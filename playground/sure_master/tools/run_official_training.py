#!/usr/bin/env python3
"""Internal full-training worker; launched once (F5) or under torchrun (DiariZen)."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from playground.sure_master.core.training import validate_training_config, validate_contract_precision
from playground.sure_master.runtime.training_state import atomic_json


def load_initial_f5(model, checkpoint: Path, *, structural: bool) -> dict:
    import torch

    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(checkpoint), device="cpu")
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = state.get("ema_model_state_dict", state.get("model_state_dict", state))
    state = {
        k.removeprefix("ema_model."): v
        for k, v in state.items()
        if k not in {"initted", "step", "update"} and hasattr(v, "shape")
    }
    for key in (
        "mel_spec.mel_stft.mel_scale.fb",
        "mel_spec.mel_stft.spectrogram.window",
    ):
        state.pop(key, None)
    own = model.state_dict()
    matched = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape}
    ratio = sum(v.numel() for v in matched.values()) / sum(
        v.numel() for v in own.values()
    )
    if structural:
        if ratio < 0.70:
            raise ValueError(
                "F5 partial initialization matches less than 70% of parameters"
            )
        model.load_state_dict(matched, strict=False)
    else:
        model.load_state_dict(state, strict=True)
    return {"matched_parameter_ratio": ratio, "strict": not structural}


def load_initial_f5_ema(ema, checkpoint: Path, *, structural: bool) -> dict:
    """Match finetune_cli: restore EMA weights and its pretrained update state."""
    import torch

    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(checkpoint), device="cpu")
    else:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = saved.get("ema_model_state_dict", saved)
    state = dict(state)
    for key in (
        "ema_model.mel_spec.mel_stft.mel_scale.fb",
        "ema_model.mel_spec.mel_stft.spectrogram.window",
    ):
        state.pop(key, None)
    if structural:
        # New architecture tensors retain the already initialized online model values.
        own = ema.state_dict()
        own.update({k: v for k, v in state.items() if k in own and own[k].shape == v.shape})
        state = own
    ema.load_state_dict(state, strict=True)
    return {"step": int(ema.step.item()), "initted": bool(ema.initted.item()),
            "initialization": "official_pretrained_ema_state"}


def f5_train(job):
    from datasets import Dataset, load_from_disk
    from f5_tts.model import CFM, DiT, Trainer
    from f5_tts.model.dataset import CustomDataset
    from f5_tts.model.utils import get_tokenizer
    from playground.sure_master.runtime.official_trainers import (
        F5TrainingSession,
        F5ComponentComplete,
        f5_trainer_class,
    )
    from playground.sure_master.runtime.process import run_bounded
    from playground.sure_master.core.artifacts import file_digest

    output, source = Path(job["output"]), Path(job["source"])
    training = job["contract"]["training"]
    resources = job["settings"]["resources"]
    manifest = Path(job["manifests"]["train_csv"])
    identity = {
        "csv": file_digest(manifest),
        "vocab": file_digest(Path(resources["vocab"])),
        "preparer": file_digest(source / "src/f5_tts/train/datasets/prepare_csv_wavs.py"),
        "vocab_mode": "official",
    }
    os.environ["SURE_F5_VOCAB_FILE"] = str(Path(resources["vocab"]).resolve())
    cache_value = str(job.get("prepared_cache") or "").strip()
    cache_root = Path(cache_value).expanduser() if cache_value else None
    if cache_root is not None:
        cache_key = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()
        prepared = cache_root / cache_key
    else:
        prepared = output / "prepared_data"
    marker = prepared / "prepared.json"
    prepared.parent.mkdir(parents=True, exist_ok=True)
    with prepared.with_suffix(".lock").open("a") as preparation_lock:
        fcntl.flock(preparation_lock, fcntl.LOCK_EX)
        if marker.exists():
            if json.loads(marker.read_text()) != identity:
                raise ValueError(
                    "Prepared F5 data belongs to another training manifest/vocabulary"
                )
        else:
            if prepared.exists():
                shutil.rmtree(prepared)
            prepared.mkdir(parents=True, exist_ok=True)
            with (output / "prepare.log").open("a") as log:
                run_bounded(
                    [
                        sys.executable,
                        str(source / "src/f5_tts/train/datasets/prepare_csv_wavs.py"),
                        str(manifest),
                        str(prepared),
                        "--workers",
                        "4",
                    ],
                    timeout=None,
                    output=log,
                    env={**os.environ, "PYTHONPATH": str(source / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")},
                )
        atomic_json(marker, identity)
    if file_digest(prepared / "vocab.txt") != identity["vocab"]:
        raise ValueError("F5 preparation changed the official vocabulary")
    try:
        raw = load_from_disk(str(prepared / "raw"))
    except FileNotFoundError:
        raw = Dataset.from_file(str(prepared / "raw.arrow"))
    with manifest.open() as stream:
        expected = list(csv.DictReader(stream, delimiter="|"))
    if len(raw) != len(expected) or list(raw["audio_path"]) != [
        r["audio_file"] for r in expected
    ]:
        raise ValueError(
            "F5 preparation skipped, reordered or replaced training samples"
        )
    durations = json.loads((prepared / "duration.json").read_text())["duration"]
    if len(durations) != len(raw):
        raise ValueError("Prepared F5 duration count mismatch")
    if os.environ.get("SURE_STAGE_TTS_AUDIO") == "1":
        from playground.sure_master.runtime.tts_staging import stage_audio
        mapping = stage_audio(list(raw["audio_path"]), identity)
        raw = raw.map(lambda row: {"audio_path": mapping[row["audio_path"]]},
                      keep_in_memory=True, load_from_cache_file=False)
    tokenizer, size = get_tokenizer(str(resources["vocab"]), "custom")
    mel = dict(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
        mel_spec_type="vocos",
    )
    model = CFM(
        transformer=DiT(**job["architecture"], text_num_embeds=size, mel_dim=100),
        mel_spec_kwargs=mel,
        vocab_char_map=tokenizer,
    )
    initialization = load_initial_f5(
        model, Path(resources["checkpoint"]), structural=job["structural"]
    )
    os.environ["SURE_F5_TRAINING_JSON"] = json.dumps(training)
    cls = f5_trainer_class(Trainer)
    trainer = cls(
        model,
        training["epochs"],
        training["learning_rate"],
        num_warmup_updates=training["num_warmup_updates"],
        save_per_updates=training["save_per_updates"],
        keep_last_n_checkpoints=training["keep_last_n_checkpoints"],
        checkpoint_path=str(output / "state"),
        batch_size_per_gpu=training["batch_size_per_gpu"],
        batch_size_type=training["batch_size_type"],
        max_samples=training["max_samples"],
        grad_accumulation_steps=training["grad_accumulation_steps"],
        max_grad_norm=training["max_grad_norm"],
        logger=None,
        log_samples=False,
        last_per_updates=training["checkpoint_every_updates"],
        accelerate_kwargs={"mixed_precision": "bf16" if job["contract"]["precision"] == "bf16" else "no"},
    )
    trainer.sure_training = F5TrainingSession(trainer, output, job["contract"])
    from playground.sure_master.runtime.f5_precision import install_precision_audit
    install_precision_audit(trainer, output, job["contract"])
    num_workers = int(os.environ.get("SURE_F5_DATALOADER_WORKERS", "16"))
    if not 0 <= num_workers <= 16:
        raise ValueError("SURE_F5_DATALOADER_WORKERS must be between 0 and 16")
    if trainer.accelerator.is_main_process:
        initialization["ema"] = load_initial_f5_ema(
            trainer.ema_model, Path(resources["checkpoint"]), structural=job["structural"]
        )
        initialization["data_loader"] = {"num_workers": num_workers, "start_method": "spawn"}
        atomic_json(output / "initialization.json", initialization)
    if "train_validation" in job["manifests"]:
        from playground.sure_master.runtime.f5_validation import F5Validation
        trainer.sure_validation = F5Validation(trainer, Path(job["manifests"]["train_validation"]), mel, job["contract"]["backend"])
    data = CustomDataset(raw, durations=durations, preprocessed_mel=False, **mel)
    try:
        trainer.train(data, num_workers=num_workers, resumable_with_seed=training["shuffle_seed"])
    except F5ComponentComplete:
        trainer.accelerator.end_training()
        return
    trainer.sure_training.finish()
    trainer.accelerator.end_training()


def sd_train(job):
    import torch
    import toml
    from accelerate import (
        Accelerator,
        DataLoaderConfiguration,
        DistributedDataParallelKwargs,
    )
    from accelerate.utils import set_seed
    from torch.utils.data import DataLoader
    from playground.sure_master.runtime.official_trainers import sd_trainer_class
    from playground.sure_master.tasks.diarization import _dataset
    from diarizen.logger import init_logging_logger

    root, output = Path(job["source"]), Path(job["output"])
    training, resources = job["contract"]["training"], job["settings"]["resources"]
    config = toml.load(root / "recipes/diar_ssl/conf/wavlm_updated_conformer.toml")
    config["model"]["args"] = dict(job["architecture"])
    config["model"]["args"]["wavlm_src"] = str(Path(resources["wavlm"]).resolve())
    config["finetune"]["finetune"] = False
    config["meta"].update(
        save_dir=str(output / "native"), exp_id="diarizen", seed=training["seed"]
    )
    config["trainer"]["args"].update(
        max_epochs=training["epochs"],
        max_steps=0,
        max_patience=training["early_stopping_patience"],
        validation_interval=1,
        freeze_wavlm=training["freeze_wavlm"],
        lr_decay=False,
        use_one_cycle_lr=False,
        gradient_accumulation_steps=1,
    )
    config_path = output / "resolved_training.toml"
    accelerator = Accelerator(
        mixed_precision="no",
        gradient_accumulation_steps=1,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True, data_seed=training["shuffle_seed"]
        ),
    )
    if accelerator.num_processes != training["world_size"]:
        raise ValueError("Official DiariZen training requires four processes")
    set_seed(training["seed"], device_specific=True)
    if accelerator.is_main_process:
        config_path.write_text(toml.dumps(config))
        init_logging_logger(config)
    accelerator.wait_for_everyone()
    config["meta"]["config_path"] = str(config_path)
    module, name = config["model"]["path"].rsplit(".", 1)
    model = getattr(importlib.import_module(module), name)(**config["model"]["args"])
    # The only inherited weights are the original SSL backbone, never a DiariZen model.
    checkpoint = torch.load(resources["wavlm"], map_location="cpu", weights_only=True)
    model.wavlm_model.load_state_dict(checkpoint["state_dict"], strict=True)
    optimizer_small = torch.optim.AdamW(
        model.wavlm_model.parameters(), lr=training["learning_rate_wavlm"]
    )
    optimizer_big = torch.optim.AdamW(
        model.non_wavlm_parameters(), lr=training["learning_rate_network"]
    )
    evolution = training.get("recipe") == "diarizen.evolution.v1"
    if evolution:
        from playground.sure_master.runtime.sd_evolution import optimizers
        if training["freeze_wavlm"]:
            model.wavlm_model.requires_grad_(False)
        custom_optimizers = optimizers(model, training)
        optimizer_small, optimizer_big = custom_optimizers["wavlm"], custom_optimizers["network"]
    if accelerator.is_main_process:
        _dataset(
            root,
            Path(job["manifests"]["train"]),
            model,
            output / "train_data",
            chunk_shift=training["chunk_shift"],
        )
        _dataset(
            root,
            Path(job["manifests"]["train_validation"]),
            model,
            output / "train_validation",
            chunk_shift=training["validation_chunk_shift"],
        )
    accelerator.wait_for_everyone()
    train_data, collate = _dataset(
        root,
        Path(job["manifests"]["train"]),
        model,
        output / "train_data",
        chunk_shift=training["chunk_shift"],
        materialize=False,
    )
    val_data, _ = _dataset(
        root,
        Path(job["manifests"]["train_validation"]),
        model,
        output / "train_validation",
        chunk_shift=training["validation_chunk_shift"],
        materialize=False,
    )
    batch = partial(
        collate,
        max_speakers_per_chunk=config["model"]["args"]["max_speakers_per_chunk"],
    )
    train_loader = DataLoader(
        train_data,
        batch_size=training["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
        collate_fn=batch,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=training["validation_batch_size"],
        shuffle=False,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
        collate_fn=batch,
    )
    model, optimizer_small, optimizer_big, train_loader, val_loader = (
        accelerator.prepare(
            model, optimizer_small, optimizer_big, train_loader, val_loader
        )
    )
    spec = importlib.util.spec_from_file_location(
        "sure_sd_official_recipe", root / "recipes/diar_ssl/trainer_dual_opt.py"
    )
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    cls = sd_trainer_class(native.Trainer)
    trainer = cls(
        accelerator=accelerator,
        config=config,
        # Native resume asserts pre-existing native checkpoint directories. Our
        # transactional TrainingStateStore owns restoration, including first launch.
        resume=False,
        model=model,
        optimizer_small=optimizer_small,
        optimizer_big=optimizer_big,
    )
    trainer.sure_schedulers = {}
    if evolution:
        from playground.sure_master.runtime.sd_evolution import schedulers
        trainer.sure_schedulers = schedulers(
            {"wavlm": optimizer_small, "network": optimizer_big}, training,
            len(train_loader) * training["epochs"])
    trainer.train_full(train_loader, val_loader, output, job["contract"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    validate_training_config(job["adapter"], job["contract"]["training"])
    validate_contract_precision(job["contract"])
    if job["contract"]["component_test"]:
        key = "SURE_F5_PROBE_UPDATES" if job["adapter"] == "tts.f5tts" else "SURE_SD_PROBE_UPDATES"
        if not 1 <= int(os.environ.get(key, "0")) <= 1000:
            raise ValueError("Component probes require a bounded update count")
    source = Path(job["source"])
    if job["contract"]["backend"] == "npu":
        import torch_npu  # noqa: F401
    from playground.sure_master.runtime.accelerator import RuntimeBackend

    backend = RuntimeBackend(job["contract"]["backend"])
    backend.seed(job["contract"]["training"]["seed"])
    sys.path[:0] = [str(source / "src"), str(source), str(source / "pyannote-audio")]
    if job["adapter"] == "tts.f5tts":
        f5_train(job)
    elif job["adapter"] == "sd.diarizen":
        sd_train(job)
    else:
        raise ValueError("Unknown official trainer")


if __name__ == "__main__":
    main()
