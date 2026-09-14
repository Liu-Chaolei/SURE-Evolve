"""Real eight-rank F5 updates, restoration, speech generation and CPU CER."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

import yaml

from playground.sure_master.core.training import (
    build_training_contract,
    canonical_digest,
)
from playground.sure_master.core.utils.slurm import atomic_json
from playground.sure_master.runtime.f5_evolution import (
    prepare_source,
    executable_identity,
)
from playground.sure_master.runtime.accelerator import RuntimeBackend
from playground.sure_master.tasks.training_jobs import (
    training_command,
    training_source_identity,
)
from playground.sure_master.tools.tts_workflow import acceptance_signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sure = yaml.safe_load(args.config.read_text())["sure"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = acceptance_signature(sure)
    if (output / "probe_result.json").exists():
        previous = json.loads((output / "probe_result.json").read_text())
        if previous.get("status") == "passed" and previous.get("signature") == identity:
            return
    os.chdir(output)
    os.environ.update(
        {
            str(k): str(v)
            for k, v in sure.get("execution_env", {}).items()
            if k
            not in {
                "ASCEND_VISIBLE_DEVICES",
                "ASCEND_RT_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
            }
        }
    )
    os.environ.update(
        SURE_ACCELERATOR="npu",
        SURE_PRECISION="fp32",
        SURE_CPU_THREADS="8",
        SURE_COMPONENT_TEST="1",
        ACCELERATE_MIXED_PRECISION="no",
    )
    backend = RuntimeBackend("npu")
    if backend.torch.npu.device_count() != 8:
        raise RuntimeError(
            "The F5 acceptance job requires exactly eight allocated NPUs"
        )
    settings = deepcopy(sure["task"])
    root = prepare_source(Path(settings["resources"]["source"]), output / "source")
    # Acceptance uses the unchanged baseline optimizer, matching formal baseline training.
    rows = [
        json.loads(line)
        for line in Path(sure["datasets"]["train"]["manifest"]).read_text().splitlines()
        if line.strip()
    ]
    # Cover realistic resampling/STFT work, not only the shortest utterances.
    rows = sorted(rows, key=lambda r: (r["duration"], r["sample_id"]))
    sample_count = 2048 if settings["training"]["batch_size_per_gpu"] >= 25600 else 256
    rows = [rows[round(i * (len(rows) - 1) / (sample_count - 1))] for i in range(sample_count)] if len(rows) >= sample_count else rows
    if len(rows) < 64:
        raise ValueError("Insufficient training samples for distributed acceptance")
    metadata = output / "metadata.csv"
    with metadata.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="|")
        writer.writerow(["audio_file", "text"])
        writer.writerows((r["audio"], r["text"]) for r in rows)
    manifest = output / "train.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    cfg = yaml.safe_load((root / "src/f5_tts/configs/F5TTS_v1_Base.yaml").read_text())
    settings["training"]["manifest"] = str(metadata)
    train_output = output / "training"
    train_output.mkdir(exist_ok=True)
    contract = build_training_contract(
        "tts.f5tts",
        settings["training"],
        cfg["model"]["arch"],
        {
            "train": manifest,
            "train_csv": metadata,
            "train_validation": Path(sure["datasets"]["train_validation"]["manifest"]),
        },
        Path(settings["resources"]["checkpoint"]),
        {**executable_identity(root), **training_source_identity("tts.f5tts", root)},
        "npu",
        component_test=True,
    )
    job = {
        "adapter": "tts.f5tts",
        "source": str(root),
        "output": str(train_output),
        "settings": settings,
        "architecture": cfg["model"]["arch"],
        "structural": False,
        "contract": contract,
        "manifests": {
            "train": str(manifest),
            "train_csv": str(metadata),
            "train_validation": sure["datasets"]["train_validation"]["manifest"],
        },
    }
    job_path = output / "training-job.json"
    atomic_json(job_path, job)
    for updates in (8, 16):
        progress = train_output / "probe_result.json"
        if progress.exists() and json.loads(progress.read_text()).get(
            "contract_digest"
        ) != canonical_digest(contract):
            raise ValueError(
                "Existing component training belongs to different code or inputs"
            )
        if (
            progress.exists()
            and json.loads(progress.read_text()).get("updates", 0) >= updates
        ):
            continue
        env = {**os.environ, "SURE_F5_PROBE_UPDATES": str(updates)}
        env.pop("ACCELERATE_TORCH_DEVICE", None)
        with (output / f"train-{updates}.log").open("a") as log:
            subprocess.run(
                training_command(job_path, "tts.f5tts", settings),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    trained = json.loads((train_output / "probe_result.json").read_text())
    if not trained.get("restored") or trained.get("updates") != 16:
        raise RuntimeError(
            "F5 checkpoint restore did not continue the previous updates"
        )
    if (train_output / "evidence/training_completion.json").exists():
        raise RuntimeError("A component run must not publish full-training evidence")
    # Export the real restored probe weights without labelling them completed training.
    import torch

    latest = json.loads((train_output / "state/latest.json").read_text())["directory"]
    saved_state = torch.load(
        train_output / "state" / latest / "training.pt",
        map_location="cpu",
        weights_only=False,
    )
    probe_checkpoint = output / "probe_weights.pt"
    torch.save(
        {
            "model_state_dict": saved_state["model"],
            "ema_model_state_dict": saved_state["ema"],
        },
        probe_checkpoint,
    )
    del saved_state
    inference_settings = deepcopy(sure["task"])
    inference_settings["resources"]["checkpoint"] = str(probe_checkpoint)
    inference_settings["resources"]["source"] = str(root)
    # Exercise a real, non-default inference setting in the trusted frozen wrapper.
    sample = json.loads(
        Path(sure["datasets"]["search"]["manifest"]).read_text().splitlines()[0]
    )
    evaluation = output / "eval.jsonl"
    evaluation.write_text(json.dumps(sample, ensure_ascii=False) + "\n")
    os.environ["SURE_EVAL_MANIFEST"] = str(evaluation)
    (output / "artifacts").mkdir(exist_ok=True)
    from playground.sure_master.tasks.f5tts import execute

    execute(
        "infer", {"inference": {"nfe_step": 8}}, inference_settings, {}, {}, backend, ""
    )
    from playground.sure_master.core.artifacts import file_digest

    samples = json.loads((output / "artifacts/samples.jsonl").read_text().strip())
    before_replay = file_digest(output / "artifacts" / samples["prediction_audio"])
    shutil.copy2(output / "artifacts" / samples["prediction_audio"], output / "replay_reference.wav")
    recorded = json.loads((output / "artifacts/model_resources.json").read_text())
    inference_settings["resources"] = recorded["resources"]
    inference_settings["inference"] = recorded["inference_config"]
    execute("infer", {}, inference_settings, {}, {}, backend, "")
    if file_digest(output / "artifacts" / samples["prediction_audio"]) != before_replay:
        raise RuntimeError(
            "F5 source/checkpoint/config replay changed the generated audio"
        )
    from playground.sure_master.tasks.tts_outputs import validate_samples

    validate_samples(evaluation, output / "artifacts/samples.jsonl")
    from playground.sure_master.core.utils.task_cards import resolve_task_card
    from playground.sure_master.core.utils.metric import SureMetricRunner

    card = resolve_task_card(sure["task_cards_path"], sure["task_id"])
    runner = SureMetricRunner(
        sure["root"],
        sure["pythonpath"],
        device="cpu",
        cache_dir=sure["cache_dir"],
        python=sys.executable,
        tts_runtime="worker",
    )
    result = runner.run(
        card,
        str(output),
        str(output / "metric"),
        {"samples_jsonl": str(output / "artifacts/samples.jsonl")},
    )
    if not result.success:
        raise RuntimeError(f"CPU TTS CER component failed: {result}")
    atomic_json(
        output / "probe_result.json",
        {
            "status": "passed",
            "signature": identity,
            "training": trained,
            "cer": result.score,
            "samples": 1,
            "replay": True,
            "image": sure["slurm"]["image"],
        },
    )


if __name__ == "__main__":
    main()
