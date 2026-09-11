"""DiariZen single-channel execution and trusted RTTM/UEM validation."""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path

from ..runtime.model_source import snapshot_source, prepare_audio_io

INFER_KEYS = {
    "segmentation_step",
    "batch_size",
    "apply_median_filtering",
    "ahc_threshold",
    "Fa",
    "Fb",
    "min_speakers",
    "max_speakers",
    "max_iters",
}
TRAIN_KEYS = {"learning_rate", "batch_size", "freeze_wavlm"}
ARCH_VALUES = {
    "num_layer": {2, 3, 4, 5},
    "ffn_hidden": {512, 768, 1024, 1536},
    "kernel_size": {15, 31, 63},
}


def rows(path: Path) -> list[dict]:
    data = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [r["session_id"] for r in data]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Diarization manifest is empty or repeats session IDs")
    return data


def read_rttm(path: Path) -> list[list[str]]:
    result = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 10 or fields[0] != "SPEAKER":
            raise ValueError(f"Invalid RTTM row {number}: {path}")
        start, length = float(fields[3]), float(fields[4])
        if not math.isfinite(start + length) or start < 0 or length <= 0:
            raise ValueError(f"Invalid RTTM times at row {number}")
        result.append(fields)
    return result


def validate_rttm_outputs(path: Path, manifest: Path) -> None:
    expected = {r["session_id"]: r for r in rows(manifest)}
    processed = json.loads((path.parent / "processed_sessions.json").read_text())
    if len(processed) != len(expected) or set(processed) != set(expected):
        raise ValueError("SD output does not cover exactly the requested sessions")
    for row in read_rttm(path):
        if row[1] not in expected:
            raise ValueError(f"Unknown output session: {row[1]}")
        if float(row[3]) + float(row[4]) > float(expected[row[1]]["duration"]) + 0.02:
            raise ValueError(f"RTTM exceeds recording duration: {row[1]}")


def clip_rttm(source: Path, manifest: Path, destination: Path) -> None:
    regions = {
        r["session_id"]: r.get("uem", [[0, r["duration"]]]) for r in rows(manifest)
    }
    result = []
    for fields in read_rttm(source):
        if fields[1] not in regions:
            continue
        start, end = float(fields[3]), float(fields[3]) + float(fields[4])
        for left, right in regions[fields[1]]:
            left, right = max(start, float(left)), min(end, float(right))
            if right > left:
                clipped = fields.copy()
                clipped[3:5] = [f"{left:.6f}", f"{right - left:.6f}"]
                result.append(" ".join(clipped))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(result) + ("\n" if result else ""))


def _dataset(root: Path, manifest: Path, model, work: Path):
    data = rows(manifest)
    work.mkdir(parents=True, exist_ok=True)
    (work / "wav.scp").write_text(
        "".join(f"{r['session_id']} {r['audio']}\n" for r in data)
    )
    (work / "rttm").write_text(
        "".join(r["reference_rttm"].rstrip() + "\n" for r in data)
    )
    uem = []
    for row in data:
        bounds = row.get("uem", [[0, row["duration"]]])
        if len(bounds) != 1:
            raise ValueError(
                "DiariZen training requires one contiguous UEM per recording"
            )
        uem.append(f"{row['session_id']} 1 {bounds[0][0]} {bounds[0][1]}\n")
    (work / "all.uem").write_text("".join(uem))
    spec = importlib.util.spec_from_file_location(
        "sure_diarizen_dataset", root / "recipes/diar_ssl/dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frames, duration, step = model.get_rf_info
    dataset = module.DiarizationDataset(
        str(work / "wav.scp"),
        str(work / "rttm"),
        str(work / "all.uem"),
        frames,
        duration,
        step,
        chunk_size=model.chunk_size,
        chunk_shift=model.chunk_size,
    )
    if not len(dataset):
        raise ValueError("No usable SD training chunks")
    return dataset, module._collate_fn


def _loss(model, batch, device):
    import torch
    from pyannote.audio.utils.loss import nll_loss
    from pyannote.audio.utils.permutation import permutate

    prediction = model(batch["xs"].to(device))
    target = batch["ts"].to(device)
    multilabel = model.powerset.to_multilabel(prediction)
    # Assignment is CPU/SciPy in upstream; neural forward/backward remain on the backend.
    target, _ = permutate(multilabel, target)
    powerset = model.powerset.to_powerset(target.float())
    return nll_loss(prediction, torch.argmax(powerset, dim=-1))


def execute(action, parameters, settings, manifest, saved, backend, parent):
    import torch
    import toml
    from functools import partial
    from torch.utils.data import DataLoader

    if set(parameters) - {"inference", "training", "architecture"}:
        raise ValueError("Expected inference/training/architecture parameter sections")
    for key, allowed in (
        ("inference", INFER_KEYS),
        ("training", TRAIN_KEYS),
        ("architecture", set(ARCH_VALUES)),
    ):
        if (
            not isinstance(parameters.get(key, {}), dict)
            or set(parameters.get(key, {})) - allowed
        ):
            raise ValueError(f"Unsupported SD {key} parameters")
    if action in {"baseline", "infer"} and (
        parameters.get("training") or parameters.get("architecture")
    ):
        raise ValueError("Inference cannot train or change structure")
    if action == "fine_tune" and parameters.get("architecture"):
        raise ValueError("Structural changes require arch")
    resources = {k: Path(v).resolve() for k, v in settings["resources"].items()}
    resources.update(saved)
    root = snapshot_source(resources["source"], Path("working/diarizen_source"))
    # Explicit device injection into the workspace copy; external source stays untouched.
    pipeline_file = root / "diarizen/pipelines/inference.py"
    text = pipeline_file.read_text()
    old = 'torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")'
    text = text.replace(old, 'torch.device(os.environ.get("SURE_MODEL_DEVICE", "cpu"))')
    pipeline_file.write_text(text)
    prepare_audio_io(pipeline_file)
    os.environ["SURE_MODEL_DEVICE"] = str(backend.device)
    sys.path[:0] = [str(root), str(root / "pyannote-audio")]
    config = toml.load(resources["model"] / "config.toml")
    baseline_config = json.loads(json.dumps(config))
    overrides = {
        **settings.get("inference", {}),
        **manifest.get("inference_config", {}),
        **parameters.get("inference", {}),
    }
    for key, value in overrides.items():
        if key not in INFER_KEYS:
            raise ValueError(f"Unsupported SD inference setting: {key}")
        section = "inference" if key in config["inference"]["args"] else "clustering"
        config[section]["args"][key] = value
    train = {**settings.get("training", {}), **parameters.get("training", {})}
    model_dir = Path("models/diarizen").resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    initial = resources["model"] / "pytorch_model.bin"
    training_report = {}
    if action in {"fine_tune", "arch"}:
        arch = parameters.get("architecture", {})
        if action == "arch" and (
            not arch
            or all(config["model"]["args"].get(k) == v for k, v in arch.items())
        ):
            raise ValueError("SD arch action requires an actual structure change")
        for key, value in arch.items():
            if value not in ARCH_VALUES[key]:
                raise ValueError(f"Unsupported {key}: {value}")
        config["model"]["args"].update(arch)
        module, cls = config["model"]["path"].rsplit(".", 1)
        model = getattr(importlib.import_module(module), cls)(**config["model"]["args"])
        state = torch.load(initial, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        own = model.state_dict()
        matched = {
            k: v for k, v in state.items() if k in own and v.shape == own[k].shape
        }
        ratio = sum(v.numel() for v in matched.values()) / sum(
            v.numel() for v in own.values()
        )
        if action == "fine_tune":
            model.load_state_dict(state, strict=True)
        else:
            if ratio < 0.70:
                raise ValueError(
                    f"SD checkpoint parameter match {ratio:.3f} is below 0.70"
                )
            model.load_state_dict(matched, strict=False)
        if train.get("freeze_wavlm", False):
            for parameter in model.wavlm_model.parameters():
                parameter.requires_grad_(False)
        model.to(backend.device).train()
        backend.verify_model(model)
        dataset, collate = _dataset(
            root,
            Path(os.environ["SURE_TRAIN_MANIFEST"]),
            model,
            Path("working/train_data"),
        )
        batch_size = int(train.get("batch_size", 1))
        steps = int(settings.get("training", {}).get("max_steps", 1000))
        lr = float(train.get("learning_rate", 2e-5))
        if (
            not 1 <= batch_size <= 16
            or not 0 < steps <= 100000
            or not 1e-7 <= lr <= 1e-3
        ):
            raise ValueError("Invalid SD training budget or optimizer settings")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=partial(
                collate,
                max_speakers_per_chunk=config["model"]["args"][
                    "max_speakers_per_chunk"
                ],
            ),
        )
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr, fused=False
        )
        iterator = iter(loader)
        for step in range(steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, batch, backend.device)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite SD training loss")
            loss.backward()
            backend.verify_model(model, gradients=True)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            optimizer.step()
        model.eval()
        validation, collate = _dataset(
            root,
            Path(os.environ["SURE_TRAIN_VALIDATION_MANIFEST"]),
            model,
            Path("working/train_validation"),
        )
        validation_batch = next(
            iter(
                DataLoader(
                    validation,
                    batch_size=1,
                    collate_fn=partial(
                        collate,
                        max_speakers_per_chunk=config["model"]["args"][
                            "max_speakers_per_chunk"
                        ],
                    ),
                )
            )
        )
        with torch.no_grad():
            validation_loss = float(_loss(model, validation_batch, backend.device))
        if not math.isfinite(validation_loss):
            raise ValueError("Nonfinite SD validation loss")
        torch.save(
            {k: v.cpu() for k, v in model.state_dict().items()},
            model_dir / "pytorch_model.bin",
        )
        training_report = {
            "steps": steps,
            "learning_rate": lr,
            "batch_size": batch_size,
            "matched_parameters": ratio,
            "train_validation_loss": validation_loss,
            "train_validation_batches": 1,
            "initial_checkpoint": str(initial),
            "seed": train.get("seed", 42),
        }
        del model, optimizer
        backend.empty_cache()
    else:
        shutil.copy2(initial, model_dir / "pytorch_model.bin")
    if (resources["model"] / "plda").is_dir():
        shutil.copytree(
            resources["model"] / "plda", model_dir / "plda", dirs_exist_ok=True
        )
    with (model_dir / "config.toml").open("w") as stream:
        toml.dump(config, stream)
    from diarizen.pipelines.inference import DiariZenPipeline

    embedding = Path("working/pyannote_embedding/pytorch_model.bin").resolve()
    embedding.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resources["embedding"], embedding)
    pipeline = DiariZenPipeline(model_dir, str(embedding), rttm_out_dir=None)
    if hasattr(pipeline._embedding, "model_"):
        backend.verify_model(pipeline._embedding.model_)
    backend.verify_model(pipeline._segmentation.model)
    manifest_path = Path(os.environ["SURE_EVAL_MANIFEST"])
    output = Path("artifacts")
    output.mkdir(exist_ok=True)
    processed = []
    with (output / "hyp.rttm").open("w") as stream:
        for row in rows(manifest_path):
            annotation = pipeline(row["audio"], sess_name=row["session_id"])
            annotation.uri = row["session_id"]
            annotation.write_rttm(stream)
            processed.append(row["session_id"])
    (output / "processed_sessions.json").write_text(json.dumps(processed))
    validate_rttm_outputs(output / "hyp.rttm", manifest_path)
    resources.update(source=root, model=model_dir)
    changed = [
        f"{section}.{key}" for section, values in parameters.items() for key in values
    ]
    candidate_type = {
        "baseline": "inference",
        "infer": "inference",
        "fine_tune": "fine_tune",
        "arch": "arch",
    }[action]
    changes = {
        "candidate_type": candidate_type,
        "idea_text": json.dumps(parameters),
        "changed_fields": changed,
        "inference_config": overrides,
        "arch_config": config["model"]["args"],
        "training_config": training_report,
        "defaults": baseline_config,
        "diff_from_defaults": parameters,
        "produced_artifacts": {
            "hyp": "artifacts/hyp.rttm",
            "checkpoint": str(model_dir / "pytorch_model.bin"),
        },
    }
    (output / "candidate_changes.json").write_text(json.dumps(changes, indent=2))
    report = {
        "resources": {k: str(v) for k, v in resources.items()},
        "model_config": config["model"],
        "inference_config": overrides,
        "provenance": {
            "action": action,
            "parent": parent,
            "training": training_report,
            "accelerator": backend.name,
            "cpu_operations": ["clustering", "speaker_assignment"],
        },
    }
    (output / "model_resources.json").write_text(json.dumps(report, indent=2))
