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
from functools import partial

from ..runtime.model_source import snapshot_source, prepare_audio_io, prepare_diarizen_inference_source

INFER_KEYS = {
    "seg_duration",
    "clustering_method",
    "min_cluster_size",
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


def mono_collate(batch, *, native_collate, max_speakers_per_chunk=4):
    """Match inference's first-channel SDM policy before stacking mixed audio."""
    result = native_collate(
        [(audio[:1], labels, name) for audio, labels, name in batch],
        max_speakers_per_chunk=max_speakers_per_chunk,
    )
    # Ascend MSE does not promote uint8 targets; binary labels are exact in FP32.
    result["ts"] = result["ts"].float()
    return result


def rows(path: Path) -> list[dict]:
    data = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [r["session_id"] for r in data]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Diarization manifest is empty or repeats session IDs")
    return data


def write_session_rttm(annotation, row: dict, stream) -> None:
    """Remove pipeline window padding outside the physical recording."""
    from pyannote.core import Segment

    bounded = annotation.crop(Segment(0.0, float(row["duration"])), mode="intersection")
    bounded.uri = row["session_id"]
    bounded.write_rttm(stream)


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


def _dataset(
    root: Path, manifest: Path, model, work: Path, *, chunk_shift=None, materialize=True
):
    if materialize:
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
        chunk_shift=chunk_shift or model.chunk_size,
    )
    if not len(dataset):
        raise ValueError("No usable SD training chunks")
    return dataset, partial(mono_collate, native_collate=module._collate_fn)


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
    import toml
    from ..core.training import validate_training_config, require_completion
    from ..runtime.official_trainers import checkpoint_identity
    from ..runtime.training_state import atomic_json
    from .training_jobs import run_training

    if set(parameters) - {"inference", "training", "architecture"}:
        raise ValueError("Expected inference/training/architecture parameter sections")
    from copy import deepcopy
    from ..runtime.sd_evolution import RECIPES, VARIABLE, prepare_source, validate_source
    evolution = settings["training"].get("recipe") in RECIPES
    settings = deepcopy(settings)
    if evolution:
        overrides = parameters.get("training", {})
        if set(overrides) - VARIABLE:
            raise ValueError("SD candidate changes fixed training controls")
        settings["training"].update(overrides)
        validate_training_config("sd.diarizen", settings["training"])
    if action == "infer" and parameters.get("training"):
        raise ValueError("Inference cannot change training parameters")
    if parameters.get("training") and not evolution:
        raise ValueError("Official DiariZen training settings are fixed")
    arch = parameters.get("architecture", {})
    if not isinstance(arch, dict) or (not evolution and set(arch) - set(ARCH_VALUES)):
        raise ValueError("Unsupported SD architecture parameters")
    if action in {"baseline", "infer", "fine_tune"} and arch:
        raise ValueError("Structural changes require arch action")
    resources = {
        key: Path(value).resolve() for key, value in settings["resources"].items()
    }
    resources.update(saved)
    source_changes = {}
    candidate_source = os.environ.get("SURE_SD_CANDIDATE_SOURCE")
    if candidate_source:
        if not evolution:
            raise ValueError("SD source editing requires the evolution recipe")
        pristine = prepare_source(resources["source"], Path("working/sd_pristine"))
        source_changes = validate_source(Path(candidate_source), pristine, requires_training=action != "infer")
        root = snapshot_source(Path(candidate_source), Path("working/diarizen_source"))
    else:
        root = snapshot_source(resources["source"], Path("working/diarizen_source"))
    prepare_diarizen_inference_source(root)
    pipeline_file = root / "diarizen/pipelines/inference.py"
    text = pipeline_file.read_text()
    text = text.replace(
        'torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")',
        'torch.device(os.environ.get("SURE_MODEL_DEVICE", "cpu"))',
    )
    pipeline_file.write_text(text)
    prepare_audio_io(pipeline_file)
    os.environ["SURE_MODEL_DEVICE"] = str(backend.device)
    sys.path[:0] = [str(root), str(root / "pyannote-audio")]
    defaults = {
        "seg_duration": 8,
        "segmentation_step": 0.1,
        "batch_size": 32,
        "apply_median_filtering": True,
        "clustering_method": "AgglomerativeClustering",
        "min_speakers": 1,
        "max_speakers": 20,
        "min_cluster_size": 30,
        "ahc_threshold": 0.70,
    }
    inference = {
        **defaults,
        **settings.get("inference", {}),
        **manifest.get("inference_config", {}),
        **parameters.get("inference", {}),
    }
    if set(inference) - INFER_KEYS:
        raise ValueError("Unsupported SD inference parameters")
    if action in {"baseline", "fine_tune", "arch"}:
        validate_training_config("sd.diarizen", settings["training"])
        config = toml.load(root / "recipes/diar_ssl/conf/wavlm_updated_conformer.toml")
        architecture = dict(config["model"]["args"])
        architecture.pop("wavlm_src", None)
        if action == "arch" and not source_changes and (
            not arch
            or all(
                architecture.get(k, 31 if k == "kernel_size" else None) == v
                for k, v in arch.items()
            )
        ):
            raise ValueError("SD arch action requires an actual structure change")
        for key, value in arch.items():
            if not evolution and value not in ARCH_VALUES[key]:
                raise ValueError(f"Unsupported {key}: {value}")
        architecture.update(arch)
        completion, evidence = run_training(
            "sd.diarizen",
            settings,
            architecture,
            root,
            backend,
            Path("models/diarizen_training"),
            structural=action == "arch",
        )
        model_dir = evidence / "model"
        config = {
            "model": {
                "path": config["model"]["path"],
                "args": {**architecture, "wavlm_src": str(resources["wavlm"])},
            },
            "inference": {
                "args": {
                    key: inference[key]
                    for key in (
                        "seg_duration",
                        "segmentation_step",
                        "batch_size",
                        "apply_median_filtering",
                    )
                }
            },
            "clustering": {
                "args": {
                    "method": inference["clustering_method"],
                    **{
                        key: inference[key]
                        for key in (
                            "min_speakers",
                            "max_speakers",
                            "min_cluster_size",
                            "ahc_threshold",
                        )
                    },
                }
            },
        }
        (model_dir / "config.toml").write_text(toml.dumps(config))
        completion["checkpoints"]["model_config"] = checkpoint_identity(
            evidence, model_dir / "config.toml"
        )
        atomic_json(evidence / "training_completion.json", completion)
        resources.update(model=model_dir, training_evidence=evidence,
                         wavlm_provenance=resources["wavlm"].with_suffix(resources["wavlm"].suffix + ".provenance.json"))
    else:
        if not saved or "training_evidence" not in resources:
            raise ValueError(
                "DiariZen inference requires a completed official-training artifact"
            )
        completion = require_completion(
            resources["training_evidence"] / "training_completion.json"
        )
        config = toml.load(resources["model"] / "config.toml")
    config["inference"]["args"].update({key: inference[key] for key in
        ("seg_duration", "segmentation_step", "batch_size", "apply_median_filtering")})
    config["clustering"]["args"].update({"method": inference["clustering_method"],
        **{key: value for key, value in inference.items() if key not in
           {"seg_duration", "segmentation_step", "batch_size", "apply_median_filtering", "clustering_method"}}})
    # Keep the immutable bundle untouched; only the local inference copy gets relocated paths.
    local = Path("working/diarizen_inference").resolve()
    local.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resources["model"] / "pytorch_model.bin", local / "pytorch_model.bin")
    config["model"]["args"]["wavlm_src"] = str(resources["wavlm"])
    (local / "config.toml").write_text(toml.dumps(config))
    from diarizen.pipelines.inference import DiariZenPipeline

    embedding = Path("working/pyannote_embedding/pytorch_model.bin").resolve()
    embedding.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resources["embedding"], embedding)
    # This official embedding checkpoint contains these pyannote metadata types.
    # Keep weights-only loading enabled on modern Torch instead of unpickling arbitrary globals.
    import torch
    from pyannote.audio.core.task import Specifications, Problem, Resolution

    with torch.serialization.safe_globals([
        torch.torch_version.TorchVersion, Specifications, Problem, Resolution,
    ]):
        pipeline = DiariZenPipeline(local, str(embedding), rttm_out_dir=None)
    backend.verify_model(pipeline._segmentation.model)
    if hasattr(pipeline._embedding, "model_"):
        backend.verify_model(pipeline._embedding.model_)
    manifest_path = Path(os.environ["SURE_EVAL_MANIFEST"])
    output = Path("artifacts")
    output.mkdir(exist_ok=True)
    processed = []
    with (output / "hyp.rttm").open("w") as stream:
        for row in rows(manifest_path):
            annotation = pipeline(row["audio"], sess_name=row["session_id"])
            write_session_rttm(annotation, row, stream)
            processed.append(row["session_id"])
    (output / "processed_sessions.json").write_text(json.dumps(processed))
    validate_rttm_outputs(output / "hyp.rttm", manifest_path)
    resources["source"] = root
    kind = (
        "arch"
        if action == "arch"
        else "inference"
        if action == "infer"
        else "fine_tune"
    )
    record = {
        "candidate_type": kind,
        "idea_text": json.dumps(parameters),
        "changed_fields": [*[f"arch_config.{k}" for k in arch], *[f"source.{k}" for k in source_changes], *[f"training.{k}" for k in parameters.get("training", {})], *[f"inference.{k}" for k in parameters.get("inference", {})]],
        "source_changes": source_changes,
        "inference_config": inference,
        "arch_config": config["model"]["args"],
        "training_config": completion,
        "defaults": {},
        "diff_from_defaults": parameters,
        "produced_artifacts": {
            "hyp": "artifacts/hyp.rttm",
            "model": str(resources["model"]),
        },
    }
    atomic_json(output / "candidate_changes.json", record)
    atomic_json(
        output / "model_resources.json",
        {
            "resources": {k: str(v) for k, v in resources.items()},
            "model_config": config["model"],
            "inference_config": inference,
            "provenance": {
                "action": action,
                "parent": parent,
                "training": completion,
                "accelerator": backend.name,
                "cpu_operations": ["clustering", "speaker_assignment"],
            },
        },
    )
