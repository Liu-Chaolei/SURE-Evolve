"""Prepare and run official, resumable training without involving the idea provider."""

from __future__ import annotations

import fcntl
import os
import sys
from pathlib import Path

from ..core.artifacts import file_digest
from ..core.training import (
    build_training_contract,
    validate_training_config,
    require_completion,
    source_identity,
)
from ..runtime.process import run_bounded
from ..runtime.training_state import atomic_json

PACKAGE = Path(__file__).resolve().parents[1]


def training_source_identity(adapter: str, root: Path) -> dict:
    external = (
        [
            "src/f5_tts/train/finetune_cli.py",
            "src/f5_tts/model/trainer.py",
            "src/f5_tts/model/dataset.py",
        ]
        if adapter == "tts.f5tts"
        else [
            "recipes/diar_ssl/conf/wavlm_updated_conformer.toml",
            "recipes/diar_ssl/trainer_dual_opt.py",
            "diarizen/trainer_dual_opt.py",
            "recipes/diar_ssl/dataset.py",
        ]
    )
    identity = source_identity(root, external)
    for name in (
        "core/training.py",
        "runtime/official_trainers.py",
        "runtime/training_state.py",
        "runtime/training_sources.py",
        "runtime/model_source.py",
        "tools/run_official_training.py",
        "tasks/training_jobs.py",
    ):
        identity["sure:" + name] = file_digest(PACKAGE / name)
    return identity


def training_command(job: Path, adapter: str, settings: dict) -> list[str]:
    training = validate_training_config(adapter, settings["training"])
    script = str(PACKAGE / "tools/run_official_training.py")
    if training["world_size"] == 1:
        return [sys.executable, script, "--job", str(job)]
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--rdzv-backend=c10d",
        "--rdzv-endpoint=127.0.0.1:0",
        "--local-addr=127.0.0.1",
        "--nnodes=1",
        f"--nproc-per-node={training['world_size']}",
        script,
        "--job",
        str(job),
    ]


def run_training(
    adapter: str,
    settings: dict,
    architecture: dict,
    source: Path,
    backend,
    output: Path,
    *,
    structural: bool = False,
) -> tuple[dict, Path]:
    training = validate_training_config(
        adapter,
        settings["training"],
        {
            "world_size": int(
                os.environ.get(
                    "SURE_TRAIN_WORLD_SIZE", settings["training"]["world_size"]
                )
            ),
            "precision": os.environ.get("SURE_PRECISION", "fp32"),
        },
    )
    manifests = {"train": Path(os.environ["SURE_TRAIN_MANIFEST"])}
    if adapter == "tts.f5tts":
        manifests["train_csv"] = Path(settings["training"]["manifest"])
        initial = Path(settings["resources"]["checkpoint"])
    else:
        manifests["train_validation"] = Path(
            os.environ["SURE_TRAIN_VALIDATION_MANIFEST"]
        )
        initial = Path(settings["resources"]["wavlm"])
    from .training_resources import validate_prepared_data, validate_wavlm_provenance

    preparation = Path(os.environ["SURE_DATA_PREPARATION"])
    checked_manifests = dict(manifests)
    if adapter == "tts.f5tts":
        checked_manifests["train_validation"] = Path(
            os.environ["SURE_TRAIN_VALIDATION_MANIFEST"]
        )
    validate_prepared_data(
        adapter,
        preparation,
        checked_manifests,
        allow_extracted_only=(
            os.environ.get("SURE_DATA_PROVENANCE_MODE") == "extracted_only"
        ),
    )
    if adapter == "sd.diarizen":
        validate_wavlm_provenance(initial)
    manifests["preparation"] = preparation
    if (
        backend.name != "cpu"
        and getattr(backend.torch, backend.name).device_count() < training["world_size"]
    ):
        raise RuntimeError(
            "Not enough allocated devices for the fixed official training batch"
        )
    identities = training_source_identity(
        adapter, Path(settings["resources"]["source"])
    )
    if adapter == "sd.diarizen":
        identities["wavlm_provenance"] = file_digest(
            initial.with_suffix(initial.suffix + ".provenance.json")
        )
    identities["torch_version"] = backend.torch.__version__
    contract = build_training_contract(
        adapter, training, architecture, manifests, initial, identities, backend.name
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    evidence = output / "evidence"
    with (output / "training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (evidence / "training_completion.json").exists():
            return require_completion(
                evidence / "training_completion.json", contract
            ), evidence
        job = {
            "adapter": adapter,
            "source": str(source.resolve()),
            "output": str(output),
            "settings": settings,
            "architecture": architecture,
            "structural": structural,
            "contract": contract,
            "manifests": {k: str(v.resolve()) for k, v in manifests.items()},
            "prepared_cache": os.environ.get("SURE_F5_PREPARED_CACHE", "")
            if adapter == "tts.f5tts"
            else "",
        }
        atomic_json(output / "job.json", job)
        env = {
            **os.environ,
            "ACCELERATE_MIXED_PRECISION": "no",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(PACKAGE.parents[1])
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
        }
        # Rank-local device selection belongs to Accelerator/torchrun, not the parent process.
        env.pop("ACCELERATE_TORCH_DEVICE", None)
        with (output / "training.log").open("a") as log:
            run_bounded(
                training_command(output / "job.json", adapter, settings),
                timeout=None,
                output=log,
                env=env,
            )
        report = require_completion(evidence / "training_completion.json", contract)
        return report, evidence
