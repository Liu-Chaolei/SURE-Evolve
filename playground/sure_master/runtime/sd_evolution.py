"""Editable SD models with framework-owned data, validation and training duration.

Optional diarizen/sure_candidate.py exports build_optimizers(model, training),
build_schedulers(optimizers, training, total_updates), transform_batch(batch),
and training_loss(model, batch, training). Validation never calls these hooks.
"""
from __future__ import annotations

import importlib
from pathlib import Path

from ..core.artifacts import file_digest
from .model_source import snapshot_source, prepare_diarizen_inference_source

RECIPE = "diarizen.evolution.v1"
VARIABLE = {"learning_rate_wavlm", "learning_rate_network", "freeze_wavlm", "candidate_options"}
PROTECTED = (
    "recipes/diar_ssl/dataset.py",
    "recipes/diar_ssl/trainer_dual_opt.py",
    "diarizen/trainer_dual_opt.py",
)


def prepare_source(source: Path, target: Path) -> Path:
    root = snapshot_source(source, target)
    prepare_diarizen_inference_source(root)
    return root


def executable_identity(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): file_digest(p)
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix in {".py", ".toml", ".yaml", ".json"}
            and not set(p.relative_to(root).parts) & {".git", ".cache", "__pycache__"}}


def validate_source(candidate: Path, pristine: Path, *, requires_training: bool) -> dict:
    candidate = candidate.absolute()
    candidate.resolve().relative_to(Path.cwd().resolve())
    if candidate.is_symlink() or any(p.is_symlink() for p in candidate.rglob("*")):
        raise ValueError("SD candidate must be a materialized workspace copy")
    for name in PROTECTED:
        if not (candidate / name).is_file() or file_digest(candidate / name) != file_digest(pristine / name):
            raise ValueError(f"Protected SD data/trainer file: {name}; use sure_candidate.py hooks")
    before, after = executable_identity(pristine), executable_identity(candidate)
    changes = {k: {"before": before.get(k), "after": after.get(k)}
               for k in sorted(before.keys() | after.keys()) if before.get(k) != after.get(k)}
    if not requires_training and any(not k.startswith("diarizen/pipelines/") for k in changes):
        raise ValueError("SD model/training source changes require training")
    return changes


def hook(name: str):
    try:
        module = importlib.import_module("diarizen.sure_candidate")
    except ModuleNotFoundError as exc:
        if exc.name != "diarizen.sure_candidate":
            raise
        return None
    return getattr(module, name, None)


def optimizers(model, training: dict):
    import torch
    custom = hook("build_optimizers")
    result = custom(model, training) if custom else {
        "wavlm": torch.optim.AdamW(model.wavlm_model.parameters(), lr=training["learning_rate_wavlm"]),
        "network": torch.optim.AdamW(model.non_wavlm_parameters(), lr=training["learning_rate_network"]),
    }
    if not isinstance(result, dict) or set(result) != {"wavlm", "network"}:
        raise ValueError("build_optimizers must return wavlm and network optimizers")
    if any(not isinstance(value, torch.optim.Optimizer) for value in result.values()):
        raise ValueError("Invalid SD optimizer")
    return result


def schedulers(optimizers: dict, training: dict, total_updates: int) -> dict:
    custom = hook("build_schedulers")
    result = custom(optimizers, training, total_updates) if custom else {}
    if not isinstance(result, dict) or any(not isinstance(k, str) or not k or
            not all(callable(getattr(v, attr, None)) for attr in ("step", "state_dict", "load_state_dict"))
            for k, v in result.items()):
        raise ValueError("build_schedulers must return named resumable schedulers")
    return result


def training_step(trainer, batch, index, native):
    import torch
    transform = hook("transform_batch")
    if transform:
        batch = transform(batch)
    loss_hook = hook("training_loss")
    if not loss_hook:
        return native(trainer, batch, index)
    trainer.optimizer_small.zero_grad()
    trainer.optimizer_big.zero_grad()
    loss = loss_hook(trainer.model, batch, trainer.sure_contract["training"])
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("training_loss must return a finite scalar tensor")
    trainer.accelerator.backward(loss)
    if trainer.accelerator.sync_gradients:
        trainer.auto_clip_grad_norm_(trainer.model)
    trainer.optimizer_small.step()
    trainer.optimizer_big.step()
    return {"Loss": loss.detach().float()}
