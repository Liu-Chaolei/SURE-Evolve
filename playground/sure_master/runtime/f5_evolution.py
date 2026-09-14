"""F5 workspace extensions; the framework owns data traversal and completion."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

from ..core.artifacts import file_digest
from .model_source import snapshot_source, prepare_f5_source
from .training_sources import prepare_f5_training_source, replace_once
from .f5_performance import optimize_trainer_source

PROTECTED = (
    "src/f5_tts/model/trainer.py",
    "src/f5_tts/model/dataset.py",
    "src/f5_tts/train/datasets/prepare_csv_wavs.py",
    "src/f5_tts/train/finetune_cli.py",
)


def prepare_source(source: Path, target: Path) -> Path:
    root = snapshot_source(source, target)
    prepare_f5_source(root)
    prepare_f5_training_source(root)
    trainer = root / PROTECTED[0]
    text = trainer.read_text()
    if "from playground.sure_master.runtime.f5_evolution import" not in text:
        text = text.replace(
            "from f5_tts.model.utils import default, exists",
            "from f5_tts.model.utils import default, exists\n"
            "from playground.sure_master.runtime.f5_evolution import evolution_optimizer, evolution_scheduler, evolution_batch",
        )
        text = replace_once(
            text,
            "self.optimizer = AdamW(model.parameters(), lr=learning_rate, fused=False)",
            "self.optimizer = evolution_optimizer(model.parameters(), learning_rate)",
        )
        text = replace_once(
            text,
            "        train_dataloader, self.scheduler = self.accelerator.prepare(",
            "        self.scheduler = evolution_scheduler(self.optimizer, total_updates, self.scheduler)\n"
            "        train_dataloader, self.scheduler = self.accelerator.prepare(",
        )
        text = replace_once(
            text,
            '                    text_inputs = batch["text"]',
            '                    batch = evolution_batch(batch)\n                    text_inputs = batch["text"]',
        )
        trainer.write_text(text)
    optimize_trainer_source(trainer)
    return root


def executable_identity(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_digest(path)
        for path in sorted((root / "src/f5_tts").rglob("*"))
        if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}
    }


def validate_source(
    candidate: Path, pristine: Path, *, requires_training: bool
) -> dict:
    candidate.resolve().relative_to(Path.cwd().resolve())
    if candidate.is_symlink() or any(p.is_symlink() for p in candidate.rglob("*")):
        raise ValueError("Candidate source must be a materialized workspace copy")
    for name in PROTECTED:
        if not (candidate / name).is_file() or file_digest(
            candidate / name
        ) != file_digest(pristine / name):
            raise ValueError(
                f"Framework training/data traversal is fixed: {name}; use sure_candidate.py hooks"
            )
    before, after = executable_identity(pristine), executable_identity(candidate)
    changes = {
        key: {"before": before.get(key), "after": after.get(key)}
        for key in sorted(before.keys() | after.keys())
        if before.get(key) != after.get(key)
    }
    if not requires_training and any(
        not key.startswith("src/f5_tts/infer/") for key in changes
    ):
        raise ValueError(
            "Model/training source changes require a completed training run"
        )
    return changes


def _hook(name: str):
    try:
        module = importlib.import_module("f5_tts.sure_candidate")
    except ModuleNotFoundError as exc:
        if exc.name != "f5_tts.sure_candidate":
            raise
        return None
    return getattr(module, name, None)


def evolution_optimizer(parameters, learning_rate):
    import torch

    hook = _hook("build_optimizer")
    training = json.loads(os.environ.get("SURE_F5_TRAINING_JSON", "{}"))
    return (
        hook(parameters, training)
        if hook
        else torch.optim.AdamW(
            parameters, lr=learning_rate,
            fused=os.environ.get("SURE_ACCELERATOR", "cpu") in {"npu", "cuda"},
        )
    )


def evolution_scheduler(optimizer, total_updates, default):
    hook = _hook("build_scheduler")
    training = json.loads(os.environ.get("SURE_F5_TRAINING_JSON", "{}"))
    return hook(optimizer, total_updates, training) if hook else default


def evolution_batch(batch):
    hook = _hook("transform_batch")
    if hook is None:
        return batch
    fields, shape, size = set(batch), batch["mel"].shape, len(batch["text"])
    result = hook(batch)
    if not isinstance(result, dict) or set(result) != fields:
        raise ValueError("Batch transformation must preserve the batch fields")
    if result["mel"].shape != shape or len(result["text"]) != size:
        raise ValueError("Batch transformation cannot change training batch size")
    return result
