"""Official training contracts and completion proofs, importable without model dependencies."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from .artifacts import file_digest, safe_relative

F5_RECIPE = "f5tts.finetune_cli.v1"
F5_DDP8_RECIPE = "f5tts.finetune_cli.v1.ddp8"
F5_EVOLUTION_RECIPE = "f5tts.evolution.v1.ddp8"
F5_BF16_RECIPE = "f5tts.evolution.v1.ddp8.bf16"
SD_RECIPE = "diarizen.wavlm_updated.v1"
REPORT_SCHEMA = "sure.training_completion.v1"

F5_TRAINING = {
    "recipe": F5_RECIPE,
    "epochs": 100,
    "learning_rate": 1e-5,
    "batch_size_type": "frame",
    "batch_size_per_gpu": 3200,
    "max_samples": 64,
    "grad_accumulation_steps": 1,
    "num_warmup_updates": 20000,
    "max_grad_norm": 1.0,
    "seed": 42,
    "shuffle_seed": 666,
    "world_size": 1,
    "save_per_updates": 50000,
    "checkpoint_every_updates": 5000,
    "keep_last_n_checkpoints": 2,
    "early_stopping_patience": 0,
    "checkpoint_selection": "final_ema",
    "initialization": "official_pretrained_weights",
}
F5_DDP8_TRAINING = {**F5_TRAINING, "recipe": F5_DDP8_RECIPE, "world_size": 8}
F5_EVOLUTION_TRAINING = {**F5_DDP8_TRAINING, "recipe": F5_EVOLUTION_RECIPE,
                         "keep_last_n_checkpoints": -1}
F5_BF16_TRAINING = {**F5_EVOLUTION_TRAINING, "recipe": F5_BF16_RECIPE,
                    "batch_size_per_gpu": 25600, "epochs": 50}
F5_VARIABLE_TRAINING = {
    "learning_rate", "batch_size_per_gpu", "max_samples",
    "grad_accumulation_steps", "num_warmup_updates", "max_grad_norm",
}
SD_TRAINING = {
    "recipe": SD_RECIPE,
    "epochs": 100,
    "learning_rate_wavlm": 2e-5,
    "learning_rate_network": 1e-3,
    "batch_size": 16,
    "validation_batch_size": 8,
    "grad_accumulation_steps": 1,
    "chunk_size": 8,
    "chunk_shift": 6,
    "validation_chunk_shift": 8,
    "drop_last": True,
    "seed": 3407,
    "shuffle_seed": 3407,
    "world_size": 4,
    "freeze_wavlm": False,
    "gradient_percentile": 90,
    "gradient_history_size": 1000,
    "validation_interval": 1,
    "early_stopping_patience": 10,
    "lr_decay": False,
    "use_one_cycle_lr": False,
    "checkpoint_every_updates": 500,
    "checkpoint_selection": "best5_validation_loss",
    "initialization": "ssl_backbone_new_diarization_network",
}


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def official_training(adapter: str) -> dict:
    recipes = {"tts.f5tts": F5_TRAINING, "sd.diarizen": SD_TRAINING}
    if adapter not in recipes:
        raise ValueError(f"No official full-training recipe for {adapter}")
    return deepcopy(recipes[adapter])


def validate_training_config(
    adapter: str, config: dict, runtime: dict | None = None
) -> dict:
    evolution = adapter == "tts.f5tts" and config.get("recipe") in {F5_EVOLUTION_RECIPE, F5_BF16_RECIPE}
    if evolution:
        expected = deepcopy(F5_BF16_TRAINING if config["recipe"] == F5_BF16_RECIPE else F5_EVOLUTION_TRAINING)
        for key in F5_VARIABLE_TRAINING:
            value = config.get(key)
            if key in {"learning_rate", "max_grad_norm"}:
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Invalid F5 training value: {key}")
            elif type(value) is not int or value < (0 if key == "num_warmup_updates" else 1):
                raise ValueError(f"Invalid F5 training value: {key}")
            expected[key] = value
    elif adapter == "sd.diarizen" and config.get("recipe") == "diarizen.evolution.v1":
        from ..runtime.sd_evolution import VARIABLE
        expected = deepcopy(SD_TRAINING)
        expected["recipe"] = "diarizen.evolution.v1"
        for key in VARIABLE:
            value = config.get(key, {} if key == "candidate_options" else expected.get(key))
            if key.startswith("learning_rate") and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
                raise ValueError(f"Invalid SD learning rate: {key}")
            if key == "freeze_wavlm" and type(value) is not bool:
                raise ValueError("freeze_wavlm must be boolean")
            if key == "candidate_options" and not isinstance(value, dict):
                raise ValueError("candidate_options must be an object")
            expected[key] = value
    elif adapter == "tts.f5tts" and config.get("recipe") == F5_DDP8_RECIPE:
        expected = deepcopy(F5_DDP8_TRAINING)
    else:
        expected = official_training(adapter)
    if "max_steps" in config and config["max_steps"] not in (None, 0):
        raise ValueError(
            "Full training does not accept the legacy max_steps cap; migrate to the official epoch recipe"
        )
    for key, value in expected.items():
        if config.get(key) != value or type(config.get(key)) is not type(value):
            raise ValueError(f"{adapter} official training requires {key}={value!r}")
    unknown = set(config) - set(expected) - {"manifest", "max_steps"}
    if unknown:
        raise ValueError(f"Unknown training controls: {sorted(unknown)}")
    if runtime:
        if int(runtime.get("world_size", 1)) != expected["world_size"]:
            raise ValueError(
                "Allocated training world_size does not match the official recipe"
            )
        precision = training_precision(adapter, config)
        if runtime.get("training_precision", runtime.get("precision", "fp32")) != precision:
            raise ValueError(f"The selected training recipe requires {precision}")
    return expected


def training_precision(adapter: str, training: dict) -> str:
    return "bf16" if adapter == "tts.f5tts" and training.get("recipe") == F5_BF16_RECIPE else "fp32"


def validate_contract_precision(contract: dict) -> None:
    expected = training_precision(contract.get("adapter"), contract.get("training", {}))
    # Legacy FP32 completion records did not always include this field.
    if contract.get("precision", "fp32") != expected:
        raise ValueError("Training precision differs from its recipe")
    if expected == "bf16" and (
        contract.get("master_parameter_precision") != "fp32"
        or contract.get("precision_mode") != "autocast"
    ):
        raise ValueError("BF16 training requires FP32 master parameters and autocast")


def source_identity(root: Path, files: list[str]) -> dict[str, str]:
    return {name: file_digest(root / name) for name in files}


def build_training_contract(
    adapter: str,
    training: dict,
    architecture: dict,
    data: dict[str, Path],
    initial: Path,
    source_files: dict,
    backend: str,
    *,
    component_test: bool = False,
) -> dict:
    """Physical device IDs and ephemeral paths are excluded; all scientific controls are pinned."""
    precision = training_precision(adapter, training)
    contract = {
        "schema_version": "sure.training_contract.v1",
        "adapter": adapter,
        "training": {k: v for k, v in training.items() if k != "manifest"},
        "architecture": architecture,
        "data": {k: file_digest(p) for k, p in data.items()},
        "initial_checkpoint_sha256": file_digest(initial),
        "source_files": source_files,
        "backend": backend,
        "precision": precision,
        "component_test": component_test,
    }
    if precision == "bf16":
        contract.update(master_parameter_precision="fp32", precision_mode="autocast")
    return contract


def validation_state(
    history: list[dict], patience_limit: int
) -> tuple[float | None, int, bool]:
    best, patience = None, 0
    for epoch, item in enumerate(history, 1):
        loss = item.get("loss")
        if (
            item.get("epoch") != epoch
            or not isinstance(loss, (float, int))
            or not math.isfinite(loss)
        ):
            raise ValueError(
                "Validation history must contain consecutive epochs and finite losses"
            )
        if best is None or loss < best:
            best, patience = float(loss), 0
        else:
            patience += 1
        if patience_limit and patience >= patience_limit and epoch != len(history):
            raise ValueError("Training continued beyond the fixed early-stop decision")
    return best, patience, bool(patience_limit and patience >= patience_limit)


def selected_epochs(history: list[dict], count: int = 5) -> list[int]:
    validation_state(history, 0)
    if len(history) < count:
        raise ValueError(f"Need {count} validated checkpoints for official averaging")
    return [
        r["epoch"]
        for r in sorted(history, key=lambda r: (r["loss"], r["epoch"]))[:count]
    ]


def validate_completion(
    report: dict, contract: dict | None = None, root: Path | None = None
) -> None:
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or report.get("status") != "completed"
    ):
        raise ValueError("Training has not completed")
    recorded = report.get("contract", {})
    if canonical_digest(recorded) != report.get("contract_digest"):
        raise ValueError("Training completion contract digest mismatch")
    if contract is not None and canonical_digest(contract) != report["contract_digest"]:
        raise ValueError("Training data, architecture or budget changed")
    if recorded.get("component_test"):
        raise ValueError("A component probe cannot be used as completed training")
    training = validate_training_config(
        recorded.get("adapter"), recorded.get("training", {})
    )
    validate_contract_precision(recorded)
    completed = report.get("epochs_completed")
    updates = report.get("optimizer_updates")
    batches = report.get("batches_per_epoch")
    if type(completed) is not int or not 1 <= completed <= training["epochs"]:
        raise ValueError("Invalid completed epoch count")
    if (
        type(batches) is not int
        or batches < 1
        or updates
        != completed * math.ceil(batches / training["grad_accumulation_steps"])
    ):
        raise ValueError(
            "Optimizer update count does not match complete DataLoader epochs"
        )
    if recorded["adapter"] == "tts.f5tts":
        if completed != training["epochs"] or report.get("stop_reason") != "max_epochs":
            raise ValueError(f"F5 must finish all {training['epochs']} epochs before scoring")
        if report.get("checkpoint_selection") != "final_ema":
            raise ValueError("F5 inference must use the final EMA checkpoint")
    else:
        history = report.get("validation_history", [])
        if len(history) != completed:
            raise ValueError(
                "SD needs full validation history for every completed epoch"
            )
        expected_validation = report.get("validation_batches_per_epoch")
        if (
            type(expected_validation) is not int
            or expected_validation < 1
            or any(row.get("batches") != expected_validation for row in history)
        ):
            raise ValueError(
                "Validation did not consume the complete planned DataLoader"
            )
        _, patience, early = validation_state(
            history, training["early_stopping_patience"]
        )
        if report.get("patience") != patience:
            raise ValueError("Early-stop counter does not match validation history")
        expected_reason = "early_stop" if early else "max_epochs"
        if not early and completed != training["epochs"]:
            raise ValueError(
                "SD stopped before its official training completion condition"
            )
        if report.get("stop_reason") != expected_reason:
            raise ValueError("Invalid SD stop reason")
        chosen = selected_epochs(history)
        if (
            report.get("selected_epochs") != chosen
            or report.get("checkpoint_selection") != "best5_validation_loss"
        ):
            raise ValueError(
                "SD checkpoint averaging does not match validation Loss/best/5"
            )
    resources = report.get("checkpoints")
    if not isinstance(resources, dict) or "final" not in resources:
        raise ValueError("Missing final checkpoint identity")
    if recorded["adapter"] == "sd.diarizen":
        if any(str(epoch) not in resources for epoch in report["selected_epochs"]):
            raise ValueError("Missing identities of averaged SD checkpoints")
    if root is not None:
        for item in resources.values():
            path = safe_relative(root, item["path"])
            if not path.is_file() or file_digest(path) != item["sha256"]:
                raise ValueError("Missing or corrupted training checkpoint")


def require_completion(path: Path, contract: dict | None = None) -> dict:
    report = json.loads(path.read_text())
    validate_completion(report, contract, path.parent)
    return report
