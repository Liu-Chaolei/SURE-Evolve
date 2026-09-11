"""Operational training limits; these never change the scientific training recipe."""

from __future__ import annotations

import math
from pathlib import Path


class TrainingBudgetPaused(RuntimeError):
    """A saved candidate needs an explicit budget decision before more submissions."""


def estimate_training_seconds(update_seconds: float, batches: int, epochs: int,
                              validation_seconds: float = 0.0) -> float:
    if not all(math.isfinite(v) and v >= 0 for v in
               (update_seconds, batches, epochs, validation_seconds)):
        raise ValueError("Training timing must be finite and nonnegative")
    return epochs * (batches * update_seconds + validation_seconds)


def check_run_pause(workspace: Path) -> None:
    marker = workspace.parent / "metric/budget_pause.json"
    if marker.exists():
        raise TrainingBudgetPaused(f"Training budget paused; review {marker}")
