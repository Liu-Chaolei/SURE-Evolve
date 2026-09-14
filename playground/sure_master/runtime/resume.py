"""Recover complete Icefall checkpoints after a Slurm time slice."""

from __future__ import annotations
import os
import pickle
from pathlib import Path


def complete_epoch(directory: Path, epoch: int) -> bool:
    """A decode retry must not restart an already completed formal trainer."""
    checkpoint = directory / f"epoch-{epoch}.pt"
    if not checkpoint.is_file():
        return False
    import torch
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        return (int(state.get("cur_epoch", 0)) == epoch
                and int(state.get("batch_idx_train", 0)) > 0
                and all(name in state for name in ("model", "optimizer", "scheduler")))
    except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError):
        return False


def resume_command(command: list[str], directory: Path) -> list[str]:
    if os.environ.get("SURE_SLURM_RESUME") != "1":
        return command
    import torch

    candidates = []
    for path in directory.glob("*.pt"):
        if not (path.stem.startswith("epoch-") or path.stem.startswith("checkpoint-")):
            continue
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if not all(
                k in checkpoint
                for k in ("model", "optimizer", "scheduler", "batch_idx_train")
            ):
                continue
            candidates.append(
                (
                    int(checkpoint["batch_idx_train"]),
                    path,
                    int(checkpoint.get("cur_epoch", 0)),
                )
            )
        except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError):
            continue  # Interrupted writes are not checkpoints.
    if not candidates:
        return command
    _, path, epoch = max(
        candidates, key=lambda row: (row[0], row[1].name.startswith("epoch-"))
    )
    result = list(command)
    option, value = (
        ("--start-batch", path.stem.split("-")[-1])
        if path.name.startswith("checkpoint-")
        else ("--start-epoch", str(epoch + 1))
    )
    if option in result:
        result[result.index(option) + 1] = value
    else:
        result.extend([option, value])
    return result
