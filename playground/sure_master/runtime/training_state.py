"""Transactional training snapshots including optimizer, RNG and exact epoch/batch position."""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from pathlib import Path

from ..core.artifacts import file_digest
from ..core.training import canonical_digest


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    with pending.open("w") as stream:
        json.dump(data, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def capture_rng(backend: str) -> dict:
    import numpy as np
    import torch

    result = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if backend != "cpu":
        result["device"] = getattr(torch, backend).get_rng_state()
    return result


def restore_rng(state: dict, backend: str) -> None:
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if backend != "cpu":
        getattr(torch, backend).set_rng_state(state["device"].cpu())


class TrainingStateStore:
    """All ranks participate in a save. Only atomically committed directories are resumable."""

    def __init__(self, root: Path, contract: dict, accelerator):
        self.root, self.contract, self.accelerator = (
            root.resolve(),
            contract,
            accelerator,
        )
        self.digest = canonical_digest(contract)
        if accelerator.is_main_process:
            self.root.mkdir(parents=True, exist_ok=True)
            marker = self.root / "contract.json"
            if marker.exists():
                if json.loads(marker.read_text())["digest"] != self.digest:
                    raise ValueError(
                        "Cannot resume after changing data, structure, budget, backend or world size"
                    )
            elif any(self.root.glob("checkpoint-*")):
                raise ValueError("Unidentified training state; use a new workspace")
            else:
                atomic_json(marker, {"digest": self.digest, "contract": contract})
        accelerator.wait_for_everyone()
        if (
            json.loads((self.root / "contract.json").read_text())["digest"]
            != self.digest
        ):
            raise ValueError("Training contract mismatch")

    def save(
        self,
        model,
        optimizers: dict,
        schedulers: dict,
        progress: dict,
        *,
        ema=None,
        extra=None,
    ) -> Path:
        import torch

        accel = self.accelerator
        pending = self.root / ".checkpoint.pending"
        if accel.is_main_process:
            if pending.exists():
                shutil.rmtree(pending)
            pending.mkdir()
        accel.wait_for_everyone()
        rng = capture_rng(self.contract["backend"])
        torch.save(
            {"rng": rng, "extra": extra or {}},
            pending / f"rank-{accel.process_index}.pt",
        )
        if accel.is_main_process:
            torch.save(
                {
                    "model": accel.unwrap_model(model).state_dict(),
                    "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
                    "schedulers": {k: v.state_dict() for k, v in schedulers.items()},
                    "ema": ema.state_dict() if ema is not None else None,
                },
                pending / "training.pt",
            )
        accel.wait_for_everyone()
        if accel.is_main_process:
            files = {p.name: file_digest(p) for p in sorted(pending.glob("*.pt"))}
            if len(files) != accel.num_processes + 1:
                raise ValueError("A training rank did not save its state")
            atomic_json(
                pending / "state.json",
                {
                    "contract_digest": self.digest,
                    "progress": progress,
                    "world_size": accel.num_processes,
                    "files": files,
                },
            )
            atomic_json(
                pending / "commit.json",
                {"state_sha256": file_digest(pending / "state.json")},
            )
            target = self.root / f"checkpoint-{time.time_ns()}"
            os.rename(pending, target)
            atomic_json(self.root / "latest.json", {"directory": target.name})
            for old in sorted(self.root.glob("checkpoint-*"), reverse=True)[2:]:
                shutil.rmtree(old)
        accel.wait_for_everyone()
        # Saving must not perturb the next training RNG sequence.
        restore_rng(rng, self.contract["backend"])
        return (
            self.root / json.loads((self.root / "latest.json").read_text())["directory"]
        )

    def latest(self) -> tuple[Path, dict] | None:
        candidates = sorted(self.root.glob("checkpoint-*"), reverse=True)
        for directory in candidates:
            try:
                commit = json.loads((directory / "commit.json").read_text())
                if file_digest(directory / "state.json") != commit["state_sha256"]:
                    continue
                metadata = json.loads((directory / "state.json").read_text())
                if metadata["contract_digest"] != self.digest:
                    raise ValueError("Training state belongs to a different contract")
                if metadata["world_size"] != self.accelerator.num_processes:
                    raise ValueError("Training world size changed")
                files = metadata["files"]
                expected = {
                    "training.pt",
                    *(f"rank-{r}.pt" for r in range(self.accelerator.num_processes)),
                }
                if set(files) != expected:
                    continue
                if all(
                    (directory / name).is_file()
                    and file_digest(directory / name) == sha
                    for name, sha in files.items()
                ):
                    return directory, metadata
            except (OSError, json.JSONDecodeError, KeyError):
                continue
        if candidates:
            raise ValueError("No intact resumable checkpoint remains")
        return None

    def restore(
        self, model, optimizers: dict, schedulers: dict, *, ema=None
    ) -> tuple[dict, dict] | None:
        import torch

        found = self.latest()
        if found is None:
            return None
        directory, metadata = found
        # These are local snapshots whose checksums and training identity were verified above.
        state = torch.load(
            directory / "training.pt", map_location="cpu", weights_only=False
        )
        if set(state["optimizers"]) != set(optimizers) or set(
            state["schedulers"]
        ) != set(schedulers):
            raise ValueError("Optimizer/scheduler topology changed")
        self.accelerator.unwrap_model(model).load_state_dict(
            state["model"], strict=True
        )
        for key, optimizer in optimizers.items():
            optimizer.load_state_dict(state["optimizers"][key])
        for key, scheduler in schedulers.items():
            scheduler.load_state_dict(state["schedulers"][key])
        if ema is not None:
            if state["ema"] is None:
                raise ValueError("Missing EMA state")
            ema.load_state_dict(state["ema"])
        rank = torch.load(
            directory / f"rank-{self.accelerator.process_index}.pt",
            map_location="cpu",
            weights_only=False,
        )
        # Restore RNG after recreating/skipping the DataLoader iterator, before the first forward.
        return metadata["progress"], rank
