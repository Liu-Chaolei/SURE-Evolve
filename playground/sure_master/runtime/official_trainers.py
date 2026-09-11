"""Durable orchestration around the upstream F5 and DiariZen training algorithms."""

from __future__ import annotations

import math
from pathlib import Path

from ..core.artifacts import file_digest
from ..core.training import (
    REPORT_SCHEMA,
    canonical_digest,
    selected_epochs,
    validation_state,
    validate_completion,
)
from .training_state import TrainingStateStore, atomic_json, restore_rng


def atomic_torch_save(value, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    torch.save(value, pending)
    pending.replace(path)


def checkpoint_identity(root: Path, path: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)}


def verify_training_device(model, accelerator, contract):
    import torch

    if accelerator.device.type != contract["backend"]:
        raise RuntimeError("Training backend differs from the declared accelerator")
    parameters = list(accelerator.unwrap_model(model).parameters())
    if not parameters or any(
        p.device.type != contract["backend"] or p.dtype != torch.float32
        for p in parameters
    ):
        raise RuntimeError(
            "Official training requires model parameters on the declared backend in FP32"
        )


def new_progress() -> dict:
    return {
        "epoch": 0,
        "batch": 0,
        "updates": 0,
        "batches_per_epoch": None,
        "validation_history": [],
    }


class F5TrainingSession:
    def __init__(self, trainer, output: Path, contract: dict):
        self.trainer, self.output, self.contract = trainer, output, contract
        self.store = TrainingStateStore(output / "state", contract, trainer.accelerator)
        self.progress = new_progress()
        self.pending_rng = None

    def configure_loader(self, loader) -> None:
        if (
            self.trainer.accelerator.num_processes
            != self.contract["training"]["world_size"]
        ):
            raise ValueError("F5 process count differs from the training contract")
        verify_training_device(
            self.trainer.model, self.trainer.accelerator, self.contract
        )
        self.batches = len(loader)
        if self.batches <= 0:
            raise ValueError("Empty F5 training DataLoader")
        self.progress["batches_per_epoch"] = self.batches

    def load(self) -> int:
        t = self.trainer
        restored = self.store.restore(
            t.model,
            {"optimizer": t.optimizer},
            {"scheduler": t.scheduler},
            ema=t.ema_model,
        )
        if restored:
            progress, rank = restored
            if progress["batches_per_epoch"] != self.batches:
                raise ValueError("F5 DataLoader length changed across resume")
            self.progress, self.pending_rng = progress, rank["rng"]
        else:
            self.save()
        return self.progress["updates"]

    def batches_for_epoch(self, loader, epoch: int):
        if self.pending_rng is not None and self.progress["batch"] == 0:
            restore_rng(self.pending_rng, self.contract["backend"])
            self.pending_rng = None
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        return enumerate(loader, start=self.progress["batch"])

    def before_forward(self):
        if self.pending_rng is not None:
            restore_rng(self.pending_rng, self.contract["backend"])
            self.pending_rng = None

    def after_update(self, epoch: int, batch: int, updates: int):
        self.progress.update(epoch=epoch, batch=batch, updates=updates)

    def epoch_end(self, epoch: int):
        if self.progress["batch"] != self.batches:
            raise ValueError("F5 epoch ended before consuming its planned batches")
        self.progress.update(epoch=epoch, batch=0)
        self.save()

    def save(self):
        if self.pending_rng is not None:
            restore_rng(self.pending_rng, self.contract["backend"])
        t = self.trainer
        self.store.save(
            t.model,
            {"optimizer": t.optimizer},
            {"scheduler": t.scheduler},
            self.progress,
            ema=t.ema_model,
        )

    def finish(self) -> dict:
        t = self.trainer
        evidence = self.output / "evidence"
        checkpoint = evidence / "final_checkpoint.pt"
        atomic_torch_save(
            {
                "model_state_dict": t.accelerator.unwrap_model(t.model).state_dict(),
                "ema_model_state_dict": t.ema_model.state_dict(),
                "update": self.progress["updates"],
            },
            checkpoint,
        )
        report = {
            "schema_version": REPORT_SCHEMA,
            "status": "completed",
            "contract": self.contract,
            "contract_digest": canonical_digest(self.contract),
            "epochs_completed": self.progress["epoch"],
            "optimizer_updates": self.progress["updates"],
            "batches_per_epoch": self.batches,
            "stop_reason": "max_epochs",
            "checkpoint_selection": "final_ema",
            "checkpoints": {"final": checkpoint_identity(evidence, checkpoint)},
        }
        validate_completion(report, self.contract, evidence)
        atomic_json(evidence / "training_completion.json", report)
        return report


def f5_trainer_class(native):
    class DurableF5Trainer(native):
        def load_checkpoint(self):
            return self.sure_training.load()

        def save_checkpoint(self, update, last=False):
            if update != self.sure_training.progress["updates"]:
                raise ValueError("F5 native and durable update counters disagree")
            self.sure_training.save()

    return DurableF5Trainer


def broadcast_value(value, accelerator):
    if accelerator.num_processes == 1:
        return value
    import torch.distributed as distributed

    values = [value]
    distributed.broadcast_object_list(values, src=0, device=accelerator.device)
    return values[0]


def average_checkpoints(paths: list[Path], target: Path) -> None:
    import torch

    if len(paths) != 5:
        raise ValueError("Official DiariZen averaging requires five checkpoints")
    state = torch.load(paths[0], map_location="cpu", weights_only=True)
    for path in paths[1:]:
        other = torch.load(path, map_location="cpu", weights_only=True)
        if set(other) != set(state) or any(
            other[k].shape != state[k].shape for k in state
        ):
            raise ValueError("Cannot average checkpoints from different architectures")
        for key in state:
            state[key] += other[key]
    for key in state:
        state[key] = state[key] / len(paths)
    atomic_torch_save(state, target)


def sd_trainer_class(native):
    class DurableSdTrainer(native):
        # Native loss, dual optimizers, clipping, validation and early-stop comparison
        # are retained. State persistence and loop cursors are owned by this class.
        def _save_checkpoint(self, epoch, is_best_epoch):
            pass  # Save all ranks atomically below, after validation state is consistent.

        def auto_clip_grad_norm_(self, model):
            norm = self.compute_grad_norm(model)
            if not math.isfinite(norm):
                raise FloatingPointError("Nonfinite DiariZen gradient norm")
            return super().auto_clip_grad_norm_(model)

        def train_full(
            self, train_loader, validation_loader, output: Path, contract: dict
        ):
            import torch

            self.sure_output, self.sure_contract = output, contract
            self.sure_store = TrainingStateStore(
                output / "state", contract, self.accelerator
            )
            if self.accelerator.num_processes != contract["training"]["world_size"]:
                raise ValueError(
                    "DiariZen process count differs from the official recipe"
                )
            verify_training_device(self.model, self.accelerator, contract)
            batches = len(train_loader)
            if not batches or not len(validation_loader):
                raise ValueError("Training and validation DataLoaders must be nonempty")
            progress = new_progress()
            progress["batches_per_epoch"] = batches
            restored = self.sure_store.restore(
                self.model,
                {"wavlm": self.optimizer_small, "network": self.optimizer_big},
                {},
            )
            pending_rng = None
            if restored:
                progress, rank = restored
                if progress["batches_per_epoch"] != batches:
                    raise ValueError("DiariZen DataLoader changed across resume")
                self.grad_history = rank["extra"].get("grad_history", [])
                pending_rng = rank["rng"]
            self.state.epochs_trained, self.state.steps_trained = (
                progress["epoch"],
                progress["updates"],
            )
            best, patience, stopped = validation_state(
                progress["validation_history"], self.max_patience
            )
            self.state.best_score = best if best is not None else float("inf")
            self.state.patience = patience
            self.state.best_score_epoch = (
                min(progress["validation_history"], key=lambda r: r["loss"])["epoch"]
                if best is not None
                else 0
            )

            def save():
                self.sure_store.save(
                    self.model,
                    {"wavlm": self.optimizer_small, "network": self.optimizer_big},
                    {},
                    progress,
                    extra={"grad_history": self.grad_history},
                )

            if restored is None:
                save()
            for epoch in range(progress["epoch"], self.max_epochs):
                if stopped:
                    break
                self.model.train()
                train_loader.set_epoch(epoch)
                skip = progress["batch"]
                if pending_rng is not None and skip == 0:
                    restore_rng(pending_rng, contract["backend"])
                    pending_rng = None
                current = (
                    self.accelerator.skip_first_batches(train_loader, skip)
                    if skip
                    else train_loader
                )
                if hasattr(current, "set_epoch"):
                    current.set_epoch(epoch)
                for index, batch in enumerate(current, start=skip):
                    if pending_rng is not None:
                        restore_rng(pending_rng, contract["backend"])
                        pending_rng = None
                    with self.accelerator.accumulate(self.model):
                        loss = self.training_step(batch, index)
                    if loss is None or not torch.isfinite(loss["Loss"]).all():
                        raise FloatingPointError("Nonfinite DiariZen training loss")
                    if self.accelerator.optimizer_step_was_skipped:
                        raise RuntimeError(
                            "Optimizer skipped an update under the fixed FP32 protocol"
                        )
                    progress.update(
                        epoch=epoch, batch=index + 1, updates=progress["updates"] + 1
                    )
                    self.state.steps_trained = progress["updates"]
                    if (
                        progress["updates"]
                        % contract["training"]["checkpoint_every_updates"]
                        == 0
                    ):
                        save()
                if progress["batch"] != batches:
                    raise ValueError(
                        "DiariZen epoch did not consume all planned batches"
                    )
                if pending_rng is not None:
                    restore_rng(pending_rng, contract["backend"])
                    pending_rng = None
                self.state.epochs_trained = epoch + 1
                path = output / "epoch_weights" / f"epoch-{epoch + 1:04d}.pt"
                if self.accelerator.is_main_process:
                    atomic_torch_save(
                        self.accelerator.unwrap_model(self.model).state_dict(), path
                    )
                self.accelerator.wait_for_everyone()
                validation_loader.set_epoch(epoch)
                score = self.validate(validation_loader)
                score = broadcast_value(score, self.accelerator)
                if hasattr(self.unwrap_model, "validation_metric"):
                    self.unwrap_model.validation_metric.reset()
                if not math.isfinite(score):
                    raise FloatingPointError("Nonfinite validation loss")
                # Reuse the official patience comparison on every rank, without native disk writes.
                stopped = self._run_early_stop_check(score)
                progress["validation_history"].append(
                    {
                        "epoch": epoch + 1,
                        "loss": float(score),
                        "batches": len(validation_loader),
                    }
                )
                progress.update(epoch=epoch + 1, batch=0)
                save()
            if self.accelerator.is_main_process:
                self.finish_full(progress, output, contract)
            self.accelerator.wait_for_everyone()

        def finish_full(self, progress, output, contract):
            import shutil

            history = progress["validation_history"]
            _, patience, stopped = validation_state(history, self.max_patience)
            epochs = selected_epochs(history)
            evidence = output / "evidence"
            selected = evidence / "selected"
            selected.mkdir(parents=True, exist_ok=True)
            identities = {}
            paths = []
            for epoch in epochs:
                target = selected / f"epoch-{epoch:04d}.pt"
                shutil.copy2(output / "epoch_weights" / target.name, target)
                paths.append(target)
                identities[str(epoch)] = checkpoint_identity(evidence, target)
            target = evidence / "model" / "pytorch_model.bin"
            average_checkpoints(paths, target)
            identities["final"] = checkpoint_identity(evidence, target)
            report = {
                "schema_version": REPORT_SCHEMA,
                "status": "completed",
                "contract": contract,
                "contract_digest": canonical_digest(contract),
                "epochs_completed": progress["epoch"],
                "optimizer_updates": progress["updates"],
                "batches_per_epoch": progress["batches_per_epoch"],
                "stop_reason": "early_stop" if stopped else "max_epochs",
                "patience": patience,
                "validation_history": history,
                "validation_batches_per_epoch": history[0]["batches"],
                "selected_epochs": epochs,
                "checkpoint_selection": "best5_validation_loss",
                "checkpoints": identities,
            }
            validate_completion(report, contract, evidence)
            atomic_json(evidence / "training_completion.json", report)

    return DurableSdTrainer
