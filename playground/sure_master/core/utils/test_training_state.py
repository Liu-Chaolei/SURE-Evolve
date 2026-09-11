"""Small CPU models prove optimizer/RNG restoration; no dataset or full training run."""

from __future__ import annotations
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
    import numpy as np
except ImportError:
    torch = None

from playground.sure_master.runtime.training_state import (
    TrainingStateStore,
    restore_rng,
)
from playground.sure_master.runtime.official_trainers import average_checkpoints


@unittest.skipIf(torch is None, "Torch CPU test environment required")
class TrainingStateTests(unittest.TestCase):
    def accelerator(self):
        return SimpleNamespace(
            is_main_process=True,
            process_index=0,
            num_processes=1,
            wait_for_everyone=lambda: None,
            unwrap_model=lambda model: model,
        )

    def test_resume_matches_uninterrupted_updates_with_optimizer_scheduler_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            torch.manual_seed(42)
            random.seed(42)
            np.random.seed(42)
            model = torch.nn.Sequential(
                torch.nn.Linear(3, 4), torch.nn.Dropout(0.3), torch.nn.Linear(4, 1)
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

            def step():
                optimizer.zero_grad()
                x = torch.randn(2, 3) * (random.random() + np.random.rand())
                loss = model(x).square().mean()
                loss.backward()
                optimizer.step()
                scheduler.step()

            step()
            store = TrainingStateStore(
                Path(temporary), {"backend": "cpu", "epochs": 100}, self.accelerator()
            )
            store.save(
                model,
                {"optimizer": optimizer},
                {"scheduler": scheduler},
                {"epoch": 0, "batch": 1, "updates": 1},
            )
            step()
            step()
            expected = {k: v.clone() for k, v in model.state_dict().items()}
            restored_model = torch.nn.Sequential(
                torch.nn.Linear(3, 4), torch.nn.Dropout(0.3), torch.nn.Linear(4, 1)
            )
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.8)
            restored_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                restored_optimizer, 0.9
            )
            progress, rank = store.restore(
                restored_model,
                {"optimizer": restored_optimizer},
                {"scheduler": restored_scheduler},
            )
            self.assertEqual(progress["updates"], 1)
            model, optimizer, scheduler = (
                restored_model,
                restored_optimizer,
                restored_scheduler,
            )
            restore_rng(rank["rng"], "cpu")
            step()
            step()
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    def test_broken_newest_snapshot_falls_back_and_changed_contract_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            store = TrainingStateStore(
                root, {"backend": "cpu", "epoch_budget": 100}, self.accelerator()
            )
            first = store.save(model, {"optimizer": optimizer}, {}, {"updates": 1})
            last = store.save(model, {"optimizer": optimizer}, {}, {"updates": 2})
            (last / "training.pt").write_bytes(b"partial write")
            self.assertEqual(store.latest()[0], first)
            with self.assertRaisesRegex(ValueError, "Cannot resume"):
                TrainingStateStore(
                    root, {"backend": "cpu", "epoch_budget": 1}, self.accelerator()
                )

    def test_top_five_weight_average_is_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index in range(5):
                path = root / f"{index}.pt"
                torch.save({"weight": torch.tensor([float(index)])}, path)
                paths.append(path)
            average_checkpoints(paths, root / "average.pt")
            state = torch.load(root / "average.pt", weights_only=True)
            torch.testing.assert_close(state["weight"], torch.tensor([2.0]))


# Executable fixtures for the actual patched upstream F5 loop. These use tiny CPU
# tensors and a deterministic frame sampler, not the F5 model or real datasets.
class _ProgressBar:
    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args):
        pass

    def set_postfix(self, **kwargs):
        pass


class _FrameSampler:
    def __init__(self, sampler, *args, **kwargs):
        self.seed = kwargs.get("random_seed", 666)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return 2

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(4, generator=generator).tolist()
        yield order[:2]
        yield order[2:]


class _Loader:
    def __init__(self, dataset, *, batch_sampler, **kwargs):
        self.batch_sampler = batch_sampler

    def __len__(self):
        return 2

    def set_epoch(self, epoch):
        self.batch_sampler.set_epoch(epoch)

    def __iter__(self):
        torch.rand(1)  # model iterator/worker creation's RNG consumption
        for ids in self.batch_sampler:
            yield {
                "mel": torch.tensor(ids, dtype=torch.float32).reshape(2, 1, 1),
                "text": ["a", "b"],
                "mel_lengths": torch.tensor([1, 1]),
            }


class _Skipped:
    def __init__(self, loader, count):
        self.loader = loader
        self.count = count

    def set_epoch(self, epoch):
        self.loader.set_epoch(epoch)

    def __iter__(self):
        import itertools

        return itertools.islice(iter(self.loader), self.count, None)


@unittest.skipIf(torch is None, "Torch CPU test environment required")
class NativeF5LoopTests(unittest.TestCase):
    def test_native_loop_resume_matches_at_mid_epoch_and_epoch_boundary(self):
        import ast
        import contextlib
        import math
        import os
        from playground.sure_master.runtime.official_trainers import (
            F5TrainingSession,
            f5_trainer_class,
        )
        from playground.sure_master.runtime.training_sources import (
            prepare_f5_training_source,
        )

        source = (
            Path(os.environ.get("SURE_F5_ROOT", "/shared/chaolei.liu/TTS/F5-TTS"))
            / "src/f5_tts/model/trainer.py"
        )
        if not source.exists():
            self.skipTest("Optional upstream source snapshot not available")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "source/src/f5_tts/model/trainer.py"
            target.parent.mkdir(parents=True)
            target.write_text(source.read_text())
            prepare_f5_training_source(root / "source")
            tree = ast.parse(target.read_text())
            method = next(
                m
                for c in tree.body
                if isinstance(c, ast.ClassDef) and c.name == "Trainer"
                for m in c.body
                if isinstance(m, ast.FunctionDef) and m.name == "train"
            )
            namespace = {
                "torch": torch,
                "math": math,
                "Dataset": object,
                "exists": lambda x: x is not None,
                "DataLoader": _Loader,
                "SequentialSampler": lambda x: x,
                "DynamicBatchSampler": _FrameSampler,
                "collate_fn": None,
                "tqdm": _ProgressBar,
                "LinearLR": torch.optim.lr_scheduler.LinearLR,
                "SequentialLR": torch.optim.lr_scheduler.SequentialLR,
            }
            exec(
                compile(
                    ast.Module(body=[method], type_ignores=[]), str(target), "exec"
                ),
                namespace,
            )

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(1, 1)
                    self.dropout = torch.nn.Dropout(0.3)

                def forward(self, mel, text, lens, noise_scheduler=None):
                    value = self.dropout(mel) * (1 + random.random() + np.random.rand())
                    return self.linear(value).square().mean(), None, None

            class EMA(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    object.__setattr__(self, "online", model)
                    self.register_buffer("value", model.linear.weight.detach().clone())

                def update(self):
                    self.value.mul_(0.9).add_(
                        self.online.linear.weight.detach(), alpha=0.1
                    )

            class Native:
                train = namespace["train"]

                def __init__(self):
                    self.accelerator = SimpleNamespace(
                        is_main_process=True,
                        is_local_main_process=True,
                        process_index=0,
                        num_processes=1,
                        device=torch.device("cpu"),
                        wait_for_everyone=lambda: None,
                        unwrap_model=lambda x: x,
                        prepare=lambda *x: x,
                        accumulate=lambda model: contextlib.nullcontext(),
                        sync_gradients=True,
                        skip_first_batches=lambda loader, num_batches: _Skipped(
                            loader, num_batches
                        ),
                        backward=lambda loss: loss.backward(),
                        clip_grad_norm_=torch.nn.utils.clip_grad_norm_,
                        log=lambda *a, **kw: None,
                        end_training=lambda: None,
                    )
                    self.model = Model()
                    self.ema_model = EMA(self.model)
                    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
                    self.epochs = 3
                    self.num_warmup_updates = 1
                    self.grad_accumulation_steps = 1
                    self.batch_size_type = "frame"
                    self.batch_size_per_gpu = 32
                    self.max_samples = 64
                    self.max_grad_norm = 1.0
                    self.last_per_updates = 1
                    self.save_per_updates = 50000
                    self.log_samples = False
                    self.logger = None
                    self.duration_predictor = None
                    self.noise_scheduler = None
                    self.is_main = True

            cls = f5_trainer_class(Native)

            def run(directory, interrupt=None):
                torch.manual_seed(42)
                random.seed(42)
                np.random.seed(42)
                trainer = cls()
                trainer.sure_training = F5TrainingSession(
                    trainer,
                    directory,
                    {
                        "backend": "cpu",
                        "training": {"world_size": 1},
                        "component_test": True,
                    },
                )
                save = trainer.save_checkpoint

                def intercepted(update, last=False):
                    save(update, last)
                    if update == interrupt:
                        raise RuntimeError("simulated preemption")

                trainer.save_checkpoint = intercepted
                trainer.train([0, 1, 2, 3], num_workers=1, resumable_with_seed=666)
                return trainer

            expected = run(root / "uninterrupted")
            for step in (3, 4):
                with self.assertRaisesRegex(RuntimeError, "preemption"):
                    run(root / f"resume-{step}", step)
                actual = run(root / f"resume-{step}")
                self.assertEqual(actual.sure_training.progress["epoch"], 3)
                self.assertEqual(actual.sure_training.progress["updates"], 6)
                for key, value in actual.model.state_dict().items():
                    torch.testing.assert_close(
                        value, expected.model.state_dict()[key], rtol=0, atol=0
                    )
                torch.testing.assert_close(
                    actual.ema_model.value, expected.ema_model.value, rtol=0, atol=0
                )


@unittest.skipIf(torch is None, "Torch CPU test environment required")
class SdLoopTests(unittest.TestCase):
    def test_dual_optimizer_loop_resume_preserves_validation_and_weights(self):
        import contextlib
        from playground.sure_master.runtime.official_trainers import sd_trainer_class
        from unittest.mock import patch

        class Native:
            def __init__(self):
                self.model = torch.nn.Sequential(
                    torch.nn.Linear(1, 2), torch.nn.Dropout(0.2), torch.nn.Linear(2, 1)
                )
                self.unwrap_model = self.model
                self.optimizer_small = torch.optim.AdamW(
                    self.model[0].parameters(), lr=0.01
                )
                self.optimizer_big = torch.optim.AdamW(
                    self.model[2].parameters(), lr=0.02
                )
                self.accelerator = SimpleNamespace(
                    is_main_process=True,
                    is_local_main_process=True,
                    num_processes=1,
                    process_index=0,
                    device=torch.device("cpu"),
                    wait_for_everyone=lambda: None,
                    unwrap_model=lambda x: x,
                    skip_first_batches=lambda loader, count: _Skipped(loader, count),
                    accumulate=lambda model: contextlib.nullcontext(),
                    optimizer_step_was_skipped=False,
                )
                self.state = SimpleNamespace()
                self.grad_history = []
                self.max_epochs = 3
                self.max_patience = 10

            def training_step(self, batch, index):
                self.optimizer_small.zero_grad()
                self.optimizer_big.zero_grad()
                output = self.model(batch["mel"]) * (
                    1 + random.random() + np.random.rand()
                )
                loss = output.square().mean()
                loss.backward()
                self.optimizer_small.step()
                self.optimizer_big.step()
                return {"Loss": loss.detach()}

            def validate(self, loader):
                self.model.eval()
                with torch.no_grad():
                    return sum(
                        float(self.model(batch["mel"]).square().mean())
                        for batch in loader
                    ) / len(loader)

            def _run_early_stop_check(self, score):
                if score < self.state.best_score:
                    self.state.best_score = score
                    self.state.patience = 0
                else:
                    self.state.patience += 1
                return self.state.patience >= self.max_patience

        cls = sd_trainer_class(Native)

        class Probe(cls):
            def finish_full(self, progress, output, contract):
                self.final_progress = progress.copy()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def run(path, interrupt=None):
                torch.manual_seed(42)
                random.seed(42)
                np.random.seed(42)
                trainer = Probe()
                save = TrainingStateStore.save

                def intercepted(store, *args, **kwargs):
                    result = save(store, *args, **kwargs)
                    progress = args[3]
                    if progress["updates"] == interrupt and progress["batch"] > 0:
                        raise RuntimeError("simulated preemption")
                    return result

                with patch.object(TrainingStateStore, "save", intercepted):
                    trainer.train_full(
                        _Loader(None, batch_sampler=_FrameSampler(None)),
                        _Loader(None, batch_sampler=_FrameSampler(None)),
                        path,
                        {
                            "backend": "cpu",
                            "training": {
                                "world_size": 1,
                                "checkpoint_every_updates": 1,
                            },
                            "component_test": True,
                        },
                    )
                return trainer

            expected = run(root / "reference")
            for step in (3, 4):
                with self.assertRaisesRegex(RuntimeError, "preemption"):
                    run(root / f"resume-{step}", step)
                actual = run(root / f"resume-{step}")
                self.assertEqual(actual.final_progress, expected.final_progress)
                for name, value in actual.model.state_dict().items():
                    torch.testing.assert_close(
                        value, expected.model.state_dict()[name], rtol=0, atol=0
                    )


def _distributed_snapshot_worker(rank, directory):
    import datetime
    import torch.distributed as dist

    root = Path(directory)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / 'rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        torch.manual_seed(42)
        model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(2, 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        model(torch.ones(2, 2) * (rank + 1)).square().mean().backward()
        optimizer.step()
        expected = {k: v.clone() for k, v in model.module.state_dict().items()}
        accelerator = SimpleNamespace(
            is_main_process=rank == 0,
            process_index=rank,
            num_processes=2,
            wait_for_everyone=dist.barrier,
            unwrap_model=lambda m: m.module,
        )
        store = TrainingStateStore(
            root / "state", {"backend": "cpu", "world_size": 2}, accelerator
        )
        store.save(
            model,
            {"optimizer": optimizer},
            {},
            {"epoch": 0, "batch": 1, "updates": 1},
            extra={"rank": rank},
        )
        for p in model.parameters():
            p.data.zero_()
        progress, saved = store.restore(model, {"optimizer": optimizer}, {})
        assert saved["extra"]["rank"] == rank and progress["updates"] == 1
        for name, value in model.module.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@unittest.skipIf(torch is None, "Torch CPU test environment required")
class DistributedSnapshotTests(unittest.TestCase):
    def test_each_rank_saves_and_restores_its_state(self):
        if not torch.distributed.is_gloo_available():
            self.skipTest("Gloo unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            torch.multiprocessing.spawn(
                _distributed_snapshot_worker, args=(temporary,), nprocs=2, join=True
            )


if __name__ == "__main__":
    unittest.main()
