"""CPU-only checks of SD extension execution and optimizer/scheduler recovery."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from playground.sure_master.core.training import SD_TRAINING
from playground.sure_master.runtime import sd_evolution
from playground.sure_master.runtime.training_state import TrainingStateStore


@unittest.skipIf(torch is None, "Requires the pinned worker Torch environment")
class SdEvolutionRuntimeTests(unittest.TestCase):
    def test_bf16_autocast_audit_observes_real_linear_outputs(self):
        from playground.sure_master.runtime.f5_precision import install_precision_audit
        accelerator = SimpleNamespace(mixed_precision="bf16", scaler=None,
            unwrap_model=lambda m: m, process_index=0)
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = SimpleNamespace(model=model, accelerator=accelerator)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            install_precision_audit(trainer, output, {"precision": "bf16", "backend": "cpu"})
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = model(torch.ones(48, 4)).square().mean()
            loss.backward()
            optimizer.step()
            self.assertTrue(trainer.sure_precision_verified)
            self.assertTrue((output / "precision-rank-0.json").exists())
            self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))

    def test_custom_loss_updates_weights_and_exactly_restores_scheduler(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.wavlm_model = torch.nn.Linear(2, 2)
                self.network = torch.nn.Linear(2, 1)

            def forward(self, xs):
                return self.network(self.wavlm_model(xs))

            def non_wavlm_parameters(self):
                return self.network.parameters()

        training = {**deepcopy(SD_TRAINING), "recipe": sd_evolution.RECIPE,
                    "world_size": 1, "candidate_options": {}}
        torch.manual_seed(3407)
        model = Model()
        hooks = {
            "training_loss": lambda m, batch, config: (m(batch["xs"]) - batch["ys"]).square().mean(),
            "transform_batch": lambda batch: {**batch, "xs": batch["xs"] * 2},
            "build_schedulers": lambda opts, config, total: {
                key: torch.optim.lr_scheduler.StepLR(value, step_size=1, gamma=0.9)
                for key, value in opts.items()},
        }
        with patch.object(sd_evolution, "hook", side_effect=hooks.get):
            opts = sd_evolution.optimizers(model, training)
            schedules = sd_evolution.schedulers(opts, training, 10)
            accel = SimpleNamespace(is_main_process=True, num_processes=1, process_index=0,
                sync_gradients=True, backward=lambda loss: loss.backward(),
                unwrap_model=lambda m: m, wait_for_everyone=lambda: None)
            trainer = SimpleNamespace(model=model, optimizer_small=opts["wavlm"], optimizer_big=opts["network"],
                accelerator=accel, auto_clip_grad_norm_=lambda m: torch.nn.utils.clip_grad_norm_(m.parameters(), 1),
                sure_contract={"training": training})
            batch = {"xs": torch.ones(3, 2), "ys": torch.zeros(3, 1)}
            before = {key: value.clone() for key, value in model.state_dict().items()}
            def update():
                sd_evolution.training_step(trainer, batch, 0, lambda *args: self.fail("custom loss was ignored"))
                for scheduler in schedules.values():
                    scheduler.step()
            update()
            self.assertTrue(any(not torch.equal(before[k], v) for k, v in model.state_dict().items()))
            with tempfile.TemporaryDirectory() as temporary:
                store = TrainingStateStore(Path(temporary), {"backend": "cpu", "training": training}, accel)
                store.save(model, opts, schedules, {"epoch": 0, "batch": 1, "updates": 1})
                update()
                expected = {key: value.clone() for key, value in model.state_dict().items()}
                expected_schedulers = {key: deepcopy(value.state_dict()) for key, value in schedules.items()}
                progress, _ = store.restore(model, opts, schedules)
                self.assertEqual(progress["updates"], 1)
                update()
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
                self.assertEqual({key: value.state_dict() for key, value in schedules.items()}, expected_schedulers)

    def test_invalid_custom_optimizer_layout_is_rejected(self):
        with patch.object(sd_evolution, "hook", return_value=lambda *args: {"single": object()}):
            with self.assertRaisesRegex(ValueError, "wavlm and network"):
                sd_evolution.optimizers(object(), {})


if __name__ == "__main__":
    unittest.main()
