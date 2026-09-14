"""Numerical equivalence with the official pretrained EMA initialization."""

import tempfile
from pathlib import Path
import unittest


class F5EMATests(unittest.TestCase):
    def test_pretrained_state_and_first_updates_match_official(self):
        import torch
        from ema_pytorch import EMA
        from safetensors.torch import save_file
        from playground.sure_master.tools.run_official_training import load_initial_f5_ema

        torch.manual_seed(42)
        model = torch.nn.Linear(3, 2)
        initial = EMA(model, include_online_model=False)
        initial.initted.fill_(True)
        initial.step.fill_(1250000)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'pretrained.safetensors'
            save_file(initial.state_dict(), str(path))
            official = EMA(model, include_online_model=False)
            official.load_state_dict(initial.state_dict())
            actual = EMA(model, include_online_model=False)
            report = load_initial_f5_ema(actual, path, structural=False)
            self.assertEqual(report['step'], 1250000)
            self.assertTrue(report['initted'])
            for _ in range(20):
                with torch.no_grad():
                    model.weight.add_(0.01)
                official.update()
                actual.update()
                for key, expected in official.state_dict().items():
                    self.assertTrue(torch.equal(expected, actual.state_dict()[key]), key)

    def test_structural_new_parameters_keep_initialization(self):
        import torch
        from ema_pytorch import EMA
        from safetensors.torch import save_file
        from playground.sure_master.tools.run_official_training import load_initial_f5_ema

        pretrained = EMA(torch.nn.Linear(3, 2), include_online_model=False)
        pretrained.initted.fill_(True)
        pretrained.step.fill_(1250000)
        target = EMA(torch.nn.Linear(4, 2), include_online_model=False)
        new_weight = target.ema_model.weight.detach().clone()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'pretrained.safetensors'
            save_file(pretrained.state_dict(), str(path))
            load_initial_f5_ema(target, path, structural=True)
        self.assertTrue(torch.equal(new_weight, target.ema_model.weight))
        self.assertTrue(torch.equal(pretrained.ema_model.bias, target.ema_model.bias))
        self.assertEqual(target.step.item(), 1250000)
