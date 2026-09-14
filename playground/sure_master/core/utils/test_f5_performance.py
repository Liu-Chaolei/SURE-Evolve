"""The combined finite check must still block invalid optimizer updates."""

import unittest


class F5PerformanceTests(unittest.TestCase):
    def test_nonfinite_values_prevent_parameter_updates(self):
        import torch
        from playground.sure_master.runtime.f5_performance import check_update_finite

        for loss, norm in ((float("nan"), 1.0), (1.0, float("inf")),
                           (float("inf"), None), (1.0, float("nan"))):
            with self.subTest(loss=loss, norm=norm):
                parameter = torch.nn.Parameter(torch.tensor(1.0))
                parameter.grad = torch.tensor(2.0)
                optimizer = torch.optim.AdamW([parameter])
                with self.assertRaises(FloatingPointError):
                    check_update_finite(torch.tensor(loss), None if norm is None else torch.tensor(norm))
                    optimizer.step()
                self.assertEqual(parameter.item(), 1.0)
                self.assertFalse(optimizer.state)

    def test_valid_checks_allow_the_same_update(self):
        import torch
        from playground.sure_master.runtime.f5_performance import check_update_finite

        for norm in (None, torch.tensor(1.0)):
            parameter = torch.nn.Parameter(torch.tensor(1.0))
            parameter.grad = torch.tensor(2.0)
            optimizer = torch.optim.AdamW([parameter])
            check_update_finite(torch.tensor(0.5), norm)
            optimizer.step()
            self.assertLess(parameter.item(), 1.0)
