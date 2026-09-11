"""Explicit NPU/CPU bridge for the TEDLIUM Zipformer recipe.

Only k2 loss/pruning bookkeeping runs on CPU. Tensor.to preserves autograd;
never detach the acoustic, language-model, or joiner outputs at this boundary.
"""
from __future__ import annotations

import k2
import torch
import torch.nn.functional as F


def swoosh_l(x):
    return F.softplus(x - 4.0) - 0.08 * x - 0.035


def swoosh_r(x):
    return F.softplus(x - 1.0) - 0.08 * x - 0.313261687


swoosh_l_forward = swoosh_l
swoosh_r_forward = swoosh_r


def swoosh_l_forward_and_deriv(x):
    return swoosh_l(x), torch.sigmoid(x - 4.0) - 0.08


def swoosh_r_forward_and_deriv(x):
    return swoosh_r(x), torch.sigmoid(x - 1.0) - 0.08


def _cpu_kwargs(kwargs):
    return {key: value.to("cpu") if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()}


def rnnt_loss_smoothed(**kwargs):
    device = kwargs["am"].device
    result = k2.rnnt_loss_smoothed(**_cpu_kwargs(kwargs))
    if isinstance(result, tuple):
        loss, gradients = result
        return loss.to(device), tuple(g.to(device) for g in gradients)
    return result.to(device)


def get_rnnt_prune_ranges(**kwargs):
    device = kwargs["px_grad"].device
    return k2.get_rnnt_prune_ranges(**_cpu_kwargs(kwargs)).to(device)


def rnnt_loss_pruned(**kwargs):
    device = kwargs["logits"].device
    return k2.rnnt_loss_pruned(**_cpu_kwargs(kwargs)).to(device)
