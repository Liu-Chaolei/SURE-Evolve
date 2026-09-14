"""Reduce host synchronization without changing the F5 optimizer update rule."""

from __future__ import annotations

from pathlib import Path


def check_update_finite(loss, grad_norm=None) -> None:
    """Check loss and clipped gradient norm together, before optimizer.step()."""
    import torch

    finite = torch.isfinite(loss.detach()).all()
    if grad_norm is not None:
        finite = finite & torch.isfinite(grad_norm.detach()).all()
    if not finite:
        raise FloatingPointError("Nonfinite F5 training loss or gradient norm")


def optimize_trainer_source(path: Path) -> None:
    from .training_sources import replace_once

    text = path.read_text()
    if "check_update_finite(loss, grad_norm)" in text:
        return
    text = replace_once(
        text, "from tqdm import tqdm",
        "from tqdm import tqdm\n"
        "from playground.sure_master.runtime.f5_performance import check_update_finite",
    )
    text = replace_once(
        text,
        '                    if not torch.isfinite(loss).all():\n'
        '                        raise FloatingPointError("Nonfinite F5 training loss")\n',
        "                    grad_norm = None\n",
    )
    text = replace_once(
        text,
        '                        if not torch.isfinite(grad_norm).all():\n'
        '                            raise FloatingPointError("Nonfinite F5 gradients")\n',
        "",
    )
    text = replace_once(
        text, "                    self.optimizer.step()",
        "                    check_update_finite(loss, grad_norm)\n"
        "                    self.optimizer.step()",
    )
    text = replace_once(
        text, "                if self.accelerator.sync_gradients:\n",
        "                log_now = False\n                if self.accelerator.sync_gradients:\n",
    )
    text = replace_once(
        text, "                    progress_bar.set_postfix(update=str(global_update), loss=loss.item())",
        "                    log_now = global_update == 1 or global_update % 50 == 0\n"
        "                    if log_now:\n"
        "                        loss_value = loss.item()\n"
        "                        progress_bar.set_postfix(update=str(global_update), loss=loss_value)",
    )
    text = replace_once(
        text, "                if self.accelerator.is_local_main_process:\n",
        "                if self.accelerator.is_local_main_process and log_now:\n",
    )
    text = text.replace('{"loss": loss.item(), "lr":', '{"loss": loss_value, "lr":')
    text = text.replace(
        'if self.logger == "tensorboard" and self.accelerator.is_main_process:',
        'if self.logger == "tensorboard" and self.accelerator.is_main_process and log_now:',
    )
    text = text.replace(
        'self.writer.add_scalar("loss", loss.item(), global_update)',
        'self.writer.add_scalar("loss", loss_value, global_update)',
    )
    import ast
    ast.parse(text)
    path.write_text(text)
