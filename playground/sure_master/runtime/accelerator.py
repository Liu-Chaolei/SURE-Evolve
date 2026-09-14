"""Task-independent accelerator configuration (model imports remain in workers)."""
from __future__ import annotations

import os
import random
from typing import Any

# Backward-compatible imports for the existing Zipformer wrappers.
from .icefall import prepare_cpu_recipe as prepare_cpu_recipe, prepare_npu_recipe as prepare_npu_recipe


def runtime_environment(config: dict[str, Any]) -> dict[str, str]:
    if not config:
        return {}
    backend = str(config.get("accelerator", "cuda"))
    if backend not in {"cuda", "npu", "cpu"}:
        raise ValueError("sure.runtime.accelerator must be cuda, npu, or cpu")
    devices = config.get("devices", ["0"])
    devices = [str(d) for d in devices] if isinstance(devices, list) else str(devices).split(",")
    precision = str(config.get("precision", "fp32"))
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("Unsupported precision")
    env = {"SURE_ACCELERATOR": backend, "SURE_PRECISION": precision,
           "SURE_CPU_THREADS": str(max(1, min(32, int(config.get("cpu_threads", 4)))))}
    if config.get("training_precision"):
        training_precision = str(config["training_precision"])
        if training_precision not in {"fp32", "bf16"}:
            raise ValueError("Unsupported training precision")
        env["SURE_TRAIN_PRECISION"] = training_precision
    if config.get("python"):
        env["SURE_WORKER_PYTHON"] = str(config["python"])
    if backend == "npu":
        # slurm-docker-run owns device visibility and maps it to local ranks.
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    elif backend == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    else:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    return env


class RuntimeBackend:
    def __init__(self, name: str | None = None):
        import torch
        self.torch = torch
        self.name = name or os.environ.get("SURE_ACCELERATOR", "cpu")
        if self.name == "npu":
            import torch_npu  # noqa: F401
        if self.name not in {"cpu", "cuda", "npu"}:
            raise ValueError(f"Unsupported backend: {self.name}")
        if self.name != "cpu":
            module = getattr(torch, self.name)
            if not module.is_available():
                raise RuntimeError(f"Requested {self.name} is unavailable; CPU fallback is disabled")
            module.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        self.device = torch.device("cpu" if self.name == "cpu" else f"{self.name}:{int(os.environ.get('LOCAL_RANK', '0'))}")
        torch.set_num_threads(int(os.environ.get("SURE_CPU_THREADS", "4")))

    def seed(self, seed: int) -> None:
        random.seed(seed)
        import numpy as np
        np.random.seed(seed)
        self.torch.manual_seed(seed)
        if self.name != "cpu":
            getattr(self.torch, self.name).manual_seed_all(seed)

    def synchronize(self) -> None:
        if self.name != "cpu":
            getattr(self.torch, self.name).synchronize()

    def empty_cache(self) -> None:
        if self.name != "cpu":
            getattr(self.torch, self.name).empty_cache()

    def verify_model(self, model, *, gradients: bool = False) -> None:
        parameters = [p for p in model.parameters() if p.requires_grad] if gradients else list(model.parameters())
        if not parameters or any(p.device.type != self.name for p in parameters):
            raise RuntimeError(f"Model parameters are not all on {self.name}")
        if gradients and not any(p.grad is not None and self.torch.isfinite(p.grad).all() for p in parameters):
            raise RuntimeError("No finite model gradients on requested accelerator")


def check_accelerator(backend: str) -> dict[str, Any]:
    runtime = RuntimeBackend(backend)
    torch = runtime.torch
    value = torch.ones(2, device=runtime.device, requires_grad=True)
    value.to("cpu").square().sum().to(runtime.device).backward()
    if value.grad is None or not torch.isfinite(value.grad).all():
        raise RuntimeError("Cross-device autograd check failed")
    runtime.synchronize()
    return {"accelerator": backend, "torch": torch.__version__, "device": str(value.device)}
