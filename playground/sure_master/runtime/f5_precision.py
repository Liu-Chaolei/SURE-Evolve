"""Verify actual autocast execution while retaining FP32 master state."""

from pathlib import Path

from .training_state import atomic_json


def install_precision_audit(trainer, output: Path, contract: dict) -> None:
    import torch

    precision = contract.get("precision", "fp32")
    expected = "bf16" if precision == "bf16" else "no"
    accelerator = trainer.accelerator
    if accelerator.mixed_precision != expected:
        raise RuntimeError("Accelerator mixed precision differs from the training contract")
    if accelerator.scaler is not None:
        raise RuntimeError("FP32/BF16 training must not use a GradScaler")
    if precision == "bf16" and contract["backend"] == "npu" and not torch.npu.is_bf16_supported():
        raise RuntimeError("This NPU runtime does not support BF16")
    model = accelerator.unwrap_model(trainer.model)
    master_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    if master_dtypes != ["torch.float32"]:
        raise RuntimeError("Training master parameters must remain FP32")
    target = torch.bfloat16 if precision == "bf16" else torch.float32
    handles = []

    def capture(module, inputs, result):
        if isinstance(result, torch.Tensor) and result.dtype == target:
            trainer.sure_precision_verified = True
            atomic_json(output / f"precision-rank-{accelerator.process_index}.json", {
                "mixed_precision": precision,
                "master_parameter_dtypes": master_dtypes,
                "observed_linear_output_dtype": str(result.dtype),
                "grad_scaler": False,
                "rank": accelerator.process_index,
            })
            for handle in handles:
                handle.remove()

    trainer.sure_precision_verified = False
    for module in [m for m in model.modules() if isinstance(m, torch.nn.Linear)][:3]:
        handles.append(module.register_forward_hook(capture))
