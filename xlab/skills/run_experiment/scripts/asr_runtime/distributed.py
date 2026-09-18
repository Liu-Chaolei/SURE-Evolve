"""Single-node Ascend DDP; visibility belongs to slurm-docker-run."""

from __future__ import annotations
import os
from datetime import timedelta


def setup_dist(rank=None, world_size=None, master_port=None, **kwargs):
    import torch
    import torch_npu  # noqa: F401

    rank = int(rank if rank is not None else os.environ.get("LOCAL_RANK", 0))
    size = int(world_size if world_size is not None else os.environ["WORLD_SIZE"])
    if torch.npu.device_count() < size:
        raise RuntimeError(
            f"Requested {size} ranks but only {torch.npu.device_count()} allocated NPUs are visible"
        )
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(master_port or 29500))
    os.environ["RANK"], os.environ["WORLD_SIZE"] = str(rank), str(size)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.npu.set_device(rank)
    torch.distributed.init_process_group(
        "hccl", rank=rank, world_size=size, timeout=timedelta(minutes=10)
    )


def cleanup_dist():
    import torch

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

