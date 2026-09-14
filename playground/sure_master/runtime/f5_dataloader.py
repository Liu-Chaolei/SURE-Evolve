"""Start audio workers without inheriting initialized Torch/NPU thread pools."""

from __future__ import annotations


def initialize_audio_worker(worker_id: int) -> None:
    import torch

    torch.set_num_threads(1)


def dataloader_options(num_workers: int) -> dict:
    if num_workers <= 0:
        return {}
    return {
        "multiprocessing_context": "spawn",
        "worker_init_fn": initialize_audio_worker,
        "timeout": 180,
    }
