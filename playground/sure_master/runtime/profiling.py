"""Opt-in real training throughput measurement for the isolated calibration job."""

from __future__ import annotations
import json
import os
import time
from pathlib import Path

_started = None
_audio_seconds = 0.0


def profile_step(batch, step, rank, world_size):
    global _started, _audio_seconds
    count = int(os.environ.get("SURE_PROFILE_STEPS", "0"))
    if not count:
        return
    import torch

    warmup = 20
    torch.npu.synchronize()
    if step == warmup:
        _started = time.monotonic()
        torch.npu.reset_peak_memory_stats()
    elif step > warmup and _started is not None:
        _audio_seconds += float(batch["supervisions"]["num_frames"].sum()) / 100
    if step < warmup + count:
        return
    elapsed = time.monotonic() - _started
    memory = torch.npu.max_memory_allocated()
    capacity = torch.npu.get_device_properties(rank).total_memory
    stats = torch.tensor(
        [_audio_seconds, elapsed, memory / capacity], device=f"npu:{rank}"
    )
    gathered = [torch.zeros_like(stats) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, stats)
    if rank == 0:
        values = [x.cpu().tolist() for x in gathered]
        duration = max(x[1] for x in values)
        report = {
            "steps": count,
            "world_size": world_size,
            "elapsed_seconds": duration,
            "audio_seconds": sum(x[0] for x in values),
            "audio_hours_per_hour": sum(x[0] for x in values) / duration,
            "peak_memory_fraction": max(x[2] for x in values),
            "ranks": values,
        }
        Path(os.environ["SURE_PROFILE_OUTPUT"]).write_text(
            json.dumps(report, indent=2) + "\n"
        )
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    raise SystemExit(0)
