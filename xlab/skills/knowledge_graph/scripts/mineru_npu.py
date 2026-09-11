#!/usr/bin/env python3
"""Run pinned MinerU with Ascend settings inherited by spawned workers."""
from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import onnxruntime
import torch_npu


# MinerU uses spawn: keep this initialization at module scope so every worker
# disables online operator compilation before constructing its models.
for cache_variable in ("ASCEND_PROCESS_LOG_PATH", "ASCEND_CACHE_PATH"):
    if os.environ.get(cache_variable):
        Path(os.environ[cache_variable]).mkdir(parents=True, exist_ok=True)
torch_npu.npu.set_compile_mode(jit_compile=False)
os.environ["MINERU_LOCAL_API_LAUNCH_MODE"] = "spawn"

# Some MinerU table classifiers construct ONNX sessions with default thread
# pools sized for the entire host. Bound those pools to this worker allocation.
_session_init = onnxruntime.InferenceSession.__init__


def _bounded_session_init(
    self: onnxruntime.InferenceSession,
    path_or_bytes: str | bytes | os.PathLike,
    sess_options: onnxruntime.SessionOptions | None = None,
    providers: Sequence[str | tuple[str, dict[str, object]]] | None = None,
    provider_options: Sequence[dict[str, object]] | None = None,
    **kwargs: object,
) -> None:
    options = sess_options or onnxruntime.SessionOptions()
    allocated = len(os.sched_getaffinity(0))
    threads = min(max(int(os.environ.get("OMP_NUM_THREADS", "16")), 1), allocated)
    options.intra_op_num_threads = min(options.intra_op_num_threads or threads, threads)
    options.inter_op_num_threads = min(options.inter_op_num_threads or 1, threads)
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    _session_init(self, path_or_bytes, options, providers, provider_options, **kwargs)


onnxruntime.InferenceSession.__init__ = _bounded_session_init

# Importing the CLI constructs Magika's ONNX session, so install the resource
# limits before that module is loaded (including in spawned processes).
from mineru.cli.client import main


if __name__ == "__main__":
    main()
