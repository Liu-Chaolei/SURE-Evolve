"""Bounded worker processes, including cleanup of their subprocess trees."""

from __future__ import annotations

import os
import signal
import subprocess
from typing import IO, Mapping, Sequence


def run_bounded(
    command: Sequence[str],
    *,
    timeout: float,
    output: IO,
    env: Mapping[str, str] | None = None,
) -> None:
    process = subprocess.Popen(
        list(command),
        env=env,
        stdout=output,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        return_code = process.wait(timeout=timeout)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            finally:
                # Parent exit does not imply all workers exited.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        except ProcessLookupError:
            pass
        process.wait()
        raise
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
