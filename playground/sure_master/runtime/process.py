"""Bounded worker processes, including cleanup of their subprocess trees."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from typing import IO, Mapping, Sequence


def run_bounded(
    command: Sequence[str],
    *,
    timeout: float | None,
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
    previous_handler = None
    if threading.current_thread() is threading.main_thread():
        previous_handler = signal.getsignal(signal.SIGTERM)
        def interrupted(_signum, _frame):
            raise KeyboardInterrupt("Worker process interrupted")
        signal.signal(signal.SIGTERM, interrupted)
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
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGTERM, previous_handler)
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
