"""Watchdog module for SURE Master timeout control."""

from __future__ import annotations

import ctypes
import threading

RUN_TIMEOUT_SECONDS = 96 * 60 * 60


class GlobalTimeoutInterrupt(BaseException):
    """Global timeout exception injected into the main thread."""


def _async_raise(target_tid, exception_type):
    ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(target_tid),
        ctypes.py_object(exception_type),
    )
    if ret == 0:
        raise ValueError("Invalid thread ID")
    if ret > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(target_tid), None)
        raise SystemError("PyThreadState_SetAsyncExc call failed")


class TimeoutWatchdog:
    """Daemon watchdog that interrupts the main thread after a hard timeout."""

    def __init__(self, timeout_seconds: int):
        self.timeout_seconds = timeout_seconds
        self.cancel_event = threading.Event()
        self.main_thread_id = threading.get_ident()
        self._thread = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._watch,
            daemon=True,
            name="SureMasterTimeoutWatchdog",
        )
        self._thread.start()

    def _watch(self) -> None:
        if not self.cancel_event.wait(self.timeout_seconds):
            _async_raise(self.main_thread_id, GlobalTimeoutInterrupt)

    def stop(self) -> None:
        self.cancel_event.set()

