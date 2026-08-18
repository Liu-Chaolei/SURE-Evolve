"""Shared cache path helpers for SURE-EVAL.

Public code must not default to a user-specific HPC path. Runtime assets can
still live on large local storage by setting ``SURE_EVAL_CACHE_DIR``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

CACHE_ENV_VAR = "SURE_EVAL_CACHE_DIR"


def get_cache_root(*, create: bool = True) -> Path:
    """Return the root cache directory for local runtime artifacts."""

    raw = os.environ.get(CACHE_ENV_VAR)
    root = Path(raw).expanduser() if raw else Path.home() / ".cache" / "sure-eval"
    if create:
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError:
            if raw:
                raise
            root = Path(tempfile.gettempdir()) / f"sure-eval-cache-{os.getuid()}"
            root.mkdir(parents=True, exist_ok=True)
    return root


def get_cache_dir(*parts: str, create: bool = True) -> Path:
    """Return a cache subdirectory under ``SURE_EVAL_CACHE_DIR``."""

    raw = os.environ.get(CACHE_ENV_VAR)
    path = get_cache_root(create=create).joinpath(*parts)
    if create:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            if raw:
                raise
            path = Path(tempfile.gettempdir()) / f"sure-eval-cache-{os.getuid()}" / Path(*parts)
            path.mkdir(parents=True, exist_ok=True)
    return path
