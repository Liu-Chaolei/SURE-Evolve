"""Canonical serialization, digests, redaction, and portable path checks."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_SECRET_KEY = re.compile(r"(api[_-]?key|token|secret|password|credential|authorization)", re.I)
_SECRET_VALUE = re.compile(r"(?i)(sk-[A-Za-z0-9_-]{12,}|bearer\s+[A-Za-z0-9._-]{12,})")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if _SECRET_KEY.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _SECRET_VALUE.sub("[REDACTED]", value)
    return value


def portable_path(path: str | Path, workspace: str | Path) -> str:
    """Return a workspace-relative path and reject host-specific references."""
    candidate = Path(path)
    root = Path(workspace).resolve()
    if candidate.is_absolute():
        raise ValueError("absolute paths are not portable artifacts")
    resolved = (root / candidate).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("path escapes workspace") from exc
    if str(relative) in ("", "."):
        raise ValueError("workspace root is not an artifact path")
    return relative.as_posix()
