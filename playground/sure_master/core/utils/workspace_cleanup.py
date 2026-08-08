from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TRUTHY = {"1", "true", "yes", "y", "on"}
FALSY = {"0", "false", "no", "n", "off"}


@dataclass
class WorkspaceCleanupConfig:
    """Retention policy for one candidate workspace."""

    enabled: bool = False
    on_success: bool = True
    on_failure: bool = False
    remove_models: bool = True
    remove_working: bool = True
    remove_runtime_cache: bool = True
    compact_metric_logs: bool = True
    keep_log_tails: bool = True
    log_tail_bytes: int = 65536

    def should_cleanup(self, *, success: bool) -> bool:
        if not self.enabled:
            return False
        return self.on_success if success else self.on_failure


def workspace_cleanup_config(
    config: Any,
    execution_env: dict[str, str] | None = None,
) -> WorkspaceCleanupConfig:
    """Build a cleanup config from sure.workspace_cleanup and runtime env."""
    raw_config = _sure_workspace_cleanup_config(config)
    resolved = WorkspaceCleanupConfig(
        enabled=_bool(raw_config.get("enabled"), False),
        on_success=_bool(raw_config.get("on_success"), True),
        on_failure=_bool(raw_config.get("on_failure"), False),
        remove_models=_bool(raw_config.get("remove_models"), True),
        remove_working=_bool(raw_config.get("remove_working"), True),
        remove_runtime_cache=_bool(raw_config.get("remove_runtime_cache"), True),
        compact_metric_logs=_bool(raw_config.get("compact_metric_logs"), True),
        keep_log_tails=_bool(raw_config.get("keep_log_tails"), True),
        log_tail_bytes=_positive_int(raw_config.get("log_tail_bytes"), 65536),
    )

    env = dict(os.environ)
    env.update(execution_env or {})
    _apply_bool_env(env, "SURE_WORKSPACE_CLEANUP_ENABLED", resolved, "enabled")
    _apply_bool_env(env, "SURE_WORKSPACE_CLEANUP_ON_SUCCESS", resolved, "on_success")
    _apply_bool_env(env, "SURE_WORKSPACE_CLEANUP_ON_FAILURE", resolved, "on_failure")
    _apply_bool_env(env, "SURE_WORKSPACE_CLEANUP_REMOVE_MODELS", resolved, "remove_models")
    _apply_bool_env(env, "SURE_WORKSPACE_CLEANUP_REMOVE_WORKING", resolved, "remove_working")
    _apply_bool_env(
        env,
        "SURE_WORKSPACE_CLEANUP_REMOVE_RUNTIME_CACHE",
        resolved,
        "remove_runtime_cache",
    )
    _apply_bool_env(
        env,
        "SURE_WORKSPACE_CLEANUP_COMPACT_METRIC_LOGS",
        resolved,
        "compact_metric_logs",
    )
    _apply_bool_env(env, "SURE_WORKSPACE_CLEANUP_KEEP_LOG_TAILS", resolved, "keep_log_tails")
    if env.get("SURE_WORKSPACE_CLEANUP_LOG_TAIL_BYTES"):
        resolved.log_tail_bytes = _positive_int(
            env.get("SURE_WORKSPACE_CLEANUP_LOG_TAIL_BYTES"),
            resolved.log_tail_bytes,
        )
    return resolved


def cleanup_candidate_workspace(
    workspace: str | Path,
    cleanup_config: WorkspaceCleanupConfig,
    *,
    success: bool,
    reason_code: str,
    score: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compact large per-candidate outputs while keeping triage artifacts."""
    root = Path(workspace)
    if not cleanup_config.should_cleanup(success=success):
        return {
            "enabled": bool(cleanup_config.enabled),
            "cleaned": False,
            "reason": "policy_disabled_for_outcome",
        }

    root.mkdir(parents=True, exist_ok=True)
    metric_dir = root / "metric"
    metric_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "enabled": True,
        "cleaned": True,
        "workspace": str(root),
        "success": bool(success),
        "reason_code": reason_code,
        "score": score,
        "metadata": metadata or {},
        "config": asdict(cleanup_config),
        "log_tails": [],
        "removed_paths": [],
        "removed_bytes": 0,
        "errors": [],
        "written_at": int(time.time()),
    }

    if cleanup_config.keep_log_tails and cleanup_config.log_tail_bytes > 0:
        manifest["log_tails"] = _save_log_tails(root, cleanup_config.log_tail_bytes)

    if cleanup_config.compact_metric_logs:
        _remove_metric_logs(root, manifest)
    if cleanup_config.remove_models:
        _remove_workspace_entry(root, "models", manifest)
    if cleanup_config.remove_working:
        _remove_workspace_entry(root, "working", manifest)
    if cleanup_config.remove_runtime_cache:
        _remove_workspace_entry(root, ".sure_runtime", manifest)

    _write_json(metric_dir / "cleanup_manifest.json", manifest)
    return manifest


def _sure_workspace_cleanup_config(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        root = config
    elif hasattr(config, "model_dump"):
        root = config.model_dump()
    elif hasattr(config, "dict"):
        root = config.dict()
    else:
        root = {}
    sure_config = root.get("sure") or {}
    if not isinstance(sure_config, dict):
        return {}
    cleanup = sure_config.get("workspace_cleanup") or {}
    return cleanup if isinstance(cleanup, dict) else {}


def _apply_bool_env(
    env: dict[str, str],
    env_name: str,
    config: WorkspaceCleanupConfig,
    field_name: str,
) -> None:
    if env_name not in env:
        return
    value = _optional_bool(env.get(env_name))
    if value is not None:
        setattr(config, field_name, value)


def _bool(value: Any, default: bool) -> bool:
    parsed = _optional_bool(value)
    return default if parsed is None else parsed


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return None
    if text in TRUTHY:
        return True
    if text in FALSY:
        return False
    return None


def _positive_int(value: Any, default: int) -> int:
    try:
        return max(1, int(float(str(value).strip())))
    except (TypeError, ValueError):
        return default


def _save_log_tails(root: Path, limit_bytes: int) -> list[dict[str, Any]]:
    tail_dir = root / "metric" / "log_tails"
    entries: list[dict[str, Any]] = []
    for path in _candidate_log_paths(root):
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if "log_tails" in rel.parts:
            continue
        target = tail_dir / _tail_name(rel)
        try:
            text = _read_tail(path, limit_bytes)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            entries.append(
                {
                    "source": str(rel),
                    "tail": str(target.relative_to(root)),
                    "source_bytes": _path_size_bytes(path),
                    "tail_bytes": target.stat().st_size,
                }
            )
        except OSError as exc:
            entries.append({"source": str(rel), "error": str(exc)})
    return entries


def _candidate_log_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for log_root in (
        root / "working",
        root / "metric",
        root / ".sure_runtime" / "duration_autotune",
    ):
        if not log_root.exists() or log_root.is_symlink():
            continue
        try:
            for path in sorted(log_root.rglob("*.log")):
                if path.is_file() and not path.is_symlink():
                    paths.append(path)
        except OSError:
            continue
    return paths


def _tail_name(rel_path: Path) -> str:
    stem = "__".join(rel_path.parts)
    digest = hashlib.sha1(str(rel_path).encode("utf-8")).hexdigest()[:8]
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
    return f"{safe}.{digest}.tail.log"


def _read_tail(path: Path, limit_bytes: int) -> str:
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - limit_bytes), os.SEEK_SET)
        return f.read().decode("utf-8", errors="replace")


def _remove_metric_logs(root: Path, manifest: dict[str, Any]) -> None:
    metric_dir = root / "metric"
    if not metric_dir.exists() or metric_dir.is_symlink():
        return
    try:
        paths = sorted(metric_dir.glob("*.log"))
    except OSError as exc:
        manifest["errors"].append({"path": "metric", "error": str(exc)})
        return
    for path in paths:
        _remove_path(root, path, manifest)


def _remove_workspace_entry(root: Path, name: str, manifest: dict[str, Any]) -> None:
    _remove_path(root, root / name, manifest)


def _remove_path(root: Path, path: Path, manifest: dict[str, Any]) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if not _safe_workspace_entry(root, path):
        manifest["errors"].append(
            {
                "path": _relative_or_absolute(root, path),
                "error": "refusing to remove path outside workspace",
            }
        )
        return
    size = _path_size_bytes(path)
    rel = _relative_or_absolute(root, path)
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError as exc:
        manifest["errors"].append({"path": rel, "error": str(exc)})
        return
    manifest["removed_paths"].append({"path": rel, "bytes": size})
    manifest["removed_bytes"] = int(manifest.get("removed_bytes") or 0) + size


def _safe_workspace_entry(root: Path, path: Path) -> bool:
    try:
        root_abs = root.absolute()
        path_abs = path.absolute()
        path_abs.relative_to(root_abs)
        return True
    except ValueError:
        return False


def _relative_or_absolute(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _path_size_bytes(path: Path) -> int:
    try:
        stat = path.lstat()
    except OSError:
        return 0
    if path.is_symlink() or not path.is_dir():
        return int(stat.st_size)
    total = int(stat.st_size)
    try:
        children = list(path.iterdir())
    except OSError:
        return total
    for child in children:
        total += _path_size_bytes(child)
    return total


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)
