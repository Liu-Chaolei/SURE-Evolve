"""Enforce the published-registry and local-staging storage boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import yaml


DECLARED_MODELS_READ_ROOT = Path("/hpc_stor03/project/oref/nfs/models")
DECLARED_RESULTS_READ_ROOT = Path("/hpc_stor03/project/oref/nfs/results")
DEFAULT_MODELS_READ_ROOT = DECLARED_MODELS_READ_ROOT.resolve(strict=False)
DEFAULT_RESULTS_READ_ROOT = DECLARED_RESULTS_READ_ROOT.resolve(strict=False)
DEFAULT_MODELS_WRITE_ROOT = Path("src/sure_eval/models")
DEFAULT_RESULTS_WRITE_ROOT = Path("results")
STORAGE_SCHEMA = "sure.eval.storage.v1"

_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:__[A-Za-z0-9][A-Za-z0-9._-]*)*$")


class PathPolicyError(ValueError):
    """Raised when a path crosses the registry/staging trust boundary."""


def validate_model_id(model_id: str) -> str:
    """Return a canonical model id or reject path-like and ambiguous values."""

    value = str(model_id).strip()
    if not value or value in {".", ".."} or not _MODEL_ID_PATTERN.fullmatch(value):
        raise PathPolicyError(
            "model_id must be one path segment containing only letters, digits, '.', '_', or '-'"
        )
    return value


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _mapping(value: Any, key: str) -> dict[str, Any]:
    result = value.get(key) if isinstance(value, dict) else None
    if not isinstance(result, dict):
        raise PathPolicyError(f"storage config requires mapping: {key}")
    return result


@dataclass(frozen=True)
class StorageConfig:
    """Resolved storage roots with explicit read and write capabilities."""

    repo_root: Path
    models_read_root: Path
    results_read_root: Path
    models_write_root: Path
    results_write_root: Path

    def __post_init__(self) -> None:
        repo_root = _resolved(self.repo_root)
        models_read_root = _resolved(self.models_read_root)
        results_read_root = _resolved(self.results_read_root)
        models_write_root = _resolved(self.models_write_root)
        results_write_root = _resolved(self.results_write_root)

        object.__setattr__(self, "repo_root", repo_root)
        object.__setattr__(self, "models_read_root", models_read_root)
        object.__setattr__(self, "results_read_root", results_read_root)
        object.__setattr__(self, "models_write_root", models_write_root)
        object.__setattr__(self, "results_write_root", results_write_root)

        for name, root in (
            ("models_write_root", models_write_root),
            ("results_write_root", results_write_root),
        ):
            if not _is_within(root, repo_root):
                raise PathPolicyError(f"{name} must resolve inside repository root: {root}")
            if _is_within(root, models_read_root) or _is_within(root, results_read_root):
                raise PathPolicyError(f"{name} overlaps a read-only NFS registry: {root}")

    def published_model(self, model_id: str, *, require_exists: bool = True) -> Path:
        return self._published_child(self.models_read_root, model_id, require_exists=require_exists)

    def declared_published_model(self, model_id: str) -> Path:
        return DECLARED_MODELS_READ_ROOT / validate_model_id(model_id)

    def published_results(self, model_id: str, *, require_exists: bool = True) -> Path:
        return self._published_child(self.results_read_root, model_id, require_exists=require_exists)

    def declared_published_results(self, model_id: str) -> Path:
        return DECLARED_RESULTS_READ_ROOT / validate_model_id(model_id)

    def staged_model(self, model_id: str, *, create: bool = False) -> Path:
        return self._staged_child(self.models_write_root, model_id, create=create)

    def staged_results(self, model_id: str, *, create: bool = False) -> Path:
        return self._staged_child(self.results_write_root, model_id, create=create)

    def assert_registry_read_path(self, path: str | Path) -> Path:
        candidate = _resolved(Path(path))
        if not (
            _is_within(candidate, self.models_read_root)
            or _is_within(candidate, self.results_read_root)
        ):
            raise PathPolicyError(f"published artifact reads must stay under the NFS registries: {candidate}")
        return candidate

    def assert_staging_write_path(self, path: str | Path) -> Path:
        candidate = _resolved(Path(path))
        if _is_within(candidate, self.models_read_root) or _is_within(
            candidate, self.results_read_root
        ):
            raise PathPolicyError(f"agents must never write to the NFS registries: {candidate}")
        if not (
            _is_within(candidate, self.models_write_root)
            or _is_within(candidate, self.results_write_root)
        ):
            raise PathPolicyError(f"artifact writes must stay under repository staging roots: {candidate}")
        return candidate

    def _published_child(self, root: Path, model_id: str, *, require_exists: bool) -> Path:
        child = _resolved(root / validate_model_id(model_id))
        if not _is_within(child, root):
            raise PathPolicyError(f"published path escapes registry root: {child}")
        if require_exists and not child.is_dir():
            raise FileNotFoundError(f"published artifact is not available: {child}")
        return child

    def _staged_child(self, root: Path, model_id: str, *, create: bool) -> Path:
        child = _resolved(root / validate_model_id(model_id))
        self.assert_staging_write_path(child)
        if create:
            child.mkdir(parents=True, exist_ok=True)
            resolved_after_create = _resolved(child)
            if not _is_within(resolved_after_create, root):
                raise PathPolicyError(f"staging path escapes through a symbolic link: {child}")
        return child


def load_storage_config(path: str | Path | None = None) -> StorageConfig:
    """Load the single authoritative storage configuration."""

    repo_root = Path(__file__).resolve().parents[3]
    config_path = Path(path) if path is not None else repo_root / "config" / "storage.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if payload.get("schema") != STORAGE_SCHEMA:
        raise PathPolicyError(f"unsupported storage config schema: {payload.get('schema')!r}")

    registry = _mapping(payload, "registry")
    staging = _mapping(payload, "staging")
    models_read = Path(str(registry.get("models_read_root", "")))
    results_read = Path(str(registry.get("results_read_root", "")))
    if models_read != DECLARED_MODELS_READ_ROOT or results_read != DECLARED_RESULTS_READ_ROOT:
        raise PathPolicyError("published registry roots are fixed and may not be overridden")

    models_write = Path(str(staging.get("models_write_root", "")))
    results_write = Path(str(staging.get("results_write_root", "")))
    if models_write.is_absolute() or results_write.is_absolute():
        raise PathPolicyError("staging roots must be repository-relative")
    if models_write != DEFAULT_MODELS_WRITE_ROOT or results_write != DEFAULT_RESULTS_WRITE_ROOT:
        raise PathPolicyError("staging roots are fixed and may not be overridden")

    return StorageConfig(
        repo_root=repo_root,
        models_read_root=models_read,
        results_read_root=results_read,
        models_write_root=repo_root / models_write,
        results_write_root=repo_root / results_write,
    )
