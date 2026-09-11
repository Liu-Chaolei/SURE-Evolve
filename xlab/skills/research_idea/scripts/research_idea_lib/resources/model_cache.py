"""Content-verified cache for immutable local model snapshots."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import BinaryIO, Mapping

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from .manifest import sha256_file

MODEL_CACHE_SCHEMA_VERSION = "xlab.research_idea.model_cache.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ModelCacheError(ValueError):
    """Raised when a model snapshot cannot be verified or cached."""


class _FileLock:
    """Cross-platform advisory file lock released automatically on process exit."""

    def __init__(self, path: Path, timeout: float) -> None:
        self.path = path
        self.timeout = timeout
        self._file: BinaryIO | None = None

    def __enter__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = self.path.open("a+b")
        if os.name == "nt" and lock_file.seek(0, os.SEEK_END) == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                _try_lock(lock_file)
                self._file = lock_file
                return
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    lock_file.close()
                    raise
                if time.monotonic() >= deadline:
                    lock_file.close()
                    raise ModelCacheError(f"Timed out waiting for model cache lock: {self.path}") from exc
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._file is None:
            return
        try:
            _unlock(self._file)
        finally:
            self._file.close()
            self._file = None


def _try_lock(lock_file: BinaryIO) -> None:
    lock_file.seek(0)
    if os.name == "nt":
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(lock_file: BinaryIO) -> None:
    lock_file.seek(0)
    if os.name == "nt":
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class ModelFile:
    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ModelSnapshot:
    model_id: str
    revision: str
    files: tuple[ModelFile, ...]
    capabilities: tuple[str, ...] = ()
    dimensions: tuple[tuple[str, int], ...] = ()

    @property
    def dimension_map(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self.dimensions))

    @property
    def cache_key(self) -> str:
        payload = {
            "model_id": self.model_id,
            "revision": self.revision,
            "files": [
                {"path": file.path, "sha256": file.sha256, "size": file.size}
                for file in sorted(self.files, key=lambda item: item.path)
            ],
        }
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CachedModel:
    snapshot: ModelSnapshot
    path: Path
    uri: str
    reused: bool


class ModelCache:
    """Resolve only explicitly supplied immutable model snapshots."""

    def __init__(self, root: Path, *, lock_timeout: float = 30.0) -> None:
        if lock_timeout < 0:
            raise ValueError("Model cache lock timeout must be non-negative")
        self.root = root.resolve()
        self.lock_timeout = lock_timeout

    def resolve(self, snapshot: ModelSnapshot, *, source: Path | None = None) -> CachedModel:
        _validate_snapshot(snapshot)
        target = self.root / "models" / snapshot.cache_key
        uri = f"xlab-cache://models/{snapshot.cache_key}"
        lock_path = self.root / "locks" / f"model-{snapshot.cache_key}.lock"

        with _FileLock(lock_path, self.lock_timeout):
            if target.exists():
                _verify_snapshot(target, snapshot)
                return CachedModel(snapshot=snapshot, path=target, uri=uri, reused=True)
            if source is None:
                raise ModelCacheError(
                    f"Model {snapshot.model_id!r} revision {snapshot.revision!r} is not cached and no explicit source was provided"
                )
            source = source.resolve()
            _verify_snapshot(source, snapshot)
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=f".{snapshot.cache_key}.", dir=target.parent))
            try:
                for expected in snapshot.files:
                    source_file = _contained_file(source, expected.path)
                    destination = staging / Path(*PurePosixPath(expected.path).parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source_file, destination)
                _write_metadata(staging, snapshot)
                _verify_snapshot(staging, snapshot)
                os.replace(staging, target)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            return CachedModel(snapshot=snapshot, path=target, uri=uri, reused=False)


def _validate_snapshot(snapshot: ModelSnapshot) -> None:
    if not snapshot.model_id.strip():
        raise ModelCacheError("Model ID must be non-empty")
    if not snapshot.revision.strip():
        raise ModelCacheError("Model revision must be non-empty and immutable")
    if not snapshot.files:
        raise ModelCacheError("Model snapshot must declare at least one file")
    seen: set[str] = set()
    for file in snapshot.files:
        if file.path in seen:
            raise ModelCacheError(f"Model snapshot repeats file {file.path!r}")
        seen.add(file.path)
        _validate_relative_path(file.path)
        if not _SHA256_RE.fullmatch(file.sha256):
            raise ModelCacheError(f"Model file {file.path!r} must declare a lowercase SHA-256 digest")
        if file.size < 0:
            raise ModelCacheError(f"Model file {file.path!r} must declare a non-negative size")


def _verify_snapshot(root: Path, snapshot: ModelSnapshot) -> None:
    if not root.is_dir():
        raise ModelCacheError(f"Model snapshot root is not a directory: {root}")
    for expected in snapshot.files:
        path = _contained_file(root, expected.path)
        if not path.is_file():
            raise ModelCacheError(f"Model snapshot is missing file {expected.path!r}")
        actual_size = path.stat().st_size
        if actual_size != expected.size:
            raise ModelCacheError(
                f"Model file {expected.path!r} has size {actual_size}, expected {expected.size}"
            )
        actual_digest = sha256_file(path)
        if actual_digest != expected.sha256:
            raise ModelCacheError(f"Model file {expected.path!r} has a digest mismatch")


def _contained_file(root: Path, relative: str) -> Path:
    _validate_relative_path(relative)
    candidate = (root / Path(*PurePosixPath(relative).parts)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ModelCacheError(f"Model file escapes its snapshot: {relative!r}") from exc
    return candidate


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ModelCacheError(f"Model file path must be normalized and relative: {value!r}")


def _write_metadata(root: Path, snapshot: ModelSnapshot) -> None:
    payload = {
        "schema_version": MODEL_CACHE_SCHEMA_VERSION,
        "model_id": snapshot.model_id,
        "revision": snapshot.revision,
        "cache_key": snapshot.cache_key,
        "capabilities": list(snapshot.capabilities),
        "dimensions": dict(snapshot.dimensions),
        "files": [
            {"path": file.path, "sha256": file.sha256, "size": file.size}
            for file in sorted(snapshot.files, key=lambda item: item.path)
        ],
    }
    (root / "xlab-model-cache.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
