from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
from pathlib import Path

from .common import JsonObject, as_mapping, atomic_write_json, read_json, sha256_file, text


def parser_context(command: Path, device: str, backend: str, method: str, language: str) -> JsonObject:
    try:
        version = importlib.metadata.version("mineru")
    except importlib.metadata.PackageNotFoundError:
        version = "uninstalled"
    config = Path(os.environ.get("MINERU_TOOLS_CONFIG_JSON", str(Path.home() / "mineru.json")))
    if not config.is_absolute():
        config = Path.home() / config
    model_files: list[tuple[str, int, int]] = []
    if config.is_file():
        settings = as_mapping(read_json(config))
        for directory in as_mapping(settings.get("models-dir")).values():
            if not text(directory):
                continue
            root = Path(text(directory))
            if root.is_dir():
                for path in sorted(root.rglob("*")):
                    if path.is_file():
                        stat = path.stat()
                        model_files.append((str(path), stat.st_size, stat.st_mtime_ns))
    return {
        "parser_version": "mineru-batched-v1",
        "mineru_version": version,
        "command_sha256": sha256_file(command),
        "npu_launcher_sha256": sha256_file(Path(__file__).parents[1] / "mineru_npu.py") if device == "npu" else None,
        "config_sha256": sha256_file(config) if config.is_file() else None,
        "model_files_sha256": hashlib.sha256(json.dumps(model_files).encode()).hexdigest(),
        "environment": {key: os.environ.get(key) for key in (
            "MINERU_MODEL_SOURCE", "MINERU_FORMULA_ENABLE", "MINERU_TABLE_ENABLE",
            "MINERU_HYBRID_BATCH_RATIO", "MINERU_PROCESSING_WINDOW_SIZE",
            "MINERU_OCR_DET_MASK_INLINE_FORMULA_ENABLE",
        )},
        "device": device,
        "backend": backend,
        "method": method,
        "language": language,
    }


def cache_key(source_sha256: str, context: JsonObject) -> str:
    return hashlib.sha256(json.dumps([source_sha256, context], sort_keys=True).encode()).hexdigest()


def cached_bundle(root: Path, key: str) -> dict[str, Path | None] | None:
    directory = root / key[:2] / key
    try:
        record = as_mapping(read_json(directory / "bundle.json"))
        if record.get("key") != key:
            return None
        for name, expected in as_mapping(record.get("files")).items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory.resolve()) or not path.is_file() or sha256_file(path) != expected:
                return None
        result: dict[str, Path | None] = {}
        for kind in ("markdown", "content_list", "middle", "model", "images"):
            name = text(as_mapping(record.get("paths")).get(kind))
            path = (directory / name).resolve() if name else None
            if path is not None and not path.is_relative_to(directory.resolve()):
                return None
            result[kind] = path
        if any(result[kind] is None or not result[kind].is_file() for kind in ("markdown", "content_list", "middle")):
            return None
        return result
    except (OSError, ValueError, TypeError):
        return None


def publish_bundle(root: Path, key: str, bundle: dict[str, Path | None]) -> dict[str, Path | None]:
    parent = root / key[:2]
    parent.mkdir(parents=True, exist_ok=True)
    existing = cached_bundle(root, key)
    if existing is not None:
        return existing
    temporary = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=parent))
    destination = parent / key
    try:
        paths: JsonObject = {}
        for kind, source in bundle.items():
            if source is None:
                paths[kind] = None
                continue
            paths[kind] = source.name
            if source.is_dir():
                shutil.copytree(source, temporary / source.name)
            else:
                shutil.copy2(source, temporary / source.name)
        files = {str(path.relative_to(temporary)): sha256_file(path) for path in temporary.rglob("*") if path.is_file()}
        atomic_write_json(temporary / "bundle.json", {"key": key, "paths": paths, "files": files})
        # The caller holds the per-key lock. Never expose a partially copied bundle.
        if destination.exists():
            shutil.rmtree(destination)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    result = cached_bundle(root, key)
    if result is None:
        raise ValueError(f"Published MinerU cache bundle failed validation: {key}")
    return result
