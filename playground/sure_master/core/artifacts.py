"""Immutable, task-independent model bundles. No model libraries are imported here."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

SCHEMA = "sure.model_artifact.v2"


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_relative(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"Invalid bundle path: {value}")
    result = root / relative
    result.resolve().relative_to(root.resolve())
    return result


def publish_bundle(
    workspace: Path,
    adapter: str,
    resources: dict[str, str | Path],
    *,
    model_config: dict | None = None,
    inference_config: dict | None = None,
    provenance: dict | None = None,
    legacy_root: Path | None = None,
) -> dict[str, Any]:
    """Copy all replay dependencies before cleanup and atomically publish by content."""
    workspace = workspace.resolve()
    store = workspace.parent / "model_artifacts"
    store.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix=".pending-", dir=store))
    try:
        manifest: dict[str, Any] = {}
        if legacy_root is not None:
            shutil.copytree(legacy_root, pending, dirs_exist_ok=True, symlinks=False)
            manifest = json.loads((pending / "manifest.json").read_text())
            (pending / "manifest.json").unlink()
        resolved = {}
        for role, raw in resources.items():
            if not role.replace("_", "").isalnum():
                raise ValueError(f"Invalid resource role: {role}")
            source = Path(raw).expanduser()
            source = source if source.is_absolute() else workspace / source
            if not source.exists():
                raise FileNotFoundError(f"Missing {role}: {source}")
            if legacy_root is not None:
                relative = source.resolve().relative_to(legacy_root.resolve())
            else:
                relative = Path("assets") / role / source.name
                target = pending / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if source.is_dir():
                    shutil.copytree(
                        source,
                        target,
                        symlinks=False,
                        ignore=shutil.ignore_patterns("__pycache__", ".git", ".cache"),
                    )
                else:
                    shutil.copy2(source, target)
            resolved[role] = relative.as_posix()
        for relative in (
            "run_sure.py",
            "artifacts/candidate_changes.json",
            "artifacts/official_baseline.json",
        ):
            source = workspace / relative
            if source.is_file():
                target = pending / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        files = {
            p.relative_to(pending).as_posix(): file_digest(p)
            for p in sorted(pending.rglob("*"))
            if p.is_file()
        }
        manifest.update(
            schema_version=SCHEMA,
            adapter=adapter,
            resources=resolved,
            files=files,
            model_config=model_config or {},
            inference_config=inference_config or {},
            provenance=provenance or {},
        )
        serialized = (
            json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
        )
        identity = hashlib.sha256(serialized.encode()).hexdigest()
        (pending / "manifest.json").write_text(serialized)
        destination = store / identity
        if destination.exists():
            load_bundle(destination / "manifest.json")
            shutil.rmtree(pending)
        else:
            os.rename(pending, destination)
        result = {
            "model_artifact": str(destination / "manifest.json"),
            "model_digest": identity,
        }
        if "checkpoint" in resolved:
            checkpoint = destination / resolved["checkpoint"]
            result.update(
                candidate_checkpoint=str(checkpoint),
                checkpoint_dir=str(checkpoint.parent),
            )
        return result
    except BaseException:
        shutil.rmtree(pending, ignore_errors=True)
        raise


def load_bundle(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    schema = payload.get("schema_version")
    if schema == "sure.model_artifact.v1":
        for name in ("checkpoint", "bpe_model"):
            resource = safe_relative(path.parent, payload[name])
            expected = payload.get(
                "checkpoint_sha256" if name == "checkpoint" else "bpe_sha256"
            )
            if not resource.is_file() or (
                expected and file_digest(resource) != expected
            ):
                raise ValueError(f"Missing or corrupt legacy {name}: {resource}")
        return {
            **payload,
            "adapter": "asr.zipformer",
            "resources": {
                "checkpoint": payload["checkpoint"],
                "tokenizer": payload["bpe_model"],
            },
        }
    if schema != SCHEMA or not payload.get("files") or not payload.get("resources"):
        raise ValueError("Invalid model bundle schema or empty resources")
    for relative, expected in payload["files"].items():
        resource = safe_relative(path.parent, relative)
        if not resource.is_file() or file_digest(resource) != expected:
            raise ValueError(f"Missing or corrupt model resource: {relative}")
    for role, relative in payload["resources"].items():
        resource = safe_relative(path.parent, relative)
        if not resource.exists():
            raise FileNotFoundError(f"Missing {role}: {relative}")
        members = (
            [resource]
            if resource.is_file()
            else [p for p in resource.rglob("*") if p.is_file()]
        )
        if not members or any(
            p.relative_to(path.parent).as_posix() not in payload["files"]
            for p in members
        ):
            raise ValueError(f"Unverified resource: {role}")
    return payload


def bundle_resources(path: str | Path) -> tuple[dict[str, Any], dict[str, Path]]:
    path = Path(path).resolve()
    manifest = load_bundle(path)
    return manifest, {
        k: safe_relative(path.parent, v) for k, v in manifest["resources"].items()
    }
