#!/usr/bin/env python3
"""Create a local model publication request; never copy or write into NFS."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sure_eval.models.registry import (
    MODEL_ARTIFACT_MANIFEST_SCHEMA,
    _file_sha256,
    model_artifact_digest,
)
from sure_eval.storage import load_storage_config, validate_model_id


EXCLUDED_PARTS = {".runtime", "__pycache__", "eval_runs", ".git"}
EXCLUDED_FILES = {
    "publication.json",
    "publication_artifacts.json",
    "publication_request.json",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write(path: Path, payload: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()
    model_id = validate_model_id(args.model_id)
    storage = load_storage_config()
    model_root = storage.staged_model(model_id)
    storage.assert_staging_write_path(model_root)
    if not model_root.is_dir():
        raise FileNotFoundError(f"staged model adapter does not exist: {model_root}")
    for required in ("config.yaml", "server.py"):
        if not (model_root / required).is_file():
            raise FileNotFoundError(f"model adapter is incomplete: missing {required}")

    files = []
    for path in sorted(model_root.rglob("*")):
        relative = path.relative_to(model_root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if relative.name in EXCLUDED_FILES:
            continue
        if path.is_symlink():
            raise ValueError(f"publication artifacts must not contain symlinks: {relative}")
        if path.is_file():
            files.append(
                {
                    "path": relative.as_posix(),
                    "sha256": _file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    artifact_manifest = {
        "schema": MODEL_ARTIFACT_MANIFEST_SCHEMA,
        "model_id": model_id,
        "files": files,
    }
    manifest_path = model_root / "publication_artifacts.json"
    _atomic_write(manifest_path, artifact_manifest)
    request = {
        "schema": "sure.eval.model_publication_request.v1",
        "status": "pending_human_review",
        "model_id": model_id,
        "source_model_root": str(model_root.relative_to(storage.repo_root)),
        "target_registry_root": str(storage.declared_published_model(model_id)),
        "artifact_sha256": model_artifact_digest(files),
        "artifact_manifest_path": manifest_path.name,
        "artifact_manifest_sha256": _file_sha256(manifest_path),
        "created_at": _utc_now(),
        "automatic_publish_allowed": False,
    }
    request_path = model_root / "publication_request.json"
    _atomic_write(request_path, request)
    print(json.dumps(request, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
