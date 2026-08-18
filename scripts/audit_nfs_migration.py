#!/usr/bin/env python3
"""Read-only audit of published registries for the v2 migration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sure_eval.models.registry import ModelRegistry
from sure_eval.results import validate_result_index
from sure_eval.storage import (
    DECLARED_MODELS_READ_ROOT,
    DECLARED_RESULTS_READ_ROOT,
    StorageConfig,
    load_storage_config,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _protocol_hints(root: Path) -> list[str]:
    hints: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file() or path.stat().st_size > 10 * 1024 * 1024:
            continue
        for part in path.parts:
            if part in {"strict_core", "standard_system", "strict_one"}:
                hints.add(part)
        if path.suffix.lower() not in {".json", ".jsonl", ".yaml", ".yml"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for protocol_id in ("strict_core", "standard_system", "strict_one"):
            if protocol_id in text:
                hints.add(protocol_id)
    return sorted(hints)


def audit(storage: StorageConfig) -> dict[str, Any]:
    models = []
    registry = ModelRegistry(storage=storage)
    if storage.models_read_root.is_dir():
        for root in sorted(path for path in storage.models_read_root.iterdir() if path.is_dir()):
            item = {"model_id": root.name, "status": "legacy_unverified", "issues": []}
            if not (root / "config.yaml").is_file():
                item["issues"].append("config_missing")
            try:
                registry.require_verified(root.name)
            except Exception as exc:
                item["issues"].append(str(exc))
            else:
                item["status"] = "v2_verified"
            models.append(item)

    results = []
    if storage.results_read_root.is_dir():
        for root in sorted(path for path in storage.results_read_root.iterdir() if path.is_dir()):
            index_path = root / "result_index.json"
            item: dict[str, Any] = {
                "model_id": root.name,
                "status": "legacy",
                "protocol_hints": _protocol_hints(root),
                "issues": [],
            }
            if index_path.is_file():
                try:
                    index = validate_result_index(root, expected_model_id=root.name)
                except Exception as exc:
                    item["status"] = "v2_invalid"
                    item["issues"].append(str(exc))
                else:
                    item["status"] = "v2_verified"
                    item["inference_run_count"] = len(index["inference_runs"])
                    item["evaluation_run_count"] = len(index["evaluation_runs"])
            else:
                item["issues"].append("result_index_v2_missing")
                if set(item["protocol_hints"]) & {"strict_core", "strict_one"}:
                    item["issues"].append(
                        "legacy_strict_label_requires_human_semantic_review; do_not_auto_relabel"
                    )
            results.append(item)
    return {
        "schema": "sure.eval.nfs_migration_audit.v1",
        "status": "completed_read_only",
        "created_at": _utc_now(),
        "roots": {
            "models": str(DECLARED_MODELS_READ_ROOT),
            "results": str(DECLARED_RESULTS_READ_ROOT),
        },
        "nfs_modified": False,
        "models": models,
        "results": results,
        "summary": {
            "model_count": len(models),
            "verified_model_count": sum(item["status"] == "v2_verified" for item in models),
            "result_model_count": len(results),
            "verified_result_count": sum(item["status"] == "v2_verified" for item in results),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-local",
        action="store_true",
        help="Also write the fixed repository-local tmp/nfs_migration_audit.json file.",
    )
    args = parser.parse_args()
    storage = load_storage_config()
    report = audit(storage)
    if args.write_local:
        output = storage.repo_root / "tmp/nfs_migration_audit.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
