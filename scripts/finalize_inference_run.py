#!/usr/bin/env python3
"""Commit one completed local inference run to the append-only model result index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sure_eval.models.registry import ModelRegistry
from sure_eval.results import AppendOnlyResultStore, file_sha256, validate_result_index
from sure_eval.results.schema import validate_dataset_identity, validate_prediction_manifest
from sure_eval.storage import load_storage_config


def _requested_dataset(value: str) -> str:
    dataset_id, separator, split = value.partition("@")
    if not separator or not dataset_id or not split:
        raise argparse.ArgumentTypeError("dataset must use '<name>__<version>@<split>'")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--inference-id", required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        type=_requested_dataset,
        help="Expected dataset identity '<name>__<version>@<split>'; repeat for each dataset.",
    )
    args = parser.parse_args()

    storage = load_storage_config()
    model = ModelRegistry(storage=storage).require_verified(args.model_id)
    store = AppendOnlyResultStore(model.name, model.artifact_sha256, storage=storage)
    run_dir = store.model_root / "inference_runs" / args.inference_id
    storage.assert_staging_write_path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"inference run does not exist: {run_dir}")

    protocol_path = run_dir / "protocol_resolution.json"
    protocol_resolution = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol = {
        "id": protocol_resolution["protocol_id"],
        "version": protocol_resolution["protocol_version"],
        "effective_params_sha256": protocol_resolution["effective_params_sha256"],
    }
    model_identity = {"id": model.name, "artifact_sha256": model.artifact_sha256}

    dataset_entries = []
    observed: set[str] = set()
    for manifest_path in sorted((run_dir / "manifests").glob("*.prediction_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset = validate_dataset_identity(manifest.get("dataset"), "prediction_manifest.dataset")
        validate_prediction_manifest(
            store.model_root,
            {
                "manifest_path": str(manifest_path.relative_to(store.model_root)),
                "manifest_sha256": file_sha256(manifest_path),
            },
            expected_model=model_identity,
            expected_dataset=dataset,
            expected_protocol=protocol,
        )
        key = f"{dataset['id']}@{dataset['split']}"
        if key in observed:
            raise ValueError(f"duplicate prediction manifest dataset: {key}")
        observed.add(key)
        dataset_entries.append(
            {
                "dataset": dataset,
                "prediction": {
                    "manifest_path": str(manifest_path.relative_to(store.model_root)),
                    "manifest_sha256": file_sha256(manifest_path),
                },
            }
        )

    expected = set(args.dataset)
    if observed != expected:
        raise ValueError(
            f"inference dataset set mismatch: expected {sorted(expected)}, observed {sorted(observed)}"
        )
    store.append_inference_entry(
        {
            "id": args.inference_id,
            "protocol": protocol,
            "datasets": dataset_entries,
        }
    )
    validate_result_index(store.model_root, expected_model_id=model.name, require_verified=False)
    store.write_publication_manifest()
    print(
        json.dumps(
            {
                "status": "committed",
                "model_id": model.name,
                "inference_id": args.inference_id,
                "datasets": sorted(observed),
                "result_index": str(store.index_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
