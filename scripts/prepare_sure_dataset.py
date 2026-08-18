#!/usr/bin/env python3
"""
Prepare canonical SURE-EVAL datasets deterministically.

This script is intentionally boring: it only downloads / converts datasets and
emits a machine-readable summary. It is meant to reduce agent-side uncertainty.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sure_eval.core.config import Config
from sure_eval.core.logging import configure_logging, get_logger
from sure_eval.datasets import DatasetManager

configure_logging(level="INFO")
logger = get_logger(__name__)


def prepare_dataset(
    manager: DatasetManager,
    dataset_name: str,
    requested_name: str | None = None,
    *,
    strict_aispeech_source: bool = True,
    annotation_source_root: str | None = None,
) -> dict[str, Any]:
    """Prepare a single dataset and return a summary."""
    source_ref = None
    if manager.is_aispeech_source_entry(dataset_name):
        source_ref = manager.resolve_aispeech_source_entry(
            dataset_name,
            annotation_source_root=annotation_source_root,
        )
        jsonl_path = manager.convert_aispeech_source_to_jsonl(source_ref)
        prepared_name = source_ref.dataset_id
    elif strict_aispeech_source:
        raise ValueError(
            "Main-flow dataset input must be an AiSpeech source root under "
            "/hpc_stor08/external_ds/aispeech/.../ds_pool/<dataset_name>; "
            f"got: {dataset_name}"
        )
    else:
        canonical_name = manager.normalize_dataset_name(dataset_name)
        jsonl_path = manager.download_and_convert(dataset_name)
        prepared_name = jsonl_path.stem
        info = manager.get_info(prepared_name) or manager.get_info(canonical_name) or {}
        return {
            "dataset": prepared_name,
            "requested_name": requested_name or dataset_name,
            "jsonl_path": str(jsonl_path),
            "task": info.get("task"),
            "language": info.get("language"),
            "source": info.get("source"),
            "num_samples": info.get("num_samples"),
            "display_name": info.get("display_name"),
            "naming_policy": "legacy_dataset_name",
        }

    info = manager.get_info(prepared_name) or {}

    return {
        "dataset": prepared_name,
        "requested_name": requested_name or dataset_name,
        "dataset_id": prepared_name,
        "source_dataset_root": str(source_ref.source_dataset_root) if source_ref else None,
        "annotation_source_root": str(source_ref.annotation_source_root) if source_ref else None,
        "source_dataset_name": source_ref.source_dataset_name if source_ref else None,
        "version_id": source_ref.version_id if source_ref else None,
        "jsonl_path": str(jsonl_path),
        "task": info.get("task"),
        "language": info.get("language"),
        "source": "aispeech_ds_pool",
        "num_samples": info.get("num_samples"),
        "display_name": info.get("display_name"),
        "naming_policy": "source_dataset_name__version_id",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare deterministic SURE-EVAL datasets")
    parser.add_argument(
        "--dataset",
        nargs="+",
        help="AiSpeech source roots: /hpc_stor08/external_ds/aispeech/.../ds_pool/<dataset_name>",
    )
    parser.add_argument("--all", action="store_true", help="Prepare all configured datasets")
    parser.add_argument("--config", type=str, help="Config path")
    parser.add_argument("--output", type=str, help="Optional JSON summary output path")
    parser.add_argument(
        "--annotation-source-root",
        type=str,
        help=(
            "Explicit read-only AiSpeech dataset root providing sample_files metadata "
            "for one audio-only --dataset source root"
        ),
    )
    parser.add_argument(
        "--allow-legacy-dataset-name",
        action="store_true",
        help="Compatibility escape hatch for non-main-flow callers. Main-flow runs must not set this.",
    )
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("Specify --dataset ... or --all")
    if args.annotation_source_root and (not args.dataset or len(args.dataset) != 1):
        parser.error("--annotation-source-root requires exactly one --dataset source root")

    cfg = Config.from_yaml(args.config) if args.config else Config.from_env()
    manager = DatasetManager(cfg)

    if args.all and not args.allow_legacy_dataset_name:
        parser.error("--all is only available with --allow-legacy-dataset-name")

    requested_names = args.dataset or list(cfg.datasets.definitions.keys())
    dataset_names: list[str] = []
    requested_by_dataset: dict[str, str] = {}
    seen: set[str] = set()
    for requested_name in requested_names:
        if manager.is_aispeech_source_entry(requested_name):
            expanded_names = [requested_name]
        elif args.allow_legacy_dataset_name:
            expanded_names = manager.expand_dataset_names([requested_name])
        else:
            raise ValueError(
                "Main-flow dataset input must be an AiSpeech source root under "
                "/hpc_stor08/external_ds/aispeech/.../ds_pool/<dataset_name>; "
                f"got: {requested_name}"
            )
        for dataset_name in expanded_names:
            if dataset_name not in seen:
                dataset_names.append(dataset_name)
                seen.add(dataset_name)
            requested_by_dataset.setdefault(dataset_name, requested_name)
    prepared: list[dict[str, Any]] = []

    for dataset_name in dataset_names:
        logger.info("Preparing dataset", dataset=dataset_name)
        prepared.append(
            prepare_dataset(
                manager,
                dataset_name,
                requested_name=requested_by_dataset.get(dataset_name),
                strict_aispeech_source=not args.allow_legacy_dataset_name,
                annotation_source_root=args.annotation_source_root,
            )
        )

    summary = {"prepared": prepared}
    output = json.dumps(summary, indent=2, ensure_ascii=False)
    print(output)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
        logger.info("Wrote preparation summary", path=str(output_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
