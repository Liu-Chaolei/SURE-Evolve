#!/usr/bin/env python3
"""Run the pinned evaluator and stage an append-only, human-reviewable result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sure_eval.evaluation_engine import EvaluationRunOrchestrator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--inference-id", required=True)
    parser.add_argument(
        "--inference-source",
        required=True,
        choices=("published", "staged"),
        help="Read an existing verified NFS inference or the current local staged inference.",
    )
    parser.add_argument(
        "--input-manifest",
        action="append",
        required=True,
        help="Evaluation input manifest; repeat for each dataset/pipeline.",
    )
    args = parser.parse_args()
    result = EvaluationRunOrchestrator().run(
        model_id=args.model_id,
        evaluation_id=args.evaluation_id,
        inference_id=args.inference_id,
        inference_source=args.inference_source,
        input_manifests=args.input_manifest,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
