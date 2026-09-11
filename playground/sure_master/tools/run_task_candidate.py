#!/usr/bin/env python3
"""Trusted model-family dispatch for candidates and frozen model replay."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from playground.sure_master.core.artifacts import bundle_resources
from playground.sure_master.tasks import get_adapter


def run(action: str, parameters: dict, artifact: str = "") -> None:
    name = os.environ["SURE_TASK_ADAPTER"]
    adapter = get_adapter(name)
    expected = os.environ.get("SURE_CANDIDATE_TYPE_HINT")
    actual = {
        "baseline": "inference",
        "infer": "inference",
        "fine_tune": "fine_tune",
        "arch": "arch",
    }[action]
    if action != "baseline" and expected and expected != actual:
        raise ValueError(f"Reviewed candidate type {expected} cannot execute {action}")
    frozen = artifact or os.environ.get("SURE_FROZEN_MODEL_ARTIFACT", "")
    if frozen and (action != "infer" or parameters):
        raise ValueError(
            "Frozen replay only accepts infer with the saved configuration"
        )
    if action == "baseline" and parameters:
        raise ValueError("Baseline parameters are fixed by the run configuration")
    parent = frozen or (
        os.environ.get("SURE_PARENT_MODEL_ARTIFACT", "") if action == "infer" else ""
    )
    manifest, resources = bundle_resources(parent) if parent else ({}, {})
    if manifest and manifest["adapter"] != name:
        raise ValueError("Model artifact belongs to another task adapter")
    os.environ["PYTHONPATH"] = (
        str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    settings = json.loads(os.environ.get("SURE_TASK_SETTINGS", "{}"))
    adapter.execute_candidate(
        action, parameters, settings, manifest, resources, parent, frozen
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action", choices=["baseline", "infer", "fine_tune", "arch"], required=True
    )
    parser.add_argument("--parameters-json", default="{}")
    parser.add_argument("--model-artifact", default="")
    args = parser.parse_args()
    parameters = json.loads(args.parameters_json)
    if not isinstance(parameters, dict):
        raise ValueError("parameters-json must be an object")
    run(args.action, parameters, args.model_artifact)


if __name__ == "__main__":
    main()
