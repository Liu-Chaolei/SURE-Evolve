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
from playground.sure_master.core.search_scope import validate_candidate_parameters


def run(action: str, parameters: dict, artifact: str = "", candidate_source: str = "") -> None:
    name = os.environ["SURE_TASK_ADAPTER"]
    adapter = get_adapter(name)
    expected = os.environ.get("SURE_CANDIDATE_TYPE_HINT")
    if candidate_source and (artifact or os.environ.get("SURE_FROZEN_MODEL_ARTIFACT")):
        raise ValueError("Frozen replay cannot override source")
    if action == "prepare_source":
        if name not in {"tts.f5tts", "sd.diarizen"} or not candidate_source or artifact or os.environ.get("SURE_FROZEN_MODEL_ARTIFACT"):
            raise ValueError("prepare_source requires an editable speech candidate directory")
        if name == "sd.diarizen":
            from playground.sure_master.runtime.sd_evolution import prepare_source
        else:
            from playground.sure_master.runtime.f5_evolution import prepare_source
        settings = json.loads(os.environ["SURE_TASK_SETTINGS"])
        source = Path(settings["resources"]["source"])
        parent = os.environ.get("SURE_PARENT_MODEL_ARTIFACT")
        if parent:
            _, resources = bundle_resources(parent)
            source = resources["source"]
        prepare_source(source, Path(candidate_source))
        return
    if action == "candidate":
        if name not in {"tts.f5tts", "sd.diarizen"} or os.environ.get("SURE_SEARCH_SCOPE") != "all":
            raise ValueError("Unified candidates require speech free exploration")
        parameters = dict(parameters)
        training = parameters.pop("requires_training", None)
        if type(training) is not bool:
            raise ValueError("Candidate must declare requires_training as a boolean")
        if expected and training != (expected != "inference"):
            raise ValueError("Candidate training requirement differs from reviewed execution")
        action = ("arch" if expected == "arch" else "fine_tune") if training else "infer"
    actual = {
        "baseline": "fine_tune" if name in {"tts.f5tts", "sd.diarizen"} else "inference",
        "infer": "inference",
        "fine_tune": "fine_tune",
        "arch": "arch",
    }[action]
    if action != "baseline" and expected and expected != actual:
        raise ValueError(f"Reviewed candidate type {expected} cannot execute {action}")
    frozen = artifact or os.environ.get("SURE_FROZEN_MODEL_ARTIFACT", "")
    if frozen and (action != "infer" or parameters or candidate_source):
        raise ValueError(
            "Frozen replay only accepts infer with the saved configuration"
        )
    if action == "baseline" and parameters:
        raise ValueError("Baseline parameters are fixed by the run configuration")
    validate_candidate_parameters(action, parameters, os.environ, frozen=bool(frozen))
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
    source_key = "SURE_SD_CANDIDATE_SOURCE" if name == "sd.diarizen" else "SURE_F5_CANDIDATE_SOURCE"
    previous = os.environ.pop(source_key, None)
    if candidate_source:
        os.environ[source_key] = candidate_source
    try:
        adapter.execute_candidate(action, parameters, settings, manifest, resources, parent, frozen)
    finally:
        os.environ.pop(source_key, None)
        if previous is not None:
            os.environ[source_key] = previous


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action", choices=["baseline", "infer", "fine_tune", "arch", "candidate", "prepare_source"], required=True
    )
    parser.add_argument("--parameters-json", default="{}")
    parser.add_argument("--model-artifact", default="")
    parser.add_argument("--candidate-source", default="")
    args = parser.parse_args()
    parameters = json.loads(args.parameters_json)
    if not isinstance(parameters, dict):
        raise ValueError("parameters-json must be an object")
    run(args.action, parameters, args.model_artifact, args.candidate_source)


if __name__ == "__main__":
    main()
