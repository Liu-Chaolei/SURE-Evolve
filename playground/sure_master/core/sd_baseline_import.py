"""Content-verified shared baseline for SD ablations."""
from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

from .artifacts import bundle_resources, file_digest
from .training import require_completion
from .utils.slurm import atomic_json


def import_sd_baseline(previous_run: Path, sure: dict, workspace: Path):
    config = Path(sure["initial_baseline_config"]).resolve()
    config.relative_to(previous_run.resolve())
    original = yaml.safe_load(config.read_text())["sure"]
    for key in ("task_id", "adapter", "task", "datasets", "execution_contract", "data_preparation", "runtime"):
        if original.get(key) != sure.get(key):
            raise ValueError(f"SD baseline protocol changed: {key}")
    state_path = previous_run / "search/workspace/metric/controller_state.json"
    state = json.loads(state_path.read_text())
    if state.get("completed_rounds") != 0 or state.get("candidates"):
        raise ValueError("SD import requires a baseline-only run")
    baseline = state["baseline"]
    if type(baseline.get("score")) not in (float, int) or not math.isfinite(baseline["score"]):
        raise ValueError("Invalid SD baseline score")
    if Path(sure["initial_solution_path"]).read_text().strip() != baseline["code"].strip():
        raise ValueError("SD baseline code differs")
    manifest, resources = bundle_resources(baseline["model_artifact"])
    if manifest["adapter"] != "sd.diarizen":
        raise ValueError("Expected SD model bundle")
    completion = require_completion(resources["training_evidence"] / "training_completion.json")
    contract = completion["contract"]
    expected_data = {key: file_digest(Path(sure["datasets"][key]["manifest"])) for key in ("train", "train_validation")}
    expected_data["preparation"] = file_digest(Path(sure["data_preparation"]))
    if contract["data"] != expected_data or contract["training"] != sure["task"]["training"]:
        raise ValueError("SD baseline training data or recipe differs")
    if contract["initial_checkpoint_sha256"] != file_digest(Path(sure["task"]["resources"]["wavlm"])):
        raise ValueError("SD SSL initialization changed")
    reports = list((previous_run / "search/workspace").glob("exp_*_draft/metric/report.json"))
    if len(reports) != 1:
        raise ValueError("Expected one baseline score report")
    report = json.loads(reports[0].read_text())
    if report["score"] != baseline["score"] or report["metric"].lower() != "der":
        raise ValueError("SD baseline score differs from report")
    fingerprints = json.loads((previous_run / "data_fingerprints.json").read_text())
    actual = {name: file_digest(Path(spec["manifest"])) for name, spec in sure["datasets"].items()}
    if actual != fingerprints:
        raise ValueError("SD data split content changed")
    atomic_json(workspace / "metric/baseline_import.json", {"source_state_sha256": file_digest(state_path),
        "source_report_sha256": file_digest(reports[0]), "score": baseline["score"],
        "model_artifact": baseline["model_artifact"], "training_performed": False})
    return dict(baseline), {"model_artifact": baseline["model_artifact"]}
