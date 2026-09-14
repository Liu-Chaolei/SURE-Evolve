"""Import a verified search baseline into a new provider run without training."""

from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

from .artifacts import bundle_resources, file_digest
from .utils.slurm import atomic_json

FIXED_ENV_KEYS = (
    "SURE_ASR_DATASET",
    "SURE_ASR_RECIPE_PROFILE",
    "SURE_ASR_EVAL_SPLITS",
    "SURE_ASR_FIXED_SEED",
    "SURE_MAX_TRAIN_EPOCHS",
    "SURE_MAX_DURATION",
    "SURE_USE_FP16",
    "SURE_BASELINE_WORLD_SIZE",
    "ASR_WORLD_SIZE",
    "SURE_ENABLE_MUSAN",
    "SURE_DECODE_METHOD",
    "SURE_BASELINE_AVG",
    "SURE_BASELINE_USE_AVERAGED_MODEL",
    "SURE_ASR_FIXED_BUDGET",
)


def import_baseline(
    previous_run: Path, sure: dict, workspace: Path
) -> tuple[dict, dict]:
    previous_run = previous_run.resolve()
    if sure.get("adapter") == "sd.diarizen":
        from .sd_baseline_import import import_sd_baseline
        return import_sd_baseline(previous_run, sure, workspace)
    if sure.get('adapter') == 'tts.f5tts':
        from .tts_baseline_import import import_tts_baseline
        return import_tts_baseline(previous_run, sure, workspace)
    state_path = previous_run / "search/workspace/metric/controller_state.json"
    state = json.loads(state_path.read_text())
    if state.get("completed_rounds") != 0 or state.get("candidates"):
        raise ValueError(
            "Baseline-only migration requires a run without completed candidates"
        )
    original = yaml.safe_load((previous_run / "execution.yaml").read_text())["sure"]
    if original["task_id"] != sure["task_id"]:
        raise ValueError("Cannot import a baseline from another task")
    for key in FIXED_ENV_KEYS:
        if str(original["execution_env"].get(key)) != str(
            sure["execution_env"].get(key)
        ):
            raise ValueError(f"Baseline execution setting changed: {key}")
    if original.get("execution_contract") != sure.get("execution_contract"):
        raise ValueError("Baseline scientific execution contract changed")
    old_sources = original["base_models"][sure["task_id"]]["source_paths"]
    new_sources = sure["base_models"][sure["task_id"]]["source_paths"]
    if old_sources != new_sources:
        raise ValueError("Baseline model/data source paths changed")
    baseline = state["baseline"]
    score = baseline.get("score")
    if not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("Imported baseline score is not finite")
    if (
        Path(sure["initial_solution_path"]).read_text().strip()
        != baseline["code"].strip()
    ):
        raise ValueError("Fixed baseline implementation changed")
    manifest_path = Path(baseline["model_artifact"])
    manifest, resources = bundle_resources(manifest_path)
    if manifest["adapter"] != "asr.zipformer":
        raise ValueError("Expected an ASR Zipformer baseline artifact")
    if file_digest(resources["tokenizer"]) != file_digest(
        Path(new_sources["data"]) / "lang_bpe_500/bpe.model"
    ):
        raise ValueError("Baseline tokenizer differs from the current data")
    copied_refs = list(
        (previous_run / "search/workspace").glob("exp_*_draft/input/ref.txt")
    )
    if len(copied_refs) != 1 or file_digest(copied_refs[0]) != file_digest(
        Path(sure["inputs"]["ref"])
    ):
        raise ValueError("Baseline reference text differs from the current search set")
    changes = json.loads(
        (manifest_path.parent / "artifacts/candidate_changes.json").read_text()
    )
    if changes["training_config"]["actual_train_epoch"] != int(
        sure["execution_env"]["SURE_MAX_TRAIN_EPOCHS"]
    ):
        raise ValueError("Baseline checkpoint epoch does not match the search budget")
    if state["attributes"]["best_model_artifact"]["model_artifact"] != str(
        manifest_path
    ):
        raise ValueError("Saved initial model is not the baseline")
    atomic_json(
        workspace / "metric/baseline_import.json",
        {
            "previous_run": str(previous_run),
            "source_state_sha256": file_digest(state_path),
            "model_artifact": str(manifest_path),
            "score": score,
            "training_performed": False,
            "reference_sha256": file_digest(copied_refs[0]),
        },
    )
    return dict(baseline), dict(state["attributes"]["best_model_artifact"])
