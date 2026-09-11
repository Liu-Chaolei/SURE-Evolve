"""Full-budget search configuration and historical model-lineage inspection."""

from __future__ import annotations
import json
from pathlib import Path

from .artifacts import load_bundle


def training_plan(artifact: str) -> dict:
    current = Path(artifact).resolve()
    seen = set()
    inference = None
    while current not in seen:
        seen.add(current)
        load_bundle(current)
        changes = current.parent / "artifacts/candidate_changes.json"
        if not changes.exists():
            raise ValueError(
                "Full training requires wrapper-recorded training provenance"
            )
        data = json.loads(changes.read_text())
        if inference is None:
            inference = data["inference_config"]
        if data.get("training_config", {}).get("actual_train_epoch"):
            return {
                "recipe": str(current.parent / "recipe"),
                "train_args": data["diff_from_defaults"]["train_extra_args"],
                "inference": inference,
                "training_artifact": str(current),
                "decode_args": json.loads(
                    (
                        Path(artifact).parent / "artifacts/candidate_changes.json"
                    ).read_text()
                )["diff_from_defaults"]["decode_extra_args"],
            }
        parent = data.get("parent_model_artifact")
        if not parent:
            raise ValueError("Inference candidate has no traceable training parent")
        current = Path(parent).resolve()
    raise ValueError("Cycle in ASR training lineage")


def promote_full_training_to_search(sure: dict) -> dict:
    """Migrate the former post-search budget before any baseline/candidate is run."""
    from copy import deepcopy

    resolved = deepcopy(sure)
    legacy = resolved.get("full_training") or {}
    if not legacy.get("enabled"):
        return resolved
    task_id = resolved.get("task_id", "asr_en_wer")
    if not task_id.startswith("asr_"):
        raise ValueError(
            "Move the final training data/budget into the task's search configuration; post-search retraining is retired"
        )
    data = legacy.get("data")
    epochs = legacy.get("epochs", 30)
    if not data or type(epochs) is not int or epochs < 1:
        raise ValueError(
            "Full search training requires a data path and positive epoch budget"
        )
    profile = resolved.setdefault("base_models", {}).setdefault(task_id, {})
    profile.setdefault("source_paths", {})["data"] = str(data)
    env = resolved.setdefault("execution_env", {})
    env.update(
        SURE_MAX_TRAIN_EPOCHS=str(epochs),
        SURE_BASELINE_EPOCH=str(epochs),
        SURE_BASELINE_USE_PRETRAINED="0",
        SURE_DECODE_ONLY="0",
    )
    contract = resolved.setdefault("execution_contract", {})
    contract.pop("training_hours", None)
    contract.update(
        training_epochs=epochs,
        training_split="full_train",
        training_data=str(data),
        training_mode="full_during_search",
    )
    marker = Path(data) / "preparation.json"
    if marker.is_file():
        metadata = json.loads(marker.read_text())
        hours = metadata.get("splits", {}).get("train", {}).get("hours")
        if hours is not None:
            contract["training_hours"] = hours
    resolved["training_mode"] = "full_during_search"
    resolved.pop("full_training")
    return resolved


def validate_full_training_data(data: Path) -> None:
    """Reject short screening datasets when a full-data run is requested."""
    marker = json.loads((data / "preparation.json").read_text())
    if not marker.get("features_ready") or "parent_fingerprint" in marker:
        raise ValueError(
            "Full-budget search requires complete training features, not a search subset"
        )
    selection = marker.get("training_selection")
    if selection is not None:
        if selection != "full":
            raise ValueError("Full-budget search cannot use a subset preparation")
        return
    # Older full preparations lack the explicit selection field. Compare against
    # the source annotations rather than accepting a 1h preparation as 'full'.
    from ..tools.prepare_tedlium import read_stm_split

    corpus = marker.get("corpus")
    if not corpus:
        raise ValueError("Full training preparation has no source corpus identity")
    expected = len(read_stm_split(Path(corpus), "train"))
    actual = marker.get("splits", {}).get("train", {}).get("utterances")
    if not expected or actual != expected:
        raise ValueError(
            "Prepared train split does not contain the complete source corpus"
        )


def retrain_selected(
    playground, baseline: dict, candidates: list[dict]
) -> tuple[dict, list[dict]]:
    """Compatibility guard: search results are final models, never submit new training."""
    if (playground.sure_config.get("full_training") or {}).get("enabled"):
        raise ValueError(
            "Post-search retraining is retired; promote_full_training_to_search before running search"
        )
    return baseline, candidates
