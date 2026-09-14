"""Explicit information boundaries for the SD factorial experiment."""
from __future__ import annotations

from copy import deepcopy


def policy(sure: dict) -> dict[str, bool]:
    raw = sure.get("ablation", {})
    if not isinstance(raw, dict) or set(raw) - {"use_literature", "use_feedback"}:
        raise ValueError("Invalid ablation policy")
    result = {key: raw.get(key, True) for key in ("use_literature", "use_feedback")}
    if any(type(value) is not bool for value in result.values()):
        raise ValueError("Ablation switches must be booleans")
    return result


def research_view(sure: dict, baseline: dict, current: dict, history: list,
                  artifacts: list, lineage: list) -> dict:
    enabled = policy(sure)["use_feedback"]
    return deepcopy({
        "current_best": current if enabled else baseline,
        "prior_rounds": history if enabled else [],
        "history_artifacts": artifacts if enabled else [],
        "parent_lineage": lineage if enabled else [],
    })
