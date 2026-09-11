"""Canonical defects used to target research idea edits."""

from __future__ import annotations

DEFECTS = frozenset(
    {
        "stagnant_novelty", "unclear_mechanism", "validation_gap",
        "brittle_single_path", "rare_regime_failure", "weak_fallback_behavior",
        "feature_dumping", "monolithic_design", "harder_to_ablate",
        "scale_mismatch", "coordination_failure", "latency_bottleneck",
        "responsibility_entanglement", "silent_failure", "drift",
        "open_loop_fragility", "theory_gap", "weak_generalization",
        "over_conservative_execution", "rollback_blindspot", "unexplored_gap",
    }
)


def validate_defects(defects: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(defects))
    unknown = set(normalized) - DEFECTS
    if unknown:
        raise ValueError(f"unknown defects: {sorted(unknown)}")
    return normalized
