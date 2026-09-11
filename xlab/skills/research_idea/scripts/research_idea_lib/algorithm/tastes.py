"""Frozen research idea idea tastes."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

try:
    from ..research_idea_spec import IDEA_TASTE_MODES, IDEA_TASTE_WEIGHTS
except ImportError:  # pragma: no cover - standalone package fallback
    IDEA_TASTE_MODES = (
        "moonshot_inventor",
        "bridge_builder",
        "steady_engineer",
        "ambitious_realist",
        "evidence_first",
    )
    _vectors = {
        "moonshot_inventor": (0.14, 0.06, 0.27, 0.22, 0.18, 0.05, 0.03, 0.02, 0.02, 0.01),
        "bridge_builder": (0.16, 0.05, 0.17, 0.10, 0.18, 0.13, 0.06, 0.02, 0.07, 0.06),
        "steady_engineer": (0.17, 0.04, 0.16, 0.07, 0.18, 0.16, 0.08, 0.02, 0.07, 0.05),
        "ambitious_realist": (0.15, 0.05, 0.22, 0.14, 0.20, 0.08, 0.06, 0.03, 0.05, 0.02),
        "evidence_first": (0.16, 0.04, 0.17, 0.04, 0.16, 0.15, 0.08, 0.02, 0.11, 0.07),
    }
    _fields = ("alignment_weight", "complexity_weight", "novelty_weight", "surprise_weight", "impact_weight", "feasibility_weight", "clarity_weight", "conciseness_weight", "risk_weight", "protocol_weight")
    IDEA_TASTE_WEIGHTS = {mode: dict(zip(_fields, vector, strict=True)) for mode, vector in _vectors.items()}


_TASTE_DETAILS = {
    "moonshot_inventor": (
        "Moonshot Inventor",
        "Prioritize 0-to-1 mechanisms and outsized upside, while tolerating higher implementation risk and structural complexity.",
        {
            "mechanism-commit-innovation": 1.0,
            "theory-transfer-injection": 0.7,
            "alternative-path-contrast": 0.15,
            "multi-scale-coordinator": 0.05,
            "hierarchical-decomposition": 0.35,
            "feedback-closed-loop": 0.05,
            "surgical-modularity": 0.25,
            "speculative-execution-with-repair": 0.2,
        },
        "Prefer one bold mechanism-level move with outsized upside. Let the core novelty live in the task-solving path, not in extra scaffolding.",
    ),
    "bridge_builder": (
        "Bridge Builder",
        "Favor cross-domain transfer: reusing strong ideas from domain A in domain B, with less emphasis on raw novelty and more on fit.",
        {
            "theory-transfer-injection": 1.0,
            "multi-scale-coordinator": 0.15,
            "hierarchical-decomposition": 0.6,
            "alternative-path-contrast": 0.15,
            "feedback-closed-loop": 0.1,
            "mechanism-commit-innovation": 0.25,
            "surgical-modularity": 0.25,
            "speculative-execution-with-repair": 0.15,
        },
        "Emphasize the transferable principle, adaptation point, and fit to the current domain. Make negative-transfer risks explicit.",
    ),
    "steady_engineer": (
        "Steady Engineer",
        "Optimize for stable, highly feasible ideas that are easy to execute and less likely to fail.",
        {
            "surgical-modularity": 1.0,
            "feedback-closed-loop": 0.35,
            "hierarchical-decomposition": 0.45,
            "alternative-path-contrast": 0.1,
            "mechanism-commit-innovation": 0.3,
            "multi-scale-coordinator": 0.05,
            "speculative-execution-with-repair": 0.2,
            "theory-transfer-injection": 0.15,
        },
        "Favor minimal, well-scoped edits with clean interfaces and clear validation. Avoid avoidable architectural sprawl.",
    ),
    "ambitious_realist": (
        "Ambitious Realist",
        "Default search posture for high-upside ideas: strongly favor novelty and impact, while keeping enough feasibility, alignment, and risk control to avoid drifting into empty moonshots.",
        {
            "mechanism-commit-innovation": 1.0,
            "theory-transfer-injection": 0.75,
            "surgical-modularity": 0.55,
            "multi-scale-coordinator": 0.08,
            "hierarchical-decomposition": 0.4,
            "alternative-path-contrast": 0.1,
            "feedback-closed-loop": 0.08,
            "speculative-execution-with-repair": 0.25,
        },
        "Push for ambitious mechanisms with real upside, but keep the causal story implementable and defensible.",
    ),
    "evidence_first": (
        "Evidence First",
        "Prefer ideas that are easy to validate, ablate, and defend with strong experimental evidence over flashy but brittle novelty.",
        {
            "surgical-modularity": 1.0,
            "feedback-closed-loop": 0.35,
            "alternative-path-contrast": 0.15,
            "hierarchical-decomposition": 0.4,
            "multi-scale-coordinator": 0.05,
            "mechanism-commit-innovation": 0.25,
            "theory-transfer-injection": 0.2,
            "speculative-execution-with-repair": 0.2,
        },
        "Prefer the lightest mechanism that yields strong ablations, stress tests, and fair comparisons. Do not add novelty that cannot be cleanly validated.",
    ),
}


@dataclass(frozen=True)
class Taste:
    mode: str
    label: str
    summary: str
    weights: Mapping[str, float]
    skill_bias: Mapping[str, float]
    instantiation_guidance: str


def get_taste(mode: str) -> Taste:
    if mode not in IDEA_TASTE_MODES:
        raise ValueError(f"unknown idea taste: {mode}")
    weights = dict(IDEA_TASTE_WEIGHTS[mode])
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError(f"weights for {mode} are not normalized")
    label, summary, skill_bias, guidance = _TASTE_DETAILS[mode]
    if set(skill_bias) != {
        "mechanism-commit-innovation",
        "alternative-path-contrast",
        "surgical-modularity",
        "multi-scale-coordinator",
        "hierarchical-decomposition",
        "feedback-closed-loop",
        "theory-transfer-injection",
        "speculative-execution-with-repair",
    }:
        raise ValueError(f"skill biases for {mode} do not cover the exact operator catalog")
    return Taste(
        mode=mode,
        label=label,
        summary=summary,
        weights=MappingProxyType(weights),
        skill_bias=MappingProxyType(dict(skill_bias)),
        instantiation_guidance=guidance,
    )
