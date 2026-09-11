"""Versioned contract for the research idea behavior retained by research_idea."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

ALGORITHM_SPEC_VERSION = "xlab.research_idea.algorithm-spec.v2"
ALGORITHM_ID = "xlab.research_idea.algorithm.v2"
WORKFLOW_ID = "xlab.research_idea.workflow.v1"
STATE_LAYOUT_ID = "xlab.research_idea.state-layout.v1"

DEFAULT_PROMPT_MODE = "default"
PRODUCTION_PROMPT_MODE = "conceptual_surprise"
PROMPT_ROUTING_PROFILE_ID = "xlab.research_idea.prompt-routing.v2"
EVIDENCE_SELECTION_PROFILE_ID = "xlab.research_idea.evidence-selection.v2"
OPERATOR_RETRIEVAL_PROFILE_ID = "xlab.research_idea.operator-retrieval.v2"
COMPONENT_NOVELTY_PROFILE_ID = "xlab.research_idea.component-novelty.v2"
SCIENTIFIC_STATE_PROFILE_ID = "xlab.research_idea.scientific-state.v2"
EVALUATION_PROFILE_ID = "xlab.research_idea.evaluation.v2"
FUSION_EVALUATION_PROFILE_ID = "xlab.research_idea.fusion-referee.v2"
IMPLEMENTATION_PROVENANCE = MappingProxyType(
    {
        "algorithm": ALGORITHM_ID,
        "spec_version": ALGORITHM_SPEC_VERSION,
        "source": "XLab package-native research idea implementation",
        "runtime": "package-native",
        "native_reconstruction": True,
    }
)

SCORE_WEIGHT_FIELDS = (
    "alignment_weight",
    "complexity_weight",
    "novelty_weight",
    "surprise_weight",
    "impact_weight",
    "feasibility_weight",
    "clarity_weight",
    "conciseness_weight",
    "risk_weight",
    "protocol_weight",
)

IDEA_TASTE_WEIGHTS: Mapping[str, Mapping[str, float]] = MappingProxyType(
    {
        "moonshot_inventor": MappingProxyType(
            {
                "alignment_weight": 0.14,
                "complexity_weight": 0.06,
                "novelty_weight": 0.27,
                "surprise_weight": 0.22,
                "impact_weight": 0.18,
                "feasibility_weight": 0.05,
                "clarity_weight": 0.03,
                "conciseness_weight": 0.02,
                "risk_weight": 0.02,
                "protocol_weight": 0.01,
            }
        ),
        "bridge_builder": MappingProxyType(
            {
                "alignment_weight": 0.16,
                "complexity_weight": 0.05,
                "novelty_weight": 0.17,
                "surprise_weight": 0.10,
                "impact_weight": 0.18,
                "feasibility_weight": 0.13,
                "clarity_weight": 0.06,
                "conciseness_weight": 0.02,
                "risk_weight": 0.07,
                "protocol_weight": 0.06,
            }
        ),
        "steady_engineer": MappingProxyType(
            {
                "alignment_weight": 0.17,
                "complexity_weight": 0.04,
                "novelty_weight": 0.16,
                "surprise_weight": 0.07,
                "impact_weight": 0.18,
                "feasibility_weight": 0.16,
                "clarity_weight": 0.08,
                "conciseness_weight": 0.02,
                "risk_weight": 0.07,
                "protocol_weight": 0.05,
            }
        ),
        "ambitious_realist": MappingProxyType(
            {
                "alignment_weight": 0.15,
                "complexity_weight": 0.05,
                "novelty_weight": 0.22,
                "surprise_weight": 0.14,
                "impact_weight": 0.20,
                "feasibility_weight": 0.08,
                "clarity_weight": 0.06,
                "conciseness_weight": 0.03,
                "risk_weight": 0.05,
                "protocol_weight": 0.02,
            }
        ),
        "evidence_first": MappingProxyType(
            {
                "alignment_weight": 0.16,
                "complexity_weight": 0.04,
                "novelty_weight": 0.17,
                "surprise_weight": 0.04,
                "impact_weight": 0.16,
                "feasibility_weight": 0.15,
                "clarity_weight": 0.08,
                "conciseness_weight": 0.02,
                "risk_weight": 0.11,
                "protocol_weight": 0.07,
            }
        ),
    }
)

IDEA_TASTE_MODES = tuple(IDEA_TASTE_WEIGHTS)


@dataclass(frozen=True)
class MetricChampion:
    label: str
    metric: str
    objective: Literal["max"] = "max"


METRIC_CHAMPIONS = (
    MetricChampion("novel", "novelty"),
    MetricChampion("feasible", "feasibility"),
    MetricChampion("concise", "conciseness"),
)
