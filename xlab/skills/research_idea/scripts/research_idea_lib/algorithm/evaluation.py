"""Ten-metric evaluation and scalarization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Mapping

from .tastes import Taste

POSITIVE_METRICS = (
    "novelty", "surprise", "feasibility", "clarity", "impact",
    "conciseness", "alignment_score", "protocol_score",
)
PENALTY_METRICS = ("risk", "complexity_penalty")
METRICS = POSITIVE_METRICS + PENALTY_METRICS


def _metric(value: float) -> int:
    return max(0, min(5, int(round(float(value)))))


def _postprocessed_metric(value: float, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not 0.0 <= number <= 5.0:
        raise ValueError(f"{field} must be between 0 and 5")
    return _metric(number)


@dataclass(frozen=True)
class EvaluationPostprocessing:
    """Optional external scores; supplied values are validated fail-closed."""

    component_novelty: float | None = None
    derived_protocol_score: float | None = None

    def __post_init__(self) -> None:
        for field in ("component_novelty", "derived_protocol_score"):
            value = getattr(self, field)
            if value is not None:
                _postprocessed_metric(value, field)


@dataclass(frozen=True)
class Evaluation:
    metrics: Mapping[str, int]
    confidence: float = 0.0
    detected_defects: tuple[str, ...] = ()
    feedback: str = ""

    @classmethod
    def from_payload(
        cls,
        metrics: Mapping[str, float],
        *,
        confidence: float = 0.0,
        detected_defects: tuple[str, ...] = (),
        feedback: str = "",
    ) -> "Evaluation":
        missing = set(METRICS) - set(metrics)
        if missing:
            raise ValueError(f"missing evaluation metrics: {sorted(missing)}")
        return cls(
            metrics={name: _metric(metrics[name]) for name in METRICS},
            confidence=max(0.0, min(1.0, float(confidence))),
            detected_defects=tuple(dict.fromkeys(detected_defects)),
            feedback=feedback,
        )

    def postprocess(self, scores: EvaluationPostprocessing) -> "Evaluation":
        """Apply Xcientist scoring fallbacks after primary payload validation."""
        if not isinstance(scores, EvaluationPostprocessing):
            raise TypeError("scores must be EvaluationPostprocessing")
        metrics = dict(self.metrics)
        if scores.component_novelty is not None:
            metrics["novelty"] = _postprocessed_metric(
                scores.component_novelty, "component_novelty"
            )
        if metrics["protocol_score"] <= 0 and scores.derived_protocol_score is not None:
            metrics["protocol_score"] = _postprocessed_metric(
                scores.derived_protocol_score, "derived_protocol_score"
            )
        return replace(self, metrics=metrics)

    def scalar(self, taste: Taste) -> float:
        scores = {
            "alignment_weight": self.metrics["alignment_score"],
            "complexity_weight": 5 - self.metrics["complexity_penalty"],
            "novelty_weight": self.metrics["novelty"],
            "surprise_weight": self.metrics["surprise"],
            "impact_weight": self.metrics["impact"],
            "feasibility_weight": self.metrics["feasibility"],
            "clarity_weight": self.metrics["clarity"],
            "conciseness_weight": self.metrics["conciseness"],
            "risk_weight": 5 - self.metrics["risk"],
            "protocol_weight": self.metrics["protocol_score"],
        }
        return sum(taste.weights[field] * value for field, value in scores.items())
