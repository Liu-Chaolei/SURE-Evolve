"""Package-native symbolic memory derived from component-removal experiments."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Literal, Mapping, Sequence

SymbolicResult = Literal["positive", "negative", "neutral", "inconclusive"]
_ALLOWED_RESULTS = {"positive", "negative", "neutral", "inconclusive"}


@dataclass(frozen=True)
class SymbolicMemoryRecord:
    component: str
    component_family: str
    result: SymbolicResult
    confidence: float = 0.5
    operation: str = "remove"
    metric: str = ""
    value: str = ""
    analysis: str = ""
    method_context: str = ""
    record_id: str = ""

    def __post_init__(self) -> None:
        if not self.component.strip() or not self.component_family.strip():
            raise ValueError("symbolic memory requires component and component_family")
        if self.result not in _ALLOWED_RESULTS:
            raise ValueError(f"unsupported symbolic result: {self.result}")
        if self.operation.casefold().strip() != "remove":
            raise ValueError("experiment symbolic memory requires component-removal results")
        if isinstance(self.confidence, bool) or not isfinite(float(self.confidence)):
            raise ValueError("symbolic confidence must be finite")
        if not 0 <= float(self.confidence) <= 1:
            raise ValueError("symbolic confidence must be in [0, 1]")

    @property
    def removal_effect(self) -> str:
        return {
            "positive": "removal_helped",
            "negative": "removal_hurt",
            "neutral": "no_directional_effect",
            "inconclusive": "unknown",
        }[self.result]

    @classmethod
    def from_payload(
        cls,
        value: Mapping[str, Any],
        *,
        component: str | None = None,
    ) -> "SymbolicMemoryRecord":
        operation = str(value.get("op") or value.get("operation") or "remove").casefold().strip()
        return cls(
            component=str(value.get("component") or component or "").strip(),
            component_family=str(value.get("component_family") or _component_family(component or value.get("component"))).strip(),
            result=str(value.get("result") or "inconclusive").casefold().strip(),  # type: ignore[arg-type]
            confidence=float(value.get("confidence", 0.5)),
            operation=operation,
            metric=str(value.get("metric") or "").strip(),
            value=str(value.get("value") or "").strip(),
            analysis=str(value.get("analysis") or "").strip(),
            method_context=str(value.get("method_context") or "").strip(),
            record_id=str(value.get("id") or value.get("record_id") or "").strip(),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "component_family": self.component_family,
            "op": self.operation,
            "result": self.result,
            "removal_effect": self.removal_effect,
            "confidence": float(self.confidence),
            "metric": self.metric,
            "value": self.value,
            "analysis": self.analysis,
            "method_context": self.method_context,
            "record_id": self.record_id,
        }


@dataclass(frozen=True)
class CandidateOutcomeRecord:
    """An observed complete intervention; never a component attribution."""

    record_id: str
    candidate_id: str
    parent_digest: str
    implementation_digest: str
    result_digest: str
    scope: dict[str, Any]
    change_set: list[dict[str, Any]]
    outcome: str
    metric: str = ""
    direction: str = ""
    parent_score: float | None = None
    candidate_score: float | None = None
    failure_category: str | None = None
    source_artifacts: list[str] = field(default_factory=list)
    kind: str = "candidate_outcome"
    schema_version: str = "xlab.candidate_outcome.v1"

    def __post_init__(self) -> None:
        if self.kind != "candidate_outcome" or self.schema_version != "xlab.candidate_outcome.v1":
            raise ValueError("Unsupported candidate outcome memory")
        if not self.record_id or not self.candidate_id or not self.result_digest or not self.scope:
            raise ValueError("Candidate outcome requires identity, scope and result provenance")
        if self.outcome not in {"improved", "worse", "tied", "incomparable", "execution_failed"}:
            raise ValueError("Invalid candidate outcome")
        for score in (self.parent_score, self.candidate_score):
            if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))
                                      or not isfinite(score)):
                raise ValueError("Memory scores must be finite numbers or null")
        if self.outcome in {"improved", "worse", "tied"}:
            if self.parent_score is None or self.candidate_score is None or not self.parent_digest:
                raise ValueError("Comparable outcome requires the measured parent")
            if self.direction not in {"lower", "higher"}:
                raise ValueError("Comparable outcome requires metric direction")
            gain = self.parent_score - self.candidate_score
            if self.direction == "higher":
                gain = -gain
            expected = "improved" if gain > 0 else "worse" if gain < 0 else "tied"
            if self.outcome != expected or self.failure_category:
                raise ValueError("Outcome contradicts recorded scores or execution status")

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    def hint(self) -> str:
        return json.dumps({"record_id": self.record_id, "candidate_id": self.candidate_id, "scope": self.scope,
                           "intervention": self.change_set, "outcome": self.outcome,
                           "parent_digest": self.parent_digest, "metric": self.metric,
                           "direction": self.direction, "parent_score": self.parent_score,
                           "candidate_score": self.candidate_score,
                           "failure_category": self.failure_category,
                           "interpretation": "Single-run whole-intervention observation; no component causality or significance claim."},
                          sort_keys=True, ensure_ascii=False)


@dataclass
class MemoryState:
    symbolic_records: list[SymbolicMemoryRecord] = field(default_factory=list)
    vector_memory_requested: bool = True
    symbolic_memory_only: bool = True
    candidate_records: list[CandidateOutcomeRecord] = field(default_factory=list)

    @property
    def vector_memory_effective(self) -> bool:
        return self.vector_memory_requested and not self.symbolic_memory_only

    @property
    def digest(self) -> str:
        payload = json.dumps(
            [record.to_payload() for record in [*self.symbolic_records, *self.candidate_records]],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def hints(self, *, method_context: str = "") -> tuple[str, ...]:
        """Return symbolic hints scoped to the current method when records declare a scope."""
        scope = " ".join(method_context.casefold().split())
        records = (
            record
            for record in self.symbolic_records
            if not record.method_context
            or record.method_context.casefold() in scope
            or scope in record.method_context.casefold()
        )
        return tuple(
            f"{record.component_family}/{record.component}: {record.removal_effect}; "
            f"confidence={record.confidence:.3f}"
            + (f"; metric={record.metric}" if record.metric else "")
            + (f"; analysis={record.analysis}" if record.analysis else "")
            for record in records
        ) + tuple(record.hint() for record in self.candidate_records)

    @classmethod
    def from_experiment_feedback(cls, feedback: Mapping[str, Any] | str | None) -> "MemoryState":
        if feedback is None or (isinstance(feedback, str) and not feedback.strip()):
            return cls()
        value: Any = feedback
        if isinstance(feedback, str):
            try:
                value = json.loads(feedback)
            except json.JSONDecodeError:
                return cls()
        if not isinstance(value, Mapping):
            raise ValueError("structured experiment feedback must be a mapping")
        if value.get("feedback_kind") == "candidate_outcome":
            records_by_id = {}
            for raw in value.get("records", []):
                record = CandidateOutcomeRecord(**raw)
                previous = records_by_id.get(record.record_id)
                if previous is not None and previous != record:
                    raise ValueError("Conflicting candidate outcome identity")
                records_by_id[record.record_id] = record
            return cls(candidate_records=[records_by_id[key] for key in sorted(records_by_id)])
        records = _records_from_feedback(value)
        return cls(records)


def _records_from_feedback(value: Mapping[str, Any]) -> list[SymbolicMemoryRecord]:
    raw_records: Any = value.get("records")
    if isinstance(raw_records, Mapping):
        raw_records = list(raw_records.values())
    if isinstance(raw_records, Sequence) and not isinstance(raw_records, (str, bytes)):
        return [SymbolicMemoryRecord.from_payload(item) for item in raw_records if isinstance(item, Mapping)]

    components = value.get("components")
    if isinstance(components, Mapping):
        return [
            SymbolicMemoryRecord.from_payload(item, component=str(name))
            for name, item in components.items()
            if isinstance(item, Mapping)
        ]

    ablation = value.get("ablation_results")
    if isinstance(ablation, Mapping):
        return _records_from_feedback(ablation)

    return []


def _component_family(value: Any) -> str:
    component = str(value or "").casefold().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", component).strip("_") or "unknown"
    role = "controller" if any(token in component for token in ("gate", "router", "control")) else "component"
    return f"{role}.{slug}"
