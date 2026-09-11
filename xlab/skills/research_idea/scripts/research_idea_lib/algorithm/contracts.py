"""Typed contracts for the package-native research idea search core."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, Sequence, cast

from ..research_idea_spec import (
    ALGORITHM_ID,
    COMPONENT_NOVELTY_PROFILE_ID,
    EVALUATION_PROFILE_ID,
    OPERATOR_RETRIEVAL_PROFILE_ID,
    PRODUCTION_PROMPT_MODE,
    PROMPT_ROUTING_PROFILE_ID,
    IDEA_TASTE_MODES,
)

PromptMode = Literal["default", "conceptual_surprise"]
TighterRole = Literal["core_mechanism", "support_module", "protocol", "guardrail"]
OperatorGroundingKind = Literal["theory_transfer", "mechanism_commit"]
OperatorAttemptOutcome = Literal[
    "applied",
    "skipped_empty",
    "explicit_empty",
    "duplicate_exhausted",
    "attachment_rejected",
    "error",
]
RolloutBlockerStage = Literal["grounding", "generation", "evaluation", "component_novelty"]
_ALLOWED_PROMPT_MODES = {"default", "conceptual_surprise"}
_ALLOWED_OPERATOR_ATTEMPT_OUTCOMES = {
    "applied",
    "skipped_empty",
    "explicit_empty",
    "duplicate_exhausted",
    "attachment_rejected",
    "error",
}
_ALLOWED_ROLLOUT_BLOCKER_STAGES = {"grounding", "generation", "evaluation", "component_novelty"}

MIN_COMPONENTS = 1
MAX_COMPONENTS = 5


def _digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clean(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _scientific_text(value: str) -> str:
    return str(value).lower()


def validate_prompt_mode(value: str) -> PromptMode:
    normalized = str(value).casefold().strip()
    if normalized not in _ALLOWED_PROMPT_MODES:
        raise ValueError(f"unsupported prompt mode: {value}")
    return normalized  # type: ignore[return-value]


@dataclass(frozen=True)
class OperatorAttemptRecord:
    parent_node_id: int
    operator: str
    plan_digest: str
    outcome: OperatorAttemptOutcome
    grounding_explicit_empty: bool = False
    query_provenance_json: str | None = None
    references_provenance_json: str | None = None
    retrieval_provenance_json: str | None = None

    def __post_init__(self) -> None:
        if self.parent_node_id < 0:
            raise ValueError("operator attempt parent node ID cannot be negative")
        if not self.operator.strip() or not self.plan_digest.strip():
            raise ValueError("operator attempt requires an operator and plan digest")
        if self.outcome not in _ALLOWED_OPERATOR_ATTEMPT_OUTCOMES:
            raise ValueError(f"unsupported operator attempt outcome: {self.outcome}")
        if not isinstance(self.grounding_explicit_empty, bool):
            raise ValueError("operator attempt grounding_explicit_empty must be a boolean")
        for field_name in (
            "query_provenance_json",
            "references_provenance_json",
            "retrieval_provenance_json",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            try:
                parsed = json.loads(value)
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"{field_name} must be valid JSON") from error
            if not isinstance(parsed, dict):
                raise ValueError(f"{field_name} must contain a JSON object")

    def to_payload(self) -> dict[str, Any]:
        return {
            "parent_node_id": self.parent_node_id,
            "operator": self.operator,
            "plan_digest": self.plan_digest,
            "outcome": self.outcome,
            "grounding_explicit_empty": self.grounding_explicit_empty,
            "query_provenance": _optional_json_object(self.query_provenance_json),
            "references_provenance": _optional_json_object(self.references_provenance_json),
            "retrieval_provenance": _optional_json_object(self.retrieval_provenance_json),
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "OperatorAttemptRecord":
        return cls(
            parent_node_id=_nonnegative_int(value.get("parent_node_id"), "operator attempt parent_node_id"),
            operator=str(value.get("operator") or "").strip(),
            plan_digest=str(value.get("plan_digest") or "").strip(),
            outcome=str(value.get("outcome") or ""),  # type: ignore[arg-type]
            grounding_explicit_empty=_boolean(
                value.get("grounding_explicit_empty", False),
                "operator attempt grounding_explicit_empty",
            ),
            query_provenance_json=_optional_json_payload(value.get("query_provenance")),
            references_provenance_json=_optional_json_payload(value.get("references_provenance")),
            retrieval_provenance_json=_optional_json_payload(value.get("retrieval_provenance")),
        )


@dataclass(frozen=True)
class RolloutBlockerRecord:
    parent_node_id: int
    operator: str
    plan_digest: str
    stage: RolloutBlockerStage
    error_type: str
    message: str

    def __post_init__(self) -> None:
        if self.parent_node_id < 0:
            raise ValueError("rollout blocker parent node ID cannot be negative")
        if not self.operator.strip() or not self.plan_digest.strip():
            raise ValueError("rollout blocker requires an operator and plan digest")
        if self.stage not in _ALLOWED_ROLLOUT_BLOCKER_STAGES:
            raise ValueError(f"unsupported rollout blocker stage: {self.stage}")
        if not self.error_type.strip() or not self.message.strip():
            raise ValueError("rollout blocker requires an error type and message")

    def to_payload(self) -> dict[str, Any]:
        return {
            "parent_node_id": self.parent_node_id,
            "operator": self.operator,
            "plan_digest": self.plan_digest,
            "stage": self.stage,
            "error_type": self.error_type,
            "message": self.message,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "RolloutBlockerRecord":
        return cls(
            parent_node_id=_nonnegative_int(value.get("parent_node_id"), "rollout blocker parent_node_id"),
            operator=str(value.get("operator") or "").strip(),
            plan_digest=str(value.get("plan_digest") or "").strip(),
            stage=str(value.get("stage") or ""),  # type: ignore[arg-type]
            error_type=str(value.get("error_type") or "").strip(),
            message=str(value.get("message") or "").strip(),
        )


def _optional_json_object(value: str | None) -> dict[str, Any] | None:
    return json.loads(value) if value is not None else None


def _optional_json_payload(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("provenance must be a mapping or null")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _boolean(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class IdeaComponent:
    name: str
    description: str

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.description.strip():
            raise ValueError("components require a name and description")


@dataclass(frozen=True)
class IdeaState:
    title: str
    abstract: str
    core_contribution: str
    method: str
    risks: str
    components: tuple[IdeaComponent, ...]
    tags: tuple[str, ...] = ()
    root_domains: tuple[str, ...] = ()
    textual_identity: str = field(init=False)
    scientific_identity: str = field(init=False)
    semantic_identity: str = field(init=False)

    def __post_init__(self) -> None:
        scientific_text = {
            "title": self.title,
            "abstract": self.abstract,
            "core_contribution": self.core_contribution,
            "method": self.method,
            "risks": self.risks,
        }
        empty_fields = [name for name, value in scientific_text.items() if not value.strip()]
        if empty_fields:
            raise ValueError(f"idea scientific text fields cannot be empty: {', '.join(empty_fields)}")
        if not MIN_COMPONENTS <= len(self.components) <= MAX_COMPONENTS:
            raise ValueError(f"an idea requires {MIN_COMPONENTS}..{MAX_COMPONENTS} components")
        object.__setattr__(self, "tags", _clean(self.tags))
        object.__setattr__(self, "root_domains", _clean(self.root_domains))
        object.__setattr__(self, "textual_identity", _digest(self.to_payload()))
        canonical_scientific_state = "|".join(
            (
                _scientific_text(self.title),
                _scientific_text(self.core_contribution),
                _scientific_text(self.method),
                ",".join(sorted(self.tags)),
                ",".join(self.root_domains),
                ",".join(sorted(component.name for component in self.components)),
                "|".join(
                    f"{component.name}:{_scientific_text(component.description)}"
                    for component in self.components
                ),
            )
        )
        scientific_identity = hashlib.sha256(
            canonical_scientific_state.encode("utf-8")
        ).hexdigest()
        object.__setattr__(self, "scientific_identity", scientific_identity)
        object.__setattr__(self, "semantic_identity", scientific_identity)

    def to_payload(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "core_contribution": self.core_contribution,
            "method": self.method,
            "risks": self.risks,
            "components": [
                {"name": component.name, "description": component.description}
                for component in self.components
            ],
            "tags": list(self.tags),
            "root_domains": list(self.root_domains),
        }


@dataclass(frozen=True)
class RankedEvidence:
    """Immutable provider-facing survey evidence in workflow rank order."""

    evidence_id: str
    kind: str
    text: str
    provenance_json: str
    paper_ids: tuple[str, ...] = ()
    source_id: str | None = None
    rank: int = 0

    def __post_init__(self) -> None:
        if not self.evidence_id.strip() or not self.kind.strip() or not self.text.strip():
            raise ValueError("ranked evidence requires an ID, kind, and text")
        if self.rank < 1:
            raise ValueError("ranked evidence requires a positive rank")
        try:
            provenance = json.loads(self.provenance_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("evidence provenance_json must be valid JSON") from error
        if not isinstance(provenance, dict) or not provenance:
            raise ValueError("ranked evidence provenance cannot be empty")
        object.__setattr__(self, "paper_ids", _clean(self.paper_ids))

    @classmethod
    def from_payload(cls, value: Mapping[str, Any], *, rank: int) -> "RankedEvidence":
        provenance = value.get("provenance")
        if not isinstance(provenance, Mapping) or not provenance:
            raise ValueError("ranked evidence provenance must be a non-empty mapping")
        raw_paper_ids = value.get("paper_ids", ())
        if isinstance(raw_paper_ids, (str, bytes)) or not isinstance(raw_paper_ids, Sequence):
            raise ValueError("ranked evidence paper_ids must be a sequence")
        return cls(
            evidence_id=str(value.get("evidence_id") or "").strip(),
            kind=str(value.get("kind") or "").strip(),
            text=str(value.get("text") or "").strip(),
            provenance_json=json.dumps(
                provenance, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
            paper_ids=tuple(str(item) for item in raw_paper_ids),
            source_id=str(value.get("source_id") or "").strip() or None,
            rank=rank,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "text": self.text,
            "provenance": json.loads(self.provenance_json),
            "paper_ids": list(self.paper_ids),
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class RefinementBoundary:
    """Machine-readable edit boundary, kept separate from human scope prose."""

    allowed_component_ids: tuple[str, ...] = ()
    allowed_fields: tuple[str, ...] = ()
    allowed_edit_kinds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_component_ids", _clean(self.allowed_component_ids))
        object.__setattr__(self, "allowed_fields", _clean(self.allowed_fields))
        object.__setattr__(self, "allowed_edit_kinds", _clean(self.allowed_edit_kinds))
        valid_fields = {"title", "abstract", "core_contribution", "method", "risks", "components"}
        invalid_fields = set(self.allowed_fields) - valid_fields
        if invalid_fields:
            raise ValueError(f"unsupported refinement fields: {sorted(invalid_fields)}")
        valid_edits = {
            "ADD_COMPONENT",
            "REMOVE_COMPONENT",
            "REPLACE_COMPONENT",
            "REWIRE",
            "ADD_PROTOCOL",
        }
        invalid_edits = set(self.allowed_edit_kinds) - valid_edits
        if invalid_edits:
            raise ValueError(f"unsupported refinement edit kinds: {sorted(invalid_edits)}")

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "RefinementBoundary":
        def strings(field_name: str) -> tuple[str, ...]:
            raw = value.get(field_name, ())
            if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
                raise ValueError(f"refinement boundary {field_name} must be a sequence")
            return tuple(str(item) for item in raw)

        return cls(
            allowed_component_ids=strings("allowed_component_ids"),
            allowed_fields=strings("allowed_fields"),
            allowed_edit_kinds=strings("allowed_edit_kinds"),
        )

    def to_payload(self) -> dict[str, list[str]]:
        return {
            "allowed_component_ids": list(self.allowed_component_ids),
            "allowed_fields": list(self.allowed_fields),
            "allowed_edit_kinds": list(self.allowed_edit_kinds),
        }


@dataclass(frozen=True)
class GroundedComponentEvidence:
    core_node_id: str
    component_id: str
    component_name: str
    component_description: str
    score: float
    domain: str | None
    resource_identity: str
    trace_identity: str

    def __post_init__(self) -> None:
        required = (
            self.core_node_id,
            self.component_id,
            self.component_name,
            self.component_description,
            self.resource_identity,
            self.trace_identity,
        )
        if not all(str(value).strip() for value in required):
            raise ValueError("grounded component evidence fields cannot be empty")
        if not 0 <= float(self.score) <= 1:
            raise ValueError("grounded component evidence score must be in [0, 1]")

    def to_payload(self) -> dict[str, Any]:
        return {
            "core_node_id": self.core_node_id,
            "component_id": self.component_id,
            "component_name": self.component_name,
            "component_description": self.component_description,
            "score": float(self.score),
            "domain": self.domain,
            "resource_identity": self.resource_identity,
            "trace_identity": self.trace_identity,
        }


@dataclass(frozen=True)
class OperatorGrounding:
    kind: OperatorGroundingKind
    query: str
    needed_content: str
    expected_role: str
    evidence: tuple[GroundedComponentEvidence, ...] = ()
    profile_id: str = OPERATOR_RETRIEVAL_PROFILE_ID

    def __post_init__(self) -> None:
        if self.kind not in {"theory_transfer", "mechanism_commit"}:
            raise ValueError(f"unsupported operator grounding kind: {self.kind}")
        if not self.query.strip() or not self.needed_content.strip() or not self.expected_role.strip():
            raise ValueError("operator grounding query fields cannot be empty")
        if not self.profile_id.strip():
            raise ValueError("operator grounding profile cannot be empty")

    @property
    def digest(self) -> str:
        return _digest(self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "query": self.query,
            "needed_content": self.needed_content,
            "expected_role": self.expected_role,
            "profile_id": self.profile_id,
            "evidence": [item.to_payload() for item in self.evidence],
        }


@dataclass(frozen=True)
class FusionValidationFeedback:
    attempt: int
    code: str
    message: str
    source_modes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.attempt < 1 or not self.code.strip() or not self.message.strip():
            raise ValueError("fusion validation feedback requires attempt, code, and message")
        object.__setattr__(self, "source_modes", _clean(self.source_modes))

    def to_payload(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "code": self.code,
            "message": self.message,
            "source_modes": list(self.source_modes),
        }


@dataclass(frozen=True)
class TighterRoleVariant:
    source_mode: str
    source_component_name: str
    source_component_description: str
    narrowed_name: str
    narrowed_description: str
    role: TighterRole
    evidence_ids: tuple[str, ...]
    narrowing_rationale: str

    def __post_init__(self) -> None:
        required = (
            self.source_mode,
            self.source_component_name,
            self.source_component_description,
            self.narrowed_name,
            self.narrowed_description,
            self.role,
            self.narrowing_rationale,
        )
        if not all(str(value).strip() for value in required):
            raise ValueError("tighter role variants require complete source provenance")
        if self.source_mode not in IDEA_TASTE_MODES:
            raise ValueError("tighter role variant source_mode must be canonical")
        if self.role not in {"core_mechanism", "support_module", "protocol", "guardrail"}:
            raise ValueError("tighter role variant role is not supported")
        evidence_ids = _clean(self.evidence_ids)
        if not evidence_ids:
            raise ValueError("tighter role variants require source-owned evidence")
        if self.narrowed_name.strip() == self.source_component_name.strip():
            raise ValueError("tighter role variant must narrow to a distinct component")
        object.__setattr__(self, "evidence_ids", evidence_ids)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TighterRoleVariant:
        if not isinstance(payload, Mapping):
            raise ValueError("tighter_role_variant must be a mapping")
        try:
            evidence = payload["evidence_ids"]
            if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
                raise ValueError("tighter role variant evidence_ids must be a sequence")
            return cls(
                source_mode=str(payload["source_mode"]).strip(),
                source_component_name=str(payload["source_component_name"]).strip(),
                source_component_description=str(payload["source_component_description"]).strip(),
                narrowed_name=str(payload["narrowed_name"]).strip(),
                narrowed_description=str(payload["narrowed_description"]).strip(),
                role=cast(TighterRole, str(payload["role"]).strip()),
                evidence_ids=tuple(str(item).strip() for item in evidence),
                narrowing_rationale=str(payload["narrowing_rationale"]).strip(),
            )
        except KeyError as error:
            raise ValueError(f"tighter role variant is missing {error.args[0]}") from error

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_mode": self.source_mode,
            "source_component_name": self.source_component_name,
            "source_component_description": self.source_component_description,
            "narrowed_name": self.narrowed_name,
            "narrowed_description": self.narrowed_description,
            "role": self.role,
            "evidence_ids": list(self.evidence_ids),
            "narrowing_rationale": self.narrowing_rationale,
        }


@dataclass(frozen=True)
class IdeaProviderContext:
    """Immutable grounding shared by every provider call in one mode search."""

    evidence: tuple[RankedEvidence, ...] = ()
    mature_idea: IdeaState | None = None
    refinement_scope: tuple[str, ...] = ()
    refinement_boundary: RefinementBoundary | None = None
    memory_hints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(item.rank != index for index, item in enumerate(self.evidence, start=1)):
            raise ValueError("ranked evidence must have contiguous ranks in tuple order")
        if len({item.evidence_id for item in self.evidence}) != len(self.evidence):
            raise ValueError("ranked evidence IDs must be unique")
        object.__setattr__(self, "refinement_scope", _clean(self.refinement_scope))
        object.__setattr__(self, "memory_hints", _clean(self.memory_hints))

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.evidence)

    @property
    def evidence_digest(self) -> str:
        return _digest([item.to_payload() for item in self.evidence])

    @property
    def memory_digest(self) -> str:
        return _digest(list(self.memory_hints))

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "evidence": [item.to_payload() for item in self.evidence],
            "mature_idea": self.mature_idea.to_payload() if self.mature_idea else None,
            "refinement_scope": list(self.refinement_scope),
            "memory_hints": list(self.memory_hints),
        }
        if self.refinement_boundary is not None:
            payload["refinement_boundary"] = self.refinement_boundary.to_payload()
        return payload


@dataclass(frozen=True)
class CacheIdentity:
    state_textual_identity: str
    state_scientific_identity: str
    idea_taste_mode: str
    prompt_mode: PromptMode
    prompt_profile: str
    runtime_profile: str
    diagnostic_profile: str
    evaluator_profile: str
    algorithm_version: str
    evidence_digest: str
    memory_digest: str = "no-memory"
    component_novelty_profile: str = COMPONENT_NOVELTY_PROFILE_ID
    component_novelty_resource_digest: str = "no-component-novelty-resource"
    operator_grounding_digest: str = "no-operator-grounding"
    digest: str = field(init=False)

    @property
    def state_semantic_identity(self) -> str:
        return self.state_scientific_identity

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_mode", validate_prompt_mode(self.prompt_mode))
        values = {
            "state_textual_identity": self.state_textual_identity,
            "state_scientific_identity": self.state_scientific_identity,
            "idea_taste_mode": self.idea_taste_mode,
            "prompt_mode": self.prompt_mode,
            "prompt_profile": self.prompt_profile,
            "runtime_profile": self.runtime_profile,
            "diagnostic_profile": self.diagnostic_profile,
            "evaluator_profile": self.evaluator_profile,
            "algorithm_version": self.algorithm_version,
            "evidence_digest": self.evidence_digest,
            "memory_digest": self.memory_digest,
            "component_novelty_profile": self.component_novelty_profile,
            "component_novelty_resource_digest": self.component_novelty_resource_digest,
            "operator_grounding_digest": self.operator_grounding_digest,
        }
        if not all(values.values()):
            raise ValueError("cache identity fields cannot be empty")
        object.__setattr__(self, "digest", _digest(values))

    @classmethod
    def for_state(
        cls,
        state: IdeaState,
        *,
        idea_taste_mode: str,
        runtime_profile: str,
        evaluator_profile: str,
        evidence_digest: str,
        memory_digest: str = "no-memory",
        prompt_mode: PromptMode = PRODUCTION_PROMPT_MODE,
        prompt_profile: str = PROMPT_ROUTING_PROFILE_ID,
        diagnostic_profile: str = EVALUATION_PROFILE_ID,
        component_novelty_profile: str = COMPONENT_NOVELTY_PROFILE_ID,
        component_novelty_resource_digest: str = "no-component-novelty-resource",
        operator_grounding_digest: str = "no-operator-grounding",
    ) -> "CacheIdentity":
        return cls(
            state_textual_identity=state.textual_identity,
            state_scientific_identity=state.scientific_identity,
            idea_taste_mode=idea_taste_mode,
            prompt_mode=prompt_mode,
            prompt_profile=prompt_profile,
            runtime_profile=runtime_profile,
            diagnostic_profile=diagnostic_profile,
            evaluator_profile=evaluator_profile,
            algorithm_version=ALGORITHM_ID,
            evidence_digest=evidence_digest,
            memory_digest=memory_digest,
            component_novelty_profile=component_novelty_profile,
            component_novelty_resource_digest=component_novelty_resource_digest,
            operator_grounding_digest=operator_grounding_digest,
        )

    def to_metadata(self) -> dict[str, str]:
        return {
            "digest": self.digest,
            "state_textual_identity": self.state_textual_identity,
            "state_scientific_identity": self.state_scientific_identity,
            "state_semantic_identity": self.state_scientific_identity,
            "idea_taste_mode": self.idea_taste_mode,
            "prompt_mode": self.prompt_mode,
            "prompt_profile": self.prompt_profile,
            "runtime_profile": self.runtime_profile,
            "diagnostic_profile": self.diagnostic_profile,
            "evaluator_profile": self.evaluator_profile,
            "algorithm_version": self.algorithm_version,
            "evidence_digest": self.evidence_digest,
            "memory_digest": self.memory_digest,
            "component_novelty_profile": self.component_novelty_profile,
            "component_novelty_resource_digest": self.component_novelty_resource_digest,
            "operator_grounding_digest": self.operator_grounding_digest,
        }


@dataclass(frozen=True)
class ProviderUsage:
    evaluator_calls: int = 0
    generation_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0

    def __add__(self, other: "ProviderUsage") -> "ProviderUsage":
        return ProviderUsage(
            evaluator_calls=self.evaluator_calls + other.evaluator_calls,
            generation_calls=self.generation_calls + other.generation_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost=self.cost + other.cost,
        )


@dataclass(frozen=True)
class EvaluationResponse:
    metrics: Mapping[str, float]
    confidence: float = 0.0
    detected_defects: tuple[str, ...] = ()
    feedback: str = ""
    usage: ProviderUsage = ProviderUsage(evaluator_calls=1)


@dataclass(frozen=True)
class GenerationResponse:
    state: IdeaState
    usage: ProviderUsage = ProviderUsage(generation_calls=1)


@dataclass(frozen=True)
class GenerationRequest:
    idea_taste_mode: str
    parent: IdeaState
    plan: Any
    seed: int
    memory_hints: tuple[str, ...] = ()
    context: IdeaProviderContext = IdeaProviderContext()
    prompt_mode: PromptMode = PRODUCTION_PROMPT_MODE
    grounding: OperatorGrounding | None = None

    def __post_init__(self) -> None:
        if not self.idea_taste_mode.strip():
            raise ValueError("generation request idea taste cannot be empty")
        object.__setattr__(self, "prompt_mode", validate_prompt_mode(self.prompt_mode))

    @property
    def mode(self) -> str:
        return self.idea_taste_mode


class ResearchIdeaProvider(Protocol):
    """Abstract provider boundary; adapters may use any current or future backend."""

    def evaluate(
        self,
        state: IdeaState,
        *,
        idea_taste_mode: str,
        prompt_mode: PromptMode = PRODUCTION_PROMPT_MODE,
        diagnostic: bool,
        context: IdeaProviderContext = IdeaProviderContext(),
    ) -> EvaluationResponse: ...

    def generate(self, request: GenerationRequest) -> GenerationResponse: ...
