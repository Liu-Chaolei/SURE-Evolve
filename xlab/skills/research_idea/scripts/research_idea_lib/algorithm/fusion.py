"""Package-native five-mode idea fusion, referee evaluation, and local repair."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from research_idea_lib.research_idea_spec import (
    ALGORITHM_ID,
    FUSION_EVALUATION_PROFILE_ID,
    IDEA_TASTE_MODES,
)

from .contracts import (
    FusionValidationFeedback,
    RefinementBoundary,
    TighterRoleVariant,
)

CANONICAL_MODES = tuple(IDEA_TASTE_MODES)
ALGORITHM_VERSION = ALGORITHM_ID
REFEREE_METRICS = (
    "novelty",
    "surprise",
    "feasibility",
    "clarity",
    "impact",
    "risk",
    "conciseness",
    "alignment_score",
    "complexity_penalty",
    "protocol_score",
)
ALLOWED_REPAIR_OPERATIONS = frozenset({"remove", "replace", "rewire"})
FUSION_EVALUATION_PROFILE = FUSION_EVALUATION_PROFILE_ID
FUSION_METRIC_WEIGHTS = {
    "alignment_score": 0.14,
    "complexity_penalty": 0.06,
    "novelty": 0.27,
    "surprise": 0.22,
    "impact": 0.18,
    "feasibility": 0.05,
    "clarity": 0.03,
    "conciseness": 0.02,
    "risk": 0.02,
    "protocol_score": 0.01,
}
_PENALTY_METRICS = frozenset({"complexity_penalty", "risk"})


@runtime_checkable
class FusionGenerator(Protocol):
    def generate(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


@runtime_checkable
class RefereeEvaluator(Protocol):
    def evaluate(self, candidate: Mapping[str, Any]) -> Mapping[str, Any]: ...


@runtime_checkable
class RepairGenerator(Protocol):
    def propose_repair(self, request: Mapping[str, Any]) -> Mapping[str, Any] | None: ...


@dataclass(frozen=True)
class FusionRequest:
    mode_inputs: Sequence[Mapping[str, Any]]
    topic: str = ""
    context: Mapping[str, Any] = field(default_factory=dict)
    minimum_components: int = 1
    refinement_scope: Sequence[str] = field(default_factory=tuple)
    refinement_boundary: RefinementBoundary | None = None
    protected_components: Sequence[str] = field(default_factory=tuple)
    max_fusion_draft_attempts: int = 5
    max_repair_steps: int = 10
    repair_patience: int = 5
    score_epsilon: float = 0.02


@dataclass(frozen=True)
class RefereeEvaluation:
    score: float
    metrics: Mapping[str, float]
    details: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "metrics": dict(self.metrics),
            "details": deepcopy(dict(self.details)),
        }


@dataclass(frozen=True)
class FusionResult:
    idea: Mapping[str, Any]
    selected_components: tuple[Mapping[str, Any], ...]
    rejected_components: tuple[Mapping[str, Any], ...]
    conflict_resolutions: tuple[Mapping[str, Any], ...]
    source_modes: tuple[str, ...]
    evaluation: RefereeEvaluation
    evolution: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any]

    @property
    def score(self) -> float:
        return self.evaluation.score

    def to_dict(self) -> dict[str, Any]:
        return {
            "idea": deepcopy(dict(self.idea)),
            "selected_components": deepcopy(list(self.selected_components)),
            "rejected_components": deepcopy(list(self.rejected_components)),
            "conflict_resolutions": deepcopy(list(self.conflict_resolutions)),
            "source_modes": list(self.source_modes),
            "evaluation": self.evaluation.to_dict(),
            "evolution": deepcopy(list(self.evolution)),
            "metadata": deepcopy(dict(self.metadata)),
        }


def fuse_five_modes(
    request: FusionRequest,
    *,
    generator: FusionGenerator,
    evaluator: RefereeEvaluator,
    repair_generator: RepairGenerator | None = None,
) -> FusionResult:
    """Fuse exactly one candidate from each canonical mode and locally repair it."""

    ordered_inputs = _ordered_mode_inputs(request.mode_inputs)
    _validate_bounds(request)
    fusion_context = deepcopy(dict(request.context))
    if request.refinement_scope:
        fusion_context["refinement_scope"] = list(request.refinement_scope)
    if request.refinement_boundary is not None:
        fusion_context["refinement_boundary"] = request.refinement_boundary.to_payload()
    generation_request = {
        "algorithm": ALGORITHM_VERSION,
        "topic": request.topic,
        "context": fusion_context,
        "source_modes": list(CANONICAL_MODES),
        "mode_inputs": deepcopy(ordered_inputs),
    }
    if request.minimum_components != 1:
        generation_request["minimum_components"] = request.minimum_components
    draft_feedback: list[FusionValidationFeedback] = []
    previous_draft = None
    validation_issues = []
    for attempt in range(1, request.max_fusion_draft_attempts + 1):
        draft_request = deepcopy(generation_request)
        draft_request["draft_attempt"] = attempt
        draft_request["max_draft_attempts"] = request.max_fusion_draft_attempts
        if draft_feedback:
            draft_request["validation_feedback"] = draft_feedback[-1].to_payload()
            if previous_draft is not None:
                draft_request["previous_draft"] = deepcopy(previous_draft)
            if validation_issues:
                draft_request["validation_issues"] = deepcopy(validation_issues)
        candidate_output = None
        try:
            candidate_output = generator.generate(draft_request)
            if not isinstance(candidate_output, Mapping):
                raise TypeError("Fusion generator must return a mapping.")
            best_idea, selected, rejected, conflicts = validate_fusion_output(
                candidate_output, ordered_inputs, minimum_components=request.minimum_components
            )
            component_count = len(_component_names(best_idea))
            if component_count < request.minimum_components:
                raise ValueError("Fused idea has fewer than minimum_components.")
            break
        except (TypeError, ValueError) as exc:
            previous_draft = getattr(exc, "previous_draft", None)
            if previous_draft is None and isinstance(candidate_output, Mapping):
                previous_draft = deepcopy(dict(candidate_output))
            validation_issues = getattr(exc, "validation_issues", [])
            draft_feedback.append(
                FusionValidationFeedback(
                    attempt=attempt,
                    code="invalid_fusion_draft",
                    message=str(exc)[:500],
                    source_modes=CANONICAL_MODES,
                )
            )
    else:
        raise ValueError(
            "Fusion generator exhausted semantic draft attempts: "
            + draft_feedback[-1].message
        )

    best_evaluation = _evaluate(evaluator, best_idea)
    evolution: list[dict[str, Any]] = [
        {
            "step": 0,
            "phase": "fusion",
            "status": "accepted",
            "score_after": best_evaluation.score,
            "evaluation": best_evaluation.to_dict(),
            "component_count": len(_component_names(best_idea)),
        }
    ]

    repairer = repair_generator
    if repairer is None and hasattr(generator, "propose_repair"):
        repairer = generator  # type: ignore[assignment]

    no_improvement = 0
    if repairer is not None:
        for step in range(1, request.max_repair_steps + 1):
            if no_improvement >= request.repair_patience:
                break
            repair_request = {
                "algorithm": ALGORITHM_VERSION,
                "step": step,
                "best_idea": deepcopy(best_idea),
                "best_evaluation": best_evaluation.to_dict(),
                "selected_components": deepcopy(selected),
                "rejected_components": deepcopy(rejected),
                "conflict_resolutions": deepcopy(conflicts),
                "source_modes": list(CANONICAL_MODES),
                "mode_inputs": deepcopy(ordered_inputs),
                "minimum_components": request.minimum_components,
                "refinement_scope": list(request.refinement_scope),
                "refinement_boundary": (
                    request.refinement_boundary.to_payload()
                    if request.refinement_boundary is not None
                    else None
                ),
                "protected_components": list(request.protected_components),
                "allowed_operations": _allowed_repair_operations(request.refinement_boundary),
                "score_epsilon": request.score_epsilon,
                "evolution": deepcopy(evolution),
            }
            proposal = repairer.propose_repair(deepcopy(repair_request))
            if proposal is None:
                evolution.append(
                    _repair_event(step, best_evaluation.score, "stopped", "repair_generator_stopped")
                )
                break

            try:
                candidate, candidate_selected, operations = _apply_repair(
                    best_idea,
                    selected,
                    proposal,
                    minimum_components=request.minimum_components,
                    refinement_boundary=request.refinement_boundary,
                    protected_components=request.protected_components,
                    mode_inputs=ordered_inputs,
                )
                _validate_component_provenance(
                    candidate_selected,
                    ordered_inputs,
                    "selected_components",
                    allow_tighter_role_variant=True,
                )
                _validate_variant_components(candidate, candidate_selected)
            except (TypeError, ValueError) as exc:
                no_improvement += 1
                evolution.append(
                    _repair_event(
                        step,
                        best_evaluation.score,
                        "rejected",
                        str(exc),
                        proposal=proposal,
                    )
                )
                continue

            candidate_evaluation = _evaluate(evaluator, candidate)
            accepted = candidate_evaluation.score > best_evaluation.score + request.score_epsilon
            evolution.append(
                {
                    "step": step,
                    "phase": "repair",
                    "status": "accepted" if accepted else "rejected",
                    "reason": "strict_score_improvement" if accepted else "insufficient_score_improvement",
                    "score_before": best_evaluation.score,
                    "score_after": candidate_evaluation.score,
                    "epsilon": request.score_epsilon,
                    "operations": deepcopy(operations),
                    "component_count_before": len(_component_names(best_idea)),
                    "component_count_after": len(_component_names(candidate)),
                    "evaluation": candidate_evaluation.to_dict(),
                }
            )
            if not accepted:
                no_improvement += 1
                continue

            best_idea = candidate
            selected = candidate_selected
            best_evaluation = candidate_evaluation
            no_improvement = 0

    metadata = {
        "algorithm": ALGORITHM_VERSION,
        "fusion_evaluation_profile": FUSION_EVALUATION_PROFILE,
        "fusion_metric_weights": dict(FUSION_METRIC_WEIGHTS),
        "canonical_modes": list(CANONICAL_MODES),
        "all_modes_completed": True,
        "fusion_draft_attempts": len(draft_feedback) + 1,
        "max_fusion_draft_attempts": request.max_fusion_draft_attempts,
        "repair_steps_attempted": sum(1 for item in evolution if item["phase"] == "repair"),
        "max_repair_steps": request.max_repair_steps,
        "repair_patience": request.repair_patience,
        "score_epsilon": request.score_epsilon,
    }
    return FusionResult(
        idea=deepcopy(best_idea),
        selected_components=tuple(deepcopy(selected)),
        rejected_components=tuple(deepcopy(rejected)),
        conflict_resolutions=tuple(deepcopy(conflicts)),
        source_modes=CANONICAL_MODES,
        evaluation=best_evaluation,
        evolution=tuple(deepcopy(evolution)),
        metadata=metadata,
    )


def run_fusion(
    mode_inputs: Sequence[Mapping[str, Any]],
    *,
    generator: FusionGenerator,
    evaluator: RefereeEvaluator,
    repair_generator: RepairGenerator | None = None,
    **options: Any,
) -> FusionResult:
    """Convenience entry point for callers coordinating through mappings."""

    return fuse_five_modes(
        FusionRequest(mode_inputs=mode_inputs, **options),
        generator=generator,
        evaluator=evaluator,
        repair_generator=repair_generator,
    )


def _ordered_mode_inputs(mode_inputs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(mode_inputs, (str, bytes)) or not isinstance(mode_inputs, Sequence):
        raise TypeError("mode_inputs must be a sequence of mappings.")
    by_mode: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    unknown: list[str] = []
    for raw in mode_inputs:
        if not isinstance(raw, Mapping):
            raise TypeError("Each mode input must be a mapping.")
        item = deepcopy(dict(raw))
        mode = str(item.get("mode") or item.get("idea_taste_mode") or "").strip()
        if mode not in CANONICAL_MODES:
            unknown.append(mode or "<missing>")
            continue
        if mode in by_mode:
            duplicates.append(mode)
            continue
        item["mode"] = mode
        by_mode[mode] = item

    missing = [mode for mode in CANONICAL_MODES if mode not in by_mode]
    if missing or duplicates or unknown or len(mode_inputs) != len(CANONICAL_MODES):
        parts = []
        if missing:
            parts.append("missing modes: " + ", ".join(missing))
        if duplicates:
            parts.append("duplicate modes: " + ", ".join(duplicates))
        if unknown:
            parts.append("unknown modes: " + ", ".join(unknown))
        if not parts:
            parts.append(f"expected exactly {len(CANONICAL_MODES)} mode inputs")
        raise ValueError("Invalid five-mode fusion input (" + "; ".join(parts) + ").")
    return [by_mode[mode] for mode in CANONICAL_MODES]


def _validate_bounds(request: FusionRequest) -> None:
    if type(request.minimum_components) is not int or request.minimum_components < 1:
        raise ValueError("minimum_components must be a positive integer.")
    if request.max_fusion_draft_attempts < 1:
        raise ValueError("max_fusion_draft_attempts must be at least one.")
    if request.max_repair_steps < 0:
        raise ValueError("max_repair_steps cannot be negative.")
    if request.repair_patience < 0:
        raise ValueError("repair_patience cannot be negative.")
    if not isfinite(float(request.score_epsilon)) or request.score_epsilon < 0:
        raise ValueError("score_epsilon must be a finite non-negative number.")


def validate_fusion_output(
    generated: Mapping[str, Any],
    mode_inputs: Sequence[Mapping[str, Any]],
    *, minimum_components: int = 1,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    idea = generated.get("idea", generated.get("fused_idea"))
    if not isinstance(idea, Mapping):
        raise ValueError("Fusion output must contain an idea mapping.")
    normalized_idea = deepcopy(dict(idea))
    for field_name in (
        "title",
        "abstract",
        "core_contribution",
        "method",
        "risks",
    ):
        if not str(normalized_idea.get(field_name) or "").strip():
            raise ValueError(f"Fused idea requires {field_name}.")
    names = _component_names(normalized_idea)
    if not names:
        raise ValueError("Fused idea must contain components.")
    if len(names) != len(set(names)):
        raise ValueError("Fused idea components must be unique.")
    if len(names) < minimum_components:
        raise ValueError("Fused idea has fewer than minimum_components.")

    metadata = generated.get("fusion_metadata")
    metadata = metadata if isinstance(metadata, Mapping) else generated
    selected = _mapping_list(metadata.get("selected_components"), "selected_components")
    rejected = _mapping_list(metadata.get("rejected_components"), "rejected_components")
    conflicts = _mapping_list(
        metadata.get("conflict_resolutions", metadata.get("conflicts_and_resolutions")),
        "conflict_resolutions",
    )
    _validate_component_provenance(selected, mode_inputs, "selected_components")
    _validate_component_provenance(rejected, mode_inputs, "rejected_components")
    _validate_conflict_provenance(conflicts, mode_inputs)
    _validate_variant_components(normalized_idea, selected)

    selected_names = {_provenance_component(item) for item in selected}
    idea_names = set(_component_names(normalized_idea))
    if selected_names != idea_names:
        raise ValueError("selected_components must describe every fused idea component exactly once.")
    return normalized_idea, selected, rejected, conflicts


def _mapping_list(value: Any, field_name: str) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"Fusion output {field_name} must be a list of mappings.")
    result = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"Fusion output {field_name} must contain only mappings.")
        result.append(deepcopy(dict(item)))
    return result


def _validate_component_provenance(
    records: Sequence[Mapping[str, Any]],
    mode_inputs: Sequence[Mapping[str, Any]],
    field_name: str,
    *,
    allow_tighter_role_variant: bool = False,
) -> None:
    evidence_by_mode = {str(item["mode"]): _source_evidence(item) for item in mode_inputs}
    components_by_mode = {str(item["mode"]): _source_components(item) for item in mode_inputs}
    names: set[tuple[str, str]] = set()
    for record in records:
        component = _provenance_component(record)
        mode = str(record.get("source_mode") or "").strip()
        evidence = _evidence_values(record)
        if not component or mode not in CANONICAL_MODES or not evidence:
            raise ValueError(f"Each {field_name} entry needs component, canonical source_mode, and evidence.")
        identity = (mode if field_name == "rejected_components" else "", component)
        if identity in names:
            raise ValueError(f"Duplicate component provenance in {field_name}: {component}")
        names.add(identity)
        if "derived_from_component" in record:
            raise ValueError(f"{field_name} uses unsupported legacy derived provenance.")

        variant_payload = record.get("tighter_role_variant")
        if variant_payload is not None and not allow_tighter_role_variant:
            raise ValueError(
                f"{field_name} does not permit tighter_role_variant; use an exact source component name."
            )
        if variant_payload is None:
            if component not in components_by_mode.get(mode, {}):
                raise ValueError(f"{field_name} component does not belong to source mode {mode}.")
        else:
            try:
                variant = TighterRoleVariant.from_payload(variant_payload)
            except ValueError as error:
                raise ValueError(f"Invalid tighter_role_variant in {field_name}: {error}") from error
            source_description = components_by_mode.get(mode, {}).get(variant.source_component_name)
            if (
                variant.source_mode != mode
                or variant.narrowed_name != component
                or source_description is None
                or variant.source_component_description != source_description
                or tuple(evidence) != variant.evidence_ids
            ):
                raise ValueError(f"{field_name} tighter_role_variant does not match its source record.")

        available = evidence_by_mode.get(mode, set())
        if not set(evidence).issubset(available):
            raise ValueError(f"{field_name} evidence does not belong to source mode {mode}.")


def _validate_variant_components(
    idea: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> None:
    descriptions = _idea_component_descriptions(idea)
    for record in records:
        payload = record.get("tighter_role_variant")
        if payload is None:
            continue
        variant = TighterRoleVariant.from_payload(payload)
        if descriptions.get(variant.narrowed_name) != variant.narrowed_description:
            raise ValueError(
                "Fused idea tighter-role component does not match its narrowed description."
            )


def _idea_component_descriptions(idea: Mapping[str, Any]) -> dict[str, str]:
    components = idea.get("components")
    if not isinstance(components, Sequence) or isinstance(components, (str, bytes)):
        return {}
    explanations = idea.get("component_explanations")
    explanations = explanations if isinstance(explanations, Mapping) else {}
    result: dict[str, str] = {}
    for component in components:
        if isinstance(component, Mapping):
            name = str(component.get("component") or component.get("name") or "").strip()
            description = str(component.get("description") or "").strip()
        else:
            name = str(component).strip()
            description = str(explanations.get(name) or "").strip()
        if name:
            result[name] = description
    return result


def _validate_conflict_provenance(
    conflicts: Sequence[Mapping[str, Any]],
    mode_inputs: Sequence[Mapping[str, Any]],
) -> None:
    evidence_by_mode = {str(item["mode"]): _source_evidence(item) for item in mode_inputs}
    for conflict in conflicts:
        resolution = str(conflict.get("resolution") or "").strip()
        source_modes = conflict.get("source_modes")
        evidence = _evidence_values(conflict)
        if (
            not resolution
            or not isinstance(source_modes, Sequence)
            or isinstance(source_modes, (str, bytes))
            or not source_modes
            or any(str(mode) not in CANONICAL_MODES for mode in source_modes)
            or not evidence
        ):
            raise ValueError("Each conflict resolution needs resolution, canonical source_modes, and evidence.")
        available = set().union(*(evidence_by_mode.get(str(mode), set()) for mode in source_modes))
        if not set(evidence).issubset(available):
            raise ValueError("Conflict evidence does not belong to its source modes.")


def _source_components(mode_input: Mapping[str, Any]) -> dict[str, str]:
    idea = mode_input.get("idea")
    if not isinstance(idea, Mapping):
        return {}
    components = idea.get("components")
    if not isinstance(components, Sequence) or isinstance(components, (str, bytes)):
        return {}
    explanations = idea.get("component_explanations")
    explanations = explanations if isinstance(explanations, Mapping) else {}
    result: dict[str, str] = {}
    for component in components:
        if isinstance(component, Mapping):
            name = str(component.get("component") or component.get("name") or "").strip()
            description = str(component.get("description") or "").strip()
        else:
            name = str(component).strip()
            description = str(explanations.get(name) or "").strip()
        if name:
            result[name] = description
    return result


def _source_component(
    mode_inputs: Sequence[Mapping[str, Any]], mode: str, name: str
) -> Any:
    for mode_input in mode_inputs:
        if str(mode_input.get("mode")) != mode:
            continue
        idea = mode_input.get("idea")
        if not isinstance(idea, Mapping):
            break
        components = idea.get("components")
        if not isinstance(components, Sequence) or isinstance(components, (str, bytes)):
            break
        for component in components:
            component_name = _component_names({"components": [component]})
            if component_name == [name]:
                return deepcopy(component)
        break
    raise ValueError(f"Replacement component does not belong to source mode {mode}.")


def _source_evidence(mode_input: Mapping[str, Any]) -> set[str]:
    evidence = mode_input.get("evidence", mode_input.get("source_evidence", []))
    if not evidence and isinstance(mode_input.get("idea"), Mapping):
        idea = mode_input["idea"]
        evidence = idea.get("evidence", idea.get("source_evidence", []))
    return set(_as_string_list(evidence))


def _evidence_values(record: Mapping[str, Any]) -> list[str]:
    return _as_string_list(record.get("evidence", record.get("source_evidence", [])))


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)):
        value = [value]
    if not isinstance(value, Sequence):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _provenance_component(record: Mapping[str, Any]) -> str:
    return str(record.get("component") or record.get("name") or "").strip()


def _component_names(idea: Mapping[str, Any]) -> list[str]:
    components = idea.get("components")
    if not isinstance(components, Sequence) or isinstance(components, (str, bytes)):
        return []
    names = []
    for component in components:
        if isinstance(component, Mapping):
            name = str(component.get("component") or component.get("name") or "").strip()
        else:
            name = str(component).strip()
        if name:
            names.append(name)
    return names


def _evaluate(evaluator: RefereeEvaluator, candidate: Mapping[str, Any]) -> RefereeEvaluation:
    raw = evaluator.evaluate(deepcopy(dict(candidate)))
    if not isinstance(raw, Mapping):
        raise TypeError("Referee evaluator must return a mapping.")
    nested = raw.get("metrics")
    metric_source = nested if isinstance(nested, Mapping) else raw
    metrics: dict[str, float] = {}
    missing = []
    for metric in REFEREE_METRICS:
        value = metric_source.get(metric)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or not 0 <= float(value) <= 5
        ):
            missing.append(metric)
        else:
            metrics[metric] = float(value)
    if missing:
        raise ValueError("Referee evaluation requires ten finite metrics; invalid: " + ", ".join(missing))
    provider_score = raw.get("score", raw.get("composite"))
    if (
        provider_score is not None
        and (
            isinstance(provider_score, bool)
            or not isinstance(provider_score, (int, float))
            or not isfinite(float(provider_score))
        )
    ):
        raise ValueError("Referee provider score must be finite when present.")
    score = sum(
        FUSION_METRIC_WEIGHTS[metric]
        * (5.0 - value if metric in _PENALTY_METRICS else value)
        for metric, value in metrics.items()
    )
    details = {
        key: deepcopy(value)
        for key, value in raw.items()
        if key not in {"score", "composite", "metrics", *REFEREE_METRICS}
    }
    details["fusion_evaluation_profile"] = FUSION_EVALUATION_PROFILE
    if provider_score is not None:
        details["provider_score"] = float(provider_score)
    return RefereeEvaluation(score, metrics, details)


def _apply_repair(
    best_idea: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    proposal: Mapping[str, Any],
    *,
    minimum_components: int,
    refinement_boundary: RefinementBoundary | None,
    protected_components: Sequence[str],
    mode_inputs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(proposal, Mapping):
        raise TypeError("Repair proposal must be a mapping.")
    raw_operations = proposal.get("operations", proposal.get("component_edits"))
    operations = _mapping_list(raw_operations, "repair operations")
    if not operations:
        raise ValueError("Repair proposal has no operations.")

    candidate = deepcopy(dict(best_idea))
    candidate_selected = deepcopy(list(selected))
    raw_components = candidate.get("components")
    if not isinstance(raw_components, list):
        raise ValueError("Fused idea components must be a list.")
    components = _component_names(candidate)
    original_count = len(components)
    allowed_components = (
        set(refinement_boundary.allowed_component_ids)
        if refinement_boundary is not None and refinement_boundary.allowed_component_ids
        else None
    )
    allowed_operations = set(_allowed_repair_operations(refinement_boundary))
    protected = {str(item) for item in protected_components}
    normalized_operations: list[dict[str, Any]] = []

    for raw in operations:
        op = str(raw.get("op") or raw.get("operation") or "").strip().lower()
        op = {
            "remove_component": "remove",
            "replace_component": "replace",
        }.get(op, op)
        if op not in allowed_operations:
            raise ValueError(f"Prohibited repair operation: {op or '<missing>'}")
        component = str(raw.get("component") or raw.get("source") or "").strip()
        target = str(raw.get("target") or raw.get("replacement") or "").strip()
        touched = {component}
        if op == "rewire":
            touched.add(target)
        if not component or component not in components:
            raise ValueError(f"Repair component is not present: {component or '<missing>'}")
        if allowed_components is not None and not touched.issubset(allowed_components):
            raise ValueError("Repair operation is outside refinement_boundary.")
        if touched & protected:
            raise ValueError("Repair operation touches a protected component.")

        if op == "remove":
            index = components.index(component)
            components.pop(index)
            raw_components.pop(index)
            candidate_selected = [
                item for item in candidate_selected if _provenance_component(item) != component
            ]
        elif op == "replace":
            if not target:
                raise ValueError("Replace operation requires a target.")
            if target != component and target in components:
                raise ValueError("Replace target already exists.")
            replacement_provenance = {
                "component": target,
                "source_mode": str(raw.get("source_mode") or "").strip(),
                "evidence": _evidence_values(raw),
            }
            variant_payload = raw.get("tighter_role_variant")
            variant: TighterRoleVariant | None = None
            if variant_payload is not None:
                replacement_provenance["tighter_role_variant"] = deepcopy(variant_payload)
            _validate_component_provenance(
                [replacement_provenance],
                mode_inputs,
                "replacement",
                allow_tighter_role_variant=True,
            )
            if variant_payload is None:
                replacement = _source_component(
                    mode_inputs, replacement_provenance["source_mode"], target
                )
            else:
                variant = TighterRoleVariant.from_payload(variant_payload)
                replacement = {
                    "name": variant.narrowed_name,
                    "description": variant.narrowed_description,
                }
            index = components.index(component)
            components[index] = target
            raw_components[index] = replacement
            if allowed_components is not None:
                allowed_components.add(target)
            for selected_index, item in enumerate(candidate_selected):
                if _provenance_component(item) == component:
                    candidate_selected[selected_index] = replacement_provenance
                    break
            explanations = candidate.get("component_explanations")
            if isinstance(explanations, Mapping):
                updated = deepcopy(dict(explanations))
                old_explanation = updated.pop(component, "")
                updated[target] = (
                    variant.narrowed_description
                    if variant is not None
                    else raw.get("explanation", old_explanation)
                )
                candidate["component_explanations"] = updated
        else:
            if not target or target not in components or target == component:
                raise ValueError("Rewire requires a distinct existing target component.")
            rewires = candidate.get("rewires")
            rewires = deepcopy(list(rewires)) if isinstance(rewires, list) else []
            rewires.append(
                {
                    "source": component,
                    "target": target,
                    "details": str(raw.get("details") or "").strip(),
                }
            )
            candidate["rewires"] = rewires
        normalized_operations.append({**deepcopy(raw), "op": op, "component": component, "target": target})

    if len(components) < minimum_components:
        raise ValueError("Repair would violate minimum_components.")
    if len(components) > original_count:
        raise ValueError("Repair cannot increase component count.")
    candidate["components"] = raw_components
    return candidate, candidate_selected, normalized_operations


def _allowed_repair_operations(
    boundary: RefinementBoundary | None,
) -> list[str]:
    if boundary is None:
        return sorted(ALLOWED_REPAIR_OPERATIONS)
    if boundary.allowed_fields and "components" not in boundary.allowed_fields:
        return []
    if not boundary.allowed_edit_kinds:
        return sorted(ALLOWED_REPAIR_OPERATIONS)
    edit_kind_operations = {
        "REMOVE_COMPONENT": "remove",
        "REPLACE_COMPONENT": "replace",
        "REWIRE": "rewire",
    }
    return sorted(
        operation
        for edit_kind, operation in edit_kind_operations.items()
        if edit_kind in boundary.allowed_edit_kinds
    )


def _repair_event(
    step: int,
    score: float,
    status: str,
    reason: str,
    *,
    proposal: Any = None,
) -> dict[str, Any]:
    event = {
        "step": step,
        "phase": "repair",
        "status": status,
        "reason": reason,
        "score_before": score,
        "score_after": score,
    }
    if proposal is not None:
        event["proposal"] = deepcopy(proposal)
    return event
