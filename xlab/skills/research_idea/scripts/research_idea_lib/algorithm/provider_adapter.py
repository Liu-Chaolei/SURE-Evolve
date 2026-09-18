"""Adapters from the package Provider contract to native search and fusion protocols."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, overload

from research_idea_lib.providers import (
    JsonValue,
    Provider,
    ProviderError,
    ProviderRequest,
    ProviderResult,
    ProviderTrace,
)

from .contracts import (
    ALGORITHM_ID,
    EvaluationResponse,
    GenerationRequest,
    GenerationResponse,
    IdeaComponent,
    IdeaProviderContext,
    IdeaState,
    OperatorGroundingKind,
    ProviderUsage,
    PromptMode,
    TighterRoleVariant,
    validate_prompt_mode,
)
from .operator_grounding import OperatorQuery

from .defects import DEFECTS, validate_defects
from .evaluation import METRICS
from .fusion import ALLOWED_REPAIR_OPERATIONS, CANONICAL_MODES, REFEREE_METRICS, validate_fusion_output
from .operators import EditKind, EditPlan
from .response_projection import unwrap_answer_object
from .root_identity import IDENTITY_FIELDS, validate_root_identity
from .prompts.idea_fusion import (
    FUSION_REPAIR_PROMPT,
    IDEA_FUSION_PROMPT,
)
from .prompts.mechanism_commit_query import get_mechanism_commit_query_prompt
from .prompts.mcts_evaluation import get_mcts_evaluation_prompt
from .prompts.skill_instantiation import get_skill_instantiation_prompt
from .prompts.theory_transfer_query import get_theory_transfer_query_prompt
from .tastes import IDEA_TASTE_MODES

IDEA_DIAGNOSTIC_OPERATION = "xlab.research_idea.idea.diagnostic.v1"
IDEA_EVALUATE_OPERATION = "xlab.research_idea.idea.evaluate.v1"
IDEA_GENERATE_OPERATION = "xlab.research_idea.idea.generate.v1"
THEORY_TRANSFER_QUERY_OPERATION = "xlab.research_idea.operator.theory-transfer-query.v1"
MECHANISM_COMMIT_QUERY_OPERATION = "xlab.research_idea.operator.mechanism-commit-query.v1"
FUSION_GENERATE_OPERATION = "xlab.research_idea.fusion.generate.v1"
FUSION_REFEREE_OPERATION = "xlab.research_idea.fusion.referee.v1"
FUSION_REPAIR_OPERATION = "xlab.research_idea.fusion.repair.v1"
MAX_REPAIR_OPERATIONS = 6


class AdapterOutputError(ValueError):
    """Raised when a provider result does not satisfy a native contract."""

    def __init__(self, message: str, *, issues: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.validation_issues = deepcopy(issues or [])
        self.previous_draft: dict[str, JsonValue] | None = None


@dataclass(frozen=True)
class AdapterUsage:
    provider_calls: int = 0
    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ProviderAdapter:
    """One provider-backed object implementing native search and fusion protocols.

    The overloads intentionally share ``generate`` and ``evaluate`` because the
    native protocols use those names with distinguishable typed inputs.
    """

    def __init__(self, provider: Provider, *, model: str, temperature: float | None = None) -> None:
        if not model.strip():
            raise ValueError("model is required")
        self._provider = provider
        self._model = model.strip()
        self._temperature = temperature
        self._usage = AdapterUsage()
        self._traces: list[ProviderTrace] = []

    @property
    def usage(self) -> AdapterUsage:
        return self._usage

    @property
    def traces(self) -> tuple[ProviderTrace, ...]:
        return tuple(self._traces)

    @overload
    def evaluate(
        self,
        value: IdeaState,
        *,
        idea_taste_mode: str,
        diagnostic: bool,
        context: IdeaProviderContext = IdeaProviderContext(),
        prompt_mode: PromptMode = "conceptual_surprise",
    ) -> EvaluationResponse: ...

    @overload
    def evaluate(self, value: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def evaluate(
        self,
        value: IdeaState | Mapping[str, Any],
        *,
        idea_taste_mode: str | None = None,
        diagnostic: bool | None = None,
        context: IdeaProviderContext | None = None,
        prompt_mode: PromptMode = "conceptual_surprise",
    ) -> EvaluationResponse | Mapping[str, Any]:
        if isinstance(value, IdeaState):
            if idea_taste_mode is None or diagnostic is None:
                raise TypeError("idea evaluation requires idea_taste_mode and diagnostic")
            return self._evaluate_idea(
                value,
                idea_taste_mode=idea_taste_mode,
                prompt_mode=prompt_mode,
                diagnostic=diagnostic,
                context=context or IdeaProviderContext(),
            )
        if idea_taste_mode is not None or diagnostic is not None or context is not None:
            raise TypeError("referee evaluation does not accept idea_taste_mode or diagnostic")
        return self._evaluate_fusion(value)

    @overload
    def generate(self, request: GenerationRequest) -> GenerationResponse: ...

    @overload
    def generate(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def generate(
        self, request: GenerationRequest | Mapping[str, Any]
    ) -> GenerationResponse | Mapping[str, Any]:
        if isinstance(request, GenerationRequest):
            return self._generate_idea(request)
        return self._generate_fusion(request)

    def theory_transfer_query(
        self, state: IdeaState, *, prompt_mode: PromptMode = "default"
    ) -> OperatorQuery:
        return self._operator_query(state, prompt_mode, kind="theory_transfer")

    def mechanism_commit_query(
        self, state: IdeaState, *, prompt_mode: PromptMode = "default"
    ) -> OperatorQuery:
        return self._operator_query(state, prompt_mode, kind="mechanism_commit")

    def _operator_query(
        self,
        state: IdeaState,
        prompt_mode: PromptMode,
        *,
        kind: OperatorGroundingKind,
    ) -> OperatorQuery:
        prompt_mode = validate_prompt_mode(prompt_mode)
        structured: dict[str, JsonValue] = {
            "algorithm": ALGORITHM_ID,
            "prompt_mode": prompt_mode,
            "idea": state.to_payload(),
        }
        if kind == "theory_transfer":
            operation = THEORY_TRANSFER_QUERY_OPERATION
            prompt = get_theory_transfer_query_prompt(prompt_mode)
            needed_field = "needed_content"
        else:
            operation = MECHANISM_COMMIT_QUERY_OPERATION
            prompt = get_mechanism_commit_query_prompt(prompt_mode)
            needed_field = "mechanism_gap"
        prompt = prompt.format(
            topic=state.title, root_domains=json.dumps(list(state.root_domains)),
            refinement_scope="See the structured user input for the current idea.",
            idea="See user input: idea.",
            edit_plan="Retrieve grounding for the named operator; no plan was supplied to this query.",
        )
        def validate_response(result: ProviderResult):
            payload = _result_object(result)
            _reject_placeholders(payload)
            expected_fields = {"query", needed_field, "expected_role"}
            if set(payload) != expected_fields:
                raise AdapterOutputError(
                    f"{kind} query output requires exactly: {', '.join(sorted(expected_fields))}"
                )
            query = _required_text(payload.get("query"), "query")
            needed_content = _required_text(payload.get(needed_field), needed_field)
            expected_role = _required_text(payload.get("expected_role"), "expected_role")
            return OperatorQuery(
                kind=kind,
                query=query,
                needed_content=needed_content,
                expected_role=expected_role,
                provenance_json=json.dumps(
                    {
                        "operation": operation,
                        "provider_trace": result.trace.to_dict(),
                        "structured_input_digest": result.trace.input_digest,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )

        result = self._complete(operation, structured, prompt, response_validator=validate_response)
        return validate_response(result)

    def propose_repair(self, request: Mapping[str, Any]) -> Mapping[str, Any] | None:
        structured = _json_mapping(request, "repair request")
        prompt = _format_repair_prompt(structured)
        def validate_response(result: ProviderResult):
            payload = _result_object(result)
            _reject_placeholders(payload)

            stop = payload.get("stop", False)
            if not isinstance(stop, bool):
                raise AdapterOutputError("repair stop must be a boolean")
            if stop:
                return None

            local_shape = "component_edits" in payload
            raw_operations = payload.get("component_edits") if local_shape else payload.get("operations")
            operations = _mapping_sequence(raw_operations, "repair operations")
            if not 1 <= len(operations) <= MAX_REPAIR_OPERATIONS:
                raise AdapterOutputError(
                    f"repair output requires 1..{MAX_REPAIR_OPERATIONS} operations"
                )
            allowed = {
                str(item).strip().lower()
                for item in structured.get("allowed_operations", sorted(ALLOWED_REPAIR_OPERATIONS))
            }
            normalized = [
                _normalize_repair_operation(item, local_shape=local_shape, allowed=allowed)
                for item in operations
            ]
            _validate_replacement_sources(normalized, structured.get("mode_inputs"))
            return {"operations": normalized}

        result = self._complete(FUSION_REPAIR_OPERATION, structured, prompt, response_validator=validate_response)
        return validate_response(result)

    def _evaluate_idea(
        self,
        state: IdeaState,
        *,
        idea_taste_mode: str,
        prompt_mode: PromptMode,
        diagnostic: bool,
        context: IdeaProviderContext,
    ) -> EvaluationResponse:
        if idea_taste_mode not in IDEA_TASTE_MODES:
            raise ValueError(f"unknown idea taste: {idea_taste_mode}")
        if not isinstance(diagnostic, bool):
            raise TypeError("diagnostic must be a boolean")
        prompt_mode = validate_prompt_mode(prompt_mode)
        structured: dict[str, JsonValue] = {
            "algorithm": ALGORITHM_ID,
            "idea_taste_mode": idea_taste_mode,
            "prompt_mode": prompt_mode,
            "diagnostic": diagnostic,
            "idea": state.to_payload(),
            "grounding": context.to_payload(),
        }
        prompt = get_mcts_evaluation_prompt(prompt_mode).format(
            topic=state.title,
            root_domains=json.dumps(list(state.root_domains), ensure_ascii=False),
            mature_idea=json.dumps(
                context.mature_idea.to_payload() if context.mature_idea else None,
                ensure_ascii=False,
            ),
            refinement_scope=json.dumps(list(context.refinement_scope), ensure_ascii=False),
            edit_plan="None",
            idea=json.dumps(state.to_payload(), ensure_ascii=False),
            defect_registry=(
                "Canonical defect tags (an empty list is valid when none applies): "
                + json.dumps(sorted(DEFECTS)) + "; ranked evidence: "
                + json.dumps([item.to_payload() for item in context.evidence], ensure_ascii=False)
            ),
            symbolic_memory_hints=json.dumps(list(context.memory_hints), ensure_ascii=False),
        )
        operation = IDEA_DIAGNOSTIC_OPERATION if diagnostic else IDEA_EVALUATE_OPERATION
        def validate_response(result: ProviderResult):
            payload = _result_object(result)
            _reject_placeholders(payload)
            metrics = _metrics(payload, METRICS, integers_only=True)
            confidence = _finite_number(payload.get("confidence"), "confidence", minimum=0, maximum=1)
            defects = _optional_string_sequence(payload.get("detected_defects"), "detected_defects")
            try:
                validated_defects = validate_defects(tuple(defects))
            except ValueError as error:
                raise AdapterOutputError(str(error)) from error
            feedback = _required_text(payload.get("feedback"), "feedback")
            return EvaluationResponse(
                metrics=metrics,
                confidence=confidence,
                detected_defects=validated_defects,
                feedback=feedback,
                usage=_algorithm_usage(result, evaluator=True),
            )

        result = self._complete(operation, structured, prompt, response_validator=validate_response)
        return validate_response(result)

    def _generate_idea(self, request: GenerationRequest) -> GenerationResponse:
        if request.mode not in IDEA_TASTE_MODES:
            raise ValueError(f"unknown idea taste: {request.mode}")
        if not isinstance(request.parent, IdeaState):
            raise TypeError("generation parent must be an IdeaState")
        if not isinstance(request.plan, EditPlan):
            raise TypeError("generation plan must be an EditPlan")
        if isinstance(request.seed, bool) or not isinstance(request.seed, int) or request.seed < 0:
            raise ValueError("generation seed must be a non-negative integer")
        if not isinstance(request.memory_hints, tuple) or any(
            not isinstance(item, str) for item in request.memory_hints
        ):
            raise TypeError("memory_hints must be a tuple of strings")
        if not isinstance(request.context, IdeaProviderContext):
            raise TypeError("generation context must be an IdeaProviderContext")
        context = IdeaProviderContext(
            evidence=request.context.evidence,
            mature_idea=request.context.mature_idea,
            refinement_scope=request.context.refinement_scope,
            refinement_boundary=request.context.refinement_boundary,
            memory_hints=request.memory_hints,
            task_context_json=request.context.task_context_json,
            research_policy_json=request.context.research_policy_json,
        )

        structured: dict[str, JsonValue] = {
            "algorithm": ALGORITHM_ID,
            "idea_taste_mode": request.idea_taste_mode,
            "prompt_mode": request.prompt_mode,
            "parent": request.parent.to_payload(),
            "plan": _plan_payload(request.plan),
            "seed": request.seed,
            "memory_hints": list(request.memory_hints),
            "grounding": context.to_payload(),
            "operator_grounding": request.grounding.to_payload() if request.grounding else None,
        }
        prompt = get_skill_instantiation_prompt(request.prompt_mode).format(
            topic=request.parent.title,
            root_domains=json.dumps(list(request.parent.root_domains), ensure_ascii=False),
            refinement_scope=json.dumps(list(context.refinement_scope), ensure_ascii=False),
            taste_guidance=request.idea_taste_mode,
            mature_idea='See user input: grounding.mature_idea (or parent if absent).',
            parent_summary='See user input: parent.',
            parent_components=json.dumps(
                [component.name for component in request.parent.components], ensure_ascii=False
            ),
            paper_context='See user input: grounding.evidence.',
            memory_bundle='See user input: memory_hints.',
            skill_references="package-local algorithm resources",
            additional_retrieval_context='See user input: operator_grounding and grounding.',
            skill_name=request.plan.operator,
            plan_objective=request.plan.rationale,
            target_defects=json.dumps(list(request.plan.target_defects), ensure_ascii=False),
            component_edits='See user input: plan.',
            validation_protocols="encoded in the typed edit plan",
            guardrails="preserve typed component bounds",
        )
        prompt += ("\nExisting replacement/removal targets MUST be copied byte-for-byte from parent_components. "
                   "Do not append a file path, role, version, or parenthetical explanation to a name. "
                   "Descriptions belong only in component_role_explanations. "
                   "If validation feedback is provided, repair the listed fields without inventing sources.\n")
        def validate_response(result: ProviderResult):
            payload = _result_object(result)
            _reject_placeholders(payload)
            state = _idea_state_from_generation(payload, request.parent, request.plan)
            _validate_generated_child(
                payload,
                parent=request.parent,
                child=state,
                plan=request.plan,
                context=request.context,
            )
            return GenerationResponse(state=state, usage=_algorithm_usage(result, generation=True))

        result = self._complete(IDEA_GENERATE_OPERATION, structured, prompt, response_validator=validate_response)
        return validate_response(result)

    def _generate_fusion(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        structured = _json_mapping(request, "fusion request")
        prompt = _format_fusion_prompt(structured)
        def validate_response(result: ProviderResult):
            payload = _result_object(result)
            try:
                _reject_placeholders(payload)
                normalized = _normalize_fusion(payload)
                _validate_fusion_sources(normalized, structured.get("mode_inputs"))
                validate_fusion_output(normalized, structured["mode_inputs"],
                                       minimum_components=structured.get("minimum_components", 1))
                mature = _fusion_mature_idea(structured)
                if mature is not None:
                    try:
                        validate_root_identity(mature, normalized["idea"], context="idea_fusion.idea")
                    except ValueError as error:
                        raise AdapterOutputError(str(error), issues=[{
                            "path": "fused_idea",
                            "message": str(error),
                            "received": {key: normalized["idea"].get(key) for key in IDENTITY_FIELDS},
                            "required_identity": {key: deepcopy(mature[key]) for key in IDENTITY_FIELDS if key in mature},
                        }]) from error
            except ValueError as error:
                if not isinstance(error, AdapterOutputError):
                    error = AdapterOutputError(str(error))
                error.previous_draft = deepcopy(payload)
                raise error
            return normalized

        result = self._complete(FUSION_GENERATE_OPERATION, structured, prompt, response_validator=validate_response)
        return validate_response(result)

    def _evaluate_fusion(self, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        structured = {
            "algorithm": ALGORITHM_ID,
            "candidate": _json_mapping(candidate, "referee candidate"),
        }
        prompt = get_mcts_evaluation_prompt().format(
            topic=str(candidate.get("title") or "fused idea"),
            root_domains=json.dumps(candidate.get("root_domains", []), ensure_ascii=False),
            mature_idea="None",
            refinement_scope="None",
            edit_plan="fusion referee evaluation",
            idea=json.dumps(structured["candidate"], ensure_ascii=False),
            defect_registry=json.dumps(sorted(DEFECTS)),
            symbolic_memory_hints="[]",
        )
        def validate_response(result: ProviderResult):
            payload = _result_object(result)
            _reject_placeholders(payload)
            metrics = _metrics(payload, REFEREE_METRICS)
            score_value = payload.get("score", payload.get("composite"))
            score = (
                _finite_number(score_value, "score", minimum=0, maximum=5)
                if score_value is not None
                else None
            )
            details = {
                key: deepcopy(value)
                for key, value in payload.items()
                if key not in {"score", "composite", "metrics", *REFEREE_METRICS}
            }
            result_payload: dict[str, Any] = {"metrics": metrics, **details}
            if score is not None:
                result_payload["score"] = score
            return result_payload

        result = self._complete(FUSION_REFEREE_OPERATION, structured, prompt, response_validator=validate_response)
        return validate_response(result)

    def _complete(
        self, operation: str, structured_input: Mapping[str, JsonValue], system_prompt: str,
        *, response_validator=None,
    ) -> ProviderResult:
        request = ProviderRequest(
            operation=operation,
            model=self._model,
            structured_input=dict(structured_input),
            system_prompt=system_prompt,
            user_prompt=json.dumps(
                structured_input, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
            output_kind="json",
            temperature=self._temperature,
            validation_profile="xlab.adapter.response.v1:" + operation if response_validator is not None else None,
            response_validator=response_validator,
        )
        try:
            result = self._provider.complete(request)
        except ProviderError as error:
            self._record_trace(error.trace)
            raise
        self._record_result(result)
        return result

    def _record_trace(self, trace: ProviderTrace) -> None:
        self._traces.append(trace)
        self._usage = AdapterUsage(
            provider_calls=self._usage.provider_calls + 1,
            attempts=self._usage.attempts + trace.attempts,
            input_tokens=self._usage.input_tokens,
            output_tokens=self._usage.output_tokens,
            total_tokens=self._usage.total_tokens,
        )

    def _record_result(self, result: ProviderResult) -> None:
        self._traces.append(result.trace)
        self._usage = AdapterUsage(
            provider_calls=self._usage.provider_calls + 1,
            attempts=self._usage.attempts + result.trace.attempts,
            input_tokens=self._usage.input_tokens + result.usage.input_tokens,
            output_tokens=self._usage.output_tokens + result.usage.output_tokens,
            total_tokens=self._usage.total_tokens + result.usage.total_tokens,
        )


ProviderResearchIdeaAdapter = ProviderAdapter
ProviderFusionAdapter = ProviderAdapter


def _result_object(result: ProviderResult) -> dict[str, JsonValue]:
    if result.json_value is None or not isinstance(result.json_value, dict):
        raise AdapterOutputError("provider operation requires a JSON object result")
    payload, _ = unwrap_answer_object(result.json_value)
    return payload


def _algorithm_usage(
    result: ProviderResult, *, evaluator: bool = False, generation: bool = False
) -> ProviderUsage:
    return ProviderUsage(
        evaluator_calls=1 if evaluator else 0,
        generation_calls=1 if generation else 0,
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
    )


def _json_mapping(value: Mapping[str, Any], field: str) -> dict[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    try:
        serialized = json.dumps(value, ensure_ascii=False, allow_nan=False)
        decoded = json.loads(serialized)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field} must contain only JSON values") from error
    if not isinstance(decoded, dict):
        raise TypeError(f"{field} must be a mapping")
    return decoded


def _plan_payload(plan: EditPlan) -> dict[str, JsonValue]:
    return {
        "operator": plan.operator,
        "target_defects": list(plan.target_defects),
        "edits": [
            {
                "kind": edit.kind.value,
                "target": edit.target,
                "replacement": edit.replacement,
            }
            for edit in plan.edits
        ],
        "rationale": plan.rationale,
        "approved": plan.approved,
        "memory_refs": list(plan.memory_refs),
    }


def _idea_state_from_generation(
    payload: Mapping[str, Any], parent: IdeaState, plan: EditPlan
) -> IdeaState:
    nested = payload.get("state", payload.get("idea"))
    source = nested if isinstance(nested, Mapping) else payload
    title = _required_text(source.get("title"), "title")
    abstract = _required_text(source.get("abstract"), "abstract")
    core = _required_text(source.get("core_contribution"), "core_contribution")
    method = _required_text(source.get("method"), "method")
    risks = _required_text(source.get("risks"), "risks")

    if "components" in source:
        components = _components(source.get("components"))
    else:
        components = _components_from_plan(source, parent, plan)
    tags = _optional_string_sequence(source.get("tags", list(parent.tags)), "tags")
    domains = _optional_string_sequence(
        source.get("root_domains", list(parent.root_domains)), "root_domains"
    )
    try:
        return IdeaState(
            title=title,
            abstract=abstract,
            core_contribution=core,
            method=method,
            risks=risks,
            components=tuple(components),
            tags=tuple(tags),
            root_domains=tuple(domains),
        )
    except ValueError as error:
        raise AdapterOutputError(str(error)) from error


def _components(value: Any) -> list[IdeaComponent]:
    records = _mapping_sequence(value, "components")
    components: list[IdeaComponent] = []
    names: set[str] = set()
    for record in records:
        name = _required_text(record.get("name", record.get("component")), "component name")
        description = _required_text(record.get("description"), "component description")
        if name in names:
            raise AdapterOutputError(f"duplicate component: {name}")
        names.add(name)
        try:
            components.append(IdeaComponent(name, description))
        except ValueError as error:
            raise AdapterOutputError(str(error)) from error
    return components


def _validate_generated_child(
    payload: Mapping[str, Any],
    *,
    parent: IdeaState,
    child: IdeaState,
    plan: EditPlan,
    context: IdeaProviderContext,
) -> None:
    nested = payload.get("state", payload.get("idea"))
    source = nested if isinstance(nested, Mapping) else payload
    mapping = source.get("component_mapping", payload.get("component_mapping"))
    roles = source.get("component_role_explanations", payload.get("component_role_explanations"))
    component_edits = {
        EditKind.ADD_COMPONENT,
        EditKind.REMOVE_COMPONENT,
        EditKind.REPLACE_COMPONENT,
    }
    if any(edit.kind in component_edits for edit in plan.edits):
        if not isinstance(mapping, Mapping) or not isinstance(roles, Mapping):
            raise AdapterOutputError(
                "generation output requires component_mapping and component_role_explanations "
                "to validate the exact approved EditPlan"
            )
        expected = _components_from_plan(
            {"component_mapping": mapping, "component_role_explanations": roles},
            parent,
            plan,
        )
        if [(item.name, item.description) for item in child.components] != [
            (item.name, item.description) for item in expected
        ]:
            raise AdapterOutputError("generated child component diff does not match the exact approved EditPlan")
    elif child.components != parent.components:
        raise AdapterOutputError("generated child changed components without an approved component edit")

    if context.mature_idea is not None:
        if child.root_domains != parent.root_domains or child.tags != parent.tags:
            raise AdapterOutputError("mature refinement cannot change root domains or tags")
        _validate_refinement_boundary(parent, child, plan, context)


def _validate_refinement_boundary(
    parent: IdeaState,
    child: IdeaState,
    plan: EditPlan,
    context: IdeaProviderContext,
) -> None:
    boundary = context.refinement_boundary
    if boundary is None:
        return
    edit_kinds = {edit.kind.value for edit in plan.edits}
    if boundary.allowed_edit_kinds and not edit_kinds.issubset(boundary.allowed_edit_kinds):
        raise AdapterOutputError("approved EditPlan uses edit kinds outside refinement boundary")

    text_fields = ("title", "abstract", "core_contribution", "method", "risks")
    changed_fields = {
        field for field in text_fields if getattr(parent, field) != getattr(child, field)
    }
    if child.components != parent.components:
        changed_fields.add("components")
    if boundary.allowed_fields and not changed_fields.issubset(boundary.allowed_fields):
        raise AdapterOutputError("generated child changed fields outside refinement boundary")

    if boundary.allowed_component_ids:
        allowed = set(boundary.allowed_component_ids)
        parent_by_name = {item.name: item.description for item in parent.components}
        child_by_name = {item.name: item.description for item in child.components}
        changed_existing = {
            name
            for name, description in parent_by_name.items()
            if child_by_name.get(name) != description
        }
        if not changed_existing.issubset(allowed):
            raise AdapterOutputError(
                "generated child changed mature components outside refinement boundary: "
                + ", ".join(sorted(changed_existing - allowed))
            )


def _components_from_plan(
    source: Mapping[str, Any], parent: IdeaState, plan: EditPlan
) -> list[IdeaComponent]:
    mapping = source.get("component_mapping")
    roles = source.get("component_role_explanations")
    if not isinstance(mapping, Mapping) or not isinstance(roles, Mapping):
        raise AdapterOutputError(
            "generation output requires typed components or component mappings and explanations"
        )
    values = {str(key): _required_text(value, f"component_mapping.{key}") for key, value in mapping.items()}
    explanations = {
        str(key): _required_text(value, f"component_role_explanations.{key}")
        for key, value in roles.items()
    }
    components = {component.name: component for component in parent.components}
    for edit in plan.edits:
        if edit.kind is EditKind.ADD_COMPONENT:
            name = values.get(edit.target)
            if name is None:
                raise AdapterOutputError(f"component_mapping is missing {edit.target}")
            components[name] = IdeaComponent(name, _role(explanations, name))
        elif edit.kind is EditKind.REMOVE_COMPONENT:
            old = values.get(edit.target, edit.target)
            if old not in components:
                raise AdapterOutputError(f"approved removal target is absent: {old}")
            components.pop(old)
        elif edit.kind is EditKind.REPLACE_COMPONENT:
            old = values.get(edit.target, edit.target)
            new = values.get(edit.replacement)
            if old not in components:
                raise AdapterOutputError(f"approved replacement target is absent: {old}")
            if new is None:
                raise AdapterOutputError(f"component_mapping is missing {edit.replacement}")
            components.pop(old)
            components[new] = IdeaComponent(new, _role(explanations, new))
    return list(components.values())


def _role(explanations: Mapping[str, str], name: str) -> str:
    if name not in explanations:
        raise AdapterOutputError(f"component_role_explanations is missing {name}")
    return explanations[name]


def _normalize_fusion(payload: Mapping[str, Any]) -> dict[str, Any]:
    idea = payload.get("idea", payload.get("fused_idea"))
    if not isinstance(idea, Mapping):
        raise AdapterOutputError("fusion output requires an idea mapping")
    required_text = (
        "title",
        "abstract",
        "core_contribution",
        "hypothesis",
        "method",
        "risks",
    )
    normalized_idea = deepcopy(dict(idea))
    for field in required_text:
        normalized_idea[field] = _required_text(idea.get(field), f"fused idea {field}")
    normalized_idea["tags"] = _optional_string_sequence(idea.get("tags"), "fused idea tags")
    normalized_idea["root_domains"] = _optional_string_sequence(
        idea.get("root_domains"), "fused idea root_domains"
    )
    raw_components = idea.get("components")
    if not isinstance(raw_components, list) or not raw_components:
        raise AdapterOutputError("fused idea components must be a non-empty list")
    component_names = []
    for item in raw_components:
        if isinstance(item, Mapping):
            component_names.append(_required_text(item.get("name", item.get("component")), "component"))
        else:
            component_names.append(_required_text(item, "component"))
    if len(component_names) != len(set(component_names)):
        raise AdapterOutputError("fused idea components must be unique")

    metadata = payload.get("fusion_metadata")
    metadata = metadata if isinstance(metadata, Mapping) else payload
    selected = _provenance(metadata.get("selected_components"), "selected_components")
    rejected = _provenance(metadata.get("rejected_components"), "rejected_components")
    conflicts = _conflicts(
        metadata.get("conflict_resolutions", metadata.get("conflicts_and_resolutions"))
    )
    selected_names = {item["component"] for item in selected}
    if selected_names != set(component_names):
        raise AdapterOutputError("selected_components must exactly cover fused components")
    return {
        "idea": normalized_idea,
        "selected_components": selected,
        "rejected_components": rejected,
        "conflict_resolutions": conflicts,
    }


def _provenance(value: Any, field: str) -> list[dict[str, Any]]:
    records = _mapping_sequence(value, field)
    normalized = []
    for record in records:
        component = _required_text(record.get("component", record.get("name")), f"{field}.component")
        mode = _required_text(record.get("source_mode"), f"{field}.source_mode")
        if mode not in CANONICAL_MODES:
            raise AdapterOutputError(f"{field} contains an unknown source mode")
        evidence = _string_sequence(
            record.get("evidence", record.get("source_evidence")), f"{field}.evidence"
        )
        normalized.append({**deepcopy(record), "component": component, "source_mode": mode, "evidence": evidence})
    return normalized


def _conflicts(value: Any) -> list[dict[str, Any]]:
    records = _mapping_sequence(value, "conflict_resolutions")
    normalized = []
    for record in records:
        resolution = _required_text(record.get("resolution"), "conflict resolution")
        source_modes = _string_sequence(record.get("source_modes"), "conflict source_modes")
        if any(mode not in CANONICAL_MODES for mode in source_modes):
            raise AdapterOutputError("conflict resolution contains an unknown source mode")
        evidence = _string_sequence(
            record.get("evidence", record.get("source_evidence")), "conflict evidence"
        )
        normalized.append(
            {**deepcopy(record), "resolution": resolution, "source_modes": source_modes, "evidence": evidence}
        )
    return normalized


def _validate_fusion_sources(
    fusion: Mapping[str, Any], mode_inputs: Any
) -> None:
    sources = _fusion_sources(mode_inputs)
    issues = []
    for field in ("selected_components", "rejected_components"):
        for index, record in enumerate(fusion[field]):
            try:
                _validate_source_record(record, sources, field)
            except AdapterOutputError as error:
                source = sources[record["source_mode"]]
                issues.append({
                    "path": f"{field}[{index}]",
                    "message": str(error),
                    "received": deepcopy(record),
                    "allowed_component_names": list(source["components"]),
                    "allowed_evidence_ids": sorted(source["evidence"]),
                })
    for index, record in enumerate(fusion["conflict_resolutions"]):
        available = set().union(
            *(sources[mode]["evidence"] for mode in record["source_modes"])
        )
        if not set(record["evidence"]).issubset(available):
            issues.append({
                "path": f"conflict_resolutions[{index}].evidence",
                "message": "conflict evidence does not belong to its source modes",
                "received": deepcopy(record["evidence"]),
                "allowed_evidence_ids": sorted(available),
            })
    if issues:
        raise AdapterOutputError(issues[0]["message"], issues=issues)


def _validate_source_record(
    record: dict[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
    field: str,
    *,
    allow_tighter_role_variant: bool = False,
) -> None:
    component = record["component"]
    mode = record["source_mode"]
    variant_payload = record.get("tighter_role_variant")
    if "derived_from_component" in record:
        raise AdapterOutputError(f"{field} uses unsupported legacy derived provenance")
    if variant_payload is not None and not allow_tighter_role_variant:
        raise AdapterOutputError(
            f"{field} does not permit tighter_role_variant; use an exact source component name"
        )
    if variant_payload is None:
        if component not in sources[mode]["components"]:
            raise AdapterOutputError(f"{field} component does not belong to source mode {mode}")
    else:
        try:
            variant = TighterRoleVariant.from_payload(variant_payload)
        except ValueError as error:
            raise AdapterOutputError(f"invalid tighter_role_variant: {error}") from error
        source_description = sources[mode]["components"].get(variant.source_component_name)
        if (
            variant.source_mode != mode
            or variant.narrowed_name != component
            or source_description is None
            or source_description != variant.source_component_description
            or tuple(record["evidence"]) != variant.evidence_ids
        ):
            raise AdapterOutputError(f"{field} tighter_role_variant does not match its source")
    if not set(record["evidence"]).issubset(sources[mode]["evidence"]):
        raise AdapterOutputError(f"{field} evidence does not belong to source mode {mode}")


def _validate_replacement_sources(operations: list[dict[str, Any]], mode_inputs: Any) -> None:
    replacements = [operation for operation in operations if operation["op"] == "replace"]
    if not replacements:
        return
    sources = _fusion_sources(mode_inputs)
    for operation in replacements:
        mode = _required_text(operation.get("source_mode"), "replacement source_mode")
        if mode not in sources:
            raise AdapterOutputError("replacement contains an unknown source mode")
        evidence = _string_sequence(
            operation.get("evidence", operation.get("source_evidence")),
            "replacement evidence",
        )
        replacement = operation["target"]
        operation["source_mode"] = mode
        operation["evidence"] = evidence
        _validate_source_record(
            {
                **operation,
                "component": replacement,
                "source_mode": mode,
                "evidence": evidence,
            },
            sources,
            "replacement",
            allow_tighter_role_variant=True,
        )


def _fusion_sources(mode_inputs: Any) -> dict[str, dict[str, Any]]:
    records = _mapping_sequence(mode_inputs, "mode_inputs")
    sources: dict[str, dict[str, Any]] = {}
    for record in records:
        mode = _required_text(record.get("mode"), "mode_inputs.mode")
        if mode not in CANONICAL_MODES:
            raise AdapterOutputError("mode_inputs contains an unknown source mode")
        idea = record.get("idea")
        if not isinstance(idea, Mapping):
            raise AdapterOutputError("mode_inputs.idea must be a mapping")
        raw_components = idea.get("components")
        if not isinstance(raw_components, list):
            raise AdapterOutputError("mode_inputs.idea.components must be a list")
        explanations = idea.get("component_explanations")
        explanations = explanations if isinstance(explanations, Mapping) else {}
        components: dict[str, str | None] = {}
        for item in raw_components:
            if isinstance(item, Mapping):
                name = _required_text(item.get("name", item.get("component")), "source component")
                raw_description = item.get("description")
            else:
                name = _required_text(item, "source component")
                raw_description = explanations.get(name)
            description = str(raw_description).strip() if raw_description is not None else None
            if not description:
                description = None
            if name in components:
                raise AdapterOutputError(f"duplicate source component: {name}")
            components[name] = description
        evidence = _string_sequence(
            record.get("evidence", record.get("source_evidence")), "source evidence"
        )
        sources[mode] = {"components": components, "evidence": set(evidence)}
    if set(sources) != set(CANONICAL_MODES):
        raise AdapterOutputError("mode_inputs must contain every canonical source mode")
    return sources


def _normalize_repair_operation(
    record: Mapping[str, Any], *, local_shape: bool, allowed: set[str]
) -> dict[str, Any]:
    raw_op = _required_text(record.get("op", record.get("operation")), "repair operation")
    op = raw_op.lower()
    op = {
        "remove_component": "remove",
        "replace_component": "replace",
    }.get(op, op)
    if op not in ALLOWED_REPAIR_OPERATIONS or op not in allowed:
        raise AdapterOutputError(f"prohibited repair operation: {raw_op}")
    component = _required_text(record.get("component", record.get("source")), "repair component")
    target_value = record.get("target", record.get("replacement", ""))
    target = "" if target_value == "" else _required_text(target_value, "repair target")
    if local_shape and op == "replace":
        component, target = target, component
    if op in {"replace", "rewire"} and not target:
        raise AdapterOutputError(f"{op} repair requires a target")
    return {**deepcopy(record), "op": op, "component": component, "target": target}


def _metrics(
    payload: Mapping[str, Any], names: tuple[str, ...], *, integers_only: bool = False
) -> dict[str, float]:
    nested = payload.get("metrics")
    source = nested if isinstance(nested, Mapping) else payload
    metrics = {
        name: _finite_number(source.get(name), name, minimum=0, maximum=5)
        for name in names
    }
    if integers_only:
        invalid = [name for name in names if not isinstance(source.get(name), int) or isinstance(source.get(name), bool)]
        if invalid:
            raise AdapterOutputError("MCTS metrics must be integers: " + ", ".join(invalid))
    return metrics


def _finite_number(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise AdapterOutputError(f"{field} must be a finite number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise AdapterOutputError(f"{field} must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise AdapterOutputError(f"{field} must be at most {maximum}")
    return number


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdapterOutputError(f"{field} must be a non-empty string")
    text = value.strip()
    _reject_placeholder_text(text, field)
    return text


def _string_sequence(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise AdapterOutputError(f"{field} must be a non-empty list of strings")
    return [_required_text(item, field) for item in value]


def _optional_string_sequence(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise AdapterOutputError(f"{field} must be a list of strings")
    return [_required_text(item, field) for item in value]


def _mapping_sequence(value: Any, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise AdapterOutputError(f"{field} must be a list of mappings")
    result = []
    for item in value:
        if not isinstance(item, Mapping):
            raise AdapterOutputError(f"{field} must contain only mappings")
        result.append(deepcopy(dict(item)))
    return result


def _reject_placeholders(value: Any, field: str = "output") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_placeholders(item, f"{field}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_placeholders(item, f"{field}[{index}]")
    elif isinstance(value, str) and value.strip():
        _reject_placeholder_text(value.strip(), field)


def _reject_placeholder_text(text: str, field: str) -> None:
    normalized = text.casefold().strip()
    exact = {"string", "placeholder", "todo", "tbd", "0-5", "0-1", "...", "…"}
    prefixes = ("generic_template_name_", "concrete_topic_specific_name_", "exact_component_name_")
    if normalized in exact or normalized.startswith(prefixes) or "{{" in text or "}}" in text:
        raise AdapterOutputError(f"{field} contains placeholder output")


def _format_fusion_prompt(request: Mapping[str, JsonValue]) -> str:
    mode_inputs = request.get("mode_inputs", [])
    prompt = IDEA_FUSION_PROMPT.format(
        topic=request.get("topic", ""),
        mature_idea="See user input: context.mature_idea_payload.",
        refinement_scope=json.dumps(request.get("context", {}).get("refinement_scope") if isinstance(request.get("context"), dict) else None),
        root_domains=json.dumps(request.get("context", {}).get("root_domains", []) if isinstance(request.get("context"), dict) else []),
        analysis="See user input: context.",
        mode_count=len(mode_inputs) if isinstance(mode_inputs, list) else 0,
        candidate_ideas_json="See user input: mode_inputs; all source details are supplied there.",
    )
    prompt += _fusion_source_catalog(mode_inputs)
    mature = _fusion_mature_idea(request)
    if mature is not None:
        prompt += (
            "\n== Frozen fused-idea identity ==\n"
            + json.dumps({key: mature[key] for key in IDENTITY_FIELDS if key in mature}, ensure_ascii=False)
            + "\nCopy these tags and root_domains into fused_idea exactly, including their order "
            "and empty arrays. They are frozen identity metadata, not a summary of your new "
            "method. Do not add, remove, rename, or reorder entries to describe the fusion.\n"
        )
    return prompt


def _fusion_mature_idea(request: Mapping[str, Any]) -> Mapping[str, Any] | None:
    context = request.get("context")
    if not isinstance(context, Mapping):
        return None
    mature = context.get("mature_idea_payload", context.get("mature_idea"))
    return mature if isinstance(mature, Mapping) else None


def _fusion_source_catalog(mode_inputs: Any) -> str:
    sources = _fusion_sources(mode_inputs)
    catalog = {
        mode: {
            "components": [
                {"name": name}
                for name, description in source["components"].items()
            ],
            "evidence_ids": sorted(source["evidence"]),
        }
        for mode, source in sources.items()
    }
    return (
        "\n== Authoritative selectable source catalog ==\n"
        + json.dumps(catalog, ensure_ascii=False)
        + "\nCopy component names and bare evidence IDs exactly from this catalog. "
        "Evidence IDs are opaque strings, not descriptions or reviewer feedback. "
        "Never append explanations, punctuation, or citations to an ID. "
        "Nested search traces, child ideas, titles, and evaluator feedback are context only; "
        "they do not add selectable components or evidence IDs. "
        "Each source_mode restricts BOTH component names and evidence IDs to that mode. "
        "If validation_feedback is supplied, correct every validation_issues entry in "
        "previous_draft and return the complete corrected JSON. Do not replace invalid "
        "evidence with an arbitrary allowed ID: choose evidence that supports the claim.\n"
    )


def _format_repair_prompt(request: Mapping[str, JsonValue]) -> str:
    prompt = FUSION_REPAIR_PROMPT.format(
        topic=request.get("topic", ""),
        root_domains=json.dumps(request.get("root_domains", []), ensure_ascii=False),
        current_idea_json="See user input: best_idea.",
        current_evaluation_json="See user input: best_evaluation.",
        candidate_ideas_json="See user input: mode_inputs.",
        atomic_op_reference=json.dumps(request.get("allowed_operations", []), ensure_ascii=False),
    )
    if "mode_inputs" in request:
        prompt += _fusion_source_catalog(request["mode_inputs"])
    return prompt
