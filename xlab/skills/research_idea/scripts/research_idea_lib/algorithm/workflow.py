"""Package-native research idea workflow orchestration.

The module deliberately owns no provider, retrieval, filesystem, environment, or
persistence policy.  Those boundaries are injected through the small protocols
below so the workflow can coordinate current adapters and future concurrent
search execution without coupling them to the package runtime.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Mapping, Protocol, Sequence

from ..providers.contracts import ProviderError, ProviderTrace, structured_input_digest
from ..research_idea_spec import WORKFLOW_ID
from ..resources.ids import stable_evidence_id
from ..resources.retrieval import CitationRegistry, OutcomeHit
from .contracts import (
    ALGORITHM_ID,
    OperatorAttemptRecord,
    RefinementBoundary,
    RolloutBlockerRecord,
)
from .fusion import CANONICAL_MODES as FUSION_CANONICAL_MODES
from .operator_grounding import MECHANISM_COMMIT_OPERATOR
from .operators import OPERATORS
from .response_projection import unwrap_answer_object
from .root_identity import (
    IDENTITY_FIELDS,
    ROOT_IDENTITY_INSTRUCTION,
    inherit_root_identity,
    validate_root_identity,
)

CANONICAL_MODES = FUSION_CANONICAL_MODES

OP_BACKGROUND = "xlab.research_idea.background.generate.v1"
OP_QUERY = "xlab.research_idea.retrieval.query.v1"
OP_RANKING = "xlab.research_idea.reference.rank.v1"
OP_ANALYSIS = "xlab.research_idea.analysis.generate.v1"
OP_REPLAN = "xlab.research_idea.analysis.replan.v1"
OP_MATERIALIZATION = "xlab.research_idea.idea.materialize.v1"
OP_RETRIEVAL = "xlab.research_idea.evidence.retrieve.v1"
OP_KEYNOTE_GROUNDING = "xlab.research_idea.keynote.ground.v1"
OP_SEARCH = "xlab.research_idea.search.run.v1"
OP_FUSION = "xlab.research_idea.fusion.generate.v1"

ALGORITHM_PROFILE_ID = ALGORITHM_ID
RUNTIME_PROFILE_ID = "xlab.research_idea.runtime.v1"
EVIDENCE_PROFILE_ID = "xlab.research_idea.evidence.v1"
SUCCESS_PROFILE_ID = "xlab.research_idea.success.v1"


class WorkflowContractError(ValueError):
    """An injected adapter violated a workflow boundary."""


class KeynoteGroundingError(RuntimeError):
    """A keynote stage failed after making traceable provider attempts."""

    def __init__(
        self,
        message: str,
        *,
        provider_traces: tuple[ProviderTrace, ...],
        usage: "BudgetUsage",
    ) -> None:
        super().__init__(message)
        self.provider_traces = provider_traces
        self.usage = usage


@dataclass(frozen=True)
class BudgetUsage:
    calls: int = 0
    iterations: int = 0
    evaluator_calls: int = 0
    generation_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.calls,
            self.iterations,
            self.evaluator_calls,
            self.generation_calls,
            self.input_tokens,
            self.output_tokens,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("budget counters must be non-negative integers")
        if isinstance(self.cost, bool) or not isinstance(self.cost, (int, float)) or not isfinite(float(self.cost)) or self.cost < 0:
            raise ValueError("budget cost must be a finite non-negative number")

    def __add__(self, other: "BudgetUsage") -> "BudgetUsage":
        return BudgetUsage(
            calls=self.calls + other.calls,
            iterations=self.iterations + other.iterations,
            evaluator_calls=self.evaluator_calls + other.evaluator_calls,
            generation_calls=self.generation_calls + other.generation_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost=self.cost + other.cost,
        )

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderOperation:
    name: str
    structured_input: Mapping[str, Any]


@dataclass(frozen=True)
class ProviderOutput:
    value: Mapping[str, Any]
    usage: BudgetUsage = BudgetUsage(calls=1)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class WorkflowProvider(Protocol):
    def execute(self, operation: ProviderOperation) -> ProviderOutput: ...


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    kind: str
    text: str
    provenance: Mapping[str, Any]
    paper_ids: tuple[str, ...] = ()
    source_id: str | None = None


@dataclass(frozen=True)
class RetrievalRequest:
    topic: str
    query: str
    background: Mapping[str, Any]
    source_context: Mapping[str, Any]


@dataclass(frozen=True)
class RetrievalOutput:
    evidence: tuple[Evidence, ...]
    resource_ids: tuple[str, ...]
    usage: BudgetUsage = BudgetUsage(calls=1)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    selected_outcome_hits: tuple[OutcomeHit, ...] = ()
    citations: CitationRegistry | None = None


class RetrievalAdapter(Protocol):
    def retrieve(self, request: RetrievalRequest) -> RetrievalOutput: ...


@dataclass(frozen=True)
class KeynoteGroundingOutput:
    ranked_keynotes: tuple[Mapping[str, Any], ...]
    curated_references: tuple[Mapping[str, Any], ...]
    provider_traces: tuple[ProviderTrace, ...]
    usage: BudgetUsage
    metadata: Mapping[str, Any] = field(default_factory=dict)


class KeynoteGrounder(Protocol):
    def run(
        self,
        *,
        topic: str,
        rag_query: str,
        selected_hits: tuple[OutcomeHit, ...],
        citations: CitationRegistry,
    ) -> KeynoteGroundingOutput: ...


@dataclass(frozen=True)
class ModeSearchRequest:
    mode: str
    seed: int
    root: Mapping[str, Any]
    context: Mapping[str, Any]
    root_signature: str
    evidence_signature: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class ModeSearchOutput:
    mode: str
    candidate: Mapping[str, Any]
    evidence_ids: tuple[str, ...]
    root_signature: str
    evidence_signature: str
    usage: BudgetUsage
    trace: Sequence[Mapping[str, Any]] = ()
    operator_attempts: tuple[OperatorAttemptRecord, ...] = ()
    rollout_blockers: tuple[RolloutBlockerRecord, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ModeSearch(Protocol):
    def run(self, request: ModeSearchRequest) -> ModeSearchOutput: ...


class SearchFactory(Protocol):
    def create(self, *, mode: str, seed: int) -> ModeSearch: ...


@dataclass(frozen=True)
class FusionWorkflowRequest:
    mode_inputs: tuple[Mapping[str, Any], ...]
    root: Mapping[str, Any]
    context: Mapping[str, Any]
    root_signature: str
    evidence_signature: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class FusionWorkflowOutput:
    idea: Mapping[str, Any]
    source_modes: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    root_signature: str
    evidence_signature: str
    usage: BudgetUsage = BudgetUsage(calls=1)
    trace: Sequence[Mapping[str, Any]] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


class FusionRunner(Protocol):
    def run(self, request: FusionWorkflowRequest) -> FusionWorkflowOutput: ...


@dataclass(frozen=True)
class WorkflowRequest:
    run_id: str
    topic: str
    source_context: Mapping[str, Any]
    seed: str = "0"
    mature_idea: Mapping[str, Any] | None = None
    refinement_scope: tuple[str, ...] = ()
    refinement_boundary: RefinementBoundary | None = None
    experiment_feedback: Mapping[str, Any] | str | None = None
    evidence: tuple[Evidence, ...] = ()
    resource_ids: tuple[str, ...] = ()
    rag_query: str | None = None
    selected_outcome_hits: tuple[OutcomeHit, ...] = ()
    citations: CitationRegistry | None = None
    retrieval_metadata: Mapping[str, Any] = field(default_factory=dict)
    research_policy: Mapping[str, Any] = field(default_factory=dict)
    task_context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowResult:
    status: str
    artifact: Mapping[str, Any]
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "success"


def run_workflow(
    request: WorkflowRequest,
    *,
    search_factory: SearchFactory,
    fusion_runner: FusionRunner,
    provider: WorkflowProvider,
    retrieval: RetrievalAdapter,
    keynote_grounder: KeynoteGrounder | None = None,
) -> WorkflowResult:
    """Execute one fail-closed research idea run and return namespace state."""

    artifact = _initial_artifact(request)
    budgets = _BudgetLedger()
    try:
        _validate_request(request)
        if request.experiment_feedback is None or request.research_policy:
            evidence, resource_ids, analysis = _normal_prelude(
                request,
                provider,
                retrieval,
                keynote_grounder,
                artifact,
                budgets,
            )
        if request.experiment_feedback is not None:
            if request.research_policy:
                # The full prelude above already retrieved and analysed fresh evidence.
                grounding = {}
            else:
                evidence = _validated_evidence(request.evidence)
                resource_ids = _nonempty_strings(request.resource_ids, "feedback resource_ids")
                _set_retrieval_namespace(artifact, evidence, resource_ids, request.retrieval_metadata)
                grounding = _run_keynote_grounding(
                    request, keynote_grounder, request.rag_query, request.selected_outcome_hits,
                    request.citations, artifact, budgets,
                )
            if not request.research_policy:
                analysis = _feedback_analysis(request, provider, evidence, grounding, artifact, budgets)
            replan = _provider_call(
                provider, OP_REPLAN,
                {"topic": request.topic, "analysis": deepcopy(dict(analysis)),
                 "experiment_feedback": deepcopy(request.experiment_feedback),
                 "mature_idea": deepcopy(dict(request.mature_idea or {})),
                 "refinement_scope": list(request.refinement_scope),
                 "refinement_boundary": request.refinement_boundary.to_payload() if request.refinement_boundary else None,
                 "evidence": [_evidence_dict(item) for item in evidence],
                 "evidence_ids": [item.evidence_id for item in evidence],
                 "research_policy": dict(request.research_policy), "task_context": dict(request.task_context)},
                artifact, budgets,
            )
            _require_mapping(replan.get("replan"), "re_analysis_replan.replan")
            artifact["analysis"]["entries"].append(deepcopy(dict(replan)))
            _stage(artifact, "re_analysis_replan")
            analysis = {**deepcopy(dict(analysis)), **deepcopy(dict(replan))}

        inheritance: dict[str, Any] = {}
        root = _select_root(request, analysis, inheritance=inheritance)
        root_signature = stable_signature(root)
        evidence_signature = stable_evidence_signature(evidence)
        evidence_ids = tuple(item.evidence_id for item in evidence)
        context = {
            "research_policy": dict(request.research_policy),
            "task_context": dict(request.task_context),
            "topic": request.topic,
            "source_context": deepcopy(dict(request.source_context)),
            "analysis": deepcopy(dict(analysis)),
            "evidence": [_evidence_dict(item) for item in evidence],
            "resource_ids": list(resource_ids),
            "mature_idea": request.mature_idea is not None,
            "mature_idea_payload": deepcopy(dict(request.mature_idea)) if request.mature_idea else None,
            "refinement_scope": list(request.refinement_scope),
            "refinement_boundary": request.refinement_boundary.to_payload() if request.refinement_boundary else None,
            "experiment_feedback": deepcopy(request.experiment_feedback),
        }
        context_signature = stable_signature(context)
        artifact["analysis"].update(
            {
                "root_idea": deepcopy(root),
                "root_signature": root_signature,
                "context_signature": context_signature,
            }
        )
        artifact["retrieval"]["evidence_signature"] = evidence_signature
        if inheritance:
            artifact["analysis"]["root_identity_inheritance"] = inheritance

        mode_outputs = _run_modes(
            request,
            search_factory,
            root,
            context,
            root_signature,
            evidence_signature,
            evidence_ids,
            artifact,
            budgets,
        )
        fusion = _run_fusion(
            fusion_runner,
            mode_outputs,
            root,
            context,
            root_signature,
            evidence_signature,
            evidence_ids,
            artifact,
            budgets,
        )
        if request.research_policy and request.mature_idea is not None:
            _validate_mature_root(request.mature_idea, fusion.idea, request.refinement_boundary,
                                  context="idea_fusion.idea")
        materialized = _provider_call(
            provider,
            OP_MATERIALIZATION,
            {
                "topic": request.topic,
                "fusion": deepcopy(dict(fusion.idea)),
                "references": deepcopy(request.source_context.get('references', [])),
                "research_policy": dict(request.research_policy),
                "task_context": dict(request.task_context),
                "evidence": [_evidence_dict(item) for item in evidence],
                "source_modes": list(CANONICAL_MODES),
                "evidence_ids": list(fusion.evidence_ids),
                "root_signature": root_signature,
                "evidence_signature": evidence_signature,
                "mature_idea": request.mature_idea is not None,
                "refinement_scope": list(request.refinement_scope),
                "refinement_boundary": (
                    request.refinement_boundary.to_payload()
                    if request.refinement_boundary
                    else None
                ),
            },
            artifact,
            budgets,
        )
        idea_result = _validate_materialization(materialized, fusion)
        _stage(artifact, "idea_materialization")
        _assert_immutable(root, root_signature, evidence, evidence_signature)

        artifact["ideation"].update(
            {
                "latest_candidate": deepcopy(dict(fusion.idea)),
                "fusion_result": _fusion_dict(fusion),
            }
        )
        artifact["persistence"].update(
            {"idea_result": idea_result, "schema_version": 1, "status": "success"}
        )
        artifact["run"].update(
            {"status": "success", "budgets": budgets.to_dict(), "completed_modes": list(CANONICAL_MODES)}
        )
        return WorkflowResult("success", artifact)
    except Exception as exc:  # adapters are untrusted; never leak nominal success
        message = f"{type(exc).__name__}: {exc}"
        artifact["run"].update({"status": "incomplete", "error": message, "budgets": budgets.to_dict()})
        artifact["persistence"].update(
            {"status": "incomplete", "schema_version": 1, "incomplete_reason": message}
        )
        return WorkflowResult("incomplete", artifact, message)


def stable_signature(value: object) -> str:
    try:
        return structured_input_digest(value)
    except (TypeError, ValueError) as exc:
        raise WorkflowContractError(f"value is not stable JSON: {exc}") from exc


def _feedback_analysis(request, provider, evidence, grounding, artifact, budgets):
    analysis = _provider_call(provider, OP_ANALYSIS, {
        "topic": request.topic, "source_context": deepcopy(dict(request.source_context)),
        "experiment_feedback": deepcopy(request.experiment_feedback),
        "evidence": [_evidence_dict(item) for item in evidence], **grounding,
        "mature_idea": deepcopy(dict(request.mature_idea or {})),
        "refinement_scope": list(request.refinement_scope),
        "refinement_boundary": request.refinement_boundary.to_payload() if request.refinement_boundary else None,
    }, artifact, budgets)
    _require_analysis(analysis, mature=bool(request.mature_idea))
    artifact["analysis"]["entries"].append(deepcopy(dict(analysis)))
    _stage(artifact, "advanced_analysis")
    return analysis


def stable_evidence_signature(evidence: Sequence[Evidence]) -> str:
    return stable_signature([_evidence_dict(item) for item in evidence])


def expected_evidence_id(evidence: Evidence) -> str:
    return stable_evidence_id(
        evidence.kind,
        evidence.text,
        paper_ids=evidence.paper_ids,
        source_id=evidence.source_id,
    )


def _normal_prelude(
    request: WorkflowRequest,
    provider: WorkflowProvider,
    retrieval: RetrievalAdapter,
    keynote_grounder: KeynoteGrounder | None,
    artifact: dict[str, Any],
    budgets: "_BudgetLedger",
) -> tuple[tuple[Evidence, ...], tuple[str, ...], Mapping[str, Any]]:
    background = _provider_call(
        provider,
        OP_BACKGROUND,
        {"topic": request.topic, "source_context": deepcopy(dict(request.source_context))},
        artifact,
        budgets,
    )
    _require_nonempty(background.get("background"), "topic_background.background")
    query = _provider_call(
        provider,
        OP_QUERY,
        {"topic": request.topic, "background": deepcopy(dict(background)),
         "mature_idea": request.mature_idea, "refinement_scope": list(request.refinement_scope),
         "experiment_feedback": request.experiment_feedback, "task_context": dict(request.task_context),
         "research_policy": dict(request.research_policy)},
        artifact,
        budgets,
    )
    query_text = _require_nonempty(query.get("query"), "rag_query.query")

    retrieval_request = RetrievalRequest(
        topic=request.topic,
        query=query_text,
        background=deepcopy(dict(background)),
        source_context=deepcopy(dict(request.source_context)),
    )
    input_digest = stable_signature(asdict(retrieval_request))
    try:
        retrieved = retrieval.retrieve(retrieval_request)
        if not isinstance(retrieved, RetrievalOutput):
            raise WorkflowContractError("retrieval adapter must return RetrievalOutput")
        evidence = _validated_evidence(retrieved.evidence)
        resource_ids = _nonempty_strings(retrieved.resource_ids, "retrieval resource_ids")
        budgets.retrieval = budgets.retrieval + retrieved.usage
        artifact["run"]["operation_trace"].append(
            {"event": "retrieval", "op_name": OP_RETRIEVAL, "status": "success", "input_digest": input_digest}
        )
    except Exception:
        artifact["run"]["operation_trace"].append(
            {"event": "retrieval", "op_name": OP_RETRIEVAL, "status": "error", "input_digest": input_digest}
        )
        raise

    ranking = _provider_call(
        provider,
        OP_RANKING,
        {
            "topic": request.topic,
            "query": query_text,
            "evidence": [_evidence_dict(item) for item in evidence],
        },
        artifact,
        budgets,
    )
    ranking_value = ranking.get("evidence_ids")
    if "evidence_ids" not in ranking and set(ranking) == {"answer"}:
        # Some OpenAI-compatible providers wrap a requested list in `answer`.
        # Preserve its ordering and still require an exact permutation below.
        ranking_value = ranking["answer"]
        artifact["retrieval"]["ranking_projection"] = {
            "source_key": "answer", "target_key": "evidence_ids",
            "raw_response_signature": stable_signature(ranking),
        }
    ranked_ids = _nonempty_strings(ranking_value, "reference_ranking.evidence_ids")
    evidence = _rank_evidence(evidence, ranked_ids)
    _set_retrieval_namespace(artifact, evidence, resource_ids, retrieved.metadata)
    grounding = _run_keynote_grounding(
        request,
        keynote_grounder,
        query_text,
        retrieved.selected_outcome_hits,
        retrieved.citations,
        artifact,
        budgets,
    )
    _stage(artifact, "knowledge_acquisition")

    analysis = _provider_call(
        provider,
        OP_ANALYSIS,
        {
            "topic": request.topic,
            "background": deepcopy(dict(background)),
            "research_policy": dict(request.research_policy),
            "task_context": dict(request.task_context),
            "experiment_feedback": request.experiment_feedback,
            "evidence": [_evidence_dict(item) for item in evidence],
            **grounding,
            "mature_idea": deepcopy(dict(request.mature_idea or {})),
            "refinement_scope": list(request.refinement_scope),
            "refinement_boundary": (
                request.refinement_boundary.to_payload()
                if request.refinement_boundary
                else None
            ),
        },
        artifact,
        budgets,
    )
    _require_analysis(analysis, mature=bool(request.mature_idea))
    artifact["analysis"]["entries"].append(deepcopy(dict(analysis)))
    _stage(artifact, "advanced_analysis")
    return evidence, resource_ids, analysis


def _run_keynote_grounding(
    request: WorkflowRequest,
    grounder: KeynoteGrounder | None,
    rag_query: str | None,
    selected_hits: tuple[OutcomeHit, ...],
    citations: CitationRegistry | None,
    artifact: dict[str, Any],
    budgets: "_BudgetLedger",
) -> dict[str, Any]:
    if grounder is None:
        return {}
    if not isinstance(rag_query, str) or not rag_query.strip():
        raise WorkflowContractError("keynote grounding requires the OutcomeRAG query")
    if not selected_hits or not isinstance(citations, CitationRegistry):
        raise WorkflowContractError(
            "keynote grounding requires selected OutcomeRAG hits and a validated CitationRegistry"
        )
    try:
        output = grounder.run(
            topic=request.topic,
            rag_query=rag_query,
            selected_hits=selected_hits,
            citations=citations,
        )
        if not isinstance(output, KeynoteGroundingOutput):
            raise WorkflowContractError(
                "keynote grounder must return KeynoteGroundingOutput"
            )
        ranked = [deepcopy(dict(item)) for item in output.ranked_keynotes]
        curated = [deepcopy(dict(item)) for item in output.curated_references]
        if not ranked or not curated:
            raise WorkflowContractError("keynote grounding cannot be empty")
        traces = [
            {
                **_provider_trace_record(
                    trace,
                    trace.operation,
                    trace.input_digest,
                    "success",
                ),
                "provider_operation": trace.operation,
            }
            for trace in output.provider_traces
        ]
        if output.usage.calls != len(traces):
            raise WorkflowContractError(
                "keynote usage calls must match provider trace count"
            )
        budgets.keynote = budgets.keynote + output.usage
        artifact["run"]["operation_trace"].extend(traces)
        if output.metadata.get("resumed") is True:
            artifact["run"]["operation_trace"].append(
                {
                    "event": "keynote_resume",
                    "op_name": OP_KEYNOTE_GROUNDING,
                    "status": "resumed",
                }
            )
        keynote_payload = {
            **deepcopy(dict(output.metadata)),
            "ranked_keynotes": ranked,
            "curated_references": curated,
            "provider_traces": [trace.to_dict() for trace in output.provider_traces],
            "provider_usage": output.usage.to_dict(),
        }
        artifact["retrieval"].update(
            {
                "ranked_keynotes": ranked,
                "keynote_metadata": deepcopy(dict(output.metadata)),
            }
        )
        artifact["retrieval"].setdefault("metadata", {})[
            "keynote_pipeline"
        ] = keynote_payload
        artifact["analysis"]["keynote_grounding"] = {
            "capsules": curated,
            "curated_references": curated,
        }
        return {"capsules": curated, "curated_references": curated}
    except KeynoteGroundingError as error:
        budgets.keynote = budgets.keynote + error.usage
        artifact["run"]["operation_trace"].extend(
            {
                **_provider_trace_record(
                    trace,
                    trace.operation,
                    trace.input_digest,
                    trace.status,
                ),
                "provider_operation": trace.operation,
            }
            for trace in error.provider_traces
        )
        artifact["run"]["operation_trace"].append(
            {"event": "keynote", "op_name": OP_KEYNOTE_GROUNDING, "status": "error"}
        )
        raise
    except Exception:
        artifact["run"]["operation_trace"].append(
            {"event": "keynote", "op_name": OP_KEYNOTE_GROUNDING, "status": "error"}
        )
        raise


def _run_modes(
    request: WorkflowRequest,
    factory: SearchFactory,
    root: dict[str, Any],
    context: dict[str, Any],
    root_signature: str,
    evidence_signature: str,
    evidence_ids: tuple[str, ...],
    artifact: dict[str, Any],
    budgets: "_BudgetLedger",
) -> tuple[ModeSearchOutput, ...]:
    runners: list[tuple[str, int, ModeSearch]] = []
    identities: set[int] = set()
    for mode in CANONICAL_MODES:
        seed = _mode_seed(request.seed, root_signature, evidence_signature, mode)
        runner = factory.create(mode=mode, seed=seed)
        if runner is None or id(runner) in identities:
            raise WorkflowContractError("search_factory must create an independent search instance per mode")
        identities.add(id(runner))
        runners.append((mode, seed, runner))

    outputs: list[ModeSearchOutput] = []
    for mode, seed, runner in runners:
        search_request = ModeSearchRequest(
            mode=mode,
            seed=seed,
            root=deepcopy(root),
            context=deepcopy(context),
            root_signature=root_signature,
            evidence_signature=evidence_signature,
            evidence_ids=evidence_ids,
        )
        try:
            output = runner.run(search_request)
            _validate_mode_output(output, mode, root_signature, evidence_signature, evidence_ids)
        except Exception:
            artifact["run"]["operation_trace"].append(
                {"event": "search", "op_name": OP_SEARCH, "mode": mode, "seed": seed, "status": "error"}
            )
            raise
        budgets.search_by_mode[mode] = output.usage
        artifact["run"]["operation_trace"].append(
            {
                "event": "search_resume" if output.metadata.get("resumed") is True else "search",
                "op_name": OP_SEARCH,
                "mode": mode,
                "seed": seed,
                "status": "resumed" if output.metadata.get("resumed") is True else "success",
            }
        )
        artifact["ideation"]["mode_candidates"].append(_mode_output_dict(output))
        outputs.append(output)
        _stage(artifact, "mcts_search", mode=mode)
    return tuple(outputs)


def _run_fusion(
    runner: FusionRunner,
    outputs: tuple[ModeSearchOutput, ...],
    root: dict[str, Any],
    context: dict[str, Any],
    root_signature: str,
    evidence_signature: str,
    evidence_ids: tuple[str, ...],
    artifact: dict[str, Any],
    budgets: "_BudgetLedger",
) -> FusionWorkflowOutput:
    request = FusionWorkflowRequest(
        mode_inputs=tuple(_mode_output_dict(output) for output in outputs),
        root=deepcopy(root),
        context=deepcopy(context),
        root_signature=root_signature,
        evidence_signature=evidence_signature,
        evidence_ids=evidence_ids,
    )
    try:
        fusion = runner.run(request)
        if not isinstance(fusion, FusionWorkflowOutput):
            raise WorkflowContractError("fusion_runner must return FusionWorkflowOutput")
        if fusion.source_modes != CANONICAL_MODES:
            raise WorkflowContractError("fusion output must contain every canonical mode in stable order")
        fusion_idea = _require_mapping(fusion.idea, "fusion.idea")
        _require_nonempty(fusion_idea.get("hypothesis"), "fusion.idea.hypothesis")
        _validate_signatures(fusion.root_signature, fusion.evidence_signature, root_signature, evidence_signature, "fusion")
        _validate_evidence_subset(fusion.evidence_ids, evidence_ids, "fusion")
    except Exception:
        artifact["run"]["operation_trace"].append(
            {"event": "fusion", "op_name": OP_FUSION, "status": "error"}
        )
        raise
    budgets.fusion = budgets.fusion + fusion.usage
    artifact["run"]["operation_trace"].append(
        {"event": "fusion", "op_name": OP_FUSION, "status": "success"}
    )
    _stage(artifact, "idea_fusion")
    return fusion


def _provider_call(
    provider: WorkflowProvider,
    name: str,
    structured_input: Mapping[str, Any],
    artifact: dict[str, Any],
    budgets: "_BudgetLedger",
) -> Mapping[str, Any]:
    provider_input = deepcopy(dict(structured_input))
    mature = provider_input.get("mature_idea")
    if name in {OP_ANALYSIS, OP_REPLAN} and isinstance(mature, Mapping) and mature:
        provider_input["root_identity_contract"] = {
            "instruction": ROOT_IDENTITY_INSTRUCTION,
            "required_values": {key: deepcopy(mature[key]) for key in IDENTITY_FIELDS if key in mature},
        }
    if name == OP_RANKING:
        provider_input["output_contract"] = {
            "instruction": 'Return exactly {"evidence_ids": ["supplied-id", "..."]}. '
                           'Use every supplied ID once; no answer wrapper or additional keys.',
            "required_key": "evidence_ids",
        }
    operation = ProviderOperation(name, provider_input)
    digest = stable_signature(operation.structured_input)
    try:
        output = provider.execute(operation)
        if not isinstance(output, ProviderOutput):
            raise WorkflowContractError(f"provider operation {name} must return ProviderOutput")
        value = _require_mapping(output.value, f"provider operation {name}")
        trace = _provider_trace_record(output.metadata.get("provider_trace"), name, digest, "success")
        budgets.provider = budgets.provider + output.usage
    except ProviderError as error:
        artifact["run"]["operation_trace"].append(
            _provider_trace_record(error.trace, name, digest, "error")
        )
        raise
    except Exception:
        artifact["run"]["operation_trace"].append(
            {"event": "llm_call", "op_name": name, "status": "error", "input_digest": digest}
        )
        raise
    artifact["run"]["operation_trace"].append(trace)
    raw_signature = stable_signature(value)
    value, wrapper_depth = unwrap_answer_object(value)
    if wrapper_depth:
        artifact["run"].setdefault("response_projections", []).append({
            "operation": name, "source_key": "answer", "target_key": "object",
            "wrapper_depth": wrapper_depth, "raw_response_signature": raw_signature,
        })
    scalar_field = {OP_BACKGROUND: "background", OP_QUERY: "query"}.get(name)
    if scalar_field and set(value) == {"answer"} and isinstance(value["answer"], str):
        artifact["run"].setdefault("response_projections", []).append({
            "operation": name, "source_key": "answer", "target_key": scalar_field,
            "raw_response_signature": stable_signature(value),
        })
        value = {scalar_field: value["answer"]}
    return deepcopy(dict(value))


def _provider_trace_record(
    value: object,
    operation: str,
    input_digest: str,
    expected_status: str,
) -> dict[str, Any]:
    raw = value.to_dict() if isinstance(value, ProviderTrace) else value
    if not isinstance(raw, Mapping):
        raise WorkflowContractError("provider output requires provider_trace metadata")
    trace = deepcopy(dict(raw))
    for trace_field in (
        "provider",
        "operation",
        "input_digest",
        "output_kind",
        "model",
        "attempts",
        "status",
        "error_code",
    ):
        if trace_field not in trace:
            raise WorkflowContractError(f"provider trace is missing {trace_field}")
    if trace["operation"] != operation:
        raise WorkflowContractError("provider trace operation does not match workflow operation")
    if trace["input_digest"] != input_digest:
        raise WorkflowContractError("provider trace input digest does not match workflow input")
    if trace["status"] != expected_status:
        raise WorkflowContractError("provider trace status does not match workflow outcome")
    return {"event": "llm_call", "op_name": operation, **trace}


def _validate_request(request: WorkflowRequest) -> None:
    _require_nonempty(request.run_id, "run_id")
    _require_nonempty(request.topic, "topic")
    _require_mapping(request.source_context, "source_context")
    mature = request.mature_idea is not None
    if mature:
        _validate_idea(request.mature_idea, "mature_idea")
    if request.refinement_scope:
        _nonempty_strings(request.refinement_scope, "refinement_scope")
    if request.refinement_boundary is not None and not isinstance(
        request.refinement_boundary, RefinementBoundary
    ):
        raise WorkflowContractError("refinement_boundary must be a RefinementBoundary")
    if request.experiment_feedback is not None and not mature:
        raise WorkflowContractError("experiment_feedback requires a mature_idea")


def _validate_mode_output(
    output: object,
    expected_mode: str,
    root_signature: str,
    evidence_signature: str,
    evidence_ids: tuple[str, ...],
) -> None:
    if not isinstance(output, ModeSearchOutput):
        raise WorkflowContractError(f"search {expected_mode} must return ModeSearchOutput")
    if output.mode != expected_mode:
        raise WorkflowContractError(f"search returned mode {output.mode!r}; expected {expected_mode!r}")
    _validate_idea(output.candidate, f"search {expected_mode} candidate")
    _validate_signatures(output.root_signature, output.evidence_signature, root_signature, evidence_signature, f"search {expected_mode}")
    _validate_evidence_subset(output.evidence_ids, evidence_ids, f"search {expected_mode}")
    _validate_rollout_records(output, expected_mode)


def _validate_rollout_records(output: ModeSearchOutput, expected_mode: str) -> None:
    context = f"search {expected_mode}"
    known_operators = {operator.name for operator in OPERATORS}
    attempts: dict[tuple[int, str, str], OperatorAttemptRecord] = {}
    for record in output.operator_attempts:
        if not isinstance(record, OperatorAttemptRecord):
            raise WorkflowContractError(
                f"{context} operator attempts must contain OperatorAttemptRecord values"
            )
        if record.operator not in known_operators:
            raise WorkflowContractError(
                f"{context} operator attempt uses unknown operator {record.operator!r}"
            )
        if record.grounding_explicit_empty and (
            record.operator != MECHANISM_COMMIT_OPERATOR
            or record.outcome not in {"applied", "explicit_empty", "error"}
        ):
            raise WorkflowContractError(
                f"{context} explicit-empty grounding is only valid for a mechanism commit attempt"
            )
        # Empty retrieval is independent of a later generation failure. Error
        # attempts still require a corresponding blocker below and are not
        # promoted to applied candidates.
        key = (record.parent_node_id, record.operator, record.plan_digest)
        if key in attempts:
            raise WorkflowContractError(
                f"{context} contains duplicate terminal attempts for one operator plan"
            )
        attempts[key] = record

    blockers: dict[tuple[int, str, str], RolloutBlockerRecord] = {}
    for record in output.rollout_blockers:
        if not isinstance(record, RolloutBlockerRecord):
            raise WorkflowContractError(
                f"{context} rollout blockers must contain RolloutBlockerRecord values"
            )
        if record.operator not in known_operators:
            raise WorkflowContractError(
                f"{context} rollout blocker uses unknown operator {record.operator!r}"
            )
        key = (record.parent_node_id, record.operator, record.plan_digest)
        if key in blockers:
            raise WorkflowContractError(
                f"{context} contains duplicate blockers for one operator plan"
            )
        blockers[key] = record

    error_keys = {key for key, record in attempts.items() if record.outcome == "error"}
    missing = error_keys - blockers.keys()
    if missing:
        raise WorkflowContractError(
            f"{context} error attempt requires one corresponding rollout blocker"
        )
    orphaned = blockers.keys() - attempts.keys()
    if orphaned:
        raise WorkflowContractError(f"{context} contains an orphan rollout blocker")
    non_error = blockers.keys() - error_keys
    if non_error:
        raise WorkflowContractError(
            f"{context} rollout blockers may only correspond to error attempts"
        )


def _validate_materialization(
    output: Mapping[str, Any], fusion: FusionWorkflowOutput
) -> dict[str, Any]:
    idea_result = _require_mapping(output.get("idea_result"), "idea_materialization.idea_result")
    _validate_idea(idea_result, "idea_materialization.idea_result")
    modes = idea_result.get("source_modes")
    if not isinstance(modes, Sequence) or isinstance(modes, (str, bytes)) or tuple(modes) != CANONICAL_MODES:
        raise WorkflowContractError("materialized idea_result must preserve canonical source_modes")
    if stable_signature(idea_result.get("components")) != stable_signature(fusion.idea.get("components")):
        raise WorkflowContractError("materialization changed the fused component boundary")
    return deepcopy(dict(idea_result))


def _validated_evidence(raw: Sequence[Evidence]) -> tuple[Evidence, ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
        raise WorkflowContractError("retrieval must return non-empty evidence")
    result: list[Evidence] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Evidence):
            raise WorkflowContractError("retrieval evidence entries must be Evidence values")
        _require_nonempty(item.kind, "evidence.kind")
        _require_nonempty(item.text, "evidence.text")
        provenance = _require_mapping(item.provenance, "evidence.provenance")
        if not provenance:
            raise WorkflowContractError("evidence provenance cannot be empty")
        if item.evidence_id != expected_evidence_id(item):
            raise WorkflowContractError(f"unstable evidence ID: {item.evidence_id!r}")
        if item.evidence_id in seen:
            raise WorkflowContractError(f"duplicate evidence ID: {item.evidence_id}")
        seen.add(item.evidence_id)
        result.append(item)
    return tuple(result)


def _rank_evidence(evidence: tuple[Evidence, ...], ranked_ids: tuple[str, ...]) -> tuple[Evidence, ...]:
    by_id = {item.evidence_id: item for item in evidence}
    if len(ranked_ids) != len(evidence) or set(ranked_ids) != set(by_id):
        raise WorkflowContractError("reference_ranking must return every retrieved evidence ID exactly once")
    return tuple(by_id[item] for item in ranked_ids)


def _select_root(
    request: WorkflowRequest, analysis: Mapping[str, Any], *,
    inheritance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if request.mature_idea is not None:
        context = "advanced_analysis.root_idea"
        root_value: object = analysis.get("root_idea") if request.experiment_feedback is None else None
        replan = analysis.get("replan")
        if isinstance(replan, Mapping):
            root_value = replan.get("root_idea", root_value)
            context = "re_analysis_replan.root_idea"
        root = deepcopy(dict(root_value)) if isinstance(root_value, Mapping) else deepcopy(dict(request.mature_idea))
        raw_signature = stable_signature(root)
        try:
            root, fields = inherit_root_identity(request.mature_idea, root, context=context)
        except ValueError as error:
            raise WorkflowContractError(str(error)) from error
        _validate_mature_root(request.mature_idea, root, request.refinement_boundary, context=context)
        if fields and inheritance is not None:
            inheritance.update(
                schema_version="xlab.root_identity_inheritance.v1", stage=context,
                fields=fields, parent_signature=stable_signature(request.mature_idea),
                raw_root_signature=raw_signature, normalized_root_signature=stable_signature(root),
            )
    else:
        root = deepcopy(dict(_require_mapping(analysis.get("root_idea"), "advanced_analysis.root_idea")))
    _validate_idea(root, "search root")
    return root


def _validate_mature_root(
    mature: Mapping[str, Any],
    root: Mapping[str, Any],
    refinement_boundary: RefinementBoundary | None,
    *, context: str = "feedback replan.root_idea",
) -> None:
    _validate_idea(root, context)
    try:
        validate_root_identity(mature, root, context=context)
    except ValueError as error:
        raise WorkflowContractError(str(error)) from error
    if refinement_boundary is None:
        return
    allowed = set(refinement_boundary.allowed_component_ids)
    if not allowed:
        return

    def components(value: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for item in value.get("components", []):
            if isinstance(item, Mapping):
                name = str(item.get("name") or item.get("component") or "").strip()
                record: Any = deepcopy(dict(item))
            else:
                name = str(item).strip()
                record = name
            if name:
                result[name] = record
        return result

    mature_components = components(mature)
    root_components = components(root)
    changed = {
        name
        for name, item in mature_components.items()
        if root_components.get(name) != item
    }
    if not changed.issubset(allowed):
        raise WorkflowContractError(
            f"{context} changed mature components outside refinement_boundary: "
            + ", ".join(sorted(changed - allowed))
        )


def _require_analysis(value: Mapping[str, Any], *, mature: bool) -> None:
    _require_mapping(value.get("analysis"), "advanced_analysis.analysis")
    if not mature:
        _validate_idea(_require_mapping(value.get("root_idea"), "advanced_analysis.root_idea"), "advanced_analysis.root_idea")


def _validate_idea(value: Mapping[str, Any], label: str) -> None:
    idea = _require_mapping(value, label)
    _require_nonempty(idea.get("title"), f"{label}.title")
    components = idea.get("components")
    if isinstance(components, (str, bytes)) or not isinstance(components, Sequence) or not components:
        raise WorkflowContractError(f"{label}.components must be a non-empty sequence")


def _validate_signatures(actual_root: str, actual_evidence: str, root: str, evidence: str, label: str) -> None:
    if actual_root != root or actual_evidence != evidence:
        raise WorkflowContractError(f"{label} changed the shared root/evidence signatures")


def _validate_evidence_subset(actual: Sequence[str], available: tuple[str, ...], label: str) -> None:
    ids = _nonempty_strings(actual, f"{label} evidence_ids")
    if not set(ids).issubset(set(available)):
        raise WorkflowContractError(f"{label} returned unknown evidence provenance")


def _assert_immutable(root: Mapping[str, Any], root_signature: str, evidence: Sequence[Evidence], evidence_signature: str) -> None:
    if stable_signature(root) != root_signature or stable_evidence_signature(evidence) != evidence_signature:
        raise WorkflowContractError("shared root or evidence changed during orchestration")


def _mode_seed(seed: str, root_signature: str, evidence_signature: str, mode: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{root_signature}\0{evidence_signature}\0{mode}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _initial_artifact(request: WorkflowRequest) -> dict[str, Any]:
    return {
        "run": {
            "research_policy": dict(request.research_policy),
            "run_id": request.run_id,
            "status": "running",
            "algorithm_profile_id": ALGORITHM_PROFILE_ID,
            "runtime_profile_id": RUNTIME_PROFILE_ID,
            "evidence_profile_id": EVIDENCE_PROFILE_ID,
            "success_profile_id": SUCCESS_PROFILE_ID,
            "algorithm_provenance": {
                "algorithm": ALGORITHM_PROFILE_ID,
                "runtime_profile": RUNTIME_PROFILE_ID,
                "evidence_profile": EVIDENCE_PROFILE_ID,
                "success_profile": SUCCESS_PROFILE_ID,
            },
            "workflow_trace": [],
            "operation_trace": [],
        },
        "retrieval": {"rag_hits": [], "references": [], "evidence": []},
        "analysis": {"entries": []},
        "ideation": {"mode_candidates": []},
        "persistence": {},
    }


def _set_retrieval_namespace(
    artifact: dict[str, Any], evidence: tuple[Evidence, ...], resource_ids: tuple[str, ...], metadata: Mapping[str, Any]
) -> None:
    records = [_evidence_dict(item) for item in evidence]
    artifact["retrieval"].update(
        {
            "rag_hits": deepcopy(records),
            "references": sorted({paper_id for item in evidence for paper_id in item.paper_ids}),
            "evidence": records,
            "evidence_ids": [item.evidence_id for item in evidence],
            "resource_ids": list(resource_ids),
            "metadata": deepcopy(dict(metadata)),
        }
    )


def _stage(artifact: dict[str, Any], stage: str, *, mode: str | None = None) -> None:
    entry = {"workflow": WORKFLOW_ID, "stage": stage, "status": "success"}
    if mode is not None:
        entry["mode"] = mode
    artifact["run"]["workflow_trace"].append(entry)


def _evidence_dict(item: Evidence) -> dict[str, Any]:
    return {
        "evidence_id": item.evidence_id,
        "kind": item.kind,
        "text": item.text,
        "provenance": deepcopy(dict(item.provenance)),
        "paper_ids": list(item.paper_ids),
        "source_id": item.source_id,
    }


def _mode_output_dict(output: ModeSearchOutput) -> dict[str, Any]:
    return {
        "mode": output.mode,
        "idea": deepcopy(dict(output.candidate)),
        "evidence": list(output.evidence_ids),
        "root_signature": output.root_signature,
        "evidence_signature": output.evidence_signature,
        "usage": output.usage.to_dict(),
        "trace": deepcopy(list(output.trace)),
        "operator_attempts": [item.to_payload() for item in output.operator_attempts],
        "rollout_blockers": [item.to_payload() for item in output.rollout_blockers],
        "metadata": deepcopy(dict(output.metadata)),
    }


def _fusion_dict(output: FusionWorkflowOutput) -> dict[str, Any]:
    return {
        "idea": deepcopy(dict(output.idea)),
        "source_modes": list(output.source_modes),
        "evidence_ids": list(output.evidence_ids),
        "root_signature": output.root_signature,
        "evidence_signature": output.evidence_signature,
        "usage": output.usage.to_dict(),
        "trace": deepcopy(list(output.trace)),
        "metadata": deepcopy(dict(output.metadata)),
    }


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkflowContractError(f"{label} must be a mapping")
    return value


def _require_nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkflowContractError(f"{label} must be a non-empty string")
    return value.strip()


def _nonempty_strings(value: object, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise WorkflowContractError(f"{label} must be a non-empty sequence")
    if any(not isinstance(item, str) for item in value):
        raise WorkflowContractError(f"{label} must contain only strings")
    result = tuple(item.strip() for item in value)
    if any(not item for item in result) or len(set(result)) != len(result):
        raise WorkflowContractError(f"{label} must contain unique non-empty strings")
    return result


@dataclass
class _BudgetLedger:
    provider: BudgetUsage = field(default_factory=BudgetUsage)
    retrieval: BudgetUsage = field(default_factory=BudgetUsage)
    search_by_mode: dict[str, BudgetUsage] = field(default_factory=dict)
    fusion: BudgetUsage = field(default_factory=BudgetUsage)
    keynote: BudgetUsage = field(default_factory=BudgetUsage)

    def to_dict(self) -> dict[str, Any]:
        search = sum(self.search_by_mode.values(), BudgetUsage())
        total = self.provider + self.retrieval + search + self.fusion + self.keynote
        return {
            "provider": self.provider.to_dict(),
            "retrieval": self.retrieval.to_dict(),
            "search": {
                "aggregate": search.to_dict(),
                "by_mode": {mode: self.search_by_mode[mode].to_dict() for mode in CANONICAL_MODES if mode in self.search_by_mode},
            },
            "fusion": self.fusion.to_dict(),
            "keynote": self.keynote.to_dict(),
            "total": total.to_dict(),
        }
