"""Production adapters for the package-native research idea workflow."""

from __future__ import annotations

import json
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from ..checkpoints import stage_completed, write_checkpoint
from ..common import read_json, run_paths, write_portable_json
from ..config import RuntimeConfig
from ..inputs import IdeaRequest
from ..providers import (
    JsonValue,
    Provider,
    ProviderError,
    ProviderRequest,
    ProviderResult,
    ProviderTrace,
)
from ..research_idea_spec import (
    ALGORITHM_ID,
    ALGORITHM_SPEC_VERSION,
    COMPONENT_NOVELTY_PROFILE_ID,
    OPERATOR_RETRIEVAL_PROFILE_ID,
    PRODUCTION_PROMPT_MODE,
    PROMPT_ROUTING_PROFILE_ID,
    WORKFLOW_ID,
)
from ..resources.component_novelty import (
    ComponentNoveltyRuntime,
    build_component_novelty_runtime,
)
from ..resources.ids import stable_evidence_id
from ..resources.manifest import sha256_file
from ..resources.retrieval import CitationRegistry, OutcomeHit, SurveyParagraph
from ..survey_repository import SurveyArtifactRepository
from .component_novelty import (
    COMPONENT_NOVELTY_OPERATION,
    ComponentNoveltyEvaluator,
    ComponentNoveltyOperation,
    ComponentNoveltyProviderOutput,
)
from .contracts import (
    IdeaComponent,
    IdeaProviderContext,
    IdeaState,
    OperatorAttemptRecord,
    RankedEvidence,
    RefinementBoundary,
    RolloutBlockerRecord,
)
from .fusion import CANONICAL_MODES, FusionRequest, fuse_five_modes
from .keynote_pipeline import KeynotePipelineRequest, KeynotePipelineResult, run_keynote_pipeline
from .memory import MemoryState
from .operator_grounding import OperatorGroundingEvaluator
from .prompts import (
    ADVANCED_ANALYSIS_PROMPT,
    RE_ANALYSIS_REPLAN_PROMPT,
    TOPIC_BACKGROUND_PROMPT,
)
from .prompts.idea_result_alignment import IDEA_RESULT_ALIGNMENT_PROMPT
from .prompts.rag_query import RAG_QUERY_PROMPT
from .provider_adapter import (
    FUSION_GENERATE_OPERATION,
    FUSION_REFEREE_OPERATION,
    FUSION_REPAIR_OPERATION,
    IDEA_DIAGNOSTIC_OPERATION,
    IDEA_EVALUATE_OPERATION,
    IDEA_GENERATE_OPERATION,
    ProviderAdapter,
)
from .search import MCTSEngine, SearchConfig, SearchResult
from .spec import SPEC_VERSION
from .workflow import (
    OP_ANALYSIS,
    OP_BACKGROUND,
    OP_MATERIALIZATION,
    OP_QUERY,
    OP_RANKING,
    OP_REPLAN,
    BudgetUsage,
    Evidence,
    FusionWorkflowOutput,
    KeynoteGroundingError,
    KeynoteGroundingOutput,
    ModeSearchOutput,
    ProviderOperation,
    ProviderOutput,
    RetrievalOutput,
    RetrievalRequest,
    RUNTIME_PROFILE_ID,
    WorkflowRequest,
    WorkflowResult,
    run_workflow,
    stable_signature,
)

_WORKFLOW_MODELS = {
    OP_BACKGROUND: "agent_model",
    OP_QUERY: "agent_model",
    OP_RANKING: "agent_model",
    OP_ANALYSIS: "agent_model",
    OP_REPLAN: "generation_model",
    OP_MATERIALIZATION: "fusion_model",
}
_PROVIDER_OPERATION_NAMES = {
    IDEA_DIAGNOSTIC_OPERATION: IDEA_DIAGNOSTIC_OPERATION,
    IDEA_EVALUATE_OPERATION: IDEA_EVALUATE_OPERATION,
    IDEA_GENERATE_OPERATION: IDEA_GENERATE_OPERATION,
    COMPONENT_NOVELTY_OPERATION: COMPONENT_NOVELTY_OPERATION,
    FUSION_GENERATE_OPERATION: FUSION_GENERATE_OPERATION,
    FUSION_REFEREE_OPERATION: FUSION_REFEREE_OPERATION,
    FUSION_REPAIR_OPERATION: FUSION_REPAIR_OPERATION,
}
_REQUIRED_RESULT_TEXT = (
    "title",
    "abstract",
    "core_contribution",
    "research_question",
    "hypothesis",
    "method",
    "introduction",
)
_REQUIRED_RESULT_LISTS = (
    "experiment_plan",
    "data_requirements",
    "baselines",
    "metrics",
    "risks",
    "components",
    "algorithm",
    "reference_papers",
)


class GenericWorkflowProvider:
    """Adapt the generic provider contract to workflow operations."""

    def __init__(self, provider: Provider, runtime: RuntimeConfig) -> None:
        self._provider = provider
        self._runtime = runtime

    def execute(self, operation: ProviderOperation) -> ProviderOutput:
        model_field = _WORKFLOW_MODELS.get(operation.name)
        if model_field is None:
            raise ValueError(f"unsupported workflow provider operation: {operation.name}")
        model = str(getattr(self._runtime, model_field))
        structured_input = _json_mapping(operation.structured_input, operation.name)
        request = ProviderRequest(
            operation=operation.name,
            model=model,
            structured_input=structured_input,
            system_prompt=_workflow_prompt(operation.name, structured_input),
            user_prompt=json.dumps(
                structured_input,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            output_kind="json",
        )
        result = self._provider.complete(request)
        if not isinstance(result.json_value, dict):
            raise ValueError(f"provider operation {operation.name} requires a JSON object")
        return ProviderOutput(
            value=deepcopy(result.json_value),
            usage=BudgetUsage(
                calls=1,
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
            ),
            metadata={"model": model, "provider_trace": result.trace.to_dict()},
        )


class SurveyRepositoryRetrieval:
    """Retrieve explicit evidence and curate selected OutcomeRAG keynotes."""

    def __init__(
        self,
        repository: SurveyArtifactRepository,
        *,
        limit: int = 12,
    ) -> None:
        self._repository = repository
        self._limit = max(1, limit)

    def retrieve(self, request: RetrievalRequest) -> RetrievalOutput:
        evidence, resource_ids, resource_usage, selected_hits, citations = self._snapshot(
            request.query
        )
        return RetrievalOutput(
            evidence=evidence,
            resource_ids=resource_ids,
            usage=BudgetUsage(calls=1),
            metadata={
                "adapter": "survey_artifact_repository",
                "evidence_profile": "xlab.research_idea.evidence.v1",
                "query_digest": stable_signature(request.query),
                "available_evidence": len(self._repository.evidence_items),
                "selected_evidence": len(evidence),
                "resource_execution": resource_usage,
            },
            selected_outcome_hits=selected_hits,
            citations=citations,
        )

    def snapshot(self, query: str) -> RetrievalOutput:
        if not query.strip():
            raise ValueError("feedback keynote grounding requires an explicit retrieval query")
        evidence, resource_ids, resource_usage, selected_hits, citations = self._snapshot(
            query
        )
        return RetrievalOutput(
            evidence=evidence,
            resource_ids=resource_ids,
            usage=BudgetUsage(calls=1),
            metadata={
                "adapter": "survey_artifact_repository",
                "evidence_profile": "xlab.research_idea.evidence.v1",
                "query_digest": stable_signature(query),
                "available_evidence": len(self._repository.evidence_items),
                "selected_evidence": len(evidence),
                "resource_execution": resource_usage,
            },
            selected_outcome_hits=selected_hits,
            citations=citations,
        )

    def _snapshot(
        self, query: str
    ) -> tuple[
        tuple[Evidence, ...],
        tuple[str, ...],
        dict[str, dict[str, Any]],
        tuple[OutcomeHit, ...],
        CitationRegistry,
    ]:
        resource_result = self._repository.retrieve_resource_evidence(query, limit=self._limit)
        selected_items = list(resource_result.evidence_items)
        resource_usage = deepcopy(resource_result.usage)
        selected_hits = resource_result.selected_outcome_hits
        citations = resource_result.citations
        if not selected_hits:
            selected_hits = _selected_outcome_hits(selected_items)
        if citations is None:
            citations = _citation_registry(self._repository)
        selected_items = [
            item for item in selected_items if str(item.get("kind") or "") != "paper_keynote"
        ]

        evidence: list[Evidence] = []
        role_evidence_ids: dict[str, list[str]] = {}
        for rank, item in enumerate(selected_items, start=1):
            text = str(item.get("text") or item.get("summary") or item.get("title") or "").strip()
            kind = str(item.get("kind") or "survey_evidence").strip()
            source_id = str(item.get("id") or "").strip() or None
            raw_paper_ids = item.get("paper_ids")
            paper_ids = tuple(
                dict.fromkeys(
                    str(value).strip()
                    for value in raw_paper_ids
                    if str(value).strip()
                )
            ) if isinstance(raw_paper_ids, list) else ()
            evidence_id = stable_evidence_id(
                kind,
                text,
                paper_ids=paper_ids,
                source_id=source_id,
            )
            resource_provenance = item.get("resource_provenance")
            resource_provenance = (
                deepcopy(dict(resource_provenance)) if isinstance(resource_provenance, Mapping) else {}
            )
            role = str(resource_provenance.get("role") or "").strip()
            if role:
                role_evidence_ids.setdefault(role, []).append(evidence_id)
            evidence.append(
                Evidence(
                    evidence_id=evidence_id,
                    kind=kind,
                    text=text,
                    provenance={
                        "repository": "survey_artifact_repository",
                        "source": str(item.get("source") or "literature_survey"),
                        "source_id": source_id,
                        "title": str(item.get("title") or "Survey evidence"),
                        "rank": rank,
                        "resource": resource_provenance,
                    },
                    paper_ids=paper_ids,
                    source_id=source_id,
                )
            )
        if not evidence:
            raise ValueError("survey repository contains no retrievable evidence")
        self._finalize_resource_receipts(resource_usage, role_evidence_ids, evidence)
        return (
            tuple(evidence),
            self._resource_ids(),
            resource_usage,
            selected_hits,
            citations,
        )

    def _finalize_resource_receipts(
        self,
        usage: dict[str, dict[str, Any]],
        role_evidence_ids: Mapping[str, list[str]],
        evidence: list[Evidence],
    ) -> None:
        portable = self._repository.source.resources.portable_resources()
        operations = {
            "graph": "expand_paper_linked_evidence",
            "keynotes": "enrich_paper_evidence",
            "outcome_model": "score_and_rank_survey_evidence",
            "component_index": "retrieve_component_novelty_context",
            "component_model": "score_component_novelty_context",
        }
        role_alias = {"component_model": "component_index"}
        ranking_ids = [item.evidence_id for item in evidence]
        for role, receipt in usage.items():
            declared = portable.get(role, {})
            if isinstance(declared, Mapping):
                receipt["logical_uri"] = str(declared.get("logical_uri") or receipt.get("logical_uri") or "")
            receipt["resource_profile"] = "xlab.research_idea.evidence.v1"
            receipt["operation"] = operations[role]
            affected_role = role_alias.get(role, role)
            affected = role_evidence_ids.get(affected_role, [])
            receipt["affected_evidence_ids"] = list(affected)
            receipt["affected_ranking_ids"] = [value for value in ranking_ids if value in affected]

    def _resource_ids(self) -> tuple[str, ...]:
        identities: list[str] = []
        for resource in self._repository.source.resources.portable_resources().values():
            if not isinstance(resource, Mapping):
                continue
            identity = str(resource.get("logical_uri") or resource.get("resource_id") or "").strip()
            if identity:
                identities.append(identity)
        identities.extend(
            str(value).strip()
            for value in self._repository.source.direct_parent_artifact_ids
            if str(value).strip()
        )
        result = tuple(dict.fromkeys(identities))
        if not result:
            raise ValueError("survey repository has no explicit resource identities")
        return result


def _selected_outcome_hits(items: Sequence[Mapping[str, Any]]) -> tuple[OutcomeHit, ...]:
    hits: list[OutcomeHit] = []
    for item in items:
        if str(item.get("kind") or "") != "survey_markdown_paragraph":
            continue
        provenance = item.get("resource_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("selected OutcomeRAG evidence requires resource provenance")
        paragraph_id = str(provenance.get("paragraph_id") or item.get("id") or "").strip()
        section_path = provenance.get("section_path")
        paper_ids = item.get("paper_ids")
        score = item.get("resource_score")
        if (
            not paragraph_id
            or not isinstance(section_path, list)
            or any(not isinstance(value, str) or not value.strip() for value in section_path)
            or not isinstance(paper_ids, list)
            or any(not isinstance(value, str) or not value.strip() for value in paper_ids)
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
        ):
            raise ValueError("selected OutcomeRAG evidence is malformed")
        text = str(item.get("text") or "").strip()
        hits.append(
            OutcomeHit(
                paragraph=SurveyParagraph(
                    paragraph_id=paragraph_id,
                    text=text,
                    section_path=tuple(section_path),
                    section_context=text,
                    paper_ids=tuple(dict.fromkeys(paper_ids)),
                ),
                score=float(score),
            )
        )
    if not hits:
        raise ValueError("resource retrieval selected no OutcomeRAG evidence")
    return tuple(hits)


def _citation_registry(repository: SurveyArtifactRepository) -> CitationRegistry:
    bundle = repository.source.resources.bundle
    if bundle is None or repository.source.keynote_cache_path is None:
        raise ValueError("keynote pipeline requires the validated keynote resource")
    descriptor = bundle.manifest.keynotes
    matches = [file for file in descriptor.files if Path(file.path).name == "keynotes.json"]
    if len(matches) != 1:
        raise ValueError("keynote resource must declare exactly one keynotes.json")
    declared = matches[0]
    path = repository.source.keynote_cache_path.resolve() / declared.path
    if (
        not path.is_file()
        or path.stat().st_size != declared.size
        or sha256_file(path) != declared.sha256
    ):
        raise ValueError("keynote resource does not match its declared digest")
    payload = read_json(path)
    return CitationRegistry.from_payloads(repository.source.citations, payload)


def _keynote_output(
    result: KeynotePipelineResult,
    *,
    resumed: bool,
) -> KeynoteGroundingOutput:
    usage = result.provider_usage
    return KeynoteGroundingOutput(
        ranked_keynotes=tuple(item.to_payload() for item in result.ranked_keynotes),
        curated_references=tuple(item.to_payload() for item in result.capsules),
        provider_traces=result.provider_traces,
        usage=BudgetUsage(
            calls=len(result.provider_traces),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        ),
        metadata={
            "selected_paper_ids": list(result.selected_paper_ids),
            "keynote_paper_ids": list(result.keynote_paper_ids),
            "resumed": resumed,
        },
    )


def _keynote_output_dict(output: KeynoteGroundingOutput) -> dict[str, Any]:
    return {
        "ranked_keynotes": [deepcopy(dict(item)) for item in output.ranked_keynotes],
        "curated_references": [
            deepcopy(dict(item)) for item in output.curated_references
        ],
        "provider_traces": [trace.to_dict() for trace in output.provider_traces],
        "usage": output.usage.to_dict(),
        "metadata": deepcopy(dict(output.metadata)),
    }


def _keynote_output_from_dict(
    value: Mapping[str, Any],
    *,
    resumed: bool,
) -> KeynoteGroundingOutput:
    ranked = value.get("ranked_keynotes")
    curated = value.get("curated_references")
    traces = value.get("provider_traces")
    usage = value.get("usage")
    if (
        not isinstance(ranked, list)
        or any(not isinstance(item, Mapping) for item in ranked)
        or not isinstance(curated, list)
        or any(not isinstance(item, Mapping) for item in curated)
        or not isinstance(traces, list)
        or any(not isinstance(item, Mapping) for item in traces)
        or not isinstance(usage, Mapping)
    ):
        raise ValueError("keynote checkpoint is malformed")
    provider_traces = tuple(
        ProviderTrace(
            provider=str(item.get("provider") or ""),
            operation=str(item.get("operation") or ""),
            input_digest=str(item.get("input_digest") or ""),
            output_kind=str(item.get("output_kind") or "json"),  # type: ignore[arg-type]
            model=str(item.get("model") or ""),
            attempts=int(item.get("attempts") or 0),
            status=str(item.get("status") or "error"),  # type: ignore[arg-type]
            error_code=(
                str(item.get("error_code")) if item.get("error_code") is not None else None
            ),
        )
        for item in traces
    )
    metadata = deepcopy(dict(value.get("metadata"))) if isinstance(value.get("metadata"), Mapping) else {}
    metadata["resumed"] = resumed
    return KeynoteGroundingOutput(
        ranked_keynotes=tuple(deepcopy(dict(item)) for item in ranked),
        curated_references=tuple(deepcopy(dict(item)) for item in curated),
        provider_traces=provider_traces,
        usage=BudgetUsage(
            calls=int(usage.get("calls") or 0),
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cost=float(usage.get("cost") or 0),
        ),
        metadata=metadata,
    )


class _RecordingKeynoteProvider:
    """Record every completed or failed keynote provider attempt."""

    def __init__(self, provider: Provider) -> None:
        self._provider = provider
        self.results: list[ProviderResult] = []
        self.error_traces: list[ProviderTrace] = []

    def complete(self, request: ProviderRequest) -> ProviderResult:
        try:
            result = self._provider.complete(request)
        except ProviderError as error:
            self.error_traces.append(error.trace)
            raise
        if isinstance(result, ProviderResult):
            self.results.append(result)
        return result

    @property
    def traces(self) -> tuple[ProviderTrace, ...]:
        return tuple(result.trace for result in self.results) + tuple(self.error_traces)

    @property
    def usage(self) -> BudgetUsage:
        return BudgetUsage(
            calls=len(self.traces),
            input_tokens=sum(result.usage.input_tokens for result in self.results),
            output_tokens=sum(result.usage.output_tokens for result in self.results),
        )


class NativeKeynoteGrounder:
    """Run and resume the explicit provider-backed keynote stage."""

    def __init__(
        self,
        provider: Provider,
        *,
        model: str,
        run_dir: Path | None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._run_dir = run_dir

    def run(
        self,
        *,
        topic: str,
        rag_query: str,
        selected_hits: tuple[OutcomeHit, ...],
        citations: CitationRegistry,
    ) -> KeynoteGroundingOutput:
        dependencies = {
            "topic": topic,
            "rag_query": rag_query,
            "selected_hits": [
                {
                    "paragraph_id": hit.paragraph.paragraph_id,
                    "score": hit.score,
                    "paper_ids": list(hit.paragraph.paper_ids),
                }
                for hit in selected_hits
            ],
            "citations": {
                "references": citations.references,
                "keynotes": citations.keynotes,
            },
            "model": self._model,
        }
        resumed = self._resume(dependencies)
        if resumed is not None:
            return resumed
        recorder = _RecordingKeynoteProvider(self._provider)
        try:
            result = run_keynote_pipeline(
                KeynotePipelineRequest(
                    topic=topic,
                    rag_query=rag_query,
                    selected_hits=selected_hits,
                    citations=citations,
                    model=self._model,
                ),
                provider=recorder,
            )
        except Exception as error:
            raise KeynoteGroundingError(
                str(error),
                provider_traces=recorder.traces,
                usage=recorder.usage,
            ) from error
        output = _keynote_output(result, resumed=False)
        self._checkpoint(output, dependencies)
        return output

    def _path(self) -> Path | None:
        return run_paths(self._run_dir)["workflow_keynote"] if self._run_dir else None

    def _resume(self, dependencies: Mapping[str, Any]) -> KeynoteGroundingOutput | None:
        path = self._path()
        if path is None or not stage_completed(
            self._run_dir,
            "keynote_grounding",
            [path],
            dependencies=dependencies,
        ):
            return None
        value = read_json(path)
        if not isinstance(value, Mapping):
            return None
        return _keynote_output_from_dict(value, resumed=True)

    def _checkpoint(
        self,
        output: KeynoteGroundingOutput,
        dependencies: Mapping[str, Any],
    ) -> None:
        path = self._path()
        if path is None:
            return
        value = _keynote_output_dict(output)
        write_portable_json(path, value)
        write_checkpoint(
            self._run_dir,
            "keynote_grounding",
            {"counts": {"provider_calls": output.usage.calls}},
            completed_stage="keynote_grounding",
            dependencies=dependencies,
            output_paths=[path],
        )


class _TracingProvider:
    def __init__(self, provider: Provider) -> None:
        self._provider = provider
        self.traces: list[tuple[ProviderTrace, str]] = []

    def complete(self, request: ProviderRequest):
        try:
            result = self._provider.complete(request)
        except ProviderError as error:
            self.traces.append((error.trace, request.model))
            raise
        self.traces.append((result.trace, request.model))
        return result


class _ComponentNoveltyProvider:
    def __init__(self, provider: _TracingProvider, *, model: str) -> None:
        self._provider = provider
        self._model = model

    def execute(self, operation: ComponentNoveltyOperation) -> ComponentNoveltyProviderOutput:
        structured = operation.to_payload()
        request = ProviderRequest(
            operation=COMPONENT_NOVELTY_OPERATION,
            model=self._model,
            structured_input=structured,
            system_prompt=(
                "Judge candidate component novelty against only the supplied retrieved Core nodes. "
                "Return strict JSON with integer retrieval_similarity, perceived_novelty, and "
                "rubric_score in 0..5, plus rationale and provenance."
            ),
            user_prompt=json.dumps(structured, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            output_kind="json",
        )
        result = self._provider.complete(request)
        value = result.json_value
        if not isinstance(value, Mapping):
            raise ValueError("component novelty provider requires a JSON object")
        provenance = value.get("provenance")
        if not isinstance(provenance, Mapping) or not provenance:
            raise ValueError("component novelty provider requires provenance")
        return ComponentNoveltyProviderOutput(
            retrieval_similarity=value.get("retrieval_similarity"),  # type: ignore[arg-type]
            perceived_novelty=value.get("perceived_novelty"),  # type: ignore[arg-type]
            rubric_score=value.get("rubric_score"),  # type: ignore[arg-type]
            rationale=str(value.get("rationale") or ""),
            provenance_json=json.dumps(provenance, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )


class _SearchProvider:
    def __init__(self, provider: Provider, runtime: RuntimeConfig) -> None:
        self._generation_model = runtime.generation_model
        self._evaluation_model = runtime.evaluation_model
        self._provider = _TracingProvider(provider)
        self.generation = ProviderAdapter(self._provider, model=self._generation_model)
        self.evaluation = ProviderAdapter(self._provider, model=self._evaluation_model)

    def evaluate(
        self,
        state: IdeaState,
        *,
        idea_taste_mode: str,
        diagnostic: bool,
        context=None,
        prompt_mode=PRODUCTION_PROMPT_MODE,
    ):
        return self.evaluation.evaluate(
            state,
            idea_taste_mode=idea_taste_mode,
            prompt_mode=prompt_mode,
            diagnostic=diagnostic,
            context=context,
        )

    def generate(self, request):
        return self.generation.generate(request)

    def trace_records(self) -> list[dict[str, Any]]:
        return [_trace_record(trace, model) for trace, model in self._provider.traces]


def _search_config(
    runtime: RuntimeConfig,
    *,
    seed: int,
    evidence_digest: str,
) -> SearchConfig:
    return SearchConfig(
        max_iterations=runtime.max_iterations,
        max_depth=runtime.max_depth,
        branching_factor=runtime.branching_factor,
        exploration_constant=runtime.exploration_constant,
        max_children=runtime.max_children,
        max_evaluator_calls=runtime.max_evaluator_calls,
        max_tokens=runtime.max_tokens,
        max_cost=runtime.max_cost,
        seed=str(seed),
        runtime_profile=stable_signature(
            {
                "runtime_profile": RUNTIME_PROFILE_ID,
                "runtime_config_semantic_version": runtime.runtime_config_semantic_version,
                "runtime_config_semantic_signature": runtime.runtime_config_semantic_signature,
            }
        ),
        evaluator_profile=runtime.evaluation_model,
        evidence_digest=evidence_digest,
        component_novelty_required=True,
    )


class NativeModeSearch:
    def __init__(
        self,
        provider: Provider,
        runtime: RuntimeConfig,
        *,
        mode: str,
        seed: int,
        novelty_runtime: ComponentNoveltyRuntime,
        run_dir: Path | None = None,
    ) -> None:
        self.mode = mode
        self.seed = seed
        self._runtime = runtime
        self._provider = _SearchProvider(provider, runtime)
        self._novelty_evaluator = ComponentNoveltyEvaluator(
            novelty_runtime.retriever,
            _ComponentNoveltyProvider(
                self._provider._provider,
                model=runtime.evaluation_model,
            ),
            embedding_model=novelty_runtime.embedding_model,
            component_index=novelty_runtime.component_index,
        )
        self._operator_grounding_evaluator = OperatorGroundingEvaluator(
            self._provider.generation,
            novelty_runtime.retriever,
        )
        self._component_novelty_resource_digest = stable_signature(
            {
                "component_novelty_profile": COMPONENT_NOVELTY_PROFILE_ID,
                "operator_retrieval_profile": OPERATOR_RETRIEVAL_PROFILE_ID,
                "embedding_model": asdict(novelty_runtime.embedding_model),
                "component_index": asdict(novelty_runtime.component_index),
            }
        )
        self._run_dir = run_dir
        self._checkpoint_trace_records: tuple[dict[str, Any], ...] = ()
        self.result: SearchResult | None = None

    def run(self, request) -> ModeSearchOutput:
        if request.mode != self.mode or request.seed != self.seed:
            raise ValueError("search factory mode/seed mismatch")
        resumed = self._resumed_output(request)
        if resumed is not None:
            return resumed
        provider_context = _idea_provider_context(request.context)
        memory = MemoryState.from_experiment_feedback(request.context.get("experiment_feedback"))
        provider_context = IdeaProviderContext(
            evidence=provider_context.evidence,
            mature_idea=provider_context.mature_idea,
            refinement_scope=provider_context.refinement_scope,
            refinement_boundary=provider_context.refinement_boundary,
            memory_hints=memory.hints(),
        )
        engine = MCTSEngine(
            self._provider,
            _search_config(
                self._runtime,
                seed=request.seed,
                evidence_digest=request.evidence_signature,
            ),
            memory=memory,
            context=provider_context,
            novelty_evaluator=self._novelty_evaluator,
            operator_grounding_evaluator=self._operator_grounding_evaluator,
            component_novelty_resource_digest=self._component_novelty_resource_digest,
        )
        self.result = engine.search(_idea_state(request.root), request.mode)
        counters = self.result.counters
        trace = tuple(_candidate_trace(candidate, request.mode) for candidate in self.result.candidates)
        metadata = {
            **deepcopy(self.result.metadata),
            "workflow_seed": request.seed,
            "rng_seed": self.result.rng_seed,
            "stop_reason": self.result.stop_reason,
            "counters": asdict(counters),
            "champions": [_champion_record(value) for value in self.result.champions],
            "best_score": self.result.best.score,
        }
        output = ModeSearchOutput(
            mode=request.mode,
            candidate=self.result.best.state.to_payload(),
            evidence_ids=provider_context.evidence_ids,
            root_signature=request.root_signature,
            evidence_signature=request.evidence_signature,
            usage=BudgetUsage(
                iterations=counters.iterations,
                evaluator_calls=counters.evaluator_calls + counters.root_diagnostics,
                generation_calls=counters.generation_calls,
                input_tokens=counters.input_tokens,
                output_tokens=counters.output_tokens,
                cost=counters.cost,
            ),
            trace=trace,
            operator_attempts=self.result.operator_attempts,
            rollout_blockers=self.result.rollout_blockers,
            metadata=metadata,
        )
        self._write_checkpoint(request, output)
        return output

    def _checkpoint_path(self) -> Path | None:
        if self._run_dir is None:
            return None
        return run_paths(self._run_dir)[f"workflow_search_{self.mode}"]

    def _dependencies(self, request) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "seed": request.seed,
            "root_signature": request.root_signature,
            "evidence_signature": request.evidence_signature,
            "evidence_ids": list(request.evidence_ids),
            "algorithm_id": ALGORITHM_ID,
            "spec_version": SPEC_VERSION,
            "runtime_profile": RUNTIME_PROFILE_ID,
            "prompt_semantic_version": PROMPT_ROUTING_PROFILE_ID,
            "operator_semantic_version": OPERATOR_RETRIEVAL_PROFILE_ID,
            "search_config": {
                "max_iterations": self._runtime.max_iterations,
                "max_depth": self._runtime.max_depth,
                "branching_factor": self._runtime.branching_factor,
                "exploration_constant": self._runtime.exploration_constant,
                "max_children": self._runtime.max_children,
                "max_evaluator_calls": self._runtime.max_evaluator_calls,
                "max_tokens": self._runtime.max_tokens,
                "max_cost": self._runtime.max_cost,
                "runtime_config_semantic_version": self._runtime.runtime_config_semantic_version,
                "runtime_config_semantic_signature": self._runtime.runtime_config_semantic_signature,
                "generation_model": self._runtime.generation_model,
                "evaluation_model": self._runtime.evaluation_model,
            },
        }

    def _resumed_output(self, request) -> ModeSearchOutput | None:
        path = self._checkpoint_path()
        if path is None or not stage_completed(
            self._run_dir,
            f"search.{self.mode}",
            [path],
            dependencies=self._dependencies(request),
        ):
            return None
        value = read_json(path)
        if not isinstance(value, Mapping):
            return None
        output = _mode_output_from_dict(value)
        metadata = dict(output.metadata)
        metadata["resumed"] = True
        trace_records = value.get("provider_trace_provenance")
        if not isinstance(trace_records, list) or any(not isinstance(item, Mapping) for item in trace_records):
            return None
        self._checkpoint_trace_records = tuple(deepcopy(dict(item)) for item in trace_records)
        return ModeSearchOutput(
            mode=output.mode,
            candidate=output.candidate,
            evidence_ids=output.evidence_ids,
            root_signature=output.root_signature,
            evidence_signature=output.evidence_signature,
            usage=output.usage,
            trace=output.trace,
            operator_attempts=output.operator_attempts,
            rollout_blockers=output.rollout_blockers,
            metadata=metadata,
        )

    def _write_checkpoint(self, request, output: ModeSearchOutput) -> None:
        path = self._checkpoint_path()
        if path is None:
            return
        value = _mode_output_dict(output)
        value["provider_trace_provenance"] = self._provider.trace_records()
        write_portable_json(path, value)
        write_checkpoint(
            self._run_dir,
            f"search.{self.mode}",
            {
                "counts": {"iterations": output.usage.iterations},
                "mode": self.mode,
                "stop_reason": output.metadata.get("stop_reason"),
            },
            completed_stage=f"search.{self.mode}",
            dependencies=self._dependencies(request),
            output_paths=[path],
        )

    def trace_records(self) -> list[dict[str, Any]]:
        if self._checkpoint_trace_records:
            return deepcopy(list(self._checkpoint_trace_records))
        return self._provider.trace_records()


class NativeSearchFactory:
    def __init__(
        self,
        provider: Provider,
        runtime: RuntimeConfig,
        *,
        novelty_runtime: ComponentNoveltyRuntime,
        run_dir: Path | None = None,
    ) -> None:
        self._provider = provider
        self._runtime = runtime
        self._novelty_runtime = novelty_runtime
        self._run_dir = run_dir
        self.runners: list[NativeModeSearch] = []

    def create(self, *, mode: str, seed: int) -> NativeModeSearch:
        runner = NativeModeSearch(
            self._provider,
            self._runtime,
            mode=mode,
            seed=seed,
            novelty_runtime=self._novelty_runtime,
            run_dir=self._run_dir,
        )
        self.runners.append(runner)
        return runner

    def trace_records(self) -> list[dict[str, Any]]:
        return [record for runner in self.runners for record in runner.trace_records()]


class NativeFusionRunner:
    def __init__(self, provider: Provider, runtime: RuntimeConfig) -> None:
        self._fusion_model = runtime.fusion_model
        self._evaluation_model = runtime.evaluation_model
        self._provider = _TracingProvider(provider)
        self._generator = ProviderAdapter(self._provider, model=self._fusion_model)
        self._evaluator = ProviderAdapter(self._provider, model=self._evaluation_model)
        self.result = None

    def run(self, request) -> FusionWorkflowOutput:
        generator_before = self._generator.usage
        evaluator_before = self._evaluator.usage
        context = dict(request.context)
        refinement_scope = tuple(
            str(value) for value in context.get("refinement_scope", []) if str(value).strip()
        )
        raw_boundary = context.get("refinement_boundary")
        refinement_boundary = (
            RefinementBoundary.from_payload(raw_boundary)
            if isinstance(raw_boundary, Mapping)
            else None
        )
        self.result = fuse_five_modes(
            FusionRequest(
                mode_inputs=request.mode_inputs,
                topic=str(context.get("topic") or ""),
                context=context,
                refinement_scope=refinement_scope,
                refinement_boundary=refinement_boundary,
            ),
            generator=self._generator,
            evaluator=self._evaluator,
            repair_generator=self._generator,
        )
        generator_usage = _usage_delta(self._generator.usage, generator_before)
        evaluator_usage = _usage_delta(self._evaluator.usage, evaluator_before)
        usage = BudgetUsage(
            calls=generator_usage[0] + evaluator_usage[0],
            input_tokens=generator_usage[1] + evaluator_usage[1],
            output_tokens=generator_usage[2] + evaluator_usage[2],
        )
        metadata = {
            **deepcopy(dict(self.result.metadata)),
            "selected_components": deepcopy(list(self.result.selected_components)),
            "rejected_components": deepcopy(list(self.result.rejected_components)),
            "conflict_resolutions": deepcopy(list(self.result.conflict_resolutions)),
            "evaluation": self.result.evaluation.to_dict(),
        }
        return FusionWorkflowOutput(
            idea=deepcopy(dict(self.result.idea)),
            source_modes=tuple(self.result.source_modes),
            evidence_ids=request.evidence_ids,
            root_signature=request.root_signature,
            evidence_signature=request.evidence_signature,
            usage=usage,
            trace=deepcopy(self.result.evolution),
            metadata=metadata,
        )

    def trace_records(self) -> list[dict[str, Any]]:
        return [_trace_record(trace, model) for trace, model in self._provider.traces]


def run_native_workflow(
    *,
    run_id: str,
    request: IdeaRequest,
    runtime: RuntimeConfig,
    repository: SurveyArtifactRepository,
    source_context: Mapping[str, Any],
    provider: Provider,
    run_dir: Path | None = None,
    novelty_runtime: ComponentNoveltyRuntime | None = None,
) -> WorkflowResult:
    """Run and compatibility-materialize the authoritative package-native workflow."""

    retrieval = SurveyRepositoryRetrieval(
        repository,
        limit=int(source_context.get("selection_limit") or 12),
    )
    keynote_grounder = NativeKeynoteGrounder(
        provider,
        model=runtime.agent_model,
        run_dir=run_dir,
    )
    workflow_provider = GenericWorkflowProvider(provider, runtime)
    novelty_runtime = novelty_runtime or build_component_novelty_runtime(repository.source.resources)
    search_factory = NativeSearchFactory(
        provider,
        runtime,
        novelty_runtime=novelty_runtime,
        run_dir=run_dir,
    )
    fusion_runner = NativeFusionRunner(provider, runtime)
    mature_idea = _mature_idea(request.mature_idea, request.refinement_scope)
    refinement_scope = (request.refinement_scope.strip(),) if request.refinement_scope.strip() else ()
    feedback_evidence: tuple[Evidence, ...] = ()
    feedback_resources: tuple[str, ...] = ()
    feedback_retrieval_metadata: Mapping[str, Any] = {}
    feedback_rag_query: str | None = None
    feedback_selected_hits: tuple[OutcomeHit, ...] = ()
    feedback_citations: CitationRegistry | None = None
    if request.experiment_feedback:
        feedback_rag_query = request.topic or repository.topic
        feedback = retrieval.snapshot(feedback_rag_query)
        feedback_evidence = feedback.evidence
        feedback_resources = feedback.resource_ids
        feedback_retrieval_metadata = feedback.metadata
        feedback_selected_hits = feedback.selected_outcome_hits
        feedback_citations = feedback.citations

    workflow_source_context = deepcopy(dict(source_context))
    if request.discussion:
        workflow_source_context["discussion"] = request.discussion
    result = run_workflow(
        WorkflowRequest(
            run_id=run_id,
            topic=request.topic or repository.topic,
            source_context=workflow_source_context,
            seed=run_id,
            mature_idea=mature_idea,
            refinement_scope=refinement_scope,
            experiment_feedback=request.experiment_feedback or None,
            evidence=feedback_evidence,
            resource_ids=feedback_resources,
            rag_query=feedback_rag_query,
            selected_outcome_hits=feedback_selected_hits,
            citations=feedback_citations,
            retrieval_metadata=feedback_retrieval_metadata,
        ),
        search_factory=search_factory,
        fusion_runner=fusion_runner,
        provider=workflow_provider,
        retrieval=retrieval,
        keynote_grounder=keynote_grounder,
    )
    artifact = deepcopy(dict(result.artifact))
    run = artifact.get("run") if isinstance(artifact.get("run"), dict) else {}
    run["operation_trace"] = [
        *run.get("operation_trace", []),
        *search_factory.trace_records(),
        *fusion_runner.trace_records(),
    ]
    run["native_workflow_trace"] = deepcopy(run.get("workflow_trace", []))
    if result.succeeded:
        run["workflow_trace"] = _public_workflow_trace(bool(request.experiment_feedback))
        _materialize_compatibility(artifact, search_factory, fusion_runner)
        try:
            _validate_complete_result(artifact)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            run.update({"status": "incomplete", "error": message})
            persistence = artifact.get("persistence")
            if isinstance(persistence, dict):
                persistence.pop("idea_result", None)
                persistence.update({"status": "incomplete", "incomplete_reason": message})
            return WorkflowResult("incomplete", artifact, message)
    return WorkflowResult(result.status, artifact, result.error)


def _mode_output_dict(output: ModeSearchOutput) -> dict[str, Any]:
    return {
        "mode": output.mode,
        "candidate": deepcopy(dict(output.candidate)),
        "evidence_ids": list(output.evidence_ids),
        "root_signature": output.root_signature,
        "evidence_signature": output.evidence_signature,
        "usage": output.usage.to_dict(),
        "trace": deepcopy(list(output.trace)),
        "operator_attempts": [item.to_payload() for item in output.operator_attempts],
        "rollout_blockers": [item.to_payload() for item in output.rollout_blockers],
        "metadata": deepcopy(dict(output.metadata)),
    }


def _mode_output_from_dict(value: Mapping[str, Any]) -> ModeSearchOutput:
    usage = value.get("usage")
    attempts = value.get("operator_attempts", [])
    blockers = value.get("rollout_blockers", [])
    if not isinstance(usage, Mapping):
        raise ValueError("search checkpoint usage must be a mapping")
    if (
        not isinstance(attempts, list)
        or any(not isinstance(item, Mapping) for item in attempts)
        or not isinstance(blockers, list)
        or any(not isinstance(item, Mapping) for item in blockers)
    ):
        raise ValueError("search checkpoint rollout records are malformed")
    return ModeSearchOutput(
        mode=str(value.get("mode") or ""),
        candidate=deepcopy(dict(value.get("candidate"))) if isinstance(value.get("candidate"), Mapping) else {},
        evidence_ids=tuple(str(item) for item in value.get("evidence_ids", [])),
        root_signature=str(value.get("root_signature") or ""),
        evidence_signature=str(value.get("evidence_signature") or ""),
        usage=BudgetUsage(
            calls=int(usage.get("calls") or 0),
            iterations=int(usage.get("iterations") or 0),
            evaluator_calls=int(usage.get("evaluator_calls") or 0),
            generation_calls=int(usage.get("generation_calls") or 0),
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cost=float(usage.get("cost") or 0),
        ),
        trace=tuple(deepcopy(value.get("trace", []))) if isinstance(value.get("trace"), list) else (),
        operator_attempts=tuple(OperatorAttemptRecord.from_payload(item) for item in attempts),
        rollout_blockers=tuple(RolloutBlockerRecord.from_payload(item) for item in blockers),
        metadata=deepcopy(dict(value.get("metadata"))) if isinstance(value.get("metadata"), Mapping) else {},
    )


def _materialize_compatibility(
    artifact: dict[str, Any],
    search_factory: NativeSearchFactory,
    fusion_runner: NativeFusionRunner,
) -> None:
    ideation = artifact["ideation"]
    persistence = artifact["persistence"]
    idea_result = persistence["idea_result"]
    mode_candidates = ideation.get("mode_candidates", [])
    iterations = [
        deepcopy(item)
        for candidate in mode_candidates
        if isinstance(candidate, Mapping)
        for item in candidate.get("trace", [])
        if isinstance(item, Mapping)
    ]
    pareto_front = {
        candidate["mode"]: {
            "title": candidate["idea"].get("title"),
            "score": candidate.get("metadata", {}).get("best_score"),
            "champions": deepcopy(candidate.get("metadata", {}).get("champions", [])),
        }
        for candidate in mode_candidates
        if isinstance(candidate, Mapping)
    }
    counters = {
        candidate["mode"]: deepcopy(candidate.get("metadata", {}).get("counters", {}))
        for candidate in mode_candidates
        if isinstance(candidate, Mapping)
    }
    stop_reasons = {
        candidate["mode"]: candidate.get("metadata", {}).get("stop_reason")
        for candidate in mode_candidates
        if isinstance(candidate, Mapping)
    }
    seeds = {
        candidate["mode"]: {
            "workflow_seed": candidate.get("metadata", {}).get("workflow_seed"),
            "rng_seed": candidate.get("metadata", {}).get("rng_seed"),
        }
        for candidate in mode_candidates
        if isinstance(candidate, Mapping)
    }
    total_iterations = sum(int(value.get("iterations") or 0) for value in counters.values())
    fusion = ideation.get("fusion_result", {})
    fusion_metadata = deepcopy(fusion.get("metadata", {})) if isinstance(fusion, Mapping) else {}
    ranked_candidates = sorted(
        [
            {
                "mode": candidate.get("mode"),
                "title": candidate.get("idea", {}).get("title"),
                "score": candidate.get("metadata", {}).get("best_score"),
            }
            for candidate in mode_candidates
            if isinstance(candidate, Mapping)
        ],
        key=lambda value: (-float(value.get("score") or 0), CANONICAL_MODES.index(value["mode"])),
    )
    fusion_evolution = {
        "source_modes": list(CANONICAL_MODES),
        "selected_components": deepcopy(fusion_metadata.get("selected_components", [])),
        "rejected_components": deepcopy(fusion_metadata.get("rejected_components", [])),
        "conflict_resolutions": deepcopy(fusion_metadata.get("conflict_resolutions", [])),
        "events": deepcopy(fusion.get("trace", [])) if isinstance(fusion, Mapping) else [],
        "ranked_candidates": ranked_candidates,
    }
    idea_result.update(
        {
            "mcts_evolution": {
                "total_iterations": total_iterations,
                "iterations": iterations,
                "pareto_front": pareto_front,
                "counters": counters,
                "stop_reasons": stop_reasons,
                "seeds": seeds,
            },
            "idea_source": "fused",
            "source_modes": list(CANONICAL_MODES),
            "fusion_metadata": fusion_metadata,
            "fusion_evolution": fusion_evolution,
            "algorithm_provenance": {
                "algorithm": ALGORITHM_ID,
                "algorithm_id": ALGORITHM_ID,
                "algorithm_spec": ALGORITHM_ID,
                "spec_version": ALGORITHM_SPEC_VERSION,
                "runtime": "package-native",
                "runtime_profile": "xlab.research_idea.runtime.v1",
                "evidence_profile": "xlab.research_idea.evidence.v1",
                "success_profile": "xlab.research_idea.success.v1",
                "native_reconstruction": True,
                "required_modes": list(CANONICAL_MODES),
                "completed_modes": list(CANONICAL_MODES),
                "all_modes_completed": True,
                "fusion_required": True,
                "fusion_used": True,
                "fusion_metadata_present": bool(fusion_metadata),
                "mcts_iterations": total_iterations,
                "pareto_front_present": bool(pareto_front),
                "search_trace_present": bool(iterations),
            },
        }
    )


def _validate_complete_result(artifact: Mapping[str, Any]) -> None:
    persistence = artifact.get("persistence")
    result = persistence.get("idea_result") if isinstance(persistence, Mapping) else None
    if not isinstance(result, Mapping):
        raise ValueError("native workflow did not persist idea_result")
    missing_text = [name for name in _REQUIRED_RESULT_TEXT if not str(result.get(name) or "").strip()]
    missing_lists = [
        name for name in _REQUIRED_RESULT_LISTS
        if not isinstance(result.get(name), list) or not result.get(name)
    ]
    if missing_text or missing_lists:
        raise ValueError(
            "native idea_result is incomplete: " + ", ".join([*missing_text, *missing_lists])
        )
    evolution = result.get("mcts_evolution")
    if not isinstance(evolution, Mapping) or int(evolution.get("total_iterations") or 0) <= 0:
        raise ValueError("native MCTS did not complete any search iterations")


def _workflow_prompt(operation: str, value: Mapping[str, JsonValue]) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if operation == OP_BACKGROUND:
        return TOPIC_BACKGROUND_PROMPT.format(topic=value.get("topic", ""))
    if operation == OP_QUERY:
        background = value.get("background")
        mature = value.get("mature_idea")
        scope = value.get("refinement_scope")
        return RAG_QUERY_PROMPT.format(
            topic=value.get("topic", ""),
            mature_idea=json.dumps(mature or background or {}, ensure_ascii=False),
            refinement_scope=json.dumps(scope or [], ensure_ascii=False),
        )
    if operation == OP_ANALYSIS:
        return ADVANCED_ANALYSIS_PROMPT.format(
            topic=value.get("topic", ""),
            mature_idea=json.dumps(value.get("mature_idea", {}), ensure_ascii=False),
            mature_idea_source="input_explicit" if value.get("mature_idea") else "empty",
            refinement_scope=json.dumps(value.get("refinement_scope", []), ensure_ascii=False),
            refinement_scope_source="input_explicit" if value.get("refinement_scope") else "empty",
            survey_contents=json.dumps(
                value.get("evidence", value.get("background", value.get("source_context", {}))),
                ensure_ascii=False,
            ),
            papers=json.dumps(value.get("evidence", []), ensure_ascii=False),
            experiment_findings=json.dumps(value.get("experiment_feedback"), ensure_ascii=False),
        ) + "\nProvider input and required contract:\n" + data
    if operation == OP_REPLAN:
        return RE_ANALYSIS_REPLAN_PROMPT.format(
            topic=value.get("topic", ""),
            mature_idea=json.dumps(value.get("mature_idea", {}), ensure_ascii=False),
            refinement_scope=json.dumps(value.get("refinement_scope", []), ensure_ascii=False),
            analysis=json.dumps(value.get("analysis", {}), ensure_ascii=False),
            ablation_results=json.dumps(value.get("experiment_feedback"), ensure_ascii=False),
        ) + "\nWrap the response in a top-level replan object and include replan.root_idea as a complete typed idea.\nProvider input:\n" + data
    if operation == OP_MATERIALIZATION:
        return IDEA_RESULT_ALIGNMENT_PROMPT.format(
            topic=value.get("topic", ""),
            mature_idea=json.dumps(value.get("mature_idea"), ensure_ascii=False),
            refinement_scope=json.dumps(value.get("refinement_scope", []), ensure_ascii=False),
            idea=json.dumps(value.get("fusion", {}), ensure_ascii=False),
            papers=json.dumps(value.get("evidence", value.get("evidence_ids", [])), ensure_ascii=False),
        ) + "\nWrap the response in idea_result and preserve the complete materialization contract in provider input:\n" + data
    if operation == OP_RANKING:
        return (
            "Return strict JSON only as evidence_ids containing every supplied evidence_id exactly once, "
            "ranked most relevant first. Do not invent IDs. Provider input:\n" + data
        )
    raise ValueError(f"unsupported workflow prompt operation: {operation}")


def _idea_provider_context(value: Mapping[str, Any]) -> IdeaProviderContext:
    raw_evidence = value.get("evidence")
    if not isinstance(raw_evidence, list):
        raise ValueError("search context evidence must be a list")
    evidence = tuple(
        RankedEvidence.from_payload(item, rank=index)
        for index, item in enumerate(raw_evidence, start=1)
        if isinstance(item, Mapping)
    )
    if len(evidence) != len(raw_evidence):
        raise ValueError("search context evidence must contain only mappings")
    mature_payload = value.get("mature_idea_payload")
    mature = _idea_state(mature_payload) if isinstance(mature_payload, Mapping) else None
    refinement_scope = _string_tuple(value.get("refinement_scope", []))
    raw_boundary = value.get("refinement_boundary")
    boundary = (
        RefinementBoundary.from_payload(raw_boundary)
        if isinstance(raw_boundary, Mapping)
        else None
    )
    return IdeaProviderContext(
        evidence=evidence,
        mature_idea=mature,
        refinement_scope=refinement_scope,
        refinement_boundary=boundary,
    )


def _idea_state(value: Mapping[str, Any]) -> IdeaState:
    components = value.get("components")
    if not isinstance(components, Sequence) or isinstance(components, (str, bytes)):
        raise ValueError("search root components must be a sequence")
    typed_components = []
    for component in components:
        if isinstance(component, Mapping):
            name = str(component.get("name") or component.get("component") or "").strip()
            description = str(component.get("description") or component.get("explanation") or "").strip()
        else:
            name = str(component).strip()
            description = ""
        if not name or not description:
            raise ValueError("search root components require names and descriptions")
        typed_components.append(IdeaComponent(name, description))
    return IdeaState(
        title=_required_text(value, "title"),
        abstract=_required_text(value, "abstract"),
        core_contribution=_required_text(value, "core_contribution"),
        method=_required_text(value, "method"),
        risks=_required_text(value, "risks"),
        components=tuple(typed_components),
        tags=_string_tuple(value.get("tags", [])),
        root_domains=_string_tuple(value.get("root_domains", [])),
    )


def _mature_idea(value: str, refinement_scope: str) -> Mapping[str, Any] | None:
    text = value.strip()
    if not text:
        return None
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, Mapping):
        return deepcopy(dict(decoded))
    del refinement_scope
    component = "mature_core"
    return {
        "title": text.split(".", 1)[0][:160],
        "abstract": text,
        "core_contribution": text,
        "method": text,
        "risks": "The supplied mature idea requires evidence-grounded validation.",
        "components": [{"name": component, "description": text}],
        "tags": [],
        "root_domains": [],
    }


def _candidate_trace(candidate, mode: str) -> dict[str, Any]:
    incoming = candidate.node_id
    return {
        "iteration": candidate.node_id,
        "node_id": candidate.node_id,
        "title": candidate.state.title,
        "score": candidate.score,
        "source": candidate.source,
        "idea_taste_mode": mode,
        "evaluation": {
            **dict(candidate.evaluation.metrics),
            "confidence": candidate.evaluation.confidence,
            "detected_defects": list(candidate.evaluation.detected_defects),
            "feedback": candidate.evaluation.feedback,
        },
        "cache_identity": candidate.cache_identity.to_metadata(),
        "path": ["root"] if incoming == 0 else ["root", str(incoming)],
    }


def _champion_record(champion) -> dict[str, Any]:
    return {
        "label": champion.label,
        "metric": champion.metric,
        "objective": champion.objective,
        "node_id": champion.candidate.node_id,
        "title": champion.candidate.state.title,
        "score": champion.candidate.score,
        "metric_value": champion.candidate.evaluation.metrics[champion.metric],
        "strategy": champion.strategy,
    }


def _trace_record(trace: ProviderTrace, model: str) -> dict[str, Any]:
    return {
        "event": "llm_call",
        "op_name": _PROVIDER_OPERATION_NAMES.get(trace.operation, trace.operation),
        "provider_operation": trace.operation,
        "status": trace.status,
        "input_digest": trace.input_digest,
        "attempts": trace.attempts,
        "error_code": trace.error_code,
        "model": model,
    }


def _public_workflow_trace(feedback: bool) -> list[dict[str, Any]]:
    stages = (
        ("advanced_analysis", "re_analysis_replan", "idea_generation")
        if feedback
        else ("knowledge_acquisition", "advanced_analysis", "idea_generation")
    )
    return [
        {"workflow": WORKFLOW_ID, "stage": stage, "status": "success"}
        for stage in stages
    ]


def _usage_delta(after, before) -> tuple[int, int, int]:
    return (
        after.provider_calls - before.provider_calls,
        after.input_tokens - before.input_tokens,
        after.output_tokens - before.output_tokens,
    )


def _required_text(value: Mapping[str, Any], field: str) -> str:
    text = str(value.get(field) or "").strip()
    if not text:
        raise ValueError(f"search root requires {field}")
    return text


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("idea tags and root_domains must be lists")
    return tuple(str(item).strip() for item in value if str(item).strip())


def _json_mapping(value: Mapping[str, Any], label: str) -> dict[str, JsonValue]:
    try:
        decoded = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} input must contain only JSON values") from error
    if not isinstance(decoded, dict):
        raise TypeError(f"{label} input must be a mapping")
    return decoded
