"""Candidate-specific component novelty retrieval and evaluation.

The defaults mirror Xcientist's component novelty scorer.  Deliberate XLab
corrections are strict failure semantics, explanation-only queries, and a
required integer rubric score instead of score clamping or fallback fields.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Protocol, Sequence

from ..research_idea_spec import COMPONENT_NOVELTY_PROFILE_ID
from .contracts import IdeaState
from .evaluation import Evaluation, EvaluationPostprocessing
from .tastes import Taste

COMPONENT_NOVELTY_OPERATION = "xlab.research_idea.component_novelty.evaluate.v1"
UPSTREAM_RETRIEVAL_TOP_K = 50
UPSTREAM_EVIDENCE_TOP_K = 5
UPSTREAM_SUPPORT_TOP_K = 3
UPSTREAM_TEXT_LIMIT = 6000


class ComponentNoveltyError(RuntimeError):
    """Component novelty could not be verified for a candidate."""


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _canonical_provenance(value: str, label: str) -> str:
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid JSON") from error
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{label} must be a non-empty JSON object")
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _score(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 5:
        raise ValueError(f"{label} must be an integer between 0 and 5")
    return value


def _clip(value: str, limit: int) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 3)] + "..."


@dataclass(frozen=True)
class DeclaredNoveltyResource:
    """Immutable identity of a manifest-declared model or component index."""

    resource_id: str
    digest: str
    algorithm_version: str
    logical_uri: str

    def __post_init__(self) -> None:
        for field in ("resource_id", "digest", "algorithm_version", "logical_uri"):
            object.__setattr__(self, field, _required_text(getattr(self, field), field))

    def to_payload(self) -> dict[str, str]:
        return {
            "resource_id": self.resource_id,
            "digest": self.digest,
            "algorithm_version": self.algorithm_version,
            "logical_uri": self.logical_uri,
        }


@dataclass(frozen=True)
class ComponentQuery:
    query_id: str
    component_name: str
    explanation: str
    query_text: str

    def __post_init__(self) -> None:
        for field in ("query_id", "component_name", "explanation", "query_text"):
            object.__setattr__(self, field, _required_text(getattr(self, field), field))

    def to_payload(self) -> dict[str, str]:
        return {
            "query_id": self.query_id,
            "component_name": self.component_name,
            "explanation": self.explanation,
            "query_text": self.query_text,
        }


@dataclass(frozen=True)
class CoreNode:
    """Immutable paper-graph Core node returned by the declared FAISS index."""

    evidence_id: str
    node_id: str
    label: str
    paper_title: str
    summary: str
    insight: str
    provenance_json: str

    def __post_init__(self) -> None:
        for field in ("evidence_id", "node_id", "label"):
            object.__setattr__(self, field, _required_text(getattr(self, field), field))
        object.__setattr__(
            self,
            "provenance_json",
            _canonical_provenance(self.provenance_json, "core node provenance_json"),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "node_id": self.node_id,
            "label": self.label,
            "paper_title": self.paper_title,
            "summary": self.summary,
            "insight": self.insight,
            "provenance": json.loads(self.provenance_json),
        }


@dataclass(frozen=True)
class ComponentHit:
    record_id: str
    node: CoreNode
    matched_component: str
    similarity: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "record_id", _required_text(self.record_id, "record_id")
        )
        object.__setattr__(
            self,
            "matched_component",
            _required_text(self.matched_component, "matched_component"),
        )
        object.__setattr__(
            self, "similarity", _finite_number(self.similarity, "similarity")
        )


@dataclass(frozen=True)
class ComponentRetrievalRequest:
    candidate_id: str
    candidate_textual_identity: str
    idea_taste_mode: str
    query: ComponentQuery
    top_k: int
    embedding_model: DeclaredNoveltyResource
    component_index: DeclaredNoveltyResource

    def __post_init__(self) -> None:
        for field in ("candidate_id", "candidate_textual_identity", "idea_taste_mode"):
            object.__setattr__(self, field, _required_text(getattr(self, field), field))
        if not isinstance(self.query, ComponentQuery):
            raise ValueError("query must be a ComponentQuery")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError("top_k must be a positive integer")

    @property
    def request_identity(self) -> str:
        payload = {
            "candidate_id": self.candidate_id,
            "candidate_textual_identity": self.candidate_textual_identity,
            "idea_taste_mode": self.idea_taste_mode,
            "query": self.query.to_payload(),
            "top_k": self.top_k,
            "embedding_model": self.embedding_model.to_payload(),
            "component_index": self.component_index.to_payload(),
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return f"component-retrieval:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True)
class ComponentRetrievalOutput:
    request_identity: str
    candidate_id: str
    candidate_textual_identity: str
    idea_taste_mode: str
    query_id: str
    hits: tuple[ComponentHit, ...]
    embedding_model: DeclaredNoveltyResource
    component_index: DeclaredNoveltyResource
    native_embedding_inference: bool
    native_faiss_search: bool
    provenance_json: str

    def __post_init__(self) -> None:
        for field in (
            "request_identity",
            "candidate_id",
            "candidate_textual_identity",
            "idea_taste_mode",
            "query_id",
        ):
            object.__setattr__(self, field, _required_text(getattr(self, field), field))
        if not isinstance(self.hits, tuple) or any(
            not isinstance(hit, ComponentHit) for hit in self.hits
        ):
            raise ValueError("hits must be a tuple of ComponentHit values")
        object.__setattr__(
            self,
            "provenance_json",
            _canonical_provenance(self.provenance_json, "retrieval provenance_json"),
        )


class ComponentRetriever(Protocol):
    """Adapter boundary for the declared embedding model and FAISS resource."""

    def retrieve(
        self, request: ComponentRetrievalRequest
    ) -> ComponentRetrievalOutput: ...


@dataclass(frozen=True)
class EvidenceSupport:
    query_id: str
    query_component: str
    query_text: str
    retrieval_record_id: str
    matched_component: str
    similarity: float

    def to_payload(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "query_component": self.query_component,
            "query_text": self.query_text,
            "retrieval_record_id": self.retrieval_record_id,
            "matched_component": self.matched_component,
            "similarity": self.similarity,
        }


@dataclass(frozen=True)
class NoveltyEvidence:
    evidence_id: str
    node_id: str
    label: str
    paper_title: str
    match_count: int
    best_similarity: float
    summary: str
    insight: str
    support: tuple[EvidenceSupport, ...]
    provenance_json: str

    def to_payload(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "node_id": self.node_id,
            "label": self.label,
            "paper_title": self.paper_title,
            "match_count": self.match_count,
            "best_similarity": self.best_similarity,
            "summary": self.summary,
            "insight": self.insight,
            "support": [item.to_payload() for item in self.support],
            "provenance": json.loads(self.provenance_json),
        }


@dataclass(frozen=True)
class ComponentNoveltyOperation:
    name: str
    candidate_id: str
    candidate_textual_identity: str
    idea_taste_mode: str
    topic: str
    idea_json: str
    queries: tuple[ComponentQuery, ...]
    evidence: tuple[NoveltyEvidence, ...]
    retrieval_top_k: int
    embedding_model: DeclaredNoveltyResource
    component_index: DeclaredNoveltyResource

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.evidence)

    def to_payload(self) -> dict[str, object]:
        return {
            "algorithm": COMPONENT_NOVELTY_PROFILE_ID,
            "operation": self.name,
            "candidate_id": self.candidate_id,
            "candidate_textual_identity": self.candidate_textual_identity,
            "idea_taste_mode": self.idea_taste_mode,
            "topic": self.topic,
            "idea": json.loads(self.idea_json),
            "components_with_explanations": [
                item.to_payload() for item in self.queries
            ],
            "retrieval_top_k": self.retrieval_top_k,
            "retrieved_nodes": [item.to_payload() for item in self.evidence],
            "evidence_ids": list(self.evidence_ids),
            "embedding_model": self.embedding_model.to_payload(),
            "component_index": self.component_index.to_payload(),
        }


@dataclass(frozen=True)
class ComponentNoveltyProviderOutput:
    retrieval_similarity: int
    perceived_novelty: int
    rubric_score: int
    rationale: str
    provenance_json: str

    def __post_init__(self) -> None:
        for field in ("retrieval_similarity", "perceived_novelty", "rubric_score"):
            object.__setattr__(self, field, _score(getattr(self, field), field))
        object.__setattr__(
            self, "rationale", _required_text(self.rationale, "rationale")
        )
        object.__setattr__(
            self,
            "provenance_json",
            _canonical_provenance(self.provenance_json, "provider provenance_json"),
        )


class ComponentNoveltyProvider(Protocol):
    """Provider boundary for the explicit 0..5 novelty judgment operation."""

    def execute(
        self, operation: ComponentNoveltyOperation
    ) -> ComponentNoveltyProviderOutput: ...


@dataclass(frozen=True)
class ComponentNoveltyResult:
    candidate_id: str
    candidate_textual_identity: str
    idea_taste_mode: str
    score: int
    judgment: ComponentNoveltyProviderOutput
    evidence: tuple[NoveltyEvidence, ...]
    retrieval_provenance_json: tuple[str, ...]
    embedding_model: DeclaredNoveltyResource
    component_index: DeclaredNoveltyResource

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.evidence)


@dataclass(frozen=True)
class NoveltyAdjustedEvaluation:
    """Evaluation after deterministic novelty replacement and rescoring."""

    evaluation: Evaluation
    composite: float
    novelty: ComponentNoveltyResult


class ComponentNoveltyEvaluator:
    """Strict candidate-specific retrieval and provider evaluation plumbing."""

    def __init__(
        self,
        retriever: ComponentRetriever,
        provider: ComponentNoveltyProvider,
        *,
        embedding_model: DeclaredNoveltyResource,
        component_index: DeclaredNoveltyResource,
        retrieval_top_k: int = UPSTREAM_RETRIEVAL_TOP_K,
        evidence_top_k: int = UPSTREAM_EVIDENCE_TOP_K,
        support_top_k: int = UPSTREAM_SUPPORT_TOP_K,
        text_limit: int = UPSTREAM_TEXT_LIMIT,
    ) -> None:
        for label, value in (
            ("retrieval_top_k", retrieval_top_k),
            ("evidence_top_k", evidence_top_k),
            ("support_top_k", support_top_k),
            ("text_limit", text_limit),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        self._retriever = retriever
        self._provider = provider
        self._embedding_model = embedding_model
        self._component_index = component_index
        self._retrieval_top_k = retrieval_top_k
        self._evidence_top_k = evidence_top_k
        self._support_top_k = support_top_k
        self._text_limit = text_limit

    def evaluate(
        self,
        state: IdeaState,
        *,
        candidate_id: str,
        idea_taste_mode: str,
        topic: str = "",
    ) -> ComponentNoveltyResult:
        """Evaluate one candidate or raise; unverifiable novelty never passes through."""

        candidate_id = _required_text(candidate_id, "candidate_id")
        idea_taste_mode = _required_text(idea_taste_mode, "idea_taste_mode")
        queries = self._queries(
            state,
            candidate_id=candidate_id,
            idea_taste_mode=idea_taste_mode,
        )
        outputs: list[ComponentRetrievalOutput] = []
        try:
            for query in queries:
                request = ComponentRetrievalRequest(
                    candidate_id=candidate_id,
                    candidate_textual_identity=state.textual_identity,
                    idea_taste_mode=idea_taste_mode,
                    query=query,
                    top_k=self._retrieval_top_k,
                    embedding_model=self._embedding_model,
                    component_index=self._component_index,
                )
                output = self._retriever.retrieve(request)
                self._validate_retrieval(request, output)
                outputs.append(output)
            evidence = self._aggregate(queries, outputs)
            if not evidence:
                raise ComponentNoveltyError(
                    "component retrieval returned no Core-node evidence"
                )
            operation = ComponentNoveltyOperation(
                name=COMPONENT_NOVELTY_OPERATION,
                candidate_id=candidate_id,
                candidate_textual_identity=state.textual_identity,
                idea_taste_mode=idea_taste_mode,
                topic=topic.strip() or state.title,
                idea_json=json.dumps(
                    state.to_payload(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                queries=queries,
                evidence=evidence,
                retrieval_top_k=self._retrieval_top_k,
                embedding_model=self._embedding_model,
                component_index=self._component_index,
            )
            judgment = self._provider.execute(operation)
            if not isinstance(judgment, ComponentNoveltyProviderOutput):
                raise ComponentNoveltyError(
                    "component novelty provider must return ComponentNoveltyProviderOutput"
                )
        except ComponentNoveltyError:
            raise
        except Exception as error:
            raise ComponentNoveltyError(
                f"candidate {candidate_id!r} component novelty verification failed: {error}"
            ) from error
        return ComponentNoveltyResult(
            candidate_id=candidate_id,
            candidate_textual_identity=state.textual_identity,
            idea_taste_mode=idea_taste_mode,
            score=judgment.rubric_score,
            judgment=judgment,
            evidence=evidence,
            retrieval_provenance_json=tuple(item.provenance_json for item in outputs),
            embedding_model=self._embedding_model,
            component_index=self._component_index,
        )

    def _queries(
        self,
        state: IdeaState,
        *,
        candidate_id: str,
        idea_taste_mode: str,
    ) -> tuple[ComponentQuery, ...]:
        queries = []
        for index, component in enumerate(state.components):
            # Intentional correction: Xcientist queries explanation only.
            query_text = component.description
            digest = hashlib.sha256(
                f"{candidate_id}\0{state.textual_identity}\0{idea_taste_mode}\0{index}\0{query_text}".encode(
                    "utf-8"
                )
            ).hexdigest()[:20]
            queries.append(
                ComponentQuery(
                    query_id=f"component-query:{digest}",
                    component_name=component.name,
                    explanation=component.description,
                    query_text=query_text,
                )
            )
        if not queries:
            raise ComponentNoveltyError("candidate has no component explanations")
        return tuple(queries)

    def _validate_retrieval(
        self,
        request: ComponentRetrievalRequest,
        output: object,
    ) -> None:
        if not isinstance(output, ComponentRetrievalOutput):
            raise ComponentNoveltyError(
                "component retriever must return ComponentRetrievalOutput"
            )
        if output.request_identity != request.request_identity:
            raise ComponentNoveltyError(
                "component retrieval returned a mismatched request identity"
            )
        if output.candidate_id != request.candidate_id:
            raise ComponentNoveltyError(
                "component retrieval returned a mismatched candidate ID"
            )
        if output.candidate_textual_identity != request.candidate_textual_identity:
            raise ComponentNoveltyError(
                "component retrieval returned a mismatched candidate textual identity"
            )
        if output.idea_taste_mode != request.idea_taste_mode:
            raise ComponentNoveltyError(
                "component retrieval returned a mismatched idea taste mode"
            )
        if output.query_id != request.query.query_id:
            raise ComponentNoveltyError(
                "component retrieval returned a mismatched query ID"
            )
        if output.embedding_model != request.embedding_model:
            raise ComponentNoveltyError(
                "component retrieval used an undeclared embedding model"
            )
        if output.component_index != request.component_index:
            raise ComponentNoveltyError(
                "component retrieval used an undeclared component index"
            )
        if not output.native_embedding_inference:
            raise ComponentNoveltyError(
                "component retrieval did not run the declared embedding model"
            )
        if not output.native_faiss_search:
            raise ComponentNoveltyError(
                "component retrieval did not run native FAISS search"
            )
        if len(output.hits) > request.top_k:
            raise ComponentNoveltyError("component retrieval exceeded top_k")

    def _aggregate(
        self,
        queries: tuple[ComponentQuery, ...],
        outputs: Sequence[ComponentRetrievalOutput],
    ) -> tuple[NoveltyEvidence, ...]:
        counts: Counter[str] = Counter()
        best: dict[str, float] = {}
        nodes: dict[str, CoreNode] = {}
        support: Mapping[str, list[EvidenceSupport]] = defaultdict(list)
        query_by_id = {item.query_id: item for item in queries}

        for output in outputs:
            query = query_by_id[output.query_id]
            for hit in output.hits:
                node = nodes.get(hit.node.node_id)
                if node is not None and node != hit.node:
                    raise ComponentNoveltyError(
                        f"Core node {hit.node.node_id!r} changed across retrieval results"
                    )
                nodes[hit.node.node_id] = hit.node
                counts[hit.node.node_id] += 1
                best[hit.node.node_id] = max(
                    best.get(hit.node.node_id, float("-inf")), hit.similarity
                )
                if len(support[hit.node.node_id]) < self._support_top_k:
                    support[hit.node.node_id].append(
                        EvidenceSupport(
                            query_id=query.query_id,
                            query_component=query.component_name,
                            query_text=_clip(query.query_text, self._text_limit),
                            retrieval_record_id=hit.record_id,
                            matched_component=hit.matched_component,
                            similarity=hit.similarity,
                        )
                    )

        ranked = sorted(
            nodes,
            key=lambda node_id: (
                -counts[node_id],
                -best[node_id],
                (nodes[node_id].paper_title or node_id).casefold(),
            ),
        )
        return tuple(
            NoveltyEvidence(
                evidence_id=nodes[node_id].evidence_id,
                node_id=node_id,
                label=nodes[node_id].label,
                paper_title=nodes[node_id].paper_title,
                match_count=counts[node_id],
                best_similarity=best[node_id],
                summary=_clip(nodes[node_id].summary, self._text_limit),
                insight=_clip(nodes[node_id].insight, self._text_limit),
                support=tuple(support[node_id]),
                provenance_json=nodes[node_id].provenance_json,
            )
            for node_id in ranked[: self._evidence_top_k]
        )


def apply_component_novelty(
    evaluation: Evaluation,
    novelty: ComponentNoveltyResult,
    taste: Taste,
) -> NoveltyAdjustedEvaluation:
    """Replace novelty deterministically and recompute the taste composite."""

    updated = evaluation.postprocess(
        EvaluationPostprocessing(component_novelty=novelty.score)
    )
    return NoveltyAdjustedEvaluation(
        evaluation=updated,
        composite=updated.scalar(taste),
        novelty=novelty,
    )
