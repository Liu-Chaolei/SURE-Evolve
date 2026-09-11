from __future__ import annotations

import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PACKAGE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from research_idea_lib.algorithm.component_novelty import (  # noqa: E402
    COMPONENT_NOVELTY_OPERATION,
    UPSTREAM_EVIDENCE_TOP_K,
    UPSTREAM_RETRIEVAL_TOP_K,
    ComponentHit,
    ComponentNoveltyError,
    ComponentNoveltyEvaluator,
    ComponentNoveltyOperation,
    ComponentNoveltyProviderOutput,
    ComponentRetrievalOutput,
    CoreNode,
    DeclaredNoveltyResource,
    apply_component_novelty,
)
from research_idea_lib.algorithm.contracts import IdeaComponent, IdeaState  # noqa: E402
from research_idea_lib.algorithm.evaluation import Evaluation, METRICS  # noqa: E402
from research_idea_lib.algorithm.tastes import get_taste  # noqa: E402


def idea() -> IdeaState:
    return IdeaState(
        title="Candidate",
        abstract="A focused candidate abstract.",
        core_contribution="A testable mechanism.",
        method="Measure a controlled intervention.",
        risks="Distribution shift.",
        components=(
            IdeaComponent("Adaptive gate", "Routes uncertain examples to a verifier."),
            IdeaComponent(
                "Repair loop", "Revises failed outputs using verifier feedback."
            ),
        ),
        tags=("agents",),
        root_domains=("computer science",),
    )


def resource(role: str) -> DeclaredNoveltyResource:
    return DeclaredNoveltyResource(
        resource_id=f"resource:{role}",
        digest=role * 8,
        algorithm_version="sentence-transformers.v1"
        if role == "model"
        else "faiss-flat-ip.v1",
        logical_uri=f"xlab-resource://bundle/{role}",
    )


def node(node_id: str, title: str) -> CoreNode:
    return CoreNode(
        evidence_id=f"evidence:{node_id}",
        node_id=node_id,
        label=f"Core {node_id}",
        paper_title=title,
        summary=f"{title} summary",
        insight=f"{title} insight",
        provenance_json=json.dumps(
            {"graph_resource_id": "graph:v1", "node_id": node_id}
        ),
    )


class RecordingRetriever:
    def __init__(self, hit_rows):
        self.hit_rows = hit_rows
        self.requests = []

    def retrieve(self, request):
        self.requests.append(request)
        hits = tuple(
            ComponentHit(
                record_id=f"record:{request.query.query_id}:{index}",
                node=item[0],
                matched_component=item[1],
                similarity=item[2],
            )
            for index, item in enumerate(self.hit_rows[len(self.requests) - 1])
        )
        return ComponentRetrievalOutput(
            request_identity=request.request_identity,
            candidate_id=request.candidate_id,
            candidate_textual_identity=request.candidate_textual_identity,
            idea_taste_mode=request.idea_taste_mode,
            query_id=request.query.query_id,
            hits=hits,
            embedding_model=request.embedding_model,
            component_index=request.component_index,
            native_embedding_inference=True,
            native_faiss_search=True,
            provenance_json=json.dumps(
                {
                    "query_id": request.query.query_id,
                    "model": request.embedding_model.resource_id,
                    "index": request.component_index.resource_id,
                }
            ),
        )


class RecordingProvider:
    def __init__(self, score=4):
        self.score = score
        self.operations: list[ComponentNoveltyOperation] = []

    def execute(self, operation):
        self.operations.append(operation)
        return ComponentNoveltyProviderOutput(
            retrieval_similarity=2,
            perceived_novelty=self.score,
            rubric_score=self.score,
            rationale="The repair mechanism departs from the nearest prior art.",
            provenance_json=json.dumps(
                {
                    "provider": "fake",
                    "operation": operation.name,
                    "candidate_id": operation.candidate_id,
                }
            ),
        )


def evaluator(retriever, provider, **kwargs) -> ComponentNoveltyEvaluator:
    return ComponentNoveltyEvaluator(
        retriever,
        provider,
        embedding_model=resource("model"),
        component_index=resource("index"),
        **kwargs,
    )


def test_queries_every_candidate_component_with_upstream_defaults_and_declared_resources():
    retriever = RecordingRetriever([[], []])
    provider = RecordingProvider()

    with pytest.raises(ComponentNoveltyError, match="no Core-node evidence"):
        evaluator(retriever, provider).evaluate(idea(), candidate_id="node-7", idea_taste_mode="moonshot_inventor")

    assert len(retriever.requests) == 2
    assert all(request.candidate_id == "node-7" for request in retriever.requests)
    assert all(
        request.idea_taste_mode == "moonshot_inventor"
        for request in retriever.requests
    )
    assert all(
        request.top_k == UPSTREAM_RETRIEVAL_TOP_K for request in retriever.requests
    )
    assert [request.query.query_text for request in retriever.requests] == [
        "Routes uncertain examples to a verifier.",
        "Revises failed outputs using verifier feedback.",
    ]
    assert all(
        request.embedding_model.resource_id == "resource:model"
        for request in retriever.requests
    )
    assert all(
        request.component_index.resource_id == "resource:index"
        for request in retriever.requests
    )
    assert provider.operations == []


def test_aggregates_core_nodes_by_match_count_then_best_similarity_and_bounds_evidence():
    alpha = node("alpha", "Zulu")
    beta = node("beta", "Alpha")
    gamma = node("gamma", "Gamma")
    retriever = RecordingRetriever(
        [
            [(alpha, "a1", 0.91), (beta, "b1", 0.99), (gamma, "g1", 1.0)],
            [(alpha, "a2", 0.40), (beta, "b2", 0.80)],
        ]
    )
    provider = RecordingProvider()

    result = evaluator(retriever, provider, evidence_top_k=2, support_top_k=1).evaluate(
        idea(), candidate_id="candidate:1", idea_taste_mode="moonshot_inventor"
    )

    assert UPSTREAM_EVIDENCE_TOP_K == 5
    assert result.evidence_ids == ("evidence:beta", "evidence:alpha")
    assert [item.match_count for item in result.evidence] == [2, 2]
    assert [item.best_similarity for item in result.evidence] == [0.99, 0.91]
    assert all(len(item.support) == 1 for item in result.evidence)
    assert len(result.retrieval_provenance_json) == 2
    assert all(
        json.loads(value)["model"] == "resource:model"
        for value in result.retrieval_provenance_json
    )

    operation = provider.operations[0]
    assert operation.name == COMPONENT_NOVELTY_OPERATION
    assert operation.candidate_textual_identity == idea().textual_identity
    assert operation.evidence_ids == result.evidence_ids
    assert operation.to_payload()["embedding_model"]["resource_id"] == "resource:model"


def test_tie_break_is_deterministic_and_provenance_is_immutable():
    alpha = node("alpha", "Same")
    beta = node("beta", "Same")
    retriever = RecordingRetriever([[(beta, "b", 0.5), (alpha, "a", 0.5)], []])
    provider = RecordingProvider()

    result = evaluator(retriever, provider).evaluate(idea(), candidate_id="candidate:2", idea_taste_mode="moonshot_inventor")

    assert result.evidence_ids == ("evidence:beta", "evidence:alpha")
    with pytest.raises(FrozenInstanceError):
        result.evidence[0].evidence_id = "changed"
    parsed = json.loads(result.evidence[0].provenance_json)
    parsed["node_id"] = "changed"
    assert json.loads(result.evidence[0].provenance_json)["node_id"] == "beta"


def test_provider_requires_explicit_integer_zero_to_five_judgment():
    with pytest.raises(ValueError, match="integer between 0 and 5"):
        ComponentNoveltyProviderOutput(
            retrieval_similarity=2,
            perceived_novelty=4,
            rubric_score=5.1,
            rationale="Invalid float.",
            provenance_json='{"provider":"fake"}',
        )
    with pytest.raises(ValueError, match="integer between 0 and 5"):
        ComponentNoveltyProviderOutput(
            retrieval_similarity=2,
            perceived_novelty=4,
            rubric_score=6,
            rationale="Out of range.",
            provenance_json='{"provider":"fake"}',
        )


def test_replaces_novelty_and_recomputes_composite_deterministically():
    core = node("core", "Prior work")
    retriever = RecordingRetriever([[(core, "prior", 0.75)], []])
    result = evaluator(retriever, RecordingProvider(score=5)).evaluate(
        idea(), candidate_id="candidate:3", idea_taste_mode="moonshot_inventor"
    )
    metrics = {name: 3 for name in METRICS}
    metrics["novelty"] = 0
    original = Evaluation.from_payload(metrics)
    taste = get_taste("moonshot_inventor")

    adjusted = apply_component_novelty(original, result, taste)

    assert original.metrics["novelty"] == 0
    assert adjusted.evaluation.metrics["novelty"] == 5
    assert adjusted.composite == adjusted.evaluation.scalar(taste)
    assert (
        adjusted.composite
        == original.scalar(taste) + 5 * taste.weights["novelty_weight"]
    )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("request_identity", "component-retrieval:tampered", "request identity"),
        ("candidate_id", "other-candidate", "candidate ID"),
        ("candidate_textual_identity", "other-text", "candidate textual identity"),
        ("idea_taste_mode", "other-mode", "idea taste mode"),
        ("query_id", "other-query", "query ID"),
    ],
)
def test_fails_closed_when_retrieval_identity_is_tampered(
    field, replacement, message
):
    core = node("core", "Prior work")

    class TamperedRetriever(RecordingRetriever):
        def retrieve(self, request):
            output = super().retrieve(request)
            values = dict(output.__dict__)
            values[field] = replacement
            return ComponentRetrievalOutput(**values)

    with pytest.raises(ComponentNoveltyError, match=message):
        evaluator(
            TamperedRetriever([[(core, "prior", 0.7)]]), RecordingProvider()
        ).evaluate(
            idea(),
            candidate_id="candidate:tampered",
            idea_taste_mode="moonshot_inventor",
        )


def test_query_identity_is_bound_to_taste_mode():
    first = RecordingRetriever([[], []])
    second = RecordingRetriever([[], []])
    with pytest.raises(ComponentNoveltyError):
        evaluator(first, RecordingProvider()).evaluate(
            idea(), candidate_id="root", idea_taste_mode="moonshot_inventor"
        )
    with pytest.raises(ComponentNoveltyError):
        evaluator(second, RecordingProvider()).evaluate(
            idea(), candidate_id="root", idea_taste_mode="evidence_first"
        )

    assert {request.query.query_id for request in first.requests}.isdisjoint(
        request.query.query_id for request in second.requests
    )
    assert first.requests[0].request_identity != second.requests[0].request_identity


@pytest.mark.parametrize(
    ("native_embedding", "native_faiss", "message"),
    [
        (False, True, "declared embedding model"),
        (True, False, "native FAISS search"),
    ],
)
def test_fails_closed_when_declared_native_resource_was_not_executed(
    native_embedding, native_faiss, message
):
    core = node("core", "Prior work")

    class UnverifiedRetriever:
        def retrieve(self, request):
            return ComponentRetrievalOutput(
                request_identity=request.request_identity,
                candidate_id=request.candidate_id,
                candidate_textual_identity=request.candidate_textual_identity,
                idea_taste_mode=request.idea_taste_mode,
                query_id=request.query.query_id,
                hits=(ComponentHit("record", core, "prior", 0.7),),
                embedding_model=request.embedding_model,
                component_index=request.component_index,
                native_embedding_inference=native_embedding,
                native_faiss_search=native_faiss,
                provenance_json='{"adapter":"test"}',
            )

    with pytest.raises(ComponentNoveltyError, match=message):
        evaluator(UnverifiedRetriever(), RecordingProvider()).evaluate(
            idea(), candidate_id="candidate:4", idea_taste_mode="moonshot_inventor"
        )


def test_fails_closed_on_retriever_or_provider_failure_without_retaining_old_novelty():
    class BrokenRetriever:
        def retrieve(self, request):
            raise OSError("FAISS unavailable")

    with pytest.raises(ComponentNoveltyError, match="FAISS unavailable"):
        evaluator(BrokenRetriever(), RecordingProvider()).evaluate(
            idea(), candidate_id="candidate:5", idea_taste_mode="moonshot_inventor"
        )

    core = node("core", "Prior work")

    class BrokenProvider:
        def execute(self, operation):
            raise TimeoutError("judgment unavailable")

    with pytest.raises(ComponentNoveltyError, match="judgment unavailable"):
        evaluator(
            RecordingRetriever([[(core, "prior", 0.7)], []]), BrokenProvider()
        ).evaluate(idea(), candidate_id="candidate:6", idea_taste_mode="moonshot_inventor")
