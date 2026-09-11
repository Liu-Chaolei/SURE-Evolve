from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PACKAGE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from research_idea_lib.algorithm.keynote_pipeline import (  # noqa: E402
    KEYNOTE_COMPRESSION_OPERATION,
    KEYNOTE_ROLLUP_OPERATION,
    KEYNOTE_SCORE_OPERATION,
    KeynotePipelineError,
    KeynotePipelineRequest,
    run_keynote_pipeline,
)
from research_idea_lib.providers import (  # noqa: E402
    JsonValue,
    ProviderRequest,
    ProviderResult,
    ProviderTrace,
    ProviderUsage,
)
from research_idea_lib.resources.retrieval import (  # noqa: E402
    CitationRegistry,
    OutcomeHit,
    SurveyParagraph,
)


class RecordingProvider:
    def __init__(
        self,
        scores: dict[str, int],
        *,
        malformed_operation: str | None = None,
        malformed_payload: dict[str, JsonValue] | None = None,
    ) -> None:
        self.scores = scores
        self.malformed_operation = malformed_operation
        self.malformed_payload = malformed_payload
        self.requests: list[ProviderRequest] = []

    def complete(self, request: ProviderRequest) -> ProviderResult:
        self.requests.append(request)
        if request.operation == self.malformed_operation:
            payload = self.malformed_payload
        elif request.operation == KEYNOTE_SCORE_OPERATION:
            structured = request.structured_input
            assert isinstance(structured, dict)
            payload = {"score": self.scores[str(structured["paper_id"])]}
        elif request.operation == KEYNOTE_COMPRESSION_OPERATION:
            structured = request.structured_input
            assert isinstance(structured, dict)
            paper_id = str(structured["paper_id"])
            payload = {
                "summary": f"Summary for {paper_id}.",
                "insight": f"Insight for {paper_id}.",
            }
        elif request.operation == KEYNOTE_ROLLUP_OPERATION:
            payload = {"summary": "One synthesis for the lower-ranked papers."}
        else:  # pragma: no cover - protects the test provider contract
            raise AssertionError(f"unexpected operation: {request.operation}")
        return ProviderResult(
            text=json.dumps(payload),
            json_value=payload,
            usage=ProviderUsage(input_tokens=2, output_tokens=1, total_tokens=3),
            trace=ProviderTrace(
                provider="recording",
                operation=request.operation,
                input_digest=request.input_digest,
                output_kind="json",
                model=request.model,
                attempts=1,
                status="success",
            ),
        )


def registry(*paper_ids: str) -> CitationRegistry:
    return CitationRegistry.from_payloads(
        {
            "references": [
                {
                    "paper_id": paper_id,
                    "title": f"Title {paper_id}",
                    "doi": f"10.example/{paper_id}",
                }
                for paper_id in paper_ids
            ]
        },
        {
            "keynotes": {
                paper_id: f"Validated keynote for {paper_id}." for paper_id in paper_ids
            }
        },
    )


def hit(
    paragraph_id: str,
    paper_ids: tuple[str, ...],
    *,
    score: float,
) -> OutcomeHit:
    return OutcomeHit(
        SurveyParagraph(
            paragraph_id=paragraph_id,
            text=f"Selected paragraph {paragraph_id}.",
            section_path=("Methods",),
            section_context=f"Selected section {paragraph_id}.",
            paper_ids=paper_ids,
        ),
        score,
    )


def request(
    citations: CitationRegistry,
    *selected_hits: OutcomeHit,
) -> KeynotePipelineRequest:
    return KeynotePipelineRequest(
        topic="Grounded research ideas",
        rag_query="mechanisms and evaluations",
        selected_hits=tuple(selected_hits),
        citations=citations,
        model="test-model",
    )


def test_ranks_only_exact_selected_citations_with_deterministic_ties() -> None:
    citations = registry("p1", "p2", "p3", "not-selected")
    provider = RecordingProvider({"p1": 70, "p2": 90, "p3": 90})

    result = run_keynote_pipeline(
        request(
            citations,
            hit("paragraph-2", ("p2", "p1"), score=0.8),
            hit("paragraph-1", ("p3", "p2"), score=0.7),
        ),
        provider=provider,
    )

    assert result.selected_paper_ids == ("p2", "p1", "p3")
    assert result.keynote_paper_ids == ("p2", "p3", "p1")
    score_requests = [
        item for item in provider.requests if item.operation == KEYNOTE_SCORE_OPERATION
    ]
    assert [item.structured_input["paper_id"] for item in score_requests] == [
        "p2",
        "p1",
        "p3",
    ]
    assert "not-selected" not in json.dumps(result.to_payload())
    payload = result.to_payload()
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    assert payload["ranked_keynotes"][0]["paper_provenance"]["doi"] == "10.example/p2"
    assert payload["ranked_keynotes"][0]["evidence"] == [
        {
            "paragraph_id": "paragraph-2",
            "outcome_score": 0.8,
            "section_path": ["Methods"],
        },
        {
            "paragraph_id": "paragraph-1",
            "outcome_score": 0.7,
            "section_path": ["Methods"],
        },
    ]
    assert all(
        item.score_trace.operation == KEYNOTE_SCORE_OPERATION
        for item in result.ranked_keynotes
    )


def test_compresses_top_five_individually_and_rolls_up_remainder_once() -> None:
    paper_ids = tuple(f"p{index}" for index in range(1, 8))
    citations = registry(*paper_ids)
    provider = RecordingProvider(
        {paper_id: 100 - index for index, paper_id in enumerate(paper_ids)}
    )

    result = run_keynote_pipeline(
        request(citations, hit("paragraph-1", paper_ids, score=0.9)),
        provider=provider,
    )

    assert [item.reference_mode for item in result.capsules] == [
        "paper_summary",
        "paper_summary",
        "paper_summary",
        "paper_summary",
        "paper_summary",
        "group_summary",
    ]
    assert [item.paper_ids for item in result.capsules[:5]] == [
        ("p1",),
        ("p2",),
        ("p3",),
        ("p4",),
        ("p5",),
    ]
    assert result.capsules[5].paper_ids == ("p6", "p7")
    operations = [item.operation for item in provider.requests]
    assert operations[:7] == [KEYNOTE_SCORE_OPERATION] * 7
    assert operations.count(KEYNOTE_SCORE_OPERATION) == len(result.keynote_paper_ids)
    assert operations.count(KEYNOTE_COMPRESSION_OPERATION) == 5
    assert operations.count(KEYNOTE_ROLLUP_OPERATION) == 1
    assert len(result.provider_traces) == 13
    assert result.provider_usage == ProviderUsage(
        input_tokens=26,
        output_tokens=13,
        total_tokens=39,
    )
    assert result.to_payload()["provider_usage"] == {
        "calls": 13,
        "input_tokens": 26,
        "output_tokens": 13,
        "total_tokens": 39,
    }
    rollup_request = next(
        item for item in provider.requests if item.operation == KEYNOTE_ROLLUP_OPERATION
    )
    assert [
        paper["paper_id"] for paper in rollup_request.structured_input["papers"]
    ] == [
        "p6",
        "p7",
    ]
    assert result.capsules[5].provider_traces[-1].operation == KEYNOTE_ROLLUP_OPERATION


@pytest.mark.parametrize(
    ("operation", "payload", "paper_count"),
    [
        (KEYNOTE_SCORE_OPERATION, {"score": True}, 1),
        (KEYNOTE_SCORE_OPERATION, {"score": 101}, 1),
        (KEYNOTE_SCORE_OPERATION, {"score": 80, "reason": "extra"}, 1),
        (KEYNOTE_COMPRESSION_OPERATION, {"summary": "", "insight": "usable"}, 1),
        (KEYNOTE_ROLLUP_OPERATION, {"summary": "usable", "extra": "field"}, 6),
    ],
)
def test_malformed_provider_output_fails_closed(
    operation: str,
    payload: dict[str, JsonValue],
    paper_count: int,
) -> None:
    paper_ids = tuple(f"p{index}" for index in range(1, paper_count + 1))
    citations = registry(*paper_ids)
    provider = RecordingProvider(
        {paper_id: 100 - index for index, paper_id in enumerate(paper_ids)},
        malformed_operation=operation,
        malformed_payload=payload,
    )

    with pytest.raises(KeynotePipelineError):
        run_keynote_pipeline(
            request(citations, hit("paragraph-1", paper_ids, score=0.9)),
            provider=provider,
        )
