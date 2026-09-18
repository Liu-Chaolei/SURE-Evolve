"""Citation-driven keynote ranking and compression for package-native research idea."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal, Protocol

from ..providers.contracts import (
    JsonValue,
    ProviderRequest,
    ProviderResult,
    ProviderTrace,
    ProviderUsage,
)
from ..resources.retrieval import CitationRegistry, OutcomeHit
from ..providers.routing import trace_matches_requested_model
from .prompts.keynote_ops import (
    KEYNOTE_GROUP_SUMMARY_PROMPT,
    KEYNOTE_SCORING_PROMPT,
    KEYNOTE_SINGLE_COMPRESSION_PROMPT,
)

KEYNOTE_SCORE_OPERATION = "xlab.research_idea.keynote.score.v1"
KEYNOTE_COMPRESSION_OPERATION = "xlab.research_idea.keynote.compress.v1"
KEYNOTE_ROLLUP_OPERATION = "xlab.research_idea.keynote.rollup.v1"
TOP_KEYNOTE_COUNT = 5

ReferenceMode = Literal["paper_summary", "group_summary"]


class KeynotePipelineError(ValueError):
    """The request or a provider output violated the keynote contract."""


class KeynoteProvider(Protocol):
    """Provider boundary injected by the workflow owner."""

    def complete(self, request: ProviderRequest) -> ProviderResult: ...


@dataclass(frozen=True)
class CitationEvidence:
    paragraph_id: str
    outcome_score: float
    section_path: tuple[str, ...]

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "paragraph_id": self.paragraph_id,
            "outcome_score": self.outcome_score,
            "section_path": list(self.section_path),
        }


@dataclass(frozen=True)
class RankedKeynote:
    paper_id: str
    title: str
    keynote: str
    relevance: int
    citation_order: int
    evidence: tuple[CitationEvidence, ...]
    paper_provenance: Mapping[str, JsonValue]
    score_trace: ProviderTrace

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "keynote": self.keynote,
            "relevance": self.relevance,
            "citation_order": self.citation_order,
            "evidence": [item.to_payload() for item in self.evidence],
            "paper_provenance": deepcopy(dict(self.paper_provenance)),
            "provider_trace": self.score_trace.to_dict(),
        }


@dataclass(frozen=True)
class KeynoteCapsule:
    title: str
    summary: str
    insight: str
    relevance: int
    reference_mode: ReferenceMode
    members: tuple[RankedKeynote, ...]
    provider_traces: tuple[ProviderTrace, ...]

    @property
    def paper_ids(self) -> tuple[str, ...]:
        return tuple(member.paper_id for member in self.members)

    @property
    def evidence(self) -> tuple[CitationEvidence, ...]:
        return tuple(item for member in self.members for item in member.evidence)

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "paper_id": self.paper_ids[0]
            if len(self.paper_ids) == 1
            else "remaining_survey_cited_papers",
            "paper_ids": list(self.paper_ids),
            "title": self.title,
            "summary": self.summary,
            "insight": self.insight,
            "score": self.relevance,
            "relevance": self.relevance,
            "reference_mode": self.reference_mode,
            "source": (
                "survey_keynote"
                if self.reference_mode == "paper_summary"
                else "survey_keynote_rollup"
            ),
            "evidence": [item.to_payload() for item in self.evidence],
            "papers": [member.to_payload() for member in self.members],
            "provider_traces": [trace.to_dict() for trace in self.provider_traces],
        }


@dataclass(frozen=True)
class KeynotePipelineRequest:
    topic: str
    rag_query: str
    selected_hits: tuple[OutcomeHit, ...]
    citations: CitationRegistry
    model: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.topic, "topic"),
            (self.rag_query, "rag_query"),
            (self.model, "model"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise KeynotePipelineError(f"{label} must be a non-empty string")
        if not isinstance(self.selected_hits, tuple) or any(
            not isinstance(hit, OutcomeHit) for hit in self.selected_hits
        ):
            raise KeynotePipelineError(
                "selected_hits must be a tuple of OutcomeHit values"
            )
        if not isinstance(self.citations, CitationRegistry):
            raise KeynotePipelineError("citations must be a validated CitationRegistry")


@dataclass(frozen=True)
class KeynotePipelineResult:
    selected_paper_ids: tuple[str, ...]
    ranked_keynotes: tuple[RankedKeynote, ...]
    capsules: tuple[KeynoteCapsule, ...]
    provider_traces: tuple[ProviderTrace, ...]
    provider_usage: ProviderUsage

    @property
    def keynote_paper_ids(self) -> tuple[str, ...]:
        return tuple(item.paper_id for item in self.ranked_keynotes)

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "selected_paper_ids": list(self.selected_paper_ids),
            "keynote_paper_ids": list(self.keynote_paper_ids),
            "ranked_keynotes": [item.to_payload() for item in self.ranked_keynotes],
            "curated_references": [item.to_payload() for item in self.capsules],
            "provider_traces": [trace.to_dict() for trace in self.provider_traces],
            "provider_usage": {
                "calls": len(self.provider_traces),
                "input_tokens": self.provider_usage.input_tokens,
                "output_tokens": self.provider_usage.output_tokens,
                "total_tokens": self.provider_usage.total_tokens,
            },
        }


def run_keynote_pipeline(
    request: KeynotePipelineRequest,
    *,
    provider: KeynoteProvider,
) -> KeynotePipelineResult:
    """Rank and compress keynotes cited by the selected OutcomeRAG hits only."""

    selected_paper_ids, candidates = _resolve_candidates(request)
    ranked: list[RankedKeynote] = []
    traces: list[ProviderTrace] = []
    usage: list[ProviderUsage] = []
    for candidate in candidates:
        structured: dict[str, JsonValue] = {
            "topic": request.topic,
            "rag_query": request.rag_query,
            "paper_id": candidate[0],
            "title": candidate[1],
            "keynote": candidate[2],
        }
        result = _complete(
            provider,
            operation=KEYNOTE_SCORE_OPERATION,
            model=request.model,
            structured_input=structured,
            prompt=KEYNOTE_SCORING_PROMPT.format(
                topic=request.topic,
                rag_query=request.rag_query,
                title=candidate[1],
                keynote=candidate[2],
            ),
            temperature=0.0,
        )
        relevance = _score_output(result)
        traces.append(result.trace)
        usage.append(result.usage)
        ranked.append(
            RankedKeynote(
                paper_id=candidate[0],
                title=candidate[1],
                keynote=candidate[2],
                relevance=relevance,
                citation_order=candidate[3],
                evidence=candidate[4],
                paper_provenance=candidate[5],
                score_trace=result.trace,
            )
        )

    ranked.sort(key=lambda item: (-item.relevance, item.citation_order, item.paper_id))
    capsules: list[KeynoteCapsule] = []
    for item in ranked[:TOP_KEYNOTE_COUNT]:
        capsule, result = _compress_one(request, provider, item)
        capsules.append(capsule)
        traces.append(result.trace)
        usage.append(result.usage)
    if len(ranked) > TOP_KEYNOTE_COUNT:
        capsule, result = _roll_up(
            request, provider, tuple(ranked[TOP_KEYNOTE_COUNT:])
        )
        capsules.append(capsule)
        traces.append(result.trace)
        usage.append(result.usage)

    return KeynotePipelineResult(
        selected_paper_ids=selected_paper_ids,
        ranked_keynotes=tuple(ranked),
        capsules=tuple(capsules),
        provider_traces=tuple(traces),
        provider_usage=ProviderUsage(
            input_tokens=sum(item.input_tokens for item in usage),
            output_tokens=sum(item.output_tokens for item in usage),
            total_tokens=sum(item.total_tokens for item in usage),
        ),
    )


def _resolve_candidates(
    request: KeynotePipelineRequest,
) -> tuple[
    tuple[str, ...],
    list[
        tuple[str, str, str, int, tuple[CitationEvidence, ...], Mapping[str, JsonValue]]
    ],
]:
    selected: list[str] = []
    evidence_by_paper: dict[str, list[CitationEvidence]] = {}
    for hit in request.selected_hits:
        score = hit.score
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not isfinite(score)
        ):
            raise KeynotePipelineError("OutcomeRAG hit scores must be finite numbers")
        paragraph = hit.paragraph
        if not paragraph.paragraph_id.strip():
            raise KeynotePipelineError("OutcomeRAG paragraph IDs cannot be empty")
        evidence = CitationEvidence(
            paragraph_id=paragraph.paragraph_id,
            outcome_score=float(score),
            section_path=paragraph.section_path,
        )
        for paper_id in paragraph.paper_ids:
            if not paper_id.strip():
                raise KeynotePipelineError("OutcomeRAG paper IDs cannot be empty")
            if paper_id not in evidence_by_paper:
                selected.append(paper_id)
                evidence_by_paper[paper_id] = []
            if all(
                item.paragraph_id != evidence.paragraph_id
                for item in evidence_by_paper[paper_id]
            ):
                evidence_by_paper[paper_id].append(evidence)

    keynotes = {
        item.paper_id: item.text for item in request.citations.keynotes_for(selected)
    }
    candidates = []
    for citation_order, paper_id in enumerate(selected):
        keynote = keynotes.get(paper_id)
        if keynote is None:
            continue
        reference = request.citations.references[paper_id]
        provenance = _json_mapping(reference, f"citation reference {paper_id!r}")
        title = str(
            reference.get("title") or reference.get("paper_title") or paper_id
        ).strip()
        candidates.append(
            (
                paper_id,
                title,
                keynote,
                citation_order,
                tuple(evidence_by_paper[paper_id]),
                provenance,
            )
        )
    return tuple(selected), candidates


def _compress_one(
    request: KeynotePipelineRequest,
    provider: KeynoteProvider,
    item: RankedKeynote,
) -> tuple[KeynoteCapsule, ProviderResult]:
    structured: dict[str, JsonValue] = {
        "topic": request.topic,
        "rag_query": request.rag_query,
        "paper_id": item.paper_id,
        "title": item.title,
        "keynote": item.keynote,
        "relevance": item.relevance,
    }
    result = _complete(
        provider,
        operation=KEYNOTE_COMPRESSION_OPERATION,
        model=request.model,
        structured_input=structured,
        prompt=KEYNOTE_SINGLE_COMPRESSION_PROMPT.format(
            topic=request.topic,
            rag_query=request.rag_query,
            title=item.title,
            keynote=item.keynote,
        ),
        temperature=0.1,
    )
    payload = _exact_output(result, {"summary", "insight"})
    summary = _nonempty_text(payload["summary"], "summary")
    insight = _nonempty_text(payload["insight"], "insight")
    return (
        KeynoteCapsule(
            title=item.title,
            summary=summary,
            insight=insight,
            relevance=item.relevance,
            reference_mode="paper_summary",
            members=(item,),
            provider_traces=(item.score_trace, result.trace),
        ),
        result,
    )


def _roll_up(
    request: KeynotePipelineRequest,
    provider: KeynoteProvider,
    items: tuple[RankedKeynote, ...],
) -> tuple[KeynoteCapsule, ProviderResult]:
    papers: list[JsonValue] = [
        {
            "paper_id": item.paper_id,
            "title": item.title,
            "score": item.relevance,
            "keynote": item.keynote,
        }
        for item in items
    ]
    structured: dict[str, JsonValue] = {
        "topic": request.topic,
        "rag_query": request.rag_query,
        "papers": papers,
    }
    result = _complete(
        provider,
        operation=KEYNOTE_ROLLUP_OPERATION,
        model=request.model,
        structured_input=structured,
        prompt=KEYNOTE_GROUP_SUMMARY_PROMPT.format(
            topic=request.topic,
            rag_query=request.rag_query,
            papers=json.dumps(papers, ensure_ascii=False, sort_keys=True),
        ),
        temperature=0.1,
    )
    payload = _exact_output(result, {"summary"})
    summary = _nonempty_text(payload["summary"], "summary")
    return (
        KeynoteCapsule(
            title=f"Remaining survey-cited papers ({len(items)})",
            summary=summary,
            insight="",
            relevance=items[0].relevance,
            reference_mode="group_summary",
            members=items,
            provider_traces=tuple(item.score_trace for item in items) + (result.trace,),
        ),
        result,
    )


def _complete(
    provider: KeynoteProvider,
    *,
    operation: str,
    model: str,
    structured_input: dict[str, JsonValue],
    prompt: str,
    temperature: float,
) -> ProviderResult:
    def validate_response(result: ProviderResult) -> None:
        if operation == KEYNOTE_SCORE_OPERATION:
            _score_output(result)
        else:
            fields = {"summary", "insight"} if operation == KEYNOTE_COMPRESSION_OPERATION else {"summary"}
            value = _exact_output(result, fields)
            for name in fields:
                _nonempty_text(value[name], name)

    request = ProviderRequest(
        operation=operation,
        model=model,
        structured_input=structured_input,
        system_prompt=prompt,
        user_prompt=json.dumps(
            structured_input,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        output_kind="json",
        temperature=temperature,
        validation_profile="xlab.keynote.response.v1:" + operation,
        response_validator=validate_response,
    )
    result = provider.complete(request)
    if not isinstance(result, ProviderResult):
        raise KeynotePipelineError(
            f"provider operation {operation} must return ProviderResult"
        )
    trace = result.trace
    if (
        not isinstance(trace, ProviderTrace)
        or trace.operation != operation
        or trace.input_digest != request.input_digest
        or trace.output_kind != "json"
        or not trace_matches_requested_model(trace, model)
        or trace.status != "success"
        or isinstance(trace.attempts, bool)
        or not isinstance(trace.attempts, int)
        or trace.attempts < 1
    ):
        raise KeynotePipelineError(
            f"provider operation {operation} returned an invalid trace"
        )
    return result


def _score_output(result: ProviderResult) -> int:
    payload = _exact_output(result, {"score"})
    score = payload["score"]
    if getattr(getattr(result, "trace", None), "model", "").startswith("glm-") and isinstance(score, str):
        try:
            score = json.loads(score)
        except json.JSONDecodeError as error:
            raise KeynotePipelineError("GLM keynote score must encode an integer from 0 to 100") from error
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 100:
        raise KeynotePipelineError("keynote score must be an integer from 0 to 100")
    return score


def _glm_field_projection(payload, fields):
    """Read explicit GLM fields, retaining the untouched wire result for audit."""
    if not isinstance(payload, dict):
        return payload
    for _ in range(3):
        if set(payload) != {"answer"}:
            break
        wrapped = payload["answer"]
        if isinstance(wrapped, str):
            try:
                decoded = json.loads(wrapped)
            except json.JSONDecodeError:
                break
            wrapped = decoded if isinstance(decoded, dict) else wrapped
        if not isinstance(wrapped, dict):
            break
        payload = wrapped
    if fields == {"summary", "insight"} and set(payload) == {"answer", "insight"}:
        # GLM sometimes names the summary `answer` while keeping the separate
        # insight explicit. Preserve both strings; never synthesize an insight.
        if isinstance(payload["answer"], str) and isinstance(payload["insight"], str):
            return {"summary": payload["answer"], "insight": payload["insight"]}
    if not fields.issubset(payload):
        return payload
    projected = {key: deepcopy(payload[key]) for key in fields}
    if "answer" not in payload:
        return projected
    alternate = payload["answer"]
    if isinstance(alternate, str):
        try:
            decoded = json.loads(alternate)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict):
            alternate = decoded
    if fields == {"score"} and isinstance(alternate, str):
        try:
            alternate = json.loads(alternate)
        except json.JSONDecodeError:
            # Prose accompanying an explicit score is explanatory metadata.
            return projected
    if len(fields) == 1 and not isinstance(alternate, dict):
        alternate = {next(iter(fields)): alternate}
    if isinstance(alternate, dict):
        for key in fields.intersection(alternate):
            actual, other = projected[key], alternate[key]
            if key == "score" and isinstance(actual, str):
                try:
                    actual = json.loads(actual)
                except json.JSONDecodeError:
                    pass
            if type(actual) is not type(other) or actual != other:
                raise KeynotePipelineError(f"GLM answer conflicts with explicit {key}")
    return projected


def _exact_output(result: ProviderResult, fields: set[str]) -> dict[str, JsonValue]:
    def without_annotations(value):
        if not isinstance(value, dict):
            return value
        return {key: child for key, child in value.items() if not (
            key in {"rationale", "reason", "explanation"} - fields and isinstance(child, str)
        )}

    # Optional prose is retained in the provider cache, not used as a score or
    # capsule field. Missing fields, conflicting answers and unknown keys fail.
    payload = result.json_value
    if getattr(getattr(result, "trace", None), "model", "").startswith("glm-"):
        raw_text = getattr(result, "text", "").strip()
        if raw_text.startswith("{"):
            # Some GLM responses concatenate an explanatory JSON object and the
            # requested result. The transport extracted only the first object.
            # Recover explicit fields only when the entire text is object JSON.
            decoder = json.JSONDecoder()
            objects = []
            remaining = raw_text
            while remaining and len(objects) < 4:
                try:
                    item, end = decoder.raw_decode(remaining)
                except json.JSONDecodeError:
                    break
                if not isinstance(item, dict):
                    break
                objects.append(item)
                remaining = remaining[end:].strip()
            if not remaining and len(objects) > 1:
                if objects[0] != payload:
                    raise KeynotePipelineError("GLM raw objects disagree with parsed provider result")
                merged = {}
                for item in objects:
                    for key, value in item.items():
                        if key in merged and (type(merged[key]) is not type(value) or merged[key] != value):
                            raise KeynotePipelineError(f"GLM JSON objects contain conflicting {key}")
                        merged[key] = deepcopy(value)
                payload = merged
        payload = _glm_field_projection(payload, fields)
    if isinstance(payload, dict) and set(payload) == fields | {"answer"}:
        expected = {key: payload[key] for key in fields}
        wrapped = without_annotations(payload["answer"])
        if fields == {"score"} and isinstance(wrapped, str):
            try:
                wrapped = json.loads(wrapped)
            except json.JSONDecodeError:
                pass
        if len(fields) == 1 and not isinstance(wrapped, dict):
            field = next(iter(fields))
            if type(wrapped) is type(expected[field]):
                wrapped = {field: wrapped}
        if isinstance(wrapped, dict) and set(wrapped) == fields and all(
            type(wrapped[key]) is type(expected[key]) and wrapped[key] == expected[key] for key in fields
        ):
            payload = expected
    if isinstance(payload, dict) and set(payload) == {"answer"}:
        wrapped = without_annotations(payload["answer"])
        if isinstance(wrapped, dict):
            payload = deepcopy(wrapped)
        elif len(fields) == 1:
            payload = {next(iter(fields)): wrapped}
        if fields == {"score"} and isinstance(payload.get("score"), str):
            try:
                payload["score"] = json.loads(payload["score"])
            except json.JSONDecodeError as error:
                raise KeynotePipelineError("wrapped keynote score must encode an integer from 0 to 100") from error
    if not isinstance(payload, dict) or set(payload) != fields:
        expected = ", ".join(sorted(fields))
        raise KeynotePipelineError(f"provider output must contain exactly: {expected}")
    return payload


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise KeynotePipelineError(
            f"provider output {field} must be a non-empty string"
        )
    return value.strip()


def _json_mapping(value: Mapping[str, Any], label: str) -> dict[str, JsonValue]:
    try:
        payload = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise KeynotePipelineError(f"{label} must contain only JSON values") from error
    if not isinstance(payload, dict):
        raise KeynotePipelineError(f"{label} must be a mapping")
    return payload
