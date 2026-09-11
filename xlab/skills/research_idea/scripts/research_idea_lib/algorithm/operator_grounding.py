"""Operator-specific component graph grounding."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol

from .contracts import (
    GroundedComponentEvidence,
    IdeaState,
    OperatorGrounding,
    OperatorGroundingKind,
    PromptMode,
)

THEORY_TRANSFER_OPERATOR = "theory-transfer-injection"
MECHANISM_COMMIT_OPERATOR = "mechanism-commit-innovation"
THEORY_TRANSFER_THRESHOLD = 0.6
MECHANISM_COMMIT_THRESHOLD = 0.6
MAX_COMPONENTS_PER_CORE_NODE = 2


@dataclass(frozen=True)
class OperatorQuery:
    kind: OperatorGroundingKind
    query: str
    needed_content: str
    expected_role: str
    provenance_json: str

    def __post_init__(self) -> None:
        if not self.query.strip() or not self.needed_content.strip() or not self.expected_role.strip():
            raise ValueError("operator query fields cannot be empty")
        _provenance(self.provenance_json, "operator query")


@dataclass(frozen=True)
class OperatorComponentHit:
    core_node_id: str
    component_id: str
    component_name: str
    component_description: str
    score: float
    domain: str | None
    paper_ids: tuple[str, ...]
    resource_identity: str
    trace_identity: str
    paper_title: str = ""
    full_name: str = ""
    label: str = ""

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
            raise ValueError("operator component hit fields cannot be empty")
        if not 0 <= float(self.score) <= 1:
            raise ValueError("operator component hit score must be in [0, 1]")


@dataclass(frozen=True)
class OperatorComponentRetrieval:
    hits: tuple[OperatorComponentHit, ...]
    provenance_json: str

    def __post_init__(self) -> None:
        _provenance(self.provenance_json, "operator component retrieval")


@dataclass(frozen=True)
class OperatorGroundingResult:
    grounding: OperatorGrounding
    skip_operator: bool
    explicit_empty: bool
    query_provenance_json: str
    references_provenance_json: str
    retrieval_provenance_json: str


class OperatorQueryProvider(Protocol):
    def theory_transfer_query(self, state: IdeaState, *, prompt_mode: PromptMode) -> OperatorQuery: ...

    def mechanism_commit_query(self, state: IdeaState, *, prompt_mode: PromptMode) -> OperatorQuery: ...


class OperatorComponentRetriever(Protocol):
    def retrieve_operator_components(self, query: str, *, limit: int) -> OperatorComponentRetrieval: ...


class OperatorGroundingEvaluator:
    def __init__(
        self,
        query_provider: OperatorQueryProvider,
        retriever: OperatorComponentRetriever,
        *,
        top_k: int = 3,
    ) -> None:
        if top_k < 1:
            raise ValueError("operator grounding top_k must be positive")
        self._query_provider = query_provider
        self._retriever = retriever
        self._top_k = top_k

    def evaluate(
        self,
        operator: str,
        state: IdeaState,
        *,
        prompt_mode: PromptMode = "default",
    ) -> OperatorGroundingResult:
        if operator == THEORY_TRANSFER_OPERATOR:
            query = self._query_provider.theory_transfer_query(state, prompt_mode=prompt_mode)
            threshold = THEORY_TRANSFER_THRESHOLD
        elif operator == MECHANISM_COMMIT_OPERATOR:
            query = self._query_provider.mechanism_commit_query(state, prompt_mode=prompt_mode)
            threshold = MECHANISM_COMMIT_THRESHOLD
        else:
            raise ValueError(f"operator does not support component grounding: {operator}")

        raw_limit = max(self._top_k * 6, 12)
        retrieval = self._retriever.retrieve_operator_components(query.query, limit=raw_limit)
        ordered_hits = sorted(
            retrieval.hits,
            key=lambda hit: (
                -hit.score,
                _normalized(hit.component_name),
                _normalized(hit.core_node_id),
            ),
        )
        by_node: dict[str, list[OperatorComponentHit]] = {}
        for hit in ordered_hits:
            matches = by_node.setdefault(hit.core_node_id, [])
            if len(matches) < MAX_COMPONENTS_PER_CORE_NODE:
                matches.append(hit)
        ranked_nodes = sorted(
            by_node,
            key=lambda node_id: (
                -by_node[node_id][0].score,
                _normalized(
                    by_node[node_id][0].paper_title
                    or by_node[node_id][0].full_name
                    or by_node[node_id][0].label
                    or node_id
                ),
            ),
        )

        root_domains = {domain.casefold().strip() for domain in state.root_domains}
        selected_nodes: list[str] = []
        for node_id in ranked_nodes:
            best = by_node[node_id][0]
            if best.score < threshold:
                continue
            if query.kind == "theory_transfer":
                if best.domain is None or not best.domain.strip():
                    continue
                if best.domain.casefold().strip() in root_domains:
                    continue
            selected_nodes.append(node_id)
            if len(selected_nodes) >= self._top_k:
                break
        selected = [hit for node_id in selected_nodes for hit in by_node[node_id]]

        evidence = tuple(
            GroundedComponentEvidence(
                core_node_id=hit.core_node_id,
                component_id=hit.component_id,
                component_name=hit.component_name,
                component_description=hit.component_description,
                score=hit.score,
                domain=hit.domain,
                resource_identity=hit.resource_identity,
                trace_identity=hit.trace_identity,
            )
            for hit in selected
        )
        grounding = OperatorGrounding(
            kind=query.kind,
            query=query.query,
            needed_content=query.needed_content,
            expected_role=query.expected_role,
            evidence=evidence,
        )
        references = {
            hit.component_id: list(hit.paper_ids)
            for hit in selected
        }
        empty = not evidence
        return OperatorGroundingResult(
            grounding=grounding,
            skip_operator=query.kind == "theory_transfer" and empty,
            explicit_empty=query.kind == "mechanism_commit" and empty,
            query_provenance_json=query.provenance_json,
            references_provenance_json=json.dumps(
                {"component_references": references}, sort_keys=True, separators=(",", ":")
            ),
            retrieval_provenance_json=retrieval.provenance_json,
        )


def _normalized(value: str) -> str:
    return value.casefold().strip()


def trace_identity(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _provenance(value: str, field: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} provenance must be valid JSON") from error
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError(f"{field} provenance cannot be empty")
    return parsed
