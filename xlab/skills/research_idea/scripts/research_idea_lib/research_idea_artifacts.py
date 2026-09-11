"""Typed read boundary for research idea namespace artifacts."""

from __future__ import annotations

from typing import Any, Mapping, TypeAlias, TypedDict, cast

JsonObject: TypeAlias = dict[str, Any]


class RunNamespace(TypedDict, total=False):
    workflow_trace: list[JsonObject]
    operation_trace: list[JsonObject]


class RetrievalNamespace(TypedDict, total=False):
    rag_hits: list[Any]
    references: list[Any]


class AnalysisNamespace(TypedDict, total=False):
    entries: list[JsonObject]
    root_idea: JsonObject


class IdeationNamespace(TypedDict, total=False):
    latest_candidate: JsonObject
    ablation_results: list[JsonObject]


class PersistenceNamespace(TypedDict, total=False):
    idea_result: JsonObject
    schema_version: int


class WorkflowArtifact(TypedDict, total=False):
    run: RunNamespace
    retrieval: RetrievalNamespace
    analysis: AnalysisNamespace
    ideation: IdeationNamespace
    persistence: PersistenceNamespace


def namespace(artifact: Mapping[str, Any], name: str) -> JsonObject:
    value = artifact.get(name)
    return value if isinstance(value, dict) else {}


def run_namespace(artifact: Mapping[str, Any]) -> RunNamespace:
    return cast(RunNamespace, namespace(artifact, "run"))


def retrieval_namespace(artifact: Mapping[str, Any]) -> RetrievalNamespace:
    return cast(RetrievalNamespace, namespace(artifact, "retrieval"))


def analysis_namespace(artifact: Mapping[str, Any]) -> AnalysisNamespace:
    return cast(AnalysisNamespace, namespace(artifact, "analysis"))


def ideation_namespace(artifact: Mapping[str, Any]) -> IdeationNamespace:
    return cast(IdeationNamespace, namespace(artifact, "ideation"))


def persistence_namespace(artifact: Mapping[str, Any]) -> PersistenceNamespace:
    return cast(PersistenceNamespace, namespace(artifact, "persistence"))


def final_idea_result(artifact: Mapping[str, Any]) -> JsonObject:
    value = persistence_namespace(artifact).get("idea_result")
    return value if isinstance(value, dict) else {}


def latest_analysis_entry(artifact: Mapping[str, Any]) -> JsonObject:
    entries = analysis_namespace(artifact).get("entries")
    if not isinstance(entries, list) or not entries:
        return {}
    entry = entries[-1]
    return entry if isinstance(entry, dict) else {}
