"""Execute validated research-idea evidence resources with native ML runtimes."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .embedding import ADAPTER_ALGORITHM, EmbeddingBackend, ResourceExecutionError, SentenceTransformerEmbedding
from .manifest import LoadedResourceBundle, ResourceDescriptor, sha256_file
from .retrieval import (
    CitationRegistry,
    FaissBackend,
    OutcomeHit,
    component_metadata_records,
    outcome_rag,
    parse_survey_markdown,
    search_component_index,
)


@dataclass(frozen=True)
class ResourceEvidenceResult:
    evidence_items: tuple[dict[str, Any], ...]
    usage: dict[str, dict[str, Any]]
    selected_outcome_hits: tuple[OutcomeHit, ...] = ()
    citations: CitationRegistry | None = None


@dataclass(frozen=True)
class RetrievalBackends:
    """Explicit inference injection, intended for package-owned runtimes and hermetic tests."""

    outcome: EmbeddingBackend
    component: EmbeddingBackend
    faiss: FaissBackend


class PackageNativeEvidenceResources:
    """Consume one validated bundle through sentence-transformers and FAISS."""

    def __init__(
        self,
        *,
        bundle: LoadedResourceBundle,
        graph_db_path: Path,
        keynote_cache_path: Path,
        component_index_dir: Path,
        outcome_model_path: Path,
        component_model_path: Path,
        outcome_backend: EmbeddingBackend | None = None,
        component_backend: EmbeddingBackend | None = None,
        faiss_backend: FaissBackend | None = None,
    ) -> None:
        self._bundle = bundle
        self._manifest = bundle.manifest
        self._graph_db_path = graph_db_path
        self._keynote_cache_path = keynote_cache_path
        self._component_index_dir = component_index_dir
        outcome, component = _select_models(self._manifest.models)
        self._outcome = outcome_backend or SentenceTransformerEmbedding.outcome(outcome, outcome_model_path)
        self._component = component_backend or SentenceTransformerEmbedding(component, component_model_path)
        self._outcome_descriptor = outcome
        self._component_descriptor = component
        self._faiss_backend = faiss_backend

    def retrieve(
        self,
        query: str,
        survey_markdown: str,
        citations: Mapping[str, Any],
        references: Sequence[Mapping[str, Any]],
        *,
        limit: int,
    ) -> ResourceEvidenceResult:
        keynote_payload = _json_file(
            _single_declared_file(self._keynote_cache_path, self._manifest.keynotes, "keynotes.json"),
            f"keynote resource {self._manifest.keynotes.resource_id!r}",
        )
        registry = CitationRegistry.from_payloads(citations, keynote_payload)
        selected_outcome_hits, ranked_survey = self._rank_survey(
            query, survey_markdown, registry
        )
        graph_items, graph_usage = _graph_neighbors(
            self._graph_db_path,
            self._manifest.graph,
            ranked_survey,
            references,
        )
        keynote_items, keynote_usage = _keynote_evidence(
            self._keynote_cache_path,
            self._manifest.keynotes,
            ranked_survey,
            graph_items,
            references,
        )
        component_items, component_usage = _component_evidence(
            self._component_index_dir,
            self._manifest.component_index,
            self._component_descriptor,
            self._component,
            query,
            limit=limit,
            faiss_backend=self._faiss_backend,
        )
        # Keynotes remain in the citation registry and grounding stage. They must
        # not crowd out the top-k main evidence before the consumer filters them.
        selected = tuple(
            deepcopy(item)
            for _, _, item in sorted(
                [
                    *[(float(item["resource_score"]), index, item) for index, item in enumerate(ranked_survey)],
                    *[(float(item["resource_score"]), len(ranked_survey) + index, item) for index, item in enumerate(graph_items)],
                    *[
                        (
                            float(item["resource_score"]),
                            len(ranked_survey) + len(graph_items) + len(keynote_items) + index,
                            item,
                        )
                        for index, item in enumerate(component_items)
                    ],
                ],
                key=lambda value: (-value[0], value[1], str(value[2].get("id") or "")),
            )[:limit]
        )
        usage = {
            "graph": graph_usage,
            "keynotes": keynote_usage,
            "outcome_model": {
                **self._outcome.usage(),
                "logical_uri": _logical_uri(self._bundle, self._manifest.models, self._outcome_descriptor),
                "records_scored": len(ranked_survey),
            },
            "component_model": {
                **self._component.usage(),
                "logical_uri": _logical_uri(self._bundle, self._manifest.models, self._component_descriptor),
                "records_scored": len(component_items),
            },
            "component_index": component_usage,
        }
        return ResourceEvidenceResult(
            selected,
            usage,
            selected_outcome_hits,
            registry,
        )

    def _rank_survey(
        self,
        query: str,
        survey_markdown: str,
        registry: CitationRegistry,
    ) -> tuple[tuple[OutcomeHit, ...], list[dict[str, Any]]]:
        paragraphs = parse_survey_markdown(survey_markdown, registry)
        selected_hits: list[OutcomeHit] = []
        seen_contexts: set[str] = set()
        for hit in outcome_rag(query, paragraphs, self._outcome):
            context_identity = hit.paragraph.section_context[:200]
            if context_identity in seen_contexts:
                continue
            seen_contexts.add(context_identity)
            selected_hits.append(hit)
            if len(selected_hits) == 5:
                break

        ranked: list[dict[str, Any]] = []
        for hit in selected_hits:
            paragraph = hit.paragraph
            ranked.append(
                {
                    "id": paragraph.paragraph_id,
                    "kind": "survey_markdown_paragraph",
                    "title": " / ".join(paragraph.section_path) or "Survey",
                    "text": paragraph.section_context,
                    "paper_ids": list(paragraph.paper_ids),
                    "source": "survey.md",
                    "resource_score": hit.score,
                    "resource_provenance": {
                        "role": "outcome_model",
                        "resource_id": self._outcome_descriptor.resource_id,
                        "descriptor_digest": self._outcome_descriptor.digest,
                        "adapter_algorithm": ADAPTER_ALGORITHM,
                        "paragraph_id": paragraph.paragraph_id,
                        "section_path": list(paragraph.section_path),
                        "adjacent_context_window": 1,
                        "citation_registry_resolved": True,
                    },
                }
            )
        return tuple(selected_hits), ranked


def _graph_neighbors(
    path: Path,
    descriptor: ResourceDescriptor,
    survey_items: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed_ids = sorted(
        {
            paper_id
            for item in survey_items
            for paper_id in _paper_ids(item)
        }
        | {str(reference.get("paper_id") or "").strip() for reference in references}
        - {""}
    )
    declared_path = _single_declared_file(path.parent, descriptor, "graph.db")
    if declared_path.resolve() != path.resolve():
        raise ResourceExecutionError(f"graph resource {descriptor.resource_id!r} resolved an undeclared database")
    uri = f"file:{path.resolve().as_posix()}?mode=ro&immutable=1"
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        columns = _sqlite_columns(connection, "nodes")
        edge_columns = _sqlite_columns(connection, "edges")
        required_nodes = {"id", "paper_id"}
        required_edges = {"source", "target"}
        if not required_nodes.issubset(columns) or not required_edges.issubset(edge_columns):
            raise ResourceExecutionError(
                f"graph resource {descriptor.resource_id!r} must expose nodes(id,paper_id) and edges(source,target)"
            )
        rows = connection.execute(
            "SELECT id, paper_id, "
            + _select_column(columns, "paper_title")
            + ", "
            + _select_column(columns, "summary")
            + " FROM nodes ORDER BY id"
        ).fetchall()
        by_node = {str(row["id"]): row for row in rows}
        seed_nodes = {node_id for node_id, row in by_node.items() if str(row["paper_id"] or "").strip() in seed_ids}
        if not seed_nodes:
            raise ResourceExecutionError(
                f"graph resource {descriptor.resource_id!r} contains no nodes linked to survey papers"
            )
        relation_select = _select_column(edge_columns, "relation", fallback="edge_type")
        summary_select = _select_column(edge_columns, "summary")
        edges = connection.execute(
            f"SELECT source, target, {relation_select}, {summary_select} FROM edges ORDER BY source, target"
        ).fetchall()
    except sqlite3.Error as error:
        raise ResourceExecutionError(f"graph resource {descriptor.resource_id!r} is malformed: {error}") from error
    finally:
        if connection is not None:
            connection.close()

    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for edge in edges:
        source = str(edge["source"])
        target = str(edge["target"])
        if source in seed_nodes:
            neighbor_id, seed_node = target, source
        elif target in seed_nodes:
            neighbor_id, seed_node = source, target
        else:
            continue
        neighbor = by_node.get(neighbor_id)
        seed = by_node.get(seed_node)
        if neighbor is None or seed is None:
            raise ResourceExecutionError(
                f"graph resource {descriptor.resource_id!r} contains an edge to an undeclared node"
            )
        paper_id = str(neighbor["paper_id"] or "").strip()
        if not paper_id or paper_id in seed_ids or (neighbor_id, seed_node) in seen:
            continue
        seen.add((neighbor_id, seed_node))
        title = str(neighbor["paper_title"] or neighbor_id).strip()
        relation = str(edge[2] or "related").strip()
        summary = str(neighbor["summary"] or edge[3] or title).strip()
        text = f"Graph neighbor {title} ({relation}): {summary}"
        items.append(
            {
                "id": f"graph:{neighbor_id}:{seed_node}",
                "kind": "graph_neighbor",
                "title": title,
                "text": text,
                "paper_ids": [paper_id],
                "source": "paper_graph.neighbors",
                "resource_score": 0.5,
                "resource_provenance": {
                    "role": "graph",
                    "resource_id": descriptor.resource_id,
                    "descriptor_digest": descriptor.digest,
                    "seed_paper_id": str(seed["paper_id"] or ""),
                    "neighbor_paper_id": paper_id,
                    "relation": relation,
                },
            }
        )
    return items, {
        "resource_id": descriptor.resource_id,
        "descriptor_digest": descriptor.digest,
        "logical_uri": _resource_uri(descriptor),
        "adapter_algorithm": "xlab.package_native.sqlite_paper_neighbors.v1",
        "capability_consumed": "paper_neighbors",
        "seed_paper_ids": seed_ids,
        "neighbor_records_consumed": len(items),
    }


def _keynote_evidence(
    root: Path,
    descriptor: ResourceDescriptor,
    survey_items: Sequence[Mapping[str, Any]],
    graph_items: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = _single_declared_file(root, descriptor, "keynotes.json")
    payload = _json_file(path, f"keynote resource {descriptor.resource_id!r}")
    if not isinstance(payload, Mapping):
        raise ResourceExecutionError(f"keynote resource {descriptor.resource_id!r} must contain a JSON object")
    records = payload.get("keynotes", payload)
    if not isinstance(records, Mapping):
        raise ResourceExecutionError(f"keynote resource {descriptor.resource_id!r} keynotes must be an object")
    wanted = {
        paper_id
        for item in (*survey_items, *graph_items)
        for paper_id in _paper_ids(item)
    } | {str(reference.get("paper_id") or "").strip() for reference in references}
    wanted.discard("")
    items: list[dict[str, Any]] = []
    consumed: list[str] = []
    for paper_id in sorted(wanted):
        if paper_id not in records:
            continue
        text = _keynote_text(records[paper_id])
        if not text:
            raise ResourceExecutionError(
                f"keynote resource {descriptor.resource_id!r} has a malformed record for paper {paper_id!r}"
            )
        consumed.append(paper_id)
        items.append(
            {
                "id": f"keynote:{paper_id}",
                "kind": "paper_keynote",
                "title": f"Keynote for {paper_id}",
                "text": text,
                "paper_ids": [paper_id],
                "source": "paper_keynotes",
                "resource_score": 0.55,
                "resource_provenance": {
                    "role": "keynotes",
                    "resource_id": descriptor.resource_id,
                    "descriptor_digest": descriptor.digest,
                    "paper_id": paper_id,
                },
            }
        )
    if not consumed:
        raise ResourceExecutionError(
            f"keynote resource {descriptor.resource_id!r} contains no records for retrieved survey or graph papers"
        )
    return items, {
        "resource_id": descriptor.resource_id,
        "descriptor_digest": descriptor.digest,
        "logical_uri": _resource_uri(descriptor),
        "adapter_algorithm": "xlab.package_native.paper_keynote_json.v1",
        "capability_consumed": "paper_keynotes",
        "paper_ids_consumed": consumed,
        "records_consumed": len(consumed),
    }


def _component_evidence(
    root: Path,
    descriptor: ResourceDescriptor,
    model_descriptor: ResourceDescriptor,
    profile: EmbeddingBackend,
    query: str,
    *,
    limit: int,
    faiss_backend: FaissBackend | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata_path = _single_declared_file(root, descriptor, "meta.json")
    index_path = _single_declared_file(root, descriptor, "faiss.index")
    payload = _json_file(metadata_path, f"component index {descriptor.resource_id!r}")
    records = component_metadata_records(payload, descriptor.resource_id)
    for component_id, record in records:
        if not _text(record):
            raise ResourceExecutionError(
                f"component index {descriptor.resource_id!r} has a metadata record without retrievable text: {component_id!r}"
            )
    hits = search_component_index(
        query=query,
        index_path=index_path,
        descriptor=descriptor,
        records=records,
        model=profile,
        limit=limit,
        backend=faiss_backend,
    )
    items = [
        {
            "id": f"component:{component_id}",
            "kind": "component_novelty_context",
            "title": str(record.get("title") or record.get("name") or component_id),
            "text": _text(record),
            "paper_ids": _paper_ids(record),
            "source": "component_index.faiss",
            "resource_score": score,
            "resource_provenance": {
                "role": "component_index",
                "resource_id": descriptor.resource_id,
                "descriptor_digest": descriptor.digest,
                "component_id": component_id,
                "component_model_id": model_descriptor.resource_id,
                "component_model_digest": model_descriptor.digest,
                "adapter_algorithm": "faiss.read_index.search.v1",
            },
        }
        for component_id, record, score in hits
    ]
    return items, {
        "resource_id": descriptor.resource_id,
        "descriptor_digest": descriptor.digest,
        "logical_uri": _resource_uri(descriptor),
        "adapter_algorithm": "faiss.read_index.search.v1",
        "capability_consumed": "component_similarity",
        "metadata_records_consumed": len(records),
        "faiss_file_consumed": {
            "logical_path": next(file.path for file in descriptor.files if Path(file.path).name == "faiss.index"),
            "sha256": next(file.sha256 for file in descriptor.files if Path(file.path).name == "faiss.index"),
            "native_faiss_inference": True,
        },
    }


def _sqlite_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    try:
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.Error as error:
        raise ResourceExecutionError(f"cannot inspect graph table {table!r}: {error}") from error
    return {str(row[1]) for row in rows}


def _select_column(columns: set[str], name: str, *, fallback: str | None = None) -> str:
    if name in columns:
        return name
    if fallback and fallback in columns:
        return fallback
    return f"NULL AS {name}"


def _keynote_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(_keynote_text(item) for item in value if _keynote_text(item)).strip()
    if isinstance(value, Mapping):
        keynote = value.get("keynote")
        if keynote is not None:
            return _keynote_text(keynote)
        return "\n".join(
            f"{key}: {_keynote_text(item)}"
            for key, item in sorted(value.items())
            if _keynote_text(item)
        ).strip()
    return ""


def _json_file(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ResourceExecutionError(f"{label} is unreadable or malformed: {error}") from error


def _single_declared_file(root: Path, descriptor: ResourceDescriptor, filename: str) -> Path:
    matches = [file for file in descriptor.files if Path(file.path).name == filename]
    if len(matches) != 1:
        raise ResourceExecutionError(
            f"resource {descriptor.resource_id!r} must declare exactly one {filename}; found {len(matches)}"
        )
    expected = matches[0]
    path = root.resolve() / expected.path
    if not path.is_file():
        raise ResourceExecutionError(f"resource {descriptor.resource_id!r} is missing {filename}")
    try:
        actual_size = path.stat().st_size
        actual_digest = sha256_file(path)
    except OSError as error:
        raise ResourceExecutionError(f"resource {descriptor.resource_id!r} cannot read {filename}") from error
    if actual_size != expected.size or actual_digest != expected.sha256:
        raise ResourceExecutionError(
            f"resource {descriptor.resource_id!r} {filename} does not match its declared digest"
        )
    return path


def _select_models(models: tuple[ResourceDescriptor, ...]) -> tuple[ResourceDescriptor, ResourceDescriptor]:
    def searchable(model: ResourceDescriptor) -> str:
        role = model.provenance_map.get("role", "")
        return " ".join((role, model.resource_id, *model.capabilities)).casefold().replace("-", "_")

    outcome = [model for model in models if "outcome" in searchable(model)]
    component = [model for model in models if any(word in searchable(model) for word in ("component", "novelty"))]
    if len(outcome) != 1 or len(component) != 1:
        raise ResourceExecutionError("validated bundle does not identify exactly one outcome and component model")
    return outcome[0], component[0]


def _logical_uri(
    bundle: LoadedResourceBundle,
    models: tuple[ResourceDescriptor, ...],
    descriptor: ResourceDescriptor,
) -> str:
    del models
    return f"xlab-model-descriptor://{bundle.manifest.bundle_id}/{descriptor.resource_id}/{descriptor.digest}"


def _resource_uri(descriptor: ResourceDescriptor) -> str:
    return f"xlab-resource-descriptor://{descriptor.resource_id}/{descriptor.digest}"


def _paper_ids(value: Mapping[str, Any]) -> list[str]:
    raw = value.get("paper_ids")
    if isinstance(raw, list):
        return list(dict.fromkeys(str(item).strip() for item in raw if str(item).strip()))
    paper_id = str(value.get("paper_id") or "").strip()
    return [paper_id] if paper_id else []


def _text(value: Mapping[str, Any]) -> str:
    parts = [value.get(name) for name in ("text", "summary", "title", "name", "description", "component")]
    return " ".join(str(part).strip() for part in parts if str(part or "").strip())
