"""Concrete declared-model/FAISS adapter for candidate component novelty."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ..algorithm.component_novelty import (
    ComponentHit,
    ComponentRetrievalOutput,
    ComponentRetrievalRequest,
    CoreNode,
    DeclaredNoveltyResource,
)
from ..algorithm.operator_grounding import (
    MAX_COMPONENTS_PER_CORE_NODE,
    OperatorComponentHit,
    OperatorComponentRetrieval,
    trace_identity,
)
from .embedding import EmbeddingBackend, ResourceExecutionError, SentenceTransformerEmbedding
from .manifest import ResourceDescriptor, sha256_file
from .resolver import ResourceResolution
from .retrieval import FaissBackend, component_metadata_records, search_component_index


@dataclass(frozen=True)
class ComponentNoveltyRuntime:
    retriever: "DeclaredFaissComponentRetriever"
    embedding_model: DeclaredNoveltyResource
    component_index: DeclaredNoveltyResource


class DeclaredFaissComponentRetriever:
    """Satisfy ComponentRetriever using only one validated declared model/index."""

    def __init__(
        self,
        *,
        index_descriptor: ResourceDescriptor,
        model_descriptor: ResourceDescriptor,
        index_root: Path,
        model_root: Path,
        index_uri: str,
        model_uri: str,
        embedding_backend: EmbeddingBackend | None = None,
        faiss_backend: FaissBackend | None = None,
    ) -> None:
        self.embedding_model = _identity(model_descriptor, model_uri)
        self.component_index = _identity(index_descriptor, index_uri)
        self._index_descriptor = index_descriptor
        self._model_descriptor = model_descriptor
        self._model = embedding_backend or SentenceTransformerEmbedding(model_descriptor, model_root)
        self._faiss = faiss_backend
        self._index_path = _declared_file(index_root, index_descriptor, "faiss.index")
        metadata_path = _declared_file(index_root, index_descriptor, "meta.json")
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ResourceExecutionError("declared component metadata is unreadable or malformed") from error
        self._records = component_metadata_records(payload, index_descriptor.resource_id)
        self._nodes = _core_nodes(self._records, index_descriptor)

    def retrieve_operator_components(
        self, query: str, *, limit: int
    ) -> OperatorComponentRetrieval:
        if not query.strip() or limit < 1:
            raise ValueError("operator component retrieval requires a query and positive limit")
        internal_search_limit = max(
            limit * MAX_COMPONENTS_PER_CORE_NODE * 8,
            limit,
        )
        hits = search_component_index(
            query=query,
            index_path=self._index_path,
            descriptor=self._index_descriptor,
            records=self._records,
            model=self._model,
            limit=internal_search_limit,
            backend=self._faiss,
        )
        resource_identity = self.component_index.resource_id
        typed_hits = tuple(
            OperatorComponentHit(
                core_node_id=str(record.get("node_id") or record.get("core_node_id") or component_id).strip(),
                component_id=component_id,
                component_name=str(
                    record.get("component") or record.get("title") or record.get("name") or component_id
                ).strip(),
                component_description=str(
                    record.get("description") or record.get("summary") or record.get("insight") or ""
                ).strip(),
                score=score,
                domain=_optional_text(record.get("domain") or record.get("root_domain")),
                paper_ids=_string_tuple(record.get("paper_ids", ())),
                resource_identity=resource_identity,
                trace_identity=trace_identity(
                    {
                        "component_id": component_id,
                        "descriptor_digest": self._index_descriptor.digest,
                        "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                        "score": score,
                    }
                ),
                paper_title=str(record.get("paper_title") or "").strip(),
                full_name=str(record.get("full_name") or "").strip(),
                label=str(record.get("label") or "").strip(),
            )
            for component_id, record, score in hits
        )
        provenance = {
            "adapter": "sentence-transformers.faiss.operator-component-retriever.v1",
            "query_text_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "raw_limit": limit,
            "internal_search_limit": internal_search_limit,
            "model": self.embedding_model.to_payload(),
            "index": self.component_index.to_payload(),
            "native_embedding_inference": True,
            "native_faiss_search": True,
        }
        return OperatorComponentRetrieval(
            hits=typed_hits,
            provenance_json=json.dumps(provenance, sort_keys=True, separators=(",", ":")),
        )

    def retrieve(self, request: ComponentRetrievalRequest) -> ComponentRetrievalOutput:
        if request.embedding_model != self.embedding_model:
            raise ResourceExecutionError("component novelty request model identity does not match declared runtime")
        if request.component_index != self.component_index:
            raise ResourceExecutionError("component novelty request index identity does not match declared runtime")
        hits = search_component_index(
            query=request.query.query_text,
            index_path=self._index_path,
            descriptor=self._index_descriptor,
            records=self._records,
            model=self._model,
            limit=request.top_k,
            backend=self._faiss,
        )
        typed_hits = tuple(
            ComponentHit(
                record_id=component_id,
                node=self._nodes[str(record.get("node_id") or record.get("id") or component_id).strip()],
                matched_component=str(record.get("component") or record.get("title") or record.get("name") or component_id),
                similarity=score,
            )
            for component_id, record, score in hits
        )
        provenance = {
            "adapter": "sentence-transformers.faiss.component-retriever.v1",
            "request_identity": request.request_identity,
            "candidate_id": request.candidate_id,
            "candidate_textual_identity": request.candidate_textual_identity,
            "idea_taste_mode": request.idea_taste_mode,
            "query_id": request.query.query_id,
            "query_text_sha256": hashlib.sha256(
                request.query.query_text.encode("utf-8")
            ).hexdigest(),
            "model": self.embedding_model.to_payload(),
            "index": self.component_index.to_payload(),
            "embedding_dimension": self._model.dimension,
            "native_embedding_inference": True,
            "native_faiss_search": True,
        }
        return ComponentRetrievalOutput(
            request_identity=request.request_identity,
            candidate_id=request.candidate_id,
            candidate_textual_identity=request.candidate_textual_identity,
            idea_taste_mode=request.idea_taste_mode,
            query_id=request.query.query_id,
            hits=typed_hits,
            embedding_model=self.embedding_model,
            component_index=self.component_index,
            native_embedding_inference=True,
            native_faiss_search=True,
            provenance_json=json.dumps(provenance, sort_keys=True, separators=(",", ":")),
        )


def build_component_novelty_runtime(
    resolution: ResourceResolution,
    *,
    embedding_backend: EmbeddingBackend | None = None,
    faiss_backend: FaissBackend | None = None,
) -> ComponentNoveltyRuntime:
    """Build a novelty runtime from one already-validated ResourceResolution."""

    if not resolution.passed or resolution.bundle is None:
        raise ResourceExecutionError("component novelty requires a validated resource resolution")
    if resolution.component_index_dir is None or resolution.component_model_path is None:
        raise ResourceExecutionError("component novelty resources did not resolve executable paths")
    descriptor = resolution.bundle.manifest.component_index
    models = [
        model
        for model in resolution.bundle.manifest.models
        if model.provenance_map.get("role", "").strip().casefold().replace("-", "_") == "component_model"
    ]
    if len(models) != 1:
        raise ResourceExecutionError("bundle must declare exactly one component_model")
    model = models[0]
    portable = resolution.portable_resources()
    index_uri = str(portable.get("component_index", {}).get("logical_uri") or "")
    model_uri = str(portable.get("component_model", {}).get("logical_uri") or "")
    if not index_uri or not model_uri:
        raise ResourceExecutionError("component novelty resources require immutable logical URIs")
    retriever = DeclaredFaissComponentRetriever(
        index_descriptor=descriptor,
        model_descriptor=model,
        index_root=resolution.component_index_dir,
        model_root=resolution.component_model_path,
        index_uri=index_uri,
        model_uri=model_uri,
        embedding_backend=embedding_backend,
        faiss_backend=faiss_backend,
    )
    return ComponentNoveltyRuntime(retriever, retriever.embedding_model, retriever.component_index)


def _identity(descriptor: ResourceDescriptor, uri: str) -> DeclaredNoveltyResource:
    return DeclaredNoveltyResource(
        resource_id=descriptor.resource_id,
        digest=descriptor.digest,
        algorithm_version=descriptor.algorithm_version,
        logical_uri=uri,
    )


def _declared_file(root: Path, descriptor: ResourceDescriptor, filename: str) -> Path:
    matches = [item for item in descriptor.files if PurePosixPath(item.path).name == filename]
    if len(matches) != 1:
        raise ResourceExecutionError(
            f"resource {descriptor.resource_id!r} must declare exactly one {filename}"
        )
    expected = matches[0]
    path = (root.resolve() / Path(*PurePosixPath(expected.path).parts)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ResourceExecutionError(f"resource {descriptor.resource_id!r} file escapes its root") from error
    if not path.is_file() or path.stat().st_size != expected.size or sha256_file(path) != expected.sha256:
        raise ResourceExecutionError(
            f"resource {descriptor.resource_id!r} {filename} does not match its declared digest"
        )
    return path


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _core_nodes(records, descriptor: ResourceDescriptor) -> dict[str, CoreNode]:
    groups: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for component_id, record in records:
        node_id = str(record.get("node_id") or record.get("id") or component_id).strip()
        groups.setdefault(node_id, []).append((component_id, record))
    nodes = {}
    for node_id, members in groups.items():
        members = sorted(members, key=lambda item: item[0])
        canonical = dict(members[0][1])
        for field in ('paper_title', 'label'):
            values = {str(record[field]).strip() for _, record in members if record.get(field)}
            if len(values) > 1:
                raise ResourceExecutionError(f'Conflicting Core node {node_id} {field}')
            if values:
                canonical[field] = next(iter(values))
        canonical['node_id'] = node_id
        canonical['evidence_id'] = f'component:{node_id}'
        # Component summaries belong to hits. Build one immutable, traceable Core
        # projection from the full declared index, never from query-dependent hits.
        canonical['summary'] = '\n'.join(
            f"{record.get('component') or component_id}: {record.get('summary') or record.get('description') or ''}"
            for component_id, record in members)
        canonical['insight'] = '\n'.join(dict.fromkeys(
            str(record.get('insight') or record.get('explanation') or '').strip()
            for _, record in members if record.get('insight') or record.get('explanation')))
        nodes[node_id] = _core_node(node_id, canonical, descriptor,
                                   component_ids=tuple(key for key, _ in members))
    return nodes


def _core_node(component_id: str, record: Mapping[str, Any], descriptor: ResourceDescriptor,
               *, component_ids: tuple[str, ...] = ()) -> CoreNode:
    node_id = str(record.get("node_id") or record.get("id") or component_id).strip()
    label = str(record.get("label") or record.get("title") or record.get("name") or component_id).strip()
    provenance = {
        "resource_id": descriptor.resource_id,
        "descriptor_digest": descriptor.digest,
        **({"component_ids": list(component_ids)} if component_ids else {"component_id": component_id}),
    }
    return CoreNode(
        evidence_id=str(record.get("evidence_id") or f"component:{component_id}"),
        node_id=node_id,
        label=label,
        paper_title=str(record.get("paper_title") or record.get("title") or ""),
        summary=str(record.get("summary") or record.get("description") or ""),
        insight=str(record.get("insight") or record.get("explanation") or record.get("description") or ""),
        provenance_json=json.dumps(provenance, sort_keys=True, separators=(",", ":")),
    )
