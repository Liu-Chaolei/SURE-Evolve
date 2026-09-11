"""Typed, production-authentic retrieval over declared local resources.

Keynote LLM ranking/compression belongs at the workflow/provider boundary.  A
provider integration can implement ``KeynoteRanker`` and consume the typed
``KeynoteCandidate`` values returned by ``CitationRegistry.keynotes_for``;
resource retrieval itself remains citation-driven and performs no provider I/O.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .embedding import EmbeddingBackend, ResourceExecutionError
from .manifest import ResourceDescriptor

_CITATION_RE = re.compile(r"\[([^\[\]\n]+)\]")
_HEADING_RE = re.compile(r"^(#{2,6})\s+(.+?)\s*$")
_NUMERIC_CITATION_RE = re.compile(r"\[\d+\]")


@dataclass(frozen=True)
class SurveyParagraph:
    paragraph_id: str
    text: str
    section_path: tuple[str, ...]
    section_context: str
    paper_ids: tuple[str, ...]


@dataclass(frozen=True)
class OutcomeHit:
    paragraph: SurveyParagraph
    score: float


@dataclass(frozen=True)
class KeynoteCandidate:
    paper_id: str
    text: str


@dataclass(frozen=True)
class RankedKeynote:
    paper_id: str
    text: str
    rank: int
    rationale: str = ""


@runtime_checkable
class KeynoteRanker(Protocol):
    """Workflow/provider integration hook for optional LLM ranking/compression."""

    def rank_and_compress(
        self, query: str, candidates: Sequence[KeynoteCandidate]
    ) -> Sequence[RankedKeynote]: ...


@dataclass(frozen=True)
class CitationRegistry:
    references: Mapping[str, Mapping[str, Any]]
    keynotes: Mapping[str, str]
    aliases: Mapping[str, str]

    @classmethod
    def from_payloads(
        cls,
        citations: Mapping[str, Any],
        keynote_payload: Any,
    ) -> "CitationRegistry":
        raw_references = citations.get("references")
        if not isinstance(raw_references, list) or not raw_references:
            raise ResourceExecutionError("citation registry must contain non-empty references")
        references: dict[str, Mapping[str, Any]] = {}
        aliases: dict[str, str] = {}
        for index, value in enumerate(raw_references, start=1):
            if not isinstance(value, Mapping):
                raise ResourceExecutionError("citation registry references must be objects")
            paper_id = str(value.get("paper_id") or value.get("id") or "").strip()
            if not paper_id or paper_id in references:
                raise ResourceExecutionError("citation registry references require unique paper_id values")
            references[paper_id] = dict(value)
            aliases[str(index)] = paper_id

        records = keynote_payload.get("keynotes", keynote_payload) if isinstance(keynote_payload, Mapping) else None
        if not isinstance(records, Mapping):
            raise ResourceExecutionError("keynote resource keynotes must be an object")
        keynotes: dict[str, str] = {}
        for paper_id, value in records.items():
            text = keynote_text(value)
            if text:
                keynotes[str(paper_id)] = text
        return cls(references=references, keynotes=keynotes, aliases=aliases)

    def resolve_markdown(self, text: str) -> tuple[str, ...]:
        resolved: list[str] = []
        for marker in _CITATION_RE.findall(text):
            for candidate in (part.strip() for part in marker.split(",")):
                marker_id = candidate.split(";", 1)[0].strip()
                paper_id = self.aliases.get(marker_id, marker_id)
                if paper_id not in self.references and paper_id.startswith("paper:") and paper_id[6:] in self.references:
                    paper_id = paper_id[6:]
                if paper_id in self.references:
                    resolved.append(paper_id)
                elif paper_id.casefold().startswith("paper:"):
                    raise ResourceExecutionError(
                        f"Survey Markdown citation {paper_id!r} is absent from the citation registry"
                    )
        return tuple(dict.fromkeys(resolved))

    def keynotes_for(self, paper_ids: Sequence[str]) -> tuple[KeynoteCandidate, ...]:
        return tuple(
            KeynoteCandidate(paper_id, self.keynotes[paper_id])
            for paper_id in dict.fromkeys(paper_ids)
            if paper_id in self.keynotes
        )


def parse_survey_markdown(markdown: str, registry: CitationRegistry) -> tuple[SurveyParagraph, ...]:
    """Mirror OutcomeRAG paragraph slicing and one-neighbor context."""

    lines = markdown.splitlines()
    without_title = lines[1:]
    references_index = next(
        (
            index
            for index, line in reversed(tuple(enumerate(without_title)))
            if line.strip().casefold().startswith("references:") or re.fullmatch(r"#{1,6}\s+references", line.strip(), re.IGNORECASE)
        ),
        None,
    )
    if references_index is not None:
        lines = without_title[:references_index]

    grouped: list[tuple[str, int, str]] = []
    section_index = -1
    section_title = ""
    paragraph_lines: list[str] = []

    def flush() -> None:
        if not paragraph_lines:
            return
        text = "\n".join(paragraph_lines).strip()
        paragraph_lines.clear()
        if text and len(text) > 20:
            grouped.append((section_title, section_index, text))

    for line in lines:
        heading = _HEADING_RE.match(line)
        if heading:
            flush()
            section_index += 1
            section_title = heading.group(2).strip()
            paragraph_lines.append(line)
        elif not line.strip():
            flush()
        else:
            paragraph_lines.append(line)
    flush()
    if not grouped:
        raise ResourceExecutionError("Survey Markdown contains no retrievable paragraphs")

    paragraphs: list[SurveyParagraph] = []
    for index, (title, current_section, text) in enumerate(grouped):
        start = max(0, index - 1)
        stop = min(len(grouped), index + 2)
        context_parts: list[str] = []
        for context_index in range(start, stop):
            context_title, context_section, context_text = grouped[context_index]
            if (
                context_index == start
                or context_section != grouped[context_index - 1][1]
            ) and context_title and not context_text.startswith("#"):
                context_parts.append(f"## {context_title}")
            context_parts.append(context_text)
        section_context = "\n\n".join(context_parts)
        paragraphs.append(
            SurveyParagraph(
                paragraph_id=f"survey-paragraph-{index + 1}",
                text=text,
                section_path=(title,) if title else (),
                section_context=section_context,
                paper_ids=registry.resolve_markdown(section_context),
            )
        )
    return tuple(paragraphs)


def outcome_rag(
    query: str,
    paragraphs: Sequence[SurveyParagraph],
    model: EmbeddingBackend,
) -> tuple[OutcomeHit, ...]:
    """Run paragraph-level semantic search with section-bounded returned context."""

    if not query.strip():
        raise ResourceExecutionError("OutcomeRAG query cannot be empty")
    texts = [query, *(_outcome_embedding_text(paragraph.text) for paragraph in paragraphs)]
    vectors = model.encode(texts)
    if tuple(vectors.shape) != (len(texts), model.dimension):
        raise ResourceExecutionError("sentence-transformer backend returned an invalid embedding matrix")
    query_vector = vectors[0]
    scores = vectors[1:] @ query_vector
    hits = [OutcomeHit(paragraph, float(scores[index])) for index, paragraph in enumerate(paragraphs)]
    return tuple(sorted(hits, key=lambda hit: (-hit.score, hit.paragraph.paragraph_id)))


def _outcome_embedding_text(text: str) -> str:
    text = _NUMERIC_CITATION_RE.sub("", text)
    return re.sub(r"^#+\s*", "", text).strip()


@runtime_checkable
class FaissIndex(Protocol):
    d: int
    ntotal: int

    def search(self, vectors: Any, limit: int) -> tuple[Any, Any]: ...


@runtime_checkable
class FaissBackend(Protocol):
    def read_index(self, path: str) -> FaissIndex: ...


def component_metadata_records(
    payload: Any, resource_id: str
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    """Return metadata in its serialized row order, matching FAISS index construction."""

    raw = payload.get("meta") if isinstance(payload, Mapping) and "meta" in payload else payload
    if isinstance(raw, Mapping):
        records = tuple(
            (str(key), value) for key, value in raw.items() if isinstance(value, Mapping)
        )
        if len(records) != len(raw):
            raise ResourceExecutionError(
                f"component index {resource_id!r} metadata values must be objects"
            )
    elif isinstance(raw, list):
        records = tuple(
            (str(value.get("id") or value.get("component_id") or index), value)
            for index, value in enumerate(raw)
            if isinstance(value, Mapping)
        )
        if len(records) != len(raw):
            raise ResourceExecutionError(
                f"component index {resource_id!r} metadata entries must be objects"
            )
    else:
        raise ResourceExecutionError(
            f"component index {resource_id!r} metadata must be an object or array"
        )
    if not records:
        raise ResourceExecutionError(
            f"component index {resource_id!r} metadata is empty"
        )
    return records


class NativeFaissBackend:
    """Lazy FAISS adapter; import/read errors fail closed."""

    def __init__(self) -> None:
        try:
            self._faiss = importlib.import_module("faiss")
        except (ImportError, OSError) as error:
            raise ResourceExecutionError(
                "FAISS runtime is unavailable; install the package-pinned faiss-cpu dependency"
            ) from error

    def read_index(self, path: str) -> FaissIndex:
        try:
            return self._faiss.read_index(path)
        except Exception as error:
            raise ResourceExecutionError(f"declared FAISS index is unreadable: {error}") from error


def search_component_index(
    *,
    query: str,
    index_path: Path,
    descriptor: ResourceDescriptor,
    records: Sequence[tuple[str, Mapping[str, Any]]],
    model: EmbeddingBackend,
    limit: int,
    backend: FaissBackend | None = None,
) -> tuple[tuple[str, Mapping[str, Any], float], ...]:
    """Execute native FAISS search over the declared component model/index pair."""

    faiss_backend = backend if backend is not None else NativeFaissBackend()
    index = faiss_backend.read_index(str(index_path.resolve()))
    declared_dimension = int(descriptor.dimension_map.get("embedding") or 0)
    if declared_dimension <= 0:
        raise ResourceExecutionError("component index must declare a positive embedding dimension")
    if int(index.d) != declared_dimension:
        raise ResourceExecutionError(
            f"FAISS index dimension {index.d} does not match declared dimension {declared_dimension}"
        )
    if model.dimension != declared_dimension:
        raise ResourceExecutionError(
            f"component model dimension {model.dimension} does not match FAISS index dimension {declared_dimension}"
        )
    if int(index.ntotal) != len(records):
        raise ResourceExecutionError(
            f"FAISS index contains {index.ntotal} vectors but metadata contains {len(records)} records"
        )
    vectors = model.encode([query])
    if tuple(vectors.shape) != (1, declared_dimension):
        raise ResourceExecutionError("component model returned an invalid query embedding")
    try:
        distances, indices = index.search(vectors, min(max(1, limit), len(records)))
    except Exception as error:
        raise ResourceExecutionError(f"FAISS component search failed: {error}") from error
    results: list[tuple[str, Mapping[str, Any], float]] = []
    for score, position in zip(distances[0], indices[0], strict=True):
        position = int(position)
        if position < 0:
            continue
        if position >= len(records):
            raise ResourceExecutionError("FAISS returned an out-of-range metadata position")
        component_id, record = records[position]
        results.append((component_id, record, float(score)))
    return tuple(results)


def keynote_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(filter(None, (keynote_text(item) for item in value))).strip()
    if isinstance(value, Mapping):
        if "keynote" in value:
            return keynote_text(value["keynote"])
        return "\n".join(
            f"{key}: {text}"
            for key, item in sorted(value.items())
            if (text := keynote_text(item))
        ).strip()
    return ""
