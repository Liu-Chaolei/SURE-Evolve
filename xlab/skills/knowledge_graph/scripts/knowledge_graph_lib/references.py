from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from .common import (
    JsonObject,
    as_list,
    as_mapping,
    compact_key,
    normalize_space,
    read_jsonl,
    text,
)


S2_FIELDS = ",".join(
    [
        "paperId",
        "title",
        "venue",
        "year",
        "externalIds",
        "url",
        "abstract",
        "authors",
        "references.paperId",
        "references.title",
        "references.venue",
        "references.year",
        "references.externalIds",
        "references.url",
        "references.abstract",
        "references.authors",
    ]
)
REFERENCE_HEADING = re.compile(
    r"(?im)^#{1,6}\s+(?:(?:\d+(?:\.\d+)*|[IVXLCDM]+)[.)]?\s+)?"
    r"(?:references|bibliography)\s*[:.]?\s*$"
)
NEXT_HEADING = re.compile(r"(?m)^#{1,6}\s+\S")
REFERENCE_ENTRY = re.compile(
    r"(?ms)(?:^|\n)\s*(?:\[(\d{1,4})\]|(\d{1,4})[.)])\s+(.+?)(?=\n\s*(?:\[\d{1,4}\]|\d{1,4}[.)])\s+|\Z)"
)
YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _truthy(value: object) -> bool:
    return text(value).lower() in {"1", "true", "yes", "on"}


def _base_url() -> str:
    configured = text(os.environ.get("KNOWLEDGE_GRAPH_SEMANTIC_SCHOLAR_API_URL"))
    if configured:
        return configured.rstrip("/")
    return "https://api.semanticscholar.org/graph/v1"


def _external_ids(metadata: JsonObject) -> JsonObject:
    values = as_mapping(metadata.get("external_ids")) or as_mapping(
        metadata.get("externalIds")
    )
    for source_key, target_key in (
        ("doi", "DOI"),
        ("arxiv_id", "ArXiv"),
        ("arxivId", "ArXiv"),
        ("corpus_id", "CorpusId"),
        ("corpusId", "CorpusId"),
    ):
        value = metadata.get(source_key)
        if text(value) and not text(values.get(target_key)):
            values[target_key] = value
    return values


def _reference_key(reference: JsonObject) -> str:
    external_ids = as_mapping(reference.get("externalIds"))
    for key in ("paperId", "DOI", "ArXiv", "CorpusId"):
        value = reference.get(key) if key == "paperId" else external_ids.get(key)
        if text(value):
            return f"{key.lower()}:{compact_key(value)}"
    title = compact_key(reference.get("title"))
    return f"title:{title}" if title else ""


def _clean_reference(reference: JsonObject, *, source: str, confidence: float) -> JsonObject:
    external_ids = as_mapping(reference.get("externalIds")) or as_mapping(
        reference.get("external_ids")
    )
    authors = []
    for value in as_list(reference.get("authors")):
        author = as_mapping(value)
        name = normalize_space(author.get("name") or value)
        if name:
            authors.append(name)
    result: JsonObject = {
        "paperId": text(reference.get("paperId") or reference.get("paper_id")) or None,
        "title": normalize_space(reference.get("title")) or None,
        "venue": normalize_space(reference.get("venue")) or None,
        "year": reference.get("year"),
        "externalIds": external_ids,
        "url": text(reference.get("url")) or None,
        "abstract": normalize_space(reference.get("abstract")) or None,
        "authors": authors,
        "source": source,
        "reference_source": source,
        "confidence": confidence,
    }
    if reference.get("raw_reference") is not None:
        result["raw_reference"] = reference.get("raw_reference")
    if reference.get("local_index") is not None:
        result["local_index"] = reference.get("local_index")
    return result


def _merge_references(groups: list[list[JsonObject]]) -> list[JsonObject]:
    merged: dict[str, JsonObject] = {}
    order: list[str] = []
    for references in groups:
        for reference in references:
            clean = as_mapping(reference)
            key = _reference_key(clean)
            if not key:
                continue
            if key not in merged:
                merged[key] = clean
                order.append(key)
                continue
            existing = merged[key]
            sources = [
                value
                for value in as_list(existing.get("sources"))
                if isinstance(value, str)
            ] or [text(existing.get("source"))]
            source = text(clean.get("source"))
            if source and source not in sources:
                sources.append(source)
            for field in ("paperId", "title", "venue", "year", "url", "abstract"):
                if not existing.get(field) and clean.get(field):
                    existing[field] = clean.get(field)
            external_ids = {**as_mapping(clean.get("externalIds")), **as_mapping(existing.get("externalIds"))}
            existing["externalIds"] = external_ids
            existing["sources"] = sources
            existing["confidence"] = max(
                float(existing.get("confidence") or 0),
                float(clean.get("confidence") or 0),
            )
    result = [merged[key] for key in order]
    result.sort(
        key=lambda item: (
            -float(item.get("confidence") or 0),
            normalize_space(item.get("title")).lower(),
            text(item.get("paperId")),
        )
    )
    return result


class SemanticScholarClient:
    def __init__(self) -> None:
        self.disabled = _truthy(os.environ.get("KNOWLEDGE_GRAPH_DISABLE_SEMANTIC_SCHOLAR"))
        self.base_url = _base_url()
        self.api_key = text(
            os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY")
        )
        self.timeout = float(
            text(os.environ.get("KNOWLEDGE_GRAPH_SEMANTIC_SCHOLAR_TIMEOUT")) or "8"
        )
        self.retries = int(
            text(os.environ.get("KNOWLEDGE_GRAPH_SEMANTIC_SCHOLAR_RETRIES")) or "1"
        )
        self._last_request_at = 0.0

    def _url(self, path: str, query: dict[str, object]) -> str:
        parsed = urlsplit(self.base_url + path)
        return urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                urlencode({key: value for key, value in query.items() if text(value)}),
                "",
            )
        )

    def _get(self, path: str, query: dict[str, object]) -> JsonObject | None:
        if self.disabled:
            return None
        url = self._url(path, query)
        headers = {"Accept": "application/json", "User-Agent": "xlab-knowledge-graph/3.0"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        for attempt in range(self.retries + 1):
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < 0.35:
                time.sleep(0.35 - elapsed)
            self._last_request_at = time.monotonic()
            try:
                request = Request(url, headers=headers, method="GET")
                with urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - public S2 API only
                    return as_mapping(json.loads(response.read().decode("utf-8")))
            except HTTPError as error:
                if error.code == 404:
                    return None
                if error.code == 429 and attempt < self.retries:
                    retry_after = text(error.headers.get("Retry-After"))
                    time.sleep(float(retry_after) if retry_after.isdigit() else 1.0)
                    continue
                return None
            except (OSError, URLError, json.JSONDecodeError, TimeoutError):
                if attempt < self.retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                return None
        return None

    def get_paper(self, identifier: str) -> JsonObject | None:
        identifier = text(identifier)
        if not identifier:
            return None
        return self._get(f"/paper/{quote(identifier, safe=':')}", {"fields": S2_FIELDS})

    def search_title(self, title: str) -> JsonObject | None:
        title = normalize_space(title)
        if not title:
            return None
        response = self._get(
            "/paper/search",
            {"query": title, "limit": 1, "fields": S2_FIELDS},
        )
        data = [as_mapping(value) for value in as_list(as_mapping(response).get("data"))]
        if not data:
            return None
        candidate = data[0]
        candidate_title = compact_key(candidate.get("title"))
        query_title = compact_key(title)
        if not candidate_title or not query_title:
            return None
        if candidate_title != query_title and (
            candidate_title not in query_title and query_title not in candidate_title
        ):
            return None
        return candidate


def _s2_references(paper: JsonObject, *, source: str) -> list[JsonObject]:
    references = []
    for value in as_list(paper.get("references")):
        reference = as_mapping(value)
        if normalize_space(reference.get("title")) or text(reference.get("paperId")):
            references.append(_clean_reference(reference, source=source, confidence=0.92))
    return references


def _resolve_semantic_references(
    client: SemanticScholarClient, document: JsonObject
) -> list[JsonObject]:
    metadata = as_mapping(document.get("metadata"))
    external_ids = _external_ids(metadata)
    attempts: list[tuple[str, str]] = []
    if text(external_ids.get("DOI")):
        attempts.append(("semantic_scholar_doi", f"DOI:{text(external_ids.get('DOI'))}"))
    if text(external_ids.get("ArXiv")):
        attempts.append(("semantic_scholar_arxiv", f"ARXIV:{text(external_ids.get('ArXiv'))}"))
    for value in (
        metadata.get("paperId"),
        metadata.get("semantic_scholar_id"),
        metadata.get("semanticScholarId"),
    ):
        if text(value):
            attempts.append(("semantic_scholar_paper_id", text(value)))
    if text(external_ids.get("CorpusId")):
        attempts.append(("semantic_scholar_corpus_id", f"CorpusId:{text(external_ids.get('CorpusId'))}"))
    seen: set[str] = set()
    for source, identifier in attempts:
        if identifier in seen:
            continue
        seen.add(identifier)
        paper = client.get_paper(identifier)
        references = _s2_references(as_mapping(paper), source=source)
        if references:
            return references
    paper = client.search_title(text(metadata.get("title") or document.get("title")))
    references = _s2_references(as_mapping(paper), source="semantic_scholar_title")
    return references


def _references_section(markdown: str) -> str:
    match = REFERENCE_HEADING.search(markdown)
    if not match:
        return ""
    start = match.end()
    next_heading = NEXT_HEADING.search(markdown, start)
    return markdown[start : next_heading.start() if next_heading else len(markdown)]


def _guess_title(raw: str) -> str:
    cleaned = normalize_space(raw)
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    quoted_title = re.search(r'[“"]([^”"\n]{3,500})[”"]', cleaned)
    if quoted_title:
        return quoted_title.group(1).strip(" .,-;:")[:300]
    year_match = YEAR.search(cleaned)
    if year_match:
        cleaned = cleaned[: year_match.start()].strip(" .,-;:")
    parts = [part.strip() for part in re.split(r"\.\s+", cleaned) if part.strip()]
    if len(parts) >= 2 and re.search(r",\s*[A-Z]\.($|\s)", parts[0]):
        return parts[1][:300]
    if len(parts) >= 2 and len(parts[0].split()) <= 3:
        return parts[1][:300]
    for part in parts:
        if len(part.split()) >= 3 and not re.search(r",\s*[A-Z]\.($|\s)", part):
            return part[:300]
    return cleaned[:300]


def _local_references(markdown_path: object) -> list[JsonObject]:
    path = Path(text(markdown_path))
    if not path.is_file():
        return []
    section = _references_section(path.read_text(encoding="utf-8", errors="replace"))
    if not section:
        return []
    references: list[JsonObject] = []
    for match in REFERENCE_ENTRY.finditer(section):
        index = text(match.group(1) or match.group(2))
        raw = normalize_space(match.group(3))
        if len(raw) < 20:
            continue
        title = _guess_title(raw)
        year_match = YEAR.search(raw)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
        references.append(
            _clean_reference(
                {
                    "paperId": f"local-ref:{digest}",
                    "title": title,
                    "year": int(year_match.group(1)) if year_match else None,
                    "raw_reference": raw,
                    "local_index": index or None,
                },
                source="local_references_section",
                confidence=0.55,
            )
        )
        if len(references) >= 300:
            break
    return references


def _upstream_references(document_manifest: JsonObject) -> dict[str, list[JsonObject]]:
    documents = [
        as_mapping(value) for value in as_list(document_manifest.get("documents"))
    ]
    metadata_by_id = {
        text(document.get("paper_id")): as_mapping(document.get("metadata"))
        for document in documents
    }
    result: dict[str, list[JsonObject]] = {}
    edge_path = Path(text(document_manifest.get("paper_edges_path")))
    if not edge_path.is_file():
        return result
    for value in read_jsonl(edge_path):
        edge = as_mapping(value)
        if edge.get("relation") != "cites":
            continue
        source = text(edge.get("source_paper_id"))
        target = text(edge.get("target_paper_id"))
        if not source or not target:
            continue
        metadata = metadata_by_id.get(target, {})
        result.setdefault(source, []).append(
            _clean_reference(
                {
                    "paperId": target,
                    "title": metadata.get("title"),
                    "venue": metadata.get("venue"),
                    "year": metadata.get("year"),
                    "externalIds": _external_ids(metadata),
                    "url": metadata.get("url"),
                    "abstract": metadata.get("abstract"),
                    "authors": metadata.get("authors"),
                },
                source="paper_collect_edges",
                confidence=1.0,
            )
        )
    return result


def build_reference_map(document_manifest: JsonObject) -> tuple[dict[str, list[JsonObject]], JsonObject]:
    upstream = _upstream_references(document_manifest)
    client = SemanticScholarClient()
    result: dict[str, list[JsonObject]] = {}
    s2_documents = 0
    local_documents = 0
    for document in [
        as_mapping(value) for value in as_list(document_manifest.get("documents"))
    ]:
        if document.get("status") != "parsed":
            continue
        paper_id = text(document.get("paper_id"))
        if not paper_id:
            continue
        upstream_refs = upstream.get(paper_id, [])
        s2_refs = _resolve_semantic_references(client, document)
        if s2_refs:
            s2_documents += 1
        local_refs = [] if s2_refs else _local_references(document.get("markdown_path"))
        if local_refs:
            local_documents += 1
        result[paper_id] = _merge_references([upstream_refs, s2_refs, local_refs])
    summary: JsonObject = {
        "upstream_edge_documents": sum(1 for refs in upstream.values() if refs),
        "semantic_scholar_documents": s2_documents,
        "local_reference_documents": local_documents,
        "total_references": sum(len(refs) for refs in result.values()),
        "semantic_scholar_enabled": not client.disabled,
    }
    return result, summary
