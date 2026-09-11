from __future__ import annotations

import html
import json
import os
import re
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path

from .common import (
    JsonObject,
    artifact_paths,
    as_list,
    as_mapping,
    atomic_write_json,
    atomic_write_jsonl,
    compact_key,
    normalize_name,
    normalize_space,
    read_json,
    read_jsonl,
    sha256_file,
    stable_id,
    text,
    utc_now,
)


NODE_TYPES = {"Paper", "Core", "Baseline", "Dataset"}
EDGE_TYPES = {
    "introduces",
    "baseline_comparison",
    "evaluated_on",
    "core_relation",
    "cites",
    "recommended_with",
}
STOPWORDS = {
    "a",
    "an",
    "algorithm",
    "and",
    "approach",
    "based",
    "by",
    "data",
    "dataset",
    "datasets",
    "for",
    "framework",
    "from",
    "graph",
    "graphs",
    "in",
    "into",
    "learning",
    "method",
    "model",
    "models",
    "network",
    "networks",
    "of",
    "on",
    "or",
    "over",
    "paper",
    "study",
    "system",
    "task",
    "tasks",
    "the",
    "to",
    "under",
    "using",
    "via",
    "with",
}
GENERIC_BASELINE_NAMES = {
    "baselines",
    "existing methods",
    "gcn",
    "gcn based kgc model",
    "gcn based kgc models",
    "gcn-based kgc model",
    "gcn-based kgc models",
    "gcns",
    "graph convolutional network",
    "graph convolutional networks",
    "kge model",
    "kge models",
    "kgc model",
    "kgc models",
    "knowledge graph embedding",
    "knowledge graph embeddings",
    "previous methods",
    "state of the art methods",
    "state-of-the-art methods",
}
GENERIC_BASELINE_FRAGMENTS = {
    "based model",
    "based models",
    "completion model",
    "completion models",
    "embedding model",
    "embedding models",
}
SURVEY_TYPES = {"Survey/Review", "Tutorial/Educational", "Position Paper"}


def _tokens(value: object) -> set[str]:
    cleaned = normalize_name(value)
    cleaned = cleaned.replace("/", " ").replace("-", " ").replace("_", " ")
    return {
        token
        for token in re.findall(r"[a-z0-9+]+", cleaned)
        if len(token) > 1 and token not in STOPWORDS
    }


def _version_tokens(value: object) -> set[str]:
    cleaned = normalize_name(value)
    versions = set(re.findall(r"[a-z]*\d+|v\d+|\+\+", cleaned))
    if "++" in cleaned:
        versions.add("++")
    return versions


def _entity_signature(
    name: object, acronym: object, keywords: object, summary: object = None
) -> dict[str, object]:
    keyword_tokens: set[str] = set()
    for value in as_list(keywords):
        keyword_tokens.update(_tokens(value))
    return {
        "name_key": compact_key(name),
        "acronym_key": compact_key(acronym),
        "name_tokens": _tokens(name),
        "acronym_tokens": _tokens(acronym),
        "keyword_tokens": keyword_tokens,
        "summary_tokens": _tokens(summary),
        "version_tokens": _version_tokens(name) | _version_tokens(acronym),
    }


def _strong_alias_match(left: dict[str, object], right: dict[str, object]) -> bool:
    for left_key in ("name_key", "acronym_key"):
        for right_key in ("name_key", "acronym_key"):
            if text(left.get(left_key)) and left.get(left_key) == right.get(right_key):
                return True
    return False


def _hard_conflict(left: dict[str, object], right: dict[str, object]) -> bool:
    left_versions = set(left.get("version_tokens") or set())
    right_versions = set(right.get("version_tokens") or set())
    if ("++" in left_versions) != ("++" in right_versions):
        return True
    if left_versions and right_versions and left_versions != right_versions:
        if "++" in left_versions or "++" in right_versions:
            return True
    left_names = set(left.get("name_tokens") or set())
    right_names = set(right.get("name_tokens") or set())
    if left_names and right_names and left_names != right_names:
        if left_names.issubset(right_names) or right_names.issubset(left_names):
            left_length = len(" ".join(sorted(left_names)))
            right_length = len(" ".join(sorted(right_names)))
            return abs(left_length - right_length) >= 3
    return False


def _weighted_overlap(left: dict[str, object], right: dict[str, object]) -> float:
    def weights(signature: dict[str, object]) -> dict[str, int]:
        result: dict[str, int] = {}
        for token in set(signature.get("name_tokens") or set()):
            result[token] = result.get(token, 0) + 5
        for token in set(signature.get("acronym_tokens") or set()):
            result[token] = result.get(token, 0) + 6
        for token in set(signature.get("keyword_tokens") or set()):
            result[token] = result.get(token, 0) + 3
        for token in set(signature.get("summary_tokens") or set()):
            result[token] = result.get(token, 0) + 1
        return result

    left_weights = weights(left)
    right_weights = weights(right)
    keys = set(left_weights) | set(right_weights)
    if not keys:
        return 0.0
    intersection = sum(min(left_weights.get(key, 0), right_weights.get(key, 0)) for key in keys)
    union = sum(max(left_weights.get(key, 0), right_weights.get(key, 0)) for key in keys)
    return intersection / max(union, 1)


def _similarity(left: dict[str, object], right: dict[str, object]) -> float:
    if _hard_conflict(left, right):
        return 0.0
    if _strong_alias_match(left, right):
        return 1.0
    left_names = set(left.get("name_tokens") or set())
    right_names = set(right.get("name_tokens") or set())
    if not left_names or not right_names or not left_names.intersection(right_names):
        return 0.0
    return _weighted_overlap(left, right)


def _core_match_confident(entity: JsonObject, core_node: JsonObject) -> bool:
    entity_signature = _entity_signature(
        entity.get("name"),
        entity.get("acronym"),
        entity.get("keywords"),
        entity.get("summary"),
    )
    core_signature = _entity_signature(
        core_node.get("full_name") or core_node.get("name"),
        core_node.get("acronym"),
        core_node.get("keywords"),
        core_node.get("summary"),
    )
    if _hard_conflict(entity_signature, core_signature):
        return False
    if _strong_alias_match(entity_signature, core_signature):
        return True
    if not set(entity_signature.get("name_tokens") or set()).intersection(
        set(core_signature.get("name_tokens") or set())
    ):
        return False
    return _weighted_overlap(entity_signature, core_signature) >= 0.90


def _is_generic_baseline_name(name: object) -> bool:
    normalized = normalize_name(name)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized or normalized in GENERIC_BASELINE_NAMES:
        return True
    if len(normalized.split()) <= 5:
        return any(fragment in normalized for fragment in GENERIC_BASELINE_FRAGMENTS)
    return False


class EntityRegistry:
    def __init__(self, node_type: str) -> None:
        self.node_type = node_type
        self.signatures: dict[str, dict[str, object]] = {}
        self.by_name: dict[str, set[str]] = defaultdict(set)
        self.by_acronym: dict[str, set[str]] = defaultdict(set)
        self.by_name_token: dict[str, set[str]] = defaultdict(set)
        self.by_keyword_token: dict[str, set[str]] = defaultdict(set)

    def add(
        self,
        node_id: str,
        name: object,
        acronym: object,
        keywords: object,
        summary: object = None,
    ) -> None:
        signature = _entity_signature(name, acronym, keywords, summary)
        self.signatures[node_id] = signature
        name_key = text(signature.get("name_key"))
        acronym_key = text(signature.get("acronym_key"))
        if name_key:
            self.by_name[name_key].add(node_id)
        if acronym_key:
            self.by_acronym[acronym_key].add(node_id)
        for token in set(signature.get("name_tokens") or set()):
            self.by_name_token[token].add(node_id)
        for token in set(signature.get("keyword_tokens") or set()):
            self.by_keyword_token[token].add(node_id)

    def candidate_ids(self, signature: dict[str, object]) -> set[str]:
        candidates: set[str] = set()
        name_key = text(signature.get("name_key"))
        acronym_key = text(signature.get("acronym_key"))
        if name_key:
            candidates.update(self.by_name.get(name_key, set()))
        if acronym_key:
            candidates.update(self.by_acronym.get(acronym_key, set()))
        for token in set(signature.get("name_tokens") or set()):
            candidates.update(self.by_name_token.get(token, set()))
        if len(candidates) < 50:
            for token in set(signature.get("keyword_tokens") or set()):
                candidates.update(self.by_keyword_token.get(token, set()))
                if len(candidates) >= 50:
                    break
        return candidates

    def match(
        self,
        name: object,
        acronym: object,
        keywords: object,
        *,
        summary: object = None,
        threshold: float,
    ) -> tuple[str | None, float]:
        signature = _entity_signature(name, acronym, keywords, summary)
        best_id: str | None = None
        best_score = 0.0
        for node_id in self.candidate_ids(signature):
            score = _similarity(signature, self.signatures[node_id])
            if score > best_score:
                best_id = node_id
                best_score = score
        return (best_id, best_score) if best_score >= threshold else (None, best_score)


def _load_extractions(manifest: JsonObject) -> list[JsonObject]:
    result: list[JsonObject] = []
    for item_value in as_list(manifest.get("extractions")):
        item = as_mapping(item_value)
        if item.get("status") != "extracted":
            continue
        path = Path(text(item.get("path")))
        extraction = as_mapping(read_json(path))
        if extraction:
            result.append(extraction)
    return result


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_list_text(value: object) -> str:
    return json.dumps(as_list(value), ensure_ascii=False, sort_keys=True)


def _first(value: object) -> object:
    values = as_list(value)
    return values[0] if values else None


def _aliases(name: object, acronym: object) -> list[str]:
    values: list[str] = []
    for value in (normalize_space(name), normalize_space(acronym)):
        if value and compact_key(value) not in {compact_key(item) for item in values}:
            values.append(value)
    return values


def _paper_domains(metadata: JsonObject, extraction_metadata: JsonObject | None = None) -> list[str]:
    extraction_metadata = extraction_metadata or {}
    values = as_list(extraction_metadata.get("domain")) or as_list(metadata.get("fields_of_study"))
    return [normalize_space(value) for value in values if normalize_space(value)]


def _paper_types(extraction_metadata: JsonObject | None = None) -> list[str]:
    extraction_metadata = extraction_metadata or {}
    return [
        normalize_space(value)
        for value in as_list(extraction_metadata.get("paper_type"))
        if normalize_space(value)
    ]


def _is_survey_paper(extraction_metadata: JsonObject | None = None) -> bool:
    return bool(SURVEY_TYPES.intersection(set(_paper_types(extraction_metadata))))


def _citation_count(metadata: JsonObject, s2_metadata: JsonObject | None = None) -> object:
    s2_metadata = s2_metadata or {}
    return metadata.get("citation_count") or s2_metadata.get("citationCount")


def _reference_count(metadata: JsonObject, s2_metadata: JsonObject | None = None) -> object:
    s2_metadata = s2_metadata or {}
    return metadata.get("reference_count") or s2_metadata.get("referenceCount")


def _field_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return _json_text(value)


def _source_metadata(extraction: JsonObject | None, documents_by_paper: dict[str, JsonObject]) -> JsonObject:
    extraction = extraction or {}
    paper_id = text(extraction.get("paper_id"))
    document = documents_by_paper.get(paper_id, {})
    metadata = as_mapping(document.get("metadata"))
    extraction_metadata = as_mapping(extraction.get("metadata"))
    s2_metadata = as_mapping(metadata.get("s2_metadata"))
    return {
        "source_venue": metadata.get("venue"),
        "pub_year": metadata.get("year"),
        "paper_title": metadata.get("title") or document.get("title") or paper_id,
        "paper_domain": _paper_domains(metadata, extraction_metadata),
        "paper_type": _paper_types(extraction_metadata),
        "is_survey": _is_survey_paper(extraction_metadata),
        "code_url": extraction_metadata.get("code_url"),
        "structured_summary": as_mapping(extraction_metadata.get("structured_summary")),
        "urls": metadata.get("urls") or metadata.get("url") or [],
        "citation_count": _citation_count(metadata, s2_metadata),
        "reference_count": _reference_count(metadata, s2_metadata),
        "influential_citation_count": s2_metadata.get("influentialCitationCount"),
        "fields_of_study": metadata.get("fields_of_study") or s2_metadata.get("fieldsOfStudy") or [],
        "publication_date": metadata.get("publication_date") or s2_metadata.get("publicationDate"),
        "tldr": metadata.get("tldr") or as_mapping(s2_metadata.get("tldr")).get("text"),
    }


def _papergraph_node_fields(
    *,
    node_type: str,
    name: object,
    acronym: object = None,
    extraction: JsonObject | None = None,
    documents_by_paper: dict[str, JsonObject] | None = None,
    core: JsonObject | None = None,
    entity: JsonObject | None = None,
    ideation_resource: JsonObject | None = None,
) -> JsonObject:
    documents_by_paper = documents_by_paper or {}
    source = _source_metadata(extraction, documents_by_paper)
    core = core or {}
    entity = entity or {}
    citation = as_mapping(entity.get("citation"))
    s2_metadata = as_mapping(entity.get("s2_metadata"))
    full_name = normalize_space(name)
    aliases = _aliases(full_name, acronym)
    citation_urls = as_list(entity.get("urls")) or as_list(citation.get("urls"))
    citation_paper_id = text(
        entity.get("citation_paperId")
        or entity.get("citation_paper_id")
        or citation.get("paper_id")
    )
    return {
        "label": full_name,
        "full_name": full_name,
        "source_venue": source.get("source_venue"),
        "pub_year": source.get("pub_year"),
        "paper_title": source.get("paper_title"),
        "paper_domain": source.get("paper_domain"),
        "paper_type": source.get("paper_type"),
        "is_survey": source.get("is_survey"),
        "core_type": core.get("type") if node_type == "Core" else None,
        "summary": core.get("summary") if node_type == "Core" else entity.get("summary"),
        "keywords": as_list(core.get("keywords")) if node_type == "Core" else as_list(entity.get("keywords")),
        "insight": core.get("insight") if node_type == "Core" else entity.get("insight"),
        "quote": core.get("quote") if node_type == "Core" else entity.get("quote"),
        "aliases": aliases,
        "code_url": source.get("code_url"),
        "structured_summary": source.get("structured_summary"),
        "problems": as_list(as_mapping(ideation_resource).get("problems")),
        "components": as_list(as_mapping(ideation_resource).get("components")),
        "innovations": as_list(as_mapping(ideation_resource).get("innovations")),
        "limitations": as_list(as_mapping(ideation_resource).get("limitations")),
        "future_work": as_list(as_mapping(ideation_resource).get("future_work")),
        "urls": source.get("urls"),
        "citation_count": source.get("citation_count") or s2_metadata.get("citationCount"),
        "reference_count": source.get("reference_count") or s2_metadata.get("referenceCount"),
        "influential_citation_count": source.get("influential_citation_count") or s2_metadata.get("influentialCitationCount"),
        "fields_of_study": source.get("fields_of_study") or s2_metadata.get("fieldsOfStudy") or [],
        "publication_date": source.get("publication_date") or s2_metadata.get("publicationDate"),
        "tldr": source.get("tldr") or as_mapping(s2_metadata.get("tldr")).get("text"),
        "citation_paper_id": citation_paper_id or None,
        "citation_paperId": citation_paper_id or None,
        "citation_title": entity.get("citation_title") or citation.get("title"),
        "citation_venue": entity.get("citation_venue") or citation.get("venue"),
        "citation_year": entity.get("citation_year"),
        "citation_urls": citation_urls,
        "s2_metadata": s2_metadata,
    }


def _papergraph_edge_fields(relation: str, evidence: JsonObject | None) -> JsonObject:
    evidence = evidence or {}
    return {
        "edge_type": relation,
        "summary": evidence.get("summary"),
        "keywords": as_list(evidence.get("keywords")),
        "metrics": as_list(evidence.get("metrics")) or None,
        "insight": evidence.get("insight"),
        "quote": evidence.get("quote"),
    }


def _add_aliases(
    alias_rows: dict[tuple[str, str], JsonObject],
    node_id: str,
    aliases: list[str],
    *,
    paper_id: str | None,
) -> None:
    for alias in aliases:
        normalized = compact_key(alias)
        if not normalized:
            continue
        alias_rows.setdefault(
            (normalized, node_id),
            {
                "alias": alias,
                "normalized_alias": normalized,
                "target_id": node_id,
                "paper_id": paper_id,
            },
        )


def _add_edge(
    edges: dict[str, JsonObject],
    *,
    source: str,
    target: str,
    relation: str,
    evidence_paper_id: str | None,
    evidence: JsonObject | None = None,
    provenance: JsonObject | None = None,
) -> None:
    if relation not in EDGE_TYPES or source == target:
        return
    evidence = evidence or {}
    edge_id = stable_id(
        "edge", source, target, relation, evidence_paper_id or ""
    )
    record: JsonObject = {
        "id": edge_id,
        "source": source,
        "target": target,
        "relation": relation,
        "evidence_paper_id": evidence_paper_id,
        "evidence": evidence,
        "provenance": provenance or {},
        **_papergraph_edge_fields(relation, evidence),
    }
    edges[edge_id] = record


def _paper_metadata(document: JsonObject) -> JsonObject:
    metadata = as_mapping(document.get("metadata"))
    return {
        "year": metadata.get("year"),
        "venue": metadata.get("venue"),
        "authors": as_list(metadata.get("authors")),
        "abstract": metadata.get("abstract"),
        "tldr": metadata.get("tldr"),
        "fields_of_study": as_list(metadata.get("fields_of_study")),
        "publication_types": as_list(metadata.get("publication_types")),
        "citation_count": metadata.get("citation_count"),
        "reference_count": metadata.get("reference_count"),
        "external_ids": as_mapping(metadata.get("external_ids")),
        "url": metadata.get("url"),
        "download": as_mapping(metadata.get("download")),
    }


def _build_database(
    path: Path,
    nodes: list[JsonObject],
    edges: list[JsonObject],
    aliases: list[JsonObject],
    extractions: list[JsonObject],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    node_columns = [
        "id",
        "node_type",
        "name",
        "label",
        "full_name",
        "acronym",
        "paper_id",
        "source_venue",
        "pub_year",
        "paper_title",
        "paper_domain",
        "paper_type",
        "is_survey",
        "core_type",
        "summary",
        "keywords",
        "insight",
        "quote",
        "aliases",
        "code_url",
        "structured_summary",
        "problems",
        "components",
        "innovations",
        "limitations",
        "future_work",
        "urls",
        "citation_count",
        "reference_count",
        "influential_citation_count",
        "fields_of_study",
        "publication_date",
        "tldr",
        "citation_paper_id",
        "citation_paperId",
        "citation_title",
        "citation_venue",
        "citation_year",
        "citation_urls",
        "raw_json",
    ]
    edge_columns = [
        "id",
        "source",
        "target",
        "relation",
        "edge_type",
        "evidence_paper_id",
        "summary",
        "keywords",
        "metrics",
        "insight",
        "quote",
        "raw_json",
    ]

    def node_row(node: JsonObject) -> tuple[object, ...]:
        values: list[object] = []
        for column in node_columns:
            if column == "raw_json":
                values.append(json.dumps(node, ensure_ascii=False, sort_keys=True))
            elif column in {
                "paper_domain",
                "paper_type",
                "keywords",
                "aliases",
                "structured_summary",
                "problems",
                "components",
                "innovations",
                "limitations",
                "future_work",
                "urls",
                "fields_of_study",
                "citation_urls",
            }:
                values.append(_field_text(node.get(column)))
            elif column == "is_survey":
                values.append("true" if node.get(column) is True else "false" if node.get(column) is False else _field_text(node.get(column)))
            else:
                values.append(_field_text(node.get(column)))
        return tuple(values)

    def edge_row(edge: JsonObject) -> tuple[object, ...]:
        values: list[object] = []
        for column in edge_columns:
            if column == "raw_json":
                values.append(json.dumps(edge, ensure_ascii=False, sort_keys=True))
            elif column in {"keywords", "metrics"}:
                values.append(_field_text(edge.get(column)))
            else:
                values.append(_field_text(edge.get(column)))
        return tuple(values)

    try:
        with sqlite3.connect(temporary) as database:
            database.execute("PRAGMA foreign_keys = ON")
            database.executescript(
                """
                CREATE TABLE nodes (
                    id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    label TEXT,
                    full_name TEXT,
                    acronym TEXT,
                    paper_id TEXT,
                    source_venue TEXT,
                    pub_year TEXT,
                    paper_title TEXT,
                    paper_domain TEXT,
                    paper_type TEXT,
                    is_survey TEXT,
                    core_type TEXT,
                    summary TEXT,
                    keywords TEXT,
                    insight TEXT,
                    quote TEXT,
                    aliases TEXT,
                    code_url TEXT,
                    structured_summary TEXT,
                    problems TEXT,
                    components TEXT,
                    innovations TEXT,
                    limitations TEXT,
                    future_work TEXT,
                    urls TEXT,
                    citation_count TEXT,
                    reference_count TEXT,
                    influential_citation_count TEXT,
                    fields_of_study TEXT,
                    publication_date TEXT,
                    tldr TEXT,
                    citation_paper_id TEXT,
                    citation_paperId TEXT,
                    citation_title TEXT,
                    citation_venue TEXT,
                    citation_year TEXT,
                    citation_urls TEXT,
                    raw_json TEXT NOT NULL
                );
                CREATE TABLE edges (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL REFERENCES nodes(id),
                    target TEXT NOT NULL REFERENCES nodes(id),
                    relation TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    evidence_paper_id TEXT,
                    summary TEXT,
                    keywords TEXT,
                    metrics TEXT,
                    insight TEXT,
                    quote TEXT,
                    raw_json TEXT NOT NULL
                );
                CREATE TABLE aliases (
                    normalized_alias TEXT NOT NULL,
                    target_id TEXT NOT NULL REFERENCES nodes(id),
                    alias TEXT NOT NULL,
                    paper_id TEXT,
                    PRIMARY KEY (normalized_alias, target_id)
                );
                CREATE TABLE paper_extractions (
                    paper_id TEXT PRIMARY KEY,
                    raw_json TEXT NOT NULL
                );
                CREATE TABLE build_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE INDEX idx_nodes_type ON nodes(node_type);
                CREATE INDEX idx_nodes_paper ON nodes(paper_id);
                CREATE INDEX idx_nodes_year ON nodes(pub_year);
                CREATE INDEX idx_nodes_survey ON nodes(is_survey);
                CREATE INDEX idx_nodes_citation_count ON nodes(citation_count);
                CREATE INDEX idx_nodes_citation_pid ON nodes(citation_paperId);
                CREATE INDEX idx_edges_source ON edges(source);
                CREATE INDEX idx_edges_target ON edges(target);
                CREATE INDEX idx_edges_relation ON edges(relation);
                CREATE INDEX idx_edges_type ON edges(edge_type);
                CREATE INDEX idx_aliases_key ON aliases(normalized_alias);
                CREATE INDEX idx_aliases_paper ON aliases(paper_id);
                """
            )
            database.executemany(
                f"""
                INSERT INTO nodes({', '.join(node_columns)})
                VALUES({', '.join('?' for _ in node_columns)})
                """,
                [node_row(node) for node in nodes],
            )
            database.executemany(
                f"""
                INSERT INTO edges({', '.join(edge_columns)})
                VALUES({', '.join('?' for _ in edge_columns)})
                """,
                [edge_row(edge) for edge in edges],
            )
            database.executemany(
                """
                INSERT INTO aliases(normalized_alias, target_id, alias, paper_id)
                VALUES(?, ?, ?, ?)
                """,
                [
                    (
                        text(alias.get("normalized_alias")),
                        text(alias.get("target_id")),
                        text(alias.get("alias")),
                        text(alias.get("paper_id")) or None,
                    )
                    for alias in aliases
                ],
            )
            database.executemany(
                "INSERT INTO paper_extractions(paper_id, raw_json) VALUES(?, ?)",
                [
                    (
                        text(extraction.get("paper_id")),
                        json.dumps(extraction, ensure_ascii=False, sort_keys=True),
                    )
                    for extraction in extractions
                ],
            )
            database.executemany(
                "INSERT INTO build_meta(key, value) VALUES(?, ?)",
                [
                    ("schema_version", "xlab.method_graph.v2"),
                    ("papergraph_compatibility", "step4v2"),
                    ("generated_at", utc_now()),
                ],
            )
            try:
                database.execute(
                    """
                    CREATE VIRTUAL TABLE node_fts USING fts5(
                        id UNINDEXED,
                        full_name,
                        acronym,
                        paper_title,
                        summary,
                        keywords,
                        aliases,
                        citation_title,
                        tldr,
                        node_type UNINDEXED,
                        content=''
                    )
                    """
                )
                database.executemany(
                    """
                    INSERT INTO node_fts(
                        id, full_name, acronym, paper_title, summary, keywords,
                        aliases, citation_title, tldr, node_type
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            text(node.get("id")),
                            text(node.get("full_name") or node.get("name")),
                            text(node.get("acronym")),
                            text(node.get("paper_title")),
                            text(node.get("summary")),
                            _field_text(node.get("keywords")),
                            _field_text(node.get("aliases")),
                            text(node.get("citation_title")),
                            text(node.get("tldr")),
                            text(node.get("node_type")),
                        )
                        for node in nodes
                    ],
                )
            except sqlite3.OperationalError:
                pass
            database.commit()
            integrity = database.execute("PRAGMA integrity_check").fetchone()
            if not integrity or integrity[0] != "ok":
                raise RuntimeError(f"SQLite integrity check failed: {integrity}")
            foreign_keys = database.execute("PRAGMA foreign_key_check").fetchall()
            if foreign_keys:
                raise RuntimeError(
                    f"SQLite foreign key check failed: {foreign_keys[:5]}"
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _visual_label(node: JsonObject) -> str:
    return text(node.get("full_name") or node.get("name") or node.get("id"))


def _visual_color(node_type: str) -> str:
    return {
        "Paper": "#d9d9d9",
        "Core": "#ffd166",
        "Baseline": "#7bdff2",
        "Dataset": "#b2f7b6",
    }.get(node_type, "#d9d9d9")


def _visual_edge_color(relation: str) -> str:
    return {
        "introduces": "#999999",
        "baseline_comparison": "#3a86ff",
        "evaluated_on": "#2a9d8f",
        "core_relation": "#e76f51",
        "cites": "#8d99ae",
        "recommended_with": "#adb5bd",
    }.get(relation, "#999999")


def _build_visualization(
    path: Path,
    nodes: list[JsonObject],
    edges: list[JsonObject],
    *,
    max_nodes: int = 240,
) -> None:
    semantic_nodes = [node for node in nodes if node.get("node_type") != "Paper"]
    selected_nodes = semantic_nodes[:max_nodes] or nodes[:max_nodes]
    selected_ids = {text(node.get("id")) for node in selected_nodes}
    selected_edges = [
        edge
        for edge in edges
        if text(edge.get("source")) in selected_ids and text(edge.get("target")) in selected_ids
    ][: max_nodes * 3]
    vis_nodes = [
        {
            "id": node.get("id"),
            "label": _visual_label(node),
            "group": node.get("node_type"),
            "color": _visual_color(text(node.get("node_type"))),
            "title": html.escape(text(node.get("summary"))[:500]),
        }
        for node in selected_nodes
    ]
    vis_edges = [
        {
            "from": edge.get("source"),
            "to": edge.get("target"),
            "label": edge.get("edge_type") or edge.get("relation"),
            "color": _visual_edge_color(text(edge.get("relation"))),
            "arrows": "to",
            "title": html.escape(text(edge.get("summary"))[:500]),
        }
        for edge in selected_edges
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    html_text = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>XLab Knowledge Graph Preview</title>
<script src=\"https://unpkg.com/vis-network/standalone/umd/vis-network.min.js\"></script>
<style>
html, body, #network {{ height: 100%; margin: 0; font-family: system-ui, sans-serif; }}
#meta {{ position: fixed; top: 12px; left: 12px; z-index: 10; background: rgba(255,255,255,.92); border: 1px solid #ddd; border-radius: 8px; padding: 10px 12px; max-width: 420px; }}
#network {{ background: #fff; }}
</style>
</head>
<body>
<div id=\"meta\"><b>XLab Knowledge Graph Preview</b><br>{len(selected_nodes)} nodes, {len(selected_edges)} edges. This static artifact is written under the run directory and does not auto-open a browser.</div>
<div id=\"network\"></div>
<script>
const nodes = new vis.DataSet({json.dumps(vis_nodes, ensure_ascii=False)});
const edges = new vis.DataSet({json.dumps(vis_edges, ensure_ascii=False)});
new vis.Network(document.getElementById('network'), {{nodes, edges}}, {{
  interaction: {{hover: true, navigationButtons: true}},
  physics: {{enabled: false}},
  layout: {{improvedLayout: true}},
  edges: {{font: {{align: 'middle'}}, smooth: true}}
}});
</script>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def _build_visualization_artifacts(
    directory: Path, nodes: list[JsonObject], edges: list[JsonObject]
) -> JsonObject:
    overview = directory / "overview.html"
    _build_visualization(overview, nodes, edges)
    return {
        "enabled": True,
        "overview_html": str(overview),
        "auto_opened": False,
    }


def build_graph(
    run_dir: Path,
    document_manifest: JsonObject,
    extraction_manifest: JsonObject,
) -> JsonObject:
    paths = artifact_paths(run_dir)
    documents = [
        as_mapping(value) for value in as_list(document_manifest.get("documents"))
    ]
    documents_by_paper = {text(document.get("paper_id")): document for document in documents}
    extractions = _load_extractions(extraction_manifest)
    extraction_by_paper = {
        text(value.get("paper_id")): value for value in extractions
    }
    nodes: dict[str, JsonObject] = {}
    edges: dict[str, JsonObject] = {}
    alias_rows: dict[tuple[str, str], JsonObject] = {}
    paper_node_by_id: dict[str, str] = {}
    core_node_by_alias: dict[str, set[str]] = defaultdict(set)
    cores_by_paper: dict[str, list[str]] = defaultdict(list)

    for document in documents:
        paper_id = text(document.get("paper_id"))
        if not paper_id:
            continue
        node_id = stable_id("paper", paper_id)
        paper_node_by_id[paper_id] = node_id
        aliases = _aliases(document.get("title") or paper_id, None)
        nodes[node_id] = {
            "id": node_id,
            "node_type": "Paper",
            "name": normalize_space(document.get("title")) or paper_id,
            "acronym": None,
            "paper_id": paper_id,
            "aliases": aliases,
            "metadata": _paper_metadata(document),
            "provenance": {
                "source_manifest": document.get("source_manifest"),
                "source_manifest_sha256": document.get("source_manifest_sha256"),
                "paper_collect_manifest": document.get("paper_collect_manifest"),
                "paper_collect_manifest_sha256": document.get("paper_collect_manifest_sha256"),
                "paper_collect_run_id": document.get("paper_collect_run_id"),
                "source_pdf_root": document.get("source_pdf_root"),
                "source_pdf": document.get("source_pdf"),
                "source_sha256": document.get("source_sha256"),
                "source_download_sha256": document.get("source_download_sha256"),
                "paper_edges_path": document.get("paper_edges_path"),
                "paper_edges_sha256": document.get("paper_edges_sha256"),
                "markdown_path": document.get("markdown_path"),
                "content_list_path": document.get("content_list_path"),
                "middle_json_path": document.get("middle_json_path"),
                "parser": document.get("parser"),
                "markdown_sha256": document.get("markdown_sha256"),
                "content_list_sha256": document.get("content_list_sha256"),
                "middle_json_sha256": document.get("middle_json_sha256"),
                "document_status": document.get("status"),
            },
        }
        _add_aliases(alias_rows, node_id, aliases, paper_id=paper_id)

    for extraction in extractions:
        paper_id = text(extraction.get("paper_id"))
        paper_node = paper_node_by_id.get(paper_id)
        if not paper_node:
            continue
        ideation = as_mapping(extraction.get("ideation_resource"))
        core_values = [as_mapping(value) for value in as_list(ideation.get("core_contributions"))]
        single_core = len(core_values) == 1
        for core in core_values:
            name = normalize_space(core.get("name"))
            if not name:
                continue
            acronym = normalize_space(core.get("acronym")) or None
            node_id = stable_id("core", paper_id, compact_key(name))
            ideation_resource = {
                key: [
                    as_mapping(value)
                    for value in as_list(ideation.get(key))
                    if text(as_mapping(value).get("related_to_core")) in {"", name}
                    or single_core
                ]
                for key in (
                    "problems",
                    "components",
                    "innovations",
                    "limitations",
                    "future_work",
                )
            }
            aliases = _aliases(name, acronym)
            nodes[node_id] = {
                "id": node_id,
                "node_type": "Core",
                "name": name,
                "acronym": acronym,
                "paper_id": paper_id,
                "aliases": aliases,
                "core_type": core.get("type"),
                "keywords": as_list(core.get("keywords")),
                "summary": core.get("summary"),
                "insight": core.get("insight"),
                "quote": core.get("quote"),
                "fallback_from_title": core.get("fallback_from_title") is True,
                "ideation_resource": ideation_resource,
                "provenance": extraction.get("provenance"),
                **_papergraph_node_fields(
                    node_type="Core",
                    name=name,
                    acronym=acronym,
                    extraction=extraction,
                    documents_by_paper=documents_by_paper,
                    core=core,
                    ideation_resource=ideation_resource,
                ),
            }
            cores_by_paper[paper_id].append(node_id)
            _add_aliases(alias_rows, node_id, aliases, paper_id=paper_id)
            for alias in aliases:
                core_node_by_alias[compact_key(alias)].add(node_id)
            _add_edge(
                edges,
                source=paper_node,
                target=node_id,
                relation="introduces",
                evidence_paper_id=paper_id,
                evidence={
                    "summary": core.get("summary"),
                    "quote": core.get("quote"),
                },
                provenance=as_mapping(extraction.get("provenance")),
            )

    baseline_registry = EntityRegistry("Baseline")
    dataset_registry = EntityRegistry("Dataset")
    graph_merge_stats: JsonObject = {
        "baseline_total": 0,
        "baseline_merged_citation_core": 0,
        "baseline_merged_registry": 0,
        "baseline_new": 0,
        "baseline_skipped_generic": 0,
        "baseline_skipped_survey": 0,
        "dataset_total": 0,
        "dataset_merged_registry": 0,
        "dataset_new": 0,
    }
    for extraction in extractions:
        paper_id = text(extraction.get("paper_id"))
        source_cores = cores_by_paper.get(paper_id, [])
        if not source_cores:
            continue
        graph_data = as_mapping(extraction.get("graph_data"))
        for relation, node_type, values, registry, default_threshold in (
            (
                "baseline_comparison",
                "Baseline",
                as_list(graph_data.get("baselines")),
                baseline_registry,
                0.92,
            ),
            (
                "evaluated_on",
                "Dataset",
                as_list(graph_data.get("datasets")),
                dataset_registry,
                0.90,
            ),
        ):
            for value in values:
                entity = as_mapping(value)
                name = normalize_space(entity.get("name"))
                if not name:
                    continue
                acronym = normalize_space(entity.get("acronym")) or None
                keywords = as_list(entity.get("keywords"))
                summary = entity.get("summary")
                entity_aliases = _aliases(name, acronym)
                target_id: str | None = None
                if node_type == "Baseline":
                    graph_merge_stats["baseline_total"] = int(graph_merge_stats["baseline_total"]) + 1
                    if _is_generic_baseline_name(name):
                        graph_merge_stats["baseline_skipped_generic"] = int(graph_merge_stats["baseline_skipped_generic"]) + 1
                        continue
                    cited_paper_id = text(entity.get("citation_paperId") or entity.get("citation_paper_id"))
                    for cited_core_id in cores_by_paper.get(cited_paper_id, []):
                        cited_core = nodes.get(cited_core_id, {})
                        if cited_core.get("is_survey") is True:
                            graph_merge_stats["baseline_skipped_survey"] = int(graph_merge_stats["baseline_skipped_survey"]) + 1
                            continue
                        if _core_match_confident(entity, cited_core):
                            target_id = cited_core_id
                            graph_merge_stats["baseline_merged_citation_core"] = int(graph_merge_stats["baseline_merged_citation_core"]) + 1
                            break
                    if not target_id:
                        core_matches: set[str] = set()
                        for alias in entity_aliases:
                            core_matches.update(core_node_by_alias.get(compact_key(alias), set()))
                        if len(core_matches) == 1:
                            candidate = next(iter(core_matches))
                            if _core_match_confident(entity, nodes.get(candidate, {})):
                                target_id = candidate
                                graph_merge_stats["baseline_merged_citation_core"] = int(graph_merge_stats["baseline_merged_citation_core"]) + 1
                    threshold = 0.88 if text(entity.get("citation_paperId") or entity.get("citation_paper_id")) else default_threshold
                else:
                    graph_merge_stats["dataset_total"] = int(graph_merge_stats["dataset_total"]) + 1
                    threshold = 0.84 if text(entity.get("citation_paperId") or entity.get("citation_paper_id")) else default_threshold
                if not target_id:
                    target_id, _ = registry.match(
                        name,
                        acronym,
                        keywords,
                        summary=summary,
                        threshold=threshold,
                    )
                    if target_id:
                        key = "baseline_merged_registry" if node_type == "Baseline" else "dataset_merged_registry"
                        graph_merge_stats[key] = int(graph_merge_stats[key]) + 1
                if not target_id:
                    target_id = stable_id(node_type, compact_key(name))
                    nodes[target_id] = {
                        "id": target_id,
                        "node_type": node_type,
                        "name": name,
                        "acronym": acronym,
                        "paper_id": None,
                        "aliases": entity_aliases,
                        "keywords": keywords,
                        "summary": summary,
                        "citation": {
                            "paper_id": entity.get("citation_paper_id")
                            or entity.get("citation_paperId"),
                            "title": entity.get("citation_title"),
                            "venue": entity.get("citation_venue"),
                            "urls": as_list(entity.get("urls")),
                        },
                        "provenance": extraction.get("provenance"),
                        **_papergraph_node_fields(
                            node_type=node_type,
                            name=name,
                            acronym=acronym,
                            extraction=extraction,
                            documents_by_paper=documents_by_paper,
                            entity=entity,
                        ),
                    }
                    registry.add(target_id, name, acronym, keywords, summary=summary)
                    key = "baseline_new" if node_type == "Baseline" else "dataset_new"
                    graph_merge_stats[key] = int(graph_merge_stats[key]) + 1
                elif text(as_mapping(nodes.get(target_id, {})).get("node_type")) == node_type:
                    registry.add(target_id, name, acronym, keywords, summary=summary)
                target_node = nodes[target_id]
                existing_aliases = [
                    text(value) for value in as_list(target_node.get("aliases"))
                ]
                existing_keys = {compact_key(value) for value in existing_aliases}
                for alias in entity_aliases:
                    if compact_key(alias) not in existing_keys:
                        existing_aliases.append(alias)
                        existing_keys.add(compact_key(alias))
                target_node["aliases"] = existing_aliases
                target_node["full_name"] = target_node.get("full_name") or target_node.get("name")
                _add_aliases(
                    alias_rows, target_id, entity_aliases, paper_id=paper_id
                )
                for source_core in source_cores:
                    _add_edge(
                        edges,
                        source=source_core,
                        target=target_id,
                        relation=relation,
                        evidence_paper_id=paper_id,
                        evidence={
                            "keywords": keywords,
                            "summary": summary,
                            "metrics": entity.get("metrics"),
                            "insight": entity.get("insight"),
                            "quote": entity.get("quote"),
                        },
                        provenance=as_mapping(extraction.get("provenance")),
                    )

        ideation = as_mapping(extraction.get("ideation_resource"))
        core_alias_map: dict[str, str] = {}
        for node_id in source_cores:
            node = nodes[node_id]
            for alias in as_list(node.get("aliases")):
                core_alias_map[compact_key(alias)] = node_id
        for relation_value in as_list(ideation.get("core_relations")):
            relation = as_mapping(relation_value)
            source = core_alias_map.get(compact_key(relation.get("source")))
            target = core_alias_map.get(compact_key(relation.get("target")))
            if source and target:
                _add_edge(
                    edges,
                    source=source,
                    target=target,
                    relation="core_relation",
                    evidence_paper_id=paper_id,
                    evidence={
                        key: relation.get(key)
                        for key in (
                            "keywords",
                            "summary",
                            "metrics",
                            "insight",
                            "quote",
                        )
                    },
                    provenance=as_mapping(extraction.get("provenance")),
                )

    edge_file = Path(text(document_manifest.get("paper_edges_path")))
    if edge_file.is_file():
        for value in read_jsonl(edge_file):
            relation_record = as_mapping(value)
            relation = text(relation_record.get("relation"))
            source_paper_id = text(relation_record.get("source_paper_id"))
            target_paper_id = text(relation_record.get("target_paper_id"))
            source = paper_node_by_id.get(source_paper_id)
            target = paper_node_by_id.get(target_paper_id)
            if source and target and relation in {"cites", "recommended_with"}:
                _add_edge(
                    edges,
                    source=source,
                    target=target,
                    relation=relation,
                    evidence_paper_id=source_paper_id,
                    evidence={"discovered_from": relation_record.get("discovered_from")},
                    provenance={
                        "source_path": str(edge_file),
                        "source_sha256": document_manifest.get("paper_edges_sha256"),
                        "paper_collect_manifest": document_manifest.get(
                            "prior_manifest_path"
                        ),
                        "paper_collect_manifest_sha256": document_manifest.get(
                            "prior_manifest_sha256"
                        ),
                    },
                )

    node_values = sorted(nodes.values(), key=lambda item: text(item.get("id")))
    edge_values = sorted(edges.values(), key=lambda item: text(item.get("id")))
    alias_values = sorted(
        alias_rows.values(),
        key=lambda item: (
            text(item.get("normalized_alias")),
            text(item.get("target_id")),
        ),
    )
    endpoint_ids = {text(node.get("id")) for node in node_values}
    invalid_edges = [
        edge
        for edge in edge_values
        if text(edge.get("source")) not in endpoint_ids
        or text(edge.get("target")) not in endpoint_ids
    ]
    if invalid_edges:
        raise RuntimeError(f"{len(invalid_edges)} graph edges have invalid endpoints")
    if any(text(node.get("node_type")) not in NODE_TYPES for node in node_values):
        raise RuntimeError("graph contains an unsupported node type")

    atomic_write_jsonl(paths["nodes"], node_values)
    atomic_write_jsonl(paths["edges"], edge_values)
    atomic_write_jsonl(paths["aliases"], alias_values)
    _build_database(
        paths["database"],
        node_values,
        edge_values,
        alias_values,
        extractions,
    )
    visualization = _build_visualization_artifacts(
        paths["visualizations"], node_values, edge_values
    )
    counts = {
        "papers": sum(node.get("node_type") == "Paper" for node in node_values),
        "cores": sum(node.get("node_type") == "Core" for node in node_values),
        "baselines": sum(
            node.get("node_type") == "Baseline" for node in node_values
        ),
        "datasets": sum(
            node.get("node_type") == "Dataset" for node in node_values
        ),
        "nodes": len(node_values),
        "edges": len(edge_values),
        "aliases": len(alias_values),
        "extractions": len(extractions),
    }
    input_provenance = {
        "input_directory": document_manifest.get("input_directory"),
        "paper_set_path": document_manifest.get("paper_set_path"),
        "paper_set_sha256": document_manifest.get("paper_set_sha256"),
        "paper_edges_path": document_manifest.get("paper_edges_path"),
        "paper_edges_sha256": document_manifest.get("paper_edges_sha256"),
        "prior_manifest_path": document_manifest.get("prior_manifest_path"),
        "prior_manifest_sha256": document_manifest.get("prior_manifest_sha256"),
        "paper_collect_run_id": document_manifest.get("paper_collect_run_id"),
        "paper_collect_skill_version": document_manifest.get(
            "paper_collect_skill_version"
        ),
        "pdf_root": document_manifest.get("pdf_root"),
    }
    graph: JsonObject = {
        "schema_version": "xlab.method_graph.v2",
        "papergraph_compatibility": {
            "algorithm": "PaperGraph Step 1/2/3/4",
            "merge": graph_merge_stats,
            "xlab_extensions": ["MinerU mandatory parse", "Paper nodes", "provenance audit"],
        },
        "generated_at": utc_now(),
        "source_artifact": document_manifest.get("paper_set_path"),
        "source_artifact_sha256": document_manifest.get("paper_set_sha256"),
        "input_provenance": input_provenance,
        "paper_documents_path": str(paths["documents"]),
        "paper_extractions_path": str(paths["extraction_manifest"]),
        "nodes_path": str(paths["nodes"]),
        "edges_path": str(paths["edges"]),
        "aliases_path": str(paths["aliases"]),
        "database_path": str(paths["database"]),
        "visualization_path": visualization.get("overview_html"),
        "visualization": visualization,
        "nodes": node_values,
        "edges": edge_values,
        "aliases": alias_values,
        "counts": counts,
    }
    atomic_write_json(paths["graph"], graph)
    return graph
