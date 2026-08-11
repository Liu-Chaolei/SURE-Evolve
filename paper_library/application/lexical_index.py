from __future__ import annotations

import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from paper_library.config import PaperLibraryConfig
from paper_library.errors import ErrorCode, PaperLibraryError
from paper_library.schemas.evidence_record import EvidenceRecord
from paper_library.schemas.research_design_card import ResearchDesignCard
from paper_library.storage.corpus import load_corpus_manifest
from paper_library.storage.jsonl import read_jsonl
from paper_library.storage.layout import BuildLayout
from paper_library.storage.manifest import CorpusManifest
from paper_library.utils.hashing import canonical_json, hash_canonical, sha256_file

LEXICAL_INDEX_SCHEMA_VERSION = "1.1.0"
_QUERY_KINDS = {"evidence", "design"}


@dataclass(frozen=True)
class LexicalIndexResult:
    corpus_version: str
    index_path: Path
    evidence_count: int
    design_count: int
    cache_hit: bool


@dataclass(frozen=True)
class LexicalQueryHit:
    record_id: str
    kind: str
    paper_id: str
    record_type: str
    score: float
    record: EvidenceRecord | ResearchDesignCard


@dataclass(frozen=True)
class LexicalQueryResult:
    corpus_version: str
    kind: str
    query: str
    hits: tuple[LexicalQueryHit, ...]


def build_lexical_index(
    config: PaperLibraryConfig,
    *,
    corpus_version: str | None = None,
    force: bool = False,
) -> LexicalIndexResult:
    if not config.indexing.lexical_enabled:
        raise PaperLibraryError(
            "Lexical indexing is disabled by configuration",
            code=ErrorCode.CONFIG_INVALID,
        )
    layout = BuildLayout(config.paths.build_root)
    layout.ensure()
    manifest, manifest_path = load_corpus_manifest(layout, corpus_version)
    index_path = layout.lexical_index(manifest.corpus_version)
    evidence_path = layout.resolve_inside(manifest.evidence_records.path)
    card_path = layout.resolve_inside(manifest.research_design_cards.path)
    evidence = list(read_jsonl(evidence_path, EvidenceRecord))
    cards = list(read_jsonl(card_path, ResearchDesignCard))
    expected = _expected_metadata(config, manifest, manifest_path, evidence, cards)
    if not force and _valid_existing_index(index_path, expected):
        return LexicalIndexResult(
            corpus_version=manifest.corpus_version,
            index_path=index_path,
            evidence_count=manifest.evidence_records.record_count,
            design_count=manifest.research_design_cards.record_count,
            cache_hit=True,
        )

    index_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{index_path.name}.", suffix=".tmp", dir=index_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_index(temporary, expected, evidence, cards)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, index_path)
        directory = os.open(index_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except sqlite3.Error as exc:
        raise PaperLibraryError(
            f"Failed to build lexical index: {exc}",
            code=ErrorCode.INDEX_FAILED,
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)

    return LexicalIndexResult(
        corpus_version=manifest.corpus_version,
        index_path=index_path,
        evidence_count=len(evidence),
        design_count=len(cards),
        cache_hit=False,
    )


def query_lexical_index(
    config: PaperLibraryConfig,
    query: str,
    *,
    kind: str,
    corpus_version: str | None = None,
    limit: int = 10,
    paper_id: str | None = None,
    record_type: str | None = None,
) -> LexicalQueryResult:
    if kind not in _QUERY_KINDS:
        raise ValueError(f"Invalid query kind: {kind}")
    if not query.strip():
        raise ValueError("Query text must not be empty")
    if limit < 1 or limit > 100:
        raise ValueError("Query limit must be between 1 and 100")

    layout = BuildLayout(config.paths.build_root)
    manifest, manifest_path = load_corpus_manifest(layout, corpus_version)
    index_path = layout.lexical_index(manifest.corpus_version)
    evidence = list(
        read_jsonl(layout.resolve_inside(manifest.evidence_records.path), EvidenceRecord)
    )
    cards = list(
        read_jsonl(
            layout.resolve_inside(manifest.research_design_cards.path),
            ResearchDesignCard,
        )
    )
    expected = _expected_metadata(config, manifest, manifest_path, evidence, cards)
    if not _valid_existing_index(index_path, expected):
        raise PaperLibraryError(
            f"Lexical index is missing or stale for {manifest.corpus_version}",
            code=ErrorCode.INDEX_FAILED,
        )

    table = "evidence_fts" if kind == "evidence" else "design_fts"
    clauses = [f"{table} MATCH ?"]
    parameters: list[object] = [query]
    if paper_id:
        clauses.append("paper_id = ?")
        parameters.append(paper_id)
    if record_type:
        clauses.append("record_type = ?")
        parameters.append(record_type)
    parameters.append(limit)
    statement = (
        "SELECT record_id, paper_id, record_type, record_json, "
        f"bm25({table}) AS score FROM {table} WHERE "
        + " AND ".join(clauses)
        + " ORDER BY score, record_id LIMIT ?"
    )

    try:
        connection = _read_only_connection(index_path)
        try:
            rows = connection.execute(statement, parameters).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise PaperLibraryError(
            f"Lexical query failed: {exc}",
            code=ErrorCode.INDEX_FAILED,
        ) from exc

    model_type = EvidenceRecord if kind == "evidence" else ResearchDesignCard
    hits = tuple(
        LexicalQueryHit(
            record_id=row[0],
            kind=kind,
            paper_id=row[1],
            record_type=row[2],
            score=float(row[4]),
            record=model_type.model_validate_json(row[3]),
        )
        for row in rows
    )
    return LexicalQueryResult(
        corpus_version=manifest.corpus_version,
        kind=kind,
        query=query,
        hits=hits,
    )


def _write_index(
    path: Path,
    metadata: dict[str, str],
    evidence: list[EvidenceRecord],
    cards: list[ResearchDesignCard],
) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute(
            "CREATE TABLE index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        for table in ("evidence_fts", "design_fts"):
            connection.execute(
                f"CREATE VIRTUAL TABLE {table} USING fts5("
                "record_id UNINDEXED, paper_id UNINDEXED, record_type UNINDEXED, "
                "title, body, metadata, record_json UNINDEXED, tokenize='unicode61')"
            )
        connection.executemany(
            "INSERT INTO index_metadata(key, value) VALUES (?, ?)",
            sorted(metadata.items()),
        )
        connection.executemany(
            "INSERT INTO evidence_fts VALUES (?, ?, ?, ?, ?, ?, ?)",
            (_evidence_row(record) for record in evidence),
        )
        connection.executemany(
            "INSERT INTO design_fts VALUES (?, ?, ?, ?, ?, ?, ?)",
            (_design_row(card) for card in cards),
        )
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise sqlite3.DatabaseError("lexical index integrity check failed")
        if connection.execute("SELECT count(*) FROM evidence_fts").fetchone()[0] != len(evidence):
            raise sqlite3.DatabaseError("evidence index count does not match corpus")
        if connection.execute("SELECT count(*) FROM design_fts").fetchone()[0] != len(cards):
            raise sqlite3.DatabaseError("design index count does not match corpus")
    finally:
        connection.close()


def _evidence_row(record: EvidenceRecord) -> tuple[str, ...]:
    facts = " ".join(
        " ".join(
            value
            for value in (
                fact.name,
                fact.metric,
                fact.dataset,
                fact.split,
                fact.unit,
            )
            if value
        )
        for fact in record.numeric_facts
    )
    conditions = " ".join(f"{key} {value}" for key, value in sorted(record.conditions.items()))
    return (
        record.evidence_id,
        record.paper_id,
        record.role.value,
        record.claim or "",
        " ".join(value for value in (record.claim, facts) if value),
        conditions,
        canonical_json(record.model_dump(mode="json")),
    )


def _design_row(card: ResearchDesignCard) -> tuple[str, ...]:
    statements = (
        card.research_problem.value,
        card.research_question.value,
        card.hypothesis.statement.value,
        card.tested_relationship.value,
        card.base_method.value,
        card.intervention_delta.value,
        card.candidate_system.value,
    )
    body = " ".join(
        value
        for value in (
            *statements,
            *card.observations,
            *card.applicability,
            *card.limitations,
        )
        if value
    )
    metadata = " ".join(
        (
            *card.baselines,
            *card.explicit_controls,
            *card.assumed_fixed,
            *card.training_objectives,
            *card.training_backends,
            *card.evaluation.datasets,
            *card.evaluation.splits,
            *card.evaluation.metrics,
            *card.evaluation.evaluation_backends,
        )
    )
    return (
        card.card_id,
        card.paper_id,
        card.card_type.value,
        card.title,
        body,
        metadata,
        canonical_json(card.model_dump(mode="json")),
    )


def _expected_metadata(
    config: PaperLibraryConfig,
    manifest: CorpusManifest,
    manifest_path: Path,
    evidence: list[EvidenceRecord],
    cards: list[ResearchDesignCard],
) -> dict[str, str]:
    index_config_hash = hash_canonical(config.indexing.model_dump(mode="json"))
    evidence_rows = sorted((_evidence_row(record) for record in evidence), key=lambda row: row[0])
    design_rows = sorted((_design_row(card) for card in cards), key=lambda row: row[0])
    return {
        "schema_version": LEXICAL_INDEX_SCHEMA_VERSION,
        "corpus_version": manifest.corpus_version,
        "corpus_manifest_sha256": sha256_file(manifest_path),
        "corpus_config_hash": manifest.config_hash,
        "index_config_hash": index_config_hash,
        "evidence_sha256": manifest.evidence_records.sha256,
        "evidence_count": str(manifest.evidence_records.record_count),
        "evidence_rows_hash": hash_canonical(evidence_rows),
        "design_sha256": manifest.research_design_cards.sha256,
        "design_count": str(manifest.research_design_cards.record_count),
        "design_rows_hash": hash_canonical(design_rows),
    }


def _valid_existing_index(path: Path, expected: dict[str, str]) -> bool:
    if not path.is_file():
        return False
    try:
        connection = _read_only_connection(path)
        try:
            metadata = dict(connection.execute("SELECT key, value FROM index_metadata"))
            if metadata != expected:
                return False
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                return False
            evidence_count = connection.execute("SELECT count(*) FROM evidence_fts").fetchone()[0]
            design_count = connection.execute("SELECT count(*) FROM design_fts").fetchone()[0]
            if evidence_count != int(expected["evidence_count"]) or design_count != int(
                expected["design_count"]
            ):
                return False
            evidence_rows = connection.execute(
                "SELECT record_id, paper_id, record_type, title, body, metadata, record_json "
                "FROM evidence_fts ORDER BY record_id"
            ).fetchall()
            design_rows = connection.execute(
                "SELECT record_id, paper_id, record_type, title, body, metadata, record_json "
                "FROM design_fts ORDER BY record_id"
            ).fetchall()
            return hash_canonical(evidence_rows) == expected[
                "evidence_rows_hash"
            ] and hash_canonical(design_rows) == expected["design_rows_hash"]
        finally:
            connection.close()
    except (OSError, sqlite3.Error, KeyError, TypeError, ValueError):
        return False


def _read_only_connection(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path.resolve()), safe='/')}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection
