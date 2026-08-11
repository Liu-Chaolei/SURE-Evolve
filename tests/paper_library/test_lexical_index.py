from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from paper_library.application.lexical_index import (
    build_lexical_index,
    query_lexical_index,
)
from paper_library.application.publish import publish_corpus
from paper_library.config import PaperLibraryConfig
from paper_library.errors import PaperLibraryError
from tests.paper_library.test_extraction_validation import (
    design_card,
    document,
    evidence_record,
)


def config_for(tmp_path: Path) -> PaperLibraryConfig:
    return PaperLibraryConfig.model_validate(
        {
            "paths": {
                "source_root": tmp_path / "papers",
                "build_root": tmp_path / "build",
            }
        }
    )


def publish_fixture(config: PaperLibraryConfig) -> str:
    evidence = evidence_record()
    result = publish_corpus(
        config,
        [document()],
        [evidence],
        [design_card(evidence.evidence_id)],
    )
    return result.manifest.corpus_version


def test_builds_and_reuses_corpus_bound_lexical_index(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    version = publish_fixture(config)

    first = build_lexical_index(config)
    second = build_lexical_index(config)

    assert first.corpus_version == version
    assert first.evidence_count == 1
    assert first.design_count == 1
    assert not first.cache_hit
    assert second.cache_hit
    assert second.index_path == first.index_path
    assert not first.index_path.with_suffix(".sqlite3-wal").exists()


def test_queries_evidence_and_design_separately(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    build_lexical_index(config, corpus_version=publish_fixture(config))

    evidence = query_lexical_index(config, "improves WER", kind="evidence")
    design = query_lexical_index(
        config,
        "intervention",
        kind="design",
        record_type="intervention",
    )

    assert len(evidence.hits) == 1
    assert evidence.hits[0].record.evidence_id == evidence.hits[0].record_id
    assert len(design.hits) == 1
    assert design.hits[0].record.card_id == design.hits[0].record_id


def test_query_does_not_create_missing_index(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    version = publish_fixture(config)
    index_path = config.paths.build_root / "indexes" / version / "lexical.sqlite3"

    with pytest.raises(PaperLibraryError, match="missing or stale"):
        query_lexical_index(config, "WER", kind="evidence")

    assert not index_path.exists()


def test_query_rejects_stale_index_metadata(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    result = build_lexical_index(config, corpus_version=publish_fixture(config))
    connection = sqlite3.connect(result.index_path)
    try:
        connection.execute(
            "UPDATE index_metadata SET value = 'stale' WHERE key = 'corpus_version'"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(PaperLibraryError, match="missing or stale"):
        query_lexical_index(config, "WER", kind="evidence")


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("body", "tampered searchable text"),
        ("record_json", "{}"),
        ("paper_id", "paper_tampered"),
        ("record_type", "negative_result"),
    ],
)
def test_query_rejects_tampered_index_rows(
    tmp_path: Path,
    column: str,
    value: str,
) -> None:
    config = config_for(tmp_path)
    result = build_lexical_index(config, corpus_version=publish_fixture(config))
    connection = sqlite3.connect(result.index_path)
    try:
        connection.execute(f"UPDATE evidence_fts SET {column} = ?", (value,))
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(PaperLibraryError, match="missing or stale"):
        query_lexical_index(config, "WER", kind="evidence")


def test_build_replaces_tampered_index(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    first = build_lexical_index(config, corpus_version=publish_fixture(config))
    connection = sqlite3.connect(first.index_path)
    try:
        connection.execute("UPDATE evidence_fts SET body = 'tampered'")
        connection.commit()
    finally:
        connection.close()

    rebuilt = build_lexical_index(config)

    assert not rebuilt.cache_hit
    evidence = query_lexical_index(config, "improves WER", kind="evidence")
    assert len(evidence.hits) == 1


def test_query_rejects_index_with_missing_table(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    result = build_lexical_index(config, corpus_version=publish_fixture(config))
    connection = sqlite3.connect(result.index_path)
    try:
        connection.execute("DROP TABLE evidence_fts")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(PaperLibraryError, match="missing or stale"):
        query_lexical_index(config, "WER", kind="evidence")


def test_query_validates_arguments(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    with pytest.raises(ValueError, match="Invalid query kind"):
        query_lexical_index(config, "WER", kind="all")
    with pytest.raises(ValueError, match="must not be empty"):
        query_lexical_index(config, " ", kind="evidence")
    with pytest.raises(ValueError, match="between 1 and 100"):
        query_lexical_index(config, "WER", kind="evidence", limit=0)
    with pytest.raises(ValueError, match="between 1 and 100"):
        query_lexical_index(config, "WER", kind="evidence", limit=101)


def test_query_handles_filters_no_results_and_invalid_fts(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    build_lexical_index(config, corpus_version=publish_fixture(config))

    no_results = query_lexical_index(
        config,
        "WER",
        kind="evidence",
        paper_id="paper_missing",
    )
    assert no_results.hits == ()

    with pytest.raises(PaperLibraryError, match="Lexical query failed"):
        query_lexical_index(config, '"unterminated', kind="evidence")
