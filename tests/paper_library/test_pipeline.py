from __future__ import annotations

import sqlite3
from pathlib import Path

import pymupdf
import pytest

from paper_library.application.pipeline import (
    extract_candidates,
    load_canonical_records,
    parse_documents,
    scan,
)
from paper_library.application.scanner import scan_pdfs
from paper_library.config import PaperLibraryConfig
from paper_library.parsing.pymupdf_parser import PyMuPDFParser
from paper_library.schemas.parsed_document import ParsedDocument
from paper_library.storage.layout import BuildLayout
from paper_library.storage.state import BuildState, StageName, StageStatus
from paper_library.utils.hashing import sha256_file


def create_pdf(path: Path, text: str = "A test ASR paper with enough extractable text.") -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def config_for(tmp_path: Path) -> PaperLibraryConfig:
    source = tmp_path / "papers"
    source.mkdir()
    return PaperLibraryConfig.model_validate(
        {
            "paths": {
                "source_root": source,
                "build_root": tmp_path / "build",
            },
            "parser": {"min_text_characters": 0},
        }
    )


def test_scan_is_deterministic_and_deduplicates_content(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    first = config.paths.source_root / "ASR"
    second = config.paths.source_root / "copy"
    first.mkdir()
    second.mkdir()
    original = first / "paper.pdf"
    duplicate = second / "renamed.pdf"
    create_pdf(original)
    duplicate.write_bytes(original.read_bytes())

    documents = scan_pdfs(config.paths.source_root)
    assert len(documents) == 1
    assert documents[0].aliases == ("ASR/paper.pdf", "copy/renamed.pdf")
    assert documents[0].domain == "asr"


def test_scan_marks_missing_without_deleting_source_state(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    pdf = config.paths.source_root / "paper.pdf"
    create_pdf(pdf)
    first = scan(config)
    assert len(first.documents) == 1
    pdf.unlink()
    second = scan(config)
    assert second.missing == 1
    assert load_canonical_records(config).documents == []


def test_changed_upstream_cache_key_marks_descendants_stale(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_pdf(config.paths.source_root / "paper.pdf")
    parse_documents(config)
    extract_candidates(config)
    layout = BuildLayout(config.paths.build_root)
    parsed = ParsedDocument.model_validate_json(
        next(layout.parsed_objects.glob("*.json")).read_text(encoding="utf-8")
    )
    state = BuildState(layout.state_database)
    candidate = state.get_stage(parsed.document_id, StageName.CANDIDATE_DESIGNS)
    assert candidate is not None
    assert candidate.status == "succeeded"

    state.start_stage(parsed.document_id, StageName.PARSE, "changed", "changed")

    candidate = state.get_stage(parsed.document_id, StageName.CANDIDATE_DESIGNS)
    assert candidate is not None
    assert candidate.status == "stale"


def test_parse_builds_object_and_hits_cache(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_pdf(config.paths.source_root / "paper.pdf")
    first = parse_documents(config)
    second = parse_documents(config)
    assert first.parsed == 1
    assert first.failed == 0
    assert second.cache_hits == 1

    layout = BuildLayout(config.paths.build_root)
    outputs = list(layout.parsed_objects.glob("*.json"))
    assert len(outputs) == 1
    parsed = ParsedDocument.model_validate_json(outputs[0].read_text(encoding="utf-8"))
    assert parsed.page_count == 1
    assert parsed.blocks

    state = BuildState(layout.state_database)
    stage = state.get_stage(parsed.document_id, StageName.PARSE)
    assert stage is not None
    assert stage.output_sha256 == sha256_file(outputs[0])
    assert stage.output_record_count == 1
    assert state.stage_is_current(
        parsed.document_id,
        StageName.PARSE,
        outputs[0].stem,
    )


def test_parse_rebuilds_corrupted_cached_artifact(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_pdf(config.paths.source_root / "paper.pdf")
    assert parse_documents(config).parsed == 1
    layout = BuildLayout(config.paths.build_root)
    output = next(layout.parsed_objects.glob("*.json"))
    parsed = ParsedDocument.model_validate_json(output.read_text(encoding="utf-8"))
    state = BuildState(layout.state_database)
    first = state.get_stage(parsed.document_id, StageName.PARSE)
    assert first is not None

    output.write_text("{}", encoding="utf-8")
    result = parse_documents(config)

    assert (result.parsed, result.cache_hits, result.failed) == (1, 0, 0)
    repaired = state.get_stage(parsed.document_id, StageName.PARSE)
    assert repaired is not None
    assert repaired.attempt == first.attempt + 1
    assert repaired.output_sha256 == sha256_file(output)
    ParsedDocument.model_validate_json(output.read_text(encoding="utf-8"))


def test_direct_extraction_invalidates_corrupted_parse(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_pdf(config.paths.source_root / "paper.pdf")
    parse_documents(config)
    layout = BuildLayout(config.paths.build_root)
    output = next(layout.parsed_objects.glob("*.json"))
    parsed = ParsedDocument.model_validate_json(output.read_text(encoding="utf-8"))
    output.write_text("{}", encoding="utf-8")

    result = extract_candidates(config)

    assert result.processed == 0
    stage = BuildState(layout.state_database).get_stage(parsed.document_id, StageName.PARSE)
    assert stage is not None
    assert stage.status == StageStatus.STALE


def test_build_state_migrates_legacy_stage_integrity_columns(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE stages (
                document_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                status TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                output_uri TEXT,
                attempt INTEGER NOT NULL DEFAULT 0,
                error_code TEXT,
                error_message TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (document_id, stage)
            );
            """
        )
        connection.execute(
            "INSERT INTO stages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "doc_test",
                "parse",
                "succeeded",
                "key",
                "input",
                "object.json",
                1,
                None,
                None,
                "now",
            ),
        )
        connection.commit()
    finally:
        connection.close()

    state = BuildState(path)
    state.initialize()
    stage = state.get_stage("doc_test", StageName.PARSE)

    assert stage is not None
    assert stage.output_sha256 is None
    assert stage.output_record_count is None


def test_parser_replaces_unpaired_unicode_surrogates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = tmp_path / "paper.pdf"
    create_pdf(pdf)
    original = pymupdf.Page.get_text

    def get_text(page: pymupdf.Page, *args: object, **kwargs: object) -> object:
        result = original(page, *args, **kwargs)
        if args and args[0] == "dict":
            result["blocks"][0]["lines"][0]["spans"][0]["text"] = "bad \ud835 text"
        return result

    monkeypatch.setattr(pymupdf.Page, "get_text", get_text)  # type: ignore[attr-defined]
    parsed = PyMuPDFParser(min_text_characters=0).parse(
        pdf,
        paper_id="paper_test",
        source_aliases=["paper.pdf"],
        domain="speech",
    )

    assert parsed.blocks[0].raw_text == "bad ? text"
    parsed.model_dump_json()


def test_structural_candidate_stage_builds_object_and_hits_cache(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_pdf(config.paths.source_root / "paper.pdf")
    parse_documents(config)

    first = extract_candidates(config)
    second = extract_candidates(config)

    assert first.processed == 1
    assert first.review_queued == 1
    assert second.cache_hits == 1
    layout = BuildLayout(config.paths.build_root)
    assert len(list(layout.candidate_objects.glob("*.jsonl"))) == 1



def test_build_layout_rejects_escape(tmp_path: Path) -> None:
    layout = BuildLayout(tmp_path / "build")
    try:
        layout.resolve_inside("../outside")
    except ValueError as exc:
        assert "escapes" in str(exc)
    else:
        raise AssertionError("expected path escape rejection")
