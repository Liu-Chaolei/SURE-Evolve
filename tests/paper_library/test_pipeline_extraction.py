from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

import pymupdf
from pydantic import BaseModel

from paper_library.application.pipeline import (
    extract_candidates,
    extract_cards,
    extract_evidence,
    parse_documents,
    validate_records,
)
from paper_library.config import PaperLibraryConfig
from paper_library.errors import ErrorCode, PaperLibraryError
from paper_library.extraction.execution import (
    CandidateDesignBatch,
    EvidenceRecordBatch,
    ResearchDesignCardBatch,
    discover_candidates,
)
from paper_library.schemas.evidence_record import EvidenceRecord
from paper_library.schemas.parsed_document import ParsedDocument
from paper_library.schemas.research_design_card import ResearchDesignCard
from paper_library.schemas.review import ReviewCandidate
from paper_library.storage.jsonl import read_jsonl
from paper_library.storage.layout import BuildLayout
from paper_library.storage.state import BuildState, StageName, StageStatus
from paper_library.utils.hashing import sha256_file

T = TypeVar("T", bound=BaseModel)


class FakeProvider:
    name = "fake"
    model = "fake-structured-v1"

    def __init__(self) -> None:
        self.calls: list[type[BaseModel]] = []

    def extract(self, *, prompt: str, response_model: type[T]) -> T:
        self.calls.append(response_model)
        payload = json.loads(
            prompt.split("<untrusted-paper-data>\n", 1)[1].split(
                "\n</untrusted-paper-data>", 1
            )[0]
        )
        if response_model is CandidateDesignBatch:
            document = payload["document"]
            return response_model.model_validate(
                {
                    "candidates": [
                        {
                            "candidate_id": "candidate_fake",
                            "paper_id": document["paper_id"],
                            "document_id": document["document_id"],
                            "card_type": "intervention",
                            "title": "Fake provider design",
                            "source_objects": [
                                {
                                    "object_type": "block",
                                    "object_id": document["blocks"][0]["block_id"],
                                }
                            ],
                        }
                    ]
                }
            )
        if response_model is EvidenceRecordBatch:
            document = payload["document"]
            block = document["blocks"][0]
            return response_model.model_validate(
                {
                    "records": [
                        {
                            "evidence_id": "evidence_fake",
                            "paper_id": document["paper_id"],
                            "document_id": document["document_id"],
                            "role": "reported_result",
                            "status": "reported",
                            "claim": block["raw_text"],
                            "source_refs": [
                                {
                                    "document_id": document["document_id"],
                                    "object_type": "block",
                                    "object_id": block["block_id"],
                                    "page_number": block["page_number"],
                                    "bbox": block["bbox"],
                                    "verbatim_text": block["raw_text"],
                                }
                            ],
                            "extractor": {"name": self.name, "version": self.model},
                            "quality": {
                                "extraction_confidence": 0.9,
                                "grounding_status": "unverified",
                                "grounding_completeness": 0,
                                "review_required": True,
                                "warnings": ["fake evidence review"],
                            },
                        }
                    ]
                }
            )
        if response_model is ResearchDesignCardBatch:
            evidence = payload["evidence_records"][0]
            absent = {"status": "not_reported"}
            reported = {"status": "reported", "value": evidence["claim"]}
            return response_model.model_validate(
                {
                    "cards": [
                        {
                            "card_id": "card_fake",
                            "paper_id": payload["paper_id"],
                            "source_document_ids": [payload["document_id"]],
                            "card_type": "intervention",
                            "title": "Fake provider card",
                            "research_problem": absent,
                            "research_question": absent,
                            "hypothesis": {
                                "hypothesis_type": "tested_relationship_only",
                                "statement": absent,
                            },
                            "tested_relationship": reported,
                            "base_method": absent,
                            "intervention_delta": absent,
                            "candidate_system": absent,
                            "evidence_ids": [evidence["evidence_id"]],
                            "field_evidence": {
                                "tested_relationship": [evidence["evidence_id"]]
                            },
                            "builder": {"name": self.name, "version": self.model},
                            "quality": {
                                "extraction_confidence": 0.85,
                                "grounding_completeness": 1,
                                "review_required": True,
                                "warnings": ["fake card review"],
                            },
                        }
                    ]
                }
            )
        raise AssertionError(f"unexpected response model: {response_model}")


class WrongSourceTypeProvider(FakeProvider):
    def extract(self, *, prompt: str, response_model: type[T]) -> T:
        result = super().extract(prompt=prompt, response_model=response_model)
        if response_model is not CandidateDesignBatch:
            return result
        values = result.model_dump()
        values["candidates"][0]["source_objects"][0]["object_type"] = "figure"
        return response_model.model_validate(values)


class RetryableFailureProvider:
    name = "fake"
    model = "retryable-failure-v1"

    def extract(self, *, prompt: str, response_model: type[T]) -> T:
        del prompt, response_model
        raise PaperLibraryError(
            "temporary provider outage",
            code=ErrorCode.EXTRACTION_FAILED,
            retryable=True,
        )


def create_pdf(path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "The proposed model improves WER by 8 percent.")
    document.save(path)
    document.close()


def config_for(tmp_path: Path) -> PaperLibraryConfig:
    source = tmp_path / "papers"
    source.mkdir()
    return PaperLibraryConfig.model_validate(
        {
            "paths": {"source_root": source, "build_root": tmp_path / "build"},
            "parser": {"min_text_characters": 0},
        }
    )


def test_fake_provider_runs_candidate_evidence_card_pipeline_and_hits_cache(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    create_pdf(config.paths.source_root / "paper.pdf")
    assert parse_documents(config).parsed == 1
    provider = FakeProvider()

    candidates = extract_candidates(config, provider=provider)
    evidence = extract_evidence(config, provider)
    cards = extract_cards(config, provider)

    assert (candidates.processed, candidates.failed) == (1, 0)
    assert (evidence.processed, evidence.failed, evidence.review_queued) == (1, 0, 1)
    assert (cards.processed, cards.failed, cards.review_queued) == (1, 0, 1)
    assert provider.calls == [
        CandidateDesignBatch,
        EvidenceRecordBatch,
        ResearchDesignCardBatch,
    ]

    layout = BuildLayout(config.paths.build_root)
    evidence_records = list(
        read_jsonl(next(layout.evidence_objects.glob("*.jsonl")), EvidenceRecord)
    )
    card_records = list(
        read_jsonl(next(layout.card_objects.glob("*.jsonl")), ResearchDesignCard)
    )
    assert evidence_records[0].quality.grounding_status == "verified"
    assert card_records[0].evidence_ids == [evidence_records[0].evidence_id]
    evidence_reviews = list(
        read_jsonl(layout.reviews / "evidence_candidates.jsonl", ReviewCandidate)
    )
    card_reviews = list(
        read_jsonl(layout.reviews / "card_candidates.jsonl", ReviewCandidate)
    )
    assert evidence_reviews[0].record_id == evidence_records[0].evidence_id
    assert card_reviews[0].record_id == card_records[0].card_id

    assert extract_candidates(config, provider=provider).cache_hits == 1
    assert extract_evidence(config, provider).cache_hits == 1
    assert extract_cards(config, provider).cache_hits == 1
    assert len(provider.calls) == 3
    state = BuildState(layout.state_database)
    document_id = evidence_records[0].document_id
    expected = {
        StageName.CANDIDATE_DESIGNS: next(layout.candidate_objects.glob("*.jsonl")),
        StageName.EVIDENCE: next(layout.evidence_objects.glob("*.jsonl")),
        StageName.CARDS: next(layout.card_objects.glob("*.jsonl")),
    }
    for stage_name, artifact in expected.items():
        stage = state.get_stage(document_id, stage_name)
        assert stage is not None
        assert stage.output_sha256 == sha256_file(artifact)
        assert stage.output_record_count == 1
    assert len(
        list(read_jsonl(layout.reviews / "evidence_candidates.jsonl", ReviewCandidate))
    ) == 1
    assert len(
        list(read_jsonl(layout.reviews / "card_candidates.jsonl", ReviewCandidate))
    ) == 1


def build_complete_document(config: PaperLibraryConfig) -> tuple[BuildLayout, str]:
    create_pdf(config.paths.source_root / "paper.pdf")
    parse_documents(config)
    provider = FakeProvider()
    extract_candidates(config, provider=provider)
    extract_evidence(config, provider)
    extract_cards(config, provider)
    layout = BuildLayout(config.paths.build_root)
    parsed = ParsedDocument.model_validate_json(
        next(layout.parsed_objects.glob("*.json")).read_text(encoding="utf-8")
    )
    return layout, parsed.document_id


def test_validation_is_read_only_and_does_not_replace_report(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    layout, document_id = build_complete_document(config)
    report = layout.reports / "validation.json"
    report.write_text('{"sentinel": true}\n', encoding="utf-8")
    before = report.read_bytes()

    result = validate_records(config, document_id_filter=document_id)

    assert result.valid
    assert report.read_bytes() == before


def test_read_only_validation_does_not_invalidate_corrupt_stage(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    layout, document_id = build_complete_document(config)
    evidence_path = next(layout.evidence_objects.glob("*.jsonl"))
    evidence_path.write_text("{}\n", encoding="utf-8")
    state = BuildState(layout.state_database)
    before = state.get_stage(document_id, StageName.EVIDENCE)
    assert before is not None and before.status == StageStatus.SUCCEEDED

    result = validate_records(config)

    after = state.get_stage(document_id, StageName.EVIDENCE)
    assert not result.valid
    assert after is not None and after.status == StageStatus.SUCCEEDED
    assert after.attempt == before.attempt


def test_validation_of_missing_build_tree_creates_nothing(tmp_path: Path) -> None:
    source = tmp_path / "papers"
    source.mkdir()
    build_root = tmp_path / "missing-build"
    config = PaperLibraryConfig.model_validate(
        {"paths": {"source_root": source, "build_root": build_root}}
    )

    result = validate_records(config)

    assert not result.valid
    assert [issue.code for issue in result.issues] == ["empty_corpus"]
    assert not build_root.exists()


def test_candidate_corruption_rebuilds_and_stales_descendants(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_pdf(config.paths.source_root / "paper.pdf")
    parse_documents(config)
    provider = FakeProvider()
    extract_candidates(config, provider=provider)
    extract_evidence(config, provider)
    extract_cards(config, provider)
    layout = BuildLayout(config.paths.build_root)
    candidate_path = next(layout.candidate_objects.glob("*.jsonl"))
    evidence = next(read_jsonl(next(layout.evidence_objects.glob("*.jsonl")), EvidenceRecord))
    state = BuildState(layout.state_database)
    candidate_before = state.get_stage(evidence.document_id, StageName.CANDIDATE_DESIGNS)
    assert candidate_before is not None
    candidate_path.write_text("{}\n", encoding="utf-8")

    result = extract_candidates(config, provider=provider)

    assert (result.processed, result.cache_hits, result.failed) == (1, 0, 0)
    candidate_after = state.get_stage(evidence.document_id, StageName.CANDIDATE_DESIGNS)
    evidence_stage = state.get_stage(evidence.document_id, StageName.EVIDENCE)
    card_stage = state.get_stage(evidence.document_id, StageName.CARDS)
    assert candidate_after is not None
    assert candidate_after.attempt == candidate_before.attempt + 1
    assert candidate_after.output_sha256 == sha256_file(candidate_path)
    assert evidence_stage is not None and evidence_stage.status == StageStatus.STALE
    assert card_stage is not None and card_stage.status == StageStatus.STALE


def test_direct_evidence_extraction_rejects_corrupted_candidate(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_pdf(config.paths.source_root / "paper.pdf")
    parse_documents(config)
    provider = FakeProvider()
    extract_candidates(config, provider=provider)
    layout = BuildLayout(config.paths.build_root)
    candidate_path = next(layout.candidate_objects.glob("*.jsonl"))
    parsed = ParsedDocument.model_validate_json(
        next(layout.parsed_objects.glob("*.json")).read_text(encoding="utf-8")
    )
    candidate_path.write_text("{}\n", encoding="utf-8")
    calls_before = len(provider.calls)

    result = extract_evidence(config, provider)

    assert result.processed == 0
    assert len(provider.calls) == calls_before
    stage = BuildState(layout.state_database).get_stage(
        parsed.document_id,
        StageName.CANDIDATE_DESIGNS,
    )
    assert stage is not None and stage.status == StageStatus.STALE


def test_candidate_source_type_must_match_referenced_object(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_pdf(config.paths.source_root / "paper.pdf")
    assert parse_documents(config).parsed == 1
    layout = BuildLayout(config.paths.build_root)
    parsed_path = next(layout.parsed_objects.glob("*.json"))
    parsed = ParsedDocument.model_validate_json(parsed_path.read_text(encoding="utf-8"))

    try:
        discover_candidates(parsed, WrongSourceTypeProvider())
    except ValueError as exc:
        assert str(exc) == "candidate references an unknown source object"
    else:
        raise AssertionError("mismatched source object type was accepted")


def test_retryable_paper_library_error_is_recorded_in_stage_state(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    create_pdf(config.paths.source_root / "paper.pdf")
    assert parse_documents(config).parsed == 1

    result = extract_candidates(config, provider=RetryableFailureProvider())

    assert (result.processed, result.failed) == (0, 1)
    layout = BuildLayout(config.paths.build_root)
    parsed_path = next(layout.parsed_objects.glob("*.json"))
    document_id = json.loads(parsed_path.read_text(encoding="utf-8"))["document_id"]
    stage = BuildState(layout.state_database).get_stage(
        document_id, StageName.CANDIDATE_DESIGNS
    )
    assert stage is not None
    assert stage.status == StageStatus.FAILED_RETRYABLE
    assert stage.attempt == 1
    assert stage.error_code == "candidate_extraction_failed"
    assert stage.error_message == "PaperLibraryError: temporary provider outage"
    assert stage.output_uri is None
