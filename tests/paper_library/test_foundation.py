from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from paper_library.config import PaperLibraryConfig
from paper_library.errors import ConfigurationError
from paper_library.schemas.common import (
    BoundingBox,
    GroundingStatus,
    NumericFact,
    Producer,
    ReportingStatus,
    ScientificEvidenceStrength,
    SourceObjectType,
    SourceReference,
)
from paper_library.schemas.evidence_record import (
    EvidenceQuality,
    EvidenceRecord,
    EvidenceRole,
)
from paper_library.schemas.parsed_document import ParsedDocument, ParsedPage
from paper_library.schemas.research_design_card import (
    CardStatement,
    DesignCardQuality,
    DesignCardType,
    EvaluationSetup,
    Hypothesis,
    HypothesisType,
    ResearchDesignCard,
)

PDF_HASH = "a" * 64
DOCUMENT_ID = f"doc_{PDF_HASH}"
PRODUCER = Producer(name="test", version="1")
EVIDENCE_ID = "evidence_claim_1"


def source_ref() -> SourceReference:
    return SourceReference(
        document_id=DOCUMENT_ID,
        object_type=SourceObjectType.BLOCK,
        object_id="block_page_1_0",
        page_number=1,
        bbox=BoundingBox(x0=0, y0=0, x1=10, y1=10),
        verbatim_text="The proposed method improves WER.",
    )


def evidence() -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=EVIDENCE_ID,
        paper_id="paper_example",
        document_id=DOCUMENT_ID,
        role=EvidenceRole.REPORTED_RESULT,
        status=ReportingStatus.REPORTED,
        claim="The proposed method improves WER.",
        source_refs=[source_ref()],
        extractor=PRODUCER,
        quality=EvidenceQuality(
            extraction_confidence=1,
            grounding_status=GroundingStatus.VERIFIED,
            grounding_completeness=1,
            scientific_strength=ScientificEvidenceStrength.MODERATE,
        ),
    )


def test_parsed_document_round_trip_and_strict_fields() -> None:
    document = ParsedDocument(
        paper_id="paper_example",
        document_id=DOCUMENT_ID,
        pdf_sha256=PDF_HASH,
        source_aliases=["ASR/example.pdf"],
        domain="asr",
        page_count=1,
        pages=[
            ParsedPage(
                page_number=1,
                width=100,
                height=200,
                raw_text="text",
                normalized_text="text",
            )
        ],
        parser=PRODUCER,
    )
    assert ParsedDocument.model_validate_json(document.model_dump_json()) == document
    with pytest.raises(ValidationError, match="Extra inputs"):
        ParsedDocument.model_validate({**document.model_dump(), "unknown": True})


def test_document_id_must_match_pdf_hash() -> None:
    with pytest.raises(ValidationError, match="derived from pdf_sha256"):
        ParsedDocument(
            paper_id="paper_example",
            document_id="doc_wrong",
            pdf_sha256=PDF_HASH,
            source_aliases=["example.pdf"],
            domain="asr",
            page_count=1,
            pages=[
                ParsedPage(
                    page_number=1,
                    width=1,
                    height=1,
                    raw_text="",
                    normalized_text="",
                )
            ],
            parser=PRODUCER,
        )


def test_reported_evidence_requires_claim_and_source() -> None:
    record = evidence()
    assert EvidenceRecord.model_validate(record.model_dump()) == record
    with pytest.raises(ValidationError, match="requires source references"):
        EvidenceRecord.model_validate({**record.model_dump(), "source_refs": []})


def test_numeric_missing_status_rejects_value() -> None:
    with pytest.raises(ValidationError, match="cannot contain values"):
        NumericFact(name="WER", status=ReportingStatus.NOT_REPORTED, value=4.2)


def test_design_card_requires_field_evidence() -> None:
    reported = CardStatement(status=ReportingStatus.REPORTED, value="Reduce recognition errors")
    inferred = CardStatement(status=ReportingStatus.INFERRED, value="A modification helps")
    missing = CardStatement(status=ReportingStatus.NOT_REPORTED)
    card = ResearchDesignCard(
        card_id="card_example_intervention",
        paper_id="paper_example",
        source_document_ids=[DOCUMENT_ID],
        card_type=DesignCardType.INTERVENTION,
        title="Example intervention",
        research_problem=reported,
        research_question=missing,
        hypothesis=Hypothesis(
            hypothesis_type=HypothesisType.CURATOR_RECONSTRUCTED,
            statement=inferred,
        ),
        tested_relationship=reported,
        base_method=reported,
        intervention_delta=reported,
        candidate_system=reported,
        evaluation=EvaluationSetup(metrics=["WER"]),
        evidence_ids=[EVIDENCE_ID],
        field_evidence={
            "research_problem": [EVIDENCE_ID],
            "tested_relationship": [EVIDENCE_ID],
            "base_method": [EVIDENCE_ID],
            "intervention_delta": [EVIDENCE_ID],
            "candidate_system": [EVIDENCE_ID],
        },
        builder=PRODUCER,
        quality=DesignCardQuality(extraction_confidence=1, grounding_completeness=1),
    )
    assert ResearchDesignCard.model_validate_json(card.model_dump_json()) == card
    with pytest.raises(ValidationError, match="research_problem"):
        ResearchDesignCard.model_validate(
            {**card.model_dump(), "field_evidence": {"tested_relationship": [EVIDENCE_ID]}}
        )


def write_config(path: Path, source: str, build: str, api_key: str = "") -> None:
    api_key_line = f"  api_key: {api_key}\n" if api_key else ""
    path.write_text(
        "paths:\n"
        f"  source_root: {source}\n"
        f"  build_root: {build}\n"
        "extraction:\n"
        "  provider: disabled\n"
        f"{api_key_line}",
        encoding="utf-8",
    )


def test_config_resolves_relative_paths_and_excludes_secret_from_hash(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    write_config(config_path, "papers", "build", "first-secret")
    first = PaperLibraryConfig.load(config_path)
    write_config(config_path, "papers", "build", "second-secret")
    second = PaperLibraryConfig.load(config_path)
    assert first.paths.source_root == (tmp_path / "papers").resolve()
    assert first.paths.build_root == (tmp_path / "build").resolve()
    assert first.content_hash() == second.content_hash()
    with pytest.raises(ConfigurationError, match="escapes build_root"):
        first.ensure_output_path(tmp_path / "outside")


def test_config_rejects_undefined_environment_variable(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    write_config(config_path, "${UNDEFINED_PAPER_LIBRARY_PATH}", "build")
    with pytest.raises(ConfigurationError, match="Undefined environment variable"):
        PaperLibraryConfig.load(config_path)
