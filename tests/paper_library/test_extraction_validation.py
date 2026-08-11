from __future__ import annotations

from pathlib import Path

import pytest

from paper_library.application.publish import publish_corpus
from paper_library.application.reviews import (
    append_review_record,
    review_candidate_for,
    review_decision_for,
)
from paper_library.application.validate import validate_corpus
from paper_library.config import PaperLibraryConfig
from paper_library.extraction.design_cards import card_id_for, ensure_stable_card_ids
from paper_library.extraction.evidence import (
    ensure_stable_evidence_ids,
    evidence_id_for,
)
from paper_library.extraction.grounding import ground_evidence, ground_reference
from paper_library.schemas.common import (
    BoundingBox,
    GroundingStatus,
    Producer,
    ReportingStatus,
    SourceObjectType,
    SourceReference,
)
from paper_library.schemas.evidence_record import (
    EvidenceQuality,
    EvidenceRecord,
    EvidenceRole,
)
from paper_library.schemas.parsed_document import ParsedDocument, ParsedPage, TextBlock
from paper_library.schemas.research_design_card import (
    CardStatement,
    DesignCardQuality,
    DesignCardType,
    EvaluationSetup,
    Hypothesis,
    HypothesisType,
    ResearchDesignCard,
)
from paper_library.schemas.review import (
    ReviewCandidate,
    ReviewDecision,
    ReviewDecisionType,
    ReviewObjectType,
)
from paper_library.storage.jsonl import read_jsonl, write_jsonl_atomic
from paper_library.storage.manifest import CurrentManifest

PDF_HASH = "b" * 64
DOCUMENT_ID = f"doc_{PDF_HASH}"
BLOCK_ID = "block_example_1"
BBOX = BoundingBox(x0=10, y0=20, x1=100, y1=40)
PRODUCER = Producer(name="test", version="1")
TEXT = "The proposed model im-\nproves WER by 8%."


def document() -> ParsedDocument:
    return ParsedDocument(
        paper_id="paper_example",
        document_id=DOCUMENT_ID,
        pdf_sha256=PDF_HASH,
        source_aliases=["ASR/example.pdf"],
        domain="asr",
        page_count=1,
        pages=[
            ParsedPage(
                page_number=1,
                width=200,
                height=300,
                raw_text=TEXT,
                normalized_text=TEXT,
                block_ids=[BLOCK_ID],
            )
        ],
        blocks=[
            TextBlock(
                block_id=BLOCK_ID,
                page_number=1,
                reading_order=0,
                bbox=BBOX,
                block_type="paragraph",
                raw_text=TEXT,
                normalized_text=TEXT,
            )
        ],
        parser=PRODUCER,
    )


def reference(verbatim_text: str = TEXT, **updates: object) -> SourceReference:
    values: dict[str, object] = {
        "document_id": DOCUMENT_ID,
        "object_type": SourceObjectType.BLOCK,
        "object_id": BLOCK_ID,
        "page_number": 1,
        "bbox": BBOX,
        "verbatim_text": verbatim_text,
    }
    values.update(updates)
    return SourceReference.model_validate(values)


def evidence_record(source: SourceReference | None = None) -> EvidenceRecord:
    record = EvidenceRecord(
        evidence_id="evidence_placeholder",
        paper_id="paper_example",
        document_id=DOCUMENT_ID,
        role=EvidenceRole.REPORTED_RESULT,
        status=ReportingStatus.REPORTED,
        claim="The model improves WER.",
        source_refs=[source or reference()],
        extractor=PRODUCER,
        quality=EvidenceQuality(
            extraction_confidence=1,
            grounding_status=GroundingStatus.VERIFIED,
            grounding_completeness=1,
        ),
    )
    return record.model_copy(update={"evidence_id": evidence_id_for(record)})


def design_card(evidence_id: str) -> ResearchDesignCard:
    reported = CardStatement(status=ReportingStatus.REPORTED, value="Model improves WER")
    absent = CardStatement(status=ReportingStatus.NOT_REPORTED)
    card = ResearchDesignCard(
        card_id="card_placeholder",
        paper_id="paper_example",
        source_document_ids=[DOCUMENT_ID],
        card_type=DesignCardType.INTERVENTION,
        title="WER intervention",
        research_problem=reported,
        research_question=absent,
        hypothesis=Hypothesis(
            hypothesis_type=HypothesisType.TESTED_RELATIONSHIP_ONLY,
            statement=absent,
        ),
        tested_relationship=reported,
        base_method=absent,
        intervention_delta=reported,
        candidate_system=absent,
        evaluation=EvaluationSetup(metrics=["WER"]),
        evidence_ids=[evidence_id],
        field_evidence={
            "research_problem": [evidence_id],
            "tested_relationship": [evidence_id],
            "intervention_delta": [evidence_id],
        },
        builder=PRODUCER,
        quality=DesignCardQuality(extraction_confidence=1, grounding_completeness=1),
    )
    return card.model_copy(update={"card_id": card_id_for(card)})


def test_grounding_checks_exact_normalized_and_forged_locations() -> None:
    parsed = document()
    exact, message = ground_reference(reference(), parsed)
    assert exact == GroundingStatus.VERIFIED
    assert message is None

    normalized, _ = ground_reference(reference("The proposed model improves WER"), parsed)
    assert normalized == GroundingStatus.NORMALIZED_MATCH

    wrong_page, message = ground_reference(reference(page_number=2), parsed)
    assert wrong_page == GroundingStatus.UNVERIFIED
    assert message == "source reference page_number does not match"

    wrong_bbox, message = ground_reference(
        reference(bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1)), parsed
    )
    assert wrong_bbox == GroundingStatus.UNVERIFIED
    assert message == "source reference bbox does not match"

    missing, message = ground_reference(reference(object_id="block_missing"), parsed)
    assert missing == GroundingStatus.UNVERIFIED
    assert message == "source object does not exist"


def test_grounding_checks_raw_offsets() -> None:
    parsed = document()
    start = TEXT.index("WER")
    status, _ = ground_reference(
        reference("WER", raw_start=start, raw_end=start + 3), parsed
    )
    assert status == GroundingStatus.VERIFIED
    status, message = ground_reference(
        reference("WER", raw_start=0, raw_end=3), parsed
    )
    assert status == GroundingStatus.UNVERIFIED
    assert message == "source reference offsets do not match verbatim text"


def test_stable_ids_and_jsonl_round_trip(tmp_path: Path) -> None:
    evidence = evidence_record()
    card = design_card(evidence.evidence_id)
    ensure_stable_evidence_ids([evidence])
    ensure_stable_card_ids([card])

    path = tmp_path / "records.jsonl"
    write_jsonl_atomic(path, [evidence])
    assert list(read_jsonl(path, EvidenceRecord)) == [evidence]

    with pytest.raises(ValueError, match="does not match stable"):
        ensure_stable_evidence_ids(
            [evidence.model_copy(update={"evidence_id": "evidence_wrong"})]
        )


def test_corpus_validation_accepts_grounded_records() -> None:
    evidence = evidence_record()
    result = validate_corpus(
        [document()],
        [evidence],
        [design_card(evidence.evidence_id)],
    )
    assert result.valid
    assert result.issues == ()


def test_corpus_validation_rejects_normalized_and_missing_links() -> None:
    normalized = evidence_record(reference("The proposed model improves WER"))
    normalized = normalized.model_copy(
        update={
            "quality": normalized.quality.model_copy(
                update={"grounding_status": GroundingStatus.NORMALIZED_MATCH}
            )
        }
    )
    card = design_card(normalized.evidence_id).model_copy(
        update={"parent_card_id": "card_missing"}
    )
    result = validate_corpus([document()], [normalized], [card])
    codes = {issue.code for issue in result.issues}
    assert not result.valid
    assert "grounding_not_exact" in codes
    assert "normalized_grounding" in codes
    assert "missing_related_card" in codes


def test_corpus_validation_rejects_stale_grounding_quality() -> None:
    stale = evidence_record(reference("The proposed model improves WER"))
    result = validate_corpus(
        [document()],
        [stale],
        [design_card(stale.evidence_id)],
        require_exact_grounding=False,
    )
    assert not result.valid
    assert "stale_grounding_quality" in {issue.code for issue in result.issues}


def test_review_records_are_stable_and_append_only(tmp_path: Path) -> None:
    candidate = review_candidate_for(
        object_type=ReviewObjectType.EVIDENCE,
        record_id="evidence_example",
        document_id=DOCUMENT_ID,
        candidate_uri="objects/evidence/example.jsonl",
        reasons=["normalized grounding"],
        upstream_hash="d" * 64,
    )
    decision = review_decision_for(
        candidate,
        decision=ReviewDecisionType.ACCEPTED,
        reviewer="curator",
        reason="Verified against PDF",
    )
    candidates_path = tmp_path / "candidates.jsonl"
    decisions_path = tmp_path / "decisions.jsonl"
    append_review_record(candidates_path, candidate)
    append_review_record(candidates_path, candidate)
    append_review_record(decisions_path, decision)
    assert list(read_jsonl(candidates_path, ReviewCandidate)) == [candidate]
    assert list(read_jsonl(decisions_path, ReviewDecision)) == [decision]


def test_publish_writes_versioned_snapshot_and_current_manifest(tmp_path: Path) -> None:
    config = PaperLibraryConfig.model_validate(
        {
            "paths": {
                "source_root": tmp_path / "papers",
                "build_root": tmp_path / "build",
            }
        }
    )
    evidence = evidence_record()
    first = publish_corpus(
        config,
        [document()],
        [evidence],
        [design_card(evidence.evidence_id)],
    )
    second = publish_corpus(
        config,
        [document()],
        [evidence],
        [design_card(evidence.evidence_id)],
    )
    assert first.manifest.corpus_version == second.manifest.corpus_version
    version = first.manifest.corpus_version
    records = config.paths.build_root / "records" / version
    assert list(read_jsonl(records / "evidence_records.jsonl", EvidenceRecord)) == [evidence]
    current = CurrentManifest.model_validate_json(
        (config.paths.build_root / "manifests" / "current.json").read_text(encoding="utf-8")
    )
    assert current.corpus_version == version


def test_publish_rejects_corrupted_existing_snapshot(tmp_path: Path) -> None:
    config = PaperLibraryConfig.model_validate(
        {
            "paths": {
                "source_root": tmp_path / "papers",
                "build_root": tmp_path / "build",
            }
        }
    )
    evidence = evidence_record()
    published = publish_corpus(
        config,
        [document()],
        [evidence],
        [design_card(evidence.evidence_id)],
    )
    evidence_path = (
        config.paths.build_root
        / "records"
        / published.manifest.corpus_version
        / "evidence_records.jsonl"
    )
    evidence_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash does not match"):
        publish_corpus(
            config,
            [document()],
            [evidence],
            [design_card(evidence.evidence_id)],
        )



def test_publish_rejects_review_required_records(tmp_path: Path) -> None:
    config = PaperLibraryConfig.model_validate(
        {
            "paths": {
                "source_root": tmp_path / "papers",
                "build_root": tmp_path / "build",
            }
        }
    )
    evidence = evidence_record()
    evidence = evidence.model_copy(
        update={"quality": evidence.quality.model_copy(update={"review_required": True})}
    )
    with pytest.raises(ValueError, match="Review-required"):
        publish_corpus(
            config,
            [document()],
            [evidence],
            [design_card(evidence.evidence_id)],
        )


def test_ground_evidence_rejects_cross_document_reference_without_schema_bypass() -> None:
    evidence = evidence_record()
    other_hash = "c" * 64
    parsed = document().model_copy(
        update={"document_id": f"doc_{other_hash}", "pdf_sha256": other_hash}
    )
    result = ground_evidence(evidence, parsed)
    assert result.status == GroundingStatus.UNVERIFIED
    assert result.completeness == 0
