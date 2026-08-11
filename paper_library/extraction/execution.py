from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field

from paper_library.extraction.candidate_designs import CandidateDesign
from paper_library.extraction.design_cards import card_id_for
from paper_library.extraction.evidence import evidence_id_for
from paper_library.extraction.grounding import ground_evidence
from paper_library.extraction.providers import StructuredExtractionProvider
from paper_library.schemas.common import GroundingStatus, SourceObjectType, StrictModel
from paper_library.schemas.evidence_record import EvidenceRecord
from paper_library.schemas.parsed_document import ParsedDocument
from paper_library.schemas.research_design_card import ResearchDesignCard

_PROMPT_DIRECTORY = Path(__file__).with_name("prompts")


class CandidateDesignBatch(StrictModel):
    candidates: list[CandidateDesign] = Field(default_factory=list)


class EvidenceRecordBatch(StrictModel):
    records: list[EvidenceRecord] = Field(default_factory=list)


class ResearchDesignCardBatch(StrictModel):
    cards: list[ResearchDesignCard] = Field(default_factory=list)


def discover_candidates(
    document: ParsedDocument,
    provider: StructuredExtractionProvider,
) -> list[CandidateDesign]:
    prompt = _build_prompt(
        "candidate_designs_v1.txt",
        {"document": document.model_dump(mode="json")},
    )
    result = provider.extract(prompt=prompt, response_model=CandidateDesignBatch)
    object_refs = _source_object_refs(document)
    for candidate in result.candidates:
        if candidate.document_id != document.document_id or candidate.paper_id != document.paper_id:
            raise ValueError("candidate identity does not match the parsed document")
        if any(
            (source.object_type, source.object_id) not in object_refs
            for source in candidate.source_objects
        ):
            raise ValueError("candidate references an unknown source object")
    return result.candidates


def extract_evidence_records(
    document: ParsedDocument,
    candidates: list[CandidateDesign],
    provider: StructuredExtractionProvider,
) -> list[EvidenceRecord]:
    prompt = _build_prompt(
        "evidence_v1.txt",
        {
            "document": document.model_dump(mode="json"),
            "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
        },
    )
    result = provider.extract(prompt=prompt, response_model=EvidenceRecordBatch)
    grounded: list[EvidenceRecord] = []
    for record in result.records:
        if record.document_id != document.document_id or record.paper_id != document.paper_id:
            raise ValueError("evidence identity does not match the parsed document")
        grounding = ground_evidence(record, document)
        quality = record.quality.model_copy(
            update={
                "grounding_status": grounding.status,
                "grounding_completeness": grounding.completeness,
                "warnings": [*record.quality.warnings, *grounding.warnings, *grounding.errors],
                "review_required": record.quality.review_required
                or grounding.status != GroundingStatus.VERIFIED,
            }
        )
        values = record.model_dump()
        values["quality"] = quality
        values["evidence_id"] = evidence_id_for(record)
        grounded.append(EvidenceRecord.model_validate(values))
    return grounded


def build_design_cards(
    document: ParsedDocument,
    evidence_records: list[EvidenceRecord],
    provider: StructuredExtractionProvider,
) -> list[ResearchDesignCard]:
    unverified = [
        record.evidence_id
        for record in evidence_records
        if record.quality.grounding_status != GroundingStatus.VERIFIED
    ]
    if unverified:
        raise ValueError("design cards can only consume exactly grounded evidence")
    prompt = _build_prompt(
        "design_cards_v1.txt",
        {
            "document_id": document.document_id,
            "paper_id": document.paper_id,
            "evidence_records": [
                record.model_dump(mode="json") for record in evidence_records
            ],
        },
    )
    result = provider.extract(prompt=prompt, response_model=ResearchDesignCardBatch)
    evidence_ids = {record.evidence_id for record in evidence_records}
    cards: list[ResearchDesignCard] = []
    for card in result.cards:
        if card.paper_id != document.paper_id:
            raise ValueError("card identity does not match the parsed document")
        if set(card.source_document_ids) != {document.document_id}:
            raise ValueError("card source documents do not match the extraction document")
        if not set(card.evidence_ids).issubset(evidence_ids):
            raise ValueError("card references evidence outside the extraction input")
        values = card.model_dump()
        values["card_id"] = card_id_for(card)
        cards.append(ResearchDesignCard.model_validate(values))
    return cards


def _build_prompt(template_name: str, payload: object) -> str:
    template = (_PROMPT_DIRECTORY / template_name).read_text(encoding="utf-8").strip()
    source = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return f"{template}\n\n<untrusted-paper-data>\n{source}\n</untrusted-paper-data>"


def _source_object_refs(document: ParsedDocument) -> set[tuple[SourceObjectType, str]]:
    values = {(SourceObjectType.BLOCK, block.block_id) for block in document.blocks}
    values.update((SourceObjectType.TABLE, table.table_id) for table in document.tables)
    values.update(
        (SourceObjectType.TABLE_CELL, cell.cell_id)
        for table in document.tables
        for cell in table.cells
    )
    values.update(
        (SourceObjectType.FIGURE, figure.figure_id) for figure in document.figures
    )
    values.update(
        (SourceObjectType.EQUATION, equation.equation_id)
        for equation in document.equations
    )
    return values
