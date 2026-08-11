from __future__ import annotations

from dataclasses import dataclass

from paper_library.parsing.normalization import grounding_normal_form
from paper_library.schemas.common import (
    BoundingBox,
    GroundingStatus,
    SourceObjectType,
    SourceReference,
)
from paper_library.schemas.evidence_record import EvidenceRecord
from paper_library.schemas.parsed_document import ParsedDocument


@dataclass(frozen=True)
class SourceObject:
    page_number: int
    bbox: BoundingBox
    raw_text: str


@dataclass(frozen=True)
class GroundingResult:
    status: GroundingStatus
    completeness: float
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def source_objects(
    document: ParsedDocument,
) -> dict[tuple[SourceObjectType, str], SourceObject]:
    values: dict[tuple[SourceObjectType, str], SourceObject] = {}
    for block in document.blocks:
        values[(SourceObjectType.BLOCK, block.block_id)] = SourceObject(
            page_number=block.page_number,
            bbox=block.bbox,
            raw_text=block.raw_text,
        )
    for table in document.tables:
        values[(SourceObjectType.TABLE, table.table_id)] = SourceObject(
            page_number=table.page_number,
            bbox=table.bbox,
            raw_text="\n".join(cell.raw_text for cell in table.cells),
        )
        for cell in table.cells:
            values[(SourceObjectType.TABLE_CELL, cell.cell_id)] = SourceObject(
                page_number=table.page_number,
                bbox=cell.bbox,
                raw_text=cell.raw_text,
            )
    for figure in document.figures:
        values[(SourceObjectType.FIGURE, figure.figure_id)] = SourceObject(
            page_number=figure.page_number,
            bbox=figure.bbox,
            raw_text=figure.caption or "",
        )
    for equation in document.equations:
        values[(SourceObjectType.EQUATION, equation.equation_id)] = SourceObject(
            page_number=equation.page_number,
            bbox=equation.bbox,
            raw_text=equation.raw_text,
        )
    return values


def source_texts(document: ParsedDocument) -> dict[tuple[SourceObjectType, str], str]:
    return {key: value.raw_text for key, value in source_objects(document).items()}


def ground_reference(
    reference: SourceReference,
    document: ParsedDocument,
) -> tuple[GroundingStatus, str | None]:
    if reference.document_id != document.document_id:
        return GroundingStatus.UNVERIFIED, "source reference document_id does not match"
    source = source_objects(document).get((reference.object_type, reference.object_id))
    if source is None:
        return GroundingStatus.UNVERIFIED, "source object does not exist"
    if reference.page_number != source.page_number:
        return GroundingStatus.UNVERIFIED, "source reference page_number does not match"
    if reference.bbox is not None and reference.bbox != source.bbox:
        return GroundingStatus.UNVERIFIED, "source reference bbox does not match"
    if reference.raw_start is not None:
        if reference.raw_end > len(source.raw_text):
            return GroundingStatus.UNVERIFIED, "source reference offsets exceed source object"
        if source.raw_text[reference.raw_start : reference.raw_end] != reference.verbatim_text:
            return GroundingStatus.UNVERIFIED, "source reference offsets do not match verbatim text"
        return GroundingStatus.VERIFIED, None
    if reference.verbatim_text in source.raw_text:
        return GroundingStatus.VERIFIED, None
    if grounding_normal_form(reference.verbatim_text) in grounding_normal_form(source.raw_text):
        return GroundingStatus.NORMALIZED_MATCH, "matched after controlled normalization"
    return GroundingStatus.UNVERIFIED, "verbatim text was not found in source object"


def ground_evidence(record: EvidenceRecord, document: ParsedDocument) -> GroundingResult:
    if not record.source_refs:
        return GroundingResult(
            status=GroundingStatus.UNVERIFIED,
            completeness=0,
            errors=("evidence has no source references",),
        )
    statuses: list[GroundingStatus] = []
    errors: list[str] = []
    warnings: list[str] = []
    for index, reference in enumerate(record.source_refs):
        status, message = ground_reference(reference, document)
        statuses.append(status)
        if message:
            formatted = f"source_refs.{index}: {message}"
            if status == GroundingStatus.NORMALIZED_MATCH:
                warnings.append(formatted)
            else:
                errors.append(formatted)
    matched = sum(
        status in {GroundingStatus.VERIFIED, GroundingStatus.NORMALIZED_MATCH}
        for status in statuses
    )
    completeness = matched / len(statuses)
    if errors:
        overall = GroundingStatus.UNVERIFIED
    elif GroundingStatus.NORMALIZED_MATCH in statuses:
        overall = GroundingStatus.NORMALIZED_MATCH
    else:
        overall = GroundingStatus.VERIFIED
    return GroundingResult(
        status=overall,
        completeness=completeness,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
