from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from paper_library.schemas.common import (
    BoundingBox,
    Identifier,
    Producer,
    SchemaVersion,
    Sha256,
    StrictModel,
    utc_now,
)

PARSED_DOCUMENT_SCHEMA_VERSION = "1.0.0"


class TextSpan(StrictModel):
    text: str
    bbox: BoundingBox
    font: str | None = None
    size: float | None = Field(default=None, ge=0)
    bold: bool = False
    italic: bool = False


class TextBlock(StrictModel):
    block_id: Identifier
    page_number: int = Field(ge=1)
    reading_order: int = Field(ge=0)
    bbox: BoundingBox
    block_type: Literal["title", "heading", "paragraph", "caption", "list", "other"]
    raw_text: str
    normalized_text: str
    section_id: Identifier | None = None
    spans: list[TextSpan] = Field(default_factory=list)


class NormalizationSegment(StrictModel):
    normalized_start: int = Field(ge=0)
    normalized_end: int = Field(ge=0)
    raw_start: int = Field(ge=0)
    raw_end: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_ranges(self) -> "NormalizationSegment":
        if self.normalized_end < self.normalized_start or self.raw_end < self.raw_start:
            raise ValueError("normalization segment offsets must be ordered")
        return self


class TableCell(StrictModel):
    cell_id: Identifier
    row: int = Field(ge=0)
    column: int = Field(ge=0)
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)
    bbox: BoundingBox
    raw_text: str
    normalized_text: str
    is_header: bool = False
    bold: bool = False
    underlined: bool = False


class Table(StrictModel):
    table_id: Identifier
    page_number: int = Field(ge=1)
    bbox: BoundingBox
    caption: str | None = None
    cells: list[TableCell] = Field(default_factory=list)
    footnotes: list[str] = Field(default_factory=list)
    extraction_warnings: list[str] = Field(default_factory=list)


class Figure(StrictModel):
    figure_id: Identifier
    page_number: int = Field(ge=1)
    bbox: BoundingBox
    caption: str | None = None
    crop_uri: str | None = None


class Equation(StrictModel):
    equation_id: Identifier
    page_number: int = Field(ge=1)
    bbox: BoundingBox
    raw_text: str
    latex: str | None = None
    number: str | None = None
    context_block_ids: list[Identifier] = Field(default_factory=list)
    symbol_definitions: dict[str, str] = Field(default_factory=dict)
    extraction_warnings: list[str] = Field(default_factory=list)


class SectionNode(StrictModel):
    section_id: Identifier
    title: str
    level: int = Field(ge=1)
    page_number: int = Field(ge=1)
    block_ids: list[Identifier] = Field(default_factory=list)
    children: list["SectionNode"] = Field(default_factory=list)


class ParsedPage(StrictModel):
    page_number: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    raw_text: str
    normalized_text: str
    normalization_map: list[NormalizationSegment] = Field(default_factory=list)
    block_ids: list[Identifier] = Field(default_factory=list)


class ParsedDocument(StrictModel):
    schema_version: SchemaVersion = PARSED_DOCUMENT_SCHEMA_VERSION
    paper_id: Identifier
    document_id: Identifier
    pdf_sha256: Sha256
    source_aliases: list[str] = Field(min_length=1)
    title: str | None = None
    domain: Literal["asr", "tts", "speech", "other"]
    page_count: int = Field(ge=1)
    pages: list[ParsedPage] = Field(min_length=1)
    sections: list[SectionNode] = Field(default_factory=list)
    blocks: list[TextBlock] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
    equations: list[Equation] = Field(default_factory=list)
    parser: Producer
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_document(self) -> "ParsedDocument":
        if self.document_id != f"doc_{self.pdf_sha256}":
            raise ValueError("document_id must be derived from pdf_sha256")
        if self.page_count != len(self.pages):
            raise ValueError("page_count must equal the number of pages")
        if [page.page_number for page in self.pages] != list(range(1, self.page_count + 1)):
            raise ValueError("pages must be ordered and contiguous from one")
        object_ids = [block.block_id for block in self.blocks]
        object_ids += [table.table_id for table in self.tables]
        object_ids += [cell.cell_id for table in self.tables for cell in table.cells]
        object_ids += [figure.figure_id for figure in self.figures]
        object_ids += [equation.equation_id for equation in self.equations]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("parsed source object IDs must be unique")
        page_numbers = {page.page_number for page in self.pages}
        referenced_pages = (
            [block.page_number for block in self.blocks]
            + [table.page_number for table in self.tables]
            + [figure.page_number for figure in self.figures]
            + [equation.page_number for equation in self.equations]
        )
        if not set(referenced_pages).issubset(page_numbers):
            raise ValueError("source objects must reference existing pages")
        return self
