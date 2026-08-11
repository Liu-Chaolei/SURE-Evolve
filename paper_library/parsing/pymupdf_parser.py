from __future__ import annotations

from pathlib import Path

import pymupdf

from paper_library.parsing.normalization import normalize_text
from paper_library.schemas.common import BoundingBox, Producer
from paper_library.schemas.parsed_document import (
    Figure,
    ParsedDocument,
    ParsedPage,
    Table,
    TableCell,
    TextBlock,
    TextSpan,
)
from paper_library.utils.hashing import hash_canonical, sha256_file
from paper_library.utils.ids import document_id, stable_id


class PyMuPDFParser:
    name = "pymupdf"
    version = pymupdf.VersionBind

    def __init__(
        self,
        *,
        min_text_characters: int = 100,
        extract_tables: bool = True,
        extract_figures: bool = True,
    ) -> None:
        self.min_text_characters = min_text_characters
        self.extract_tables = extract_tables
        self.extract_figures = extract_figures

    def config_hash(self) -> str:
        return hash_canonical(
            {
                "min_text_characters": self.min_text_characters,
                "extract_tables": self.extract_tables,
                "extract_figures": self.extract_figures,
            }
        )

    def parse(
        self,
        path: Path,
        *,
        paper_id: str,
        source_aliases: list[str],
        domain: str,
        pdf_sha256: str | None = None,
    ) -> ParsedDocument:
        digest = pdf_sha256 or sha256_file(path)
        expected_document_id = document_id(digest)
        warnings: list[str] = []
        blocks: list[TextBlock] = []
        tables: list[Table] = []
        figures: list[Figure] = []
        pages: list[ParsedPage] = []

        with pymupdf.open(path) as pdf:
            if pdf.is_encrypted and not pdf.authenticate(""):
                raise ValueError(f"Encrypted PDF cannot be opened: {path}")
            if pdf.page_count < 1:
                raise ValueError(f"PDF has no pages: {path}")
            metadata = pdf.metadata or {}
            title = _safe_text(metadata.get("title") or "").strip() or None
            for page_index, page in enumerate(pdf):
                page_number = page_index + 1
                page_dict = page.get_text("dict", sort=True)
                page_block_ids: list[str] = []
                raw_parts: list[str] = []
                reading_order = 0
                for source_block in page_dict.get("blocks", []):
                    block_type = int(source_block.get("type", 0))
                    if block_type == 1:
                        if self.extract_figures:
                            bbox = _bbox(source_block.get("bbox"))
                            figures.append(
                                Figure(
                                    figure_id=stable_id(
                                        "figure", expected_document_id, page_number, reading_order
                                    ),
                                    page_number=page_number,
                                    bbox=bbox,
                                )
                            )
                        reading_order += 1
                        continue
                    if block_type != 0:
                        reading_order += 1
                        continue
                    spans: list[TextSpan] = []
                    line_texts: list[str] = []
                    for line in source_block.get("lines", []):
                        line_parts: list[str] = []
                        for source_span in line.get("spans", []):
                            text = _safe_text(source_span.get("text", ""))
                            if not text:
                                continue
                            flags = int(source_span.get("flags", 0))
                            spans.append(
                                TextSpan(
                                    text=text,
                                    bbox=_bbox(source_span.get("bbox")),
                                    font=source_span.get("font"),
                                    size=float(source_span.get("size", 0)),
                                    bold=bool(flags & 16),
                                    italic=bool(flags & 2),
                                )
                            )
                            line_parts.append(text)
                        if line_parts:
                            line_texts.append("".join(line_parts))
                    raw_text = "\n".join(line_texts).strip()
                    if not raw_text:
                        reading_order += 1
                        continue
                    normalized, _ = normalize_text(raw_text)
                    block_id = stable_id(
                        "block", expected_document_id, page_number, reading_order
                    )
                    blocks.append(
                        TextBlock(
                            block_id=block_id,
                            page_number=page_number,
                            reading_order=reading_order,
                            bbox=_bbox(source_block.get("bbox")),
                            block_type="paragraph",
                            raw_text=raw_text,
                            normalized_text=normalized,
                            spans=spans,
                        )
                    )
                    page_block_ids.append(block_id)
                    raw_parts.append(raw_text)
                    reading_order += 1

                if self.extract_tables and hasattr(page, "find_tables"):
                    try:
                        finder = page.find_tables()
                        for table_index, source_table in enumerate(finder.tables):
                            table_id = stable_id(
                                "table", expected_document_id, page_number, table_index
                            )
                            extracted = source_table.extract()
                            cells: list[TableCell] = []
                            for row_index, row in enumerate(extracted):
                                for column_index, value in enumerate(row):
                                    raw_text = _safe_text(value or "")
                                    normalized, _ = normalize_text(raw_text)
                                    cells.append(
                                        TableCell(
                                            cell_id=stable_id(
                                                "cell", table_id, row_index, column_index
                                            ),
                                            row=row_index,
                                            column=column_index,
                                            bbox=_bbox(source_table.bbox),
                                            raw_text=raw_text,
                                            normalized_text=normalized,
                                            is_header=row_index == 0,
                                        )
                                    )
                            tables.append(
                                Table(
                                    table_id=table_id,
                                    page_number=page_number,
                                    bbox=_bbox(source_table.bbox),
                                    cells=cells,
                                    extraction_warnings=[
                                        "Cell bounding boxes use the enclosing table bbox in V1."
                                    ],
                                )
                            )
                    except Exception as exc:  # noqa: BLE001
                        warnings.append(
                            f"Table extraction failed on page {page_number}: {type(exc).__name__}: {exc}"
                        )

                page_raw = "\n\n".join(raw_parts)
                page_normalized, normalization_map = normalize_text(page_raw)
                pages.append(
                    ParsedPage(
                        page_number=page_number,
                        width=float(page.rect.width),
                        height=float(page.rect.height),
                        raw_text=page_raw,
                        normalized_text=page_normalized,
                        normalization_map=normalization_map,
                        block_ids=page_block_ids,
                    )
                )

        text_count = sum(len(page.normalized_text) for page in pages)
        if text_count < self.min_text_characters:
            warnings.append(
                f"Extracted only {text_count} normalized text characters; OCR review is recommended."
            )
        return ParsedDocument(
            paper_id=paper_id,
            document_id=expected_document_id,
            pdf_sha256=digest,
            source_aliases=source_aliases,
            title=title,
            domain=domain,  # type: ignore[arg-type]
            page_count=len(pages),
            pages=pages,
            blocks=blocks,
            tables=tables,
            figures=figures,
            parser=Producer(
                name=self.name,
                version=self.version,
                config_hash=self.config_hash(),
            ),
            warnings=warnings,
        )


def _safe_text(value: object) -> str:
    return str(value).encode("utf-8", errors="replace").decode("utf-8")


def _bbox(value: object) -> BoundingBox:
    if value is None:
        return BoundingBox(x0=0, y0=0, x1=0, y1=0)
    coordinates = tuple(float(item) for item in value)  # type: ignore[arg-type]
    return BoundingBox(
        x0=coordinates[0],
        y0=coordinates[1],
        x1=coordinates[2],
        y1=coordinates[3],
    )
