from __future__ import annotations

from pathlib import Path
from typing import Protocol

from paper_library.schemas.parsed_document import ParsedDocument


class DocumentParser(Protocol):
    name: str
    version: str

    def parse(
        self,
        path: Path,
        *,
        paper_id: str,
        source_aliases: list[str],
        domain: str,
        pdf_sha256: str | None = None,
    ) -> ParsedDocument:
        """Parse one immutable PDF into a provenance-preserving document."""
