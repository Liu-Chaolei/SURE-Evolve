"""Build a provenance-preserving ASR/TTS research paper library."""

from paper_library.schemas.evidence_record import EvidenceRecord
from paper_library.schemas.parsed_document import ParsedDocument
from paper_library.schemas.research_design_card import ResearchDesignCard

__all__ = ["EvidenceRecord", "ParsedDocument", "ResearchDesignCard"]
__version__ = "0.1.0"
