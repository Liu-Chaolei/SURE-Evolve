from __future__ import annotations

from datetime import datetime

from pydantic import Field

from paper_library.schemas.common import Sha256, StrictModel, utc_now

CORPUS_MANIFEST_SCHEMA_VERSION = "1.0.0"


class RecordArtifact(StrictModel):
    path: str = Field(min_length=1)
    record_count: int = Field(ge=0)
    sha256: Sha256


class CorpusManifest(StrictModel):
    schema_version: str = CORPUS_MANIFEST_SCHEMA_VERSION
    corpus_version: str = Field(pattern=r"^corpus_[0-9a-f]{24}$")
    config_hash: Sha256
    parsed_documents: RecordArtifact
    evidence_records: RecordArtifact
    research_design_cards: RecordArtifact
    source_pdf_hashes: list[Sha256] = Field(default_factory=list)
    warning_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)


class CurrentManifest(StrictModel):
    corpus_version: str = Field(pattern=r"^corpus_[0-9a-f]{24}$")
    manifest_path: str = Field(min_length=1)
    updated_at: datetime = Field(default_factory=utc_now)
