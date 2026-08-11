from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field, model_validator

from paper_library.schemas.common import (
    GroundingStatus,
    Identifier,
    NumericFact,
    Producer,
    ReportingStatus,
    SchemaVersion,
    ScientificEvidenceStrength,
    SourceReference,
    StrictModel,
    utc_now,
)

EVIDENCE_RECORD_SCHEMA_VERSION = "1.0.0"


class EvidenceRole(StrEnum):
    RESEARCH_PROBLEM = "research_problem"
    RESEARCH_GAP = "research_gap"
    AUTHOR_HYPOTHESIS = "author_hypothesis"
    PROPOSED_MECHANISM = "proposed_mechanism"
    METHOD_DESCRIPTION = "method_description"
    IMPLEMENTATION_DETAIL = "implementation_detail"
    DATA_SETUP = "data_setup"
    TRAINING_SETUP = "training_setup"
    EVALUATION_PROTOCOL = "evaluation_protocol"
    BASELINE_DEFINITION = "baseline_definition"
    CONTROL_DEFINITION = "control_definition"
    REPORTED_RESULT = "reported_result"
    ABLATION_RESULT = "ablation_result"
    NEGATIVE_RESULT = "negative_result"
    TRADEOFF = "tradeoff"
    LIMITATION = "limitation"
    FUTURE_WORK = "future_work"


class EvidenceQuality(StrictModel):
    extraction_confidence: float = Field(ge=0, le=1)
    grounding_status: GroundingStatus
    grounding_completeness: float = Field(ge=0, le=1)
    scientific_strength: ScientificEvidenceStrength = ScientificEvidenceStrength.UNSPECIFIED
    review_required: bool = False
    warnings: list[str] = Field(default_factory=list)


class EvidenceRecord(StrictModel):
    schema_version: SchemaVersion = EVIDENCE_RECORD_SCHEMA_VERSION
    evidence_id: Identifier
    paper_id: Identifier
    document_id: Identifier
    role: EvidenceRole
    status: ReportingStatus
    claim: str | None = None
    conditions: dict[str, str] = Field(default_factory=dict)
    numeric_facts: list[NumericFact] = Field(default_factory=list)
    source_refs: list[SourceReference] = Field(default_factory=list)
    extractor: Producer
    quality: EvidenceQuality
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_record(self) -> "EvidenceRecord":
        absent = {
            ReportingStatus.NOT_REPORTED,
            ReportingStatus.NOT_EXTRACTED,
            ReportingStatus.NOT_APPLICABLE,
        }
        if self.status in absent and self.claim is not None:
            raise ValueError(f"{self.status.value} evidence cannot contain a claim")
        if self.status in {ReportingStatus.REPORTED, ReportingStatus.INFERRED}:
            if not self.claim:
                raise ValueError(f"{self.status.value} evidence requires a claim")
            if not self.source_refs:
                raise ValueError(f"{self.status.value} evidence requires source references")
        if any(ref.document_id != self.document_id for ref in self.source_refs):
            raise ValueError("evidence source references cannot cross documents")
        for fact in self.numeric_facts:
            if any(index >= len(self.source_refs) for index in fact.source_ref_indexes):
                raise ValueError("numeric fact source_ref_indexes are out of range")
        if self.quality.grounding_status == GroundingStatus.VERIFIED and not self.source_refs:
            raise ValueError("verified evidence requires source references")
        return self
