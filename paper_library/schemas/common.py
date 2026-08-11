from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Identifier = Annotated[str, StringConstraints(min_length=3, pattern=r"^[a-z][a-z0-9_-]*$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
SchemaVersion = Annotated[str, StringConstraints(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ReportingStatus(StrEnum):
    REPORTED = "reported"
    INFERRED = "inferred"
    NOT_REPORTED = "not_reported"
    NOT_EXTRACTED = "not_extracted"
    NOT_APPLICABLE = "not_applicable"
    CONFLICTING = "conflicting"


class GroundingStatus(StrEnum):
    VERIFIED = "verified"
    NORMALIZED_MATCH = "normalized_match"
    UNVERIFIED = "unverified"
    CONFLICTING = "conflicting"


class ScientificEvidenceStrength(StrEnum):
    UNSPECIFIED = "unspecified"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    CONFLICTING = "conflicting"


class SourceObjectType(StrEnum):
    BLOCK = "block"
    TABLE = "table"
    TABLE_CELL = "table_cell"
    FIGURE = "figure"
    EQUATION = "equation"


class BoundingBox(StrictModel):
    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def validate_extent(self) -> "BoundingBox":
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("bbox upper bounds must not precede lower bounds")
        return self


class Producer(StrictModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    config_hash: Sha256 | None = None
    prompt_version: str | None = None
    model: str | None = None


class SourceReference(StrictModel):
    document_id: Identifier
    object_type: SourceObjectType
    object_id: Identifier
    page_number: int = Field(ge=1)
    bbox: BoundingBox | None = None
    raw_start: int | None = Field(default=None, ge=0)
    raw_end: int | None = Field(default=None, ge=0)
    verbatim_text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_offsets(self) -> "SourceReference":
        if (self.raw_start is None) != (self.raw_end is None):
            raise ValueError("raw offsets must be both present or both absent")
        if self.raw_start is not None and self.raw_end <= self.raw_start:
            raise ValueError("raw_end must be greater than raw_start")
        return self


class NumericFact(StrictModel):
    name: str = Field(min_length=1)
    status: ReportingStatus = ReportingStatus.REPORTED
    value: float | None = None
    unit: str | None = None
    metric: str | None = None
    direction: Literal["higher_is_better", "lower_is_better", "neutral"] | None = None
    dataset: str | None = None
    split: str | None = None
    baseline_value: float | None = None
    candidate_value: float | None = None
    delta: float | None = None
    variance: float | None = None
    seeds: int | None = Field(default=None, ge=1)
    significance: str | None = None
    source_ref_indexes: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status_value(self) -> "NumericFact":
        numeric_values = (
            self.value,
            self.baseline_value,
            self.candidate_value,
            self.delta,
            self.variance,
        )
        has_value = any(value is not None for value in numeric_values)
        if self.status in {
            ReportingStatus.NOT_REPORTED,
            ReportingStatus.NOT_EXTRACTED,
            ReportingStatus.NOT_APPLICABLE,
        } and has_value:
            raise ValueError(f"{self.status.value} numeric facts cannot contain values")
        if self.status in {ReportingStatus.REPORTED, ReportingStatus.INFERRED} and not has_value:
            raise ValueError(f"{self.status.value} numeric facts require a value")
        return self


class ValidationIssue(StrictModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    severity: Literal["info", "warning", "error"]
    record_id: str | None = None
    field_path: str | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
