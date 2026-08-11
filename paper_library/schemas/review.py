from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field, model_validator

from paper_library.schemas.common import Identifier, StrictModel, utc_now


class ReviewObjectType(StrEnum):
    EVIDENCE = "evidence"
    CARD = "card"


class ReviewDecisionType(StrEnum):
    ACCEPTED = "accepted"
    REVISED = "revised"
    REJECTED = "rejected"


class ReviewCandidate(StrictModel):
    review_id: Identifier
    object_type: ReviewObjectType
    record_id: Identifier
    document_id: Identifier
    candidate_uri: str = Field(min_length=1)
    reasons: list[str] = Field(min_length=1)
    upstream_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime = Field(default_factory=utc_now)


class ReviewDecision(StrictModel):
    decision_id: Identifier
    review_id: Identifier
    decision: ReviewDecisionType
    reviewer: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    revised_record_uri: str | None = None
    upstream_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_revision(self) -> ReviewDecision:
        if self.decision == ReviewDecisionType.REVISED and not self.revised_record_uri:
            raise ValueError("revised decisions require revised_record_uri")
        if self.decision != ReviewDecisionType.REVISED and self.revised_record_uri is not None:
            raise ValueError("only revised decisions can contain revised_record_uri")
        return self
