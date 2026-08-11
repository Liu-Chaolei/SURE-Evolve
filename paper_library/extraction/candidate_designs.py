from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from paper_library.schemas.common import Identifier, SourceObjectType, StrictModel
from paper_library.schemas.research_design_card import DesignCardType
from paper_library.utils.ids import stable_id


class CandidateSource(StrictModel):
    object_type: SourceObjectType
    object_id: Identifier


class CandidateDesign(StrictModel):
    candidate_id: Identifier
    paper_id: Identifier
    document_id: Identifier
    card_type: DesignCardType
    title: str = Field(min_length=1)
    source_objects: list[CandidateSource] = Field(default_factory=list)
    parent_candidate_id: Identifier | None = None
    rationale: str | None = None


class CandidateDiscoveryStrategy(StrEnum):
    STRUCTURAL = "structural"
    PROVIDER = "provider"


def structural_candidates(document_id: str, paper_id: str, title: str | None) -> list[CandidateDesign]:
    candidate_title = title or "Paper-level research system"
    return [
        CandidateDesign(
            candidate_id=stable_id("candidate", paper_id, "system"),
            paper_id=paper_id,
            document_id=document_id,
            card_type=DesignCardType.SYSTEM,
            title=candidate_title,
            rationale="Deterministic paper-level fallback candidate; requires review and refinement.",
        )
    ]
