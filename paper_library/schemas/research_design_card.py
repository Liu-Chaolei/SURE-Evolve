from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field, model_validator

from paper_library.schemas.common import (
    Identifier,
    NumericFact,
    Producer,
    ReportingStatus,
    SchemaVersion,
    ScientificEvidenceStrength,
    StrictModel,
    utc_now,
)

RESEARCH_DESIGN_CARD_SCHEMA_VERSION = "1.0.0"


class DesignCardType(StrEnum):
    INTERVENTION = "intervention"
    ABLATION = "ablation"
    DIAGNOSTIC = "diagnostic"
    NEGATIVE_RESULT = "negative_result"
    TRADEOFF = "tradeoff"
    SYSTEM = "system"


class CardRelationType(StrEnum):
    DEPENDS_ON = "depends_on"
    PART_OF = "part_of"
    EXTENDS = "extends"
    EVALUATED_WITH = "evaluated_with"


class HypothesisType(StrEnum):
    AUTHOR_EXPLICIT = "author_explicit"
    CURATOR_RECONSTRUCTED = "curator_reconstructed"
    TESTED_RELATIONSHIP_ONLY = "tested_relationship_only"


class CardStatement(StrictModel):
    status: ReportingStatus
    value: str | None = None

    @model_validator(mode="after")
    def validate_value(self) -> "CardStatement":
        absent = {
            ReportingStatus.NOT_REPORTED,
            ReportingStatus.NOT_EXTRACTED,
            ReportingStatus.NOT_APPLICABLE,
        }
        if self.status in absent and self.value is not None:
            raise ValueError(f"{self.status.value} statements cannot contain values")
        if self.status in {ReportingStatus.REPORTED, ReportingStatus.INFERRED} and not self.value:
            raise ValueError(f"{self.status.value} statements require values")
        return self


class Hypothesis(StrictModel):
    hypothesis_type: HypothesisType
    statement: CardStatement


class CardRelation(StrictModel):
    relation_type: CardRelationType
    target_card_id: Identifier


class EvaluationSetup(StrictModel):
    datasets: list[str] = Field(default_factory=list)
    splits: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    evaluation_backends: list[str] = Field(default_factory=list)


class DesignCardQuality(StrictModel):
    extraction_confidence: float = Field(ge=0, le=1)
    grounding_completeness: float = Field(ge=0, le=1)
    scientific_strength: ScientificEvidenceStrength = ScientificEvidenceStrength.UNSPECIFIED
    review_required: bool = False
    warnings: list[str] = Field(default_factory=list)


class ResearchDesignCard(StrictModel):
    schema_version: SchemaVersion = RESEARCH_DESIGN_CARD_SCHEMA_VERSION
    card_id: Identifier
    paper_id: Identifier
    source_document_ids: list[Identifier] = Field(min_length=1)
    card_type: DesignCardType
    title: str = Field(min_length=1)
    parent_card_id: Identifier | None = None
    child_card_ids: list[Identifier] = Field(default_factory=list)
    relations: list[CardRelation] = Field(default_factory=list)
    research_problem: CardStatement
    research_question: CardStatement
    hypothesis: Hypothesis
    tested_relationship: CardStatement
    base_method: CardStatement
    intervention_delta: CardStatement
    candidate_system: CardStatement
    baselines: list[str] = Field(default_factory=list)
    explicit_controls: list[str] = Field(default_factory=list)
    assumed_fixed: list[str] = Field(default_factory=list)
    training_objectives: list[str] = Field(default_factory=list)
    training_backends: list[str] = Field(default_factory=list)
    evaluation: EvaluationSetup = Field(default_factory=EvaluationSetup)
    results: list[NumericFact] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    applicability: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evidence_ids: list[Identifier] = Field(min_length=1)
    field_evidence: dict[str, list[Identifier]] = Field(default_factory=dict)
    builder: Producer
    quality: DesignCardQuality
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_card(self) -> "ResearchDesignCard":
        if self.card_id in self.child_card_ids:
            raise ValueError("a card cannot be its own child")
        if self.parent_card_id == self.card_id:
            raise ValueError("a card cannot be its own parent")
        evidence = set(self.evidence_ids)
        referenced = {item for items in self.field_evidence.values() for item in items}
        if not referenced.issubset(evidence):
            raise ValueError("field_evidence must reference evidence_ids on the card")
        factual_fields = {
            "research_problem": self.research_problem,
            "research_question": self.research_question,
            "tested_relationship": self.tested_relationship,
            "base_method": self.base_method,
            "intervention_delta": self.intervention_delta,
            "candidate_system": self.candidate_system,
        }
        for path, statement in factual_fields.items():
            if statement.status == ReportingStatus.REPORTED and not self.field_evidence.get(path):
                raise ValueError(f"reported field {path!r} requires field-level evidence")
        if (
            self.hypothesis.statement.status == ReportingStatus.REPORTED
            and not self.field_evidence.get("hypothesis.statement")
        ):
            raise ValueError("reported hypothesis requires field-level evidence")
        if self.results and not any(path.startswith("results") for path in self.field_evidence):
            raise ValueError("reported results require field-level evidence")
        return self
