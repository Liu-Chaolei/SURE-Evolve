from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from paper_library.extraction.grounding import ground_evidence
from paper_library.schemas.common import (
    GroundingStatus,
    ReportingStatus,
    ValidationIssue,
)
from paper_library.schemas.evidence_record import EvidenceRecord
from paper_library.schemas.parsed_document import ParsedDocument
from paper_library.schemas.research_design_card import ResearchDesignCard


@dataclass(frozen=True)
class CorpusValidationResult:
    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


def validate_corpus(
    documents: Iterable[ParsedDocument],
    evidence_records: Iterable[EvidenceRecord],
    cards: Iterable[ResearchDesignCard],
    *,
    minimum_grounding_completeness: float = 0.9,
    require_exact_grounding: bool = True,
) -> CorpusValidationResult:
    if not 0 <= minimum_grounding_completeness <= 1:
        raise ValueError("minimum_grounding_completeness must be between zero and one")
    issues: list[ValidationIssue] = []
    document_map = _unique_map(documents, "document_id", issues)
    evidence_map = _unique_map(evidence_records, "evidence_id", issues)
    card_map = _unique_map(cards, "card_id", issues)

    for evidence in evidence_map.values():
        document = document_map.get(evidence.document_id)
        if document is None:
            issues.append(
                ValidationIssue(
                    code="missing_document",
                    message=f"Evidence references missing document {evidence.document_id}",
                    severity="error",
                    record_id=evidence.evidence_id,
                )
            )
            continue
        result = ground_evidence(evidence, document)
        if evidence.paper_id != document.paper_id:
            issues.append(
                ValidationIssue(
                    code="evidence_paper_mismatch",
                    message=(
                        f"Evidence paper {evidence.paper_id} does not match "
                        f"document paper {document.paper_id}"
                    ),
                    severity="error",
                    record_id=evidence.evidence_id,
                )
            )
        if (
            evidence.quality.grounding_status != result.status
            or abs(evidence.quality.grounding_completeness - result.completeness) > 1e-9
        ):
            issues.append(
                ValidationIssue(
                    code="stale_grounding_quality",
                    message=(
                        "Stored grounding quality does not match recomputed "
                        f"status={result.status.value}, completeness={result.completeness:.3f}"
                    ),
                    severity="error",
                    record_id=evidence.evidence_id,
                )
            )
        if result.completeness < minimum_grounding_completeness:
            issues.append(
                ValidationIssue(
                    code="low_grounding_completeness",
                    message=(
                        f"Grounding completeness {result.completeness:.3f} is below "
                        f"{minimum_grounding_completeness:.3f}"
                    ),
                    severity="error",
                    record_id=evidence.evidence_id,
                )
            )
        if require_exact_grounding and result.status != GroundingStatus.VERIFIED:
            issues.append(
                ValidationIssue(
                    code="grounding_not_exact",
                    message=f"Grounding status is {result.status.value}",
                    severity="error",
                    record_id=evidence.evidence_id,
                )
            )
        for message in result.errors:
            issues.append(
                ValidationIssue(
                    code="ungrounded_source",
                    message=message,
                    severity="error",
                    record_id=evidence.evidence_id,
                )
            )
        for message in result.warnings:
            issues.append(
                ValidationIssue(
                    code="normalized_grounding",
                    message=message,
                    severity="warning",
                    record_id=evidence.evidence_id,
                )
            )

    for card in card_map.values():
        source_paper_ids: set[str] = set()
        for document_id in card.source_document_ids:
            document = document_map.get(document_id)
            if document is None:
                issues.append(
                    ValidationIssue(
                        code="missing_card_document",
                        message=f"Card references missing document {document_id}",
                        severity="error",
                        record_id=card.card_id,
                    )
                )
            else:
                source_paper_ids.add(document.paper_id)
        if source_paper_ids and source_paper_ids != {card.paper_id}:
            issues.append(
                ValidationIssue(
                    code="card_paper_mismatch",
                    message="Card source documents do not all match its paper_id",
                    severity="error",
                    record_id=card.card_id,
                )
            )
        for evidence_id in card.evidence_ids:
            evidence = evidence_map.get(evidence_id)
            if evidence is None:
                issues.append(
                    ValidationIssue(
                        code="missing_card_evidence",
                        message=f"Card references missing evidence {evidence_id}",
                        severity="error",
                        record_id=card.card_id,
                    )
                )
            elif evidence.document_id not in card.source_document_ids:
                issues.append(
                    ValidationIssue(
                        code="cross_document_card_evidence",
                        message=f"Evidence {evidence_id} is outside card source documents",
                        severity="error",
                        record_id=card.card_id,
                    )
                )
            elif evidence.paper_id != card.paper_id:
                issues.append(
                    ValidationIssue(
                        code="cross_paper_card_evidence",
                        message=f"Evidence {evidence_id} belongs to another paper",
                        severity="error",
                        record_id=card.card_id,
                    )
                )
        related_ids = [card.parent_card_id, *card.child_card_ids]
        related_ids += [relation.target_card_id for relation in card.relations]
        for related_id in filter(None, related_ids):
            if related_id not in card_map:
                issues.append(
                    ValidationIssue(
                        code="missing_related_card",
                        message=f"Card relation references missing card {related_id}",
                        severity="error",
                        record_id=card.card_id,
                    )
                )
        for index, result in enumerate(card.results):
            if result.status == ReportingStatus.REPORTED:
                path = f"results.{index}"
                if not card.field_evidence.get(path):
                    issues.append(
                        ValidationIssue(
                            code="missing_result_evidence",
                            message=f"Reported result {path} lacks field-level evidence",
                            severity="error",
                            record_id=card.card_id,
                            field_path=path,
                        )
                    )
    return CorpusValidationResult(issues=tuple(issues))


def _unique_map(
    records: Iterable[object],
    attribute: str,
    issues: list[ValidationIssue],
) -> dict[str, object]:
    values: dict[str, object] = {}
    for record in records:
        record_id = str(getattr(record, attribute))
        if record_id in values:
            issues.append(
                ValidationIssue(
                    code="duplicate_id",
                    message=f"Duplicate {attribute}: {record_id}",
                    severity="error",
                    record_id=record_id,
                )
            )
        values[record_id] = record
    return values
