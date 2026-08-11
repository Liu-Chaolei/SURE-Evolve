from __future__ import annotations

from pathlib import Path

from paper_library.schemas.review import ReviewCandidate, ReviewDecision
from paper_library.storage.jsonl import read_jsonl, write_jsonl_atomic
from paper_library.utils.hashing import hash_canonical
from paper_library.utils.ids import stable_id


def review_candidate_for(
    *,
    object_type: str,
    record_id: str,
    document_id: str,
    candidate_uri: str,
    reasons: list[str],
    upstream_hash: str,
) -> ReviewCandidate:
    return ReviewCandidate(
        review_id=stable_id("review", object_type, record_id, upstream_hash),
        object_type=object_type,
        record_id=record_id,
        document_id=document_id,
        candidate_uri=candidate_uri,
        reasons=reasons,
        upstream_hash=upstream_hash,
    )


def review_decision_for(
    candidate: ReviewCandidate,
    *,
    decision: str,
    reviewer: str,
    reason: str,
    revised_record_uri: str | None = None,
) -> ReviewDecision:
    identity = hash_canonical(
        {
            "review_id": candidate.review_id,
            "decision": decision,
            "reviewer": reviewer,
            "reason": reason,
            "revised_record_uri": revised_record_uri,
        }
    )
    return ReviewDecision(
        decision_id=stable_id("decision", identity),
        review_id=candidate.review_id,
        decision=decision,
        reviewer=reviewer,
        reason=reason,
        revised_record_uri=revised_record_uri,
        upstream_hash=candidate.upstream_hash,
    )


def append_review_record(path: Path, record: ReviewCandidate | ReviewDecision) -> None:
    existing: list[ReviewCandidate | ReviewDecision] = []
    if path.is_file():
        model_type = ReviewCandidate if isinstance(record, ReviewCandidate) else ReviewDecision
        existing = list(read_jsonl(path, model_type))
        record_id = record.review_id if isinstance(record, ReviewCandidate) else record.decision_id
        existing_ids = {
            item.review_id if isinstance(item, ReviewCandidate) else item.decision_id
            for item in existing
        }
        if record_id in existing_ids:
            return
    write_jsonl_atomic(path, [*existing, record])
