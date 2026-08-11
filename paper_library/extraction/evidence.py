from __future__ import annotations

from collections.abc import Iterable

from paper_library.schemas.evidence_record import EvidenceRecord
from paper_library.utils.ids import stable_id


def evidence_id_for(record: EvidenceRecord) -> str:
    anchors = sorted(
        (
            reference.document_id,
            reference.object_type.value,
            reference.object_id,
            reference.raw_start,
            reference.raw_end,
            reference.verbatim_text,
        )
        for reference in record.source_refs
    )
    return stable_id(
        "evidence",
        record.schema_version,
        record.role.value,
        anchors,
    )


def ensure_stable_evidence_ids(records: Iterable[EvidenceRecord]) -> None:
    seen: set[str] = set()
    for record in records:
        expected = evidence_id_for(record)
        if record.evidence_id != expected:
            raise ValueError(
                f"Evidence ID {record.evidence_id!r} does not match stable source anchors: {expected}"
            )
        if record.evidence_id in seen:
            raise ValueError(f"Duplicate Evidence ID: {record.evidence_id}")
        seen.add(record.evidence_id)
