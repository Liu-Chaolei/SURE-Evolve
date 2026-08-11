from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from paper_library.application.validate import CorpusValidationResult, validate_corpus
from paper_library.config import PaperLibraryConfig
from paper_library.extraction.design_cards import ensure_stable_card_ids
from paper_library.extraction.evidence import ensure_stable_evidence_ids
from paper_library.schemas.evidence_record import EvidenceRecord
from paper_library.schemas.parsed_document import ParsedDocument
from paper_library.schemas.research_design_card import ResearchDesignCard
from paper_library.storage.atomic import write_model_atomic
from paper_library.storage.corpus import load_corpus_manifest
from paper_library.storage.jsonl import write_jsonl_atomic
from paper_library.storage.layout import BuildLayout
from paper_library.storage.manifest import (
    CorpusManifest,
    CurrentManifest,
    RecordArtifact,
)
from paper_library.utils.hashing import hash_canonical, sha256_file


@dataclass(frozen=True)
class PublishResult:
    manifest: CorpusManifest
    validation: CorpusValidationResult


def publish_corpus(
    config: PaperLibraryConfig,
    documents: list[ParsedDocument],
    evidence_records: list[EvidenceRecord],
    cards: list[ResearchDesignCard],
) -> PublishResult:
    documents = sorted(documents, key=lambda item: item.document_id)
    evidence_records = sorted(evidence_records, key=lambda item: item.evidence_id)
    cards = sorted(cards, key=lambda item: item.card_id)
    ensure_stable_evidence_ids(evidence_records)
    ensure_stable_card_ids(cards)
    validation = validate_corpus(
        documents,
        evidence_records,
        cards,
        minimum_grounding_completeness=config.validation.minimum_grounding_completeness,
        require_exact_grounding=config.validation.require_exact_grounding,
    )
    if not validation.valid:
        messages = "; ".join(issue.message for issue in validation.issues if issue.severity == "error")
        raise ValueError(f"Corpus validation failed: {messages}")
    if not config.validation.publish_review_required:
        review_required = [
            record.evidence_id for record in evidence_records if record.quality.review_required
        ] + [card.card_id for card in cards if card.quality.review_required]
        if review_required:
            raise ValueError(
                "Review-required records cannot be published: " + ", ".join(review_required)
            )

    layout = BuildLayout(config.paths.build_root)
    layout.ensure()
    identity = {
        "config_hash": config.content_hash(),
        "documents": [
            document.model_dump(mode="json", exclude={"created_at"})
            for document in documents
        ],
        "evidence": [
            record.model_dump(mode="json", exclude={"created_at"})
            for record in evidence_records
        ],
        "cards": [
            card.model_dump(mode="json", exclude={"created_at"}) for card in cards
        ],
    }
    corpus_version = f"corpus_{hash_canonical(identity)[:24]}"
    destination = layout.records / corpus_version
    if destination.exists():
        manifest, _ = load_corpus_manifest(layout, corpus_version)
        _update_current(layout, manifest)
        return PublishResult(manifest=manifest, validation=validation)

    staging = Path(tempfile.mkdtemp(prefix=f".{corpus_version}.", dir=layout.staging))
    try:
        parsed_path = staging / "parsed_documents.jsonl"
        evidence_path = staging / "evidence_records.jsonl"
        cards_path = staging / "research_design_cards.jsonl"
        write_jsonl_atomic(parsed_path, documents)
        write_jsonl_atomic(evidence_path, evidence_records)
        write_jsonl_atomic(cards_path, cards)
        manifest = CorpusManifest(
            corpus_version=corpus_version,
            config_hash=config.content_hash(),
            parsed_documents=_artifact(parsed_path, len(documents), corpus_version),
            evidence_records=_artifact(evidence_path, len(evidence_records), corpus_version),
            research_design_cards=_artifact(cards_path, len(cards), corpus_version),
            source_pdf_hashes=sorted(document.pdf_sha256 for document in documents),
            warning_count=sum(len(document.warnings) for document in documents)
            + sum(len(record.quality.warnings) for record in evidence_records)
            + sum(len(card.quality.warnings) for card in cards)
            + sum(issue.severity == "warning" for issue in validation.issues),
        )
        staging.rename(destination)
        write_model_atomic(layout.manifests / f"{corpus_version}.json", manifest)
        _update_current(layout, manifest)
        return PublishResult(manifest=manifest, validation=validation)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _artifact(path: Path, count: int, corpus_version: str) -> RecordArtifact:
    return RecordArtifact(
        path=f"records/{corpus_version}/{path.name}",
        record_count=count,
        sha256=sha256_file(path),
    )


def _update_current(layout: BuildLayout, manifest: CorpusManifest) -> None:
    write_model_atomic(
        layout.manifests / "current.json",
        CurrentManifest(
            corpus_version=manifest.corpus_version,
            manifest_path=f"manifests/{manifest.corpus_version}.json",
        ),
    )
