from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from paper_library.application.publish import PublishResult, publish_corpus
from paper_library.application.reviews import append_review_record, review_candidate_for
from paper_library.application.scanner import SourceDocument, scan_pdfs
from paper_library.application.validate import CorpusValidationResult, validate_corpus
from paper_library.config import PaperLibraryConfig
from paper_library.errors import BuildStageError, ErrorCode, PaperLibraryError
from paper_library.extraction.candidate_designs import (
    CandidateDesign,
    structural_candidates,
)
from paper_library.extraction.execution import (
    build_design_cards,
    discover_candidates,
    extract_evidence_records,
)
from paper_library.extraction.providers import StructuredExtractionProvider
from paper_library.parsing.pymupdf_parser import PyMuPDFParser
from paper_library.schemas.common import ValidationIssue
from paper_library.schemas.evidence_record import EvidenceRecord
from paper_library.schemas.parsed_document import ParsedDocument
from paper_library.schemas.research_design_card import ResearchDesignCard
from paper_library.schemas.review import ReviewObjectType
from paper_library.storage.atomic import write_json_atomic, write_model_atomic
from paper_library.storage.jsonl import read_jsonl, write_jsonl_atomic
from paper_library.storage.layout import BuildLayout
from paper_library.storage.state import BuildState, StageName, StageRecord, StageStatus
from paper_library.utils.hashing import hash_canonical, sha256_file


@dataclass(frozen=True)
class ScanResult:
    documents: list[SourceDocument]
    missing: int


@dataclass(frozen=True)
class ParseResult:
    parsed: int
    cache_hits: int
    failed: int


@dataclass(frozen=True)
class ExtractionResult:
    processed: int
    cache_hits: int
    failed: int
    review_queued: int = 0


@dataclass(frozen=True)
class CanonicalRecords:
    documents: list[ParsedDocument]
    evidence_records: list[EvidenceRecord]
    cards: list[ResearchDesignCard]
    incomplete_document_ids: list[str]


@dataclass(frozen=True)
class BuildResult:
    scan: ScanResult
    parse: ParseResult
    candidates: ExtractionResult
    evidence: ExtractionResult
    cards: ExtractionResult
    validation: CorpusValidationResult
    publication: PublishResult


def initialize(config: PaperLibraryConfig) -> tuple[BuildLayout, BuildState]:
    layout = BuildLayout(config.paths.build_root)
    layout.ensure()
    state = BuildState(layout.state_database)
    state.initialize()
    return layout, state


def scan(config: PaperLibraryConfig) -> ScanResult:
    _, state = initialize(config)
    documents = scan_pdfs(config.paths.source_root)
    _, missing = state.sync_sources(documents)
    return ScanResult(documents=documents, missing=missing)


def parse_documents(
    config: PaperLibraryConfig,
    *,
    force: bool = False,
    document_id_filter: str | None = None,
) -> ParseResult:
    layout, state = initialize(config)
    scanned = scan_pdfs(config.paths.source_root)
    state.sync_sources(scanned)
    parser = PyMuPDFParser(
        min_text_characters=config.parser.min_text_characters,
        extract_tables=config.parser.extract_tables,
        extract_figures=config.parser.extract_figures,
    )
    parsed = cache_hits = failed = 0
    for source in scanned:
        if document_id_filter and source.document_id != document_id_filter:
            continue
        cache_key = hash_canonical(
            {
                "stage": StageName.PARSE.value,
                "pdf_sha256": source.pdf_sha256,
                "parser": parser.name,
                "parser_version": parser.version,
                "parser_config": parser.config_hash(),
                "schema_version": "1.0.0",
            }
        )
        output = layout.parsed_objects / f"{cache_key}.json"
        if _cache_hit(
            layout,
            state,
            source.document_id,
            StageName.PARSE,
            cache_key,
            output,
            force,
        ):
            cache_hits += 1
            continue
        state.start_stage(source.document_id, StageName.PARSE, cache_key, source.pdf_sha256)
        try:
            document = parser.parse(
                source.primary_path,
                paper_id=source.paper_id,
                source_aliases=list(source.aliases),
                domain=source.domain,
                pdf_sha256=source.pdf_sha256,
            )
            write_model_atomic(output, document)
            state.finish_stage(
                source.document_id,
                StageName.PARSE,
                output_uri=output.relative_to(layout.root).as_posix(),
                output_sha256=sha256_file(output),
                output_record_count=1,
            )
            parsed += 1
        except Exception as exc:
            _fail(state, source.document_id, StageName.PARSE, "parse_failed", exc)
            failed += 1
            if config.runtime.fail_fast:
                raise
    return ParseResult(parsed=parsed, cache_hits=cache_hits, failed=failed)


def extract_candidates(
    config: PaperLibraryConfig,
    *,
    provider: StructuredExtractionProvider | None = None,
    force: bool = False,
    document_id_filter: str | None = None,
) -> ExtractionResult:
    layout, state = initialize(config)
    processed = cache_hits = failed = review_queued = 0
    for document, parse_key in _parsed_documents(layout, state, document_id_filter):
        strategy = "provider" if provider is not None else "structural"
        cache_key = _extraction_cache_key(
            config, StageName.CANDIDATE_DESIGNS, [parse_key], provider, strategy
        )
        output = layout.candidate_objects / f"{cache_key}.jsonl"
        if _cache_hit(
            layout,
            state,
            document.document_id,
            StageName.CANDIDATE_DESIGNS,
            cache_key,
            output,
            force,
        ):
            cache_hits += 1
            continue
        state.start_stage(document.document_id, StageName.CANDIDATE_DESIGNS, cache_key, parse_key)
        try:
            candidates = (
                discover_candidates(document, provider)
                if provider is not None
                else structural_candidates(document.document_id, document.paper_id, document.title)
            )
            write_jsonl_atomic(output, candidates)
            state.finish_stage(
                document.document_id,
                StageName.CANDIDATE_DESIGNS,
                output_uri=output.relative_to(layout.root).as_posix(),
                output_sha256=sha256_file(output),
                output_record_count=len(candidates),
            )
            processed += 1
            if provider is None:
                review_queued += len(candidates)
        except Exception as exc:
            _fail(state, document.document_id, StageName.CANDIDATE_DESIGNS, "candidate_extraction_failed", exc)
            failed += 1
            if config.runtime.fail_fast:
                raise
    return ExtractionResult(processed, cache_hits, failed, review_queued)


def extract_evidence(
    config: PaperLibraryConfig,
    provider: StructuredExtractionProvider,
    *,
    force: bool = False,
    document_id_filter: str | None = None,
) -> ExtractionResult:
    layout, state = initialize(config)
    processed = cache_hits = failed = review_queued = 0
    for document, parse_key in _parsed_documents(layout, state, document_id_filter):
        candidate_stage = state.get_stage(document.document_id, StageName.CANDIDATE_DESIGNS)
        if not _usable_stage(candidate_stage) or not _stage_artifact_is_valid(
            layout,
            state,
            candidate_stage,
        ):
            continue
        candidate_path = layout.resolve_inside(candidate_stage.output_uri or "")
        cache_key = _extraction_cache_key(
            config, StageName.EVIDENCE, [parse_key, candidate_stage.cache_key], provider
        )
        output = layout.evidence_objects / f"{cache_key}.jsonl"
        if _cache_hit(
            layout,
            state,
            document.document_id,
            StageName.EVIDENCE,
            cache_key,
            output,
            force,
        ):
            cache_hits += 1
            continue
        state.start_stage(document.document_id, StageName.EVIDENCE, cache_key, candidate_stage.cache_key)
        try:
            candidates = list(read_jsonl(candidate_path, CandidateDesign))
            records = extract_evidence_records(document, candidates, provider)
            write_jsonl_atomic(output, records)
            state.finish_stage(
                document.document_id,
                StageName.EVIDENCE,
                output_uri=output.relative_to(layout.root).as_posix(),
                output_sha256=sha256_file(output),
                output_record_count=len(records),
            )
            processed += 1
            review_queued += _queue_evidence_reviews(layout, output, records)
        except Exception as exc:
            _fail(state, document.document_id, StageName.EVIDENCE, "evidence_extraction_failed", exc)
            failed += 1
            if config.runtime.fail_fast:
                raise
    return ExtractionResult(processed, cache_hits, failed, review_queued)


def extract_cards(
    config: PaperLibraryConfig,
    provider: StructuredExtractionProvider,
    *,
    force: bool = False,
    document_id_filter: str | None = None,
) -> ExtractionResult:
    layout, state = initialize(config)
    processed = cache_hits = failed = review_queued = 0
    for document, parse_key in _parsed_documents(layout, state, document_id_filter):
        evidence_stage = state.get_stage(document.document_id, StageName.EVIDENCE)
        if not _usable_stage(evidence_stage) or not _stage_artifact_is_valid(
            layout,
            state,
            evidence_stage,
        ):
            continue
        evidence_path = layout.resolve_inside(evidence_stage.output_uri or "")
        cache_key = _extraction_cache_key(
            config, StageName.CARDS, [parse_key, evidence_stage.cache_key], provider
        )
        output = layout.card_objects / f"{cache_key}.jsonl"
        if _cache_hit(
            layout,
            state,
            document.document_id,
            StageName.CARDS,
            cache_key,
            output,
            force,
        ):
            cache_hits += 1
            continue
        state.start_stage(document.document_id, StageName.CARDS, cache_key, evidence_stage.cache_key)
        try:
            records = list(read_jsonl(evidence_path, EvidenceRecord))
            cards = build_design_cards(document, records, provider)
            write_jsonl_atomic(output, cards)
            state.finish_stage(
                document.document_id,
                StageName.CARDS,
                output_uri=output.relative_to(layout.root).as_posix(),
                output_sha256=sha256_file(output),
                output_record_count=len(cards),
            )
            processed += 1
            review_queued += _queue_card_reviews(layout, output, cards)
        except Exception as exc:
            _fail(state, document.document_id, StageName.CARDS, "card_extraction_failed", exc)
            failed += 1
            if config.runtime.fail_fast:
                raise
    return ExtractionResult(processed, cache_hits, failed, review_queued)


def load_canonical_records(
    config: PaperLibraryConfig,
    *,
    document_id_filter: str | None = None,
    readonly: bool = False,
) -> CanonicalRecords:
    if readonly:
        layout = BuildLayout(config.paths.build_root)
        state = BuildState(layout.state_database)
        if not layout.state_database.is_file():
            return CanonicalRecords([], [], [], [])
    else:
        layout, state = initialize(config)
    documents: list[ParsedDocument] = []
    evidence_records: list[EvidenceRecord] = []
    cards: list[ResearchDesignCard] = []
    incomplete_document_ids: list[str] = []
    for document, parse_key in _parsed_documents(
        layout,
        state,
        document_id_filter,
        readonly=readonly,
    ):
        candidate_stage = state.get_stage(
            document.document_id,
            StageName.CANDIDATE_DESIGNS,
            readonly=readonly,
        )
        evidence_stage = state.get_stage(
            document.document_id,
            StageName.EVIDENCE,
            readonly=readonly,
        )
        card_stage = state.get_stage(
            document.document_id,
            StageName.CARDS,
            readonly=readonly,
        )
        lineage_is_current = bool(
            _usable_stage(candidate_stage)
            and _stage_artifact_is_valid(
                layout, state, candidate_stage, invalidate_invalid=not readonly
            )
            and candidate_stage.input_hash == parse_key
            and _usable_stage(evidence_stage)
            and _stage_artifact_is_valid(
                layout, state, evidence_stage, invalidate_invalid=not readonly
            )
            and evidence_stage.input_hash == candidate_stage.cache_key
            and _usable_stage(card_stage)
            and _stage_artifact_is_valid(
                layout, state, card_stage, invalidate_invalid=not readonly
            )
            and card_stage.input_hash == evidence_stage.cache_key
        )
        if not lineage_is_current:
            incomplete_document_ids.append(document.document_id)
            continue
        evidence_path = layout.resolve_inside(evidence_stage.output_uri or "")
        card_path = layout.resolve_inside(card_stage.output_uri or "")
        if not evidence_path.is_file() or not card_path.is_file():
            raise ValueError(f"Successful extraction artifact is missing for {document.document_id}")
        documents.append(document)
        evidence_records.extend(read_jsonl(evidence_path, EvidenceRecord))
        cards.extend(read_jsonl(card_path, ResearchDesignCard))
    return CanonicalRecords(
        documents,
        evidence_records,
        cards,
        incomplete_document_ids,
    )


def validate_records(
    config: PaperLibraryConfig,
    *,
    document_id_filter: str | None = None,
    write_report: bool = False,
) -> CorpusValidationResult:
    layout = BuildLayout(config.paths.build_root)
    records = load_canonical_records(
        config,
        document_id_filter=document_id_filter,
        readonly=not write_report,
    )
    result = validate_corpus(
        records.documents,
        records.evidence_records,
        records.cards,
        minimum_grounding_completeness=config.validation.minimum_grounding_completeness,
        require_exact_grounding=config.validation.require_exact_grounding,
    )
    issues = list(result.issues)
    for document_id in records.incomplete_document_ids:
        issues.append(
            ValidationIssue(
                code="incomplete_document",
                message="Document lacks current successful evidence or card output",
                severity="error",
                record_id=document_id,
            )
        )
    if not records.documents and not records.incomplete_document_ids:
        issues.append(
            ValidationIssue(
                code="empty_corpus",
                message="No current parsed documents are available for validation",
                severity="error",
            )
        )
    result = CorpusValidationResult(issues=tuple(issues))
    report = {
        "valid": result.valid,
        "document_count": len(records.documents),
        "evidence_count": len(records.evidence_records),
        "card_count": len(records.cards),
        "issues": [issue.model_dump(mode="json") for issue in result.issues],
    }
    if write_report and document_id_filter is None:
        write_json_atomic(layout.reports / "validation.json", report)
    return result


def publish_records(config: PaperLibraryConfig) -> PublishResult:
    records = load_canonical_records(config)
    if records.incomplete_document_ids:
        raise PaperLibraryError(
            "Cannot publish documents without current evidence and cards: "
            + ", ".join(records.incomplete_document_ids),
            code=ErrorCode.VALIDATION_FAILED,
        )
    if not records.documents:
        raise PaperLibraryError(
            "Cannot publish an empty corpus",
            code=ErrorCode.VALIDATION_FAILED,
        )
    return publish_corpus(
        config,
        records.documents,
        records.evidence_records,
        records.cards,
    )


def build_library(
    config: PaperLibraryConfig,
    provider: StructuredExtractionProvider,
    *,
    force: bool = False,
    document_id_filter: str | None = None,
) -> BuildResult:
    scan_result = scan(config)
    parse_result = parse_documents(
        config,
        force=force,
        document_id_filter=document_id_filter,
    )
    candidate_result = extract_candidates(
        config,
        provider=provider,
        force=force,
        document_id_filter=document_id_filter,
    )
    evidence_result = extract_evidence(
        config,
        provider,
        force=force,
        document_id_filter=document_id_filter,
    )
    card_result = extract_cards(
        config,
        provider,
        force=force,
        document_id_filter=document_id_filter,
    )
    failures = (
        parse_result.failed
        + candidate_result.failed
        + evidence_result.failed
        + card_result.failed
    )
    if failures:
        raise BuildStageError(
            f"Build stages failed for {failures} document-stage executions"
        )
    validation = validate_records(config, write_report=True)
    if not validation.valid:
        messages = "; ".join(
            issue.message for issue in validation.issues if issue.severity == "error"
        )
        raise PaperLibraryError(
            f"Corpus validation failed: {messages}",
            code=ErrorCode.VALIDATION_FAILED,
        )
    publication = publish_records(config)
    return BuildResult(
        scan_result,
        parse_result,
        candidate_result,
        evidence_result,
        card_result,
        validation,
        publication,
    )


def scan_result_json(result: ScanResult) -> dict[str, object]:
    return {
        "document_count": len(result.documents),
        "missing_count": result.missing,
        "documents": [
            {**asdict(document), "primary_path": str(document.primary_path)}
            for document in result.documents
        ],
    }


def _parsed_documents(
    layout: BuildLayout,
    state: BuildState,
    document_id_filter: str | None,
    *,
    readonly: bool = False,
) -> list[tuple[ParsedDocument, str]]:
    documents: list[tuple[ParsedDocument, str]] = []
    for stage in state.present_stage_records(StageName.PARSE, readonly=readonly):
        if document_id_filter and stage.document_id != document_id_filter:
            continue
        if not _usable_stage(stage) or not _stage_artifact_is_valid(
            layout,
            state,
            stage,
            invalidate_invalid=not readonly,
        ):
            continue
        path = layout.resolve_inside(stage.output_uri or "")
        try:
            document = ParsedDocument.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            if not readonly:
                state.invalidate_stage(stage.document_id, StageName.PARSE)
            continue
        if document.document_id != stage.document_id:
            if not readonly:
                state.invalidate_stage(stage.document_id, StageName.PARSE)
            continue
        documents.append((document, stage.cache_key))
    return documents


def _extraction_cache_key(
    config: PaperLibraryConfig,
    stage: StageName,
    upstream_keys: list[str],
    provider: StructuredExtractionProvider | None,
    strategy: str | None = None,
) -> str:
    return hash_canonical(
        {
            "stage": stage.value,
            "upstream": upstream_keys,
            "provider": provider.name if provider else "disabled",
            "model": provider.model if provider else None,
            "prompt_version": config.extraction.prompt_version,
            "temperature": config.extraction.temperature,
            "strategy": strategy,
        }
    )


def _cache_hit(
    layout: BuildLayout,
    state: BuildState,
    document_id: str,
    stage: StageName,
    cache_key: str,
    output: Path,
    force: bool,
) -> bool:
    if force:
        return False
    record = state.get_stage(document_id, stage)
    expected_uri = output.relative_to(layout.root).as_posix()
    if (
        not _usable_stage(record)
        or record.cache_key != cache_key
        or record.output_uri != expected_uri
    ):
        return False
    return _stage_artifact_is_valid(layout, state, record)


def _stage_artifact_is_valid(
    layout: BuildLayout,
    state: BuildState,
    stage: StageRecord,
    *,
    invalidate_invalid: bool = True,
) -> bool:
    valid = False
    try:
        path = layout.resolve_inside(stage.output_uri or "")
        valid = bool(
            stage.output_sha256
            and stage.output_record_count is not None
            and stage.output_record_count >= 0
            and path.is_file()
            and sha256_file(path) == stage.output_sha256
            and _artifact_record_count(path) == stage.output_record_count
        )
    except (OSError, ValueError):
        valid = False
    if not valid and invalidate_invalid:
        state.invalidate_stage(stage.document_id, stage.stage)
    return valid


def _artifact_record_count(path: Path) -> int:
    if path.suffix == ".json":
        return 1
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _usable_stage(stage: StageRecord | None) -> bool:
    return bool(
        stage
        and stage.status == StageStatus.SUCCEEDED
        and stage.output_uri
    )


def _queue_evidence_reviews(
    layout: BuildLayout,
    artifact_path: Path,
    records: list[EvidenceRecord],
) -> int:
    count = 0
    for record in records:
        if not record.quality.review_required:
            continue
        upstream_hash = hash_canonical(record.model_dump(mode="json", exclude={"created_at"}))
        append_review_record(
            layout.reviews / "evidence_candidates.jsonl",
            review_candidate_for(
                object_type=ReviewObjectType.EVIDENCE,
                record_id=record.evidence_id,
                document_id=record.document_id,
                candidate_uri=artifact_path.relative_to(layout.root).as_posix(),
                reasons=record.quality.warnings or ["evidence requires review"],
                upstream_hash=upstream_hash,
            ),
        )
        count += 1
    return count


def _queue_card_reviews(
    layout: BuildLayout,
    artifact_path: Path,
    cards: list[ResearchDesignCard],
) -> int:
    count = 0
    for card in cards:
        if not card.quality.review_required:
            continue
        upstream_hash = hash_canonical(card.model_dump(mode="json", exclude={"created_at"}))
        append_review_record(
            layout.reviews / "card_candidates.jsonl",
            review_candidate_for(
                object_type=ReviewObjectType.CARD,
                record_id=card.card_id,
                document_id=card.source_document_ids[0],
                candidate_uri=artifact_path.relative_to(layout.root).as_posix(),
                reasons=card.quality.warnings or ["design card requires review"],
                upstream_hash=upstream_hash,
            ),
        )
        count += 1
    return count


def _fail(
    state: BuildState,
    document_id: str,
    stage: StageName,
    error_code: str,
    exc: Exception,
) -> None:
    state.fail_stage(
        document_id,
        stage,
        error_code=error_code,
        error_message=f"{type(exc).__name__}: {exc}",
        retryable=isinstance(exc, PaperLibraryError) and exc.retryable,
    )
