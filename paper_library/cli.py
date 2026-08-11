from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from enum import IntEnum

from paper_library.application.lexical_index import (
    build_lexical_index,
    query_lexical_index,
)
from paper_library.application.pipeline import (
    build_library,
    extract_candidates,
    extract_cards,
    extract_evidence,
    initialize,
    parse_documents,
    publish_records,
    scan,
    scan_result_json,
    validate_records,
)
from paper_library.config import PaperLibraryConfig
from paper_library.errors import (
    BuildStageError,
    ConfigurationError,
    ErrorCode,
    PaperLibraryError,
)
from paper_library.extraction.providers import create_provider


class ExitCode(IntEnum):
    SUCCESS = 0
    PARTIAL_FAILURE = 2
    CONFIG_ERROR = 3
    VALIDATION_ERROR = 4
    OPTIONAL_DEPENDENCY_MISSING = 5
    INTERNAL_ERROR = 10


def _load_config(args: argparse.Namespace) -> PaperLibraryConfig:
    return PaperLibraryConfig.load(args.config)


def _cmd_validate_config(args: argparse.Namespace) -> int:
    config = _load_config(args)
    print(
        json.dumps(
            {
                "valid": True,
                "config": str(config.config_path),
                "config_hash": config.content_hash(),
                "source_root": str(config.paths.source_root),
                "build_root": str(config.paths.build_root),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return ExitCode.SUCCESS


def _cmd_scan(args: argparse.Namespace) -> int:
    result = scan(_load_config(args))
    print(json.dumps(scan_result_json(result), ensure_ascii=False, sort_keys=True))
    return ExitCode.SUCCESS


def _cmd_parse(args: argparse.Namespace) -> int:
    result = parse_documents(
        _load_config(args),
        force=args.force,
        document_id_filter=args.document_id,
    )
    print(json.dumps(result.__dict__, ensure_ascii=False, sort_keys=True))
    return ExitCode.PARTIAL_FAILURE if result.failed else ExitCode.SUCCESS


def _cmd_extract(args: argparse.Namespace) -> int:
    config = _load_config(args)
    if args.kind == "candidates":
        provider = None if config.extraction.provider == "disabled" else create_provider(config.extraction)
        result = extract_candidates(
            config,
            provider=provider,
            force=args.force,
            document_id_filter=args.document_id,
        )
    else:
        provider = create_provider(config.extraction)
        if config.extraction.provider == "disabled":
            raise PaperLibraryError(
                f"Extraction provider must be enabled for {args.kind}",
                code=ErrorCode.CONFIG_INVALID,
            )
        runner = extract_evidence if args.kind == "evidence" else extract_cards
        result = runner(
            config,
            provider,
            force=args.force,
            document_id_filter=args.document_id,
        )
    print(json.dumps(result.__dict__, ensure_ascii=False, sort_keys=True))
    return ExitCode.PARTIAL_FAILURE if result.failed else ExitCode.SUCCESS


def _cmd_validate(args: argparse.Namespace) -> int:
    result = validate_records(
        _load_config(args),
        document_id_filter=args.document_id,
    )
    print(
        json.dumps(
            {
                "valid": result.valid,
                "issues": [issue.model_dump(mode="json") for issue in result.issues],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return ExitCode.SUCCESS if result.valid else ExitCode.VALIDATION_ERROR


def _cmd_publish(args: argparse.Namespace) -> int:
    result = publish_records(_load_config(args))
    print(result.manifest.model_dump_json())
    return ExitCode.SUCCESS


def _cmd_build(args: argparse.Namespace) -> int:
    config = _load_config(args)
    if config.extraction.provider == "disabled":
        raise PaperLibraryError(
            "Extraction provider must be enabled for a complete build",
            code=ErrorCode.CONFIG_INVALID,
        )
    provider = create_provider(config.extraction)
    result = build_library(
        config,
        provider,
        force=args.force,
        document_id_filter=args.document_id,
    )
    print(
        json.dumps(
            {
                "scan": scan_result_json(result.scan),
                "parse": result.parse.__dict__,
                "candidates": result.candidates.__dict__,
                "evidence": result.evidence.__dict__,
                "cards": result.cards.__dict__,
                "valid": result.validation.valid,
                "corpus_version": result.publication.manifest.corpus_version,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return ExitCode.SUCCESS


def _cmd_status(args: argparse.Namespace) -> int:
    _, state = initialize(_load_config(args))
    print(json.dumps({"stages": state.counts()}, ensure_ascii=False, sort_keys=True))
    return ExitCode.SUCCESS


def _cmd_index(args: argparse.Namespace) -> int:
    result = build_lexical_index(
        _load_config(args),
        corpus_version=args.corpus_version,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "corpus_version": result.corpus_version,
                "index_path": str(result.index_path),
                "evidence_count": result.evidence_count,
                "design_count": result.design_count,
                "cache_hit": result.cache_hit,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return ExitCode.SUCCESS


def _cmd_query(args: argparse.Namespace) -> int:
    result = query_lexical_index(
        _load_config(args),
        args.text,
        kind=args.kind,
        corpus_version=args.corpus_version,
        limit=args.limit,
        paper_id=args.paper_id,
        record_type=args.record_type,
    )
    print(
        json.dumps(
            {
                "corpus_version": result.corpus_version,
                "kind": result.kind,
                "query": result.query,
                "hits": [
                    {
                        "record_id": hit.record_id,
                        "paper_id": hit.paper_id,
                        "record_type": hit.record_type,
                        "score": hit.score,
                        "record": hit.record.model_dump(mode="json"),
                    }
                    for hit in result.hits
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return ExitCode.SUCCESS


def _not_implemented(command: str) -> Callable[[argparse.Namespace], int]:
    def handler(args: argparse.Namespace) -> int:
        _load_config(args)
        raise PaperLibraryError(
            f"Command {command!r} is not available in the foundational release",
            code=ErrorCode.VALIDATION_FAILED,
        )

    return handler


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to paper-library YAML configuration")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper-library",
        description="Build and query a provenance-preserving ASR/TTS paper library.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_config = subparsers.add_parser("validate-config", help="Validate configuration")
    _add_config_argument(validate_config)
    validate_config.set_defaults(func=_cmd_validate_config)

    scan_command = subparsers.add_parser("scan", help="Scan source PDFs")
    _add_config_argument(scan_command)
    scan_command.set_defaults(func=_cmd_scan)

    parse_command = subparsers.add_parser("parse", help="Parse source PDFs")
    _add_config_argument(parse_command)
    parse_command.add_argument("--force", action="store_true")
    parse_command.add_argument("--document-id")
    parse_command.set_defaults(func=_cmd_parse)

    status_command = subparsers.add_parser("status", help="Show build state")
    _add_config_argument(status_command)
    status_command.set_defaults(func=_cmd_status)

    build_command = subparsers.add_parser("build", help="Run the complete build pipeline")
    _add_config_argument(build_command)
    build_command.add_argument("--force", action="store_true")
    build_command.add_argument("--document-id")
    build_command.set_defaults(func=_cmd_build)

    validate_command = subparsers.add_parser("validate", help="Validate canonical records")
    _add_config_argument(validate_command)
    validate_command.add_argument("--document-id")
    validate_command.set_defaults(func=_cmd_validate)

    publish_command = subparsers.add_parser("publish", help="Publish validated records")
    _add_config_argument(publish_command)
    publish_command.set_defaults(func=_cmd_publish)

    index_command = subparsers.add_parser("index", help="Build retrieval indexes")
    _add_config_argument(index_command)
    index_command.add_argument("--corpus-version")
    index_command.add_argument("--force", action="store_true")
    index_command.set_defaults(func=_cmd_index)

    query_command = subparsers.add_parser("query", help="Query a published index")
    _add_config_argument(query_command)
    query_command.add_argument("text")
    query_command.add_argument("--kind", required=True, choices=("evidence", "design"))
    query_command.add_argument("--corpus-version")
    query_command.add_argument("--limit", type=int, default=10)
    query_command.add_argument("--paper-id")
    query_command.add_argument("--record-type")
    query_command.set_defaults(func=_cmd_query)

    retry_command = subparsers.add_parser("retry", help="Retry failed stages")
    _add_config_argument(retry_command)
    retry_command.set_defaults(func=_not_implemented("retry"))

    extract = subparsers.add_parser("extract", help="Run an extraction stage")
    _add_config_argument(extract)
    extract.add_argument("kind", choices=("candidates", "evidence", "cards"))
    extract.add_argument("--force", action="store_true")
    extract.add_argument("--document-id")
    extract.set_defaults(func=_cmd_extract)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code = int(args.func(args))
    except ConfigurationError as exc:
        print(f"ERROR [{exc.code.value}]: {exc}", file=sys.stderr)
        code = ExitCode.CONFIG_ERROR
    except BuildStageError as exc:
        print(f"ERROR [{exc.code.value}]: {exc}", file=sys.stderr)
        code = ExitCode.PARTIAL_FAILURE
    except PaperLibraryError as exc:
        print(f"ERROR [{exc.code.value}]: {exc}", file=sys.stderr)
        if exc.code == ErrorCode.CONFIG_INVALID:
            code = ExitCode.CONFIG_ERROR
        elif exc.code == ErrorCode.OPTIONAL_DEPENDENCY_MISSING:
            code = ExitCode.OPTIONAL_DEPENDENCY_MISSING
        else:
            code = ExitCode.VALIDATION_ERROR
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR [{type(exc).__name__}]: {exc}", file=sys.stderr)
        code = ExitCode.INTERNAL_ERROR
    raise SystemExit(code)
