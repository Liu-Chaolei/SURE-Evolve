#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

from paper_collect_lib.audit import audit_run
from paper_collect_lib.common import (
    artifact_paths,
    atomic_write_json,
    atomic_write_jsonl,
    normalize_space,
    run_lock,
    text,
    utc_now,
)
from paper_collect_lib.download import download_run
from paper_collect_lib.metadata import ingest_run
from paper_collect_lib.pipeline import collect_run
from paper_collect_lib.providers import (
    PAPER_FIELDS,
    S2_REFERENCES,
    S2_SEARCH,
    TAVILY_SEARCH,
    ProviderRequestError,
)


SKILL_NAME = "paper_collect"
SKILL_VERSION = "2.2.0"
ACADEMIC_DOMAINS = [
    "arxiv.org",
    "openreview.net",
    "aclanthology.org",
    "proceedings.mlr.press",
    "semanticscholar.org",
]


def _run_dir(value: str) -> Path:
    run_dir = Path(value).expanduser().resolve()
    expected = text(os.environ.get("XLAB_RUN_DIR"))
    if expected and run_dir != Path(expected).expanduser().resolve():
        raise ValueError("--run-dir must match the active XLab run directory.")
    return run_dir


def _run_id(args: argparse.Namespace, run_dir: Path) -> str:
    return text(getattr(args, "run_id", "")) or run_dir.name


def _query_entry(
    *,
    query_id: str,
    provider: str,
    purpose: str,
    operation: str,
    arguments: dict[str, object],
) -> dict[str, object]:
    return {
        "id": query_id,
        "provider": provider,
        "purpose": purpose,
        "operation": operation,
        "arguments": arguments,
    }


def _query_plan(query: str, facets: list[str]) -> dict[str, object]:
    current_year = datetime.now(timezone.utc).year
    entries: list[dict[str, object]] = [
        _query_entry(
            query_id="tavily-survey",
            provider="tavily",
            purpose="survey",
            operation=TAVILY_SEARCH,
            arguments={
                "query": f"{query} survey review research papers",
                "search_depth": "advanced",
                "max_results": 20,
                "include_domains": ACADEMIC_DOMAINS,
                "include_raw_content": False,
            },
        ),
        _query_entry(
            query_id="tavily-seminal",
            provider="tavily",
            purpose="seminal",
            operation=TAVILY_SEARCH,
            arguments={
                "query": f"{query} seminal foundational influential papers",
                "search_depth": "advanced",
                "max_results": 20,
                "include_domains": ACADEMIC_DOMAINS,
                "include_raw_content": False,
            },
        ),
        _query_entry(
            query_id="tavily-recent",
            provider="tavily",
            purpose="recent",
            operation=TAVILY_SEARCH,
            arguments={
                "query": f"{query} recent advances {current_year - 2} {current_year}",
                "search_depth": "advanced",
                "max_results": 20,
                "include_domains": ACADEMIC_DOMAINS,
                "include_raw_content": False,
                "start_date": f"{current_year - 2}-01-01",
            },
        ),
        _query_entry(
            query_id="s2-topic",
            provider="semantic_scholar",
            purpose="topic",
            operation=S2_SEARCH,
            arguments={"query": query, "limit": 100, "fields": PAPER_FIELDS},
        ),
        _query_entry(
            query_id="s2-recent",
            provider="semantic_scholar",
            purpose="recent",
            operation=S2_SEARCH,
            arguments={
                "query": query,
                "limit": 100,
                "year": f"{current_year - 2}-{current_year}",
                "fields": PAPER_FIELDS,
            },
        ),
    ]
    for index, facet in enumerate(facets, 1):
        facet_query = f"{query} {facet}"
        entries.extend(
            [
                _query_entry(
                    query_id=f"mcp-facet-{index}",
                    provider="web_search_mcp",
                    purpose="facet",
                    operation=TAVILY_SEARCH,
                    arguments={
                        "query": f"{facet_query} academic papers",
                        "search_depth": "advanced",
                        "max_results": 20,
                        "include_domains": ACADEMIC_DOMAINS,
                        "include_raw_content": False,
                    },
                ),
                _query_entry(
                    query_id=f"s2-facet-{index}",
                    provider="semantic_scholar",
                    purpose="facet",
                    operation=S2_SEARCH,
                    arguments={
                        "query": facet_query,
                        "limit": 100,
                        "fields": PAPER_FIELDS,
                    },
                ),
            ]
        )
    return {
        "schema_version": "xlab.paper_query_plan.v1",
        "strategy": "mcp-web-search-seeds-then-semantic-scholar-expansion",
        "generated_at": utc_now(),
        "queries": entries,
    }


def _parse_invocation_arguments(value: str) -> dict[str, object]:
    source = value.strip()
    if not source or source[0] not in {'"', "'"}:
        raise ValueError(
            'The topic must be the first argument and enclosed in quotes, for example: '
            '"/xlab collect-papers \\"graph neural networks\\" --target-papers 500".'
        )
    try:
        tokens = shlex.split(source, posix=True)
    except ValueError as error:
        raise ValueError(f"Cannot parse paper_collect arguments: {error}") from error
    if not tokens or not normalize_space(tokens[0]):
        raise ValueError("The quoted topic must not be empty.")

    parser = argparse.ArgumentParser(prog="/xlab collect-papers", add_help=False)
    parser.add_argument("--facet", action="append", default=[])
    parser.add_argument("--target-papers", type=int)
    parser.add_argument("--max-papers", type=int)
    parser.add_argument("--download-workers", type=int)
    parser.add_argument("--download-per-host", type=int)
    try:
        options, unknown = parser.parse_known_args(tokens[1:])
    except SystemExit as error:
        raise ValueError("Invalid paper_collect option value.") from error
    if unknown:
        raise ValueError(f"Unknown paper_collect arguments: {' '.join(unknown)}")
    return {
        "query": tokens[0],
        "facets": options.facet,
        "target_papers": options.target_papers,
        "max_papers": options.max_papers,
        "download_workers": options.download_workers,
        "download_per_host": options.download_per_host,
    }


def _init_run_unlocked(args: argparse.Namespace) -> dict[str, object]:
    run_dir = _run_dir(args.run_dir)
    paths = artifact_paths(run_dir)
    invocation_arguments = text(getattr(args, "arguments", ""))
    if invocation_arguments:
        parsed = _parse_invocation_arguments(invocation_arguments)
        args.query = parsed["query"]
        args.facet = parsed["facets"]
        for name in (
            "target_papers",
            "max_papers",
            "download_workers",
            "download_per_host",
        ):
            value = parsed[name]
            if value is not None:
                setattr(args, name, value)
    query = normalize_space(args.query)
    if not query:
        raise ValueError("--query or --arguments must provide a non-empty topic.")
    facets = []
    seen: set[str] = set()
    for raw_facet in args.facet:
        facet = normalize_space(raw_facet)
        if not facet or facet.lower() in seen:
            continue
        seen.add(facet.lower())
        facets.append(facet)
    if len(facets) > 8:
        raise ValueError("At most eight --facet values are allowed.")
    if args.target_papers < 1:
        raise ValueError("--target-papers must be at least 1.")
    if args.max_papers is not None and args.max_papers < 1:
        raise ValueError("--max-papers must be at least 1.")
    if not 0 <= args.min_graph_connected_ratio <= 1:
        raise ValueError("--min-graph-connected-ratio must be between 0 and 1.")
    target_papers = args.target_papers
    max_papers = (
        args.max_papers
        if args.max_papers is not None
        else max(target_papers * 3, 1500)
    )
    if target_papers > max_papers:
        raise ValueError("--target-papers cannot exceed --max-papers")
    if not 1 <= args.download_workers <= 16:
        raise ValueError("--download-workers must be between 1 and 16.")
    if not 1 <= args.download_per_host <= args.download_workers:
        raise ValueError(
            "--download-per-host must be between 1 and --download-workers."
        )
    if args.download_timeout <= 0:
        raise ValueError("--download-timeout must be greater than zero.")
    if args.download_retries < 0:
        raise ValueError("--download-retries must not be negative.")
    if args.max_pdf_bytes < 1024:
        raise ValueError("--max-pdf-bytes must be at least 1024.")
    request = {
        "schema_version": "xlab.paper_collection_request.v2",
        "query": query,
        "facets": facets,
        "invocation_arguments": invocation_arguments or None,
        "target_papers": target_papers,
        "max_papers": max_papers,
        "seed_limit": args.seed_limit,
        "relation_limit_per_seed": args.relation_limit,
        "recommendation_limit": args.recommendation_limit,
        "max_semantic_scholar_calls": args.max_s2_calls,
        "max_enrichment_calls": args.max_enrichment_calls,
        "novelty_stop_ratio": args.novelty_stop_ratio,
        "min_metadata_usable_ratio": args.min_metadata_usable_ratio,
        "min_core_ratio": args.min_core_ratio,
        "min_abstract_coverage": args.min_abstract_coverage,
        "min_graph_connected_ratio": args.min_graph_connected_ratio,
        "download_workers": args.download_workers,
        "download_per_host": args.download_per_host,
        "download_timeout_seconds": args.download_timeout,
        "download_retries": args.download_retries,
        "max_pdf_bytes": args.max_pdf_bytes,
        "created_at": utc_now(),
    }
    for directory in (
        paths["artifacts"],
        paths["metadata"],
        paths["pdfs"],
        paths["logs"],
    ):
        directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths["request"], request)
    plan = _query_plan(query, facets)
    atomic_write_json(paths["query_plan"], plan)
    for path in (
        paths["provider_results"],
        paths["downloads"],
        paths["download_results"],
        paths["failures"],
        paths["papers_jsonl"],
        paths["edges_jsonl"],
    ):
        if not path.exists():
            atomic_write_jsonl(path, [])
    return {
        "run_id": _run_id(args, run_dir),
        "query": query,
        "facet_count": len(facets),
        "target_papers": target_papers,
        "max_papers": max_papers,
        "initial_query_count": len(plan["queries"]),
        "request_path": str(paths["request"]),
        "query_plan_path": str(paths["query_plan"]),
    }


def init_run(args: argparse.Namespace) -> dict[str, object]:
    run_dir = _run_dir(args.run_dir)
    with run_lock(run_dir, "init"):
        return _init_run_unlocked(args)


def _provider_record(
    provider: str,
    operation: str,
    input_value: dict[str, object],
    payload: object,
) -> dict[str, object]:
    return {
        "captured_at": utc_now(),
        "provider": provider,
        "operation": operation,
        "input": input_value,
        "is_error": False,
        "http_status": 200,
        "attempts": 1,
        "request_id": f"smoke-{operation.rsplit('.', 1)[-1]}",
        "payload": payload,
    }


def smoke_run(args: argparse.Namespace) -> dict[str, object]:
    run_dir = _run_dir(args.run_dir)
    init_args = argparse.Namespace(
        run_dir=str(run_dir),
        run_id=args.run_id,
        query="graph neural networks",
        facet=["semi-supervised node classification"],
        target_papers=2,
        max_papers=10,
        seed_limit=4,
        relation_limit=20,
        recommendation_limit=10,
        max_s2_calls=20,
        max_enrichment_calls=5,
        novelty_stop_ratio=0.05,
        min_metadata_usable_ratio=0.8,
        min_core_ratio=0.15,
        min_abstract_coverage=0.5,
        min_graph_connected_ratio=0.2,
        download_workers=4,
        download_per_host=2,
        download_timeout=5.0,
        download_retries=1,
        max_pdf_bytes=10 * 1024 * 1024,
    )
    init_run(init_args)
    paths = artifact_paths(run_dir)
    records = [
        _provider_record(
            "tavily",
            TAVILY_SEARCH,
            {
                "query": "graph neural networks survey review research papers",
                "max_results": 20,
            },
            {
                "results": [
                    {
                        "title": "Graph Neural Networks: A Review of Methods and Applications",
                        "url": "https://doi.org/10.1000/gnn.review",
                        "content": "A review of graph neural network methods.",
                        "score": 0.93,
                    }
                ]
            },
        ),
        _provider_record(
            "semantic_scholar",
            S2_SEARCH,
            {"query": "graph neural networks", "limit": 100},
            {
                "total": 2,
                "data": [
                    {
                        "paperId": "S2-GNN-REVIEW",
                        "title": "Graph Neural Networks: A Review of Methods and Applications",
                        "authors": [{"authorId": "A1", "name": "Ada Researcher"}],
                        "year": 2021,
                        "abstract": "This review organizes graph neural network methods and applications.",
                        "venue": "Journal of Graph Learning",
                        "citationCount": 120,
                        "referenceCount": 80,
                        "externalIds": {"DOI": "10.1000/gnn.review"},
                    },
                    {
                        "paperId": "S2-GCN",
                        "title": "Semi-Supervised Classification with Graph Convolutional Networks",
                        "authors": [{"authorId": "A2", "name": "Grace Scientist"}],
                        "year": 2017,
                        "abstract": "We study semi-supervised node classification with graph convolutional networks.",
                        "venue": "Learning Representations",
                        "citationCount": 5000,
                        "referenceCount": 20,
                        "externalIds": {"DOI": "10.1000/gcn"},
                    },
                ],
            },
        ),
        _provider_record(
            "semantic_scholar",
            S2_REFERENCES,
            {"paperId": "S2-GNN-REVIEW", "limit": 20, "offset": 0},
            {
                "data": [
                    {
                        "citedPaper": {
                            "paperId": "S2-GCN",
                            "title": "Semi-Supervised Classification with Graph Convolutional Networks",
                            "authors": [{"authorId": "A2", "name": "Grace Scientist"}],
                            "year": 2017,
                            "abstract": "We study semi-supervised node classification with graph convolutional networks.",
                            "externalIds": {"DOI": "10.1000/gcn"},
                        }
                    }
                ]
            },
        ),
    ]
    atomic_write_jsonl(paths["provider_results"], records)
    ingest = ingest_run(run_dir)
    download = download_run(run_dir)
    audit = audit_run(
        run_dir,
        run_id=_run_id(args, run_dir),
        skill_name=SKILL_NAME,
        skill_version=SKILL_VERSION,
    )
    return {"ingest": ingest, "download": download, "audit": audit}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a broad, graph-ready paper collection with MCP web search and "
            "Semantic Scholar API clients."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--run-dir", required=True)
    init_parser.add_argument("--run-id")
    query_source = init_parser.add_mutually_exclusive_group(required=True)
    query_source.add_argument("--query")
    query_source.add_argument("--arguments")
    init_parser.add_argument("--facet", action="append", default=[])
    init_parser.add_argument("--target-papers", type=int, default=500)
    init_parser.add_argument("--max-papers", type=int)
    init_parser.add_argument("--seed-limit", type=int, default=40)
    init_parser.add_argument("--relation-limit", type=int, default=200)
    init_parser.add_argument("--recommendation-limit", type=int, default=100)
    init_parser.add_argument("--max-s2-calls", type=int, default=240)
    init_parser.add_argument("--max-enrichment-calls", type=int, default=80)
    init_parser.add_argument("--novelty-stop-ratio", type=float, default=0.05)
    init_parser.add_argument("--min-metadata-usable-ratio", type=float, default=0.8)
    init_parser.add_argument("--min-core-ratio", type=float, default=0.15)
    init_parser.add_argument("--min-abstract-coverage", type=float, default=0.5)
    init_parser.add_argument(
        "--min-graph-connected-ratio",
        type=float,
        default=0.2,
    )
    init_parser.add_argument("--download-workers", type=int, default=8)
    init_parser.add_argument("--download-per-host", type=int, default=2)
    init_parser.add_argument("--download-timeout", type=float, default=60.0)
    init_parser.add_argument("--download-retries", type=int, default=3)
    init_parser.add_argument(
        "--max-pdf-bytes",
        type=int,
        default=100 * 1024 * 1024,
    )

    for name in ("collect", "ingest", "download", "audit", "run", "resume", "smoke"):
        command_parser = subparsers.add_parser(name)
        command_parser.add_argument("--run-dir", required=True)
        command_parser.add_argument("--run-id")
    download_parser = subparsers.choices["download"]
    download_parser.add_argument("--workers", type=int)
    download_parser.add_argument("--per-host", type=int)
    download_parser.add_argument("--timeout", type=float)
    download_parser.add_argument("--max-bytes", type=int)
    download_parser.add_argument("--retries", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir = _run_dir(args.run_dir)
    if args.command == "init":
        result = init_run(args)
    elif args.command in {"collect", "resume"}:
        result = collect_run(run_dir)
    elif args.command == "ingest":
        result = ingest_run(run_dir)
    elif args.command == "download":
        result = download_run(
            run_dir,
            workers=args.workers,
            per_host_workers=args.per_host,
            timeout_seconds=args.timeout,
            max_bytes=args.max_bytes,
            retries=args.retries,
        )
    elif args.command == "audit":
        result = audit_run(
            run_dir,
            run_id=_run_id(args, run_dir),
            skill_name=SKILL_NAME,
            skill_version=SKILL_VERSION,
        )
    elif args.command == "run":
        collection = collect_run(run_dir)
        download = download_run(run_dir)
        audit = audit_run(
            run_dir,
            run_id=_run_id(args, run_dir),
            skill_name=SKILL_NAME,
            skill_version=SKILL_VERSION,
        )
        result = {
            "collection": collection,
            "download": download,
            "audit": audit,
        }
    else:
        result = smoke_run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if args.command == "smoke" and result["audit"].get("passed") is not True:
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
        ProviderRequestError,
    ) as error:
        print(
            json.dumps(
                {
                    "error": f"{type(error).__name__}: {error}",
                    "status": "failed",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from error
