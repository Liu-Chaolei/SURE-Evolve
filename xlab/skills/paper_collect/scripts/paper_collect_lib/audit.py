from __future__ import annotations

import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .common import (
    JsonObject,
    artifact_paths,
    as_list,
    as_mapping,
    atomic_write_json,
    integer,
    is_safe_relative_path,
    number,
    read_json,
    read_jsonl,
    run_lock,
    text,
    utc_now,
)
from .download import validate_pdf


def _check(
    checks: JsonObject,
    blocking_errors: list[str],
    name: str,
    passed: bool,
    message: str,
) -> None:
    checks[name] = {"passed": passed, "message": message}
    if not passed:
        blocking_errors.append(message)


def _provider_signature(record: Mapping[str, object]) -> str:
    operation = text(record.get("operation"))
    arguments = dict(as_mapping(record.get("input")))
    if operation in {
        "semantic_scholar.get_paper_citations",
        "semantic_scholar.get_paper_references",
        "semantic_scholar.get_recommendations",
    }:
        arguments.pop("fields", None)
    return f"{operation}\0{json.dumps(arguments, ensure_ascii=False, sort_keys=True)}"


def _provider_coverage(
    records: list[object],
) -> tuple[bool, bool, int, int, int]:
    latest: dict[str, Mapping[str, object]] = {}
    for raw_record in records:
        record = as_mapping(raw_record)
        if text(record.get("operation")):
            latest[_provider_signature(record)] = record
    successful_operations = [
        text(record.get("operation"))
        for record in latest.values()
        if record.get("is_error") is not True
    ]
    failed_calls = sum(
        record.get("is_error") is True for record in latest.values()
    )
    return (
        "tavily.search" in successful_operations,
        any(
            operation.startswith("semantic_scholar.")
            for operation in successful_operations
        ),
        len(successful_operations),
        len(latest),
        failed_calls,
    )


def _audit_run_unlocked(
    run_dir: Path,
    *,
    run_id: str,
    skill_name: str = "paper_collect",
    skill_version: str = "2.2.0",
) -> JsonObject:
    paths = artifact_paths(run_dir)
    request = as_mapping(read_json(paths["request"], {}))
    collection = as_mapping(read_json(paths["collection"], {}))
    papers = [
        as_mapping(item)
        for item in as_list(collection.get("papers"))
        if as_mapping(item)
    ]
    edges = [
        as_mapping(item)
        for item in read_jsonl(paths["edges_jsonl"])
        if as_mapping(item)
    ]
    provider_records = read_jsonl(paths["provider_results"])
    target_papers = integer(request.get("target_papers")) or 500
    minimum_usable_ratio = number(request.get("min_metadata_usable_ratio"))
    if minimum_usable_ratio is None:
        minimum_usable_ratio = 0.8
    minimum_core_ratio = number(request.get("min_core_ratio"))
    if minimum_core_ratio is None:
        minimum_core_ratio = 0.15
    minimum_abstract_coverage = number(request.get("min_abstract_coverage"))
    if minimum_abstract_coverage is None:
        minimum_abstract_coverage = 0.5
    minimum_graph_connected_ratio = number(
        request.get("min_graph_connected_ratio")
    )
    if minimum_graph_connected_ratio is None:
        minimum_graph_connected_ratio = 0.2

    checks: JsonObject = {}
    blockers: list[str] = []
    warnings: list[str] = []
    _check(
        checks,
        blockers,
        "collection_schema",
        collection.get("schema_version") == "xlab.paper_set.v3",
        "Paper collection must use schema xlab.paper_set.v3.",
    )
    if len(papers) < target_papers:
        warnings.append(
            f"Collected {len(papers)} papers; target is {target_papers}."
        )

    ids = [text(paper.get("id")) for paper in papers]
    dedupe_keys = [text(paper.get("dedupe_key")) for paper in papers]
    _check(
        checks,
        blockers,
        "required_identity",
        all(ids)
        and all(text(paper.get("title")) for paper in papers)
        and all(dedupe_keys),
        "Every selected paper must have id, title, and dedupe_key.",
    )
    _check(
        checks,
        blockers,
        "unique_identity",
        len(ids) == len(set(ids)) and len(dedupe_keys) == len(set(dedupe_keys)),
        "Selected papers contain duplicate ids or dedupe keys.",
    )

    required_metadata_fields = {
        "id",
        "title",
        "authors",
        "external_ids",
        "pdf_candidates",
        "discovery",
        "dedupe_key",
        "metadata_status",
        "relevance",
        "download",
    }
    incomplete_metadata = []
    for paper in papers:
        relevance = as_mapping(paper.get("relevance"))
        download = as_mapping(paper.get("download"))
        missing = sorted(required_metadata_fields - set(paper))
        if (
            missing
            or not isinstance(paper.get("authors"), list)
            or not isinstance(paper.get("external_ids"), Mapping)
            or not isinstance(paper.get("pdf_candidates"), list)
            or not as_list(paper.get("discovery"))
            or number(relevance.get("score")) is None
            or number(relevance.get("direct_score")) is None
            or text(download.get("status")) not in {
                "not_requested",
                "downloaded",
                "existing",
                "failed",
                "unavailable",
            }
        ):
            incomplete_metadata.append(text(paper.get("id")) or text(paper.get("title")))
    _check(
        checks,
        blockers,
        "paper_metadata_schema",
        not incomplete_metadata,
        "Selected papers must satisfy the paper_metadata@1 required fields.",
    )

    metadata_statuses = [text(paper.get("metadata_status")) for paper in papers]
    usable_count = sum(status in {"complete", "partial"} for status in metadata_statuses)
    usable_ratio = usable_count / max(1, len(papers))
    _check(
        checks,
        blockers,
        "metadata_usable_ratio",
        usable_ratio >= minimum_usable_ratio,
        (
            f"Usable metadata ratio is {usable_ratio:.3f}; "
            f"minimum is {minimum_usable_ratio:.3f}."
        ),
    )
    _check(
        checks,
        blockers,
        "no_invalid_metadata",
        all(status != "invalid" for status in metadata_statuses),
        "Selected papers include invalid metadata records.",
    )

    tiers = [
        text(as_mapping(paper.get("relevance")).get("tier"))
        for paper in papers
    ]
    core_count = sum(tier == "core" for tier in tiers)
    core_ratio = core_count / max(1, len(papers))
    _check(
        checks,
        blockers,
        "relevance_tiers",
        all(tier in {"core", "related"} for tier in tiers),
        "Selected papers must be classified as core or related.",
    )
    core_ratio_message = (
        f"Core-paper ratio is {core_ratio:.3f}; "
        f"minimum is {minimum_core_ratio:.3f}."
    )
    core_ratio_passed = core_ratio >= minimum_core_ratio
    checks["minimum_core_ratio"] = {
        "passed": core_ratio_passed,
        "message": core_ratio_message,
    }
    if not core_ratio_passed:
        warnings.append(core_ratio_message)
    unsupported_relevance = [
        paper
        for paper in papers
        if (number(as_mapping(paper.get("relevance")).get("lexical_score")) or 0.0)
        == 0
        and (
            integer(as_mapping(paper.get("relevance")).get("graph_support"))
            or 0
        )
        == 0
    ]
    _check(
        checks,
        blockers,
        "relevance_evidence",
        not unsupported_relevance,
        "Selected papers include records with no textual or graph relevance evidence.",
    )

    paper_ids = set(ids)
    allowed_edge_relations = {"cites", "recommended_with"}
    invalid_edges = [
        edge
        for edge in edges
        if edge.get("schema_version") != "xlab.paper_edge.v1"
        or text(edge.get("source_paper_id")) not in paper_ids
        or text(edge.get("target_paper_id")) not in paper_ids
        or text(edge.get("relation")) not in allowed_edge_relations
        or not text(edge.get("discovered_from"))
    ]
    _check(
        checks,
        blockers,
        "edge_references",
        not invalid_edges,
        "Citation edges must match paper_edge@1 and reference retained paper nodes.",
    )
    _check(
        checks,
        blockers,
        "graph_nonempty",
        bool(edges),
        "The graph-ready paper collection must contain at least one retained relation.",
    )
    connected_ids = {
        text(edge.get("source_paper_id")) for edge in edges
    } | {text(edge.get("target_paper_id")) for edge in edges}
    graph_connected_ratio = len(paper_ids & connected_ids) / max(1, len(papers))
    _check(
        checks,
        blockers,
        "minimum_graph_connected_ratio",
        graph_connected_ratio >= minimum_graph_connected_ratio,
        (
            f"Graph-connected paper ratio is {graph_connected_ratio:.3f}; "
            f"minimum is {minimum_graph_connected_ratio:.3f}."
        ),
    )

    (
        tavily_ok,
        semantic_scholar_ok,
        successful_api_calls,
        logical_api_calls,
        failed_api_calls,
    ) = _provider_coverage(provider_records)
    _check(
        checks,
        blockers,
        "tavily_seed_evidence",
        tavily_ok,
        "No successful MCP web-search seed evidence was captured.",
    )
    _check(
        checks,
        blockers,
        "semantic_scholar_evidence",
        semantic_scholar_ok,
        "No successful Semantic Scholar evidence was captured.",
    )
    search_state = as_mapping(collection.get("search"))
    stop_reason = text(search_state.get("stop_reason"))
    _check(
        checks,
        blockers,
        "search_converged",
        integer(search_state.get("followup_count")) == 0
        and bool(stop_reason),
        "Search still has pending Semantic Scholar follow-up actions.",
    )
    _check(
        checks,
        blockers,
        "provider_expansion_complete",
        stop_reason
        not in {
            "provider_expansion_failed",
            "graph_expansion_exhausted",
            "no_graph_evidence",
        },
        f"Provider graph expansion stopped with failure reason: {stop_reason or 'missing'}.",
    )
    if stop_reason == "call_budget_exhausted":
        warnings.append(
            "Semantic Scholar call budget was exhausted before the target count."
        )

    pending_downloads = []
    invalid_downloads = []
    downloaded_count = 0
    failed_downloads = 0
    unavailable_downloads = 0
    downloaded_papers: list[tuple[str, str, str]] = []
    unsafe_download_paths = []
    missing_download_hashes = []
    for paper in papers:
        download = as_mapping(paper.get("download"))
        status = text(download.get("status"))
        if as_list(paper.get("pdf_candidates")) and status in {"", "not_requested"}:
            pending_downloads.append(text(paper.get("id")))
        if status in {"downloaded", "existing"}:
            downloaded_count += 1
            relative_path = text(download.get("path"))
            expected_sha256 = text(download.get("sha256"))
            if not is_safe_relative_path(relative_path):
                unsafe_download_paths.append(text(paper.get("id")))
                continue
            resolved_path = (paths["pdfs"] / relative_path).resolve()
            try:
                resolved_path.relative_to(paths["pdfs"].resolve())
            except ValueError:
                unsafe_download_paths.append(text(paper.get("id")))
                continue
            if not expected_sha256:
                missing_download_hashes.append(text(paper.get("id")))
            downloaded_papers.append(
                (
                    text(paper.get("id")),
                    relative_path,
                    expected_sha256,
                )
            )
        elif status == "failed":
            failed_downloads += 1
        elif status == "unavailable":
            unavailable_downloads += 1
    audit_workers = max(
        1,
        min(integer(request.get("download_workers")) or 4, 4),
    )
    with ThreadPoolExecutor(max_workers=audit_workers) as executor:
        validations = {
            executor.submit(validate_pdf, paths["pdfs"] / relative_path): (
                paper_id,
                relative_path,
                expected_sha256,
            )
            for paper_id, relative_path, expected_sha256 in downloaded_papers
        }
        for future in as_completed(validations):
            paper_id, relative_path, expected_sha256 = validations[future]
            try:
                validation = future.result()
            except Exception as error:
                validation = {
                    "valid": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            actual_sha256 = text(validation.get("sha256"))
            if (
                validation.get("valid") is not True
                or not expected_sha256
                or not actual_sha256
                or expected_sha256 != actual_sha256
            ):
                invalid_downloads.append(
                    {
                        "paper_id": paper_id,
                        "path": relative_path,
                        "error": (
                            "sha256_missing_or_mismatch"
                            if validation.get("valid") is True
                            else validation.get("error")
                        ),
                    }
                )
    _check(
        checks,
        blockers,
        "download_attempts_complete",
        not pending_downloads,
        "Some papers with PDF candidates have not reached a terminal download status.",
    )
    _check(
        checks,
        blockers,
        "download_paths_run_local",
        not unsafe_download_paths,
        "Downloaded PDF paths must be relative paths under artifacts/pdfs.",
    )
    _check(
        checks,
        blockers,
        "download_hashes_present",
        not missing_download_hashes,
        "Downloaded PDFs must record sha256 hashes.",
    )
    _check(
        checks,
        blockers,
        "downloaded_pdfs_valid",
        not invalid_downloads,
        "One or more downloaded PDF files failed integrity validation.",
    )

    abstract_coverage = sum(bool(text(paper.get("abstract"))) for paper in papers) / max(
        1, len(papers)
    )
    if abstract_coverage < minimum_abstract_coverage:
        warnings.append(
            (
                f"Abstract coverage is {abstract_coverage:.3f}; "
                f"desired minimum is {minimum_abstract_coverage:.3f}."
            )
        )

    passed = not blockers
    report = {
        "schema_version": "xlab.paper_collection_report.v1",
        "passed": passed,
        "generated_at": utc_now(),
        "counts": {
            "papers": len(papers),
            "target_papers": target_papers,
            "core_papers": core_count,
            "related_papers": sum(tier == "related" for tier in tiers),
            "edges": len(edges),
            "successful_api_calls": successful_api_calls,
            "logical_api_calls": logical_api_calls,
            "failed_api_calls": failed_api_calls,
            "downloaded_pdfs": downloaded_count,
            "failed_downloads": failed_downloads,
            "unavailable_downloads": unavailable_downloads,
        },
        "metrics": {
            "metadata_usable_ratio": round(usable_ratio, 6),
            "abstract_coverage": round(abstract_coverage, 6),
            "core_ratio": round(core_ratio, 6),
            "graph_connected_ratio": round(graph_connected_ratio, 6),
        },
        "checks": checks,
        "blocking_errors": blockers,
        "warnings": warnings,
    }
    atomic_write_json(paths["report"], report)

    base = f".xlab/runs/{run_id}/artifacts"
    status = "success" if passed else "incomplete"
    manifest = {
        "schema_version": "2",
        "run_id": run_id,
        "skill_name": skill_name,
        "skill_version": skill_version,
        "status": status,
        "created_at": utc_now(),
        "inputs": {
            "query": request.get("query"),
            "facets": request.get("facets"),
            "target_papers": target_papers,
            "max_papers": request.get("max_papers"),
        },
        "outputs": {
            "paper_collection": f"{base}/papers.manifest.json",
            "papers": f"{base}/metadata/papers.jsonl",
            "edges": f"{base}/metadata/edges.jsonl",
            "pdf_dir": f"{base}/pdfs",
            "collection_report": f"{base}/collection_report.json",
            "collected_count": len(papers),
            "edge_count": len(edges),
            "downloaded_pdf_count": downloaded_count,
        },
        "validation": {
            "passed": passed,
            "blocking_errors": blockers,
            "warnings": warnings,
            "report": f"{base}/collection_report.json",
        },
        "artifacts": [
            {
                "type": "paper_set",
                "schema_version": "3",
                "path": f"{base}/papers.manifest.json",
            },
            {
                "type": "paper_edges",
                "schema_version": "1",
                "path": f"{base}/metadata/edges.jsonl",
            },
            {
                "type": "collection_report",
                "schema_version": "1",
                "path": f"{base}/collection_report.json",
            },
            {
                "type": "search_log",
                "schema_version": "1",
                "path": f"{base}/logs/provider_results.jsonl",
            },
        ],
    }
    atomic_write_json(paths["manifest"], manifest)
    return {
        "passed": passed,
        "status": status,
        "paper_count": len(papers),
        "edge_count": len(edges),
        "downloaded_pdf_count": downloaded_count,
        "blocking_errors": blockers,
        "warnings": warnings,
        "report_path": str(paths["report"]),
        "manifest_path": str(paths["manifest"]),
    }


def audit_run(
    run_dir: Path,
    *,
    run_id: str,
    skill_name: str = "paper_collect",
    skill_version: str = "2.2.0",
) -> JsonObject:
    with run_lock(run_dir, "audit"):
        return _audit_run_unlocked(
            run_dir,
            run_id=run_id,
            skill_name=skill_name,
            skill_version=skill_version,
        )
