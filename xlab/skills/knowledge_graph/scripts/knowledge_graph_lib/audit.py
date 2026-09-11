from __future__ import annotations

import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from .common import (
    JsonObject,
    artifact_paths,
    as_list,
    as_mapping,
    atomic_write_json,
    number,
    read_json,
    read_jsonl,
    sha256_file,
    text,
    utc_now,
)
from .graph import EDGE_TYPES, NODE_TYPES


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / max(denominator, 1)


def _configured_ratio(request: JsonObject, key: str, default: float) -> float:
    value = number(request.get(key))
    return default if value is None else value


def _count(value: object) -> int:
    parsed = number(value)
    return -1 if parsed is None else int(parsed)


def _check_sha(
    *,
    paper_id: str,
    path_value: object,
    sha_value: object,
    label: str,
    blocking: list[str],
) -> None:
    path = Path(text(path_value))
    if not path.is_file():
        blocking.append(f"Parsed paper {paper_id} is missing {label}: {path}.")
    elif not text(sha_value) or sha256_file(path) != text(sha_value):
        blocking.append(f"Parsed paper {paper_id} has a stale {label} checksum.")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _path(value: object) -> Path | None:
    raw = text(value)
    return Path(raw) if raw else None


def _same_path(left: object, right: object) -> bool:
    left_path = _path(left)
    right_path = _path(right)
    return (
        left_path is not None
        and right_path is not None
        and left_path.resolve() == right_path.resolve()
    )


def _check_file_hash(
    *,
    path_value: object,
    sha_value: object,
    label: str,
    blocking: list[str],
) -> Path | None:
    path = _path(path_value)
    if path is None or not path.is_file():
        blocking.append(f"{label} is missing: {path_value or '<missing>'}.")
        return path
    actual = sha256_file(path)
    if text(sha_value) != actual:
        blocking.append(f"{label} has a stale checksum.")
    return path.resolve()


def _canonical_input_checks(
    request: JsonObject,
    documents: JsonObject,
    graph: JsonObject,
    document_records: list[JsonObject],
    blocking: list[str],
) -> JsonObject:
    input_directory = _path(documents.get("input_directory") or request.get("input_directory"))
    paper_set = _path(documents.get("paper_set_path") or request.get("input_path"))
    edge_path = _path(documents.get("paper_edges_path") or request.get("paper_edges_path"))
    prior_manifest = _path(
        documents.get("prior_manifest_path") or request.get("prior_manifest")
    )
    pdf_root = _path(documents.get("pdf_root") or request.get("pdf_root"))
    provenance: JsonObject = {
        "input_directory": str(input_directory) if input_directory else None,
        "paper_set_path": str(paper_set) if paper_set else None,
        "paper_set_sha256": documents.get("paper_set_sha256"),
        "paper_edges_path": str(edge_path) if edge_path else None,
        "paper_edges_sha256": documents.get("paper_edges_sha256"),
        "prior_manifest_path": str(prior_manifest) if prior_manifest else None,
        "prior_manifest_sha256": documents.get("prior_manifest_sha256"),
        "paper_collect_run_id": documents.get("paper_collect_run_id"),
        "paper_collect_skill_version": documents.get("paper_collect_skill_version"),
        "pdf_root": str(pdf_root) if pdf_root else None,
    }
    if input_directory is None or not input_directory.is_dir():
        blocking.append("Input directory from paper_documents.json is missing.")
        return provenance
    canonical_paper_set = (input_directory / "artifacts" / "papers.manifest.json").resolve()
    canonical_edges = (input_directory / "artifacts" / "metadata" / "edges.jsonl").resolve()
    canonical_pdf_root = (input_directory / "artifacts" / "pdfs").resolve()
    if paper_set is None or paper_set.resolve() != canonical_paper_set:
        blocking.append("paper_documents.json does not use the canonical paper_collect paper set.")
    if edge_path is None or edge_path.resolve() != canonical_edges:
        blocking.append("paper_documents.json does not use the canonical paper_collect edge file.")
    if pdf_root is None or pdf_root.resolve() != canonical_pdf_root:
        blocking.append("paper_documents.json does not use the canonical paper_collect PDF root.")
    if paper_set is not None:
        _check_file_hash(
            path_value=paper_set,
            sha_value=documents.get("paper_set_sha256"),
            label="paper_collect paper set",
            blocking=blocking,
        )
    if edge_path is not None:
        _check_file_hash(
            path_value=edge_path,
            sha_value=documents.get("paper_edges_sha256"),
            label="paper_collect edge file",
            blocking=blocking,
        )
    if prior_manifest is not None:
        _check_file_hash(
            path_value=prior_manifest,
            sha_value=documents.get("prior_manifest_sha256"),
            label="paper_collect run manifest",
            blocking=blocking,
        )
        prior = as_mapping(read_json(prior_manifest))
        if prior.get("schema_version") != "2":
            blocking.append("Prior paper_collect manifest has an unsupported schema version.")
        if text(prior.get("skill_name")) != "paper_collect":
            blocking.append("Prior manifest was not produced by paper_collect.")
        if text(prior.get("status")) != "success":
            blocking.append("Prior paper_collect manifest is not successful.")
        if as_mapping(prior.get("validation")).get("passed") is not True:
            blocking.append("Prior paper_collect validation did not pass.")
    else:
        blocking.append("paper_documents.json is missing prior paper_collect manifest provenance.")
    for key in ("paper_set_sha256", "paper_edges_sha256", "prior_manifest_sha256"):
        request_key = "input_sha256" if key == "paper_set_sha256" else key
        if text(request.get(request_key)) != text(documents.get(key)):
            blocking.append(f"request.json {request_key} does not match paper_documents.json {key}.")
    graph_provenance = as_mapping(graph.get("input_provenance"))
    for key, value in provenance.items():
        if key in graph_provenance and text(graph_provenance.get(key)) != text(value):
            blocking.append(f"method_graph input provenance {key} is inconsistent.")
    if text(graph.get("source_artifact_sha256")) != text(documents.get("paper_set_sha256")):
        blocking.append("method_graph source artifact checksum is inconsistent.")
    for document in document_records:
        paper_id = text(document.get("paper_id"))
        if not _same_path(document.get("source_manifest"), paper_set):
            blocking.append(f"Paper {paper_id} does not point to the canonical paper set.")
        if not _same_path(document.get("paper_collect_manifest"), prior_manifest):
            blocking.append(f"Paper {paper_id} does not point to the prior manifest.")
        if text(document.get("source_manifest_sha256")) != text(documents.get("paper_set_sha256")):
            blocking.append(f"Paper {paper_id} has stale paper set provenance.")
        if text(document.get("paper_collect_manifest_sha256")) != text(documents.get("prior_manifest_sha256")):
            blocking.append(f"Paper {paper_id} has stale prior-manifest provenance.")
        if text(document.get("paper_edges_sha256")) != text(documents.get("paper_edges_sha256")):
            blocking.append(f"Paper {paper_id} has stale edge-file provenance.")
        source_pdf = _path(document.get("source_pdf"))
        if source_pdf is None:
            continue
        if pdf_root is None or not _is_within(source_pdf, pdf_root):
            blocking.append(f"Paper {paper_id} PDF is outside the paper_collect PDF root.")
        actual_pdf_sha256 = sha256_file(source_pdf) if source_pdf.is_file() else ""
        if not actual_pdf_sha256:
            blocking.append(f"Paper {paper_id} source PDF is missing.")
        if text(document.get("source_sha256")) != actual_pdf_sha256:
            blocking.append(f"Paper {paper_id} source PDF checksum is stale.")
        if text(document.get("source_download_sha256")) != actual_pdf_sha256:
            blocking.append(f"Paper {paper_id} source PDF no longer matches paper_collect download metadata.")
    return provenance


def _database_checks(
    path: Path,
    *,
    nodes: int,
    edges: int,
    aliases: int,
    extractions: int,
    blocking: list[str],
) -> JsonObject:
    checks: JsonObject = {
        "integrity": None,
        "foreign_key_violations": None,
        "node_rows": None,
        "edge_rows": None,
        "alias_rows": None,
        "extraction_rows": None,
        "queryable": False,
        "wide_node_columns": False,
        "wide_edge_columns": False,
        "edge_type_matches_relation": False,
        "fts_queryable": False,
    }
    required_node_columns = {
        "full_name",
        "paper_title",
        "paper_domain",
        "paper_type",
        "summary",
        "keywords",
        "aliases",
        "citation_paperId",
        "citation_title",
        "tldr",
        "raw_json",
    }
    required_edge_columns = {
        "relation",
        "edge_type",
        "summary",
        "keywords",
        "metrics",
        "insight",
        "quote",
        "raw_json",
    }
    if path.is_file():
        try:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as database:
                integrity = database.execute("PRAGMA integrity_check").fetchone()
                foreign_keys = database.execute("PRAGMA foreign_key_check").fetchall()
                node_rows = database.execute("SELECT COUNT(*) FROM nodes").fetchone()
                edge_rows = database.execute("SELECT COUNT(*) FROM edges").fetchone()
                alias_rows = database.execute("SELECT COUNT(*) FROM aliases").fetchone()
                extraction_rows = database.execute(
                    "SELECT COUNT(*) FROM paper_extractions"
                ).fetchone()
                sample = database.execute("SELECT id FROM nodes LIMIT 1").fetchone()
                node_columns = {
                    row[1] for row in database.execute("PRAGMA table_info(nodes)")
                }
                edge_columns = {
                    row[1] for row in database.execute("PRAGMA table_info(edges)")
                }
                mismatch = database.execute(
                    "SELECT COUNT(*) FROM edges WHERE relation != edge_type"
                ).fetchone()
                try:
                    fts_source = database.execute(
                        "SELECT full_name FROM nodes WHERE full_name != '' LIMIT 1"
                    ).fetchone()
                    fts_terms = re.findall(r"[A-Za-z0-9]+", fts_source[0] if fts_source else "")
                    fts_query = fts_terms[0] if fts_terms else ""
                    fts_sample = (
                        database.execute(
                            "SELECT id FROM node_fts WHERE node_fts MATCH ? LIMIT 1",
                            (fts_query,),
                        ).fetchone()
                        if fts_query
                        else database.execute("SELECT id FROM node_fts LIMIT 1").fetchone()
                    )
                except sqlite3.Error:
                    fts_sample = None
                checks.update(
                    {
                        "integrity": integrity[0] if integrity else None,
                        "foreign_key_violations": len(foreign_keys),
                        "node_rows": node_rows[0] if node_rows else None,
                        "edge_rows": edge_rows[0] if edge_rows else None,
                        "alias_rows": alias_rows[0] if alias_rows else None,
                        "extraction_rows": extraction_rows[0] if extraction_rows else None,
                        "queryable": sample is not None,
                        "wide_node_columns": required_node_columns.issubset(node_columns),
                        "wide_edge_columns": required_edge_columns.issubset(edge_columns),
                        "edge_type_matches_relation": mismatch is not None and mismatch[0] == 0,
                        "fts_queryable": fts_sample is not None or nodes == 0,
                        "missing_node_columns": sorted(required_node_columns - node_columns),
                        "missing_edge_columns": sorted(required_edge_columns - edge_columns),
                    }
                )
        except sqlite3.Error as error:
            blocking.append(f"SQLite graph is not queryable: {error}.")
    expected = {
        "integrity": "ok",
        "foreign_key_violations": 0,
        "node_rows": nodes,
        "edge_rows": edges,
        "alias_rows": aliases,
        "extraction_rows": extractions,
        "wide_node_columns": True,
        "wide_edge_columns": True,
        "edge_type_matches_relation": True,
        "fts_queryable": True,
    }
    for key, value in expected.items():
        if checks.get(key) != value:
            blocking.append(
                f"SQLite {key.replace('_', ' ')} is {checks.get(key)!r}; expected {value!r}."
            )
    if nodes and checks.get("queryable") is not True:
        blocking.append("SQLite sample graph query returned no rows.")
    return checks


def audit_run(run_dir: Path) -> JsonObject:
    paths = artifact_paths(run_dir)
    request = as_mapping(read_json(paths["request"]))
    resources = as_mapping(read_json(paths["resource_plan"]))
    documents = as_mapping(read_json(paths["documents"]))
    mineru_manifest = as_mapping(read_json(paths["mineru_manifest"]))
    structures = as_mapping(read_json(paths["structure_manifest"]))
    extractions = as_mapping(read_json(paths["extraction_manifest"]))
    graph = as_mapping(read_json(paths["graph"]))
    nodes = [as_mapping(value) for value in read_jsonl(paths["nodes"])]
    edges = [as_mapping(value) for value in read_jsonl(paths["edges"])]
    aliases = [as_mapping(value) for value in read_jsonl(paths["aliases"])]
    blocking: list[str] = []
    warnings: list[str] = []

    required_files = [
        paths["request"],
        paths["resource_plan"],
        paths["documents"],
        paths["mineru_manifest"],
        paths["structure_manifest"],
        paths["extraction_manifest"],
        paths["nodes"],
        paths["edges"],
        paths["aliases"],
        paths["database"],
        paths["graph"],
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        blocking.append(f"Missing required graph artifacts: {', '.join(missing)}.")
    if request.get("schema_version") != "xlab.knowledge_graph_request.v2":
        blocking.append("request.json has an unsupported schema version.")
    if resources.get("schema_version") != "xlab.knowledge_graph_resources.v1":
        blocking.append("resource_plan.json has an unsupported schema version.")
    if documents.get("schema_version") != "xlab.paper_documents.v2":
        blocking.append("paper_documents.json has an unsupported schema version.")
    if mineru_manifest.get("schema_version") != "xlab.mineru_manifest.v1":
        blocking.append("mineru.manifest.json has an unsupported schema version.")
    if structures.get("schema_version") != "xlab.paper_structures.v1":
        blocking.append("paper_structures manifest has an unsupported schema version.")
    if extractions.get("schema_version") != "xlab.paper_extractions.v2":
        blocking.append("extractions manifest has an unsupported schema version.")

    document_records = [
        as_mapping(value) for value in as_list(documents.get("documents"))
    ]
    input_provenance = _canonical_input_checks(
        request, documents, graph, document_records, blocking
    )
    structure_records = [
        as_mapping(value) for value in as_list(structures.get("structures"))
    ]
    extraction_records = [
        as_mapping(value) for value in as_list(extractions.get("extractions"))
    ]
    total_documents = len(document_records)
    if any(document.get("status") == "pending" for document in document_records):
        blocking.append("PDF parsing still has pending papers.")
    if any(record.get("status") == "pending" for record in extraction_records):
        blocking.append("LLM extraction still has pending papers.")
    downloaded = sum(
        Path(text(document.get("source_pdf"))).is_file()
        for document in document_records
        if text(document.get("source_pdf"))
    )
    parsed = sum(document.get("status") == "parsed" for document in document_records)
    structured = sum(
        record.get("status") == "structured" for record in structure_records
    )
    extracted = sum(
        record.get("status") == "extracted" for record in extraction_records
    )
    summary = as_mapping(documents.get("summary"))
    if _count(summary.get("total")) != total_documents:
        blocking.append("Paper document total does not match its records.")
    if _count(summary.get("parsed")) != parsed:
        blocking.append("Paper document parsed count does not match its records.")
    paper_ids = [text(document.get("paper_id")) for document in document_records]
    nonempty_ids = [paper_id for paper_id in paper_ids if paper_id]
    if len(nonempty_ids) != len(set(nonempty_ids)):
        blocking.append("Input contains duplicate stable paper IDs.")
    if len(nonempty_ids) != total_documents:
        blocking.append("One or more input papers have no stable paper ID.")
    if total_documents < int(number(request.get("min_papers")) or 1):
        blocking.append(
            f"Only {total_documents} papers were supplied; minimum is "
            f"{int(number(request.get('min_papers')) or 1)}."
        )
    if downloaded == 0:
        blocking.append("The selected collection contains no downloaded PDFs.")
    if parsed == 0:
        blocking.append("MinerU did not parse any paper.")

    for document in document_records:
        if document.get("status") != "parsed":
            continue
        paper_id = text(document.get("paper_id"))
        if document.get("parser") != "mineru":
            blocking.append(f"Parsed paper {paper_id} was not produced by MinerU.")
        for path_key, sha_key, label in (
            ("markdown_path", "markdown_sha256", "Markdown"),
            ("content_list_path", "content_list_sha256", "content_list JSON"),
            ("middle_json_path", "middle_json_sha256", "middle/layout JSON"),
        ):
            _check_sha(
                paper_id=paper_id,
                path_value=document.get(path_key),
                sha_value=document.get(sha_key),
                label=label,
                blocking=blocking,
            )
    mineru_coverage = _ratio(parsed, downloaded)
    structure_success = _ratio(structured, parsed)
    extraction_success = _ratio(extracted, structured)
    thresholds = {
        "MinerU coverage": (
            mineru_coverage,
            _configured_ratio(request, "min_mineru_coverage", 0.9),
        ),
        "PaperGraph Step 1 success": (
            structure_success,
            _configured_ratio(request, "min_structure_success", 0.98),
        ),
        "PaperGraph Step 2 success": (
            extraction_success,
            _configured_ratio(request, "min_extraction_success", 0.9),
        ),
    }
    for label, (actual, minimum) in thresholds.items():
        if actual < minimum:
            blocking.append(f"{label} is {actual:.3f}; minimum is {minimum:.3f}.")

    structure_ids: list[str] = []
    for record in structure_records:
        if record.get("status") != "structured":
            continue
        paper_id = text(record.get("paper_id"))
        structure_ids.append(paper_id)
        structure_path = Path(text(record.get("path")))
        value = as_mapping(read_json(structure_path))
        if value.get("schema_version") != "xlab.paper_structure.v1":
            blocking.append(f"Paper {paper_id} has an invalid Step 1 checkpoint.")
        if not as_list(value.get("structure")):
            blocking.append(f"Paper {paper_id} has no Step 1 structure sections.")
        provenance = as_mapping(value.get("provenance"))
        if text(provenance.get("source_manifest_sha256")) != text(
            documents.get("paper_set_sha256")
        ):
            blocking.append(f"Paper {paper_id} Step 1 has stale paper-set provenance.")
        if text(provenance.get("paper_collect_manifest_sha256")) != text(
            documents.get("prior_manifest_sha256")
        ):
            blocking.append(f"Paper {paper_id} Step 1 has stale prior-run provenance.")
        if text(provenance.get("paper_edges_sha256")) != text(
            documents.get("paper_edges_sha256")
        ):
            blocking.append(f"Paper {paper_id} Step 1 has stale citation-edge provenance.")
        mineru_output = as_mapping(value.get("mineru_output"))
        for key in ("markdown", "content_list_json", "middle_json"):
            if not Path(text(mineru_output.get(key))).is_file():
                blocking.append(f"Paper {paper_id} Step 1 is missing MinerU {key}.")
    if len(structure_ids) != len(set(structure_ids)):
        blocking.append("Step 1 manifest contains duplicate paper IDs.")

    extraction_ids: list[str] = []
    for record in extraction_records:
        if record.get("status") != "extracted":
            continue
        paper_id = text(record.get("paper_id"))
        extraction_ids.append(paper_id)
        extraction_path = Path(text(record.get("path")))
        value = as_mapping(read_json(extraction_path))
        quality = as_mapping(value.get("quality"))
        ideation = as_mapping(value.get("ideation_resource"))
        provenance = as_mapping(value.get("provenance"))
        if value.get("schema_version") != "xlab.paper_extraction.v2":
            blocking.append(f"Paper {paper_id} has an invalid Step 2 checkpoint.")
        if quality.get("llm_validated") is not True or quality.get("grounded") is not True:
            blocking.append(f"Paper {paper_id} Step 2 did not pass LLM grounding checks.")
        if as_list(quality.get("passes")) != ["main", "graph", "calibration"]:
            blocking.append(f"Paper {paper_id} is missing a validated LLM pass.")
        if text(provenance.get("source_manifest_sha256")) != text(
            documents.get("paper_set_sha256")
        ):
            blocking.append(f"Paper {paper_id} Step 2 has stale paper-set provenance.")
        if text(provenance.get("paper_collect_manifest_sha256")) != text(
            documents.get("prior_manifest_sha256")
        ):
            blocking.append(f"Paper {paper_id} Step 2 has stale prior-run provenance.")
        if text(provenance.get("paper_edges_sha256")) != text(
            documents.get("paper_edges_sha256")
        ):
            blocking.append(f"Paper {paper_id} Step 2 has stale citation-edge provenance.")
        structure_path = Path(text(provenance.get("structure_path")))
        if not structure_path.is_file() or text(provenance.get("structure_sha256")) != sha256_file(
            structure_path
        ):
            blocking.append(f"Paper {paper_id} Step 2 has stale Step 1 provenance.")
        if not as_list(ideation.get("core_contributions")):
            blocking.append(f"Paper {paper_id} has no grounded Core contribution.")
    if len(extraction_ids) != len(set(extraction_ids)):
        blocking.append("Step 2 manifest contains duplicate paper IDs.")

    node_ids = [text(node.get("id")) for node in nodes]
    node_id_set = set(node_ids)
    edge_ids = [text(edge.get("id")) for edge in edges]
    if len(node_ids) != len(node_id_set):
        blocking.append("Graph contains duplicate node IDs.")
    if len(edge_ids) != len(set(edge_ids)):
        blocking.append("Graph contains duplicate edge IDs.")
    invalid_node_types = sorted(
        {
            text(node.get("node_type"))
            for node in nodes
            if text(node.get("node_type")) not in NODE_TYPES
        }
    )
    invalid_relations = sorted(
        {
            text(edge.get("relation"))
            for edge in edges
            if text(edge.get("relation")) not in EDGE_TYPES
        }
    )
    if invalid_node_types:
        blocking.append(f"Unsupported node types: {', '.join(invalid_node_types)}.")
    if invalid_relations:
        blocking.append(f"Unsupported edge relations: {', '.join(invalid_relations)}.")
    invalid_endpoints = [
        edge
        for edge in edges
        if text(edge.get("source")) not in node_id_set
        or text(edge.get("target")) not in node_id_set
    ]
    if invalid_endpoints:
        blocking.append(f"{len(invalid_endpoints)} edges have missing endpoints.")
    graph_nodes = [as_mapping(value) for value in as_list(graph.get("nodes"))]
    graph_edges = [as_mapping(value) for value in as_list(graph.get("edges"))]
    graph_aliases = [as_mapping(value) for value in as_list(graph.get("aliases"))]
    if graph.get("schema_version") != "xlab.method_graph.v2":
        blocking.append("method_graph.json has an unsupported schema version.")
    if graph_nodes != nodes or graph_edges != edges or graph_aliases != aliases:
        blocking.append("method_graph.json does not match its JSONL artifacts.")
    graph_counts = as_mapping(graph.get("counts"))
    if _count(graph_counts.get("nodes")) != len(nodes):
        blocking.append("method_graph.json node count is inconsistent.")
    if _count(graph_counts.get("edges")) != len(edges):
        blocking.append("method_graph.json edge count is inconsistent.")

    counts_by_type = Counter(text(node.get("node_type")) for node in nodes)
    counts_by_relation = Counter(text(edge.get("relation")) for edge in edges)
    min_nodes = int(number(request.get("min_nodes")) or 1)
    min_edges = int(number(request.get("min_edges")) or 1)
    if len(nodes) < min_nodes:
        blocking.append(f"Graph has {len(nodes)} nodes; minimum is {min_nodes}.")
    if len(edges) < min_edges:
        blocking.append(f"Graph has {len(edges)} edges; minimum is {min_edges}.")
    if counts_by_type["Core"] == 0:
        blocking.append("Graph has no Core nodes.")
    expected_papers = len(set(nonempty_ids))
    if counts_by_type["Paper"] != expected_papers:
        blocking.append(
            f"Graph contains {counts_by_type['Paper']} Paper nodes for "
            f"{expected_papers} unique input papers."
        )

    extracted_ids = set(extraction_ids)
    core_paper_ids = {
        text(node.get("paper_id"))
        for node in nodes
        if node.get("node_type") == "Core"
    }
    core_coverage = _ratio(len(core_paper_ids & extracted_ids), len(extracted_ids))
    min_core = _configured_ratio(request, "min_core_coverage", 0.9)
    if core_coverage < min_core:
        blocking.append(
            f"Core coverage is {core_coverage:.3f}; minimum is {min_core:.3f}."
        )
    semantic_edges = [
        edge
        for edge in edges
        if edge.get("relation")
        in {"introduces", "baseline_comparison", "evaluated_on", "core_relation"}
    ]
    grounded_edges = [
        edge
        for edge in semantic_edges
        if text(as_mapping(edge.get("evidence")).get("quote"))
    ]
    evidence_coverage = _ratio(len(grounded_edges), len(semantic_edges))
    min_evidence = _configured_ratio(request, "min_evidence_coverage", 0.95)
    if evidence_coverage < min_evidence:
        blocking.append(
            f"Semantic-edge quote coverage is {evidence_coverage:.3f}; "
            f"minimum is {min_evidence:.3f}."
        )

    alias_targets: dict[str, set[str]] = defaultdict(set)
    node_type_by_id = {
        text(node.get("id")): text(node.get("node_type")) for node in nodes
    }
    orphan_aliases = 0
    for alias in aliases:
        target = text(alias.get("target_id"))
        if target not in node_id_set:
            orphan_aliases += 1
        normalized = text(alias.get("normalized_alias"))
        if normalized:
            alias_targets[normalized].add(target)
    if orphan_aliases:
        blocking.append(f"{orphan_aliases} aliases point to missing nodes.")
    ambiguous = 0
    for targets in alias_targets.values():
        by_type: dict[str, set[str]] = defaultdict(set)
        for target in targets:
            by_type[node_type_by_id.get(target, "")].add(target)
        if any(len(values) > 1 for values in by_type.values()):
            ambiguous += 1
    ambiguous_ratio = _ratio(ambiguous, len(alias_targets))
    if ambiguous_ratio > _configured_ratio(
        request, "max_ambiguous_alias_ratio", 0.2
    ):
        warnings.append(f"Ambiguous alias ratio is {ambiguous_ratio:.3f}.")

    parsed_ids = {
        text(document.get("paper_id"))
        for document in document_records
        if document.get("status") == "parsed"
    }
    baseline_ids = {
        text(edge.get("evidence_paper_id"))
        for edge in edges
        if edge.get("relation") == "baseline_comparison"
    }
    dataset_ids = {
        text(edge.get("evidence_paper_id"))
        for edge in edges
        if edge.get("relation") == "evaluated_on"
    }
    baseline_coverage = _ratio(len(parsed_ids & baseline_ids), len(parsed_ids))
    dataset_coverage = _ratio(len(parsed_ids & dataset_ids), len(parsed_ids))
    if baseline_coverage < _configured_ratio(
        request, "desired_baseline_coverage", 0.2
    ):
        warnings.append(f"Baseline coverage is {baseline_coverage:.3f}.")
    if dataset_coverage < _configured_ratio(
        request, "desired_dataset_coverage", 0.2
    ):
        warnings.append(f"Dataset coverage is {dataset_coverage:.3f}.")

    database_checks = _database_checks(
        paths["database"],
        nodes=len(nodes),
        edges=len(edges),
        aliases=len(aliases),
        extractions=extracted,
        blocking=blocking,
    )
    passed = not blocking
    report: JsonObject = {
        "schema_version": "xlab.graph_report.v2",
        "generated_at": utc_now(),
        "passed": passed,
        "blocking_errors": blocking,
        "warnings": warnings,
        "counts": {
            "input_papers": total_documents,
            "downloaded_pdfs": downloaded,
            "parsed_papers": parsed,
            "structured_papers": structured,
            "extracted_papers": extracted,
            "nodes": len(nodes),
            "edges": len(edges),
            "aliases": len(aliases),
            "node_types": dict(counts_by_type),
            "edge_relations": dict(counts_by_relation),
        },
        "coverage": {
            "mineru": round(mineru_coverage, 6),
            "papergraph_step1": round(structure_success, 6),
            "papergraph_step2": round(extraction_success, 6),
            "core": round(core_coverage, 6),
            "baseline": round(baseline_coverage, 6),
            "dataset": round(dataset_coverage, 6),
            "semantic_edge_evidence": round(evidence_coverage, 6),
        },
        "deduplication": {
            "duplicate_node_ids": len(node_ids) - len(node_id_set),
            "duplicate_edge_ids": len(edge_ids) - len(set(edge_ids)),
            "ambiguous_aliases": ambiguous,
            "ambiguous_alias_ratio": round(ambiguous_ratio, 6),
        },
        "resources": resources,
        "input_provenance": input_provenance,
        "database": database_checks,
        "artifacts": {
            "resource_plan": str(paths["resource_plan"]),
            "mineru_manifest": str(paths["mineru_manifest"]),
            "paper_structures": str(paths["structure_manifest"]),
            "paper_extractions": str(paths["extraction_manifest"]),
            "method_graph": str(paths["graph"]),
            "database": str(paths["database"]),
            "nodes": str(paths["nodes"]),
            "edges": str(paths["edges"]),
            "aliases": str(paths["aliases"]),
        },
    }
    atomic_write_json(paths["report"], report)

    run_id = text(request.get("run_id")) or run_dir.name
    skill_version = text(request.get("skill_version")) or "3.0.0"
    base = f".xlab/runs/{run_id}/artifacts"
    manifest: JsonObject = {
        "schema_version": "2",
        "run_id": run_id,
        "skill_name": "knowledge_graph",
        "skill_version": skill_version,
        "status": "success" if passed else "incomplete",
        "created_at": utc_now(),
        "inputs": {
            "input_directory": request.get("input_directory"),
            "paper_set": request.get("input_path"),
            "paper_set_sha256": request.get("paper_set_sha256") or request.get("input_sha256"),
            "paper_edges": request.get("paper_edges_path"),
            "paper_edges_sha256": request.get("paper_edges_sha256"),
            "prior_manifest": request.get("prior_manifest"),
            "prior_manifest_sha256": request.get("prior_manifest_sha256"),
            "paper_collect_run_id": request.get("paper_collect_run_id"),
            "pdf_root": request.get("pdf_root"),
            "llm_model": as_mapping(request.get("llm")).get("model"),
            "mineru_backend": request.get("resolved_mineru_backend"),
        },
        "outputs": {
            "resource_plan": f"{base}/resource_plan.json",
            "mineru_manifest": f"{base}/mineru.manifest.json",
            "paper_structures": f"{base}/paper_structures.manifest.json",
            "paper_extractions": f"{base}/extractions.manifest.json",
            "method_graph": f"{base}/method_graph.json",
            "graph_db": f"{base}/graph.db",
            "graph_report": f"{base}/graph_report.json",
            "node_count": len(nodes),
            "edge_count": len(edges),
            "extracted_paper_count": extracted,
        },
        "validation": {
            "passed": passed,
            "blocking_errors": blocking,
            "warnings": warnings,
            "report": f"{base}/graph_report.json",
        },
        "artifacts": [
            {
                "type": "mineru_documents",
                "schema_version": "1",
                "path": f"{base}/mineru.manifest.json",
            },
            {
                "type": "paper_structures",
                "schema_version": "1",
                "path": f"{base}/paper_structures.manifest.json",
            },
            {
                "type": "paper_extractions",
                "schema_version": "2",
                "path": f"{base}/extractions.manifest.json",
            },
            {
                "type": "method_graph",
                "schema_version": "2",
                "path": f"{base}/method_graph.json",
            },
            {
                "type": "graph_db",
                "schema_version": "1",
                "path": f"{base}/graph.db",
            },
            {
                "type": "graph_report",
                "schema_version": "2",
                "path": f"{base}/graph_report.json",
            },
        ],
    }
    atomic_write_json(paths["manifest"], manifest)
    graph["audit"] = {
        "passed": passed,
        "report_path": str(paths["report"]),
        "blocking_error_count": len(blocking),
        "warning_count": len(warnings),
    }
    graph["input_provenance"] = input_provenance
    graph["source_artifact_sha256"] = documents.get("paper_set_sha256")
    atomic_write_json(paths["graph"], graph)
    return report
