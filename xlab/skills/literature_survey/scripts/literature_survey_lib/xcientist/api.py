from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from ..common import utc_now
from ..config import RuntimeConfig
from ..inputs import SurveyRequest
from ..survey_agent import AGENT_EXPORT_VERSION
from .config import load_survey_agent_config

AGENT_SCHEMA_VERSION = "xlab.literature_survey.agent.v1"


def generate_survey_from_xlab_context(
    *,
    run_dir: str | Path,
    request: SurveyRequest,
    runtime: RuntimeConfig,
    papers: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    graph_context: dict[str, Any],
) -> dict[str, Any]:
    """Generate an XLab SurveyAgent result using the integrated Xcientist engine."""
    if not provider_api_key_set():
        raise RuntimeError("OPENAI_API_KEY or LLM_API_KEY is required for SurveyAgent synthesis.")
    if not papers:
        raise RuntimeError("SurveyAgent synthesis requires at least one real paper.")

    run_dir_path = Path(run_dir).expanduser().resolve()
    graph_db_path = graph_db_path_from_context(graph_context)
    if graph_db_path is None:
        graph_db_path = materialize_graph_db_from_context(run_dir_path, papers, graph_context)

    from .engine import run_integrated_survey_engine

    config = load_survey_agent_config(
        run_dir=run_dir_path,
        topic=request.topic,
        graph_db=graph_db_path,
        overrides=runtime_overrides(runtime),
    )
    engine_result = run_integrated_survey_engine(
        config=config,
        request=request,
        runtime=runtime,
        papers=papers,
        clusters=clusters,
        graph_context=graph_context,
        graph_db_path=graph_db_path,
    )
    return normalize_agent_result(engine_result, runtime)


def provider_api_key_set() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("LLM_API_KEY", "").strip())


def runtime_overrides(runtime: RuntimeConfig) -> dict[str, Any]:
    """Map non-secret XLab runtime settings into SurveyAgent config overrides."""
    return {
        "APIInfo.chat_timeout": runtime.request_timeout_seconds,
        "ModuleInfo.SurveyGenerator.use_full_text_in_survey_generation": runtime.full_text_enabled,
    }


def graph_db_path_from_context(graph_context: dict[str, Any]) -> Path | None:
    artifacts = graph_context.get("artifacts") if isinstance(graph_context, dict) else None
    graph_db = artifacts.get("graph_db") if isinstance(artifacts, dict) else None
    if not graph_db:
        return None
    path = Path(str(graph_db)).expanduser().resolve()
    return path if path.exists() else None


def materialize_graph_db_from_context(run_dir: Path, papers: list[dict[str, Any]], graph_context: dict[str, Any]) -> Path:
    """Create a run-local PaperGraphRetriever-compatible graph DB from XLab evidence."""
    graph_db = run_dir / "state" / "xcientist" / "graph.db"
    if graph_db.exists():
        return graph_db
    graph_db.parent.mkdir(parents=True, exist_ok=True)

    method_graph = method_graph_from_context(graph_context)
    method_nodes = method_graph.get("nodes", []) if isinstance(method_graph.get("nodes"), list) else []
    method_edges = method_graph.get("edges", []) if isinstance(method_graph.get("edges"), list) else []
    graph_node_by_paper_id = graph_node_ids_by_paper_id(method_nodes)

    with sqlite3.connect(graph_db) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                node_type TEXT,
                paper_id TEXT,
                paper_title TEXT,
                pub_year INTEGER,
                source_venue TEXT,
                full_name TEXT,
                acronym TEXT,
                summary TEXT,
                keynote TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                source TEXT,
                target TEXT,
                edge_type TEXT,
                summary TEXT,
                keywords TEXT,
                insight TEXT
            )
            """
        )
        paper_by_id = {str(paper.get("id")): paper for paper in papers if paper.get("id") is not None}
        for paper_id, paper in paper_by_id.items():
            graph_id = graph_node_by_paper_id.get(paper_id, f"paper:{paper_id}")
            upsert_graph_paper_node(connection, graph_id, paper_id, paper)

        for node in method_nodes:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id") or "").strip()
            provenance = node.get("provenance") if isinstance(node.get("provenance"), dict) else {}
            metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
            paper_id = str(node.get("paper_id") or provenance.get("paper_id") or metadata.get("paper_id") or "").strip()
            if not node_id or not paper_id or paper_id not in paper_by_id:
                continue
            upsert_graph_paper_node(connection, node_id, paper_id, paper_by_id[paper_id], node_type=str(node.get("node_type") or "Core"))

        for edge in method_edges:
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source") or "").strip()
            target = str(edge.get("target") or "").strip()
            if not source or not target:
                continue
            relation = str(edge.get("relation") or edge.get("edge_type") or "related")
            connection.execute(
                "INSERT INTO edges (source, target, edge_type, summary, keywords, insight) VALUES (?, ?, ?, ?, ?, ?)",
                (source, target, relation, relation, "xlab-method-graph", "Imported from XLab method graph context."),
            )
        connection.commit()
    return graph_db


def method_graph_from_context(graph_context: dict[str, Any]) -> dict[str, Any]:
    artifacts = graph_context.get("artifacts") if isinstance(graph_context, dict) else None
    method_graph_path = artifacts.get("method_graph") if isinstance(artifacts, dict) else None
    if not method_graph_path:
        return {}
    path = Path(str(method_graph_path)).expanduser().resolve()
    if not path.exists() or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def graph_node_ids_by_paper_id(nodes: list[Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "").strip()
        provenance = node.get("provenance") if isinstance(node.get("provenance"), dict) else {}
        metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
        paper_id = str(node.get("paper_id") or provenance.get("paper_id") or metadata.get("paper_id") or "").strip()
        if node_id and paper_id and paper_id not in mapping and node.get("node_type") == "Paper":
            mapping[paper_id] = node_id
    return mapping


def upsert_graph_paper_node(
    connection: sqlite3.Connection,
    graph_id: str,
    paper_id: str,
    paper: dict[str, Any],
    *,
    node_type: str = "Core",
) -> None:
    title = str(paper.get("title") or paper_id).strip()
    abstract = str(paper.get("abstract") or title).strip()
    year = paper.get("year") if isinstance(paper.get("year"), int) else None
    venue = str(paper.get("venue") or "").strip()
    keynote = graph_keynote_from_paper(title=title, abstract=abstract, venue=venue)
    connection.execute(
        """
        INSERT OR REPLACE INTO nodes
            (id, node_type, paper_id, paper_title, pub_year, source_venue, full_name, acronym, summary, keynote)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (graph_id, node_type or "Core", paper_id, title, year, venue, title, "", abstract, keynote),
    )


def graph_keynote_from_paper(*, title: str, abstract: str, venue: str) -> str:
    summary = abstract or title
    domain = venue or "XLab literature survey"
    return (
        f"Paper Title: {title}\n"
        "Paper Type: Core\n"
        f"Domain: {domain}\n"
        f"Quote: {summary[:240]}\n"
        f"Summary: {summary}\n"
        "Keywords: literature survey; citation traceability; knowledge graph; scientific agents\n"
        "Insight: This paper contributes reusable evidence for graph-grounded survey synthesis and research planning."
    )


def normalize_agent_result(engine_result: dict[str, Any], runtime: RuntimeConfig) -> dict[str, Any]:
    return {
        "schema_version": AGENT_SCHEMA_VERSION,
        "export_version": AGENT_EXPORT_VERSION,
        "mode": "xcientist_survey_agent",
        "generated_at": str(engine_result.get("generated_at") or utc_now()),
        "llm": {
            "model": runtime.model,
            "base_url": runtime.llm_base_url,
            "context_window": runtime.llm_context_window,
        },
        "outline": engine_result.get("outline", {}),
        "paper_assignment": engine_result.get("paper_assignment", {}),
        "markdown_body": str(engine_result.get("markdown_body") or ""),
        "sections": engine_result.get("sections", []),
        "key_claims": engine_result.get("key_claims", []),
        "research_gaps": engine_result.get("research_gaps", []),
        "cited_paper_ids": engine_result.get("cited_paper_ids", []),
        "engine": {
            "schema_version": engine_result.get("schema_version"),
            "mode": engine_result.get("mode"),
            "counts": engine_result.get("counts", {}),
            "xcientist": engine_result.get("xcientist", {}),
        },
    }
