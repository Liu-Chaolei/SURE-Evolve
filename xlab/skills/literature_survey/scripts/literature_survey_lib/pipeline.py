from __future__ import annotations

from pathlib import Path
import re
import os
from typing import Any

from .audit import audit_artifacts
from .checkpoints import checkpoint_signature, stage_completed, write_checkpoint
from .common import (
    append_diagnostic,
    append_pipeline_event,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    file_digest,
    read_json,
    read_jsonl,
    run_paths,
    utc_now,
)
from .config import RuntimeConfig
from .inputs import SurveyRequest
from .llm import llm_status
from .manifest import build_final_manifest, write_final_manifest
from .survey_agent import AGENT_EXPORT_VERSION, AGENT_SCHEMA_VERSION, run_survey_agent, survey_agent_requested
from .paper_sources import (
    build_chronology,
    build_clusters,
    extract_papers,
    graph_papers_and_context,
    normalize_papers,
    parent_artifacts_for_source,
    resolve_graph_artifacts,
    synthetic_topic_papers,
)

SKILL_VERSION = "1.0.0"


def run_pipeline(
    *,
    cwd: Path,
    run_dir: Path,
    run_id: str,
    request: SurveyRequest,
    runtime: RuntimeConfig,
    requested_status: str | None = None,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    paths = run_paths(run_dir)
    for directory in (paths["artifacts"], paths["state"], paths["logs"]):
        directory.mkdir(parents=True, exist_ok=True)
    request_json = request.to_json(cwd)
    runtime_json = runtime.to_json()
    atomic_write_json(paths["request"], request_json)
    atomic_write_json(paths["runtime_config"], runtime_json)
    source_signatures = source_signatures_for_request(request, cwd)
    request_signature, runtime_signature, source_signature = checkpoint_signature(request_json, runtime_json, source_signatures)

    ingest_sources(cwd, run_dir, request, runtime, allow_synthetic, request_signature, runtime_signature, source_signatures, source_signature)
    papers = list(read_jsonl(paths["papers_jsonl"]))
    if papers:
        ingest_graph_context(cwd, run_dir, request, runtime, request_signature, runtime_signature, source_signatures, source_signature)
        cluster(run_dir, request, runtime, request_signature, runtime_signature, source_signatures, source_signature)
        if survey_agent_requested(runtime):
            try:
                survey_agent(run_dir, request, runtime, request_signature, runtime_signature, source_signatures, source_signature)
            except Exception as error:  # noqa: BLE001 - provider/runtime failures must still yield resumable artifacts.
                message = f"Survey Agent generation failed: {error}"
                append_diagnostic(run_dir, "error", message, code="survey_agent_failed")
                write_checkpoint(
                    run_dir,
                    "survey_agent",
                    {"counts": {"sections": 0, "claims": 0, "gaps": 0}, "mode": "survey_agent_failed"},
                    blocked_stage="survey_agent",
                    request_signature=request_signature,
                    runtime_config_signature=runtime_signature,
                    source_signatures=source_signatures,
                    last_error=message,
                )
        else:
            record_survey_agent_unavailable(run_dir, request, runtime, request_signature, runtime_signature, source_signatures, source_signature)
        citation_traces(run_dir, runtime, request_signature, runtime_signature, source_signatures, source_signature)
        render_artifacts(run_dir, run_id, request, runtime, request_signature, runtime_signature, source_signatures, source_signature)
    else:
        render_empty_artifacts(run_dir, run_id, request, runtime)

    return audit_and_manifest(cwd, run_dir, run_id, request_json, requested_status=requested_status)


def source_signatures_for_request(request: SurveyRequest, cwd: Path) -> dict[str, str | None]:
    signatures: dict[str, str | None] = {
        "graph_path": file_digest(request.graph_path) if request.graph_path and request.graph_path.is_file() else None,
        "input": file_digest(request.input_path) if request.input_path and request.input_path.is_file() else None,
    }
    alias_path = os.environ.get("XLAB_LITERATURE_SURVEY_CITATION_ALIASES")
    signatures["citation_aliases"] = file_digest(Path(alias_path)) if alias_path else None
    if request.graph_path and request.graph_path.exists():
        graph_artifacts = resolve_graph_artifacts(request.graph_path, cwd)
        for key, artifact_path in graph_artifacts.items():
            signatures[f"graph_{key}"] = file_digest(artifact_path) if artifact_path else None
    return signatures


def ingest_sources(
    cwd: Path,
    run_dir: Path,
    request: SurveyRequest,
    runtime: RuntimeConfig,
    allow_synthetic: bool,
    request_signature: str,
    runtime_signature: str,
    source_signatures: dict[str, str | None],
    source_signature: str,
) -> None:
    paths = run_paths(run_dir)
    if stage_completed(
        run_dir,
        "ingest_sources",
        [paths["papers_jsonl"], paths["references_jsonl"], paths["paper_index"]],
        request_signature=request_signature,
        runtime_config_signature=runtime_signature,
        source_signature=source_signature,
    ):
        append_pipeline_event(run_dir, "ingest_sources", "skipped")
        return

    warnings = list(request.compatibility_warnings)
    raw_papers: list[dict[str, Any]] = []
    graph_context: dict[str, Any] = {}
    graph_warnings: list[str] = []
    source_type = "missing"
    source_path: Path | None = None

    if request.graph_path:
        source_path = request.graph_path
        if request.graph_path.exists():
            graph_papers, graph_context, graph_warnings = graph_papers_and_context(request.graph_path, cwd, runtime.graph_context_limit)
            raw_papers = graph_papers
            if graph_papers:
                source_type = "knowledge_graph"
        else:
            graph_warnings.append(f"Graph path does not exist: {request.graph_path}")
    warnings.extend(graph_warnings)

    if not raw_papers and request.input_path:
        source_path = request.input_path
        if request.input_path.exists() and request.input_path.is_file():
            raw_papers = extract_papers(read_json(request.input_path), cwd)
            source_type = "paper_manifest" if raw_papers else "missing"
        else:
            warnings.append(f"Input paper manifest does not exist or is not a file: {request.input_path}")

    if not raw_papers and (allow_synthetic or runtime.allow_synthetic):
        raw_papers = synthetic_topic_papers(request.topic, request.min_papers)
        source_type = "synthetic"
        warnings.append("Synthetic papers were used for a smoke/test run; normal runs require --graph or --input.")

    normalized, stats = normalize_papers(raw_papers, request.max_papers)
    references = [reference_from_paper(paper) for paper in normalized]
    abstract_count = sum(1 for paper in normalized if str(paper.get("abstract") or "").strip())
    if stats.get("truncated_count", 0) > 0:
        warnings.append(f"Input source contained {stats['source_count']} papers; only {len(normalized)} were materialized by max_papers.")
    if stats.get("duplicate_count", 0) > 0:
        warnings.append(f"Removed {stats['duplicate_count']} duplicate papers during source ingestion.")
    if normalized and abstract_count / len(normalized) < 0.5:
        warnings.append("Less than half of normalized papers include abstracts; the survey uses titles and graph context where needed.")
    if not normalized:
        warnings.append("No real papers were available; provide --graph <knowledge_graph> or --input <paper_set> to generate a successful survey.")

    atomic_write_jsonl(paths["papers_jsonl"], normalized)
    atomic_write_jsonl(paths["references_jsonl"], references)
    atomic_write_json(paths["graph_context"], graph_context or {"available": False})
    parent_artifacts = parent_artifacts_for_source(source_path, cwd, source_type)
    paper_index = {
        "schema_version": "xlab.literature_survey.paper_index.v1",
        "source_type": source_type,
        "source_path": str(source_path) if source_path else None,
        "source_signatures": source_signatures,
        "parent_artifacts": parent_artifacts,
        "paper_count": len(normalized),
        "source_count": stats.get("source_count", 0),
        "duplicate_count": stats.get("duplicate_count", 0),
        "truncated_count": stats.get("truncated_count", 0),
        "abstract_count": abstract_count,
        "abstract_coverage": abstract_count / len(normalized) if normalized else 0,
        "warnings": warnings,
        "generated_at": utc_now(),
    }
    atomic_write_json(paths["paper_index"], paper_index)
    for warning in warnings:
        append_diagnostic(run_dir, "warning", warning)
    write_checkpoint(
        run_dir,
        "ingest_sources",
        {"counts": {"papers": len(normalized)}, "paper_index": paper_index},
        completed_stage="ingest_sources",
        request_signature=request_signature,
        runtime_config_signature=runtime_signature,
        source_signatures=source_signatures,
        warnings=warnings,
    )
    append_pipeline_event(run_dir, "ingest_sources", "success", {"papers": len(normalized), "source_type": source_type})


def ingest_graph_context(
    cwd: Path,
    run_dir: Path,
    request: SurveyRequest,
    runtime: RuntimeConfig,
    request_signature: str,
    runtime_signature: str,
    source_signatures: dict[str, str | None],
    source_signature: str,
) -> None:
    paths = run_paths(run_dir)
    if stage_completed(
        run_dir,
        "ingest_graph_context",
        [paths["graph_context"]],
        request_signature=request_signature,
        runtime_config_signature=runtime_signature,
        source_signature=source_signature,
    ):
        append_pipeline_event(run_dir, "ingest_graph_context", "skipped")
        return
    graph_context = read_json(paths["graph_context"]) if paths["graph_context"].exists() else {"available": False}
    if request.graph_path and request.graph_path.exists():
        _papers, graph_context, warnings = graph_papers_and_context(request.graph_path, cwd, runtime.graph_context_limit)
        for warning in warnings:
            append_diagnostic(run_dir, "warning", warning)
        atomic_write_json(paths["graph_context"], graph_context)
        write_checkpoint(
            run_dir,
            "ingest_graph_context",
            {"counts": {"graph_terms": len(graph_context.get("core_terms", [])) if isinstance(graph_context, dict) else 0}},
            completed_stage="ingest_graph_context",
            request_signature=request_signature,
            runtime_config_signature=runtime_signature,
            source_signatures=source_signatures,
            warnings=warnings,
        )
    else:
        write_checkpoint(
            run_dir,
            "ingest_graph_context",
            {"counts": {"graph_terms": 0}},
            completed_stage="ingest_graph_context",
            request_signature=request_signature,
            runtime_config_signature=runtime_signature,
            source_signatures=source_signatures,
        )
    append_pipeline_event(run_dir, "ingest_graph_context", "success")


def cluster(
    run_dir: Path,
    request: SurveyRequest,
    runtime: RuntimeConfig,
    request_signature: str,
    runtime_signature: str,
    source_signatures: dict[str, str | None],
    source_signature: str,
) -> None:
    paths = run_paths(run_dir)
    if stage_completed(
        run_dir,
        "cluster",
        [paths["clusters_json"], paths["chronology_json"]],
        request_signature=request_signature,
        runtime_config_signature=runtime_signature,
        source_signature=source_signature,
    ):
        append_pipeline_event(run_dir, "cluster", "skipped")
        return
    papers = list(read_jsonl(paths["papers_jsonl"]))
    graph_context = read_json(paths["graph_context"]) if paths["graph_context"].exists() else {}
    clusters = build_clusters(request.topic, papers, runtime.cluster_limit, graph_context if isinstance(graph_context, dict) else None)
    chronology = build_chronology(papers)
    atomic_write_json(paths["clusters_json"], clusters)
    atomic_write_json(paths["chronology_json"], chronology)
    write_checkpoint(
        run_dir,
        "cluster",
        {"counts": {"papers": len(papers), "clusters": len(clusters)}, "llm_path": survey_agent_requested(runtime)},
        completed_stage="cluster",
        request_signature=request_signature,
        runtime_config_signature=runtime_signature,
        source_signatures=source_signatures,
    )
    append_pipeline_event(run_dir, "cluster", "success", {"clusters": len(clusters)})


def survey_agent(
    run_dir: Path,
    request: SurveyRequest,
    runtime: RuntimeConfig,
    request_signature: str,
    runtime_signature: str,
    source_signatures: dict[str, str | None],
    source_signature: str,
) -> None:
    paths = run_paths(run_dir)
    cached_result = read_json(paths["survey_agent_result"]) if paths["survey_agent_result"].exists() else {}
    evaluation_path = paths["state"] / "xcientist" / "evaluation.json"
    evaluation = read_json(evaluation_path) if evaluation_path.exists() else {}
    evaluation_needs_retry = evaluation.get("error") or evaluation.get("evaluation_version") != 2
    if not evaluation_needs_retry and cached_result.get("export_version") == AGENT_EXPORT_VERSION and stage_completed(
        run_dir,
        "survey_agent",
        [
            paths["survey_agent_outline"],
            paths["survey_agent_assignment"],
            paths["survey_agent_draft"],
            paths["survey_agent_result"],
            paths["sections_jsonl"],
            paths["claims_jsonl"],
            paths["gaps_jsonl"],
        ],
        request_signature=request_signature,
        runtime_config_signature=runtime_signature,
        source_signature=source_signature,
    ):
        append_pipeline_event(run_dir, "survey_agent", "skipped")
        return
    papers = list(read_jsonl(paths["papers_jsonl"]))
    clusters = read_json(paths["clusters_json"])
    graph_context = read_json(paths["graph_context"]) if paths["graph_context"].exists() else {}
    append_pipeline_event(run_dir, "survey_agent", "started", {"papers": len(papers), "model": runtime.model})
    result = run_survey_agent(
        run_dir=run_dir,
        request=request,
        runtime=runtime,
        papers=papers,
        clusters=clusters if isinstance(clusters, list) else [],
        graph_context=graph_context if isinstance(graph_context, dict) else {},
    )
    sections = result["sections"] if isinstance(result.get("sections"), list) else []
    claims = result["key_claims"] if isinstance(result.get("key_claims"), list) else []
    gaps = result["research_gaps"] if isinstance(result.get("research_gaps"), list) else []
    atomic_write_json(paths["survey_agent_outline"], result.get("outline", {}))
    atomic_write_json(paths["survey_agent_assignment"], result.get("paper_assignment", {}))
    atomic_write_text(paths["survey_agent_draft"], str(result.get("markdown_body") or ""))
    atomic_write_json(paths["survey_agent_result"], result)
    atomic_write_jsonl(paths["sections_jsonl"], sections)
    atomic_write_jsonl(paths["claims_jsonl"], claims)
    atomic_write_jsonl(paths["gaps_jsonl"], gaps)
    write_checkpoint(
        run_dir,
        "survey_agent",
        {"counts": {"sections": len(sections), "claims": len(claims), "gaps": len(gaps)}, "mode": result.get("mode")},
        completed_stage="survey_agent",
        last_error="",
        request_signature=request_signature,
        runtime_config_signature=runtime_signature,
        source_signatures=source_signatures,
    )
    append_pipeline_event(run_dir, "survey_agent", "success", {"sections": len(sections), "claims": len(claims), "gaps": len(gaps)})


def record_survey_agent_unavailable(
    run_dir: Path,
    request: SurveyRequest,
    runtime: RuntimeConfig,
    request_signature: str,
    runtime_signature: str,
    source_signatures: dict[str, str | None],
    source_signature: str,
) -> None:
    paths = run_paths(run_dir)
    runtime_info = llm_status(runtime)
    reason = runtime_info.get("reason") or "SurveyAgent provider configuration is unavailable."
    message = "Survey Agent generation skipped because LLM refinement is unavailable; provide OPENAI_API_KEY or enable the SurveyAgent runtime."
    append_diagnostic(run_dir, "error", message, code="survey_agent_unavailable")
    result = {
        "schema_version": AGENT_SCHEMA_VERSION,
        "mode": "survey_agent_unavailable",
        "engine": {"mode": "survey_agent_unavailable", "reason": reason},
        "topic": request.topic,
        "outline": {},
        "paper_assignment": {},
        "markdown_body": "",
        "sections": [],
        "key_claims": [],
        "research_gaps": [],
        "warnings": [message, str(reason)],
    }
    atomic_write_json(paths["survey_agent_outline"], {})
    atomic_write_json(paths["survey_agent_assignment"], {})
    atomic_write_text(paths["survey_agent_draft"], "")
    atomic_write_json(paths["survey_agent_result"], result)
    atomic_write_jsonl(paths["sections_jsonl"], [])
    atomic_write_jsonl(paths["claims_jsonl"], [])
    atomic_write_jsonl(paths["gaps_jsonl"], [])
    write_checkpoint(
        run_dir,
        "survey_agent",
        {"counts": {"sections": 0, "claims": 0, "gaps": 0}, "mode": "survey_agent_unavailable"},
        blocked_stage="survey_agent",
        request_signature=request_signature,
        runtime_config_signature=runtime_signature,
        source_signatures=source_signatures,
        last_error=message,
    )
    append_pipeline_event(run_dir, "survey_agent", "blocked", {"reason": reason})



def citation_traces(
    run_dir: Path,
    runtime: RuntimeConfig,
    request_signature: str,
    runtime_signature: str,
    source_signatures: dict[str, str | None],
    source_signature: str,
) -> None:
    paths = run_paths(run_dir)
    if stage_completed(
        run_dir,
        "citation_traces",
        [paths["traces_jsonl"]],
        request_signature=request_signature,
        runtime_config_signature=runtime_signature,
        source_signature=source_signature,
    ):
        append_pipeline_event(run_dir, "citation_traces", "skipped")
        return
    papers = list(read_jsonl(paths["papers_jsonl"]))
    by_id = {paper["id"]: paper for paper in papers}
    claims = list(read_jsonl(paths["claims_jsonl"]))
    traces = []
    for claim in claims:
        traces.append(
            {
                "claim_id": claim["id"],
                "claim": claim["claim"],
                "paper_ids": claim["paper_ids"],
                "evidence": [paper_summary(by_id[paper_id], runtime.evidence_chars) for paper_id in claim["paper_ids"] if paper_id in by_id],
            }
        )
    atomic_write_jsonl(paths["traces_jsonl"], traces)
    write_checkpoint(
        run_dir,
        "citation_traces",
        {"counts": {"traces": len(traces)}},
        completed_stage="citation_traces",
        request_signature=request_signature,
        runtime_config_signature=runtime_signature,
        source_signatures=source_signatures,
    )
    append_pipeline_event(run_dir, "citation_traces", "success", {"traces": len(traces)})


def render_artifacts(
    run_dir: Path,
    run_id: str,
    request: SurveyRequest,
    runtime: RuntimeConfig,
    request_signature: str,
    runtime_signature: str,
    source_signatures: dict[str, str | None],
    source_signature: str,
) -> None:
    paths = run_paths(run_dir)
    if stage_completed(
        run_dir,
        "render_artifacts",
        [paths["survey_md"], paths["survey_json"], paths["citations_json"]],
        request_signature=request_signature,
        runtime_config_signature=runtime_signature,
        source_signature=source_signature,
    ):
        append_pipeline_event(run_dir, "render_artifacts", "skipped")
        return
    papers = list(read_jsonl(paths["papers_jsonl"]))
    clusters = read_json(paths["clusters_json"])
    chronology = read_json(paths["chronology_json"])
    sections = list(read_jsonl(paths["sections_jsonl"]))
    claims = list(read_jsonl(paths["claims_jsonl"]))
    gaps = list(read_jsonl(paths["gaps_jsonl"]))
    references = list(read_jsonl(paths["references_jsonl"]))
    traces = list(read_jsonl(paths["traces_jsonl"]))
    graph_context = read_json(paths["graph_context"]) if paths["graph_context"].exists() else {}
    paper_index = read_json(paths["paper_index"])
    runtime_info = llm_status(runtime)
    agent_result = read_json(paths["survey_agent_result"]) if paths["survey_agent_result"].exists() else None
    agent_markdown = str(agent_result.get("markdown_body") or "") if isinstance(agent_result, dict) else ""
    citations = {
        "schema_version": "xlab.citation_trace.v1",
        "run_id": run_id,
        "references": references,
        "traces": traces,
    }
    survey = {
        "schema_version": "xlab.literature_survey.v1",
        "run_id": run_id,
        "topic": request.topic,
        "language": request.language,
        "depth": request.depth,
        "paper_count": len(papers),
        "clusters": clusters,
        "chronology": chronology,
        "sections": sections,
        "key_claims": claims,
        "research_gaps": gaps,
        "references": references,
        "citation_trace_path": "citations.json",
        "runtime": {
            "mode": "survey-agent" if agent_markdown else "survey-agent-unavailable",
            "full_text_requested": request.full_text,
            "llm": runtime_info,
            "semantic_scholar_api_key_set": runtime.semantic_scholar_api_key_set,
            "survey_agent_engine": "integrated_xcientist_survey_agent",
        },
        "source": paper_index,
        "graph_context": graph_context,
        "generated_at": utc_now(),
    }
    markdown = render_survey_agent_markdown(request.topic, agent_markdown, references) if agent_markdown else render_unavailable_markdown(request.topic, runtime_info)
    atomic_write_text(paths["survey_md"], markdown)
    atomic_write_json(paths["survey_json"], survey)
    atomic_write_json(paths["citations_json"], citations)
    write_checkpoint(
        run_dir,
        "render_artifacts",
        {"counts": {"papers": len(papers), "sections": len(sections), "claims": len(claims), "gaps": len(gaps), "traces": len(traces)}},
        completed_stage="render_artifacts",
        request_signature=request_signature,
        runtime_config_signature=runtime_signature,
        source_signatures=source_signatures,
    )
    append_pipeline_event(run_dir, "render_artifacts", "success")


def render_empty_artifacts(run_dir: Path, run_id: str, request: SurveyRequest, runtime: RuntimeConfig) -> None:
    paths = run_paths(run_dir)
    graph_context = read_json(paths["graph_context"]) if paths["graph_context"].exists() else {"available": False}
    paper_index = read_json(paths["paper_index"]) if paths["paper_index"].exists() else {"source_type": "missing", "paper_count": 0}
    survey = {
        "schema_version": "xlab.literature_survey.v1",
        "run_id": run_id,
        "topic": request.topic,
        "language": request.language,
        "depth": request.depth,
        "paper_count": 0,
        "clusters": [],
        "chronology": [],
        "sections": [],
        "key_claims": [],
        "research_gaps": [],
        "references": [],
        "citation_trace_path": "citations.json",
        "runtime": {
            "mode": "missing-source",
            "full_text_requested": request.full_text,
            "llm": llm_status(runtime),
            "semantic_scholar_api_key_set": runtime.semantic_scholar_api_key_set,
            "survey_agent_engine": "integrated_xcientist_survey_agent",
        },
        "source": paper_index,
        "graph_context": graph_context,
        "generated_at": utc_now(),
    }
    citations = {"schema_version": "xlab.citation_trace.v1", "run_id": run_id, "references": [], "traces": []}
    markdown = f"# {request.topic}\n\nNo real papers were available. Provide `--graph <knowledge_graph>` or `--input <paper_set>` and rerun `resume`.\n"
    atomic_write_text(paths["survey_md"], markdown)
    atomic_write_json(paths["survey_json"], survey)
    atomic_write_json(paths["citations_json"], citations)


def audit_and_manifest(cwd: Path, run_dir: Path, run_id: str, request_json: dict[str, Any], *, requested_status: str | None = None) -> dict[str, Any]:
    paths = run_paths(run_dir)
    report = audit_artifacts(run_dir, int(request_json.get("min_papers") or 3))
    status = "success" if report.get("passed") is True else "incomplete"
    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []
    if requested_status == "success" and status != "success":
        warnings.append("Requested status success was ignored because audit did not pass.")
    elif requested_status == "incomplete" and status == "success":
        warnings.append("Requested status incomplete downgraded an otherwise successful survey manifest.")
        status = "incomplete"
    report["warnings"] = warnings
    atomic_write_json(paths["report_json"], report)
    survey = read_json(paths["survey_json"]) if paths["survey_json"].exists() else {}
    citations = read_json(paths["citations_json"]) if paths["citations_json"].exists() else {}
    manifest = build_final_manifest(
        cwd=cwd,
        run_dir=run_dir,
        run_id=run_id,
        skill_version=SKILL_VERSION,
        status=status,
        request=request_json,
        survey=survey if isinstance(survey, dict) else {},
        citations=citations if isinstance(citations, dict) else {},
        report=report,
    )
    write_final_manifest(paths["manifest_json"], manifest)
    return {"manifest": manifest, "report": report, "status": status}


def reference_from_paper(paper: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": paper["id"],
        "title": paper["title"],
        "year": paper.get("year"),
        "venue": paper.get("venue"),
        "authors": paper.get("authors", []),
        "url": paper.get("url"),
        "source": paper.get("source", "input"),
    }


def paper_summary(paper: dict[str, Any], limit: int) -> str:
    abstract = paper.get("abstract") or paper.get("title") or ""
    return str(abstract)[:limit]


def render_survey_agent_markdown(topic: str, body: str, references: list[dict[str, Any]]) -> str:
    lines = [f"# {topic}", "", body.strip(), "", "## References"]
    cited = set(re.findall(r"\[paper:([^\]]+)\]", body))
    for index, reference in enumerate((item for item in references if item["paper_id"] in cited), start=1):
        year = f" ({reference['year']})" if reference.get("year") else ""
        authors = ", ".join(reference.get("authors") or [])
        author_text = f"{authors}. " if authors else ""
        venue = f" {reference['venue']}." if reference.get("venue") else ""
        url = reference.get("url") or ""
        link = f" <{url}>" if url.startswith(("https://", "http://")) else ""
        lines.append(f"{index}. [{reference['paper_id']}] {author_text}{reference['title']}{year}.{venue}{link}")
    lines.append("")
    return "\n".join(lines)


def render_unavailable_markdown(topic: str, runtime_info: dict[str, object]) -> str:
    lines = [
        f"# {topic}",
        "",
        "The integrated SurveyAgent path did not run, so no survey draft was generated.",
        "Provide the required SurveyAgent API configuration and rerun `resume` with the same run directory.",
    ]
    if runtime_info.get("reason"):
        lines.extend(["", f"> Note: {runtime_info['reason']}"])
    lines.append("")
    return "\n".join(lines)
