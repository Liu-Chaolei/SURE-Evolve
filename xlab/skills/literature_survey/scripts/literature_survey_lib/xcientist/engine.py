from __future__ import annotations

import re
import unicodedata
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from ..checkpoints import write_checkpoint
from ..common import append_pipeline_event, atomic_write_json, atomic_write_jsonl, atomic_write_text, read_json, run_paths, utc_now
from .adapters import IntegratedSurveyContext, build_integrated_context, materialize_context_papers
from .modules.judge import Judge
from .utils.prose_cleanup import remove_repeated_prose
from .utils.editorial_corrections import apply_editorial_corrections

ENGINE_SCHEMA_VERSION = "xlab.literature_survey.xcientist_engine.v1"


def run_integrated_survey_engine(
    *,
    config: DictConfig,
    request: Any,
    runtime: Any,
    papers: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    graph_context: dict[str, Any],
    graph_db_path: Path | None = None,
) -> dict[str, Any]:
    """Run the integrated Xcientist SurveyAgent flow on XLab-provided evidence."""
    if not papers:
        raise RuntimeError("Integrated SurveyAgent requires at least one paper from --graph or --input.")

    run_dir = Path(str(config.BasicInfo.base_dir)).expanduser().resolve()
    state_dir = Path(str(config.BasicInfo.cache_path)).parent
    state_dir.mkdir(parents=True, exist_ok=True)
    write_engine_progress(run_dir, state_dir, "context", counts={"papers": len(papers)})

    context = build_integrated_context(
        config=config,
        papers=papers,
        graph_db_path=graph_db_path,
    )

    seed_path = state_dir / "seed_papers.json"
    if seed_path.exists():
        seed_paper_ids = [str(paper_id) for paper_id in read_json_list(seed_path)]
    else:
        seed_paper_ids = list(context.paper_by_id)
        atomic_write_json(seed_path, seed_paper_ids)
    context.work_collector.graph_paper_ids.update(seed_paper_ids)
    write_engine_progress(run_dir, state_dir, "seed_papers", counts={"seed_papers": len(seed_paper_ids)})

    expanded_path = state_dir / "expanded_papers.json"
    if expanded_path.exists():
        expanded_paper_ids = [str(paper_id) for paper_id in read_json_list(expanded_path)]
    else:
        expanded_paper_ids = context.work_collector.expand_seed_papers_by_reference_and_citation(seed_paper_ids)
        atomic_write_json(expanded_path, expanded_paper_ids)
    collected_papers = unique_preserve_order([*seed_paper_ids, *expanded_paper_ids])
    if not collected_papers:
        collected_papers = list(context.paper_by_id)
    materialized_papers = materialize_context_papers(context, collected_papers)
    papers = materialized_papers
    sync_pipeline_papers(run_dir, papers)
    atomic_write_json(state_dir / "collected_papers.json", collected_papers)
    write_engine_progress(
        run_dir,
        state_dir,
        "collect_papers",
        counts={"seed_papers": len(seed_paper_ids), "expanded_papers": len(expanded_paper_ids), "collected_papers": len(collected_papers)},
    )

    context.database.build_with_graph()
    keynotes_path = state_dir / "keynotes.json"
    if keynotes_path.exists():
        keynotes_state = read_json(keynotes_path)
        failed = {str(paper_id) for paper_id in keynotes_state.get("failed_paper_ids", [])} if isinstance(keynotes_state, dict) else set()
    else:
        failed_papers = context.work_analyzer.read_papers_and_write_keynotes(collected_papers)
        failed = {str(paper_id) for paper_id in failed_papers or []}
        keynote_records: dict[str, Any] = {}
        keynote_errors: dict[str, str] = {}
        for paper_id in collected_papers:
            if paper_id in failed:
                continue
            try:
                keynote_records[paper_id] = context.work_analyzer.get_paper_keynote(paper_id)
            except Exception as error:
                failed.add(paper_id)
                keynote_errors[paper_id] = str(error)
        atomic_write_json(keynotes_path, {"keynotes": keynote_records, "failed_paper_ids": sorted(failed), "errors": keynote_errors})
    collected_papers = [paper_id for paper_id in collected_papers if paper_id not in failed]
    if not collected_papers:
        raise RuntimeError("Integrated SurveyAgent could not materialize any paper notes.")
    materialize_context_papers(context, collected_papers)
    write_engine_progress(run_dir, state_dir, "keynotes", counts={"collected_papers": len(collected_papers), "failed_papers": len(failed)})

    clustering_path = state_dir / "clustering_result.json"
    if clustering_path.exists():
        cached_clustering = read_json(clustering_path)
        clustering_result = cached_clustering if isinstance(cached_clustering, list) else []
    else:
        clustering_result = context.work_analyzer.cluster_papers(collected_papers)
        atomic_write_json(clustering_path, clustering_result)
    context.work_analyzer.log_clusters(clustering_result)
    atomic_write_json(run_paths(run_dir)["clusters_json"], clusters_for_artifacts(clustering_result))
    write_engine_progress(run_dir, state_dir, "clustering", counts={"clusters": len(clustering_result)})

    analysis_path = state_dir / "analysis_context.json"
    if analysis_path.exists():
        analysis_state = read_json(analysis_path)
        relation_graph = relation_graphs_from_json(analysis_state.get("relation_graph")) if isinstance(analysis_state, dict) else None
        relation_table = analysis_state.get("relation_table") if isinstance(analysis_state, dict) else None
        intra_analysis_results = analysis_state.get("intra_analysis_results", []) if isinstance(analysis_state, dict) else []
        inter_analysis_results = str(analysis_state.get("inter_analysis_results", "")) if isinstance(analysis_state, dict) else ""
        context.work_analyzer.relation_analysis_graph = relation_graph
        context.work_analyzer.relation_analysis_table = relation_table
    else:
        relation_graph = None
        relation_table = None
        intra_analysis_results: list[list[dict[str, Any]]] = []
        inter_analysis_results = ""

        if config.ModuleInfo.SurveyGenerator.include_relation_graph:
            relation_graph = context.work_analyzer.build_relationship_graphs(clustering_result)
        if config.ModuleInfo.SurveyGenerator.include_relation_table:
            relation_table = context.work_analyzer.generate_cluster_tables(clustering_result)
        if config.ModuleInfo.SurveyGenerator.include_initial_analysis:
            intra_analysis_results = context.work_analyzer.intra_cluster_analysis(clustering_result)
            context.work_analyzer.log_intra_cluster_analysis(intra_analysis_results)
            inter_analysis_results = context.work_analyzer.inter_cluster_analysis(intra_analysis_results)
            context.work_analyzer.log_inter_cluster_analysis(inter_analysis_results)
        atomic_write_json(
            analysis_path,
            {
                "relation_graph": relation_graphs_for_json(relation_graph),
                "relation_table": relation_table,
                "intra_analysis_results": intra_analysis_results,
                "inter_analysis_results": inter_analysis_results,
            },
        )
    write_engine_progress(run_dir, state_dir, "analysis", counts={"clusters": len(clustering_result)})

    outline_path = state_dir / "outline.raw.json"
    if outline_path.exists():
        outline = read_json(outline_path)
        if config.ModuleInfo.SurveyGenerator.outline_generation_in_steps and outline.get("assignment_version") != 2:
            outline = context.survey_generator.generate_outline_assign_papers(
                outline, intra_analysis_results, inter_analysis_results, collected_papers
            )
            atomic_write_json(outline_path, outline)
            superseded = state_dir / ("superseded-assignment-" + utc_now().replace(":", "-"))
            for name in ("draft.raw.json", "draft.raw.md", "refined.raw.md", "refined.titled.md", "refined.titled.meta.json", "references.raw.json", "evaluation.json", "engine_result.json"):
                path = state_dir / name
                if path.exists():
                    superseded.mkdir(parents=True, exist_ok=True)
                    path.replace(superseded / name)
    else:
        try:
            outline = context.survey_generator.generate_outline(intra_analysis_results, inter_analysis_results, collected_papers)
        except Exception as error:
            atomic_write_json(state_dir / "outline.error.json", {"error": str(error), "generated_at": utc_now()})
            raise
        atomic_write_json(outline_path, outline)
    context.survey_generator.log_outline(outline)
    write_engine_progress(run_dir, state_dir, "outline", counts={"outline_sections": len(outline.get("sections", [])) if isinstance(outline, dict) else 0})

    draft_json_path = state_dir / "draft.raw.json"
    draft_md_path = state_dir / "draft.raw.md"
    if draft_json_path.exists():
        draft = read_json(draft_json_path)
        if draft.get("review_version") != 2:
            write_engine_progress(run_dir, state_dir, "review", counts={"draft_sections": len(draft.get("section_drafts", []))})
            superseded = state_dir / ("superseded-review-" + utc_now().replace(":", "-"))
            superseded.mkdir(parents=True, exist_ok=True)
            atomic_write_json(superseded / "draft.raw.json", draft)
            draft = context.survey_generator.review_and_revise_survey_in_parts(draft, outline)
            draft["review_version"] = 2
            atomic_write_json(draft_json_path, draft)
            for name in ("draft.raw.md", "refined.raw.md", "refined.titled.md", "refined.titled.meta.json", "references.raw.json", "evaluation.json", "engine_result.json"):
                path = state_dir / name
                if path.exists():
                    path.replace(superseded / name)
    else:
        try:
            unreviewed_path = state_dir / "draft.unreviewed.json"
            unreviewed = read_json(unreviewed_path) if unreviewed_path.exists() else {}
            if unreviewed.get("outline") == outline and unreviewed.get("full_draft"):
                draft = unreviewed
            else:
                draft = context.survey_generator.draft_survey(intra_analysis_results, inter_analysis_results, outline)
                atomic_write_json(unreviewed_path, draft)
            write_engine_progress(run_dir, state_dir, "review", counts={"draft_sections": len(draft.get("section_drafts", []))})
            draft = context.survey_generator.review_and_revise_survey_in_parts(draft, outline)
            draft["review_version"] = 2
        except Exception as error:
            atomic_write_json(state_dir / "draft.error.json", {"error": str(error), "generated_at": utc_now()})
            raise
        atomic_write_json(draft_json_path, draft)
        atomic_write_text(draft_md_path, str(draft.get("full_draft") or ""))
    if not draft_md_path.exists():
        atomic_write_text(draft_md_path, str(draft.get("full_draft") or ""))
    write_engine_progress(run_dir, state_dir, "draft", counts={"draft_sections": len(draft.get("section_drafts", [])) if isinstance(draft, dict) else 0})

    refined_md_path = state_dir / "refined.raw.md"
    references_path = state_dir / "references.raw.json"
    reference_meta_path = state_dir / "references.raw.meta.json"
    reference_meta = read_json(reference_meta_path) if reference_meta_path.exists() else {}
    if refined_md_path.exists() and references_path.exists() and reference_meta.get("resolution_version") != 3:
        archive = state_dir / ("superseded-reference-resolution-" + utc_now().replace(":", "-"))
        archive.mkdir(parents=True, exist_ok=True)
        for name in ("refined.raw.md", "references.raw.json", "evaluation.json", "engine_result.json"):
            path = state_dir / name
            if path.exists():
                path.replace(archive / name)
    if refined_md_path.exists() and references_path.exists():
        survey_with_references = refined_md_path.read_text(encoding="utf-8")
        references = [str(reference) for reference in read_json_list(references_path)]
        context.survey_generator.save_survey(survey_with_references, references)
    else:
        try:
            titled_path = state_dir / "refined.titled.md"
            titled_meta_path = state_dir / "refined.titled.meta.json"
            titled_meta = read_json(titled_meta_path) if titled_meta_path.exists() else {}
            draft_digest = hashlib.sha256(json.dumps(draft, sort_keys=True).encode()).hexdigest()
            if titled_path.exists() and titled_meta.get("draft_sha256") == draft_digest and titled_meta.get("model") == runtime.model:
                alias_path = state_dir / "citation_aliases.json"
                editorial = read_json(alias_path).get("editorial_corrections", []) if alias_path.exists() else []
                titled_text, correction_report = apply_editorial_corrections(
                    titled_path.read_text(), editorial, set(context.database.reference_papers)
                )
                atomic_write_json(state_dir / "editorial_corrections.json", correction_report)
                survey_with_references, references = context.survey_generator.finalize_citations(titled_text, outline)
            else:
                survey_with_references, references = context.survey_generator.refine_draft(draft)
        except Exception as error:
            atomic_write_json(state_dir / "citation_resolutions.json", list(context.database.resolution_records.values()))
            atomic_write_json(state_dir / "refine.error.json", {"error": str(error), "generated_at": utc_now()})
            raise
        references = [str(reference) for reference in references]
        context.survey_generator.save_survey(survey_with_references, references)
        atomic_write_text(refined_md_path, survey_with_references)
        atomic_write_json(references_path, references)
        atomic_write_json(reference_meta_path, {"resolution_version": 3})
    survey_with_references, repetitions_removed = remove_repeated_prose(survey_with_references)
    if repetitions_removed:
        atomic_write_text(refined_md_path, survey_with_references)
        context.survey_generator.save_survey(survey_with_references, references)
        atomic_write_json(state_dir / "prose_cleanup.json", {"removed_exact_paragraph_repetitions": repetitions_removed})
    write_engine_progress(run_dir, state_dir, "refine", counts={"references": len(references)})

    known_references = set(context.paper_by_id) | set(context.work_collector.data_manager.xlab_paper_by_id)
    if not set(references).issubset(known_references):
        raise ValueError("Survey references include papers outside the input graph")
    papers = materialize_context_papers(context, unique_preserve_order([*collected_papers, *references]))
    sync_pipeline_papers(run_dir, papers)

    evaluation_path = state_dir / "evaluation.json"
    evaluation = read_json(evaluation_path) if evaluation_path.exists() else {}
    if evaluation.get("evaluation_version") != 2 or evaluation.get("error"):
        evaluation = evaluate_survey(config, context, survey_with_references, references)
    write_engine_progress(run_dir, state_dir, "evaluation", counts={"references": len(references)})

    markdown_body = normalize_xlab_markdown(survey_with_references, references, context)
    enriched_ids = ids_from_text(markdown_body, known_references)
    papers = materialize_context_papers(context, unique_preserve_order([*collected_papers, *references, *enriched_ids]))
    sync_pipeline_papers(run_dir, papers)
    atomic_write_json(state_dir / "citation_resolutions.json", list(context.database.resolution_records.values()))
    normalized_outline = normalize_outline(outline, collected_papers)
    paper_assignment = paper_assignment_from_outline(normalized_outline, papers)
    sections = sections_from_outline_and_markdown(normalized_outline, markdown_body, context)
    markdown_body = ensure_markdown_citations(markdown_body, sections, list(context.paper_by_id), inline_limit=runtime.inline_citation_limit)
    key_claims = claims_from_sections(sections, request.topic)
    research_gaps = gaps_from_sections(sections, request.topic)
    cited_paper_ids = sorted(collect_cited_paper_ids(markdown_body, sections, key_claims, research_gaps))
    if not cited_paper_ids:
        cited_paper_ids = sorted({paper_id for section in sections for paper_id in section.get("paper_ids", [])})

    result = {
        "schema_version": ENGINE_SCHEMA_VERSION,
        "mode": "integrated_xcientist_survey_agent",
        "generated_at": utc_now(),
        "llm": {
            "model": runtime.model,
            "base_url": runtime.llm_base_url,
            "context_window": runtime.llm_context_window,
        },
        "counts": {
            "seed_papers": len(seed_paper_ids),
            "expanded_papers": len(expanded_paper_ids),
            "collected_papers": len(collected_papers),
            "clusters": len(clustering_result),
            "references": len(references),
        },
        "outline": normalized_outline,
        "paper_assignment": paper_assignment,
        "markdown_body": markdown_body,
        "sections": sections,
        "key_claims": key_claims,
        "research_gaps": research_gaps,
        "cited_paper_ids": cited_paper_ids,
        "xcientist": {
            "collected_paper_ids": collected_papers,
            "failed_paper_ids": sorted(failed),
            "raw_references": [str(reference) for reference in references],
            "relation_graph": relation_graphs_for_json(relation_graph),
            "relation_table": relation_table,
            "evaluation": evaluation,
            "graph_db_path": str(graph_db_path) if graph_db_path else None,
            "internal_state_dir": str(state_dir),
        },
    }
    atomic_write_json(state_dir / "engine_result.json", result)
    write_engine_progress(run_dir, state_dir, "complete", counts=result["counts"])
    return result


def read_json_list(path: Path) -> list[Any]:
    value = read_json(path)
    return value if isinstance(value, list) else []


def relation_graphs_for_json(relation_graph: Any) -> dict[str, list[dict[str, Any]]] | None:
    if not isinstance(relation_graph, dict):
        return None
    serialized: dict[str, list[dict[str, Any]]] = {}
    for cluster_name, graph in relation_graph.items():
        if hasattr(graph, "edges"):
            serialized[str(cluster_name)] = [
                {
                    "source": str(source),
                    "target": str(target),
                    "type": str(data.get("type", "unspecified")),
                    "analysis": str(data.get("analysis", "")),
                }
                for source, target, data in graph.edges(data=True)
            ]
        elif isinstance(graph, list):
            serialized[str(cluster_name)] = [edge for edge in graph if isinstance(edge, dict)]
    return serialized


def relation_graphs_from_json(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    try:
        import networkx as nx
    except Exception:
        return relation_graphs_for_json(value)
    graphs: dict[str, Any] = {}
    for cluster_name, edges in value.items():
        graph = nx.DiGraph()
        if isinstance(edges, list):
            for edge in edges:
                if not isinstance(edge, dict):
                    continue
                source = str(edge.get("source") or "")
                target = str(edge.get("target") or "")
                if not source or not target:
                    continue
                graph.add_edge(
                    source,
                    target,
                    type=str(edge.get("type", "unspecified")),
                    analysis=str(edge.get("analysis", "")),
                )
        graphs[str(cluster_name)] = graph
    return graphs


def sync_pipeline_papers(run_dir: Path, papers: list[dict[str, Any]]) -> None:
    paths = run_paths(run_dir)
    atomic_write_jsonl(paths["papers_jsonl"], papers)
    atomic_write_jsonl(paths["references_jsonl"], [reference_from_paper(paper) for paper in papers])
    paper_index = read_json(paths["paper_index"]) if paths["paper_index"].exists() else {}
    if isinstance(paper_index, dict):
        paper_index["paper_count"] = len(papers)
        paper_index["source_count"] = max(int(paper_index.get("source_count") or 0), len(papers))
        paper_index["abstract_count"] = sum(1 for paper in papers if str(paper.get("abstract") or "").strip())
        paper_index["abstract_coverage"] = paper_index["abstract_count"] / len(papers) if papers else 0
        atomic_write_json(paths["paper_index"], paper_index)


def reference_from_paper(paper: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": paper["id"],
        "title": paper["title"],
        "year": paper.get("year"),
        "venue": paper.get("venue"),
        "authors": paper.get("authors", []),
        "url": paper.get("url"),
        "source": paper.get("source", "survey_agent"),
    }


def clusters_for_artifacts(clustering_result: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for index, cluster in enumerate(clustering_result, start=1):
        if not isinstance(cluster, dict):
            continue
        papers = cluster.get("papers") if isinstance(cluster.get("papers"), list) else []
        paper_ids = [str(paper.get("id")) for paper in papers if isinstance(paper, dict) and paper.get("id") is not None]
        if not paper_ids:
            paper_ids = unique_preserve_order(cluster.get("paper_ids", []))
        name = str(cluster.get("cluster_name") or cluster.get("name") or f"Cluster {index}")
        clusters.append(
            {
                "id": str(cluster.get("cluster_id") or cluster.get("id") or f"cluster-{index}"),
                "name": name,
                "keywords": cluster.get("keywords", []) if isinstance(cluster.get("keywords"), list) else [],
                "paper_ids": paper_ids,
                "summary": str(cluster.get("summary") or cluster.get("description") or name),
            }
        )
    return clusters


def write_engine_progress(run_dir: Path, state_dir: Path, step: str, *, counts: dict[str, Any] | None = None) -> None:
    progress = {"schema_version": ENGINE_SCHEMA_VERSION, "step": step, "updated_at": utc_now(), "counts": counts or {}}
    atomic_write_json(state_dir / "progress.json", progress)
    append_pipeline_event(run_dir, f"survey_agent.{step}", "checkpoint", progress["counts"])
    write_checkpoint(
        run_dir,
        f"survey_agent.{step}",
        {"counts": counts or {}, "mode": "integrated_xcientist_survey_agent"},
        attempted_stage="survey_agent",
    )



def evaluate_survey(config: DictConfig, context: IntegratedSurveyContext, survey: str, references: list[str]) -> dict[str, Any]:
    try:
        judge = Judge(config, context.work_analyzer)
        scores, reasons = judge.evaluate(survey, references)
        evaluation = {"evaluation_version": 2, "scores": scores, "reasons": reasons, "error": None}
    except Exception as error:  # noqa: BLE001 - evaluation should not discard a completed survey.
        evaluation = {"evaluation_version": 2, "scores": {}, "reasons": {}, "error": str(error)}
    state_dir = Path(str(config.BasicInfo.cache_path)).parent
    atomic_write_json(state_dir / "evaluation.json", evaluation)
    return evaluation


def unique_known_paper_ids(values: list[str], paper_by_id: dict[str, dict[str, Any]]) -> list[str]:
    result: list[str] = []
    for value in values:
        paper_id = str(value)
        if paper_id in paper_by_id and paper_id not in result:
            result.append(paper_id)
    return result


def normalize_xlab_markdown(text: str, references: list[str], context: IntegratedSurveyContext) -> str:
    body = strip_reference_list(strip_top_heading(text or ""))
    body = replace_numeric_citations(body, references)
    titles = {}
    for paper_id, paper in context.paper_by_id.items():
        title = " ".join(unicodedata.normalize("NFKC", str(paper.get("title") or "")).casefold().split())
        if title:
            titles.setdefault(title, set()).add(paper_id)

    def cite_explicit_title(match):
        title = " ".join(unicodedata.normalize("NFKC", match.group(1)).casefold().split())
        matches = titles.get(title, set())
        if len(matches) != 1:
            return match.group(0)
        line_start = body.rfind("\n", 0, match.start()) + 1
        line_end = body.find("\n", match.end())
        before = body[line_start:match.start()].strip()
        after = body[match.end():line_end if line_end >= 0 else len(body)].strip()
        if (not before or re.fullmatch(r"#{1,6}", before)) and not after:
            return match.group(0)
        citation = f"[paper:{next(iter(matches))}]"
        return match.group(0) if after.startswith(citation) else match.group(0) + " " + citation

    body = re.sub(r"\*\*([^*\n]+)\*\*", cite_explicit_title, body)
    body = replace_title_citations(body, context)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return body


def strip_top_heading(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].lstrip().startswith("# "):
        return "\n".join(lines[1:]).strip()
    return text.strip()


def strip_reference_list(text: str) -> str:
    patterns = [r"\n\s*##\s+References\b", r"\n\s*References\s*:\s*\n"]
    cut = len(text)
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            cut = min(cut, match.start())
    return text[:cut].strip()


def replace_numeric_citations(text: str, references: list[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        raw = match.group(1)
        paper_ids: list[str] = []
        for value in re.findall(r"\d+", raw):
            index = int(value) - 1
            if 0 <= index < len(references):
                paper_id = str(references[index])
                if paper_id not in paper_ids:
                    paper_ids.append(paper_id)
        if not paper_ids:
            return match.group(0)
        return " ".join(f"[paper:{paper_id}]" for paper_id in paper_ids)

    return re.sub(r"\[((?:\d+\s*[,;]?\s*)+)\]", replace, text)


def replace_title_citations(text: str, context: IntegratedSurveyContext) -> str:
    def replace(match: re.Match[str]) -> str:
        title = match.group(1).strip()
        try:
            paper_id, _matched_title, _score = context.database.resolve_title_to_paper_id(
                title,
                min_title_similarity=context.config.ModuleInfo.SurveyGenerator.valid_title_min_similarity,
            )
        except Exception:
            raise ValueError(f"Unresolved citation remained during Markdown export: {title}")
        return f"[paper:{paper_id}]"

    return re.sub(r"<([^<>]+)>", replace, text)


def normalize_outline(outline: dict[str, Any], collected_papers: list[str]) -> dict[str, Any]:
    sections: list[dict[str, Any]] = []
    for index, section in enumerate(outline.get("sections", []) if isinstance(outline, dict) else [], start=1):
        if not isinstance(section, dict):
            continue
        subsections: list[dict[str, Any]] = []
        for sub_index, subsection in enumerate(section.get("subsections", []) if isinstance(section.get("subsections"), list) else [], start=1):
            if not isinstance(subsection, dict):
                continue
            subsections.append(
                {
                    "id": f"section-{index}-subsection-{sub_index}",
                    "title": str(subsection.get("title") or f"Subsection {index}.{sub_index}"),
                    "description": str(subsection.get("description") or ""),
                    "paper_ids": unique_preserve_order(subsection.get("papers_to_use", [])),
                }
            )
        section_ids = unique_preserve_order(section.get("papers_to_use", []))
        for subsection in subsections:
            section_ids.extend(paper_id for paper_id in subsection["paper_ids"] if paper_id not in section_ids)
        sections.append(
            {
                "id": f"section-{index}",
                "title": str(section.get("title") or f"Section {index}"),
                "description": str(section.get("description") or ""),
                "paper_ids": section_ids,
                "subsections": subsections,
            }
        )
    assigned = {paper_id for section in sections for paper_id in section["paper_ids"]}
    exclusions = {paper["paper_id"]: paper for paper in outline.get("excluded_papers", []) if paper.get("paper_id") not in assigned}
    return {"title": str(outline.get("title") or "Literature Survey"), "sections": sections, "excluded_papers": list(exclusions.values())}


def unique_preserve_order(values: Any) -> list[str]:
    result: list[str] = []
    if not isinstance(values, list):
        return result
    for value in values:
        paper_id = str(value).strip()
        if paper_id and paper_id not in result:
            result.append(paper_id)
    return result


def paper_assignment_from_outline(outline: dict[str, Any], papers: list[dict[str, Any]]) -> dict[str, Any]:
    titles = {str(paper["id"]): str(paper.get("title") or paper["id"]) for paper in papers}
    exclusions = {paper["paper_id"]: paper.get("exclusion_reason") for paper in outline.get("excluded_papers", [])}
    by_paper: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"section_ids": [], "subsection_ids": []})
    for section in outline.get("sections", []):
        section_id = str(section.get("id"))
        for paper_id in section.get("paper_ids", []):
            if section_id not in by_paper[str(paper_id)]["section_ids"]:
                by_paper[str(paper_id)]["section_ids"].append(section_id)
        for subsection in section.get("subsections", []):
            subsection_id = str(subsection.get("id"))
            for paper_id in subsection.get("paper_ids", []):
                if subsection_id not in by_paper[str(paper_id)]["subsection_ids"]:
                    by_paper[str(paper_id)]["subsection_ids"].append(subsection_id)
    assignments = []
    for paper_id, title in titles.items():
        item = by_paper[paper_id]
        assignments.append(
            {
                "paper_id": paper_id,
                "paper_title": title,
                "section_ids": item["section_ids"],
                "subsection_ids": item["subsection_ids"],
                "exclusion_reason": exclusions.get(paper_id),
            }
        )
    return {"assignments": assignments}


def sections_from_outline_and_markdown(outline: dict[str, Any], markdown_body: str, context: IntegratedSurveyContext) -> list[dict[str, Any]]:
    body_by_title = section_bodies_by_title(markdown_body)
    sections: list[dict[str, Any]] = []
    all_ids = set(context.paper_by_id)
    for index, section in enumerate(outline.get("sections", []), start=1):
        title = str(section.get("title") or f"Section {index}")
        content = body_by_title.get(normalize_heading_title(title), "").strip()
        if not content:
            raise ValueError(f"Generated survey is missing the outline section: {title}")
        paper_ids = ids_from_text(content, all_ids)
        sections.append(
            {
                "id": str(section.get("id") or f"section-{index}"),
                "title": title,
                "summary": first_sentence(content) or str(section.get("description") or title),
                "paper_ids": paper_ids,
                "content_markdown": content,
                "evidence": first_sentence(content),
            }
        )
    if not sections:
        raise ValueError("Generated survey has no outline sections")
    return sections


def section_bodies_by_title(markdown_body: str) -> dict[str, str]:
    result: dict[str, str] = {}
    lines = markdown_body.splitlines()
    headings = []
    for index, line in enumerate(lines):
        match = re.match(r"^(#{2,6})\s+(.*)$", line)
        if match:
            headings.append((index, len(match.group(1)), normalize_heading_title(match.group(2))))
    for position, (start, level, title) in enumerate(headings):
        end = next((index for index, next_level, _ in headings[position + 1:] if next_level <= level), len(lines))
        result[title] = "\n".join(lines[start:end]).strip()
    return result


def normalize_heading_title(title: str) -> str:
    title = re.sub(r"^\d+(?:\.\d+)*\s*[.)-]?\s*", "", str(title).strip())
    return " ".join(re.findall(r"[A-Za-z0-9]+", title.lower()))


def ensure_markdown_citations(markdown_body: str, sections: list[dict[str, Any]], paper_ids: list[str], inline_limit: int) -> str:
    if "[paper:" in markdown_body:
        return markdown_body
    raise ValueError("Generated survey body has no traceable citations")


def claims_from_sections(sections: list[dict[str, Any]], topic: str) -> list[dict[str, Any]]:
    return [
        {
            "id": f"claim-{index}",
            "claim": section.get("summary") or f"Integrated SurveyAgent identifies {section.get('title')} as a recurring theme for {topic}.",
            "paper_ids": section.get("paper_ids", []),
            "confidence": "medium",
        }
        for index, section in enumerate(sections, start=1)
        if section.get("paper_ids")
    ]


def gaps_from_sections(sections: list[dict[str, Any]], topic: str) -> list[dict[str, Any]]:
    gaps = []
    cues = re.compile(r"\b(?:open (?:challenges?|questions?|problems?)|future (?:work|research|directions?)|remains? (?:unclear|unresolved|challenging)|research gaps?|limitations?)\b", re.IGNORECASE)
    for section in sections:
        allowed = set(section.get("paper_ids", []))
        for paragraph in re.split(r"\n\s*\n", section.get("content_markdown", "")):
            if not cues.search(paragraph) or paragraph.lstrip().startswith("#"):
                continue
            ids = ids_from_text(paragraph, allowed)
            if ids:
                gaps.append({"id": f"gap-{len(gaps) + 1}", "gap": paragraph.strip(), "paper_ids": ids})
    return gaps


def collect_cited_paper_ids(markdown_body: str, sections: list[dict[str, Any]], claims: list[dict[str, Any]], gaps: list[dict[str, Any]]) -> set[str]:
    used = set(re.findall(r"\[paper:([^\]]+)\]", markdown_body or ""))
    for collection in (sections, claims, gaps):
        for item in collection:
            used.update(str(paper_id) for paper_id in item.get("paper_ids", []) if paper_id)
    return used


def ids_from_text(text: str, allowed: set[str]) -> list[str]:
    result: list[str] = []
    for paper_id in re.findall(r"\[paper:([^\]]+)\]", text or ""):
        if paper_id in allowed and paper_id not in result:
            result.append(paper_id)
    return result


def first_sentence(text: str) -> str:
    prose = "\n".join(line for line in (text or "").splitlines() if not re.match(r"^\s*(?:#{1,6}\s|\|)", line))
    stripped = " ".join(prose.split())
    if not stripped:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", stripped, maxsplit=1)
    return parts[0][:500]
