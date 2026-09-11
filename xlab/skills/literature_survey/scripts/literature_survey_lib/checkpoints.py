from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import atomic_write_json, read_json, run_paths, stable_signature, utc_now


STAGE_ORDER = [
    "ingest_sources",
    "ingest_graph_context",
    "cluster",
    "survey_agent",
    "citation_traces",
    "render_artifacts",
]


def write_checkpoint(
    run_dir: Path,
    phase: str,
    data: dict[str, Any] | None = None,
    *,
    completed_stage: str | None = None,
    attempted_stage: str | None = None,
    blocked_stage: str | None = None,
    request_signature: str | None = None,
    runtime_config_signature: str | None = None,
    source_signatures: dict[str, str | None] | None = None,
    warnings: list[str] | None = None,
    last_error: str | None = None,
) -> dict[str, Any]:
    previous = read_checkpoint(run_dir) or {}
    same_signature = True
    if request_signature and previous.get("request_signature") not in {None, request_signature}:
        same_signature = False
    if runtime_config_signature and previous.get("runtime_config_signature") not in {None, runtime_config_signature}:
        same_signature = False
    previous_sources = previous.get("source_signatures")
    if (
        source_signatures is not None
        and previous_sources is not None
        and stable_signature(previous_sources) != stable_signature(source_signatures)
    ):
        same_signature = False

    completed = stage_list(previous, "completed_stages") if same_signature else []
    attempted = stage_list(previous, "attempted_stages") if same_signature else []
    blocked = stage_list(previous, "blocked_stages") if same_signature else []

    stage_to_mark = completed_stage or blocked_stage or attempted_stage
    if stage_to_mark:
        completed = prune_conflicting_stages(completed, stage_to_mark)
        attempted = prune_conflicting_stages(attempted, stage_to_mark)
        blocked = prune_conflicting_stages(blocked, stage_to_mark)

    if attempted_stage:
        attempted = append_unique(attempted, attempted_stage)
    if completed_stage:
        attempted = append_unique(attempted, completed_stage)
        completed = append_unique(completed, completed_stage)
        blocked = [stage for stage in blocked if stage != completed_stage]
    if blocked_stage:
        attempted = append_unique(attempted, blocked_stage)
        blocked = append_unique(blocked, blocked_stage)
        completed = remove_stage_and_downstream(completed, blocked_stage)

    merged_warnings = list(previous.get("warnings", [])) if same_signature and isinstance(previous.get("warnings"), list) else []
    for warning in warnings or []:
        if warning not in merged_warnings:
            merged_warnings.append(warning)
    checkpoint = {
        "schema_version": "xlab.literature_survey.checkpoint.v1",
        "phase": phase,
        "current_phase": phase,
        "updated_at": utc_now(),
        "resumable": True,
        "completed_stages": completed,
        "attempted_stages": attempted,
        "blocked_stages": blocked,
        "request_signature": request_signature or (previous.get("request_signature") if same_signature else None),
        "runtime_config_signature": runtime_config_signature or (previous.get("runtime_config_signature") if same_signature else None),
        "source_signatures": source_signatures or (previous.get("source_signatures") if same_signature else None) or {},
        "outputs": stage_outputs(run_dir),
        "counts": data.get("counts", {}) if isinstance(data, dict) else (previous.get("counts", {}) if same_signature else {}),
        "warnings": merged_warnings,
        "last_error": last_error if last_error is not None else (previous.get("last_error") if same_signature else None),
        "data": data or {},
    }
    atomic_write_json(run_paths(run_dir)["checkpoint"], checkpoint)
    return checkpoint


def stage_list(checkpoint: dict[str, Any], key: str) -> list[str]:
    value = checkpoint.get(key)
    return [str(stage) for stage in value if stage is not None] if isinstance(value, list) else []


def append_unique(stages: list[str], stage: str) -> list[str]:
    if stage not in stages:
        stages.append(stage)
    return stages


def prune_conflicting_stages(completed: list[str], completed_stage: str) -> list[str]:
    if completed_stage not in STAGE_ORDER:
        return completed
    stage_index = STAGE_ORDER.index(completed_stage)
    return [stage for stage in completed if stage not in STAGE_ORDER[stage_index + 1 :]]


def remove_stage_and_downstream(stages: list[str], stage: str) -> list[str]:
    if stage not in STAGE_ORDER:
        return [candidate for candidate in stages if candidate != stage]
    stage_index = STAGE_ORDER.index(stage)
    return [candidate for candidate in stages if candidate not in STAGE_ORDER[stage_index:]]


def read_checkpoint(run_dir: Path) -> dict[str, Any] | None:
    path = run_paths(run_dir)["checkpoint"]
    if not path.exists():
        return None
    value = read_json(path)
    return value if isinstance(value, dict) else None


def checkpoint_signature(
    request: dict[str, Any],
    runtime_config: dict[str, Any],
    source_signatures: dict[str, str | None] | None = None,
) -> tuple[str, str, str]:
    return stable_signature(request), stable_signature(runtime_config), stable_signature(source_signatures or {})


def stage_completed(
    run_dir: Path,
    stage: str,
    required_paths: list[Path],
    *,
    request_signature: str,
    runtime_config_signature: str,
    source_signature: str | None = None,
) -> bool:
    checkpoint = read_checkpoint(run_dir)
    if not checkpoint:
        return False
    completed = checkpoint.get("completed_stages") if isinstance(checkpoint.get("completed_stages"), list) else []
    if stage not in completed:
        return False
    if checkpoint.get("request_signature") != request_signature:
        return False
    if checkpoint.get("runtime_config_signature") != runtime_config_signature:
        return False
    if source_signature and stable_signature(checkpoint.get("source_signatures") or {}) != source_signature:
        return False
    if not all(path.exists() for path in required_paths):
        return False
    if stage not in STAGE_ORDER:
        return True
    stage_index = STAGE_ORDER.index(stage)
    downstream_paths = {
        "ingest_sources": ["graph_context", "clusters_json", "chronology_json", "traces_jsonl", "survey_md", "survey_json", "citations_json"],
        "ingest_graph_context": ["clusters_json", "chronology_json", "traces_jsonl", "survey_md", "survey_json", "citations_json"],
        "cluster": ["survey_agent_outline", "survey_agent_assignment", "survey_agent_draft", "survey_agent_result", "traces_jsonl", "survey_md", "survey_json", "citations_json"],
        "survey_agent": ["traces_jsonl", "survey_md", "survey_json", "citations_json"],
        "citation_traces": ["survey_md", "survey_json", "citations_json"],
    }
    paths = run_paths(run_dir)
    for earlier_stage in STAGE_ORDER[:stage_index]:
        if earlier_stage not in completed:
            continue
        for key in downstream_paths.get(earlier_stage, []):
            if not paths[key].exists():
                return False
    return True


def stage_outputs(run_dir: Path) -> dict[str, str]:
    paths = run_paths(run_dir)
    keys = [
        "papers_jsonl",
        "paper_index",
        "graph_context",
        "clusters_json",
        "chronology_json",
        "survey_agent_outline",
        "survey_agent_assignment",
        "survey_agent_draft",
        "survey_agent_result",
        "sections_jsonl",
        "claims_jsonl",
        "gaps_jsonl",
        "references_jsonl",
        "traces_jsonl",
        "survey_md",
        "survey_json",
        "citations_json",
        "report_json",
        "manifest_json",
    ]
    return {key: str(paths[key]) for key in keys if paths[key].exists()}
