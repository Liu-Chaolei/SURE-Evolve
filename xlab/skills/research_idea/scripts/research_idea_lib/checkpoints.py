from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .common import file_digest, read_json, run_paths, stable_signature, utc_now, write_portable_json
from .research_idea_spec import STATE_LAYOUT_ID


CHECKPOINT_SCHEMA_VERSION = "xlab.research_idea.checkpoint.v2"
_UNSET_LAST_ERROR = object()
STAGE_ORDER = [
    "ingest_survey",
    "organize_context",
    "resource_preflight",
    "knowledge_acquisition",
    "advanced_analysis",
    "re_analysis_replan",
    "idea_generation",
    "materialize_xlab_artifact",
    "audit",
]

_STAGE_OUTPUT_KEYS = {
    "ingest_survey": ("survey_context",),
    "organize_context": ("context",),
    "resource_preflight": ("resource_preflight",),
    "knowledge_acquisition": ("workflow_retrieval",),
    "advanced_analysis": ("workflow_analysis",),
    "re_analysis_replan": ("workflow_analysis",),
    "idea_generation": ("workflow_ideation", "workflow_persistence"),
    "materialize_xlab_artifact": (
        "workflow_artifact",
        "workflow_run",
        "workflow_retrieval",
        "workflow_analysis",
        "workflow_ideation",
        "workflow_persistence",
        "idea_result_json",
        "research_idea_json",
        "idea_trace_json",
    ),
    "audit": ("report_json", "manifest_json"),
}
_SECRET_CAPABILITY_KEYS = {
    "api_key",
    "api_key_set",
    "has_api_key",
    "llm_enabled",
    "openai_api_key",
    "openai_api_key_set",
    "provider_available",
    "secret",
    "secret_available",
    "token",
}


def canonical_stage(stage: str) -> str:
    return stage


def stage_dependency_signature(stage: str, dependencies: Mapping[str, Any] | None = None) -> str:
    """Return a semantic signature for one stage, excluding live secret capability."""
    return stable_signature({"stage": canonical_stage(stage), "dependencies": _semantic_value(dependencies or {})})


def write_checkpoint(
    run_dir: Path,
    phase: str,
    data: dict[str, Any] | None = None,
    *,
    completed_stage: str | None = None,
    request_signature: str | None = None,
    runtime_config_signature: str | None = None,
    source_signatures: dict[str, str | None] | None = None,
    warnings: list[str] | None = None,
    last_error: str | None | object = _UNSET_LAST_ERROR,
    dependency_signature: str | None = None,
    dependencies: Mapping[str, Any] | None = None,
    output_paths: Mapping[str, Path] | Iterable[Path] | None = None,
) -> dict[str, Any]:
    """Write a v2 checkpoint while retaining the v1 caller surface.

    New callers can provide a stage dependency signature (or raw dependencies) and
    explicit outputs. Existing pipeline callers are assigned a stage-scoped
    compatibility signature and the known outputs for that stage.
    """
    previous = read_checkpoint(run_dir) or {}
    phase = canonical_stage(phase)
    completed_stage = canonical_stage(completed_stage) if completed_stage else None
    completed = _canonical_stages(previous.get("completed_stages"))
    stages = dict(previous.get("stages")) if isinstance(previous.get("stages"), dict) else {}

    if completed_stage:
        completed = prune_conflicting_stages(completed, completed_stage)
        stages = {stage: record for stage, record in stages.items() if stage in completed}
        if completed_stage not in completed:
            completed.append(completed_stage)

        signature = dependency_signature
        if signature is None and dependencies is not None:
            signature = stage_dependency_signature(completed_stage, dependencies)
        if signature is None:
            signature = _compatibility_dependency_signature(
                completed_stage,
                request_signature=request_signature,
                runtime_config_signature=runtime_config_signature,
                source_signature=_source_signature(source_signatures),
            )
        stages[completed_stage] = {
            "completed_at": utc_now(),
            "dependency_signature": signature,
            "output_digests": _output_digests(run_dir, completed_stage, output_paths),
        }

    merged_warnings = list(previous.get("warnings", [])) if isinstance(previous.get("warnings"), list) else []
    for warning in warnings or []:
        if warning not in merged_warnings:
            merged_warnings.append(warning)

    stored_sources = _semantic_value(source_signatures) if source_signatures is not None else previous.get("source_signatures")
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "state_layout_id": STATE_LAYOUT_ID,
        "phase": phase,
        "current_phase": phase,
        "updated_at": utc_now(),
        "resumable": True,
        "completed_stages": completed,
        "request_signature": request_signature if request_signature is not None else previous.get("request_signature"),
        "runtime_config_signature": runtime_config_signature if runtime_config_signature is not None else previous.get("runtime_config_signature"),
        "source_signatures": stored_sources if isinstance(stored_sources, dict) else {},
        "stages": stages,
        "outputs": stage_outputs(run_dir),
        "counts": data.get("counts", {}) if isinstance(data, dict) else previous.get("counts", {}),
        "warnings": merged_warnings,
        "last_error": previous.get("last_error") if last_error is _UNSET_LAST_ERROR else last_error,
        "data": data or {},
    }
    if isinstance(previous.get("migration"), dict):
        checkpoint["migration"] = previous["migration"]
    write_portable_json(run_paths(run_dir)["checkpoint"], checkpoint)
    return checkpoint


def prune_conflicting_stages(completed: list[str], completed_stage: str) -> list[str]:
    completed_stage = canonical_stage(completed_stage)
    completed = _canonical_stages(completed)
    if completed_stage not in STAGE_ORDER:
        return completed
    stage_index = STAGE_ORDER.index(completed_stage)
    return [stage for stage in completed if stage not in STAGE_ORDER[stage_index + 1 :]]


def read_checkpoint(run_dir: Path) -> dict[str, Any] | None:
    path = run_paths(run_dir)["checkpoint"]
    if not path.exists():
        return None
    value = read_json(path)
    if not isinstance(value, dict):
        return None
    if value.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        return None
    if value.get("state_layout_id") != STATE_LAYOUT_ID:
        return None
    return value


def checkpoint_signature(
    request: dict[str, Any],
    runtime_config: dict[str, Any],
    source_signatures: dict[str, str | None] | None = None,
) -> tuple[str, str, str]:
    """Compatibility signatures for current pipeline callers.

    Live provider/secret capability is deliberately excluded: it must be probed
    at execution time rather than becoming resumable semantic state.
    """
    return (
        stable_signature(_semantic_value(request)),
        stable_signature(_semantic_value(runtime_config)),
        _source_signature(source_signatures),
    )


def stage_completed(
    run_dir: Path,
    stage: str,
    required_paths: list[Path],
    *,
    request_signature: str | None = None,
    runtime_config_signature: str | None = None,
    source_signature: str | None = None,
    dependency_signature: str | None = None,
    dependencies: Mapping[str, Any] | None = None,
) -> bool:
    """Validate a completed stage's dependency signature and output bytes."""
    checkpoint = read_checkpoint(run_dir)
    if (
        not checkpoint
        or checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or checkpoint.get("state_layout_id") != STATE_LAYOUT_ID
    ):
        return False
    stage = canonical_stage(stage)
    completed = _canonical_stages(checkpoint.get("completed_stages"))
    if stage not in completed:
        return False

    expected_signature = dependency_signature
    if expected_signature is None and dependencies is not None:
        expected_signature = stage_dependency_signature(stage, dependencies)
    if expected_signature is None:
        expected_signature = _compatibility_dependency_signature(
            stage,
            request_signature=request_signature,
            runtime_config_signature=runtime_config_signature,
            source_signature=source_signature,
        )
    stages = checkpoint.get("stages") if isinstance(checkpoint.get("stages"), dict) else {}
    record = stages.get(stage) if isinstance(stages.get(stage), dict) else {}
    if record.get("dependency_signature") != expected_signature:
        return False
    stored_digests = record.get("output_digests") if isinstance(record.get("output_digests"), dict) else {}
    for path in required_paths:
        relative = _run_relative_path(run_dir, path)
        if relative is None:
            return False
        digest = file_digest(path)
        if digest is None or stored_digests.get(relative) != digest:
            return False
    return True


def stage_outputs(run_dir: Path) -> dict[str, str]:
    """Return existing known outputs as structurally contained run-relative paths."""
    paths = run_paths(run_dir)
    keys = [
        "survey_context",
        "context",
        "resource_preflight",
        "workflow_artifact",
        "workflow_run",
        "workflow_retrieval",
        "workflow_analysis",
        "workflow_ideation",
        "workflow_persistence",
        "workflow_keynote",
        "idea_result_json",
        "research_idea_json",
        "idea_trace_json",
        "report_json",
        "manifest_json",
    ]
    outputs: dict[str, str] = {}
    for key in keys:
        path = paths[key]
        if not path.exists():
            continue
        relative = _run_relative_path(run_dir, path)
        if relative is not None:
            outputs[key] = relative
    return outputs


def _compatibility_dependency_signature(
    stage: str,
    *,
    request_signature: str | None,
    runtime_config_signature: str | None,
    source_signature: str | None,
) -> str:
    stage = canonical_stage(stage)
    dependencies: dict[str, str | None] = {"request": request_signature, "source": source_signature}
    if stage != "ingest_survey":
        dependencies["runtime_config"] = runtime_config_signature
    return stage_dependency_signature(stage, dependencies)


def _source_signature(source_signatures: Mapping[str, Any] | None) -> str:
    return stable_signature(_semantic_value(source_signatures or {}))


def _semantic_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _semantic_value(item)
            for key, item in value.items()
            if str(key).lower() not in _SECRET_CAPABILITY_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_semantic_value(item) for item in value]
    return value


def _canonical_stages(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        stage = canonical_stage(str(item))
        if stage not in result:
            result.append(stage)
    return result


def _run_relative_path(run_dir: Path, path: Path | str) -> str | None:
    root = run_dir.resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    try:
        relative = candidate.resolve().relative_to(root)
    except (OSError, ValueError):
        return None
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        return None
    return relative.as_posix()


def _explicit_output_paths(output_paths: Mapping[str, Path] | Iterable[Path]) -> list[Path]:
    if isinstance(output_paths, Mapping):
        return [Path(path) for path in output_paths.values()]
    return [Path(path) for path in output_paths]


def _output_digests(
    run_dir: Path,
    stage: str,
    output_paths: Mapping[str, Path] | Iterable[Path] | None,
) -> dict[str, str]:
    paths = run_paths(run_dir)
    candidates = (
        _explicit_output_paths(output_paths)
        if output_paths is not None
        else [paths[key] for key in _STAGE_OUTPUT_KEYS.get(canonical_stage(stage), ())]
    )
    result: dict[str, str] = {}
    for path in candidates:
        relative = _run_relative_path(run_dir, path)
        digest = file_digest(path)
        if relative is not None and digest is not None:
            result[relative] = digest
    return result
