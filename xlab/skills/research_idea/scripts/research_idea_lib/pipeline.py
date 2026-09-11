from __future__ import annotations

import importlib.util
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any

from .artifact_mapping import incomplete_idea_result, incomplete_research_idea, map_to_research_idea
from .checkpoints import checkpoint_signature, read_checkpoint, stage_completed, write_checkpoint
from .common import append_diagnostic, append_pipeline_event, atomic_write_json, ensure_run_directories, read_json, run_paths, stable_signature, write_portable_json
from .config import RuntimeConfig, provider_api_key
from .inputs import IdeaRequest
from .research_idea_artifacts import final_idea_result, retrieval_namespace
from .research_idea_spec import ALGORITHM_ID, ALGORITHM_SPEC_VERSION, IMPLEMENTATION_PROVENANCE
from .llm import llm_status
from .manifest import finalize_run
from .providers import OpenAICompatibleConfig, OpenAICompatibleProvider
from .survey_repository import SurveyArtifactRepository

SKILL_VERSION = "1.0.0"
ALGORITHM_IMPLEMENTATION = {
    "name": "xlab-native-research-idea",
    "algorithm_spec": ALGORITHM_ID,
    "algorithm_spec_version": ALGORITHM_SPEC_VERSION,
    "provenance": dict(IMPLEMENTATION_PROVENANCE),
    "success_policy": "xlab.research_idea.success.v1",
    "resource_profile": "xlab.research_idea.evidence.v1",
    "lightweight_success_paths": False,
}


def run_pipeline(
    *,
    cwd: Path,
    run_dir: Path,
    run_id: str,
    request: IdeaRequest,
    runtime: RuntimeConfig,
    requested_status: str | None = None,
) -> dict[str, Any]:
    ensure_run_directories(run_dir)
    request_json = request.to_json(cwd)
    runtime_json = runtime_contract_json(runtime.to_json())
    completed_result = completed_success(run_dir, run_id, requested_status=requested_status)
    if completed_result is not None:
        return completed_result
    repository = ingest_survey(cwd, run_dir, request, runtime, request_json, runtime_json)
    source_context = organize_context(cwd, run_dir, request, runtime, request_json, runtime_json, repository)
    validation = repository.validation()
    if not validation.get("passed"):
        reason = "; ".join(str(error) for error in validation.get("blocking_errors", [])) or "Survey validation failed."
        append_diagnostic(run_dir, "error", reason, code="survey_validation_failed")
        materialize_incomplete(cwd, run_dir, run_id, request, runtime, request_json, source_context, reason, requested_status=requested_status)
        return audit_and_manifest(cwd, run_dir, run_id, request_json, requested_status=requested_status)
    if not runtime.provider_available:
        status = llm_status(runtime)
        reason = str(status.get("reason") or "research idea provider configuration is unavailable.")
        append_diagnostic(run_dir, "error", reason, code="provider_unavailable")
        materialize_incomplete(cwd, run_dir, run_id, request, runtime, request_json, source_context, reason, requested_status=requested_status)
        return audit_and_manifest(cwd, run_dir, run_id, request_json, requested_status=requested_status)
    preflight = resource_preflight(cwd, run_dir, request, runtime, request_json, runtime_json, repository)
    if not preflight.get("passed"):
        details = "; ".join(str(error) for error in preflight.get("blocking_errors", []))
        reason = "Resource preflight failed: " + (details or "The research idea evidence and runtime resources are incomplete.")
        append_diagnostic(run_dir, "error", reason, code="resource_preflight_failed")
        materialize_incomplete(cwd, run_dir, run_id, request, runtime, request_json, source_context, reason, requested_status=requested_status)
        return audit_and_manifest(cwd, run_dir, run_id, request_json, requested_status=requested_status)
    try:
        run_research_idea(cwd, run_dir, run_id, request, runtime, request_json, runtime_json, repository, source_context)
    except Exception as error:  # noqa: BLE001 - preserve resumable diagnostics instead of crashing the whole skill.
        reason = f"research idea generation failed: {error}"
        append_diagnostic(run_dir, "error", reason, code="research_idea_failed")
        materialize_incomplete(cwd, run_dir, run_id, request, runtime, request_json, source_context, reason, requested_status=requested_status)
    return audit_and_manifest(cwd, run_dir, run_id, request_json, requested_status=requested_status)


def completed_success(
    run_dir: Path,
    run_id: str,
    *,
    requested_status: str | None,
) -> dict[str, Any] | None:
    if requested_status == "incomplete":
        return None
    checkpoint = read_checkpoint(run_dir)
    completed = checkpoint.get("completed_stages") if isinstance(checkpoint, dict) else []
    paths = run_paths(run_dir)
    if "audit" not in completed or not all(
        paths[key].is_file()
        for key in ("idea_result_json", "research_idea_json", "idea_trace_json", "report_json", "manifest_json")
    ):
        return None
    manifest = read_json(paths["manifest_json"])
    report = read_json(paths["report_json"])
    if not (
        isinstance(manifest, dict)
        and manifest.get("run_id") == run_id
        and manifest.get("status") == "success"
        and isinstance(report, dict)
        and report.get("passed") is True
    ):
        return None
    return {"manifest": manifest, "report": report, "status": "success"}


def ingest_survey(
    cwd: Path,
    run_dir: Path,
    request: IdeaRequest,
    runtime: RuntimeConfig,
    request_json: dict[str, Any],
    runtime_json: dict[str, Any],
) -> SurveyArtifactRepository:
    paths = run_paths(run_dir)
    repository = SurveyArtifactRepository.from_request(request, cwd, model_cache_root=run_paths(run_dir)["memory"] / "resource-cache")
    request_signature, runtime_signature, source_signature = checkpoint_signature(request_json, runtime_json, repository.source.signatures)
    if stage_completed(
        run_dir,
        "ingest_survey",
        [paths["survey_context"]],
        request_signature=request_signature,
        runtime_config_signature=runtime_signature,
        source_signature=source_signature,
    ):
        append_pipeline_event(run_dir, "ingest_survey", "skipped")
        return repository
    validation = repository.validation()
    survey_context = repository.source_context(request)
    survey_context["validation"] = validation
    atomic_write_json(paths["survey_context"], survey_context)
    for warning in validation.get("warnings", []):
        append_diagnostic(run_dir, "warning", str(warning))
    for blocker in validation.get("blocking_errors", []):
        append_diagnostic(run_dir, "error", str(blocker), code="survey_validation")
    write_checkpoint(
        run_dir,
        "ingest_survey",
        {"counts": {"references": len(repository.references), "evidence": len(repository.evidence_items)}, "validation": validation},
        completed_stage="ingest_survey",
        request_signature=request_signature,
        runtime_config_signature=runtime_signature,
        source_signatures=repository.source.signatures,
        warnings=[str(warning) for warning in validation.get("warnings", [])],
        last_error=None if validation.get("passed") else "; ".join(str(error) for error in validation.get("blocking_errors", [])),
    )
    append_pipeline_event(run_dir, "ingest_survey", "success" if validation.get("passed") else "blocked", {"references": len(repository.references), "evidence": len(repository.evidence_items)})
    return repository


def organize_context(
    cwd: Path,
    run_dir: Path,
    request: IdeaRequest,
    runtime: RuntimeConfig,
    request_json: dict[str, Any],
    runtime_json: dict[str, Any],
    repository: SurveyArtifactRepository,
) -> dict[str, Any]:
    del cwd
    paths = run_paths(run_dir)
    request_signature, runtime_signature, source_signature = checkpoint_signature(request_json, runtime_json, repository.source.signatures)
    if stage_completed(
        run_dir,
        "organize_context",
        [paths["context"]],
        request_signature=request_signature,
        runtime_config_signature=runtime_signature,
        source_signature=source_signature,
    ):
        append_pipeline_event(run_dir, "organize_context", "skipped")
        return read_json(paths["context"])
    source_context = repository.source_context(request)
    context = {
        "schema_version": "xlab.research_idea.context.v1",
        "topic": request.topic or repository.topic,
        "mature_idea": request.mature_idea,
        "refinement_scope": request.refinement_scope,
        "discussion": request.discussion,
        "experiment_feedback": request.experiment_feedback,
        "workflow": ["advanced_analysis", "re_analysis_replan", "idea_generation"] if request.experiment_feedback else ["knowledge_acquisition", "advanced_analysis", "idea_generation"],
        "contract_mode": bool(request.mature_idea),
        "llm": llm_status(runtime),
        "source_context": source_context,
    }
    atomic_write_json(paths["context"], context)
    write_checkpoint(
        run_dir,
        "organize_context",
        {"counts": {"rag_hits": len(source_context.get("selected_evidence", []))}, "workflow": context["workflow"]},
        completed_stage="organize_context",
        request_signature=request_signature,
        runtime_config_signature=runtime_signature,
        source_signatures=repository.source.signatures,
    )
    append_pipeline_event(run_dir, "organize_context", "success", {"workflow": context["workflow"]})
    return source_context


def resource_preflight(
    cwd: Path,
    run_dir: Path,
    request: IdeaRequest,
    runtime: RuntimeConfig,
    request_json: dict[str, Any],
    runtime_json: dict[str, Any],
    repository: SurveyArtifactRepository,
) -> dict[str, Any]:
    del cwd, request
    paths = run_paths(run_dir)
    request_signature, runtime_signature, source_signature = checkpoint_signature(request_json, runtime_json, repository.source.signatures)
    if stage_completed(
        run_dir,
        "resource_preflight",
        [paths["resource_preflight"]],
        request_signature=request_signature,
        runtime_config_signature=runtime_signature,
        source_signature=source_signature,
    ):
        append_pipeline_event(run_dir, "resource_preflight", "skipped")
        value = read_json(paths["resource_preflight"])
        return value if isinstance(value, dict) else {"passed": False, "blocking_errors": ["Stored parity preflight is not a JSON object."]}

    checks: dict[str, bool] = {}
    blockers: list[str] = []
    warnings: list[str] = []
    resolution = repository.source.resources

    checks["resource_manifest_declared"] = resolution.manifest_path is not None
    checks["resource_bundle_compatible"] = resolution.passed
    blockers.extend(resolution.blocking_errors)

    checks["provider_available"] = runtime.provider_available
    if not checks["provider_available"]:
        blockers.append("OPENAI_API_KEY is not set; full research idea provider execution cannot run.")

    checks["required_python_modules"] = required_python_modules_available()
    if not checks["required_python_modules"]:
        blockers.append("The package-native research idea runtime modules are incomplete or unavailable.")

    graph_db = repository.source.graph_db_path
    checks["graph_db_present"] = path_is_file(graph_db)
    checks["graph_db_has_nodes"] = sqlite_table_has_rows(graph_db, "nodes") if checks["graph_db_present"] else False
    if resolution.passed:
        if not checks["graph_db_present"]:
            blockers.append("Validated graph resource must contain a readable graph database file at its declared resource root.")
        elif not checks["graph_db_has_nodes"]:
            blockers.append("Paper graph database must contain a readable nodes table before research idea can run.")

    component_index_dir = repository.source.component_index_dir
    checks["component_index_dir_present"] = path_is_dir(component_index_dir)
    checks["component_faiss_present"] = path_is_file(component_index_dir / "faiss.index" if component_index_dir else None)
    checks["component_metadata_present"] = path_is_file(component_index_dir / "meta.json" if component_index_dir else None)
    checks["component_metadata_nonempty"] = component_metadata_nonempty(component_index_dir / "meta.json" if component_index_dir else None) if checks["component_metadata_present"] else False
    if resolution.passed:
        if not (checks["component_index_dir_present"] and checks["component_faiss_present"] and checks["component_metadata_present"]):
            blockers.append("Validated component-index resource must contain faiss.index and meta.json.")
        elif not checks["component_metadata_nonempty"]:
            blockers.append("Core-component metadata must contain indexed component records.")

    outcome_model = repository.source.outcome_model_path
    component_model = repository.source.component_model_path
    checks["outcome_sentence_transformer_present"] = path_exists(outcome_model)
    checks["component_novelty_model_present"] = path_exists(component_model)
    if resolution.passed and not checks["outcome_sentence_transformer_present"]:
        blockers.append("OutcomeRAG model descriptor did not resolve through the immutable XLab model cache.")
    if resolution.passed and not checks["component_novelty_model_present"]:
        blockers.append("Component novelty model descriptor did not resolve through the immutable XLab model cache.")

    checks["keynote_cache_present"] = path_exists(repository.source.keynote_cache_path) or survey_references_have_keynotes(repository.references)
    if resolution.passed and not checks["keynote_cache_present"]:
        blockers.append("Validated keynote resource is unavailable to research idea knowledge acquisition.")

    checks["cache_path_present"] = all(
        uri.startswith("xlab-cache://models/") for uri in resolution.model_uris.values()
    ) and len(resolution.model_uris) == 2
    checks["survey_resource_paths_resolved"] = all(
        path_exists(path)
        for path in (
            repository.source.graph_db_path,
            repository.source.component_index_dir,
            repository.source.outcome_model_path,
            repository.source.component_model_path,
        )
    )
    if resolution.passed and not checks["survey_resource_paths_resolved"]:
        blockers.append("The validated bundle did not resolve every resource required by the research idea retrieval stack.")

    portable_resources = resolution.portable_resources()
    preflight = {
        "schema_version": "xlab.research_idea.resource_preflight.v1",
        "implementation": ALGORITHM_IMPLEMENTATION,
        "passed": not blockers,
        "checks": checks,
        "blocking_errors": blockers,
        "warnings": warnings,
        "resource_manifest": resolution.portable_manifest(),
        "resources": portable_resources,
        "direct_parent_artifact_ids": list(repository.source.direct_parent_artifact_ids),
        "source_signatures": repository.source.signatures,
    }
    atomic_write_json(paths["resource_preflight"], preflight)
    for warning in warnings:
        append_diagnostic(run_dir, "warning", warning, code="resource_preflight")
    for blocker in blockers:
        append_diagnostic(run_dir, "error", blocker, code="resource_preflight")
    write_checkpoint(
        run_dir,
        "resource_preflight",
        {"counts": {"blocking_errors": len(blockers), "warnings": len(warnings)}, "checks": checks},
        completed_stage="resource_preflight",
        request_signature=request_signature,
        runtime_config_signature=runtime_signature,
        source_signatures=repository.source.signatures,
        warnings=warnings,
        last_error=None if not blockers else "; ".join(blockers),
    )
    append_pipeline_event(run_dir, "resource_preflight", "success" if not blockers else "blocked", {"checks": checks})
    return preflight


def required_python_modules_available() -> bool:
    return all(
        importlib.util.find_spec(name) is not None
        for name in (
            "research_idea_lib.algorithm.workflow",
            "research_idea_lib.algorithm.search",
            "research_idea_lib.algorithm.fusion",
            "research_idea_lib.algorithm.provider_adapter",
        )
    )


def path_exists(path: Path | None) -> bool:
    return path is not None and path.exists()


def path_is_file(path: Path | None) -> bool:
    return path is not None and path.exists() and path.is_file()


def path_is_dir(path: Path | None) -> bool:
    return path is not None and path.exists() and path.is_dir()


def sqlite_table_has_rows(path: Path | None, table: str) -> bool:
    if not path_is_file(path):
        return False
    uri = f"file:{path.as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        try:
            row = conn.execute(f"select 1 from {table} limit 1").fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return False
    return row is not None


def component_metadata_nonempty(path: Path | None) -> bool:
    if not path_is_file(path):
        return False
    try:
        value = read_json(path)
    except Exception:
        return False
    meta = value.get("meta") if isinstance(value, dict) else None
    return isinstance(meta, dict) and bool(meta)


def survey_references_have_keynotes(references: list[dict[str, Any]]) -> bool:
    for reference in references:
        if not isinstance(reference, dict):
            continue
        if str(reference.get("keynote") or reference.get("summary") or reference.get("abstract") or "").strip():
            return True
    return False



def run_research_idea(
    cwd: Path,
    run_dir: Path,
    run_id: str,
    request: IdeaRequest,
    runtime: RuntimeConfig,
    request_json: dict[str, Any],
    runtime_json: dict[str, Any],
    repository: SurveyArtifactRepository,
    source_context: dict[str, Any],
) -> None:
    del cwd
    paths = run_paths(run_dir)
    request_signature, runtime_signature, source_signature = checkpoint_signature(request_json, runtime_json, repository.source.signatures)
    if stage_completed(
        run_dir,
        "materialize_xlab_artifact",
        [paths["idea_result_json"], paths["research_idea_json"], paths["idea_trace_json"], paths["workflow_artifact"]],
        request_signature=request_signature,
        runtime_config_signature=runtime_signature,
        source_signature=source_signature,
    ):
        write_checkpoint(run_dir, "materialize_xlab_artifact", last_error=None)
        append_pipeline_event(run_dir, "materialize_xlab_artifact", "skipped")
        return
    from .algorithm.runtime_adapters import run_native_workflow

    provider = OpenAICompatibleProvider(
        api_key=provider_api_key(),
        endpoint=runtime.chat_completions_url,
        config=OpenAICompatibleConfig(
            timeout_seconds=runtime.request_timeout_seconds,
            max_attempts=runtime.max_retries + 1,
        ),
    )
    workflow_result = run_native_workflow(
        run_id=run_id,
        request=request,
        runtime=runtime,
        repository=repository,
        source_context=source_context,
        provider=provider,
        run_dir=run_dir,
    )
    if not workflow_result.succeeded:
        raise RuntimeError(workflow_result.error or "Package-native research idea workflow was incomplete.")
    workflow_artifact = dict(workflow_result.artifact)
    for namespace, path_key in (
        ("run", "workflow_run"),
        ("retrieval", "workflow_retrieval"),
        ("analysis", "workflow_analysis"),
        ("ideation", "workflow_ideation"),
        ("persistence", "workflow_persistence"),
    ):
        value = workflow_artifact.get(namespace)
        if not isinstance(value, dict):
            raise RuntimeError(f"Package-native research idea omitted the {namespace} namespace.")
        write_portable_json(paths[path_key], value)
    idea_result = final_idea_result(workflow_artifact)
    if not idea_result:
        raise RuntimeError("research idea did not produce persistence.idea_result.")
    idea_result["status"] = "success"
    source_context = merge_workflow_retrieval_context(source_context, workflow_artifact)
    research_idea = map_to_research_idea(run_id=run_id, request=request, idea_result=idea_result, workflow_artifact=workflow_artifact, source_context=source_context)
    research_idea["status"] = "success"
    trace = build_trace(run_id, request, research_idea, idea_result, workflow_artifact, source_context, status="success")
    write_portable_json(paths["workflow_artifact"], workflow_artifact)
    write_portable_json(paths["idea_result_json"], idea_result)
    write_portable_json(paths["research_idea_json"], research_idea)
    write_portable_json(paths["idea_trace_json"], trace)
    for event in research_idea.get("workflow_trace", []) if isinstance(research_idea.get("workflow_trace"), list) else []:
        stage = str(event.get("stage") if isinstance(event, dict) else "")
        if stage:
            write_checkpoint(
                run_dir,
                stage,
                {"counts": event.get("counts", {}) if isinstance(event, dict) else {}},
                completed_stage=stage,
                request_signature=request_signature,
                runtime_config_signature=runtime_signature,
                source_signatures=repository.source.signatures,
            )
            append_pipeline_event(run_dir, stage, "success", event.get("counts", {}) if isinstance(event, dict) else {})
    write_checkpoint(
        run_dir,
        "materialize_xlab_artifact",
        {"counts": artifact_counts(research_idea, idea_result)},
        completed_stage="materialize_xlab_artifact",
        request_signature=request_signature,
        runtime_config_signature=runtime_signature,
        source_signatures=repository.source.signatures,
        last_error=None,
    )
    append_pipeline_event(run_dir, "materialize_xlab_artifact", "success", artifact_counts(research_idea, idea_result))


def merge_workflow_retrieval_context(source_context: dict[str, Any], workflow_artifact: dict[str, Any]) -> dict[str, Any]:
    merged = dict(source_context)
    retrieval = retrieval_namespace(workflow_artifact)
    selected: list[dict[str, Any]] = []
    for batch in retrieval.get("rag_hits", []) if isinstance(retrieval.get("rag_hits"), list) else []:
        hits: list[Any] = []
        if isinstance(batch, dict) and isinstance(batch.get("hits"), list):
            hits = batch["hits"]
        elif isinstance(batch, dict):
            provenance = batch.get("provenance") if isinstance(batch.get("provenance"), dict) else {}
            hits = [{
                "id": batch.get("evidence_id") or batch.get("source_id"),
                "kind": batch.get("kind"),
                "title": provenance.get("title") or batch.get("kind"),
                "summary": batch.get("text"),
                "text": batch.get("text"),
                "paper_ids": batch.get("paper_ids", []),
                "source": provenance.get("source") or "literature_survey",
                "rank": provenance.get("rank"),
            }]
        elif isinstance(batch, list):
            hits = batch
        for hit in hits:
            if isinstance(hit, dict):
                selected.append(hit)
    merged["selected_evidence"] = selected
    merged["selected_evidence_status"] = "selected_outcome_evidence" if selected else "no_outcome_evidence"
    metadata = retrieval.get("metadata")
    keynote = metadata.get("keynote_pipeline") if isinstance(metadata, dict) else None
    if isinstance(keynote, dict):
        merged["keynote_pipeline"] = deepcopy(keynote)
        merged["curated_references"] = deepcopy(keynote.get("curated_references", []))
        merged["keynote_paper_ids"] = deepcopy(keynote.get("keynote_paper_ids", []))
    return merged



def materialize_incomplete(
    cwd: Path,
    run_dir: Path,
    run_id: str,
    request: IdeaRequest,
    runtime: RuntimeConfig,
    request_json: dict[str, Any],
    source_context: dict[str, Any],
    reason: str,
    *,
    requested_status: str | None = None,
) -> None:
    del cwd, requested_status
    paths = run_paths(run_dir)
    runtime_json = runtime_contract_json(runtime.to_json())
    request_signature, runtime_signature, _source_signature = checkpoint_signature(request_json, runtime_json, source_context.get("source_signatures") if isinstance(source_context.get("source_signatures"), dict) else {})
    topic = request.topic or str(source_context.get("topic") or "Research idea")
    idea_result = incomplete_idea_result(topic, reason)
    idea_result["status"] = "incomplete"
    idea_result["incomplete_reason"] = reason
    research_idea = incomplete_research_idea(run_id=run_id, request=request, idea_result=idea_result, source_context=source_context, reason=reason)
    research_idea["status"] = "incomplete"
    research_idea["incomplete_reason"] = reason
    trace = build_trace(
        run_id,
        request,
        research_idea,
        idea_result,
        {"run": {}, "retrieval": {}, "analysis": {}, "ideation": {}, "persistence": {}},
        source_context,
        status="incomplete",
        incomplete_reason=reason,
    )
    write_portable_json(paths["idea_result_json"], idea_result)
    write_portable_json(paths["research_idea_json"], research_idea)
    write_portable_json(paths["idea_trace_json"], trace)
    write_checkpoint(
        run_dir,
        "idea_generation",
        {"counts": {"candidates": 0, "mcts_iterations": 0}, "mode": "incomplete"},
        request_signature=request_signature,
        runtime_config_signature=runtime_signature,
        last_error=reason,
    )
    append_pipeline_event(run_dir, "idea_generation", "blocked", {"reason": reason})


def build_trace(
    run_id: str,
    request: IdeaRequest,
    research_idea: dict[str, Any],
    idea_result: dict[str, Any],
    workflow_artifact: dict[str, Any],
    source_context: dict[str, Any],
    *,
    status: str,
    incomplete_reason: str | None = None,
) -> dict[str, Any]:
    del request
    trace = {
        "schema_version": "xlab.research_idea.trace.v2",
        "status": status,
        "run_id": run_id,
        "topic": research_idea.get("topic") or source_context.get("topic") or "Research idea",
        "workflow_trace": research_idea.get("workflow_trace", []),
        "operation_trace": research_idea.get("operation_trace", []),
        "source_context": source_context,
        "mcts_evolution": idea_result.get("mcts_evolution") if isinstance(idea_result.get("mcts_evolution"), dict) else {},
        "fusion_evolution": idea_result.get("fusion_evolution") if isinstance(idea_result.get("fusion_evolution"), dict) else {},
        "fusion_metadata": idea_result.get("fusion_metadata") if isinstance(idea_result.get("fusion_metadata"), dict) else {},
        "workflow_namespaces": sorted(workflow_artifact.keys()) if isinstance(workflow_artifact, dict) else [],
    }
    if incomplete_reason:
        trace["incomplete_reason"] = incomplete_reason
    return trace


def artifact_counts(research_idea: dict[str, Any], idea_result: dict[str, Any]) -> dict[str, int]:
    mcts = idea_result.get("mcts_evolution") if isinstance(idea_result.get("mcts_evolution"), dict) else {}
    fusion = idea_result.get("fusion_evolution") if isinstance(idea_result.get("fusion_evolution"), dict) else {}
    ranked = fusion.get("ranked_candidates") if isinstance(fusion.get("ranked_candidates"), list) else []
    return {
        "evidence": len(research_idea.get("source_evidence", [])) if isinstance(research_idea.get("source_evidence"), list) else 0,
        "components": len(idea_result.get("components", [])) if isinstance(idea_result.get("components"), list) else 0,
        "candidates": len(ranked),
        "mcts_iterations": int(mcts.get("total_iterations") or 0),
    }


def runtime_contract_json(runtime_json: dict[str, Any]) -> dict[str, Any]:
    """Attach package-native algorithm contract metadata to the persisted runtime profile."""

    value = dict(runtime_json)
    value["algorithm_implementation"] = ALGORITHM_IMPLEMENTATION
    value["algorithm_signature"] = stable_signature(ALGORITHM_IMPLEMENTATION)
    return value


def audit_and_manifest(cwd: Path, run_dir: Path, run_id: str, request_json: dict[str, Any], *, requested_status: str | None = None) -> dict[str, Any]:
    return finalize_run(
        cwd=cwd,
        run_dir=run_dir,
        run_id=run_id,
        skill_version=SKILL_VERSION,
        request=request_json,
        requested_status=requested_status,
    )

