from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .checkpoints import read_checkpoint
from .common import portable_output_findings, read_json, read_jsonl, run_paths, text_contains_secret_shape, utc_now
from .research_idea_spec import ALGORITHM_ID, IDEA_TASTE_MODES
from .providers.routing import routing_credentials, routing_policy_from_environment

RESEARCH_IDEA_MODES = IDEA_TASTE_MODES
ALGORITHM_PROVENANCE = {
    "algorithm": ALGORITHM_ID,
    "algorithm_id": ALGORITHM_ID,
    "algorithm_spec": ALGORITHM_ID,
    "runtime_profile": "xlab.research_idea.runtime.v1",
    "evidence_profile": "xlab.research_idea.evidence.v1",
    "success_profile": "xlab.research_idea.success.v1",
}

REQUIRED_IDEA_FIELDS = (
    "schema_version",
    "topic",
    "research_question",
    "hypothesis",
    "method",
    "expected_contribution",
    "experiment_plan",
    "data_requirements",
    "baselines",
    "metrics",
    "risks",
    "source_evidence",
)
REQUIRED_RESULT_FIELDS = (
    "title",
    "abstract",
    "core_contribution",
    "research_question",
    "hypothesis",
    "method",
    "experiment_plan",
    "data_requirements",
    "baselines",
    "metrics",
    "risks",
    "introduction",
    "components",
    "algorithm",
    "reference_papers",
    "mcts_evolution",
    "algorithm_provenance",
)
REQUIRED_PROVIDER_OPS = (
    "xlab.research_idea.analysis.generate.v1",
    "xlab.research_idea.idea.materialize.v1",
    "xlab.research_idea.fusion.generate.v1",
    "xlab.research_idea.component_novelty.evaluate.v1",
)
COMPONENT_NOVELTY_PROVIDER_OPERATION = "xlab.research_idea.component_novelty.evaluate.v1"
PLACEHOLDER_TEXT = {
    "",
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
    "unspecified",
    "not provided",
    "tbd",
    "todo",
    "placeholder",
    "primary_task_metric",
    "baseline",
    "dataset",
}


def audit_artifacts(run_dir: Path) -> dict[str, Any]:
    paths = run_paths(run_dir)
    blocking_errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}
    checkpoint = read_checkpoint(run_dir) or {}

    public_artifact_errors: list[str] = []
    for name, path in (
        ("idea_result.json", paths["idea_result_json"]),
        ("research_idea.json", paths["research_idea_json"]),
        ("idea_trace.json", paths["idea_trace_json"]),
    ):
        checks[f"has_{name.replace('.', '_')}"] = path.exists()
        if not path.exists():
            public_artifact_errors.append(f"Missing artifacts/{name}.")
    blocking_errors.extend(public_artifact_errors)
    if public_artifact_errors:
        return build_report(False, blocking_errors, warnings, {}, checks, run_dir, checkpoint=checkpoint)

    checks["has_state_resource_preflight_json"] = paths["resource_preflight"].exists()
    checks["has_state_workflow_artifact_json"] = paths["workflow_artifact"].exists()
    if not checks["has_state_resource_preflight_json"]:
        blocking_errors.append("Missing state/resource_preflight.json with resource-preflight provenance.")
    if not checks["has_state_workflow_artifact_json"]:
        blocking_errors.append("Missing state/research_idea/artifact.json with research idea namespace provenance.")

    idea_result = load_object(paths["idea_result_json"], blocking_errors, "idea_result.json")
    adapted_provenance = algorithm_provenance(idea_result)
    if adapted_provenance:
        idea_result["algorithm_provenance"] = adapted_provenance
    idea = load_object(paths["research_idea_json"], blocking_errors, "research_idea.json")
    trace = load_object(paths["idea_trace_json"], blocking_errors, "idea_trace.json")
    preflight = load_object(paths["resource_preflight"], blocking_errors, "resource preflight") if paths["resource_preflight"].exists() else {}
    workflow_artifact = load_object(paths["workflow_artifact"], blocking_errors, "artifact.json") if paths["workflow_artifact"].exists() else {}
    request = read_json(paths["request"]) if paths["request"].exists() else {}
    policy = request.get("research_policy", {})
    task_only = policy.get("evidence_mode") == "task_only"

    checks["idea_schema_version"] = idea.get("schema_version") == "xlab.research_idea.v2"
    if not checks["idea_schema_version"]:
        blocking_errors.append("research_idea.json schema_version must be xlab.research_idea.v2.")
    checks["trace_schema_version"] = trace.get("schema_version") == "xlab.research_idea.trace.v2"
    if not checks["trace_schema_version"]:
        blocking_errors.append("idea_trace.json schema_version must be xlab.research_idea.trace.v2.")

    missing_idea_fields = [field for field in REQUIRED_IDEA_FIELDS if empty(idea.get(field))]
    checks["required_research_idea_fields"] = not missing_idea_fields
    if missing_idea_fields:
        blocking_errors.append(f"research_idea.json is missing required content: {', '.join(missing_idea_fields)}.")
    missing_result_fields = [field for field in REQUIRED_RESULT_FIELDS if empty(idea_result.get(field))
                             and not (task_only and field == "reference_papers")]
    checks["required_idea_result_fields"] = not missing_result_fields
    if missing_result_fields:
        blocking_errors.append(f"idea_result.json is missing required research idea content: {', '.join(missing_result_fields)}.")

    result_field_errors = generated_field_errors(
        idea_result,
        text_fields=("title", "abstract", "core_contribution", "research_question", "hypothesis", "method"),
        list_fields=("experiment_plan", "data_requirements", "baselines", "metrics", "risks"),
        label="idea_result.json",
    )
    idea_field_errors = generated_field_errors(
        idea,
        text_fields=("research_question", "hypothesis", "method", "expected_contribution"),
        list_fields=("experiment_plan", "data_requirements", "baselines", "metrics", "risks"),
        label="research_idea.json",
    )
    checks["provider_generated_public_fields"] = not (result_field_errors or idea_field_errors)
    blocking_errors.extend(result_field_errors)
    blocking_errors.extend(idea_field_errors)

    preflight_checks = preflight.get("checks") if isinstance(preflight.get("checks"), dict) else {}
    preflight_implementation = preflight.get("implementation") if isinstance(preflight.get("implementation"), dict) else {}
    required_preflight_checks = (
        "provider_available",
        "required_python_modules",
        "graph_db_present",
        "graph_db_has_nodes",
        "component_index_dir_present",
        "component_faiss_present",
        "component_metadata_present",
        "component_metadata_nonempty",
        "outcome_sentence_transformer_present",
        "component_novelty_model_present",
        "keynote_cache_present",
        "survey_resource_paths_resolved",
    )
    if task_only:
        required_preflight_checks = ("provider_available", "required_python_modules", "task_context_present", "literature_disabled")
    missing_preflight_checks = [name for name in required_preflight_checks if preflight_checks.get(name) is not True]
    resource_identity_errors = ([] if task_only and preflight.get("resources") == {}
                                else validate_resource_identities(preflight))
    checks["resource_preflight_passed"] = preflight.get("passed") is True
    checks["resource_preflight_algorithm_spec"] = preflight_implementation.get("algorithm_spec") == ALGORITHM_ID
    checks["resource_preflight_success_profile"] = preflight_implementation.get("success_policy") == ALGORITHM_PROVENANCE["success_profile"]
    checks["resource_preflight_evidence_profile"] = preflight_implementation.get("resource_profile") == ALGORITHM_PROVENANCE["evidence_profile"]
    checks["resource_preflight_required_checks"] = not missing_preflight_checks
    checks["resource_preflight_resource_identities"] = not resource_identity_errors
    if not checks["resource_preflight_passed"]:
        blocking_errors.append("The resource preflight must pass research idea resource validation.")
    if not checks["resource_preflight_algorithm_spec"]:
        blocking_errors.append(f"The resource preflight must identify algorithm spec {ALGORITHM_ID}.")
    if not checks["resource_preflight_success_profile"]:
        blocking_errors.append("The resource preflight must identify the required success profile.")
    if not checks["resource_preflight_evidence_profile"]:
        blocking_errors.append("The resource preflight must identify the required evidence profile.")
    if missing_preflight_checks:
        blocking_errors.append("Resource preflight did not prove required resources/checks: " + ", ".join(missing_preflight_checks) + ".")
    if resource_identity_errors:
        blocking_errors.extend(resource_identity_errors)

    required_workflow_namespaces = ("run", "retrieval", "analysis", "ideation", "persistence")
    missing_workflow_namespaces = [namespace for namespace in required_workflow_namespaces if not isinstance(workflow_artifact.get(namespace), dict)]
    checks["workflow_artifact_namespaces"] = not missing_workflow_namespaces
    if missing_workflow_namespaces:
        blocking_errors.append("state/research_idea/artifact.json must contain research idea namespaces: " + ", ".join(missing_workflow_namespaces) + ".")
    workflow_retrieval = workflow_artifact.get("retrieval") if isinstance(workflow_artifact.get("retrieval"), dict) else {}
    workflow_analysis = workflow_artifact.get("analysis") if isinstance(workflow_artifact.get("analysis"), dict) else {}
    workflow_ideation = workflow_artifact.get("ideation") if isinstance(workflow_artifact.get("ideation"), dict) else {}
    workflow_persistence = workflow_artifact.get("persistence") if isinstance(workflow_artifact.get("persistence"), dict) else {}
    checks["workflow_retrieval_namespace_provenance"] = bool(workflow_retrieval.get("rag_hits") and (task_only or workflow_retrieval.get("references")))
    checks["workflow_analysis_namespace_provenance"] = bool(workflow_analysis.get("entries") and workflow_analysis.get("root_idea"))
    checks["workflow_ideation_namespace_provenance"] = bool(workflow_ideation.get("latest_candidate") and workflow_ideation.get("mode_candidates") and workflow_ideation.get("fusion_result"))
    checks["workflow_persistence_namespace_provenance"] = bool(workflow_persistence.get("idea_result"))
    if not checks["workflow_retrieval_namespace_provenance"]:
        blocking_errors.append("research idea artifact must preserve retrieval namespace provenance for RAG hits and references.")
    if not checks["workflow_analysis_namespace_provenance"]:
        blocking_errors.append("research idea artifact must preserve provider-generated analysis namespace provenance.")
    if not checks["workflow_ideation_namespace_provenance"]:
        blocking_errors.append("research idea artifact must preserve research idea candidate and fusion namespace provenance.")
    if not checks["workflow_persistence_namespace_provenance"]:
        blocking_errors.append("research idea artifact must preserve persisted idea_result namespace provenance.")

    source_context = idea.get("source_context") if isinstance(idea.get("source_context"), dict) else trace.get("source_context") if isinstance(trace.get("source_context"), dict) else {}
    selected_evidence = source_context.get("selected_evidence") if isinstance(source_context.get("selected_evidence"), list) else []
    source_evidence = idea.get("source_evidence") if isinstance(idea.get("source_evidence"), list) else []
    references = source_context.get("references") if isinstance(source_context.get("references"), list) else []
    if task_only:
        local_records = workflow_retrieval.get("evidence", [])
        checks["no_literature_evidence"] = (
            idea_result.get("reference_papers") == [] and references == []
            and workflow_retrieval.get("references") == []
            and all(item.get("kind") in {"task_context", "experiment_observation"}
                    and not item.get("paper_ids") for item in local_records)
        )
        if not checks["no_literature_evidence"]:
            blocking_errors.append("Task-only research contains literature evidence or references.")
    reference_ids = {str(reference.get("paper_id")) for reference in references if isinstance(reference, dict) and reference.get("paper_id") is not None}
    evidence_ids = {str(pid) for item in source_evidence if isinstance(item, dict) for pid in item.get("paper_ids", []) if pid is not None}
    unresolved = sorted(evidence_ids - reference_ids) if reference_ids else []
    checks["source_evidence_present"] = bool(source_evidence)
    checks["source_evidence_resolves"] = not unresolved
    checks["source_context_present"] = bool(source_context)
    checks["rag_hits_present"] = bool(selected_evidence)
    if not checks["source_evidence_present"]:
        blocking_errors.append("research_idea.json must include source_evidence tied to the survey.")
    if unresolved:
        blocking_errors.append(f"Unresolved source_evidence paper ids: {', '.join(unresolved)}.")
    if not checks["source_context_present"]:
        blocking_errors.append("research_idea.json or idea_trace.json must include source_context.")
    if not checks["rag_hits_present"]:
        blocking_errors.append("source_context must include selected survey evidence / RAG hits.")

    components = idea_result.get("components") if isinstance(idea_result.get("components"), list) else []
    algorithm = idea_result.get("algorithm") if isinstance(idea_result.get("algorithm"), list) else []
    baselines = idea.get("baselines") if isinstance(idea.get("baselines"), list) else []
    metrics = idea.get("metrics") if isinstance(idea.get("metrics"), list) else []
    checks["experiment_ready"] = bool(components and algorithm and baselines and metrics)
    if not checks["experiment_ready"]:
        blocking_errors.append("A successful idea requires components, algorithm/experiment plan, baselines, and metrics.")

    workflow_trace = trace.get("workflow_trace") if isinstance(trace.get("workflow_trace"), list) else []
    operation_trace = trace.get("operation_trace") if isinstance(trace.get("operation_trace"), list) else []
    workflow_run = workflow_artifact.get("run") if isinstance(workflow_artifact.get("run"), dict) else {}
    workflow_operation_trace = workflow_run.get("operation_trace") if isinstance(workflow_run.get("operation_trace"), list) else []
    combined_operation_trace = [*operation_trace, *workflow_operation_trace]
    request = read_json(paths["request"]) if paths["request"].exists() else {}
    feedback_supplied = bool(request.get("experiment_feedback")) if isinstance(request, dict) else False
    stage_names = [str(item.get("stage")) for item in workflow_trace if isinstance(item, dict)]
    expected_stage_path = ["advanced_analysis", "re_analysis_replan", "idea_generation"] if feedback_supplied else ["knowledge_acquisition", "advanced_analysis", "idea_generation"]
    if policy and feedback_supplied:
        expected_stage_path.insert(0, "knowledge_acquisition")
    checks["research_idea_workflow_trace"] = stage_names == expected_stage_path
    if not checks["research_idea_workflow_trace"]:
        workflow_label = "feedback research idea" if feedback_supplied else "research idea"
        blocking_errors.append("idea_trace.json must include the strict " + workflow_label + " workflow path: " + " -> ".join(expected_stage_path) + ".")
    failed_or_degraded = [str(item.get("stage") or "<unknown>") for item in workflow_trace if isinstance(item, dict) and str(item.get("status") or "") != "success"]
    checks["research_idea_workflow_no_degraded_status"] = not failed_or_degraded
    if failed_or_degraded:
        blocking_errors.append("research idea workflow contains non-success stage statuses: " + ", ".join(failed_or_degraded) + ".")
    checks["feedback_replan_recorded"] = (not feedback_supplied) or ("re_analysis_replan" in stage_names and idea.get("replanning_trigger") is not None)
    if not checks["feedback_replan_recorded"]:
        blocking_errors.append("Experiment feedback was supplied but re_analysis_replan metadata was not recorded.")
    provider_success_ops = {
        str(item.get("op_name") or "")
        for item in combined_operation_trace
        if isinstance(item, dict) and item.get("event") == "llm_call" and item.get("status") == "success"
    }
    workflow_provider_success_ops = {
        str(item.get("op_name") or "")
        for item in workflow_operation_trace
        if isinstance(item, dict) and item.get("event") == "llm_call" and item.get("status") == "success"
    }
    required_ops = tuple(op for op in REQUIRED_PROVIDER_OPS if not (task_only and op == COMPONENT_NOVELTY_PROVIDER_OPERATION))
    missing_provider_ops = [op for op in required_ops if op not in provider_success_ops]
    missing_workflow_provider_ops = [op for op in required_ops if op not in workflow_provider_success_ops]
    if task_only and COMPONENT_NOVELTY_PROVIDER_OPERATION in provider_success_ops:
        blocking_errors.append("Task-only research executed a literature novelty operation.")
    novelty_trace_tampered = any(
        isinstance(item, dict)
        and item.get("event") == "llm_call"
        and item.get("status") == "success"
        and item.get("op_name") == COMPONENT_NOVELTY_PROVIDER_OPERATION
        and item.get("provider_operation") != COMPONENT_NOVELTY_PROVIDER_OPERATION
        for item in combined_operation_trace
    )
    workflow_novelty_trace_tampered = any(
        isinstance(item, dict)
        and item.get("event") == "llm_call"
        and item.get("status") == "success"
        and item.get("op_name") == COMPONENT_NOVELTY_PROVIDER_OPERATION
        and item.get("provider_operation") != COMPONENT_NOVELTY_PROVIDER_OPERATION
        for item in workflow_operation_trace
    )
    checks["provider_call_trace_present"] = bool(provider_success_ops)
    checks["provider_required_ops_present"] = not missing_provider_ops
    checks["component_novelty_provider_operation_authentic"] = not novelty_trace_tampered
    checks["workflow_artifact_provider_call_trace_present"] = bool(workflow_provider_success_ops)
    checks["workflow_artifact_provider_required_ops_present"] = not missing_workflow_provider_ops
    checks["workflow_artifact_component_novelty_provider_operation_authentic"] = (
        not workflow_novelty_trace_tampered
    )
    if not checks["provider_call_trace_present"]:
        blocking_errors.append("research idea success requires provider-call operation trace events.")
    if missing_provider_ops:
        blocking_errors.append("research idea provider-call trace is missing required operations: " + ", ".join(missing_provider_ops) + ".")
    if novelty_trace_tampered:
        blocking_errors.append(
            "research idea component novelty trace has a mismatched native provider operation."
        )
    if not checks["workflow_artifact_provider_call_trace_present"]:
        blocking_errors.append("research idea artifact must preserve provider-call operation trace events.")
    if missing_workflow_provider_ops:
        blocking_errors.append("research idea artifact provider-call trace is missing required operations: " + ", ".join(missing_workflow_provider_ops) + ".")
    if workflow_novelty_trace_tampered:
        blocking_errors.append(
            "research idea artifact component novelty trace has a mismatched native provider operation."
        )

    mcts = idea_result.get("mcts_evolution") if isinstance(idea_result.get("mcts_evolution"), dict) else {}
    mcts_iterations = mcts.get("iterations") if isinstance(mcts.get("iterations"), list) else []
    checks["mcts_iterations_present"] = int(mcts.get("total_iterations") or 0) > 0 and bool(mcts_iterations)
    if not checks["mcts_iterations_present"]:
        blocking_errors.append("idea_result.json must include MCTS evolution with iterations.")
    checks["mcts_evaluations_present"] = all(
        isinstance(item, dict) and isinstance(item.get("evaluation"), dict) and not empty(item.get("evaluation"))
        for item in mcts_iterations
    ) if mcts_iterations else False
    if not checks["mcts_evaluations_present"]:
        blocking_errors.append("MCTS evolution must include evaluation payloads for search iterations.")
    mode_iteration_set = {str(item.get("idea_taste_mode") or "") for item in mcts_iterations if isinstance(item, dict) and str(item.get("idea_taste_mode") or "").strip()}
    checks["mcts_iterations_cover_all_modes"] = all(mode in mode_iteration_set for mode in RESEARCH_IDEA_MODES)
    if not checks["mcts_iterations_cover_all_modes"]:
        missing_modes = [mode for mode in RESEARCH_IDEA_MODES if mode not in mode_iteration_set]
        blocking_errors.append("MCTS evolution must include search iterations for all research idea modes: " + ", ".join(missing_modes) + ".")
    checks["pareto_front_present"] = isinstance(mcts.get("pareto_front"), dict) and bool(mcts.get("pareto_front"))
    if not checks["pareto_front_present"]:
        blocking_errors.append("MCTS evolution must include Pareto candidate provenance.")
    pareto = mcts.get("pareto_front") if isinstance(mcts.get("pareto_front"), dict) else {}
    checks["pareto_front_covers_all_modes"] = all(mode in pareto for mode in RESEARCH_IDEA_MODES)
    if not checks["pareto_front_covers_all_modes"]:
        missing_pareto_modes = [mode for mode in RESEARCH_IDEA_MODES if mode not in pareto]
        blocking_errors.append("MCTS Pareto provenance must cover all research idea modes: " + ", ".join(missing_pareto_modes) + ".")

    provenance = algorithm_provenance(idea_result)
    source_modes = idea_result.get("source_modes") if isinstance(idea_result.get("source_modes"), list) else []
    completed_modes = provenance.get("completed_modes") if isinstance(provenance.get("completed_modes"), list) else source_modes
    completed_mode_set = {str(mode) for mode in completed_modes}
    checks["algorithm_provenance_present"] = bool(provenance)
    checks["research_idea_algorithm_spec"] = provenance.get("algorithm") == ALGORITHM_ID
    checks["research_idea_all_modes"] = all(mode in completed_mode_set for mode in RESEARCH_IDEA_MODES)
    checks["research_idea_fusion_used"] = idea_result.get("idea_source") == "fused" and provenance.get("fusion_used") is True
    checks["fusion_metadata_present"] = isinstance(idea_result.get("fusion_metadata"), dict) and bool(idea_result.get("fusion_metadata")) and provenance.get("fusion_metadata_present") is True
    checks["fusion_evolution_present"] = isinstance(idea_result.get("fusion_evolution"), dict) and bool(idea_result.get("fusion_evolution"))
    if not checks["algorithm_provenance_present"]:
        blocking_errors.append("idea_result.json must include algorithm_provenance.")
    if not checks["research_idea_algorithm_spec"]:
        blocking_errors.append(f"A successful idea requires algorithm spec {ALGORITHM_ID}.")
    if not checks["research_idea_all_modes"]:
        missing_modes = [mode for mode in RESEARCH_IDEA_MODES if mode not in completed_mode_set]
        blocking_errors.append("research idea must complete all idea taste modes: " + ", ".join(missing_modes) + ".")
    if not checks["research_idea_fusion_used"]:
        blocking_errors.append("research idea success requires fused output, not a raw single-mode candidate.")
    if not checks["fusion_metadata_present"]:
        blocking_errors.append("research idea fusion metadata is required for success.")
    if not checks["fusion_evolution_present"]:
        blocking_errors.append("research idea fusion evolution is required for success.")

    if isinstance(checkpoint.get("warnings"), list):
        warnings.extend(str(warning) for warning in checkpoint["warnings"] if warning)
    last_error = checkpoint.get("last_error") if isinstance(checkpoint, dict) else None
    if isinstance(last_error, str) and last_error.strip():
        warnings.append(last_error.strip())
        if "provider" in last_error.lower() or "research_idea" in last_error.lower() or "survey" in last_error.lower():
            blocking_errors.append(last_error.strip())

    leak_findings = scan_artifact_safety(run_dir, exclude={paths["report_json"]})
    checks["portable_artifact_paths"] = not leak_findings["nonportable_paths"]
    checks["no_api_secret_leak"] = not leak_findings["secrets"]
    if leak_findings["nonportable_paths"]:
        blocking_errors.append("Run artifacts contain non-portable paths outside the run or project directory.")
    if leak_findings["secrets"]:
        blocking_errors.append("Run artifacts appear to contain an API secret value or prefix.")

    warnings = sorted(set(warnings))
    counts = {
        "references": len(references),
        "evidence": len(source_evidence),
        "rag_hits": len(selected_evidence),
        "components": len(components),
        "algorithm_steps": len(algorithm),
        "baselines": len(baselines),
        "metrics": len(metrics),
        "candidates": count_candidates(idea_result),
        "mcts_iterations": int(mcts.get("total_iterations") or 0),
        "research_idea_modes": len(completed_mode_set),
        "workflow_stages": len(workflow_trace),
        "operations": len(combined_operation_trace),
        "provider_success_ops": len(provider_success_ops),
    }
    return build_report(not blocking_errors, blocking_errors, warnings, counts, checks, run_dir, checkpoint=checkpoint, source=source_context)


def load_object(path: Path, blockers: list[str], label: str) -> dict[str, Any]:
    value = read_json(path)
    if isinstance(value, dict):
        return value
    blockers.append(f"{label} must contain a JSON object.")
    return {}


def validate_resource_identities(preflight: dict[str, Any]) -> list[str]:
    resources = preflight.get("resources")
    if not isinstance(resources, dict):
        return ["Resource preflight must record canonical portable resource identities."]

    errors: list[str] = []
    for role in ("survey", "graph", "component_index", "keynotes", "outcome_model", "component_model"):
        resource = resources.get(role)
        if not isinstance(resource, dict):
            errors.append(f"Resource preflight is missing the {role} resource identity.")
            continue
        resource_id = str(resource.get("resource_id") or "").strip()
        logical_uri = str(resource.get("logical_uri") or "").strip()
        if not resource_id:
            errors.append(f"Resource preflight {role} resource must include resource_id.")
        if role in {"outcome_model", "component_model"}:
            if not logical_uri.startswith("xlab-cache://models/"):
                errors.append(f"Resource preflight {role} resource must use an xlab-cache://models/ logical URI.")
        elif not logical_uri.startswith("xlab-resource://"):
            errors.append(f"Resource preflight {role} resource must use an xlab-resource:// logical URI.")
    return errors


def algorithm_provenance(idea_result: dict[str, Any]) -> dict[str, Any]:
    canonical = idea_result.get("algorithm_provenance")
    return canonical if isinstance(canonical, dict) else {}


def generated_field_errors(
    payload: dict[str, Any],
    *,
    text_fields: tuple[str, ...],
    list_fields: tuple[str, ...],
    label: str,
) -> list[str]:
    errors: list[str] = []
    for field in text_fields:
        value = str(payload.get(field) or "").strip()
        if placeholder(value):
            errors.append(f"{label} field {field} must contain provider-generated non-placeholder text.")
    for field in list_fields:
        value = payload.get(field)
        if not isinstance(value, list) or not value:
            errors.append(f"{label} field {field} must contain a provider-generated non-empty list.")
            continue
        if any(placeholder(str(item or "")) for item in value):
            errors.append(f"{label} field {field} contains empty or placeholder content.")
    return errors


def placeholder(value: str) -> bool:
    text = str(value or "").strip()
    normalized = text.lower().strip(" .;:-_")
    return not text or normalized in PLACEHOLDER_TEXT


def empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


def count_candidates(idea_result: dict[str, Any]) -> int:
    fusion = idea_result.get("fusion_evolution") if isinstance(idea_result.get("fusion_evolution"), dict) else {}
    ranked = fusion.get("ranked_candidates") if isinstance(fusion.get("ranked_candidates"), list) else []
    return len(ranked)


def scan_artifact_safety(run_dir: Path, *, exclude: set[Path] | None = None) -> dict[str, list[str]]:
    nonportable_paths: list[str] = []
    secrets: list[str] = []
    secret_values = [os.environ.get("OPENAI_API_KEY", "")]
    routing_policy = routing_policy_from_environment()
    if routing_policy is not None:
        secret_values.extend(routing_credentials(routing_policy).values())
    secret_needles = []
    for value in secret_values:
        if not value:
            continue
        secret_needles.append(value)
        if len(value) >= 8:
            secret_needles.append(value[:8])
    excluded = {path.resolve() for path in (exclude or set())}
    for path in candidate_text_files(run_dir):
        if path.resolve() in excluded:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(needle and needle in text for needle in secret_needles) or text_contains_secret_shape(text):
            secrets.append(str(path))
        if path.suffix.lower() != ".json":
            continue
        try:
            value = read_json(path)
        except Exception:
            continue
        findings = portable_output_findings(value)
        if findings["secrets"] and str(path) not in secrets:
            secrets.append(str(path))
        path_findings = findings["nonportable_paths"]
        nonportable_paths.extend(f"{path}:{field}" for field in path_findings)
    return {"nonportable_paths": nonportable_paths, "secrets": secrets}


def candidate_text_files(run_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".json", ".jsonl", ".log", ".txt", ".md"}:
            files.append(path)
    return files


def build_report(
    passed: bool,
    blocking_errors: list[str],
    warnings: list[str],
    counts: dict[str, int],
    checks: dict[str, bool],
    run_dir: Path,
    *,
    source: dict[str, Any] | None = None,
    checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths = run_paths(run_dir)
    return {
        "schema_version": "xlab.research_idea.report.v2",
        "status": "success" if passed else "incomplete",
        "generated_at": utc_now(),
        "passed": passed,
        "blocking_errors": blocking_errors,
        "blockers": blocking_errors,
        "warnings": warnings,
        "counts": counts,
        "checks": checks,
        "source": source or {},
        "stages": (checkpoint or {}).get("completed_stages", []),
        "completed_stages": (checkpoint or {}).get("completed_stages", []),
        "algorithm_provenance": dict(ALGORITHM_PROVENANCE),
        "checkpoint": checkpoint or {},
        "metrics": {
            "diagnostic_count": sum(1 for _ in read_jsonl(paths["diagnostics_log"])) if paths["diagnostics_log"].exists() else 0,
            "pipeline_event_count": sum(1 for _ in read_jsonl(paths["pipeline_log"])) if paths["pipeline_log"].exists() else 0,
        },
        "files": {
            "idea_result": "idea_result.json",
            "research_idea": "research_idea.json",
            "idea_trace": "idea_trace.json",
            "report": "idea_report.json",
        },
    }
