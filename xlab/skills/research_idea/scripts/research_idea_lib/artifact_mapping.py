from __future__ import annotations

from typing import Any

from .inputs import IdeaRequest
from .research_idea_artifacts import ideation_namespace, latest_analysis_entry, retrieval_namespace
from .materialization_validation import align_public_materialization
from .research_idea_spec import ALGORITHM_ID

_PLACEHOLDER_TEXT = {
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


def require_generated_text(payload: dict[str, Any], key: str) -> str:
    raw = payload.get(key)
    if not isinstance(raw, str):
        raise ValueError(f"research idea idea_result field {key} must be provider-generated text.")
    value = raw.strip()
    normalized = value.lower().strip(" .;:-_")
    if not value or normalized in _PLACEHOLDER_TEXT:
        raise ValueError(f"research idea idea_result is missing provider-generated {key}.")
    return value


def require_generated_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"research idea idea_result field {key} must be a provider-generated non-empty list.")
    items: list[str] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, str):
            raise ValueError(f"research idea idea_result field {key} must contain only text entries.")
        text = raw.strip()
        normalized = text.lower().strip(" .;:-_")
        if not text or normalized in _PLACEHOLDER_TEXT:
            raise ValueError(f"research idea idea_result field {key} contains empty or placeholder content.")
        if normalized in seen:
            continue
        seen.add(normalized)
        items.append(text)
    if not items:
        raise ValueError(f"research idea idea_result field {key} must contain provider-generated entries.")
    return items


def map_to_research_idea(
    *,
    run_id: str,
    request: IdeaRequest,
    idea_result: dict[str, Any],
    workflow_artifact: dict[str, Any],
    source_context: dict[str, Any],
) -> dict[str, Any]:
    ideation = ideation_namespace(workflow_artifact)
    fusion_result = ideation.get("fusion_result") if isinstance(ideation.get("fusion_result"), dict) else {}
    analysis_payload = latest_analysis_entry(workflow_artifact)
    topic = request.topic or str(source_context.get("topic") or idea_result.get("title") or "Research idea")
    retrieval = retrieval_namespace(workflow_artifact)
    evidence_registry = retrieval.get("evidence")
    references = source_context.get("references")
    if not isinstance(evidence_registry, list):
        raise ValueError("research idea retrieval namespace must contain an evidence registry.")
    if not isinstance(references, list):
        raise ValueError("source context references must be a list.")
    fused_evidence = align_public_materialization(
        idea_result,
        fusion_result,
        evidence_registry,
        references,
    )
    risks = require_generated_list(idea_result, "risks")
    source_evidence = build_source_evidence(fused_evidence)
    return {
        "schema_version": "xlab.research_idea.v2",
        "run_id": run_id,
        "topic": topic,
        "research_question": require_generated_text(idea_result, "research_question"),
        "hypothesis": require_generated_text(idea_result, "hypothesis"),
        "method": require_generated_text(idea_result, "method"),
        "expected_contribution": require_generated_text(idea_result, "core_contribution"),
        "experiment_plan": require_generated_list(idea_result, "experiment_plan"),
        "data_requirements": require_generated_list(idea_result, "data_requirements"),
        "baselines": require_generated_list(idea_result, "baselines"),
        "metrics": require_generated_list(idea_result, "metrics"),
        "risks": risks,
        "source_evidence": source_evidence,
        "idea_result": idea_result,
        "source_context": source_context,
        "workflow_trace": extract_trace(workflow_artifact, "workflow_trace"),
        "operation_trace": extract_trace(workflow_artifact, "operation_trace"),
        "mcts_evolution": idea_result.get("mcts_evolution") if isinstance(idea_result.get("mcts_evolution"), dict) else {},
        "fusion_evolution": idea_result.get("fusion_evolution") if isinstance(idea_result.get("fusion_evolution"), dict) else {},
        "fusion_metadata": idea_result.get("fusion_metadata") if isinstance(idea_result.get("fusion_metadata"), dict) else {},
        "idea_contract": idea_result.get("idea_contract") if isinstance(idea_result.get("idea_contract"), dict) else idea_contract(request),
        "replanning_trigger": analysis_payload.get("replan") if request.experiment_feedback else None,
        "previous_results": ideation.get("ablation_results") if request.experiment_feedback else None,
        "generated_from": "survey_grounded_mcts",
    }


def incomplete_idea_result(topic: str, reason: str) -> dict[str, Any]:
    return {
        "title": "",
        "abstract": "",
        "core_contribution": "",
        "method": "",
        "risks": [],
        "introduction": (
            f"Package-native research idea generation conforming to {ALGORITHM_ID} "
            f"did not complete for {topic}: {reason}"
        ),
        "components": [],
        "algorithm": [],
        "reference_papers": [],
        "mcts_evolution": {"mode": "incomplete", "reason": reason},
        "idea_source": "incomplete",
        "source_modes": [],
        "fusion_metadata": {},
        "fusion_evolution": {},
        "idea_contract": {},
    }


def incomplete_research_idea(
    *,
    run_id: str,
    request: IdeaRequest,
    idea_result: dict[str, Any],
    source_context: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    topic = request.topic or str(source_context.get("topic") or "Research idea")
    return {
        "schema_version": "xlab.research_idea.v2",
        "run_id": run_id,
        "topic": topic,
        "research_question": "",
        "hypothesis": "",
        "method": "",
        "expected_contribution": "",
        "experiment_plan": [],
        "data_requirements": [],
        "baselines": [],
        "metrics": [],
        "risks": [reason] if reason else [],
        "source_evidence": build_source_evidence([]),
        "idea_result": idea_result,
        "source_context": source_context,
        "workflow_trace": [],
        "operation_trace": [],
        "mcts_evolution": idea_result.get("mcts_evolution") if isinstance(idea_result.get("mcts_evolution"), dict) else {},
        "fusion_evolution": {},
        "fusion_metadata": {},
        "idea_contract": idea_contract(request),
        "replanning_trigger": request.experiment_feedback or None,
        "previous_results": None,
        "generated_from": "incomplete",
    }


def build_source_evidence(evidence_registry: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for item in evidence_registry:
        evidence_id = str(item["evidence_id"]).strip()
        provenance = item.get("provenance") if isinstance(item.get("provenance"), dict) else {}
        paper_ids = _strict_string_list(item.get("paper_ids"), f"source evidence {evidence_id!r} paper_ids")
        evidence.append(
            {
                "kind": str(item.get("kind") or "survey_evidence"),
                "id": evidence_id,
                "title": str(provenance.get("title") or item.get("kind") or "Survey evidence"),
                "summary": str(item.get("text") or ""),
                "paper_ids": paper_ids,
                "source": str(provenance.get("source") or "literature_survey"),
                "rank": provenance.get("rank"),
                "score": provenance.get("score"),
            }
        )
    return evidence


def validate_reference_papers(
    idea_result: dict[str, Any],
    source_context: dict[str, Any],
    source_evidence: list[dict[str, Any]],
) -> list[str]:
    """Validate provider bibliography against references linked by attributed evidence."""
    references = _strict_string_list(idea_result.get("reference_papers"), "reference_papers")
    if not references:
        return []
    attributed_paper_ids = {
        paper_id
        for item in source_evidence
        for paper_id in _strict_string_list(item.get("paper_ids"), "source evidence paper_ids")
    }
    attributed_titles: dict[str, str] = {}
    raw_references = source_context.get("references")
    if not isinstance(raw_references, list):
        raise ValueError("source context references must be a list.")
    for item in raw_references:
        if not isinstance(item, dict):
            raise ValueError("source context references must contain only mappings.")
        paper_id = item.get("paper_id")
        title = item.get("title")
        if isinstance(paper_id, str) and paper_id in attributed_paper_ids and isinstance(title, str) and title.strip():
            attributed_titles[title.strip().casefold()] = title.strip()
    unsupported = [reference for reference in references if reference.casefold() not in attributed_titles]
    if unsupported:
        raise ValueError(f"research idea idea_result contains reference_papers without attributed evidence: {unsupported!r}.")
    return [attributed_titles[reference.casefold()] for reference in references]


def _strict_string_list(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must contain only text entries.")
    return list(dict.fromkeys(item.strip() for item in value if item.strip()))


def extract_trace(workflow_artifact: dict[str, Any], key: str) -> list[dict[str, Any]]:
    run = workflow_artifact.get("run") if isinstance(workflow_artifact.get("run"), dict) else {}
    trace = run.get(key)
    return [item for item in trace if isinstance(item, dict)] if isinstance(trace, list) else []


def idea_contract(request: IdeaRequest) -> dict[str, Any]:
    return {
        "mature_idea": request.mature_idea,
        "refinement_scope": request.refinement_scope,
        "contract_mode": bool(request.mature_idea),
    }
