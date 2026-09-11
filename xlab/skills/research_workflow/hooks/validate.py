#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import Any


STAGE_CHAIN = ("collect", "graph", "survey", "idea", "novelty")
EXPECTED_STAGE_ARTIFACTS = {
    "collect": {"paper_set", "paper_edges", "collection_report"},
    "graph": {"mineru_documents", "paper_structures", "paper_extractions", "method_graph", "graph_db", "graph_report"},
    "survey": {"literature_survey_document", "literature_survey_json", "citation_trace", "survey_report"},
    "idea": {"research_idea", "research_idea_result", "research_idea_trace", "research_idea_report"},
    "novelty": {"novelty_report"},
}
NOVELTY_REQUIRED = (
    "schema_version",
    "idea_claims",
    "closest_work",
    "overlap",
    "differences",
    "evidence_gaps",
    "risk_level",
    "confidence",
    "recommendations",
)
NOVELTY_ARRAY_FIELDS = ("idea_claims", "closest_work", "overlap", "differences", "evidence_gaps", "recommendations")
RISK_LEVELS = {"low", "medium", "high", "unknown"}
CONFIDENCE_LEVELS = {"low", "medium", "high"}


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def is_record(value: Any) -> bool:
    return isinstance(value, dict)


def event_value(context: dict[str, Any]) -> dict[str, Any]:
    event = context.get("event")
    return event if is_record(event) else {}


def context_paths(context: dict[str, Any]) -> tuple[Path, Path]:
    cwd = Path(str(context.get("cwd") or ".")).resolve()
    run = context.get("run") if is_record(context.get("run")) else {}
    run_dir = Path(str(context.get("runDir") or run.get("runDir") or cwd)).resolve()
    return cwd, run_dir


def is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_allowed_path(context: dict[str, Any], path_value: Any) -> tuple[Path | None, str | None]:
    if not isinstance(path_value, str) or not path_value.strip():
        return None, "Artifact path is missing."
    cwd, run_dir = context_paths(context)
    raw = Path(path_value).expanduser()
    candidates = [raw.resolve()] if raw.is_absolute() else [(cwd / raw).resolve(), (run_dir / raw).resolve()]
    first_allowed: Path | None = None
    for candidate in candidates:
        allowed = candidate == cwd or candidate == run_dir or is_inside(candidate, cwd) or is_inside(candidate, run_dir)
        if not allowed:
            continue
        first_allowed = first_allowed or candidate
        if candidate.is_file():
            return candidate, None
    if first_allowed:
        return first_allowed, f"Artifact path does not exist: {path_value}."
    return None, f"Artifact path must stay inside the project or run workspace: {path_value}."


def read_json_file(path: Path) -> tuple[Any, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as error:
        return None, f"{path} is not valid JSON: {error}"
    except OSError as error:
        return None, f"Cannot read {path}: {error}"


def read_workflow(context: dict[str, Any]) -> dict[str, Any] | None:
    _, run_dir = context_paths(context)
    path = run_dir / "workflow.json"
    if not path.is_file():
        return None
    value, error = read_json_file(path)
    if error or not is_record(value):
        return None
    return value


def declared_stages(context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    skill = context.get("skill") if is_record(context.get("skill")) else {}
    workflow = skill.get("workflow") if is_record(skill.get("workflow")) else {}
    stages = workflow.get("stages") if isinstance(workflow.get("stages"), list) else []
    declared: dict[str, dict[str, Any]] = {}
    for stage in stages:
        if not is_record(stage) or not stage.get("id"):
            continue
        stage_id = str(stage["id"])
        declared[stage_id] = {
            "id": stage_id,
            "skill": str(stage.get("skill") or stage_id),
            "needs": [str(item) for item in stage.get("needs", []) if item is not None] if isinstance(stage.get("needs"), list) else [],
            "optional": bool(stage.get("optional")),
            "max_attempts": stage.get("maxAttempts") or stage.get("max_attempts"),
        }
    return declared


def workflow_stages(workflow: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not workflow:
        return {}
    stages = workflow.get("stages") if isinstance(workflow.get("stages"), list) else []
    return {str(stage.get("id")): stage for stage in stages if is_record(stage) and stage.get("id")}


def has_artifacts(value: Any) -> bool:
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, dict):
        return len(value) > 0
    return False


def artifact_list(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if is_record(item)] if isinstance(value, list) else []


def artifact_types(artifacts: Any) -> set[str]:
    return {str(item.get("type")) for item in artifact_list(artifacts) if item.get("type")}


def artifact_cards(artifacts: Any) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for artifact in artifact_list(artifacts):
        cards.append({
            "type": artifact.get("type"),
            "name": artifact.get("type"),
            "path": artifact.get("path"),
            "status": "ready",
        })
    return cards


def stage_summary(workflow: dict[str, Any] | None) -> list[dict[str, Any]]:
    stages = workflow.get("stages") if workflow and isinstance(workflow.get("stages"), list) else []
    return [
        {
            "id": stage.get("id"),
            "skill": stage.get("skill"),
            "status": stage.get("status"),
            "attempts": stage.get("attempts"),
            "needs": stage.get("needs", []),
        }
        for stage in stages
        if is_record(stage)
    ]


def validate_stage_artifacts(stage_id: str, artifacts: Any) -> list[str]:
    expected = EXPECTED_STAGE_ARTIFACTS.get(stage_id, set())
    if not expected:
        return []
    if not isinstance(artifacts, list):
        return [f"Stage {stage_id} completion requires a typed artifact list."]
    malformed = [index for index, item in enumerate(artifacts) if not is_record(item) or not item.get("type") or not item.get("path")]
    errors = [f"Stage {stage_id} has malformed artifact entries at indexes: {malformed}."] if malformed else []
    missing = sorted(expected - artifact_types(artifacts))
    if missing:
        errors.append(f"Stage {stage_id} is missing required artifact types: {', '.join(missing)}.")
    return errors


def find_artifact(artifacts: Any, artifact_type: str) -> dict[str, Any] | None:
    return next((item for item in artifact_list(artifacts) if item.get("type") == artifact_type), None)


def load_artifact_json(context: dict[str, Any], artifacts: Any, artifact_type: str) -> tuple[dict[str, Any] | None, Path | None, str | None]:
    artifact = find_artifact(artifacts, artifact_type)
    if not artifact:
        return None, None, f"Final manifest is missing required artifact type: {artifact_type}."
    path, error = resolve_allowed_path(context, artifact.get("path"))
    if error:
        return None, path, error
    assert path is not None
    value, read_error = read_json_file(path)
    if read_error:
        return None, path, read_error
    if not is_record(value):
        return None, path, f"{artifact_type} must be a JSON object."
    return value, path, None


def validate_novelty_report(report: dict[str, Any], *, success_gate: bool) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    missing = [field for field in NOVELTY_REQUIRED if field not in report]
    if missing:
        errors.append(f"Novelty report is missing fields: {', '.join(missing)}.")
    if report.get("schema_version") != "xlab.novelty_report.v1":
        errors.append("Novelty report schema_version must be xlab.novelty_report.v1.")
    for field in NOVELTY_ARRAY_FIELDS:
        value = report.get(field)
        if not isinstance(value, list):
            errors.append(f"Novelty report field {field} must be an array.")
    if isinstance(report.get("idea_claims"), list) and not report["idea_claims"]:
        errors.append("Novelty report requires at least one idea claim.")
    if isinstance(report.get("recommendations"), list) and not report["recommendations"]:
        errors.append("Novelty report requires at least one recommendation.")
    if report.get("risk_level") not in RISK_LEVELS:
        errors.append("Novelty report risk_level must be low, medium, high, or unknown.")
    if report.get("confidence") not in CONFIDENCE_LEVELS:
        errors.append("Novelty report confidence must be low, medium, or high.")
    if success_gate:
        if report.get("confidence") not in {"medium", "high"}:
            errors.append("Research workflow success requires medium or high novelty evidence confidence.")
        if report.get("risk_level") not in {"low", "medium"}:
            errors.append("Research workflow success requires low or medium duplication risk; preserve high/unknown-risk reports and start a new idea or evidence pass.")
    counters = {
        "closest_work": len(report.get("closest_work")) if isinstance(report.get("closest_work"), list) else 0,
        "evidence_gaps": len(report.get("evidence_gaps")) if isinstance(report.get("evidence_gaps"), list) else 0,
        "idea_claims": len(report.get("idea_claims")) if isinstance(report.get("idea_claims"), list) else 0,
    }
    return errors, counters


def handle_pre_start(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "state_patch": {
            "phase": {"id": "workflow", "label": "Research workflow ready", "status": "running", "progress": 0},
            "message": "Workflow chain: collect → graph → survey → idea → novelty.",
            "checkpoint": {
                "id": "workflow",
                "label": "Declared XLab workflow stages",
                "resumable": True,
                "resume_hint": "Use /xlab resume-run <run-id> to continue from the next runnable stage.",
                "data": {"stages": list(STAGE_CHAIN)},
            },
            "next_actions": [
                "Start collect with xlab_stage action=start stage_id=collect.",
                "Pass typed artifact arrays to xlab_stage action=complete at every stage boundary.",
                "Finish only after novelty succeeds with medium/high confidence and low/medium risk.",
            ],
        },
    }


def handle_pre_stage(context: dict[str, Any]) -> dict[str, Any]:
    event = event_value(context)
    stage_id = str(event.get("stage_id") or "")
    declared = declared_stages(context)
    if stage_id not in declared:
        return blocked(f"Unknown workflow stage: {stage_id or '<missing>'}.", "Use a stage id declared in research_workflow/xlab.skill.json.")
    workflow = read_workflow(context)
    states = workflow_stages(workflow)
    needs = declared[stage_id]["needs"]
    missing_success = [need for need in needs if states.get(need, {}).get("status") != "success"]
    missing_artifacts = [need for need in needs if states.get(need, {}).get("status") == "success" and not has_artifacts(states.get(need, {}).get("artifacts"))]
    if missing_success or missing_artifacts:
        problems = []
        if missing_success:
            problems.append(f"upstream stages not successful: {', '.join(missing_success)}")
        if missing_artifacts:
            problems.append(f"upstream stages missing artifact handoff: {', '.join(missing_artifacts)}")
        return blocked(
            f"Cannot start {stage_id}; " + "; ".join(problems) + ".",
            "Complete upstream stages with typed artifacts before starting this stage.",
            stage_id=stage_id,
            workflow=workflow,
        )
    return {
        "ok": True,
        "state_patch": {
            "phase": {"id": stage_id, "label": f"Starting workflow stage: {stage_id}", "status": "running"},
            "message": f"Stage {stage_id} may start; dependency artifacts are ready.",
            "checkpoint": {"id": stage_id, "label": f"Stage gate: {stage_id}", "resumable": True, "data": {"needs": needs, "workflow": stage_summary(workflow)}},
            "next_actions": [f"Run the packaged {declared[stage_id]['skill']} stage runtime, then call xlab_stage action=complete with typed artifacts."],
        },
    }


def handle_post_stage(context: dict[str, Any]) -> dict[str, Any]:
    event = event_value(context)
    action = str(event.get("action") or "")
    stage_id = str(event.get("stage_id") or "")
    declared = declared_stages(context)
    if stage_id not in declared:
        return blocked(f"Unknown workflow stage: {stage_id or '<missing>'}.", "Use a stage id declared in research_workflow/xlab.skill.json.")
    if action == "complete":
        artifacts = event.get("artifacts")
        errors = validate_stage_artifacts(stage_id, artifacts)
        counters: dict[str, int] = {}
        if stage_id == "novelty" and not errors:
            report, _path, error = load_artifact_json(context, artifacts, "novelty_report")
            if error:
                errors.append(error)
            elif report is not None:
                report_errors, counters = validate_novelty_report(report, success_gate=True)
                errors.extend(report_errors)
        if errors:
            return blocked(
                " ".join(errors),
                "Repair the stage artifacts or mark the stage blocked with the blocker summary.",
                stage_id=stage_id,
            )
        return {
            "ok": True,
            "state_patch": {
                "phase": {"id": stage_id, "label": f"Completed workflow stage: {stage_id}", "status": "success", "progress": 1},
                "message": f"Stage {stage_id} artifacts satisfy the workflow contract.",
                "counters": counters,
                "artifacts": artifact_cards(artifacts),
            },
        }
    if action in {"fail", "block"}:
        error = event.get("error")
        if not isinstance(error, str) or not error.strip():
            return blocked(f"Stage {stage_id} {action} transitions require an error summary.", "Call xlab_stage again with the error field populated.", stage_id=stage_id)
        return {
            "ok": True,
            "state_patch": {
                "phase": {"id": stage_id, "label": f"Workflow stage {action}: {stage_id}", "status": "failed" if action == "fail" else "blocked"},
                "diagnostics": [{"severity": "error", "code": f"stage_{action}", "message": error}],
                "next_actions": ["Repair the blocker and resume the run, or start a new upstream attempt if the blocker is evidence or novelty quality."],
            },
        }
    if action == "skip":
        if not declared[stage_id].get("optional"):
            return blocked(f"Required stage {stage_id} cannot be skipped.", "Complete or block the required stage instead.", stage_id=stage_id)
        return {"ok": True, "state_patch": {"phase": {"id": stage_id, "label": f"Skipped optional stage: {stage_id}", "status": "skipped"}}}
    return {"ok": True}


def handle_pre_finish(context: dict[str, Any]) -> dict[str, Any]:
    event = event_value(context)
    finish = event.get("finish") if is_record(event.get("finish")) else {}
    manifest = event.get("manifest") if is_record(event.get("manifest")) else {}
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), list) else []
    report, path, error = load_artifact_json(context, artifacts, "novelty_report")
    if error:
        return blocked(error, "Create a valid novelty_report artifact and call xlab_finish again.")
    assert report is not None
    success_gate = finish.get("status") == "success"
    errors, counters = validate_novelty_report(report, success_gate=success_gate)
    if errors:
        repair = "Expand collection, regenerate the idea, or finish as incomplete when evidence remains low or risk remains high."
        return blocked(" ".join(errors), repair)
    return {
        "ok": True,
        "state_patch": {
            "phase": {"id": "validate", "label": "Research workflow validated", "status": "success", "progress": 1},
            "message": "Final novelty report satisfies the workflow finish gate.",
            "counters": counters,
            "artifacts": [{"type": "novelty_report", "name": "novelty_report", "path": str(path), "status": "ready"}],
        },
    }


def handle_resume(context: dict[str, Any]) -> dict[str, Any]:
    workflow = read_workflow(context)
    stages = workflow_stages(workflow)
    running = next((stage for stage in STAGE_CHAIN if stages.get(stage, {}).get("status") == "running"), None)
    blocked_stage = next((stage for stage in STAGE_CHAIN if stages.get(stage, {}).get("status") in {"blocked", "failed"}), None)
    next_stage = running or next_runnable_stage(workflow)
    next_actions: list[str]
    if running:
        next_actions = [f"Continue {running}, then call xlab_stage action=complete or action=block."]
    elif blocked_stage:
        next_actions = [f"Repair blocked stage {blocked_stage}, then resume or start a fresh upstream attempt."]
    elif next_stage:
        next_actions = [f"Start next stage with xlab_stage action=start stage_id={next_stage}."]
    else:
        next_actions = ["All required stages appear complete; call xlab_finish with the final manifest."]
    counters = count_stage_statuses(workflow)
    diagnostics = []
    if blocked_stage:
        diagnostics.append({
            "severity": "warning",
            "code": "workflow_blocked_stage",
            "message": f"Stage {blocked_stage} is {stages.get(blocked_stage, {}).get('status')}.",
            "repair": stages.get(blocked_stage, {}).get("error") or "Inspect stage artifacts and rerun the blocked stage.",
        })
    return {
        "ok": True,
        "state_patch": {
            "phase": {"id": next_stage or "finish", "label": "Research workflow resume point", "status": "running" if next_stage else "success"},
            "counters": counters,
            "diagnostics": diagnostics,
            "checkpoint": {"id": next_stage or "finish", "label": "Workflow resume state", "resumable": bool(next_stage), "data": {"workflow": stage_summary(workflow)}},
            "next_actions": next_actions,
        },
    }


def next_runnable_stage(workflow: dict[str, Any] | None) -> str | None:
    states = workflow_stages(workflow)
    for stage_id in STAGE_CHAIN:
        state = states.get(stage_id, {})
        if state.get("status") != "pending":
            continue
        needs = state.get("needs") if isinstance(state.get("needs"), list) else []
        if all(states.get(str(need), {}).get("status") == "success" for need in needs):
            return stage_id
    return None


def count_stage_statuses(workflow: dict[str, Any] | None) -> dict[str, int]:
    counts = {"pending": 0, "running": 0, "success": 0, "failed": 0, "blocked": 0, "skipped": 0}
    for stage in workflow_stages(workflow).values():
        status = stage.get("status")
        if status in counts:
            counts[status] += 1
    return counts


def handle_cancel(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "state_patch": {
            "phase": {"id": "cancel", "label": "Research workflow cancelled", "status": "skipped"},
            "checkpoint": {"id": "cancel", "label": "Cancelled by user", "resumable": False},
        },
    }


def handle_error(context: dict[str, Any]) -> dict[str, Any]:
    event = event_value(context)
    message = event.get("message") or event.get("error") or "Research workflow error."
    return {
        "ok": True,
        "state_patch": {
            "phase": {"id": "error", "label": "Research workflow error", "status": "failed"},
            "diagnostics": [{"severity": "error", "code": "workflow_error", "message": str(message), "data": event}],
            "next_actions": ["Inspect workflow.json and stage artifacts, repair the failing stage, then use /xlab resume-run <run-id>."],
        },
    }


def blocked(message: str, repair: str, *, stage_id: str | None = None, workflow: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "ok": False,
        "message": message,
        "repair": repair,
        "state_patch": {
            "phase": {"id": stage_id or "validate", "label": "Research workflow gate", "status": "blocked"},
            "diagnostics": [{"severity": "error", "code": "research_workflow_gate", "message": message, "repair": repair}],
            "checkpoint": {"id": stage_id or "validate", "label": "Blocked workflow gate", "resumable": True, "data": {"workflow": stage_summary(workflow)} if workflow else {}},
            "next_actions": [repair],
        },
    }


def dispatch(context: dict[str, Any]) -> dict[str, Any]:
    point = str(context.get("point") or "pre_finish")
    if point == "pre_start":
        return handle_pre_start(context)
    if point == "pre_stage":
        return handle_pre_stage(context)
    if point == "post_stage":
        return handle_post_stage(context)
    if point == "pre_finish":
        return handle_pre_finish(context)
    if point == "on_resume":
        return handle_resume(context)
    if point == "on_cancel":
        return handle_cancel(context)
    if point == "on_error":
        return handle_error(context)
    return {"ok": True}


def main() -> None:
    try:
        context = json.load(sys.stdin)
        if not is_record(context):
            emit({"ok": False, "message": "Hook context must be a JSON object."})
            return
        emit(dispatch(context))
    except Exception as error:  # pragma: no cover - command hooks must fail closed.
        emit({
            "ok": False,
            "message": f"Research workflow hook failed: {error}",
            "repair": "Inspect xlab/skills/research_workflow/hooks/validate.py and rerun the hook after fixing the validator error.",
            "state_patch": {"phase": {"id": "validate", "label": "Research workflow hook failed", "status": "failed"}},
        })


if __name__ == "__main__":
    main()
