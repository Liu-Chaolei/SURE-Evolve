#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import Any


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


def artifact_list(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if is_record(item)] if isinstance(value, list) else []


def manifest_artifact(context: dict[str, Any], artifact_type: str) -> dict[str, Any] | None:
    manifest = event_value(context).get("manifest") if is_record(event_value(context).get("manifest")) else {}
    return next((item for item in artifact_list(manifest.get("artifacts")) if item.get("type") == artifact_type), None)


def load_artifact_json(context: dict[str, Any], artifact_type: str) -> tuple[dict[str, Any] | None, Path | None, str | None]:
    artifact = manifest_artifact(context, artifact_type)
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


def read_workflow(context: dict[str, Any]) -> dict[str, Any] | None:
    _, run_dir = context_paths(context)
    path = run_dir / "workflow.json"
    if not path.is_file():
        return None
    value, error = read_json_file(path)
    return value if not error and is_record(value) else None


def workflow_stages(workflow: dict[str, Any] | None) -> list[dict[str, Any]]:
    stages = workflow.get("stages") if workflow and isinstance(workflow.get("stages"), list) else []
    return [stage for stage in stages if is_record(stage)]


def declared_chain(context: dict[str, Any]) -> list[str]:
    skill = context.get("skill") if is_record(context.get("skill")) else {}
    workflow = skill.get("workflow") if is_record(skill.get("workflow")) else {}
    stages = workflow.get("stages") if isinstance(workflow.get("stages"), list) else []
    return [str(stage.get("id")) for stage in stages if is_record(stage) and stage.get("id")]


def count_statuses(workflow: dict[str, Any] | None) -> dict[str, int]:
    counts = {"pending": 0, "running": 0, "success": 0, "failed": 0, "blocked": 0, "skipped": 0}
    for stage in workflow_stages(workflow):
        status = stage.get("status")
        if status in counts:
            counts[status] += 1
    return counts


def lifecycle(context: dict[str, Any]) -> dict[str, Any]:
    point = str(context.get("point") or "pre_finish")
    workflow = read_workflow(context)
    chain = declared_chain(context)
    if point == "pre_start":
        return {"ok": True, "state_patch": {"phase": {"id": "workflow", "label": "Model MCP workflow ready", "status": "running", "progress": 0}, "message": f"Workflow stages: {' → '.join(chain)}.", "checkpoint": {"id": "workflow", "label": "Model MCP lifecycle", "resumable": True, "data": {"stages": chain}}, "next_actions": ["Wrap the model command into a package, validate the stdio lifecycle, and register only after validation passes.", "Use xlab_stage at every declared stage boundary."]}}
    if point == "pre_stage":
        stage_id = str(event_value(context).get("stage_id") or "stage")
        if stage_id == "register":
            return {"ok": True, "state_patch": {"phase": {"id": stage_id, "label": "Starting optional MCP registration", "status": "running"}, "diagnostics": [{"severity": "warning", "code": "external_mutation", "message": "Registration mutates local MCP configuration; confirm the validated package and backup path before proceeding."}], "checkpoint": {"id": stage_id, "label": "Stage gate", "resumable": True}}}
        return {"ok": True, "state_patch": {"phase": {"id": stage_id, "label": f"Starting {stage_id}", "status": "running"}, "checkpoint": {"id": stage_id, "label": "Stage gate", "resumable": True}}}
    if point == "post_stage":
        event = event_value(context)
        stage_id = str(event.get("stage_id") or "stage")
        action = str(event.get("action") or "")
        if action == "complete" and not artifact_list(event.get("artifacts")):
            return blocked(f"Stage {stage_id} completion requires a typed artifact list.", "Call xlab_stage action=complete with the artifacts emitted by the packaged stage.")
        status = "success" if action == "complete" else "blocked" if action == "block" else "failed" if action == "fail" else "skipped"
        return {"ok": True, "state_patch": {"phase": {"id": stage_id, "label": f"Stage {stage_id} {action}", "status": status}, "counters": count_statuses(workflow)}}
    if point == "on_resume":
        return {"ok": True, "state_patch": {"phase": {"id": "resume", "label": "Model MCP resume point", "status": "running"}, "counters": count_statuses(workflow), "checkpoint": {"id": "workflow", "label": "Workflow resume state", "resumable": True, "data": {"stages": workflow_stages(workflow)}}, "next_actions": ["Resume the next pending or blocked MCP packaging stage."]}}
    if point == "on_cancel":
        return {"ok": True, "state_patch": {"phase": {"id": "cancel", "label": "Model MCP workflow cancelled", "status": "skipped"}, "checkpoint": {"id": "cancel", "resumable": False}}}
    if point == "on_error":
        event = event_value(context)
        return {"ok": True, "state_patch": {"phase": {"id": "error", "label": "Model MCP workflow error", "status": "failed"}, "diagnostics": [{"severity": "error", "code": "workflow_error", "message": str(event.get("message") or event.get("error") or "Model MCP workflow error."), "data": event}], "next_actions": ["Repair the failing stage, then use /xlab resume-run <run-id>."]}}
    return {"ok": True}


def validate_package_paths(context: dict[str, Any], package: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("server", "config", "input_schema", "example_input"):
        path, error = resolve_allowed_path(context, package.get(field))
        if error:
            errors.append(f"MCP package field {field}: {error}")
    return errors


def blocked(message: str, repair: str) -> dict[str, Any]:
    return {"ok": False, "message": message, "repair": repair, "state_patch": {"phase": {"id": "validate", "label": "Model MCP gate", "status": "blocked"}, "diagnostics": [{"severity": "error", "code": "mcp_gate", "message": message, "repair": repair}], "next_actions": [repair]}}


def validate_finish(context: dict[str, Any]) -> dict[str, Any]:
    package, package_path, package_error = load_artifact_json(context, "mcp_package")
    if package_error:
        return blocked(package_error, "Generate a valid mcp_package artifact before finishing.")
    validation, validation_path, validation_error = load_artifact_json(context, "mcp_validation")
    if validation_error:
        return blocked(validation_error, "Run mcp_validate against the generated package before finishing.")
    assert package is not None and validation is not None
    errors = validate_package_paths(context, package)
    if package.get("schema_version") != "xlab.mcp_package.v1":
        errors.append("MCP package schema_version must be xlab.mcp_package.v1.")
    if validation.get("schema_version") != "xlab.mcp_validation.v1":
        errors.append("MCP validation schema_version must be xlab.mcp_validation.v1.")
    if validation.get("passed") is not True:
        errors.append("MCP validation did not pass.")
    if validation.get("tool_name") != package.get("tool_name"):
        errors.append("MCP validation tool_name does not match the generated package.")
    for field in ("initialized", "listed", "called"):
        if validation.get(field) is not True:
            errors.append(f"MCP validation field {field} must be true.")
    if errors:
        return blocked(" ".join(errors), "Run mcp_validate against the generated package before finishing.")
    return {"ok": True, "state_patch": {"phase": {"id": "validate", "label": "Model MCP validated", "status": "success", "progress": 1}, "counters": {"validated_tools": 1}, "artifacts": [{"type": "mcp_package", "name": "mcp_package", "path": str(package_path), "status": "ready"}, {"type": "mcp_validation", "name": "mcp_validation", "path": str(validation_path), "status": "ready"}]}}


def dispatch(context: dict[str, Any]) -> dict[str, Any]:
    if str(context.get("point") or "pre_finish") != "pre_finish":
        return lifecycle(context)
    return validate_finish(context)


def main() -> None:
    try:
        context = json.load(sys.stdin)
        emit(dispatch(context if is_record(context) else {}))
    except Exception as error:
        emit({"ok": False, "message": f"Model MCP hook failed: {error}", "repair": "Fix hooks/validate.py and rerun the hook."})


if __name__ == "__main__":
    main()
