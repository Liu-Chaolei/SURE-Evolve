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


def handle_lifecycle(context: dict[str, Any]) -> dict[str, Any]:
    point = str(context.get("point") or "pre_finish")
    if point == "pre_start":
        return {"ok": True, "state_patch": {"phase": {"id": "discussion", "label": "Research discussion ready", "status": "running", "progress": 0}, "message": "Discussion must include multiple roles, cited turns, source artifacts, and actionable conclusions.", "checkpoint": {"id": "discussion", "label": "Discussion lifecycle", "resumable": True}, "next_actions": ["Collect source artifacts and role context before finalizing the discussion.", "Finish only after every turn has citations and actions are explicit."]}}
    if point == "on_resume":
        return {"ok": True, "state_patch": {"phase": {"id": "resume", "label": "Research discussion resume point", "status": "running"}, "checkpoint": {"id": "discussion", "label": "Discussion resume state", "resumable": True}, "next_actions": ["Continue the bounded discussion or finalize the research_discussion artifact."]}}
    if point == "on_cancel":
        return {"ok": True, "state_patch": {"phase": {"id": "cancel", "label": "Research discussion cancelled", "status": "skipped"}, "checkpoint": {"id": "cancel", "resumable": False}}}
    if point == "on_error":
        event = event_value(context)
        return {"ok": True, "state_patch": {"phase": {"id": "error", "label": "Research discussion error", "status": "failed"}, "diagnostics": [{"severity": "error", "code": "discussion_error", "message": str(event.get("message") or event.get("error") or "Research discussion error."), "data": event}], "next_actions": ["Repair the discussion artifact or source evidence, then use /xlab resume-run <run-id>."]}}
    return {"ok": True}


def blocked(message: str, repair: str) -> dict[str, Any]:
    return {"ok": False, "message": message, "repair": repair, "state_patch": {"phase": {"id": "validate", "label": "Research discussion gate", "status": "blocked"}, "diagnostics": [{"severity": "error", "code": "discussion_gate", "message": message, "repair": repair}], "next_actions": [repair]}}


def validate_finish(context: dict[str, Any]) -> dict[str, Any]:
    result, path, error = load_artifact_json(context, "research_discussion")
    if error:
        return blocked(error, "Create a valid research_discussion artifact before finishing.")
    assert result is not None
    roles = result.get("roles")
    turns = result.get("turns")
    actions = result.get("actions")
    source_artifacts = result.get("source_artifacts")
    valid = (
        isinstance(roles, list)
        and len(roles) >= 2
        and isinstance(turns, list)
        and len(turns) >= 2
        and isinstance(actions, list)
        and len(actions) >= 1
        and isinstance(source_artifacts, list)
        and len(source_artifacts) >= 1
        and all(is_record(turn) and turn.get("citations") for turn in turns)
    )
    if not valid:
        return blocked("Discussion lacks roles, cited turns, source artifacts, or actionable conclusions.", "Complete a bounded evidence-linked discussion before finishing.")
    return {"ok": True, "state_patch": {"phase": {"id": "validate", "label": "Research discussion validated", "status": "success", "progress": 1}, "counters": {"rounds": int(result.get("round_count", 0) or 0), "turns": len(turns), "actions": len(actions)}, "artifacts": [{"type": "research_discussion", "name": "research_discussion", "path": str(path), "status": "ready"}]}}


def dispatch(context: dict[str, Any]) -> dict[str, Any]:
    if str(context.get("point") or "pre_finish") != "pre_finish":
        return handle_lifecycle(context)
    return validate_finish(context)


def main() -> None:
    try:
        context = json.load(sys.stdin)
        emit(dispatch(context if is_record(context) else {}))
    except Exception as error:
        emit({"ok": False, "message": f"Research discussion hook failed: {error}", "repair": "Fix hooks/validate.py and rerun the hook."})


if __name__ == "__main__":
    main()
