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


def load_manifest(context: dict[str, Any]) -> tuple[dict[str, Any] | None, Path | None, str | None]:
    artifact = manifest_artifact(context, "communication_manifest")
    if not artifact:
        return None, None, "Final manifest is missing required artifact type: communication_manifest."
    path, error = resolve_allowed_path(context, artifact.get("path"))
    if error:
        return None, path, error
    assert path is not None
    value, read_error = read_json_file(path)
    if read_error:
        return None, path, read_error
    if not is_record(value):
        return None, path, "communication_manifest must be a JSON object."
    return value, path, None


def handle_lifecycle(context: dict[str, Any]) -> dict[str, Any]:
    point = str(context.get("point") or "pre_finish")
    if point == "pre_start":
        return {"ok": True, "state_patch": {"phase": {"id": "gateway", "label": "Science gateway ready", "status": "running", "progress": 0}, "message": "Gateway runs a local stdio MCP forum/direct-message service with auditable SQLite storage.", "checkpoint": {"id": "gateway", "label": "Science gateway lifecycle", "resumable": True}, "next_actions": ["Start the service with a run-local database path.", "Export communication_manifest only after reading back messages and threads."]}}
    if point == "on_resume":
        return {"ok": True, "state_patch": {"phase": {"id": "resume", "label": "Science gateway resume point", "status": "running"}, "checkpoint": {"id": "gateway", "label": "Gateway resume state", "resumable": True}, "next_actions": ["Continue gateway operation or export a communication_manifest for finish validation."]}}
    if point == "on_cancel":
        return {"ok": True, "state_patch": {"phase": {"id": "cancel", "label": "Science gateway cancelled", "status": "skipped"}, "checkpoint": {"id": "cancel", "resumable": False}}}
    if point == "on_error":
        event = event_value(context)
        return {"ok": True, "state_patch": {"phase": {"id": "error", "label": "Science gateway error", "status": "failed"}, "diagnostics": [{"severity": "error", "code": "gateway_error", "message": str(event.get("message") or event.get("error") or "Science gateway error."), "data": event}], "next_actions": ["Inspect the local database and service logs, then resume or export the communication manifest."]}}
    return {"ok": True}


def blocked(message: str, repair: str) -> dict[str, Any]:
    return {"ok": False, "message": message, "repair": repair, "state_patch": {"phase": {"id": "validate", "label": "Communication records gate", "status": "blocked"}, "diagnostics": [{"severity": "error", "code": "gateway_gate", "message": message, "repair": repair}], "next_actions": [repair]}}


def validate_finish(context: dict[str, Any]) -> dict[str, Any]:
    manifest, path, error = load_manifest(context)
    if error:
        return blocked(error, "Read back messages, export the gateway database, and call xlab_finish again.")
    assert manifest is not None
    database, database_error = resolve_allowed_path(context, manifest.get("database"))
    messages = manifest.get("messages")
    threads = manifest.get("threads")
    valid = (
        manifest.get("schema_version") == "xlab.communication_manifest.v1"
        and database_error is None
        and isinstance(messages, list)
        and isinstance(threads, list)
        and manifest.get("message_count") == len(messages)
        and manifest.get("thread_count") == len(threads)
    )
    if not valid:
        detail = database_error or "Communication export is missing, inconsistent, or not backed by a readable database."
        return blocked(detail, "Read back messages, export the gateway database, and call xlab_finish again.")
    return {"ok": True, "state_patch": {"phase": {"id": "validate", "label": "Communication records validated", "status": "success", "progress": 1}, "counters": {"threads": int(manifest["thread_count"]), "messages": int(manifest["message_count"])}, "artifacts": [{"type": "communication_manifest", "name": "communication_manifest", "path": str(path), "status": "ready"}]}}


def dispatch(context: dict[str, Any]) -> dict[str, Any]:
    if str(context.get("point") or "pre_finish") != "pre_finish":
        return handle_lifecycle(context)
    return validate_finish(context)


def main() -> None:
    try:
        context = json.load(sys.stdin)
        emit(dispatch(context if is_record(context) else {}))
    except Exception as error:
        emit({"ok": False, "message": f"Science gateway hook failed: {error}", "repair": "Fix hooks/validate.py and rerun the hook."})


if __name__ == "__main__":
    main()
