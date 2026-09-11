#!/usr/bin/env python3
import json
import shlex
import sys
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = (
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
ARRAY_FIELDS = ("idea_claims", "closest_work", "overlap", "differences", "evidence_gaps", "recommendations")
RISK_LEVELS = {"low", "medium", "high", "unknown"}
CONFIDENCE_LEVELS = {"low", "medium", "high"}
COMMON_REPORT_PATHS = (
    "artifacts/novelty_report.json",
    "artifacts/novelty.json",
    "novelty_report.json",
)


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


def resolve_allowed_path(context: dict[str, Any], path_value: Any, *, must_exist: bool = True) -> tuple[Path | None, str | None]:
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
        if not must_exist or candidate.is_file():
            return candidate, None
    if first_allowed:
        if must_exist:
            return first_allowed, f"Artifact path does not exist: {path_value}."
        return first_allowed, None
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


def find_artifact(artifacts: Any, artifact_type: str) -> dict[str, Any] | None:
    return next((item for item in artifact_list(artifacts) if item.get("type") == artifact_type), None)


def state_artifacts(context: dict[str, Any]) -> list[dict[str, Any]]:
    event = event_value(context)
    state = event.get("state") if is_record(event.get("state")) else {}
    artifacts = state.get("artifacts") if isinstance(state.get("artifacts"), list) else []
    return artifact_list(artifacts)


def load_report_from_manifest(context: dict[str, Any]) -> tuple[dict[str, Any] | None, Path | None, str | None]:
    event = event_value(context)
    manifest = event.get("manifest") if is_record(event.get("manifest")) else {}
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), list) else []
    artifact = find_artifact(artifacts, "novelty_report")
    if not artifact:
        return None, None, "Final manifest is missing required artifact type: novelty_report."
    return load_report_path(context, artifact.get("path"))


def load_report_path(context: dict[str, Any], path_value: Any) -> tuple[dict[str, Any] | None, Path | None, str | None]:
    path, error = resolve_allowed_path(context, path_value)
    if error:
        return None, path, error
    assert path is not None
    value, read_error = read_json_file(path)
    if read_error:
        return None, path, read_error
    if not is_record(value):
        return None, path, "Novelty report must be a JSON object."
    return value, path, None


def discover_existing_report(context: dict[str, Any]) -> tuple[dict[str, Any] | None, Path | None, str | None]:
    for artifact in state_artifacts(context):
        if artifact.get("type") == "novelty_report" and artifact.get("path"):
            report, path, error = load_report_path(context, artifact.get("path"))
            if report is not None or error:
                return report, path, error
    _, run_dir = context_paths(context)
    for relative in COMMON_REPORT_PATHS:
        path = (run_dir / relative).resolve()
        if path.is_file():
            value, error = read_json_file(path)
            if error:
                return None, path, error
            if not is_record(value):
                return None, path, "Novelty report must be a JSON object."
            return value, path, None
    return None, None, "No novelty report artifact found yet."


def parse_flag_values(args_text: str) -> dict[str, str]:
    try:
        tokens = shlex.split(args_text)
    except ValueError:
        return {}
    flags: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            if "=" in key:
                key, value = key.split("=", 1)
                flags[key] = value
            elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                flags[key] = tokens[index + 1]
                index += 1
            else:
                flags[key] = ""
        index += 1
    return flags


def path_like(value: str) -> bool:
    text = value.strip()
    return bool(text) and (
        "/" in text
        or "\\" in text
        or text.startswith(".")
        or text.endswith((".json", ".jsonl", ".md", ".yaml", ".yml"))
    )


def validate_json_input(context: dict[str, Any], label: str, value: str, errors: list[str], warnings: list[str]) -> None:
    if not path_like(value):
        warnings.append(f"{label} looks like a workspace handle rather than a path; resolve it to a concrete artifact before running scripts/check.py.")
        return
    path, error = resolve_allowed_path(context, value)
    if error:
        errors.append(f"{label} is not readable: {error}")
    elif path:
        json_value, read_error = read_json_file(path)
        if read_error:
            errors.append(f"{label} is invalid JSON: {read_error}")
        elif not is_record(json_value):
            errors.append(f"{label} must be a JSON object.")


def validate_input_paths(context: dict[str, Any]) -> tuple[list[str], list[str]]:
    flags = parse_flag_values(str(context.get("args") or ""))
    errors: list[str] = []
    warnings: list[str] = []
    idea_path = flags.get("idea") or flags.get("idea_path") or flags.get("idea_artifact")
    papers_path = flags.get("papers") or flags.get("paper_set") or flags.get("papers_path")
    graph_path = flags.get("graph") or flags.get("graph_path")
    # Slash-command args may be natural language, workspace handles, or
    # parent-workflow context rather than concrete file flags. Missing or
    # handle-like paths are guidance warnings; invalid concrete paths are blockers.
    if idea_path:
        validate_json_input(context, "Idea artifact", idea_path, errors, warnings)
    else:
        warnings.append("No --idea path was provided in slash-command args; ensure the prompt includes a concrete idea artifact before running scripts/check.py.")
    evidence_path = papers_path or graph_path
    if evidence_path:
        validate_json_input(context, "Evidence artifact", evidence_path, errors, warnings)
    else:
        warnings.append("No --papers or --graph evidence path was provided; low retrieval coverage must finish as incomplete.")
    return errors, warnings


def validate_report(report: dict[str, Any], finish_status: str | None) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_FIELDS if field not in report]
    if missing:
        errors.append(f"Novelty report is missing fields: {', '.join(missing)}.")
    if report.get("schema_version") != "xlab.novelty_report.v1":
        errors.append("Novelty report schema_version must be xlab.novelty_report.v1.")
    for field in ARRAY_FIELDS:
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
    if finish_status == "success":
        if report.get("confidence") == "low" or report.get("risk_level") == "unknown":
            errors.append("Low-confidence or unknown novelty evidence must finish as incomplete.")
        if report.get("risk_level") == "high":
            errors.append("High duplication risk cannot finish as success; preserve the report and generate a new idea attempt or finish incomplete.")
    counters = {
        "idea_claims": len(report.get("idea_claims")) if isinstance(report.get("idea_claims"), list) else 0,
        "closest_work": len(report.get("closest_work")) if isinstance(report.get("closest_work"), list) else 0,
        "evidence_gaps": len(report.get("evidence_gaps")) if isinstance(report.get("evidence_gaps"), list) else 0,
        "overlap": len(report.get("overlap")) if isinstance(report.get("overlap"), list) else 0,
    }
    return errors, counters


def handle_pre_start(context: dict[str, Any]) -> dict[str, Any]:
    errors, warnings = validate_input_paths(context)
    diagnostics = [
        {"severity": "warning", "code": "novelty_input_warning", "message": warning}
        for warning in warnings
    ]
    if errors:
        message = " ".join(errors)
        return blocked(message, "Provide a run-local idea artifact plus --papers or --graph evidence, then rerun /xlab check-idea-novelty.", diagnostics=diagnostics)
    return {
        "ok": True,
        "state_patch": {
            "phase": {"id": "preflight", "label": "Novelty check preflight", "status": "running", "progress": 0},
            "message": "Novelty check must ground duplication risk in an idea artifact and paper or graph evidence.",
            "diagnostics": diagnostics,
            "checkpoint": {
                "id": "novelty_inputs",
                "label": "Novelty input contract",
                "resumable": True,
                "resume_hint": "Use /xlab resume-run <run-id> after repairing missing evidence or report fields.",
            },
            "next_actions": [
                "Run scripts/check.py with --idea, --papers, and --output inside the project or run workspace.",
                "Inspect graph relations and cite evidence for overlap, differences, and gaps.",
                "Finish as incomplete when confidence is low or risk is unknown/high.",
            ],
        },
    }


def handle_pre_finish(context: dict[str, Any]) -> dict[str, Any]:
    finish = event_value(context).get("finish") if is_record(event_value(context).get("finish")) else {}
    finish_status = str(finish.get("status") or "")
    report, path, error = load_report_from_manifest(context)
    if error:
        return blocked(error, "Create a valid novelty_report artifact and call xlab_finish again.")
    assert report is not None
    errors, counters = validate_report(report, finish_status)
    if errors:
        return blocked(" ".join(errors), "Expand evidence or finish the run as incomplete.")
    return {
        "ok": True,
        "state_patch": {
            "phase": {"id": "validate", "label": "Novelty evidence validated", "status": "success", "progress": 1},
            "message": "Novelty report shape, confidence, and risk gates passed for the requested finish status.",
            "counters": counters,
            "artifacts": [{"type": "novelty_report", "name": "novelty_report", "path": str(path), "status": "ready"}],
        },
    }


def handle_resume(context: dict[str, Any]) -> dict[str, Any]:
    report, path, error = discover_existing_report(context)
    diagnostics: list[dict[str, Any]] = []
    next_actions: list[str]
    counters: dict[str, int] = {}
    status = "running"
    if error:
        diagnostics.append({"severity": "warning", "code": "novelty_report_missing", "message": error, "repair": "Run scripts/check.py and inspect graph evidence before finishing."})
        next_actions = ["Generate artifacts/novelty_report.json, then call xlab_finish with a manifest containing novelty_report."]
    else:
        assert report is not None
        errors, counters = validate_report(report, None)
        if errors:
            diagnostics.append({"severity": "error", "code": "novelty_report_invalid", "message": " ".join(errors), "repair": "Repair the novelty report fields before finishing."})
            next_actions = ["Repair novelty report shape and evidence fields, then resume."]
        else:
            status = "success"
            next_actions = ["Call xlab_finish with success if confidence/risk are acceptable, otherwise finish incomplete with the report preserved."]
    return {
        "ok": True,
        "state_patch": {
            "phase": {"id": "resume", "label": "Novelty check resume point", "status": status},
            "counters": counters,
            "diagnostics": diagnostics,
            "checkpoint": {"id": "novelty_report", "label": "Novelty report state", "resumable": status != "success", "data": {"path": str(path) if path else None}},
            "next_actions": next_actions,
        },
    }


def handle_cancel(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "state_patch": {
            "phase": {"id": "cancel", "label": "Novelty check cancelled", "status": "skipped"},
            "checkpoint": {"id": "cancel", "label": "Cancelled by user", "resumable": False},
        },
    }


def handle_error(context: dict[str, Any]) -> dict[str, Any]:
    event = event_value(context)
    report, path, error = discover_existing_report(context)
    diagnostics = [{"severity": "error", "code": "novelty_error", "message": str(event.get("message") or event.get("error") or "Novelty check error."), "data": event}]
    if error:
        diagnostics.append({"severity": "warning", "code": "novelty_report_state", "message": error})
    elif report is not None:
        report_errors, _ = validate_report(report, None)
        if report_errors:
            diagnostics.append({"severity": "error", "code": "novelty_report_invalid", "message": " ".join(report_errors)})
        else:
            diagnostics.append({"severity": "info", "code": "novelty_report_available", "message": f"Existing novelty report is available at {path}."})
    return {
        "ok": True,
        "state_patch": {
            "phase": {"id": "error", "label": "Novelty check error", "status": "failed"},
            "diagnostics": diagnostics,
            "next_actions": ["Repair the reported issue, regenerate or validate the novelty report, then use /xlab resume-run <run-id>."],
        },
    }


def blocked(message: str, repair: str, *, diagnostics: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    all_diagnostics = diagnostics or []
    all_diagnostics.append({"severity": "error", "code": "novelty_gate", "message": message, "repair": repair})
    return {
        "ok": False,
        "message": message,
        "repair": repair,
        "state_patch": {
            "phase": {"id": "validate", "label": "Validating novelty evidence", "status": "blocked"},
            "diagnostics": all_diagnostics,
            "checkpoint": {"id": "novelty_gate", "label": "Blocked novelty gate", "resumable": True},
            "next_actions": [repair],
        },
    }


def dispatch(context: dict[str, Any]) -> dict[str, Any]:
    point = str(context.get("point") or "pre_finish")
    if point == "pre_start":
        return handle_pre_start(context)
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
            "message": f"Novelty hook failed: {error}",
            "repair": "Inspect xlab/skills/novelty_check/hooks/validate.py and rerun after fixing the validator error.",
            "state_patch": {"phase": {"id": "validate", "label": "Novelty hook failed", "status": "failed"}},
        })


if __name__ == "__main__":
    main()
