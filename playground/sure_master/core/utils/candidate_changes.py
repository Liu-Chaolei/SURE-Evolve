from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "candidate_type",
    "idea_text",
    "changed_fields",
    "arch_config",
    "training_config",
    "inference_config",
    "defaults",
    "diff_from_defaults",
    "produced_artifacts",
}

METRIC_OUTPUT_FILES = (
    "score_summary.json",
    "metric_error.json",
    "metric_gpu_attempts.json",
    "pipeline_spec.json",
    "remote_training_result.json",
)
ARTIFACT_OUTPUT_FILES = (
    "hyp.txt",
    "samples.jsonl",
    "sample_output.jsonl",
    "candidate_status.json",
)
INPUT_ROLES = {
    "ref",
    "source",
    "prompt",
    "prompts",
    "input",
    "text",
    "audio",
    "wav",
    "wavs",
}


def candidate_changes_path(workspace: str | Path) -> Path:
    return Path(workspace) / "artifacts" / "candidate_changes.json"


def validate_candidate_changes(workspace: str | Path) -> list[str]:
    """Validate the structured candidate change record."""
    path = candidate_changes_path(workspace)
    if not path.is_file():
        return ["candidate change record is missing: artifacts/candidate_changes.json"]

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"candidate change record is invalid JSON: {exc}"]
    except OSError as exc:
        return [f"candidate change record could not be read: {exc}"]

    if not isinstance(payload, dict):
        return ["candidate change record must be a JSON object"]

    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS.difference(payload))
    if missing:
        errors.append("candidate change record is missing required field(s): " + ", ".join(missing))

    if "candidate_type" in payload and not str(payload.get("candidate_type") or "").strip():
        errors.append("candidate change record candidate_type must be non-empty")
    if "idea_text" in payload and not isinstance(payload.get("idea_text"), str):
        errors.append("candidate change record idea_text must be a string")
    if "changed_fields" in payload and not isinstance(payload.get("changed_fields"), list):
        errors.append("candidate change record changed_fields must be a list")
    for field in (
        "arch_config",
        "training_config",
        "inference_config",
        "defaults",
        "diff_from_defaults",
        "produced_artifacts",
    ):
        if field in payload and not isinstance(payload.get(field), dict):
            errors.append(f"candidate change record {field} must be an object")
    return errors


def validate_arch_candidate_changes(workspace: str | Path) -> list[str]:
    """Require architecture candidates to disclose a real architecture diff."""
    errors = validate_candidate_changes(workspace)
    if errors:
        return errors

    path = candidate_changes_path(workspace)
    payload = json.loads(path.read_text(encoding="utf-8"))
    normalized = normalize_arch_candidate_changes_payload(payload)
    if normalized is not payload:
        write_candidate_changes(path, normalized)
        payload = normalized
    candidate_type = str(payload.get("candidate_type") or "").strip().lower()
    if candidate_type not in {"arch", "architecture", "structure"}:
        errors.append("architecture candidate change record candidate_type must be arch")

    arch_config = payload.get("arch_config")
    if not isinstance(arch_config, dict) or not arch_config:
        errors.append("architecture candidate change record arch_config must contain changed structure values")

    changed_fields = payload.get("changed_fields")
    has_arch_field = (
        isinstance(changed_fields, list)
        and any(str(field).startswith("arch_config.") for field in changed_fields)
    )
    diff = payload.get("diff_from_defaults")
    arch_diff = diff.get("arch_config") if isinstance(diff, dict) else None
    if not has_arch_field and not (isinstance(arch_diff, dict) and arch_diff):
        errors.append(
            "architecture candidate change record must include arch_config diff_from_defaults "
            "or changed_fields entries"
        )
    return errors


def normalize_arch_candidate_changes_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Backfill the nested arch diff shape for legacy F5-TTS arch records."""
    if not isinstance(payload, dict):
        return payload
    changed_fields = payload.get("changed_fields")
    if not isinstance(changed_fields, list):
        return payload
    arch_fields = [
        str(field).split(".", 1)[1]
        for field in changed_fields
        if str(field).startswith("arch_config.") and "." in str(field)
    ]
    if not arch_fields:
        return payload

    arch_config = payload.get("arch_config")
    if not isinstance(arch_config, dict) or not arch_config:
        return payload

    diff = payload.get("diff_from_defaults")
    if not isinstance(diff, dict):
        diff = {}
    arch_diff = diff.get("arch_config")
    if isinstance(arch_diff, dict) and arch_diff:
        return payload

    defaults = payload.get("defaults")
    default_arch = defaults.get("arch_config") if isinstance(defaults, dict) else {}
    if not isinstance(default_arch, dict):
        default_arch = {}

    backfilled = {
        key: {
            "default": default_arch.get(key),
            "value": arch_config.get(key),
        }
        for key in arch_fields
        if key in arch_config
    }
    if not backfilled:
        return payload

    normalized = dict(payload)
    normalized_diff = dict(diff)
    normalized_diff["arch_config"] = backfilled
    normalized["diff_from_defaults"] = normalized_diff
    return normalized


def write_candidate_changes(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        target,
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
    )


def clear_candidate_outputs(
    workspace: str | Path,
    role_paths: dict[str, str | None] | None = None,
) -> None:
    """Remove stale candidate outputs before executing a new candidate."""
    root = Path(workspace)
    artifacts = root / "artifacts"
    metric = root / "metric"
    for name in ARTIFACT_OUTPUT_FILES:
        _unlink_if_file(artifacts / name)
    for name in METRIC_OUTPUT_FILES:
        _unlink_if_file(metric / name)

    if role_paths:
        for role, value in role_paths.items():
            if value is None:
                continue
            role_name = str(role).strip().lower()
            path_text = str(value)
            if role_name in INPUT_ROLES or path_text.startswith("literal:"):
                continue
            path = Path(path_text)
            if not path.is_absolute():
                path = root / path
            try:
                resolved = path.resolve()
                if root.resolve() not in {resolved, *resolved.parents}:
                    continue
            except OSError:
                pass
            _unlink_if_file(path)


def write_candidate_status(
    workspace: str | Path,
    *,
    success: bool,
    reason_code: str,
    stage: str,
    candidate_type: str,
    score: float | None = None,
    score_valid: bool = False,
    metric_accepted: bool = False,
    remote_success: bool | None = None,
    rung: str | None = None,
    phase: str | None = None,
    idea_id: str | None = None,
    missing_artifacts: list[str] | None = None,
    artifact_guard_errors: list[str] | None = None,
    execution_info: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
    metric_feedback: str = "",
    terminal_output: str = "",
) -> Path:
    """Write a compact machine-readable status for candidate triage."""
    root = Path(workspace)
    path = root / "artifacts" / "candidate_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "success": bool(success),
        "reason_code": reason_code,
        "stage": stage,
        "phase": phase or "",
        "rung": rung or "",
        "idea_id": idea_id or "",
        "candidate_type": candidate_type,
        "score": score,
        "score_valid": bool(score_valid),
        "metric_accepted": bool(metric_accepted),
        "remote_success": remote_success,
        "missing_artifacts": missing_artifacts or [],
        "artifact_guard_errors": artifact_guard_errors or [],
        "execution_info": execution_info or {},
        "details": details or {},
        "metric_feedback": metric_feedback,
        "terminal_output_tail": terminal_output[-12000:] if terminal_output else "",
        "written_at": int(time.time()),
    }
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
    )
    return path


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _unlink_if_file(path: Path) -> None:
    try:
        if path.is_file() or path.is_symlink():
            path.unlink()
    except FileNotFoundError:
        return
