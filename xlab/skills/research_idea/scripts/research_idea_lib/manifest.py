from __future__ import annotations

import re
import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .audit import ALGORITHM_PROVENANCE, algorithm_provenance, audit_artifacts
from .checkpoints import write_checkpoint
from .common import append_pipeline_event, read_json, run_paths, utc_now, write_portable_json

PACKAGE_VERSION = "2.0.0"
SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"
PUBLIC_ARTIFACT_SCHEMAS = {
    "research_idea.json": "idea.schema.json",
    "idea_result.json": "idea_result.schema.json",
    "idea_trace.json": "idea_trace.schema.json",
    "idea_report.json": "idea_report.schema.json",
}
_ABSOLUTE_PATH_RE = re.compile(
    r"(?:(?<![:A-Za-z0-9])/(?!/)|(?<![A-Za-z0-9])[A-Za-z]:[\\/])[^\s\"'<>]*"
)
_PUBLIC_INPUT_FIELDS = (
    "topic",
    "mature_idea",
    "refinement_scope",
    "discussion",
    "experiment_feedback",
    "resume",
)


def build_final_manifest(
    *,
    cwd: Path,
    run_dir: Path,
    run_id: str,
    skill_version: str,
    status: str,
    request: dict[str, Any],
    idea: dict[str, Any],
    idea_result: dict[str, Any],
    trace: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    artifacts_dir = run_dir / "artifacts"
    del cwd
    counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
    parent_artifact_ids = direct_parent_artifact_ids(request, idea, trace)
    inputs = public_manifest_inputs(request, idea, trace)
    return {
        "schema_version": "2",
        "run_id": run_id,
        "skill_name": "research_idea",
        "skill_version": skill_version,
        "status": status,
        "created_at": utc_now(),
        "inputs": inputs,
        "outputs": {
            "topic": idea.get("topic") or request.get("topic"),
            "title": idea_result.get("title"),
            "research_question": idea.get("research_question"),
            "reference_count": counts.get("references", 0),
            "evidence_count": counts.get("evidence", 0),
            "rag_hit_count": counts.get("rag_hits", 0),
            "component_count": counts.get("components", 0),
            "candidate_count": counts.get("candidates", 0),
            "mcts_iterations": counts.get("mcts_iterations", 0),
            "source_modes": idea_result.get("source_modes", []),
        },
        "validation": {
            "passed": report.get("passed") is True,
            "blocking_errors": report.get("blocking_errors", []),
            "warnings": report.get("warnings", []),
        },
        "artifacts": [
            {
                "type": "research_idea",
                "schema_version": "2",
                "path": portable_artifact_path(run_dir, artifacts_dir / "research_idea.json"),
                "parents": parent_artifact_ids,
            },
            {
                "type": "research_idea_result",
                "schema_version": "2",
                "path": portable_artifact_path(run_dir, artifacts_dir / "idea_result.json"),
                "parents": parent_artifact_ids,
                "metadata": {"title": idea_result.get("title")},
            },
            {
                "type": "research_idea_trace",
                "schema_version": "2",
                "path": portable_artifact_path(run_dir, artifacts_dir / "idea_trace.json"),
                "parents": parent_artifact_ids,
            },
            {
                "type": "research_idea_report",
                "schema_version": "2",
                "path": portable_artifact_path(run_dir, artifacts_dir / "idea_report.json"),
                "parents": parent_artifact_ids,
            },
        ],
        "trace": {
            "workflow_stages": len(trace.get("workflow_trace", [])) if isinstance(trace.get("workflow_trace"), list) else 0,
            "operation_events": len(trace.get("operation_trace", [])) if isinstance(trace.get("operation_trace"), list) else 0,
        },
    }


def public_manifest_inputs(
    request: dict[str, Any],
    idea: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    inputs = {key: sanitize_public_paths(request[key]) for key in _PUBLIC_INPUT_FIELDS if key in request}
    source_context = idea.get("source_context") if isinstance(idea.get("source_context"), dict) else trace.get("source_context") if isinstance(trace.get("source_context"), dict) else {}
    survey: dict[str, Any] = {}
    parent_ids = source_context.get("direct_parent_artifact_ids")
    if isinstance(parent_ids, list):
        survey["artifact_ids"] = sorted({str(value).strip() for value in parent_ids if str(value).strip()})
    logical_path = validated_logical_path(request.get("survey_path"))
    if logical_path is not None:
        survey["path"] = logical_path
    if survey:
        inputs["survey"] = survey
    return inputs


def validated_logical_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or "\\" in text or not re.fullmatch(r"[A-Za-z0-9._/-]+", text):
        return None
    return text


def sanitize_public_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: ({"private_context_digest": "sha256:" + hashlib.sha256(
                json.dumps(child, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
                if key == "task_context" and isinstance(child, dict) and set(child) != {"private_context_digest"}
                else sanitize_public_paths(child))
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [sanitize_public_paths(child) for child in value]
    if isinstance(value, str):
        # Adjacent slashes in source excerpts can expose a second match after
        # substitution. Reach a fixed point so publication remains idempotent.
        while _ABSOLUTE_PATH_RE.search(value):
            value = _ABSOLUTE_PATH_RE.sub("[redacted-path]", value)
        return value
    return value


def public_path_findings(value: Any, prefix: str = "") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            field = f"{prefix}.{key}" if prefix else str(key)
            findings.extend(public_path_findings(child, field))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(public_path_findings(child, f"{prefix}[{index}]"))
    elif isinstance(value, str) and _ABSOLUTE_PATH_RE.search(value):
        findings.append(prefix or "<root>")
    return findings


def validate_public_projection(artifacts: dict[str, dict[str, Any]], manifest: dict[str, Any]) -> None:
    findings: list[str] = []
    for filename, artifact in artifacts.items():
        findings.extend(f"{filename}:{field}" for field in public_path_findings(artifact))
    findings.extend(f"manifest.json:{field}" for field in public_path_findings(manifest))
    if findings:
        raise ValueError("PUBLIC_PATH_DISCLOSURE: refusing unsafe public projection at " + ", ".join(findings))


def direct_parent_artifact_ids(request: dict[str, Any], idea: dict[str, Any], trace: dict[str, Any]) -> list[str]:
    source_context = idea.get("source_context") if isinstance(idea.get("source_context"), dict) else trace.get("source_context") if isinstance(trace.get("source_context"), dict) else {}
    values: list[Any] = []
    for payload in (request, source_context):
        for key in ("direct_parent_artifact_ids", "parent_artifact_ids", "source_artifact_ids", "artifact_ids"):
            raw = payload.get(key)
            values.extend(raw if isinstance(raw, list) else [])
        for key in ("parent_artifact_id", "source_artifact_id", "artifact_id", "survey_artifact_id", "resource_id", "bundle_id"):
            values.append(payload.get(key))
    resources = source_context.get("resources")
    if isinstance(resources, dict):
        resource_values = resources.values()
    elif isinstance(resources, list):
        resource_values = resources
    else:
        resource_values = ()
    for resource in resource_values:
        if not isinstance(resource, dict):
            continue
        values.append(resource.get("resource_id"))
        raw_parents = resource.get("parent_artifact_ids")
        values.extend(raw_parents if isinstance(raw_parents, list) else [])
    return sorted({str(value).strip() for value in values if value is not None and str(value).strip()})


def portable_artifact_path(run_dir: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(run_dir.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Final artifact must be contained by the run directory: {resolved}") from error


def finalize_run(
    *,
    cwd: Path,
    run_dir: Path,
    run_id: str,
    skill_version: str,
    request: dict[str, Any],
    requested_status: str | None = None,
) -> dict[str, Any]:
    paths = run_paths(run_dir)
    skill_version = PACKAGE_VERSION
    report = audit_artifacts(run_dir)
    status = "success" if report.get("passed") is True else "incomplete"
    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []
    if requested_status == "success" and status != "success":
        warnings.append("Requested status success was ignored because audit did not pass.")
    elif requested_status == "incomplete" and status == "success":
        reason = "Run was explicitly finalized as incomplete."
        warnings.append("Requested status incomplete downgraded an otherwise successful idea manifest.")
        report["passed"] = False
        report["blocking_errors"] = [reason]
        status = "incomplete"
    report["status"] = "success" if report.get("passed") is True else "incomplete"
    report["warnings"] = sorted(set(str(warning) for warning in warnings))
    report = sanitize_public_paths(report)

    idea = sanitize_public_paths(load_optional_object(paths["research_idea_json"]))
    idea_result = sanitize_public_paths(load_optional_object(paths["idea_result_json"]))
    trace = sanitize_public_paths(load_optional_object(paths["idea_trace_json"]))
    completed_stages = canonical_stages(report.get("completed_stages"))
    blockers = [str(value) for value in report.get("blocking_errors", []) if str(value).strip()]
    provenance = algorithm_provenance(idea_result) or dict(ALGORITHM_PROVENANCE)
    migrate_public_artifact(
        idea_result,
        schema_version="xlab.research_idea.result.v2",
        status=status,
        blockers=blockers,
        completed_stages=completed_stages,
        provenance=provenance,
    )
    migrate_public_artifact(
        idea,
        schema_version="xlab.research_idea.v2",
        status=status,
        blockers=blockers,
        completed_stages=completed_stages,
        provenance=provenance,
    )
    migrate_public_artifact(
        trace,
        schema_version="xlab.research_idea.trace.v2",
        status=status,
        blockers=blockers,
        completed_stages=completed_stages,
        provenance=provenance,
    )
    report["blockers"] = blockers
    report["completed_stages"] = completed_stages
    report["algorithm_provenance"] = provenance
    checks = report.get("checks") if isinstance(report.get("checks"), dict) else {}
    report["checks"] = checks
    checks["generated_public_artifact_schemas"] = True
    checks["emitted_public_artifact_schemas"] = True

    artifacts = public_artifacts(idea=idea, idea_result=idea_result, trace=trace, report=report)
    schema_errors = validate_public_artifacts(artifacts)
    if schema_errors:
        status = "incomplete"
        schema_blockers = [f"Public artifact schema validation failed: {error}" for error in schema_errors]
        blockers = list(dict.fromkeys([*blockers, *schema_blockers]))
        provenance = dict(ALGORITHM_PROVENANCE)
        report["passed"] = False
        report["status"] = status
        report["blocking_errors"] = blockers
        report["blockers"] = blockers
        checks["generated_public_artifact_schemas"] = False
        for artifact, schema_version in (
            (idea_result, "xlab.research_idea.result.v2"),
            (idea, "xlab.research_idea.v2"),
            (trace, "xlab.research_idea.trace.v2"),
        ):
            migrate_public_artifact(
                artifact,
                schema_version=schema_version,
                status=status,
                blockers=blockers,
                completed_stages=completed_stages,
                provenance=provenance,
            )
        report["algorithm_provenance"] = provenance
        artifacts = public_artifacts(idea=idea, idea_result=idea_result, trace=trace, report=report)
        incomplete_errors = validate_public_artifacts(artifacts, grouped=True)
        for filename in incomplete_errors:
            artifacts[filename] = controlled_incomplete_artifact(
                filename,
                artifacts[filename],
                blockers=blockers,
                completed_stages=completed_stages,
            )
        idea = artifacts["research_idea.json"]
        idea_result = artifacts["idea_result.json"]
        trace = artifacts["idea_trace.json"]
        report = artifacts["idea_report.json"]
        remaining_errors = validate_public_artifacts(artifacts)
        if remaining_errors:
            raise ValueError("Controlled incomplete artifacts failed schema validation: " + "; ".join(remaining_errors))

    artifacts = public_artifacts(idea=idea, idea_result=idea_result, trace=trace, report=report)
    remaining_errors = validate_public_artifacts(artifacts)
    if remaining_errors:
        raise ValueError("Final public artifacts failed schema validation: " + "; ".join(remaining_errors))
    manifest = build_final_manifest(
        cwd=cwd,
        run_dir=run_dir,
        run_id=run_id,
        skill_version=skill_version,
        status=status,
        request=request,
        idea=idea,
        idea_result=idea_result,
        trace=trace,
        report=report,
    )
    validate_public_projection(artifacts, manifest)
    write_portable_json(paths["idea_result_json"], idea_result)
    write_portable_json(paths["research_idea_json"], idea)
    write_portable_json(paths["idea_trace_json"], trace)
    write_portable_json(paths["report_json"], report)
    write_portable_json(paths["manifest_json"], manifest)
    write_checkpoint(
        run_dir,
        "audit",
        {"counts": report.get("counts", {}), "status": status},
        completed_stage="audit",
        request_signature=None,
        runtime_config_signature=None,
    )
    append_pipeline_event(run_dir, "audit", "success" if status == "success" else "incomplete", {"status": status})
    return {"manifest": manifest, "report": report, "status": status}


def public_artifacts(
    *,
    idea: dict[str, Any],
    idea_result: dict[str, Any],
    trace: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        "research_idea.json": idea,
        "idea_result.json": idea_result,
        "idea_trace.json": trace,
        "idea_report.json": report,
    }


def validate_public_artifacts(
    artifacts: dict[str, dict[str, Any]],
    *,
    grouped: bool = False,
) -> list[str] | dict[str, list[str]]:
    grouped_errors: dict[str, list[str]] = {}
    for filename, schema_filename in PUBLIC_ARTIFACT_SCHEMAS.items():
        schema = read_json(SCHEMA_DIR / schema_filename)
        if not isinstance(schema, dict):
            grouped_errors[filename] = [f"{schema_filename} must contain a JSON object"]
            continue
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as error:
            grouped_errors[filename] = [f"{schema_filename} is not a valid Draft 2020-12 schema: {error.message}"]
            continue
        errors = sorted(Draft202012Validator(schema).iter_errors(artifacts[filename]), key=lambda error: list(error.absolute_path))
        if errors:
            grouped_errors[filename] = [schema_error_message(filename, error) for error in errors]
    if grouped:
        return grouped_errors
    return [error for errors in grouped_errors.values() for error in errors]


def schema_error_message(filename: str, error: Any) -> str:
    location = ".".join(str(value) for value in error.absolute_path)
    return f"{filename}{'.' + location if location else ''}: {error.message}"


def controlled_incomplete_artifact(
    filename: str,
    artifact: dict[str, Any],
    *,
    blockers: list[str],
    completed_stages: list[str],
) -> dict[str, Any]:
    reason = blockers[0] if blockers else "Research idea generation is incomplete."
    common = {
        "status": "incomplete",
        "blockers": blockers or [reason],
        "completed_stages": completed_stages,
        "algorithm_provenance": dict(ALGORITHM_PROVENANCE),
    }
    if filename == "idea_report.json":
        report = {
            "schema_version": "xlab.research_idea.report.v2",
            "generated_at": str(artifact.get("generated_at") or utc_now()),
            "passed": False,
            "blocking_errors": common["blockers"],
            "warnings": artifact.get("warnings") if isinstance(artifact.get("warnings"), list) else [],
            "counts": artifact.get("counts") if isinstance(artifact.get("counts"), dict) else {},
            "checks": artifact.get("checks") if isinstance(artifact.get("checks"), dict) else {},
            **common,
        }
        report["checks"]["emitted_public_artifact_schemas"] = True
        return report
    schema_versions = {
        "research_idea.json": "xlab.research_idea.v2",
        "idea_result.json": "xlab.research_idea.result.v2",
        "idea_trace.json": "xlab.research_idea.trace.v2",
    }
    return {
        "schema_version": schema_versions[filename],
        "incomplete_reason": reason,
        **common,
    }


def migrate_public_artifact(
    artifact: dict[str, Any],
    *,
    schema_version: str,
    status: str,
    blockers: list[str],
    completed_stages: list[str],
    provenance: dict[str, Any],
) -> None:
    artifact["schema_version"] = schema_version
    artifact["status"] = status
    artifact["blockers"] = blockers if status == "incomplete" else []
    artifact["completed_stages"] = completed_stages
    artifact["algorithm_provenance"] = provenance
    if status == "incomplete":
        artifact["incomplete_reason"] = str(artifact.get("incomplete_reason") or (blockers[0] if blockers else "Research idea generation is incomplete."))
    else:
        artifact.pop("incomplete_reason", None)


def canonical_stages(value: Any) -> list[str]:
    stages = value if isinstance(value, list) else []
    canonical = [str(stage) for stage in stages if str(stage).strip()]
    return list(dict.fromkeys(canonical))


def load_optional_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = read_json(path)
    return value if isinstance(value, dict) else {}
