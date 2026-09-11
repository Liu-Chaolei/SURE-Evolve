from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import atomic_write_json, project_relative, read_json, utc_now


def parent_ids_from_source(*values: dict[str, Any]) -> list[str]:
    parent_ids: list[str] = []
    for value in values:
        source = value.get("source") if isinstance(value.get("source"), dict) else value
        parent_artifacts = source.get("parent_artifacts") if isinstance(source, dict) else None
        if not isinstance(parent_artifacts, list):
            continue
        for parent in parent_artifacts:
            if not isinstance(parent, dict):
                continue
            artifact_id = parent.get("artifact_id")
            if isinstance(artifact_id, str) and artifact_id.strip():
                parent_ids.append(artifact_id.strip())
    return sorted(set(parent_ids))


def artifact_entry(
    *,
    artifact_type: str,
    schema_version: str,
    path: str,
    parents: list[str],
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "type": artifact_type,
        "schema_version": schema_version,
        "path": path,
    }
    if parents:
        entry["parents"] = parents
    return entry


def build_final_manifest(
    *,
    cwd: Path,
    run_dir: Path,
    run_id: str,
    skill_version: str,
    status: str,
    request: dict[str, Any],
    survey: dict[str, Any],
    citations: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    artifacts_dir = run_dir / "artifacts"
    counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
    parents = parent_ids_from_source(survey, report)
    return {
        "schema_version": "2",
        "run_id": run_id,
        "skill_name": "literature_survey",
        "skill_version": skill_version,
        "status": status,
        "created_at": utc_now(),
        "inputs": request,
        "outputs": {
            "topic": survey.get("topic"),
            "paper_count": counts.get("papers", survey.get("paper_count", 0)),
            "cluster_count": counts.get("clusters", 0),
            "section_count": counts.get("sections", 0),
            "claim_count": counts.get("claims", 0),
            "gap_count": counts.get("gaps", 0),
            "trace_count": counts.get("traces", 0),
        },
        "validation": {
            "passed": report.get("passed") is True,
            "blocking_errors": report.get("blocking_errors", []),
            "warnings": report.get("warnings", []),
        },
        "artifacts": [
            artifact_entry(
                artifact_type="literature_survey_document",
                schema_version="1",
                path=project_relative(cwd, artifacts_dir / "survey.md"),
                parents=parents,
            ),
            artifact_entry(
                artifact_type="literature_survey_json",
                schema_version="1",
                path=project_relative(cwd, artifacts_dir / "survey.json"),
                parents=parents,
            ),
            artifact_entry(
                artifact_type="citation_trace",
                schema_version="1",
                path=project_relative(cwd, artifacts_dir / "citations.json"),
                parents=parents,
            ),
            artifact_entry(
                artifact_type="survey_report",
                schema_version="1",
                path=project_relative(cwd, artifacts_dir / "survey_report.json"),
                parents=parents,
            ),
        ],
    }


def write_final_manifest(path: Path, manifest: dict[str, Any]) -> None:
    previous = read_json(path) if path.exists() else {}
    declaration = manifest.get("resource_manifest") or previous.get("resource_manifest")
    if declaration:
        manifest["resource_manifest"] = declaration
    atomic_write_json(path, manifest)
