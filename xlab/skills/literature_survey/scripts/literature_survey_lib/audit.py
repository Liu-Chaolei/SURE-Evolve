from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .common import project_root_from_run_dir, read_json, read_jsonl, run_paths, utc_now
from .checkpoints import read_checkpoint


REQUIRED_SURVEY_FIELDS = (
    "schema_version",
    "run_id",
    "topic",
    "paper_count",
    "sections",
    "clusters",
    "key_claims",
    "research_gaps",
    "references",
)
REQUIRED_XCIENTIST_STATE_FILES = (
    "seed_papers.json",
    "expanded_papers.json",
    "collected_papers.json",
    "keynotes.json",
    "clustering_result.json",
    "analysis_context.json",
    "outline.raw.json",
    "draft.raw.json",
    "draft.raw.md",
    "refined.raw.md",
    "references.raw.json",
    "evaluation.json",
    "engine_result.json",
)
HOST_PATH_RE = "/" + "hpc_stor"
AISTOR_HOST_PATH_RE = "/aistor" + HOST_PATH_RE
EXTERNAL_CHECKOUT_NAME_RE = "agent_workspace/" + "Xcientist-2"
EXTERNAL_CHECKOUT_RE = re.compile(
    re.escape(EXTERNAL_CHECKOUT_NAME_RE)
    + r"|"
    + r"XAgora|XForge|src/agents/survey_agent|src\.agents\.survey_agent"
)
HOST_EXTERNAL_RE = re.compile(re.escape(HOST_PATH_RE) + r"|" + re.escape(AISTOR_HOST_PATH_RE))
SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"


def schema_path(name: str) -> Path:
    return SCHEMA_DIR / name


def type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def validate_json_schema(value: Any, schema: dict[str, Any], label: str) -> list[str]:
    errors: list[str] = []
    if "const" in schema and value != schema["const"]:
        errors.append(f"{label} must equal {schema['const']!r}.")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{label} must be one of {schema['enum']!r}.")

    expected_type = schema.get("type")
    if expected_type is not None:
        expected_types = expected_type if isinstance(expected_type, list) else [expected_type]
        if not any(isinstance(item, str) and type_matches(value, item) for item in expected_types):
            errors.append(f"{label} must be {', '.join(str(item) for item in expected_types)}.")
            return errors

    if isinstance(value, str) and isinstance(schema.get("minLength"), int) and len(value) < schema["minLength"]:
        errors.append(f"{label} must contain at least {schema['minLength']} character(s).")
    if isinstance(value, list):
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
            errors.append(f"{label} must contain at least {schema['minItems']} item(s).")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                errors.extend(validate_json_schema(item, item_schema, f"{label}[{index}]"))
    if isinstance(value, dict):
        required = schema.get("required")
        if isinstance(required, list):
            for field in required:
                if isinstance(field, str) and field not in value:
                    errors.append(f"{label}.{field} is required.")
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for field, field_schema in properties.items():
                if field in value and isinstance(field_schema, dict):
                    errors.extend(validate_json_schema(value[field], field_schema, f"{label}.{field}"))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append(f"{label} must be >= {minimum}.")
        if isinstance(maximum, (int, float)) and value > maximum:
            errors.append(f"{label} must be <= {maximum}.")
    return errors


def validate_schema_file(value: Any, schema_file: str, label: str) -> list[str]:
    schema = read_json(schema_path(schema_file))
    if not isinstance(schema, dict):
        return [f"Schema {schema_file} must be a JSON object."]
    return validate_json_schema(value, schema, label)


def record_schema_check(
    *,
    checks: dict[str, bool],
    blocking_errors: list[str],
    check_name: str,
    schema_file: str,
    value: Any,
    label: str,
) -> None:
    errors = validate_schema_file(value, schema_file, label)
    checks[check_name] = not errors
    blocking_errors.extend(errors)


def collect_referenced_ids(survey: dict[str, Any], citations: dict[str, Any]) -> set[str]:
    used: set[str] = set()
    for section in survey.get("sections", []):
        if isinstance(section, dict):
            used.update(str(paper_id) for paper_id in section.get("paper_ids", []) if paper_id is not None)
    for claim in survey.get("key_claims", []):
        if isinstance(claim, dict):
            used.update(str(paper_id) for paper_id in claim.get("paper_ids", []) if paper_id is not None)
    for gap in survey.get("research_gaps", []):
        if isinstance(gap, dict):
            used.update(str(paper_id) for paper_id in gap.get("paper_ids", []) if paper_id is not None)
    for trace in citations.get("traces", []):
        if isinstance(trace, dict):
            used.update(str(paper_id) for paper_id in trace.get("paper_ids", []) if paper_id is not None)
    return used


def audit_artifacts(run_dir: Path, min_papers: int = 3) -> dict[str, Any]:
    paths = run_paths(run_dir)
    survey_md_path = paths["survey_md"]
    survey_json_path = paths["survey_json"]
    citations_path = paths["citations_json"]
    report_path = paths["report_json"]
    blocking_errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}

    for name, path in (("survey.md", survey_md_path), ("survey.json", survey_json_path), ("citations.json", citations_path)):
        if not path.exists():
            blocking_errors.append(f"Missing artifacts/{name}.")
            checks[f"has_{name.replace('.', '_')}"] = False
        else:
            checks[f"has_{name.replace('.', '_')}"] = True
    if blocking_errors:
        return build_report(False, blocking_errors, warnings, {}, checks, run_dir, report_path)

    survey = read_json(survey_json_path)
    citations = read_json(citations_path)
    request = read_json(paths["request"]) if paths["request"].exists() else {}
    paper_index = read_json(paths["paper_index"]) if paths["paper_index"].exists() else {}
    checkpoint = read_checkpoint(run_dir) or {}
    if not isinstance(request, dict):
        blocking_errors.append("state/request.json must be a JSON object.")
        request = {}
    record_schema_check(
        checks=checks,
        blocking_errors=blocking_errors,
        check_name="request_schema",
        schema_file="request.schema.json",
        value=request,
        label="state/request.json",
    )
    if not isinstance(survey, dict):
        blocking_errors.append("survey.json must be a JSON object.")
        survey = {}
    if not isinstance(citations, dict):
        blocking_errors.append("citations.json must be a JSON object.")
        citations = {}
    record_schema_check(
        checks=checks,
        blocking_errors=blocking_errors,
        check_name="survey_schema",
        schema_file="literature_survey.schema.json",
        value=survey,
        label="artifacts/survey.json",
    )
    record_schema_check(
        checks=checks,
        blocking_errors=blocking_errors,
        check_name="citation_trace_schema",
        schema_file="citation_trace.schema.json",
        value=citations,
        label="artifacts/citations.json",
    )
    checks["survey_schema_version"] = survey.get("schema_version") == "xlab.literature_survey.v1"
    if not checks["survey_schema_version"]:
        blocking_errors.append("survey.json schema_version must be xlab.literature_survey.v1.")
    checks["citation_schema_version"] = citations.get("schema_version") == "xlab.citation_trace.v1"
    if not checks["citation_schema_version"]:
        blocking_errors.append("citations.json schema_version must be xlab.citation_trace.v1.")
    missing_fields = [field for field in REQUIRED_SURVEY_FIELDS if not survey.get(field)]
    checks["required_survey_fields"] = not missing_fields
    if missing_fields:
        blocking_errors.append(f"survey.json is missing required content: {', '.join(missing_fields)}.")

    references = citations.get("references") if isinstance(citations.get("references"), list) else survey.get("references", [])
    reference_ids = {
        str(reference.get("paper_id"))
        for reference in references
        if isinstance(reference, dict) and reference.get("paper_id") is not None
    }
    used_ids = collect_referenced_ids(survey, citations)
    unresolved = sorted(used_ids - reference_ids)
    checks["citation_ids_resolve"] = not unresolved
    if unresolved:
        blocking_errors.append(f"Unresolved citation ids: {', '.join(unresolved)}.")
    paper_count = survey.get("paper_count") if isinstance(survey.get("paper_count"), int) else len(reference_ids)
    checks["minimum_paper_count"] = paper_count >= min_papers
    if not checks["minimum_paper_count"]:
        blocking_errors.append(f"Expected at least {min_papers} papers, found {paper_count}.")

    source_type = paper_index.get("source_type") if isinstance(paper_index, dict) else None
    survey_source = survey.get("source") if isinstance(survey.get("source"), dict) else {}
    survey_source_type = survey_source.get("source_type") if isinstance(survey_source, dict) else None
    real_source_types = {source_type, survey_source_type} - {None}
    checks["real_paper_source"] = bool(real_source_types) and not (real_source_types & {"missing", "synthetic"})
    if not checks["real_paper_source"]:
        blocking_errors.append("A successful literature survey requires real papers from --graph or --input; synthetic/topic-only evidence is incomplete.")
    reference_source_types = {
        str(reference.get("source"))
        for reference in references
        if isinstance(reference, dict) and reference.get("source") is not None
    }
    checks["no_synthetic_references"] = "synthetic" not in reference_source_types
    if not checks["no_synthetic_references"]:
        blocking_errors.append("A successful literature survey cannot cite synthetic placeholder references.")
    source_signatures = checkpoint.get("source_signatures") if isinstance(checkpoint.get("source_signatures"), dict) else {}
    checks["source_signature_present"] = any(value for value in source_signatures.values())
    if not checks["source_signature_present"]:
        blocking_errors.append("A successful literature survey requires source signatures for the input paper set or graph artifacts.")

    runtime = survey.get("runtime") if isinstance(survey.get("runtime"), dict) else {}
    runtime_mode = runtime.get("mode") if isinstance(runtime, dict) else None
    checks["survey_agent_mode"] = runtime_mode == "survey-agent"
    if not checks["survey_agent_mode"]:
        blocking_errors.append("A successful literature survey requires the integrated SurveyAgent outline/draft/refine path.")
    audit_survey_agent_provenance(run_dir, survey, references, checks, blocking_errors)

    markdown = survey_md_path.read_text(encoding="utf-8")
    checks["markdown_length"] = len(markdown.strip()) >= 400
    if not checks["markdown_length"]:
        blocking_errors.append("survey.md is too short for a successful literature survey.")
    checks["markdown_citations"] = "[paper:" in markdown
    if not checks["markdown_citations"]:
        blocking_errors.append("survey.md must contain [paper:<id>] citations.")
    traces = citations.get("traces") if isinstance(citations.get("traces"), list) else []
    checks["traces_present"] = bool(traces)
    if not checks["traces_present"]:
        blocking_errors.append("citations.json must include trace entries.")

    if isinstance(paper_index, dict):
        warnings.extend(str(warning) for warning in paper_index.get("warnings", []) if warning)
        if paper_index.get("truncated_count", 0):
            warnings.append(f"Input source was truncated by max_papers; {paper_index.get('truncated_count')} papers were not materialized.")
        if paper_index.get("duplicate_count", 0):
            warnings.append(f"Duplicate papers removed: {paper_index.get('duplicate_count')}.")
    if isinstance(checkpoint.get("warnings"), list):
        warnings.extend(str(warning) for warning in checkpoint["warnings"] if warning)
    last_error = checkpoint.get("last_error") if isinstance(checkpoint, dict) else None
    if isinstance(last_error, str) and last_error.strip():
        warnings.append(last_error.strip())
        if last_error.startswith("Survey Agent generation failed:") or "Survey Agent generation skipped" in last_error:
            checks["survey_agent_completed"] = False
            blocking_errors.append("Survey Agent generation failed; inspect diagnostics and rerun resume after repairing provider configuration.")

    leak_findings = scan_for_forbidden_text(run_dir)
    checks["no_external_checkout_reference"] = not leak_findings["external"]
    checks["no_api_secret_leak"] = not leak_findings["secrets"]
    if leak_findings["external"]:
        blocking_errors.append("Run artifacts reference external research-agent checkouts; keep literature_survey self-contained.")
    if leak_findings["secrets"]:
        blocking_errors.append("Run artifacts appear to contain an API secret value or prefix.")

    warnings = sorted(set(warnings))
    counts = {
        "papers": paper_count,
        "clusters": len(survey.get("clusters", [])) if isinstance(survey.get("clusters"), list) else 0,
        "sections": len(survey.get("sections", [])) if isinstance(survey.get("sections"), list) else 0,
        "claims": len(survey.get("key_claims", [])) if isinstance(survey.get("key_claims"), list) else 0,
        "gaps": len(survey.get("research_gaps", [])) if isinstance(survey.get("research_gaps"), list) else 0,
        "references": len(reference_ids),
        "traces": len(traces),
    }
    return build_report(not blocking_errors, blocking_errors, warnings, counts, checks, run_dir, report_path, source=paper_index, checkpoint=checkpoint)


def audit_survey_agent_provenance(
    run_dir: Path,
    survey: dict[str, Any],
    references: list[Any],
    checks: dict[str, bool],
    blocking_errors: list[str],
) -> None:
    paths = run_paths(run_dir)
    agent_path = paths["survey_agent_result"]
    checks["has_state_survey_agent_result_json"] = agent_path.exists()
    if not checks["has_state_survey_agent_result_json"]:
        blocking_errors.append("state/survey_agent_result.json is required for integrated SurveyAgent provenance.")
        return
    agent_result = read_json(agent_path)
    if not isinstance(agent_result, dict):
        checks["survey_agent_schema_version"] = False
        blocking_errors.append("state/survey_agent_result.json must be a JSON object.")
        return
    engine = agent_result.get("engine") if isinstance(agent_result.get("engine"), dict) else {}
    xcientist = engine.get("xcientist") if isinstance(engine.get("xcientist"), dict) else {}
    checks["survey_agent_schema_version"] = agent_result.get("schema_version") == "xlab.literature_survey.agent.v1"
    checks["survey_agent_integrated_mode"] = agent_result.get("mode") == "xcientist_survey_agent"
    checks["survey_agent_engine_integrated"] = engine.get("mode") == "integrated_xcientist_survey_agent"
    runtime = survey.get("runtime") if isinstance(survey.get("runtime"), dict) else {}
    checks["survey_runtime_engine_integrated"] = runtime.get("survey_agent_engine") == "integrated_xcientist_survey_agent"
    checks["survey_agent_sections_claims_gaps_present"] = all(
        isinstance(agent_result.get(key), list) and bool(agent_result.get(key))
        for key in ("sections", "key_claims", "research_gaps")
    )
    reference_ids = {
        str(reference.get("paper_id"))
        for reference in references
        if isinstance(reference, dict) and reference.get("paper_id") is not None
    }
    cited = {str(paper_id) for paper_id in agent_result.get("cited_paper_ids", []) if paper_id is not None}
    checks["survey_agent_cited_papers_resolve"] = bool(cited) and cited.issubset(reference_ids)
    internal_state_dir = str(xcientist.get("internal_state_dir") or "")
    checks["survey_agent_xcientist_provenance"] = (
        bool(xcientist.get("collected_paper_ids"))
        and bool(xcientist.get("raw_references"))
        and bool(internal_state_dir)
        and is_run_local(Path(internal_state_dir), run_dir)
    )
    state_dir = Path(internal_state_dir) if internal_state_dir else paths["state"] / "xcientist"
    checks["survey_agent_xcientist_state_files"] = all((state_dir / name).exists() for name in REQUIRED_XCIENTIST_STATE_FILES)

    if not checks["survey_agent_schema_version"]:
        blocking_errors.append("state/survey_agent_result.json schema_version must be xlab.literature_survey.agent.v1.")
    if not checks["survey_agent_integrated_mode"]:
        blocking_errors.append("SurveyAgent result mode must be xcientist_survey_agent.")
    if not checks["survey_agent_engine_integrated"] or not checks["survey_runtime_engine_integrated"]:
        blocking_errors.append("A successful survey requires integrated_xcientist_survey_agent provenance in both state and survey runtime metadata.")
    if not checks["survey_agent_sections_claims_gaps_present"]:
        blocking_errors.append("SurveyAgent result must contain non-empty sections, key claims, and research gaps.")
    if not checks["survey_agent_cited_papers_resolve"]:
        blocking_errors.append("SurveyAgent cited_paper_ids must be non-empty and resolve to survey references.")
    if not checks["survey_agent_xcientist_provenance"]:
        blocking_errors.append("SurveyAgent engine result must include run-local Xcientist provenance with collected papers and raw references.")
    if not checks["survey_agent_xcientist_state_files"]:
        blocking_errors.append("Integrated SurveyAgent state is incomplete; expected Xcientist outline/draft/refine/evaluation state files.")


def is_run_local(path: Path, run_dir: Path) -> bool:
    try:
        path.resolve().relative_to(run_dir.resolve())
        return True
    except ValueError:
        return False


def scan_for_forbidden_text(run_dir: Path) -> dict[str, list[str]]:
    external: list[str] = []
    secrets: list[str] = []
    secret_values = [
        os.environ.get("OPENAI_API_KEY", ""),
        os.environ.get("LLM_API_KEY", ""),
        os.environ.get("SEMANTIC_SCHOLAR_API_KEY", ""),
        os.environ.get("S2_API_KEY", ""),
    ]
    secret_needles = []
    for value in secret_values:
        if not value:
            continue
        secret_needles.append(value)
        if len(value) >= 8:
            secret_needles.append(value[:8])
    allowed_paths = [str(run_dir.resolve()), str(project_root_from_run_dir(run_dir))]
    for path in candidate_text_files(run_dir):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        redacted = text
        for allowed_path in allowed_paths:
            redacted = redacted.replace(allowed_path, "<run-local>")
        if EXTERNAL_CHECKOUT_RE.search(redacted) or (HOST_EXTERNAL_RE.search(redacted) and "Xcientist-2" in redacted):
            external.append(str(path))
        if any(needle and needle in text for needle in secret_needles):
            secrets.append(str(path))
    return {"external": external, "secrets": secrets}


def candidate_text_files(run_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".json", ".jsonl", ".log", ".txt", ".md", ".yaml", ".yml"}:
            files.append(path)
    return files


def build_report(
    passed: bool,
    blocking_errors: list[str],
    warnings: list[str],
    counts: dict[str, int],
    checks: dict[str, bool],
    run_dir: Path,
    report_path: Path,
    *,
    source: dict[str, Any] | None = None,
    checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths = run_paths(run_dir)
    checkpoint = checkpoint or {}
    report = {
        "schema_version": "xlab.literature_survey.report.v1",
        "generated_at": utc_now(),
        "passed": passed,
        "blocking_errors": blocking_errors,
        "warnings": warnings,
        "counts": counts,
        "checks": checks,
        "source": source or {},
        "stages": checkpoint.get("completed_stages", []),
        "attempted_stages": checkpoint.get("attempted_stages", []),
        "blocked_stages": checkpoint.get("blocked_stages", []),
        "checkpoint": checkpoint,
        "metrics": {
            "state_papers": sum(1 for _ in read_jsonl(paths["papers_jsonl"])) if paths["papers_jsonl"].exists() else 0,
            "state_traces": sum(1 for _ in read_jsonl(paths["traces_jsonl"])) if paths["traces_jsonl"].exists() else 0,
        },
        "files": {
            "survey_markdown": str(paths["survey_md"]),
            "survey_json": str(paths["survey_json"]),
            "citation_trace": str(paths["citations_json"]),
            "report": str(report_path),
        },
    }
    report_schema_errors = validate_schema_file(report, "survey_report.schema.json", "artifacts/survey_report.json")
    if report_schema_errors:
        report["checks"]["survey_report_schema"] = False
        report["blocking_errors"].extend(report_schema_errors)
        report["passed"] = False
    else:
        report["checks"]["survey_report_schema"] = True
    return report
