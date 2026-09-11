from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .common import file_digest, read_json, stable_id, stable_signature
from .inputs import IdeaRequest
from .resources import ResourceResolution, resolve_survey_resources
from .resources.evidence import PackageNativeEvidenceResources, ResourceEvidenceResult, RetrievalBackends

_COMPONENT_FAISS_FILE = "faiss.index"
_COMPONENT_METADATA_FILE = "meta.json"


@dataclass
class SurveySource:
    survey_path: Path
    survey_json_path: Path | None
    survey_md_path: Path | None
    citations_path: Path | None
    report_path: Path | None
    manifest_path: Path | None
    manifest: dict[str, Any] = field(default_factory=dict)
    survey: dict[str, Any] = field(default_factory=dict)
    citations: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    graph_db_path: Path | None = None
    component_index_dir: Path | None = None
    outcome_model_path: Path | None = None
    component_model_path: Path | None = None
    keynote_cache_path: Path | None = None
    cache_path: Path | None = None
    resources: ResourceResolution = field(default_factory=ResourceResolution)
    warnings: list[str] = field(default_factory=list)

    @property
    def topic(self) -> str:
        return str(self.survey.get("topic") or self.manifest.get("outputs", {}).get("topic") or "Research idea")

    @property
    def survey_run_dir(self) -> Path | None:
        if self.manifest_path is not None:
            return self.manifest_path.parent
        for path in (self.survey_json_path, self.survey_md_path, self.citations_path, self.report_path):
            if path is not None and path.parent.name == "artifacts":
                return path.parent.parent
        return self.survey_path if self.survey_path.is_dir() else None

    @property
    def signatures(self) -> dict[str, str | None]:
        return {
            "survey": file_digest(self.survey_json_path),
            "survey_markdown": file_digest(self.survey_md_path),
            "citations": file_digest(self.citations_path),
            "report": file_digest(self.report_path),
            "manifest": file_digest(self.manifest_path),
            "resource_manifest": file_digest(self.resources.manifest_path),
            "resource_bundle": stable_signature(self.resources.portable_resources()),
            "graph_db": file_digest(self.graph_db_path),
            "component_index_faiss": file_digest(self.component_index_dir / _COMPONENT_FAISS_FILE if self.component_index_dir else None),
            "component_index_metadata": file_digest(self.component_index_dir / _COMPONENT_METADATA_FILE if self.component_index_dir else None),
            "resource_layout": stable_signature(self.resources.portable_resources()),
        }

    @property
    def direct_parent_artifact_ids(self) -> tuple[str, ...]:
        return self.resources.direct_parent_artifact_ids


@dataclass
class SurveyArtifactRepository:
    source: SurveySource
    evidence_items: list[dict[str, Any]]
    references: list[dict[str, Any]]
    retrieval_backends: RetrievalBackends | None = None

    @classmethod
    def from_request(cls, request: IdeaRequest, cwd: Path, *, model_cache_root: Path | None = None) -> "SurveyArtifactRepository":
        source = resolve_survey_source(request.survey_path, cwd, model_cache_root=model_cache_root)
        references = normalize_references(source)
        evidence_items = collect_evidence_items(source, references)
        return cls(source=source, evidence_items=evidence_items, references=references)

    @property
    def topic(self) -> str:
        return self.source.topic

    def validation(self) -> dict[str, Any]:
        survey = self.source.survey
        report = self.source.report
        manifest = self.source.manifest
        runtime = survey.get("runtime") if isinstance(survey.get("runtime"), dict) else {}
        manifest_validation = manifest.get("validation") if isinstance(manifest.get("validation"), dict) else {}
        report_checks = report.get("checks") if isinstance(report.get("checks"), dict) else {}
        checks = {
            "survey_file_loaded": bool(survey),
            "survey_markdown_loaded": self.source.survey_md_path is not None and self.source.survey_md_path.exists(),
            "survey_schema_version": survey.get("schema_version") == "xlab.literature_survey.v1",
            "manifest_present": bool(manifest),
            "manifest_success": manifest.get("status") == "success" and manifest_validation.get("passed") is True,
            "report_present": bool(report),
            "report_passed": report.get("passed") is True,
            "survey_agent_mode": runtime.get("mode") == "survey-agent" and report_checks.get("survey_agent_mode") is not False,
            "citations_present": bool(self.source.citations),
            "citation_traces_present": bool(self.source.citations.get("traces")),
            "references_present": bool(self.references),
            "evidence_present": bool(self.evidence_items),
            "citations_resolve": citation_traces_resolve(self.source.citations, self.references),
        }
        blockers: list[str] = []
        if not checks["survey_file_loaded"]:
            blockers.append("Could not load a literature survey artifact from --survey.")
        if not checks["survey_markdown_loaded"]:
            blockers.append("Research idea generation requires the literature_survey Markdown artifact for original OutcomeRAG retrieval.")
        if not checks["survey_schema_version"]:
            blockers.append("Survey artifact must use schema_version xlab.literature_survey.v1.")
        if not checks["manifest_present"]:
            blockers.append("Research idea generation requires the source literature_survey manifest.json artifact.")
        elif not checks["manifest_success"]:
            blockers.append("The source literature_survey manifest is not successful; rerun or finish it before idea generation.")
        if not checks["report_present"]:
            blockers.append("Research idea generation requires the source survey_report.json artifact.")
        elif not checks["report_passed"]:
            blockers.append("The source survey report did not pass validation; idea generation remains incomplete.")
        if not checks["survey_agent_mode"]:
            blockers.append("Research idea generation requires a completed integrated SurveyAgent survey, not an unavailable or fallback survey.")
        if not checks["citations_present"]:
            blockers.append("Research idea generation requires the source citations.json artifact.")
        elif not checks["citation_traces_present"]:
            blockers.append("Research idea generation requires citation traces from the source literature survey.")
        if not checks["references_present"]:
            blockers.append("Survey artifact does not contain reference papers.")
        if not checks["evidence_present"]:
            blockers.append("Survey artifact does not contain traceable sections, claims, gaps, or citation traces.")
        if not checks["citations_resolve"]:
            blockers.append("Citation traces must resolve to reference papers in citations.json before idea generation can run.")
        return {"passed": not blockers, "checks": checks, "blocking_errors": blockers, "warnings": list(self.source.warnings)}

    def source_context(self, request: IdeaRequest, *, limit: int = 12) -> dict[str, Any]:
        return {
            "schema_version": "xlab.research_idea.source_context.v1",
            "topic": request.topic or self.topic,
            "reference_count": len(self.references),
            "evidence_count": len(self.evidence_items),
            "references": self.references,
            "selected_evidence": [],
            "selected_evidence_status": "pending_outcome_evidence",
            "selection_limit": limit,
            "survey_runtime": self.source.survey.get("runtime") if isinstance(self.source.survey.get("runtime"), dict) else {},
            "resource_manifest": self.source.resources.portable_manifest(),
            "resources": self.source.resources.portable_resources(),
            "direct_parent_artifact_ids": list(self.source.direct_parent_artifact_ids),
            "source_signatures": self.source.signatures,
        }

    def retrieve_resource_evidence(self, query: str, *, limit: int) -> ResourceEvidenceResult:
        """Execute a validated resource bundle; production never falls back to substitutes."""

        resolution = self.source.resources
        if resolution.bundle is None:
            raise ValueError("research_idea retrieval requires a validated executable resource bundle")
        required = {
            "graph_db": self.source.graph_db_path,
            "keynotes": self.source.keynote_cache_path,
            "component_index": self.source.component_index_dir,
            "outcome_model": self.source.outcome_model_path,
            "component_model": self.source.component_model_path,
        }
        missing = [name for name, path in required.items() if path is None]
        if missing:
            raise ValueError("validated resource bundle did not resolve executable resources: " + ", ".join(missing))
        if self.source.survey_md_path is None:
            raise ValueError("OutcomeRAG requires the declared Survey Markdown artifact")
        try:
            survey_markdown = self.source.survey_md_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ValueError("OutcomeRAG cannot read the Survey Markdown artifact") from error
        adapter = PackageNativeEvidenceResources(
            bundle=resolution.bundle,
            graph_db_path=self.source.graph_db_path,  # type: ignore[arg-type]
            keynote_cache_path=self.source.keynote_cache_path,  # type: ignore[arg-type]
            component_index_dir=self.source.component_index_dir,  # type: ignore[arg-type]
            outcome_model_path=self.source.outcome_model_path,  # type: ignore[arg-type]
            component_model_path=self.source.component_model_path,  # type: ignore[arg-type]
            outcome_backend=self.retrieval_backends.outcome if self.retrieval_backends else None,
            component_backend=self.retrieval_backends.component if self.retrieval_backends else None,
            faiss_backend=self.retrieval_backends.faiss if self.retrieval_backends else None,
        )
        return adapter.retrieve(
            query,
            survey_markdown,
            self.source.citations,
            self.references,
            limit=limit,
        )



def resolve_survey_source(path: Path, cwd: Path, *, model_cache_root: Path | None = None) -> SurveySource:
    warnings: list[str] = []
    resolved = path.resolve()
    manifest_path: Path | None = None
    survey_json_path: Path | None = None
    survey_md_path: Path | None = None
    citations_path: Path | None = None
    report_path: Path | None = None
    manifest: dict[str, Any] = {}

    if resolved.is_dir():
        manifest_candidate = resolved / "manifest.json"
        artifacts = resolved / "artifacts"
        manifest_path = manifest_candidate if manifest_candidate.exists() else None
        survey_json_path = artifacts / "survey.json"
        survey_md_path = artifacts / "survey.md"
        citations_path = artifacts / "citations.json"
        report_path = artifacts / "survey_report.json"
    elif resolved.name == "manifest.json":
        manifest_path = resolved
    else:
        if resolved.suffix.lower() in {".md", ".markdown"}:
            survey_md_path = resolved
        else:
            survey_json_path = resolved
        if resolved.name in {"survey.json", "survey.md"}:
            sibling_survey_json = resolved.parent / "survey.json"
            sibling_survey_md = resolved.parent / "survey.md"
            sibling_citations = resolved.parent / "citations.json"
            sibling_report = resolved.parent / "survey_report.json"
            sibling_manifest = resolved.parent.parent / "manifest.json"
            survey_json_path = sibling_survey_json if sibling_survey_json.exists() else survey_json_path
            survey_md_path = sibling_survey_md if sibling_survey_md.exists() else survey_md_path
            citations_path = sibling_citations if sibling_citations.exists() else None
            report_path = sibling_report if sibling_report.exists() else None
            manifest_path = sibling_manifest if sibling_manifest.exists() else None

    if manifest_path and manifest_path.exists():
        value = read_json(manifest_path)
        manifest = value if isinstance(value, dict) else {}
        for artifact in manifest.get("artifacts", []) if isinstance(manifest.get("artifacts"), list) else []:
            if not isinstance(artifact, dict) or not artifact.get("path"):
                continue
            artifact_path = resolve_artifact_path(str(artifact["path"]), cwd, manifest_path.parent)
            artifact_type = artifact.get("type")
            if artifact_type in {"literature_survey_json", "survey_json"}:
                survey_json_path = artifact_path
            elif artifact_type in {"literature_survey_document", "survey_markdown", "survey_md"}:
                survey_md_path = artifact_path
            elif artifact_type in {"citation_trace", "citations"}:
                citations_path = artifact_path
            elif artifact_type in {"survey_report", "literature_survey_report"}:
                report_path = artifact_path
        if survey_json_path is None:
            sibling = manifest_path.parent / "artifacts" / "survey.json"
            if sibling.exists():
                survey_json_path = sibling
        if survey_md_path is None:
            sibling = manifest_path.parent / "artifacts" / "survey.md"
            if sibling.exists():
                survey_md_path = sibling
        if citations_path is None:
            sibling = manifest_path.parent / "artifacts" / "citations.json"
            if sibling.exists():
                citations_path = sibling
        if report_path is None:
            sibling = manifest_path.parent / "artifacts" / "survey_report.json"
            if sibling.exists():
                report_path = sibling

    survey = read_optional_json_object(survey_json_path, warnings, "survey")
    citations = read_optional_json_object(citations_path, warnings, "citations")
    report = read_optional_json_object(report_path, warnings, "survey report")
    resources = resolve_survey_resources(
        survey_manifest_path=manifest_path,
        survey_manifest=manifest,
        survey=survey,
        survey_json_path=survey_json_path,
        cwd=cwd,
        model_cache_root=model_cache_root,
    )
    return SurveySource(
        survey_path=resolved,
        survey_json_path=survey_json_path,
        survey_md_path=survey_md_path,
        citations_path=citations_path,
        report_path=report_path,
        manifest_path=manifest_path,
        manifest=manifest,
        survey=survey,
        citations=citations,
        report=report,
        graph_db_path=resources.graph_db_path,
        component_index_dir=resources.component_index_dir,
        outcome_model_path=resources.outcome_model_path,
        component_model_path=resources.component_model_path,
        keynote_cache_path=resources.keynote_cache_path,
        cache_path=model_cache_root,
        resources=resources,
        warnings=warnings,
    )



def resolve_artifact_path(value: str, cwd: Path, manifest_root: Path) -> Path:
    """Resolve a manifest artifact without allowing it to escape authorized roots."""

    path = Path(value).expanduser()
    cwd = cwd.resolve()
    manifest_root = manifest_root.resolve()
    candidates = (
        (path.resolve(),)
        if path.is_absolute()
        else ((manifest_root / path).resolve(), (cwd / path).resolve())
    )
    allowed = [candidate for candidate in candidates
               if _path_is_within(candidate, manifest_root) or _path_is_within(candidate, cwd)]
    for candidate in allowed:
        if candidate.is_file():
            return candidate
    if allowed:
        # Preserve the unresolved path for the caller's missing-artifact diagnostic.
        return allowed[0]
    raise ValueError(f"Survey artifact path escapes its manifest run and current workspace: {value!r}")


def _path_is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def read_optional_json_object(path: Path | None, warnings: list[str], label: str) -> dict[str, Any]:
    if path is None or not path.exists() or not path.is_file():
        warnings.append(f"Missing {label} artifact: {path}")
        return {}
    value = read_json(path)
    if isinstance(value, dict):
        return value
    warnings.append(f"Expected {label} artifact to contain a JSON object: {path}")
    return {}


def citation_traces_resolve(citations: dict[str, Any], references: list[dict[str, Any]]) -> bool:
    reference_ids = {str(reference.get("paper_id")) for reference in references if isinstance(reference, dict) and reference.get("paper_id") is not None}
    traces = citations.get("traces") if isinstance(citations.get("traces"), list) else []
    if not traces or not reference_ids:
        return False
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        paper_ids = normalize_paper_ids(trace.get("paper_ids"))
        if paper_ids and not all(paper_id in reference_ids for paper_id in paper_ids):
            return False
    return True



def normalize_references(source: SurveySource) -> list[dict[str, Any]]:
    raw = source.citations.get("references") if isinstance(source.citations.get("references"), list) else source.survey.get("references", [])
    references: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, reference in enumerate(raw if isinstance(raw, list) else []):
        if isinstance(reference, dict):
            paper_id = str(reference.get("paper_id") or reference.get("id") or reference.get("title") or f"paper-{index + 1}")
            title = str(reference.get("title") or paper_id)
            year = reference.get("year")
            venue = reference.get("venue")
            authors = reference.get("authors", []) if isinstance(reference.get("authors"), list) else []
            url = reference.get("url")
            source = reference.get("source", "survey")
        else:
            paper_id = str(reference).strip()
            title = paper_id
            year = None
            venue = None
            authors = []
            url = None
            source = "survey"
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        references.append(
            {
                "paper_id": paper_id,
                "title": title,
                "year": year,
                "venue": venue,
                "authors": authors,
                "url": url,
                "source": source,
            }
        )
    return references


def collect_evidence_items(source: SurveySource, references: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reference_titles = {str(reference.get("paper_id")): str(reference.get("title") or reference.get("paper_id")) for reference in references}
    items: list[dict[str, Any]] = []
    for kind, key, text_field in (
        ("section", "sections", "summary"),
        ("key_claim", "key_claims", "claim"),
        ("research_gap", "research_gaps", "gap"),
    ):
        for index, value in enumerate(source.survey.get(key, []) if isinstance(source.survey.get(key), list) else []):
            if not isinstance(value, dict):
                continue
            paper_ids = normalize_paper_ids(value.get("paper_ids"))
            text = str(value.get(text_field) or value.get("text") or value.get("title") or "")
            title = str(value.get("title") or value.get("cluster") or value.get("theme") or kind.replace("_", " ").title())
            if not text and not title:
                continue
            items.append(
                {
                    "id": str(value.get("id") or stable_id(kind, title, text, index)),
                    "kind": kind,
                    "title": title,
                    "summary": text,
                    "text": text,
                    "paper_ids": paper_ids,
                    "paper_titles": [reference_titles.get(paper_id, paper_id) for paper_id in paper_ids],
                    "source": key,
                }
            )
    for index, trace in enumerate(source.citations.get("traces", []) if isinstance(source.citations.get("traces"), list) else []):
        if not isinstance(trace, dict):
            continue
        paper_ids = normalize_paper_ids(trace.get("paper_ids"))
        claim = str(trace.get("claim") or trace.get("claim_id") or "")
        evidence = trace.get("evidence") if isinstance(trace.get("evidence"), list) else []
        evidence_text = " ".join(str(item) for item in evidence if item)
        if not claim and not evidence_text:
            continue
        items.append(
            {
                "id": str(trace.get("claim_id") or stable_id("trace", claim, index)),
                "kind": "citation_trace",
                "title": str(trace.get("claim_id") or "Citation trace"),
                "summary": claim,
                "text": " ".join(part for part in [claim, evidence_text] if part),
                "paper_ids": paper_ids,
                "paper_titles": [reference_titles.get(paper_id, paper_id) for paper_id in paper_ids],
                "source": "citations.traces",
            }
        )
    return items


def normalize_paper_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if value is None:
        return []
    return [str(value)]


