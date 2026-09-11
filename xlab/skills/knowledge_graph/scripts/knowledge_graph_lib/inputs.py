from __future__ import annotations

import json
from pathlib import Path

from .common import (
    JsonObject,
    artifact_paths,
    as_list,
    as_mapping,
    atomic_write_json,
    normalize_space,
    sha256_file,
    text,
    utc_now,
)


DOWNLOADED_STATUSES = {"downloaded", "existing"}


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _safe_relative_path(value: object) -> Path | None:
    raw = text(value)
    if not raw or raw.startswith("~"):
        return None
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    return candidate


def _declared_path(value: object, *, cwd: Path, collection_root: Path) -> Path | None:
    raw = text(value)
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    anchors = [cwd, collection_root]
    for anchor in anchors:
        resolved = (anchor / candidate).resolve()
        if resolved.exists():
            return resolved
    if raw.startswith("artifacts/"):
        return (collection_root / candidate).resolve()
    return (cwd / candidate).resolve()


def _artifact_path(manifest: JsonObject, artifact_type: str) -> object:
    for value in as_list(manifest.get("artifacts")):
        artifact = as_mapping(value)
        if artifact.get("type") == artifact_type:
            return artifact.get("path")
    return None


def locate_collection(
    input_directory: Path, cwd: Path
) -> tuple[Path, Path, Path, JsonObject]:
    directory = input_directory.expanduser().resolve()
    cwd = cwd.expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Input must be a completed paper_collect directory: {directory}")

    run_manifest = directory / "manifest.json"
    if not run_manifest.is_file():
        raise ValueError("Input directory must be a successful paper_collect run with manifest.json.")
    try:
        manifest = as_mapping(json.loads(run_manifest.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read prior run manifest: {error}") from error

    if manifest.get("schema_version") != "2":
        raise ValueError("Prior paper_collect manifest must use schema_version 2.")
    if text(manifest.get("skill_name")) != "paper_collect":
        raise ValueError(
            f"Input run was produced by {text(manifest.get('skill_name')) or '<missing>'}, not paper_collect."
        )
    if text(manifest.get("status")) != "success":
        raise ValueError(
            "The selected paper_collect run is not complete: "
            f"manifest status is {text(manifest.get('status')) or '<missing>'}."
        )
    if as_mapping(manifest.get("validation")).get("passed") is not True:
        raise ValueError("The selected paper_collect run did not pass validation.")

    collection_root = directory
    paper_set_path = collection_root / "artifacts" / "papers.manifest.json"
    if not paper_set_path.is_file():
        raise ValueError("Input directory must contain canonical artifacts/papers.manifest.json.")

    outputs = as_mapping(manifest.get("outputs"))
    for label, raw_path in (
        ("paper_set artifact", _artifact_path(manifest, "paper_set")),
        ("outputs.paper_collection", outputs.get("paper_collection")),
    ):
        resolved = _declared_path(raw_path, cwd=cwd, collection_root=collection_root)
        if resolved is None or resolved != paper_set_path.resolve():
            raise ValueError(
                f"Prior paper_collect manifest {label} must point to {paper_set_path}."
            )

    pdf_root = collection_root / "artifacts" / "pdfs"
    declared_pdf_root = _declared_path(
        outputs.get("pdf_dir"), cwd=cwd, collection_root=collection_root
    )
    if declared_pdf_root is None or declared_pdf_root != pdf_root.resolve():
        raise ValueError(
            f"Prior paper_collect manifest outputs.pdf_dir must point to {pdf_root}."
        )
    if not pdf_root.is_dir():
        raise ValueError(f"Prior paper_collect PDF root is missing: {pdf_root}")

    return collection_root, paper_set_path, run_manifest, manifest


def _edge_path(collection_root: Path, paper_set: JsonObject) -> Path:
    files = as_mapping(paper_set.get("files"))
    relative = _safe_relative_path(files.get("edges"))
    canonical = (collection_root / "artifacts" / "metadata" / "edges.jsonl").resolve()
    if relative is None:
        raise ValueError("paper_collect files.edges must be a safe relative path.")
    if relative.parts[:1] == ("artifacts",):
        candidate = (collection_root / relative).resolve()
    else:
        candidate = (collection_root / "artifacts" / relative).resolve()
    if candidate != canonical:
        raise ValueError(f"paper_collect files.edges must point to {canonical}.")
    if not candidate.is_file():
        raise ValueError(f"paper_collect edge artifact is missing: {candidate}")
    return candidate


def _pdf_path(
    collection_root: Path,
    paper: JsonObject,
    pdf_root: Path,
) -> tuple[Path | None, str | None]:
    download = as_mapping(paper.get("download"))
    if text(download.get("status")) not in DOWNLOADED_STATUSES:
        return None, None
    relative = _safe_relative_path(download.get("path"))
    if relative is None:
        return None, "downloaded PDF path is not a safe relative path"
    candidate = (pdf_root / relative).resolve()
    if not _is_within(candidate, pdf_root):
        return None, "downloaded PDF path escapes the canonical paper_collect PDF root"
    if not candidate.is_file():
        return None, f"downloaded PDF is missing: {candidate}"
    if candidate.suffix.lower() != ".pdf":
        return None, f"downloaded artifact is not a PDF: {candidate.name}"
    expected_sha256 = text(download.get("sha256"))
    if not expected_sha256:
        return None, "downloaded PDF is missing its paper_collect sha256"
    actual_sha256 = sha256_file(candidate)
    if actual_sha256 != expected_sha256:
        return None, "downloaded PDF sha256 does not match paper_collect metadata"
    if not _is_within(candidate, collection_root / "artifacts" / "pdfs"):
        return None, "downloaded PDF is outside canonical artifacts/pdfs"
    return candidate, None


def prepare_input_documents(
    run_dir: Path,
    collection_root: Path,
    paper_set_path: Path,
    prior_manifest_path: Path,
    prior_manifest: JsonObject,
) -> JsonObject:
    paths = artifact_paths(run_dir)
    try:
        paper_set = as_mapping(json.loads(paper_set_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read paper collection: {error}") from error
    if paper_set.get("schema_version") != "xlab.paper_set.v3":
        raise ValueError("Selected collection must use schema_version xlab.paper_set.v3.")

    pdf_root = (collection_root / "artifacts" / "pdfs").resolve()
    edge_path = _edge_path(collection_root, paper_set)
    prior_manifest_sha256 = sha256_file(prior_manifest_path)
    paper_set_sha256 = sha256_file(paper_set_path)
    paper_edges_sha256 = sha256_file(edge_path)
    paper_collect_run_id = text(prior_manifest.get("run_id")) or collection_root.name

    documents: list[JsonObject] = []
    seen_ids: set[str] = set()
    for paper_value in as_list(paper_set.get("papers")):
        paper = as_mapping(paper_value)
        paper_id = text(paper.get("id") or paper.get("paper_id"))
        source_pdf, pdf_blocker = _pdf_path(collection_root, paper, pdf_root)
        blocker: str | None = None
        status = "pending"
        if not paper_id:
            status = "failed"
            blocker = "missing stable paper ID"
        elif paper_id in seen_ids:
            status = "failed"
            blocker = f"duplicate stable paper ID: {paper_id}"
        elif pdf_blocker is not None:
            raise ValueError(f"Paper {paper_id}: {pdf_blocker}.")
        elif source_pdf is None:
            status = "metadata_only"
            blocker = "no downloaded PDF"
        if paper_id:
            seen_ids.add(paper_id)
        source_sha256 = sha256_file(source_pdf) if source_pdf else None
        download = as_mapping(paper.get("download"))
        source_download_sha256 = text(download.get("sha256")) or None
        documents.append(
            {
                "paper_id": paper_id,
                "title": normalize_space(paper.get("title")) or paper_id,
                "status": status,
                "parser": None,
                "source_pdf": str(source_pdf) if source_pdf else None,
                "source_sha256": source_sha256,
                "source_download_sha256": source_download_sha256,
                "source_manifest": str(paper_set_path),
                "source_manifest_sha256": paper_set_sha256,
                "paper_collect_manifest": str(prior_manifest_path),
                "paper_collect_manifest_sha256": prior_manifest_sha256,
                "paper_collect_run_id": paper_collect_run_id,
                "source_pdf_root": str(pdf_root),
                "paper_edges_path": str(edge_path),
                "paper_edges_sha256": paper_edges_sha256,
                "markdown_path": None,
                "content_list_path": None,
                "middle_json_path": None,
                "model_json_path": None,
                "images_dir": None,
                "markdown_sha256": None,
                "content_list_sha256": None,
                "middle_json_sha256": None,
                "blocker": blocker,
                "failure_phase": "input" if status == "failed" else None,
                "metadata": paper,
            }
        )

    manifest: JsonObject = {
        "schema_version": "xlab.paper_documents.v2",
        "generated_at": utc_now(),
        "input_directory": str(collection_root),
        "paper_set_path": str(paper_set_path),
        "paper_set_sha256": paper_set_sha256,
        "paper_edges_path": str(edge_path),
        "paper_edges_sha256": paper_edges_sha256,
        "prior_manifest_path": str(prior_manifest_path),
        "prior_manifest_sha256": prior_manifest_sha256,
        "paper_collect_run_id": paper_collect_run_id,
        "paper_collect_skill_version": text(prior_manifest.get("skill_version")),
        "pdf_root": str(pdf_root),
        "documents": documents,
        "summary": {
            "total": len(documents),
            "downloaded_pdfs": sum(item.get("status") == "pending" for item in documents),
            "parsed": 0,
            "metadata_only": sum(
                item.get("status") == "metadata_only" for item in documents
            ),
            "failed": sum(item.get("status") == "failed" for item in documents),
            "reused": 0,
        },
    }
    atomic_write_json(paths["documents"], manifest)
    return manifest
