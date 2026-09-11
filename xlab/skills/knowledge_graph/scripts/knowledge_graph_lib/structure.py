from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .common import (
    JsonObject,
    append_jsonl,
    artifact_paths,
    as_list,
    as_mapping,
    atomic_write_json,
    normalize_space,
    read_json,
    sha256_file,
    text,
    utc_now,
)
from .references import build_reference_map


STRUCTURER_VERSION = "papergraph-step1-adapted-v1"
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _checkpoint_path(directory: Path, paper_id: str) -> Path:
    digest = hashlib.sha256(paper_id.encode("utf-8")).hexdigest()[:24]
    return directory / f"{digest}.json"


def _content_records(value: object) -> list[JsonObject]:
    result: list[JsonObject] = []

    def visit(candidate: object) -> None:
        if isinstance(candidate, list):
            for item in candidate:
                visit(item)
            return
        item = as_mapping(candidate)
        if not item:
            return
        if text(item.get("type")).lower() in {
            "text",
            "image",
            "table",
            "equation",
            "interline_equation",
        } and any(
            item.get(key) is not None
            for key in (
                "text",
                "content",
                "img_path",
                "image_path",
                "table_body",
                "latex",
            )
        ):
            result.append(item)
            return
        for key in (
            "pages",
            "pdf_info",
            "content_list",
            "contents",
            "blocks",
            "para_blocks",
            "children",
            "body",
            "items",
        ):
            visit(item.get(key))

    visit(value)
    return result


def _caption(record: JsonObject) -> str:
    for key in ("caption", "image_caption", "table_caption"):
        value = record.get(key)
        if isinstance(value, list):
            result = normalize_space(" ".join(text(item) for item in value))
        else:
            result = normalize_space(value)
        if result:
            return result
    return ""


def _block_metadata(record: JsonObject) -> JsonObject:
    return {
        key: record.get(key)
        for key in ("page_idx", "bbox", "type", "text_level")
        if record.get(key) is not None
    }


def _fallback_markdown_structure(markdown: str) -> list[JsonObject]:
    matches = list(HEADING.finditer(markdown))
    if not matches:
        return [
            {
                "title": "Document",
                "level": 1,
                "content": [{"type": "text", "data": markdown.strip()}],
            }
        ]
    sections: list[JsonObject] = []
    prefix = markdown[: matches[0].start()].strip()
    if prefix:
        sections.append(
            {
                "title": "Preamble",
                "level": 1,
                "content": [{"type": "text", "data": prefix}],
            }
        )
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        body = markdown[match.end() : end].strip()
        sections.append(
            {
                "title": normalize_space(match.group(2)),
                "level": len(match.group(1)),
                "content": ([{"type": "text", "data": body}] if body else []),
            }
        )
    return sections


def _structure_from_content_list(
    records: list[JsonObject], markdown: str
) -> list[JsonObject]:
    sections: list[JsonObject] = []
    current: JsonObject = {"title": "Preamble", "level": 1, "content": []}
    sections.append(current)
    for record in records:
        block_type = text(record.get("type")).lower()
        value = text(record.get("text") or record.get("content"))
        level_value = record.get("text_level")
        level = int(level_value) if isinstance(level_value, int) else None
        if block_type == "text" and level is not None and value:
            current = {
                "title": normalize_space(value),
                "level": min(max(level, 1), 6),
                "content": [],
            }
            sections.append(current)
            continue
        metadata = _block_metadata(record)
        content = as_list(current.get("content"))
        if block_type == "text" and value:
            content.append({"type": "text", "data": value, "metadata": metadata})
        elif block_type == "image":
            content.append(
                {
                    "type": "image",
                    "caption": _caption(record),
                    "rel_path": text(
                        record.get("img_path")
                        or record.get("image_path")
                        or record.get("path")
                    )
                    or None,
                    "metadata": metadata,
                }
            )
        elif block_type == "table":
            content.append(
                {
                    "type": "table",
                    "markdown": text(
                        record.get("table_body")
                        or record.get("table_markdown")
                        or record.get("text")
                    ),
                    "caption": _caption(record),
                    "metadata": metadata,
                }
            )
        elif block_type in {"equation", "interline_equation"}:
            content.append(
                {
                    "type": "equation",
                    "latex": text(
                        record.get("text")
                        or record.get("latex")
                        or record.get("content")
                    ),
                    "metadata": metadata,
                }
            )
        current["content"] = content
    nonempty = [
        section
        for section in sections
        if as_list(section.get("content"))
        or normalize_space(section.get("title")) not in {"", "Preamble"}
    ]
    return nonempty or _fallback_markdown_structure(markdown)


def _layout_summary(value: JsonObject) -> JsonObject:
    pages = [as_mapping(item) for item in as_list(value.get("pdf_info"))]
    return {
        "page_count": len(pages),
        "paragraph_block_count": sum(
            len(as_list(page.get("para_blocks"))) for page in pages
        ),
        "discarded_block_count": sum(
            len(as_list(page.get("discarded_blocks"))) for page in pages
        ),
        "backend": value.get("_backend"),
        "mineru_version": value.get("_version_name"),
    }


def _structure_document(
    document: JsonObject, references: list[JsonObject]
) -> JsonObject:
    references_sha256 = _json_sha256(references)
    paper_id = text(document.get("paper_id"))
    markdown_path = Path(text(document.get("markdown_path")))
    content_list_path = Path(text(document.get("content_list_path")))
    middle_path = Path(text(document.get("middle_json_path")))
    markdown = markdown_path.read_text(encoding="utf-8", errors="replace")
    content_value = json.loads(content_list_path.read_text(encoding="utf-8"))
    middle = as_mapping(json.loads(middle_path.read_text(encoding="utf-8")))
    records = _content_records(content_value)
    if not markdown.strip():
        raise ValueError("MinerU Markdown is empty")
    if not records:
        raise ValueError("MinerU content_list has no layout records")
    if not middle:
        raise ValueError("MinerU middle JSON is not an object")
    metadata = as_mapping(document.get("metadata"))
    return {
        "schema_version": "xlab.paper_structure.v1",
        "structurer": STRUCTURER_VERSION,
        "generated_at": utc_now(),
        "paper_id": paper_id,
        "source": str(markdown_path.parent),
        "metadata": {
            "title": normalize_space(document.get("title")) or paper_id,
            "authors": as_list(metadata.get("authors")),
            "year": metadata.get("year"),
            "venue": metadata.get("venue"),
            "abstract": metadata.get("abstract"),
            "tldr": metadata.get("tldr"),
            "external_ids": as_mapping(metadata.get("external_ids")),
            "urls": [
                value
                for value in (
                    metadata.get("url"),
                    as_mapping(metadata.get("open_access_pdf")).get("url"),
                )
                if text(value)
            ],
        },
        "raw_stats": {
            "sections": 0,
            "markdown_chars": len(markdown),
            "content_blocks": len(records),
        },
        "structure": _structure_from_content_list(records, markdown),
        "semantic_references": references,
        "layout_summary": _layout_summary(middle),
        "mineru_output": {
            "paper_dir": str(markdown_path.parent),
            "markdown": str(markdown_path),
            "content_list_json": str(content_list_path),
            "middle_json": str(middle_path),
            "model_json": document.get("model_json_path"),
            "images_dir": document.get("images_dir"),
        },
        "provenance": {
            "source_manifest": document.get("source_manifest"),
            "source_manifest_sha256": document.get("source_manifest_sha256"),
            "paper_collect_manifest": document.get("paper_collect_manifest"),
            "paper_collect_manifest_sha256": document.get("paper_collect_manifest_sha256"),
            "paper_collect_run_id": document.get("paper_collect_run_id"),
            "paper_edges_path": document.get("paper_edges_path"),
            "paper_edges_sha256": document.get("paper_edges_sha256"),
            "semantic_references_sha256": references_sha256,
            "source_pdf_root": document.get("source_pdf_root"),
            "source_pdf": document.get("source_pdf"),
            "source_sha256": document.get("source_sha256"),
            "source_download_sha256": document.get("source_download_sha256"),
            "markdown_sha256": document.get("markdown_sha256"),
            "content_list_sha256": document.get("content_list_sha256"),
            "middle_json_sha256": document.get("middle_json_sha256"),
        },
    }


def _cache_valid(checkpoint: Path, document: JsonObject) -> JsonObject | None:
    value = as_mapping(read_json(checkpoint))
    provenance = as_mapping(value.get("provenance"))
    edge_path = Path(text(document.get("paper_edges_path")))
    current_edges_sha256 = sha256_file(edge_path) if edge_path.is_file() else ""
    if (
        value.get("schema_version") == "xlab.paper_structure.v1"
        and value.get("structurer") == STRUCTURER_VERSION
        and text(provenance.get("markdown_sha256"))
        == text(document.get("markdown_sha256"))
        and text(provenance.get("content_list_sha256"))
        == text(document.get("content_list_sha256"))
        and text(provenance.get("middle_json_sha256"))
        == text(document.get("middle_json_sha256"))
        and text(provenance.get("source_sha256"))
        == text(document.get("source_sha256"))
        and text(provenance.get("source_download_sha256"))
        == text(document.get("source_download_sha256"))
        and text(provenance.get("source_manifest_sha256"))
        == text(document.get("source_manifest_sha256"))
        and text(provenance.get("paper_collect_manifest_sha256"))
        == text(document.get("paper_collect_manifest_sha256"))
        and text(provenance.get("paper_edges_sha256"))
        == text(document.get("paper_edges_sha256"))
        == current_edges_sha256
        and text(provenance.get("semantic_references_sha256"))
        == text(document.get("semantic_references_sha256"))
    ):
        return value
    return None


def structure_documents(
    run_dir: Path, document_manifest: JsonObject, *, workers: int
) -> JsonObject:
    paths = artifact_paths(run_dir)
    paths["structures"].mkdir(parents=True, exist_ok=True)
    references_by_paper, reference_summary = build_reference_map(document_manifest)
    documents = [
        as_mapping(value)
        for value in as_list(document_manifest.get("documents"))
        if as_mapping(value).get("status") == "parsed"
    ]
    successes: dict[str, Path] = {}
    failures: list[JsonObject] = []
    jobs = {}
    reused = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for document in documents:
            paper_id = text(document.get("paper_id"))
            checkpoint = _checkpoint_path(paths["structures"], paper_id)
            references = references_by_paper.get(paper_id, [])
            document["semantic_references_sha256"] = _json_sha256(references)
            if _cache_valid(checkpoint, document):
                successes[paper_id] = checkpoint
                reused += 1
                continue
            future = executor.submit(
                _structure_document,
                document,
                references,
            )
            jobs[future] = (paper_id, checkpoint)
        for future in as_completed(jobs):
            paper_id, checkpoint = jobs[future]
            try:
                structure = future.result()
                structure["raw_stats"] = {
                    **as_mapping(structure.get("raw_stats")),
                    "sections": len(as_list(structure.get("structure"))),
                }
                atomic_write_json(checkpoint, structure)
                successes[paper_id] = checkpoint
            except Exception as error:
                failure = {
                    "phase": "structure",
                    "paper_id": paper_id,
                    "error": str(error)[:2000],
                    "recorded_at": utc_now(),
                }
                failures.append(failure)
                append_jsonl(paths["failures"], failure)
    records: list[JsonObject] = [
        {"paper_id": paper_id, "path": str(successes[paper_id]), "status": "structured"}
        for paper_id in sorted(successes)
    ]
    records.extend(
        {
            "paper_id": text(failure.get("paper_id")),
            "path": None,
            "status": "failed",
            "error": failure.get("error"),
        }
        for failure in sorted(failures, key=lambda item: text(item.get("paper_id")))
    )
    manifest: JsonObject = {
        "schema_version": "xlab.paper_structures.v1",
        "structurer": STRUCTURER_VERSION,
        "generated_at": utc_now(),
        "documents_path": str(paths["documents"]),
        "references": reference_summary,
        "structures": records,
        "summary": {
            "eligible": len(documents),
            "structured": len(successes),
            "failed": len(failures),
            "reused": reused,
        },
    }
    atomic_write_json(paths["structure_manifest"], manifest)
    return manifest
