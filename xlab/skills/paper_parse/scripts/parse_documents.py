#!/usr/bin/env python3
"""Normalize local scholarly documents without referencing source repositories."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def read_text_source(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="replace"), "text"
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            chunks = [
                str(item.get("text") or item.get("content") or "")
                for item in value
                if isinstance(item, dict)
            ]
            return "\n\n".join(chunk for chunk in chunks if chunk.strip()), "mineru_content_list"
        if isinstance(value, dict):
            return str(value.get("markdown") or value.get("text") or ""), "structured_json"
    if suffix == ".pdf" and shutil.which("pdftotext"):
        result = subprocess.run(
            ["pdftotext", str(path), "-"],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.stdout, "pdftotext"
    raise ValueError(f"no parser available for {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    papers = source.get("papers")
    if not isinstance(papers, list):
        raise ValueError("fetch manifest must contain papers")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    documents: list[dict[str, object]] = []
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        document: dict[str, object] = {
            "paper_id": str(paper.get("id") or ""),
            "title": str(paper.get("title") or ""),
            "status": "metadata_only",
            "parser": None,
            "text_path": None,
            "sha256": None,
            "blocker": paper.get("blocker"),
        }
        raw_path = paper.get("document_path") or paper.get("pdf_path")
        if raw_path:
            try:
                text, parser_name = read_text_source(Path(str(raw_path)))
                if not text.strip():
                    raise ValueError("parser returned empty text")
                destination = output_dir / f"{document['paper_id'] or hashlib.sha256(text.encode()).hexdigest()[:16]}.md"
                destination.write_text(text.strip() + "\n", encoding="utf-8")
                document.update(
                    {
                        "status": "parsed",
                        "parser": parser_name,
                        "text_path": str(destination),
                        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                        "blocker": None,
                    }
                )
            except Exception as exc:
                document.update({"status": "failed", "blocker": str(exc)})
        documents.append(document)

    manifest = {
        "schema_version": "xlab.paper_documents.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_path": args.input,
        "documents": documents,
        "summary": {
            "total": len(documents),
            "parsed": sum(item["status"] == "parsed" for item in documents),
            "failed": sum(item["status"] == "failed" for item in documents),
        },
    }
    Path(args.manifest).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": args.manifest, **manifest["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
