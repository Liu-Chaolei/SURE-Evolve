from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_jsonl(path: Path, values: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                yield value


def stable_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha1("\0".join(str(part) for part in parts if part is not None).encode("utf-8")).hexdigest()
    return f"{prefix}:{digest[:12]}"


def stable_signature(value: object) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def file_digest(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_root_from_run_dir(run_dir: Path) -> Path:
    resolved = run_dir.resolve()
    if resolved.parent.name == "runs" and resolved.parent.parent.name == ".xlab":
        return resolved.parent.parent.parent
    return Path.cwd().resolve()


def project_relative(cwd: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(cwd.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def resolve_project_path(cwd: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (cwd / path).resolve()


def run_paths(run_dir: Path) -> dict[str, Path]:
    artifacts = run_dir / "artifacts"
    state = run_dir / "state"
    logs = run_dir / "logs"
    return {
        "artifacts": artifacts,
        "state": state,
        "logs": logs,
        "request": state / "request.json",
        "runtime_config": state / "runtime_config.json",
        "checkpoint": state / "checkpoint.json",
        "papers_jsonl": state / "papers.jsonl",
        "paper_index": state / "paper_index.json",
        "graph_context": state / "graph_context.json",
        "clusters_json": state / "clusters.json",
        "chronology_json": state / "chronology.json",
        "survey_agent_outline": state / "survey_agent_outline.json",
        "survey_agent_assignment": state / "survey_agent_assignment.json",
        "survey_agent_draft": state / "survey_agent_draft.md",
        "survey_agent_result": state / "survey_agent_result.json",
        "sections_jsonl": state / "sections.jsonl",
        "claims_jsonl": state / "key_claims.jsonl",
        "gaps_jsonl": state / "research_gaps.jsonl",
        "references_jsonl": state / "references.jsonl",
        "traces_jsonl": state / "traces.jsonl",
        "pipeline_log": logs / "pipeline.jsonl",
        "diagnostics_log": logs / "diagnostics.jsonl",
        "survey_md": artifacts / "survey.md",
        "survey_json": artifacts / "survey.json",
        "citations_json": artifacts / "citations.json",
        "report_json": artifacts / "survey_report.json",
        "manifest_json": run_dir / "manifest.json",
    }


def append_diagnostic(run_dir: Path, severity: str, message: str, *, code: str | None = None, data: object | None = None) -> None:
    entry: dict[str, object] = {"timestamp": utc_now(), "severity": severity, "message": message}
    if code:
        entry["code"] = code
    if data is not None:
        entry["data"] = data
    append_jsonl(run_paths(run_dir)["diagnostics_log"], entry)


def append_pipeline_event(run_dir: Path, stage: str, status: str, data: object | None = None) -> None:
    entry: dict[str, object] = {"timestamp": utc_now(), "stage": stage, "status": status}
    if data is not None:
        entry["data"] = data
    append_jsonl(run_paths(run_dir)["pipeline_log"], entry)
