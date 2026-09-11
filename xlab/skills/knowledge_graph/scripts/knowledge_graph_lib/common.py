from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


JsonObject = dict[str, object]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def is_mapping(value: object) -> bool:
    return isinstance(value, Mapping)


def as_mapping(value: object) -> JsonObject:
    if not is_mapping(value):
        return {}
    return {str(key): item for key, item in value.items()}


def as_list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def integer(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_space(value: object) -> str:
    return re.sub(r"\s+", " ", text(value)).strip()


def normalize_name(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", normalize_space(value)).lower()
    return re.sub(r"[^a-z0-9+]+", " ", normalized).strip()


def compact_key(value: object) -> str:
    return re.sub(r"[^a-z0-9+]+", "", normalize_name(value))


def stable_id(kind: str, *parts: object) -> str:
    source = "\x1f".join(normalize_space(part) for part in parts)
    digest = hashlib.sha256(f"{kind}\x1e{source}".encode("utf-8")).hexdigest()[:20]
    return f"{kind.lower()}:{digest}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


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
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_json(path: Path, default: object = None) -> object:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[object]:
    if not path.exists():
        return []
    values: list[object] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                values.append(json.loads(stripped))
            except json.JSONDecodeError as error:
                values.append(
                    {
                        "_parse_error": f"line {line_number}: {error}",
                        "_raw": stripped[:2000],
                    }
                )
    return values


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def artifact_paths(run_dir: Path) -> dict[str, Path]:
    artifacts = run_dir / "artifacts"
    return {
        "artifacts": artifacts,
        "request": artifacts / "request.json",
        "resource_plan": artifacts / "resource_plan.json",
        "documents": artifacts / "paper_documents.json",
        "mineru": artifacts / "mineru",
        "mineru_staging": artifacts / ".mineru-input",
        "mineru_manifest": artifacts / "mineru.manifest.json",
        "structures": artifacts / "paper_structures",
        "structure_manifest": artifacts / "paper_structures.manifest.json",
        "llm": artifacts / "llm",
        "extractions": artifacts / "extractions",
        "extraction_manifest": artifacts / "extractions.manifest.json",
        "nodes": artifacts / "nodes.jsonl",
        "edges": artifacts / "edges.jsonl",
        "aliases": artifacts / "entity_aliases.jsonl",
        "database": artifacts / "graph.db",
        "visualizations": artifacts / "visualizations",
        "graph": artifacts / "method_graph.json",
        "report": artifacts / "graph_report.json",
        "failures": artifacts / "failures.jsonl",
        "logs": artifacts / "logs" / "pipeline.jsonl",
        "manifest": run_dir / "manifest.json",
    }


@contextmanager
def run_lock(run_dir: Path, phase: str) -> Iterator[None]:
    lock_path = run_dir / "artifacts" / ".locks" / "run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            owner = handle.read().strip()
            raise RuntimeError(
                f"another knowledge_graph phase holds the run lock{f': {owner}' if owner else ''}"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps({"pid": os.getpid(), "phase": phase, "started_at": utc_now()})
        )
        handle.flush()
        try:
            yield
        finally:
            handle.seek(0)
            handle.truncate()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
