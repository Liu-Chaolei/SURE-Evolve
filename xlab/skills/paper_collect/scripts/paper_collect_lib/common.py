from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import fcntl


JsonObject = dict[str, object]

_ARXIV_RE = re.compile(
    r"(?:arxiv\s*[:/]\s*|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_STOP_WORDS = {
    "about",
    "after",
    "also",
    "among",
    "and",
    "are",
    "before",
    "between",
    "for",
    "from",
    "into",
    "its",
    "latest",
    "method",
    "methods",
    "paper",
    "papers",
    "research",
    "review",
    "survey",
    "that",
    "the",
    "their",
    "this",
    "through",
    "using",
    "with",
}


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


def integer(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_space(value: object) -> str:
    return re.sub(r"\s+", " ", text(value)).strip()


def normalize_title(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", normalize_space(value)).lower()
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def normalize_doi(value: object) -> str:
    doi = text(value).lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix) :]
    return doi.rstrip(".,;) ]")


def normalize_arxiv(value: object) -> str:
    arxiv_id = text(value)
    match = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?", arxiv_id)
    return match.group(1) if match else arxiv_id.removeprefix("ARXIV:").strip()


def extract_external_ids(value: object) -> JsonObject:
    source = as_mapping(value)
    result: JsonObject = {}
    for key, item in source.items():
        item_text = text(item)
        if not item_text:
            continue
        lowered = key.lower()
        if lowered == "doi":
            result["DOI"] = normalize_doi(item_text)
        elif lowered == "arxiv":
            result["ArXiv"] = normalize_arxiv(item_text)
        else:
            result[key] = item_text
    return result


def extract_ids_from_text(value: object) -> JsonObject:
    source = text(value)
    result: JsonObject = {}
    doi = _DOI_RE.search(source)
    arxiv = _ARXIV_RE.search(source)
    if doi:
        result["DOI"] = normalize_doi(doi.group(0))
    if arxiv:
        result["ArXiv"] = normalize_arxiv(arxiv.group(1))
    return result


def identity_keys(paper: Mapping[str, object]) -> list[str]:
    keys: list[str] = []
    paper_id = text(paper.get("paper_id"))
    external_ids = extract_external_ids(paper.get("external_ids"))
    doi = text(external_ids.get("DOI"))
    arxiv = text(external_ids.get("ArXiv"))
    title_key = normalize_title(paper.get("title"))
    if paper_id:
        keys.append(f"s2:{paper_id.lower()}")
    if doi:
        keys.append(f"doi:{doi}")
    if arxiv:
        keys.append(f"arxiv:{arxiv.lower()}")
    if title_key:
        keys.append(f"title:{title_key}")
    return keys


def stable_paper_id(paper: Mapping[str, object]) -> str:
    keys = identity_keys(paper)
    for prefix in ("s2:", "doi:", "arxiv:", "title:"):
        match = next((key for key in keys if key.startswith(prefix)), None)
        if match:
            if prefix != "title:":
                return match
            digest = hashlib.sha1(match.encode("utf-8")).hexdigest()[:16]
            return f"title:{digest}"
    return ""


def topic_tokens(query: str, facets: Iterable[object] = ()) -> set[str]:
    combined = " ".join([query, *(text(facet) for facet in facets)])
    tokens = re.findall(r"[a-z0-9][a-z0-9+-]{1,}", normalize_title(combined))
    return {token for token in tokens if len(token) > 2 and token not in _STOP_WORDS}


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


@contextmanager
def run_lock(run_dir: Path, phase: str) -> Iterator[None]:
    lock_path = run_dir / "artifacts" / ".locks" / "run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            holder = normalize_space(handle.read()) or "unknown holder"
            raise RuntimeError(
                f"Another paper_collect phase is active for this run: {holder}"
            ) from error
        handle.seek(0)
        handle.truncate()
        json.dump(
            {
                "pid": os.getpid(),
                "phase": phase,
                "acquired_at": utc_now(),
            },
            handle,
            ensure_ascii=False,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def relative_to_run(path: Path, run_dir: Path) -> str:
    return path.resolve().relative_to(run_dir.resolve()).as_posix()


def is_safe_relative_path(value: object) -> bool:
    path_text = text(value)
    if not path_text:
        return False
    path = Path(path_text)
    return not path.is_absolute() and ".." not in path.parts


def artifact_paths(run_dir: Path) -> dict[str, Path]:
    artifacts = run_dir / "artifacts"
    metadata = artifacts / "metadata"
    logs = artifacts / "logs"
    return {
        "run": run_dir,
        "artifacts": artifacts,
        "metadata": metadata,
        "pdfs": artifacts / "pdfs",
        "logs": logs,
        "request": artifacts / "request.json",
        "query_plan": artifacts / "query_plan.json",
        "provider_results": logs / "provider_results.jsonl",
        "candidates": artifacts / "raw_candidates.jsonl",
        "papers_jsonl": metadata / "papers.jsonl",
        "edges_jsonl": metadata / "edges.jsonl",
        "seeds": metadata / "seeds.json",
        "followups": artifacts / "followups.json",
        "state": artifacts / "state.json",
        "downloads": logs / "downloads.jsonl",
        "download_results": logs / "download_results.jsonl",
        "failures": artifacts / "failures.jsonl",
        "collection": artifacts / "papers.manifest.json",
        "report": artifacts / "collection_report.json",
        "manifest": run_dir / "manifest.json",
    }


def find_named_list(value: object, names: set[str]) -> list[object]:
    if isinstance(value, list):
        return value
    mapping = as_mapping(value)
    for name in names:
        candidate = mapping.get(name)
        if isinstance(candidate, list):
            return candidate
    for key in ("result", "data", "payload", "response"):
        candidate = mapping.get(key)
        if candidate is None:
            continue
        found = find_named_list(candidate, names)
        if found:
            return found
    return []


def find_paper_mapping(value: object) -> JsonObject:
    mapping = as_mapping(value)
    if text(mapping.get("paperId")) or (
        text(mapping.get("title"))
        and (
            is_mapping(mapping.get("externalIds"))
            or mapping.get("authors") is not None
            or mapping.get("year") is not None
        )
    ):
        return mapping
    for key in ("result", "data", "paper", "payload", "response"):
        candidate = mapping.get(key)
        if candidate is None:
            continue
        found = find_paper_mapping(candidate)
        if found:
            return found
    return {}
