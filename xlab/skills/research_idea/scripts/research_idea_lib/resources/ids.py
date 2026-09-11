"""Stable identifiers for survey papers and evidence."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")


def stable_paper_id(paper: Mapping[str, Any]) -> str:
    """Prefer normalized scholarly IDs, with a stable metadata fallback."""

    external_ids = paper.get("external_ids")
    external_ids = external_ids if isinstance(external_ids, Mapping) else {}
    semantic_scholar = _first_string(paper, "paper_id", "semantic_scholar_id", "s2_id") or _first_string(
        external_ids, "CorpusId", "SemanticScholar"
    )
    if semantic_scholar:
        lowered = semantic_scholar.casefold()
        if lowered.startswith("s2:"):
            semantic_scholar = semantic_scholar[3:]
        return f"s2:{semantic_scholar.strip().casefold()}"

    doi = _first_string(paper, "doi") or _first_string(external_ids, "DOI", "doi")
    if doi:
        normalized = doi.casefold().removeprefix("https://doi.org/").removeprefix("http://doi.org/").removeprefix("doi:")
        return f"doi:{normalized.strip()}"

    arxiv = _first_string(paper, "arxiv_id", "arxiv") or _first_string(external_ids, "ArXiv", "arxiv")
    if arxiv:
        normalized = arxiv.casefold().removeprefix("https://arxiv.org/abs/").removeprefix("http://arxiv.org/abs/").removeprefix("arxiv:")
        return f"arxiv:{normalized.strip()}"

    payload = {
        "title": _normalize_text(_first_string(paper, "title") or ""),
        "year": str(paper.get("year") or "").strip(),
        "authors": [_normalize_text(str(author)) for author in paper.get("authors", []) if str(author).strip()]
        if isinstance(paper.get("authors"), list)
        else [],
    }
    return f"paper:{_stable_digest(payload)[:20]}"


def stable_evidence_id(
    kind: str,
    text: str,
    *,
    paper_ids: tuple[str, ...] | list[str] = (),
    source_id: str | None = None,
) -> str:
    """Build an order-independent evidence identity from normalized content."""

    payload = {
        "kind": kind.strip().casefold(),
        "text": _normalize_text(text),
        "paper_ids": sorted({paper_id.strip().casefold() for paper_id in paper_ids if paper_id.strip()}),
        "source_id": source_id.strip().casefold() if source_id and source_id.strip() else None,
    }
    return f"evidence:{_stable_digest(payload)[:20]}"


def _first_string(value: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _normalize_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value).strip().casefold()


def _stable_digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
