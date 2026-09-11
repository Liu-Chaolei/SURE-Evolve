from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .common import project_relative, read_json, resolve_project_path, stable_id

STOPWORDS = {
    "about",
    "after",
    "also",
    "among",
    "based",
    "between",
    "from",
    "into",
    "method",
    "model",
    "paper",
    "study",
    "that",
    "their",
    "these",
    "this",
    "using",
    "with",
}


def tokens(text: str) -> list[str]:
    cleaned = "".join(character.lower() if character.isalnum() else " " for character in text)
    return [token for token in cleaned.split() if len(token) > 3 and token not in STOPWORDS]


def follow_manifest_artifacts(value: Any, cwd: Path) -> list[dict[str, Any]]:
    return list(iter_manifest_artifact_papers(value, cwd))


def extract_papers(value: Any, cwd: Path) -> list[dict[str, Any]]:
    return list(iter_papers(value, cwd))


def parent_artifacts_for_source(source_path: Path | None, cwd: Path, source_type: str) -> list[dict[str, Any]]:
    if source_path is None or not source_path.exists():
        return []
    if source_type == "knowledge_graph":
        return graph_parent_artifacts(source_path, cwd)
    if source_type == "paper_manifest":
        return input_parent_artifacts(source_path, cwd)
    return []


def iter_papers(value: Any, cwd: Path) -> Iterator[dict[str, Any]]:
    if isinstance(value, list):
        for paper in value:
            if isinstance(paper, dict):
                yield paper
        return
    if not isinstance(value, dict):
        return
    for key in ("papers", "items", "references", "citations"):
        candidate = value.get(key)
        if isinstance(candidate, list):
            for paper in candidate:
                if isinstance(paper, dict):
                    yield paper
            return
    yield from iter_manifest_artifact_papers(value, cwd)


def iter_manifest_artifact_papers(value: Any, cwd: Path) -> Iterator[dict[str, Any]]:
    if not isinstance(value, dict):
        return
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list):
        return
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        artifact_type = artifact.get("type")
        path_value = artifact.get("path")
        if artifact_type not in {"paper_set", "paper_collection", "literature_survey_json"}:
            continue
        if not isinstance(path_value, str) or not path_value.strip():
            continue
        path = resolve_project_path(cwd, path_value)
        if path is None or not path.exists() or not path.is_file():
            continue
        yield from iter_papers(read_json(path), cwd)
        return


def normalize_papers(papers: Iterable[dict[str, Any]], max_papers: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_count = 0
    duplicate_count = 0
    truncated_count = 0
    for index, paper in enumerate(papers):
        source_count += 1
        title = str(paper.get("title") or paper.get("name") or f"Untitled paper {index + 1}").strip()
        paper_id = str(
            paper.get("id")
            or paper.get("paper_id")
            or paper.get("paperId")
            or paper.get("corpusId")
            or stable_id("paper", title, paper.get("year"), index)
        )
        if paper_id in seen:
            duplicate_count += 1
            continue
        seen.add(paper_id)
        authors = paper.get("authors")
        if isinstance(authors, list):
            normalized_authors = [str(author.get("name") if isinstance(author, dict) else author) for author in authors]
        else:
            normalized_authors = []
        year = paper.get("year")
        open_access = paper.get("openAccessPdf")
        url = paper.get("url")
        if not url and isinstance(open_access, dict):
            url = open_access.get("url")
        if len(normalized) < max_papers:
            normalized.append(
                {
                    "id": paper_id,
                    "title": title,
                    "abstract": str(paper.get("abstract") or paper.get("summary") or "").strip(),
                    "year": year if isinstance(year, int) else None,
                    "venue": str(paper.get("venue") or paper.get("publicationVenue") or "").strip(),
                    "authors": normalized_authors,
                    "url": url,
                    "citation_count": paper.get("citationCount") or paper.get("citation_count") or 0,
                    "source": str(paper.get("source") or "input"),
                }
            )
        else:
            truncated_count += 1
    return normalized, {"source_count": source_count, "duplicate_count": duplicate_count, "truncated_count": truncated_count}


def synthetic_topic_papers(topic: str, minimum: int) -> list[dict[str, Any]]:
    aspects = ["foundations", "systems", "evaluation", "applications", "limitations"]
    count = max(3, minimum)
    return [
        {
            "id": stable_id("paper", topic, aspect),
            "title": f"{topic}: {aspect.title()}",
            "abstract": f"Synthetic placeholder evidence about {aspect} for {topic}. Replace this with a paper_collect manifest for a real survey.",
            "year": 2020 + (index % 5),
            "venue": "synthetic",
            "authors": [],
            "url": None,
            "citation_count": 0,
            "source": "synthetic",
        }
        for index, aspect in enumerate(aspects[:count])
    ]


def file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def xlab_file_artifact_digest(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    digest.update(b"file\0")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_final_manifest(value: Any) -> bool:
    return isinstance(value, dict) and value.get("schema_version") == "2" and isinstance(value.get("artifacts"), list)


def manifest_artifact_entries(manifest_path: Path, cwd: Path) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path)
    if not is_final_manifest(manifest):
        return []
    entries: list[dict[str, Any]] = []
    artifacts = manifest.get("artifacts", []) if isinstance(manifest.get("artifacts"), list) else []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        path_value = artifact.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            continue
        artifact_path = resolve_manifest_artifact_path(manifest_path, cwd, path_value)
        artifact_sha256 = xlab_file_artifact_digest(artifact_path)
        entries.append(
            {
                "skill_name": manifest.get("skill_name"),
                "skill_version": manifest.get("skill_version"),
                "run_id": manifest.get("run_id"),
                "manifest_path": project_relative(cwd, manifest_path),
                "manifest_sha256": file_sha256(manifest_path),
                "artifact_type": artifact.get("type"),
                "artifact_schema_version": artifact.get("schema_version"),
                "artifact_path": project_relative(cwd, artifact_path) if artifact_path else path_value,
                "artifact_sha256": artifact_sha256,
                "content_sha256": file_sha256(artifact_path),
                "artifact_id": f"sha256:{artifact_sha256}" if artifact_sha256 else None,
                "parents": artifact.get("parents", []) if isinstance(artifact.get("parents"), list) else [],
            }
        )
    return entries


def resolve_manifest_artifact_path(manifest_path: Path, cwd: Path, path_value: str) -> Path | None:
    candidates = [resolve_project_path(cwd, path_value)]
    if not Path(path_value).expanduser().is_absolute():
        candidates.append((manifest_path.parent / path_value).resolve())
    return next((candidate for candidate in candidates if candidate and candidate.exists() and candidate.is_file()), candidates[0])


def input_parent_artifacts(input_path: Path, cwd: Path) -> list[dict[str, Any]]:
    value = read_json(input_path)
    if is_final_manifest(value):
        entries = manifest_artifact_entries(input_path, cwd)
        return [entry for entry in entries if entry.get("artifact_type") in {"paper_set", "paper_collection", "literature_survey_json"}]
    digest = file_sha256(input_path)
    if not digest:
        return []
    return [
        {
            "skill_name": None,
            "skill_version": None,
            "run_id": None,
            "manifest_path": None,
            "manifest_sha256": None,
            "artifact_type": "paper_manifest",
            "artifact_schema_version": value.get("schema_version") if isinstance(value, dict) else None,
            "artifact_path": project_relative(cwd, input_path),
            "artifact_sha256": digest,
            "artifact_id": None,
            "parents": [],
        }
    ]


def graph_parent_artifacts(graph_path: Path, cwd: Path) -> list[dict[str, Any]]:
    artifacts = resolve_graph_artifacts(graph_path, cwd)
    manifest_path = artifacts.get("manifest")
    if manifest_path and manifest_path.exists():
        entries = manifest_artifact_entries(manifest_path, cwd)
        graph_entries = [entry for entry in entries if entry.get("artifact_type") in {"method_graph", "graph_db", "graph_report"}]
        if graph_entries:
            return graph_entries
    parents: list[dict[str, Any]] = []
    for artifact_type in ("method_graph", "graph_db"):
        artifact_path = artifacts.get(artifact_type)
        digest = file_sha256(artifact_path)
        if artifact_path and digest:
            parents.append(
                {
                    "skill_name": "knowledge_graph",
                    "skill_version": None,
                    "run_id": None,
                    "manifest_path": project_relative(cwd, manifest_path) if manifest_path else None,
                    "manifest_sha256": file_sha256(manifest_path) if manifest_path else None,
                    "artifact_type": artifact_type,
                    "artifact_schema_version": None,
                    "artifact_path": project_relative(cwd, artifact_path),
                    "artifact_sha256": digest,
                    "artifact_id": None,
                    "parents": [],
                }
            )
    return parents


def resolve_graph_artifacts(graph_path: Path, cwd: Path) -> dict[str, Path | None]:
    path = graph_path.resolve()
    manifest_path: Path | None = None
    method_graph_path: Path | None = None
    graph_db_path: Path | None = None
    if path.is_dir():
        manifest_path = path / "manifest.json" if (path / "manifest.json").exists() else None
        method_graph_path = path / "artifacts" / "method_graph.json" if (path / "artifacts" / "method_graph.json").exists() else None
        graph_db_path = path / "artifacts" / "graph.db" if (path / "artifacts" / "graph.db").exists() else None
    elif path.name == "manifest.json" and path.exists():
        manifest_path = path
    elif path.suffix == ".db" and path.exists():
        graph_db_path = path
    elif path.exists():
        method_graph_path = path
    if manifest_path and manifest_path.exists():
        manifest = read_json(manifest_path)
        if isinstance(manifest, dict):
            for artifact in manifest.get("artifacts", []) if isinstance(manifest.get("artifacts"), list) else []:
                if not isinstance(artifact, dict):
                    continue
                artifact_path = artifact.get("path")
                if not isinstance(artifact_path, str):
                    continue
                candidates = [resolve_project_path(cwd, artifact_path)]
                manifest_relative = (manifest_path.parent / artifact_path).resolve() if not Path(artifact_path).expanduser().is_absolute() else None
                if manifest_relative is not None:
                    candidates.append(manifest_relative)
                resolved = next((candidate for candidate in candidates if candidate and candidate.exists()), None)
                if not resolved:
                    continue
                if artifact.get("type") == "method_graph":
                    method_graph_path = method_graph_path or resolved
                if artifact.get("type") == "graph_db":
                    graph_db_path = graph_db_path or resolved
    return {"manifest": manifest_path, "method_graph": method_graph_path, "graph_db": graph_db_path}


def graph_papers_and_context(graph_path: Path, cwd: Path, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    artifacts = resolve_graph_artifacts(graph_path, cwd)
    warnings: list[str] = []
    if artifacts["graph_db"]:
        papers, context = graph_db_papers_and_context(artifacts["graph_db"], limit)
        if papers:
            context["artifacts"] = {key: str(value) for key, value in artifacts.items() if value}
            return papers, context, warnings
        warnings.append("Knowledge graph database did not yield Paper nodes; falling back to method_graph JSON if available.")
    if artifacts["method_graph"]:
        papers, context = method_graph_papers_and_context(artifacts["method_graph"], limit)
        context["artifacts"] = {key: str(value) for key, value in artifacts.items() if value}
        return papers, context, warnings
    warnings.append("No readable graph_db or method_graph artifact was found for --graph.")
    return [], {"source": str(graph_path), "available": False}, warnings


def graph_db_papers_and_context(path: Path, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    papers: list[dict[str, Any]] = []
    paper_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cores: list[str] = []
    relation_counts: Counter[str] = Counter()
    uri = f"file:{path}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error:
        return [], {"source_type": "graph_db", "path": str(path), "available": False}
    try:
        table_names = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if isinstance(row[0], str)
        }
        if "nodes" not in table_names:
            return [], {"source_type": "graph_db", "path": str(path), "available": False, "reason": "missing nodes table"}
        columns = [row[1] for row in connection.execute("PRAGMA table_info(nodes)").fetchall()]
        type_column = "node_type" if "node_type" in columns else "type" if "type" in columns else None
        if type_column:
            paper_rows = connection.execute(
                f"SELECT * FROM nodes WHERE {type_column} = ? LIMIT ?", ("Paper", limit)
            ).fetchall()
            papers = [paper_from_graph_node(dict(zip(columns, row)), index) for index, row in enumerate(paper_rows)]
        row_limit = max(limit * 40, limit)
        rows = connection.execute("SELECT * FROM nodes LIMIT ?", (row_limit,)).fetchall()
        for row in rows:
            node = dict(zip(columns, row))
            node_type = str(node.get("node_type") or node.get("type") or "")
            name = str(node.get("name") or node.get("full_name") or node.get("label") or node.get("title") or "")
            paper_id = str(node.get("paper_id") or "").strip()
            if node_type == "Paper":
                continue
            if paper_id and len(paper_records) < limit:
                paper_records[paper_id].append(node)
            elif paper_id and paper_id in paper_records:
                paper_records[paper_id].append(node)
            if node_type == "Core" and name and len(cores) < limit:
                cores.append(name)
        if not papers and paper_records:
            papers = [paper_from_graph_records(records, index) for index, records in enumerate(paper_records.values())]
        if "edges" in table_names:
            edge_columns = [row[1] for row in connection.execute("PRAGMA table_info(edges)").fetchall()]
            relation_column = "relation" if "relation" in edge_columns else "edge_type" if "edge_type" in edge_columns else None
            if relation_column:
                for (relation,) in connection.execute(f"SELECT {relation_column} FROM edges LIMIT ?", (limit * 10,)).fetchall():
                    relation_counts[str(relation)] += 1
    finally:
        connection.close()
    return papers[:limit], {
        "source_type": "graph_db",
        "path": str(path),
        "available": True,
        "core_terms": cores,
        "relation_counts": dict(relation_counts),
        "counts": {"papers": len(papers), "core_terms": len(cores)},
    }


def method_graph_papers_and_context(path: Path, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    value = read_json(path)
    nodes = value.get("nodes", []) if isinstance(value, dict) else []
    edges = value.get("edges", []) if isinstance(value, dict) else []
    papers: list[dict[str, Any]] = []
    core_terms: list[str] = []
    relation_counts: Counter[str] = Counter()
    if isinstance(nodes, list):
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_type = node.get("node_type")
            if node_type == "Paper" and len(papers) < limit:
                papers.append(paper_from_graph_node(node, len(papers)))
            elif node_type == "Core" and len(core_terms) < limit:
                core_terms.append(str(node.get("name") or node.get("id") or ""))
    if isinstance(edges, list):
        for edge in edges[: limit * 10]:
            if isinstance(edge, dict):
                relation_counts[str(edge.get("relation") or "unknown")] += 1
    return papers, {
        "source_type": "method_graph",
        "path": str(path),
        "available": True,
        "core_terms": [term for term in core_terms if term],
        "relation_counts": dict(relation_counts),
        "counts": value.get("counts", {}) if isinstance(value, dict) else {},
    }


def paper_from_graph_node(node: dict[str, Any], index: int) -> dict[str, Any]:
    return paper_from_graph_records([node], index)


def paper_from_graph_records(records: list[dict[str, Any]], index: int) -> dict[str, Any]:
    records = [
        {**_parse_json_object(record.get("raw_json")), **{key: value for key, value in record.items() if value is not None and value != ""}}
        for record in records
    ]
    first = records[0] if records else {}
    provenance = _parse_json_object(first.get("provenance"))
    metadata = _parse_json_object(first.get("metadata"))
    evidence = _parse_json_object(first.get("evidence"))
    paper_id = _first_text(
        first.get("paper_id"),
        provenance.get("paper_id"),
        metadata.get("paper_id"),
        evidence.get("paper_id"),
        first.get("id"),
        stable_id("paper", first.get("paper_title") or first.get("name") or first.get("label"), index),
    )
    title = _first_text(
        first.get("paper_title"),
        first.get("title"),
        metadata.get("title"),
        evidence.get("title"),
        first.get("name"),
        first.get("full_name"),
        first.get("label"),
        f"Graph paper {index + 1}",
    )
    abstract_parts: list[str] = []
    for record in records:
        abstract_parts.extend(
            _unique_texts(
                record.get("abstract"),
                record.get("tldr"),
                record.get("summary"),
                record.get("insight"),
                record.get("quote"),
            )
        )
        structured = _structured_summary(record)
        abstract_parts.extend(
            _unique_texts(
                structured.get("background"),
                structured.get("method"),
                structured.get("result"),
                structured.get("conclusion"),
            )
        )
    abstract = " ".join(_unique_texts(metadata.get("abstract"), evidence.get("abstract"), provenance.get("summary"), *abstract_parts))
    venue = _first_text(metadata.get("venue"), evidence.get("venue"), first.get("citation_venue"), first.get("source_venue"), default="")
    if venue == paper_id:
        venue = ""
    authors = metadata.get("authors") if isinstance(metadata.get("authors"), list) else []
    url = _first_url(metadata.get("url"), evidence.get("url"), first.get("url"), first.get("urls"), first.get("citation_urls"), first.get("code_url"))
    return {
        "id": paper_id,
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "year": _parse_int(metadata.get("year"), first.get("year"), first.get("pub_year")),
        "venue": venue,
        "authors": authors,
        "url": url,
        "citation_count": _parse_int(metadata.get("citation_count"), first.get("citation_count")) or 0,
        "source": "knowledge_graph",
    }


def _first_text(*values: object, default: str = "") -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def _unique_texts(*values: object) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _parse_json_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _structured_summary(record: dict[str, Any]) -> dict[str, Any]:
    structured = _parse_json_object(record.get("structured_summary"))
    if structured:
        return structured
    raw = _parse_json_object(record.get("raw_json"))
    nested = _parse_json_object(raw.get("structured_summary"))
    return nested or raw


def _parse_int(*values: object) -> int | None:
    for value in values:
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip():
            try:
                return int(float(value.strip()))
            except ValueError:
                continue
    return None


def _first_url(*values: object) -> str | None:
    for value in values:
        if isinstance(value, list):
            nested = _first_url(*value)
            if nested:
                return nested
        if isinstance(value, str) and value.strip():
            text = value.strip()
            parsed = _parse_json_object(text)
            if parsed:
                nested = _first_url(parsed.get("url"), parsed.get("urls"))
                if nested:
                    return nested
            if text.startswith("["):
                try:
                    parsed_list = json.loads(text)
                except json.JSONDecodeError:
                    parsed_list = []
                if isinstance(parsed_list, list):
                    nested = _first_url(*parsed_list)
                    if nested:
                        return nested
            if text.startswith("http://") or text.startswith("https://"):
                return text
    return None


def build_clusters(topic: str, papers: list[dict[str, Any]], limit: int = 5, graph_context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    keyword_counts = Counter(token for paper in papers for token in tokens(f"{paper['title']} {paper.get('abstract', '')}"))
    graph_terms = [str(term).lower() for term in (graph_context or {}).get("core_terms", []) if str(term).strip()]
    keywords: list[str] = []
    for term in graph_terms:
        keyword = tokens(term)
        if keyword:
            keywords.append(keyword[0])
        if len(keywords) >= limit:
            break
    for keyword, _ in keyword_counts.most_common(limit):
        if keyword not in keywords:
            keywords.append(keyword)
        if len(keywords) >= limit:
            break
    keywords = keywords or tokens(topic)[:limit] or [topic]
    clusters: list[dict[str, Any]] = []
    for keyword in keywords[:limit]:
        paper_ids = [paper["id"] for paper in papers if keyword in tokens(f"{paper['title']} {paper.get('abstract', '')}")]
        if not paper_ids:
            paper_ids = [papers[len(clusters) % len(papers)]["id"]]
        clusters.append(
            {
                "id": stable_id("cluster", topic, keyword),
                "name": keyword,
                "keywords": [keyword],
                "paper_ids": sorted(set(paper_ids)),
                "summary": f"Papers in this cluster discuss {keyword} in relation to {topic}.",
            }
        )
    covered = {paper_id for cluster in clusters for paper_id in cluster["paper_ids"]}
    missing = [paper["id"] for paper in papers if paper["id"] not in covered]
    if missing:
        clusters[0]["paper_ids"] = sorted(set(clusters[0]["paper_ids"] + missing))
    return clusters


def build_chronology(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_year: dict[int, list[str]] = defaultdict(list)
    for paper in papers:
        year = paper.get("year")
        if isinstance(year, int):
            by_year[year].append(paper["id"])
    return [{"year": year, "paper_ids": sorted(ids)} for year, ids in sorted(by_year.items())]
