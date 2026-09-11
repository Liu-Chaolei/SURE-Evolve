from __future__ import annotations

import re
import sqlite3
import json
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from ..paper_sources import normalize_papers, paper_from_graph_node
from .modules.data_manager import DataManager
from .modules.database import Database
from .modules.paper_graph_retriever import PaperGraphRetriever
from .modules.survey_generator import SurveyGenerator
from .modules.work_analyzer import WorkAnalyzer
from .modules.work_collector import WorkCollector
from .utils.utils import get_hash


@dataclass
class IntegratedSurveyContext:
    config: DictConfig
    paper_ids: list[str]
    paper_by_id: dict[str, dict[str, Any]]
    work_collector: WorkCollector
    database: Database
    work_analyzer: WorkAnalyzer
    survey_generator: SurveyGenerator
    graph_db_path: Path | None = None


class SeededDataManager(DataManager):
    """Original DataManager with XLab graph/input papers preloaded as run-local hints."""

    def __init__(self, config: DictConfig, papers: list[dict[str, Any]]):
        super().__init__(config)
        self.xlab_paper_by_id = {str(paper["id"]): paper for paper in papers if paper.get("id") is not None}
        for paper_id, paper in self.xlab_paper_by_id.items():
            self._cache_xlab_paper(paper_id, paper)

    def _cache_xlab_paper(self, paper_id: str, paper: dict[str, Any]) -> None:
        title = str(paper.get("title") or paper_id).strip()
        abstract = str(paper.get("abstract") or title).strip()
        if title and abstract:
            self.paper_abstract_cache[get_hash(paper_id)] = {
                "paper_id": paper_id,
                "title": title,
                "abstract": abstract,
            }

    def get_paper_title_abstract(self, paper_id: str, retry: int = 1) -> tuple[str, str]:
        paper = self.xlab_paper_by_id.get(str(paper_id))
        if paper is not None:
            title = str(paper.get("title") or paper_id).strip()
            abstract = str(paper.get("abstract") or title).strip()
            if title and abstract:
                return title, abstract
        return super().get_paper_title_abstract(paper_id, retry=retry)

    def get_paper_title(self, paper_id: str, retry: int = 3) -> str:
        paper = self.xlab_paper_by_id.get(str(paper_id))
        if paper is not None:
            title = str(paper.get("title") or "").strip()
            if title:
                return title
        return super().get_paper_title(paper_id, retry=retry)

    def get_paper_raw_markdown(self, paper_id: str) -> str:
        try:
            return super().get_paper_raw_markdown(paper_id)
        except Exception:
            paper = self.xlab_paper_by_id.get(str(paper_id))
            if paper is None:
                raise
            title = str(paper.get("title") or paper_id).strip()
            abstract = str(paper.get("abstract") or title).strip()
            return f"# {title}\n\n{abstract}".strip()

    def get_paper_with_title(self, title: str):
        local = self._xlab_paper_with_title(title)
        if local is not None:
            return local
        return super().get_paper_with_title(title)

    def get_paper_with_title_batch(self, titles: list[str]):
        results: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        for title in titles:
            local = self._xlab_paper_with_title(title)
            if local is None:
                missing.append(title)
            else:
                results[title] = local
        if missing:
            results.update(super().get_paper_with_title_batch(missing))
        return results

    def _xlab_paper_with_title(self, title: str) -> dict[str, Any] | None:
        normalized = normalize_title(title)
        if not normalized:
            return None
        best: tuple[float, str, dict[str, Any]] | None = None
        for paper_id, paper in self.xlab_paper_by_id.items():
            candidate_title = str(paper.get("title") or paper_id)
            candidate_normalized = normalize_title(candidate_title)
            if not candidate_normalized:
                continue
            if normalized == candidate_normalized:
                best = (1.0, paper_id, paper)
                break
            score = title_similarity(normalized, candidate_normalized)
            if normalized in candidate_normalized or candidate_normalized in normalized:
                score = max(score, 0.95)
            if best is None or score > best[0]:
                best = (score, paper_id, paper)
        if best is None or best[0] < 0.8:
            return None
        _score, paper_id, paper = best
        external_ids = dict(paper.get("externalIds") or {}) if isinstance(paper.get("externalIds"), dict) else {}
        if "." in paper_id and not any(key.lower() == "arxiv" for key in external_ids):
            external_ids["ArXiv"] = paper_id
        result = {
            "paperId": paper_id,
            "title": str(paper.get("title") or paper_id),
            "abstract": str(paper.get("abstract") or ""),
            "externalIds": external_ids,
            "openAccessPdf": {"url": paper.get("url")} if paper.get("url") else {},
            "authors": paper.get("authors") or [],
            "year": paper.get("year"),
            "venue": paper.get("venue") or "",
            "api_platform": "xlab",
        }
        return result


def build_integrated_context(
    *,
    config: DictConfig,
    papers: list[dict[str, Any]],
    graph_db_path: Path | None = None,
) -> IntegratedSurveyContext:
    return build_original_context(config=config, papers=papers, graph_db_path=graph_db_path)


def build_original_context(*, config: DictConfig, papers: list[dict[str, Any]], graph_db_path: Path | None = None) -> IntegratedSurveyContext:
    local_papers = load_local_graph_papers(config)
    alias_path = Path(config.BasicInfo.base_dir) / "state/xcientist/citation_aliases.json"
    if alias_path.exists():
        for reference in json.loads(alias_path.read_text()).get("verified_references", []):
            local_papers[reference["paper_id"]] = {**reference, "id": reference["paper_id"]}
    for paper in papers:
        if paper.get("id") is not None:
            paper_id = str(paper["id"])
            local_papers[paper_id] = {**local_papers.get(paper_id, {}), **paper}
    data_manager = SeededDataManager(config, list(local_papers.values()))
    work_collector = WorkCollector(config, data_manager=data_manager)
    paper_graph_retriever = getattr(work_collector, "paper_graph_retriever", None)
    if paper_graph_retriever is not None:
        paper_graph_retriever.data_manager = data_manager
    if config.ModuleInfo.WorkAnalyzer.use_local_paper_graph_keynotes and paper_graph_retriever is None:
        paper_graph_retriever = PaperGraphRetriever(config, data_manager=data_manager)
    work_collector.graph_paper_ids.update(str(paper["id"]) for paper in papers if paper.get("id") is not None)
    database = Database(config, work_collector)
    work_analyzer = WorkAnalyzer(config, work_collector, paper_graph_retriever=paper_graph_retriever)
    survey_generator = SurveyGenerator(config, work_analyzer, database)
    paper_by_id = {str(paper["id"]): dict(paper) for paper in papers if paper.get("id") is not None}
    return IntegratedSurveyContext(
        config=config,
        paper_ids=list(paper_by_id),
        paper_by_id=paper_by_id,
        work_collector=work_collector,
        database=database,
        work_analyzer=work_analyzer,
        survey_generator=survey_generator,
        graph_db_path=graph_db_path,
    )


def load_local_graph_papers(config: DictConfig) -> dict[str, dict[str, Any]]:
    """Hydrate only the run-local graph copy and preload expansion metadata."""
    path = Path(str(config.ModuleInfo.PaperGraphRetriever.db_path)).resolve()
    state_dir = Path(str(config.BasicInfo.base_dir)).resolve() / "state"
    if not path.is_relative_to(state_dir) or not path.is_file():
        return {}
    papers = {}
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(nodes)")}
        if not {"node_type", "raw_json", "paper_title", "summary", "pub_year", "source_venue"}.issubset(columns):
            return {}
        rows = connection.execute("SELECT * FROM nodes WHERE node_type = 'Paper'").fetchall()
        for index, row in enumerate(rows):
            paper = paper_from_graph_node(dict(row), index)
            papers[paper["id"]] = paper
            connection.execute(
                "UPDATE nodes SET paper_title = ?, summary = ?, pub_year = ?, source_venue = ? WHERE id = ?",
                (paper["title"], paper["abstract"], paper["year"], paper["venue"], row["id"]),
            )
    return papers


def materialize_context_papers(context: IntegratedSurveyContext, paper_ids: Iterable[str]) -> list[dict[str, Any]]:
    materialized: list[dict[str, Any]] = []
    for raw_paper_id in paper_ids:
        paper_id = str(raw_paper_id).strip()
        if not paper_id:
            continue
        if paper_id not in context.paper_by_id:
            context.paper_by_id[paper_id] = paper_metadata_from_context(context, paper_id)
        normalized, _stats = normalize_papers([context.paper_by_id[paper_id]], 1)
        context.paper_by_id[paper_id] = normalized[0]
        materialized.append(normalized[0])
    context.paper_ids = list(context.paper_by_id)
    return materialized


def paper_metadata_from_context(context: IntegratedSurveyContext, paper_id: str) -> dict[str, Any]:
    paper = xlab_paper_from_collector(context.work_collector, paper_id)
    if paper is not None:
        return dict(paper)

    title = ""
    abstract = ""
    authors: list[Any] = []
    year: int | None = None
    venue = ""
    url = None

    graph_metadata = reference_graph_metadata(getattr(context.work_collector, "reference_graph", None), paper_id)
    if not graph_metadata:
        graph_metadata = reference_graph_metadata(getattr(context.work_analyzer, "reference_graph", None), paper_id)
    if graph_metadata:
        title = str(graph_metadata.get("title") or graph_metadata.get("paper_title") or "")
        abstract = str(graph_metadata.get("abstract") or graph_metadata.get("summary") or "")
        authors = graph_metadata.get("authors") or []
        year = parse_year(graph_metadata.get("year") or graph_metadata.get("pub_year"))
        venue = str(graph_metadata.get("venue") or graph_metadata.get("source_venue") or "")
        url = graph_metadata.get("url")

    if not title or not abstract:
        try:
            fetched_title, fetched_abstract = context.work_collector.get_paper_title_abstract(paper_id, retry=1)
            title = title or fetched_title
            abstract = abstract or fetched_abstract
        except Exception:
            pass
    if not title:
        try:
            title = context.work_collector.get_paper_title(paper_id, retry=1)
        except Exception:
            title = paper_id
    if not abstract:
        abstract = title

    normalized_authors: list[str] = []
    if isinstance(authors, list):
        normalized_authors = [str(author.get("name") if isinstance(author, dict) else author) for author in authors]
    elif isinstance(authors, str) and authors.strip():
        normalized_authors = [authors.strip()]

    return {
        "id": paper_id,
        "title": title,
        "abstract": abstract,
        "year": year,
        "venue": venue,
        "authors": normalized_authors,
        "url": url,
        "citation_count": 0,
        "source": "survey_agent",
    }


def xlab_paper_from_collector(work_collector: WorkCollector, paper_id: str) -> dict[str, Any] | None:
    paper_by_id = getattr(work_collector, "xlab_paper_by_id", None)
    if isinstance(paper_by_id, dict) and paper_id in paper_by_id:
        return paper_by_id[paper_id]
    data_manager = getattr(work_collector, "data_manager", None)
    data_paper_by_id = getattr(data_manager, "xlab_paper_by_id", None)
    if isinstance(data_paper_by_id, dict) and paper_id in data_paper_by_id:
        return data_paper_by_id[paper_id]
    return None


def reference_graph_metadata(reference_graph: Any, paper_id: str) -> dict[str, Any]:
    if reference_graph is None:
        return {}
    if isinstance(reference_graph, dict):
        value = reference_graph.get(paper_id)
        return dict(value) if isinstance(value, dict) else {}
    nodes = getattr(reference_graph, "nodes", None)
    if nodes is not None:
        try:
            value = nodes.get(paper_id, {})
        except Exception:
            value = {}
        return dict(value) if isinstance(value, dict) else {}
    return {}


def parse_year(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(float(value.strip()))
        except ValueError:
            return None
    return None


def normalize_title(title: str) -> str:
    return " ".join(re.findall(r"[A-Za-z0-9]+", str(title).lower()))


def title_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    left_terms = set(left.split())
    right_terms = set(right.split())
    if not left_terms or not right_terms:
        return 0.0
    jaccard = len(left_terms & right_terms) / len(left_terms | right_terms)
    sequence = SequenceMatcher(None, left, right).ratio()
    return max(jaccard, sequence)
