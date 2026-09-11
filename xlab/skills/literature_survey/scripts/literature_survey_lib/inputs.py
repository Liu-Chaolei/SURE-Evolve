from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .common import read_json, resolve_project_path
from .config import RuntimeConfig, load_runtime_config


@dataclass
class SurveyRequest:
    topic: str
    input_path: Path | None = None
    graph_path: Path | None = None
    language: str = "en"
    depth: str = "standard"
    max_papers: int = 24
    min_papers: int = 3
    full_text: bool = False
    resume: bool = False
    raw_args: str = ""
    facets: list[str] = field(default_factory=list)
    compatibility_warnings: list[str] = field(default_factory=list)

    def to_json(self, cwd: Path) -> dict[str, Any]:
        return {
            "schema_version": "xlab.literature_survey.request.v1",
            "topic": self.topic,
            "input_path": self.input_path and _relative(cwd, self.input_path),
            "graph_path": self.graph_path and _relative(cwd, self.graph_path),
            "language": self.language,
            "depth": self.depth,
            "max_papers": self.max_papers,
            "min_papers": self.min_papers,
            "full_text": self.full_text,
            "resume": self.resume,
            "facets": self.facets,
            "raw_args": self.raw_args,
            "compatibility_warnings": self.compatibility_warnings,
            "source_preference": "graph" if self.graph_path else "input" if self.input_path else "missing",
        }


def _relative(cwd: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(cwd.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def normalize_key_value_tokens(tokens: list[str]) -> list[str]:
    """Accept simple key=value shorthand without weakening argparse validation."""

    normalized: list[str] = []
    for token in tokens:
        if token.startswith("--") or "=" not in token:
            normalized.append(token)
            continue
        key, value = token.split("=", 1)
        flag = "--" + key.strip().replace("_", "-")
        if flag in {"--full-text", "--resume"}:
            if value.strip().lower() in {"1", "true", "yes", "on"}:
                normalized.append(flag)
            continue
        normalized.extend([flag, value])
    return normalized


def parse_request_args(raw_args: str, cwd: Path, runtime: RuntimeConfig | None = None) -> SurveyRequest:
    runtime = runtime or load_runtime_config()
    tokens = shlex.split(raw_args)
    if tokens and tokens[0] == "--":
        tokens = tokens[1:]
    tokens = normalize_key_value_tokens(tokens)
    parser = argparse.ArgumentParser(prog="/xlab write-literature-survey", add_help=False)
    parser.add_argument("positional", nargs="*")
    parser.add_argument("--topic")
    parser.add_argument("--input")
    parser.add_argument("--papers")
    parser.add_argument("--paper-set")
    parser.add_argument("--graph")
    parser.add_argument("--language", default=runtime.default_language)
    parser.add_argument("--depth", choices=("brief", "standard", "deep"), default=runtime.default_depth)
    parser.add_argument("--max-papers", type=int, default=runtime.default_max_papers)
    parser.add_argument("--min-papers", type=int, default=runtime.default_min_papers)
    parser.add_argument("--full-text", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--facet", action="append", default=[])
    namespace, unknown = parser.parse_known_args(tokens)
    if unknown:
        raise ValueError(f"Unsupported literature_survey arguments: {' '.join(unknown)}")
    topic = namespace.topic or " ".join(namespace.positional).strip()
    input_value = namespace.input or namespace.papers or namespace.paper_set
    input_path = resolve_project_path(cwd, input_value)
    graph_path = resolve_project_path(cwd, namespace.graph)
    if not topic:
        inferred_path = graph_path if graph_path and graph_path.exists() else input_path
        if inferred_path and inferred_path.exists() and inferred_path.is_file():
            topic = infer_topic(read_json(inferred_path))
    if not topic:
        raise ValueError("Provide a topic and a knowledge graph path, or a topic and input paper manifest path.")
    max_papers = max(1, min(int(namespace.max_papers), int(runtime.max_papers_limit)))
    min_papers = max(1, int(namespace.min_papers))
    if min_papers > max_papers:
        raise ValueError("--min-papers cannot exceed --max-papers.")
    warnings = compatibility_warnings(namespace, runtime, has_graph=graph_path is not None, has_input=input_path is not None)
    return SurveyRequest(
        topic=topic,
        input_path=input_path,
        graph_path=graph_path,
        language=namespace.language,
        depth=namespace.depth,
        max_papers=max_papers,
        min_papers=min_papers,
        full_text=bool(namespace.full_text),
        resume=bool(namespace.resume),
        raw_args=raw_args,
        facets=list(namespace.facet or []),
        compatibility_warnings=warnings,
    )


def infer_topic(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("query", "topic", "title", "name"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        inputs = value.get("inputs")
        if isinstance(inputs, dict):
            for key in ("query", "topic"):
                candidate = inputs.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
    return "Literature survey"


def compatibility_warnings(namespace: argparse.Namespace, runtime: RuntimeConfig, *, has_graph: bool, has_input: bool) -> list[str]:
    warnings: list[str] = []
    if not has_graph and has_input:
        warnings.append("Using --input paper manifests is supported as a compatibility path; normal survey runs should prefer --graph.")
    if not has_graph and not has_input:
        warnings.append("No --graph or --input source was provided; normal runs cannot produce a successful real-paper survey.")
    if namespace.full_text and not runtime.full_text_enabled:
        warnings.append("--full-text was requested, but full-text synthesis is not enabled in this runtime; Survey Agent will use graph paper notes.")
    if not (runtime.llm_refinement_enabled and runtime.llm_api_key_set):
        warnings.append("SurveyAgent requires LLM refinement and OPENAI_API_KEY; the run will stay incomplete until provider access is available.")
    if namespace.facet:
        warnings.append("--facet is accepted for compatibility; scope should normally be controlled by upstream paper collection or knowledge graph construction.")
    if namespace.depth != runtime.default_depth:
        warnings.append("--depth is an advanced compatibility override; Shanghai Cloud defaults should be used for normal runs.")
    if namespace.language != runtime.default_language:
        warnings.append("--language is recorded as metadata; SurveyAgent output language follows provider behavior.")
    return warnings


def request_from_json(value: dict[str, Any], cwd: Path) -> SurveyRequest:
    return SurveyRequest(
        topic=str(value.get("topic") or "Literature survey"),
        input_path=resolve_project_path(cwd, str(value["input_path"])) if value.get("input_path") else None,
        graph_path=resolve_project_path(cwd, str(value["graph_path"])) if value.get("graph_path") else None,
        language=str(value.get("language") or "en"),
        depth=str(value.get("depth") or "standard"),
        max_papers=int(value.get("max_papers") or 24),
        min_papers=int(value.get("min_papers") or 3),
        full_text=bool(value.get("full_text")),
        resume=bool(value.get("resume")),
        raw_args=str(value.get("raw_args") or ""),
        facets=[str(facet) for facet in value.get("facets", []) if facet is not None] if isinstance(value.get("facets"), list) else [],
        compatibility_warnings=[str(warning) for warning in value.get("compatibility_warnings", []) if warning is not None]
        if isinstance(value.get("compatibility_warnings"), list)
        else [],
    )
