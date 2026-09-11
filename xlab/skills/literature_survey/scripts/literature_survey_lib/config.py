from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

DEFAULT_LLM_BASE_URL = "https://api.minimaxi.com/v1"
DEFAULT_LLM_MODEL = "MiniMax-M3"


@dataclass(frozen=True)
class RuntimeConfig:
    model: str
    llm_base_url: str
    llm_api_key_set: bool
    semantic_scholar_api_key_set: bool
    llm_context_window: int
    max_workers: int
    request_timeout_seconds: int
    max_retries: int
    default_max_papers: int
    max_papers_limit: int
    default_min_papers: int
    default_language: str
    default_depth: str
    cluster_limit: int
    inline_citation_limit: int
    evidence_chars: int
    graph_context_limit: int
    allow_synthetic: bool
    llm_refinement_enabled: bool
    full_text_enabled: bool

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def load_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        model=_first_env("LLM_MODEL", "XLAB_LITERATURE_SURVEY_MODEL", default=DEFAULT_LLM_MODEL),
        llm_base_url=_first_env("LLM_BASE_URL", "LLM_API_BASE", "XLAB_LITERATURE_SURVEY_LLM_BASE_URL", default=DEFAULT_LLM_BASE_URL),
        llm_api_key_set=bool(_first_env("OPENAI_API_KEY", "LLM_API_KEY", default="")),
        semantic_scholar_api_key_set=bool(_first_env("SEMANTIC_SCHOLAR_API_KEY", "S2_API_KEY", default="")),
        llm_context_window=_bounded_int("LLM_CONTEXT_WINDOW", default=512_000, minimum=8_000, maximum=1_000_000),
        max_workers=_bounded_int("XLAB_LITERATURE_SURVEY_MAX_WORKERS", default=4, minimum=1, maximum=16),
        request_timeout_seconds=_bounded_int("XLAB_LITERATURE_SURVEY_TIMEOUT_SECONDS", default=300, minimum=10, maximum=600),
        max_retries=_bounded_int("XLAB_LITERATURE_SURVEY_MAX_RETRIES", default=2, minimum=0, maximum=5),
        default_max_papers=_bounded_int("XLAB_LITERATURE_SURVEY_DEFAULT_MAX_PAPERS", default=24, minimum=1, maximum=200),
        max_papers_limit=_bounded_int("XLAB_LITERATURE_SURVEY_MAX_PAPERS_LIMIT", default=200, minimum=3, maximum=500),
        default_min_papers=_bounded_int("XLAB_LITERATURE_SURVEY_DEFAULT_MIN_PAPERS", default=3, minimum=1, maximum=50),
        default_language=os.environ.get("XLAB_LITERATURE_SURVEY_DEFAULT_LANGUAGE", "en").strip() or "en",
        default_depth=_choice("XLAB_LITERATURE_SURVEY_DEFAULT_DEPTH", default="standard", choices={"brief", "standard", "deep"}),
        cluster_limit=_bounded_int("XLAB_LITERATURE_SURVEY_CLUSTER_LIMIT", default=5, minimum=1, maximum=20),
        inline_citation_limit=_bounded_int("XLAB_LITERATURE_SURVEY_INLINE_CITATION_LIMIT", default=8, minimum=1, maximum=50),
        evidence_chars=_bounded_int("XLAB_LITERATURE_SURVEY_EVIDENCE_CHARS", default=500, minimum=80, maximum=5000),
        graph_context_limit=_bounded_int("XLAB_LITERATURE_SURVEY_GRAPH_CONTEXT_LIMIT", default=24, minimum=1, maximum=200),
        allow_synthetic=_truthy("XLAB_LITERATURE_SURVEY_ALLOW_SYNTHETIC", default=False),
        llm_refinement_enabled=_truthy("XLAB_LITERATURE_SURVEY_ENABLE_LLM_REFINEMENT", default=True),
        full_text_enabled=_truthy("XLAB_LITERATURE_SURVEY_ENABLE_FULL_TEXT", default=False),
    )


def runtime_config_from_json(value: dict[str, Any]) -> RuntimeConfig:
    defaults = load_runtime_config().to_json()
    if "openai_api_key_set" in value and "llm_api_key_set" not in value:
        value = dict(value)
        value["llm_api_key_set"] = value["openai_api_key_set"]
    defaults.update({key: value[key] for key in defaults if key in value})
    return RuntimeConfig(**defaults)


def _first_env(*names: str, default: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


def _bounded_int(name: str, *, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(value, maximum))


def _truthy(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _choice(name: str, *, default: str, choices: set[str]) -> str:
    raw = os.environ.get(name, "").strip().lower()
    return raw if raw in choices else default
