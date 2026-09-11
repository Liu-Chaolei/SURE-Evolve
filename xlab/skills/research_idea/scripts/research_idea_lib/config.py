from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from ipaddress import ip_address
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

from .research_idea_spec import IDEA_TASTE_MODES as SPEC_IDEA_TASTE_MODES

DEFAULT_BASE_URL = "https://api.openai.com/v1"
_SECRET_BEARING_PATH = re.compile(r"(?:^|[/_.-])(?:api[_-]?key|token|secret|password|passwd|sk-[a-z0-9])", re.IGNORECASE)
DEFAULT_AGENT_MODEL = "gpt-5-mini"
DEFAULT_FUSION_MODEL = "gpt-5.4"
DEFAULT_MAX_ITERATIONS = 64
DEFAULT_MAX_DEPTH = 3
DEFAULT_BRANCHING_FACTOR = 3
DEFAULT_EXPLORATION_CONSTANT = 1.2
# This versions persisted search/accounting semantics, not the public
# xlab.research_idea.runtime.v1 artifact contract.
RUNTIME_CONFIG_SEMANTIC_VERSION = "xlab.research_idea.runtime-config-semantics.v2"
_RUNTIME_SEMANTICS = {
    "max_iterations": DEFAULT_MAX_ITERATIONS,
    "max_depth": DEFAULT_MAX_DEPTH,
    "branching_factor": DEFAULT_BRANCHING_FACTOR,
    "exploration_constant": DEFAULT_EXPLORATION_CONSTANT,
    "max_children": None,
    "max_evaluator_calls": None,
    "max_tokens": None,
    "max_cost": None,
    "root_diagnostic_counts_as_iteration": False,
}
RUNTIME_CONFIG_SEMANTIC_SIGNATURE = hashlib.sha256(
    json.dumps(_RUNTIME_SEMANTICS, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
IDEA_TASTE_MODES = list(SPEC_IDEA_TASTE_MODES)


@dataclass(frozen=True)
class RuntimeConfig:
    """Persisted, non-secret execution settings.

    Provider readiness is deliberately computed from the current process so a run
    initialized without credentials can resume after credentials are configured.
    """

    agent_model: str
    generation_model: str
    evaluation_model: str
    fusion_model: str
    openai_base_url: str
    chat_completions_url: str
    max_iterations: int
    max_depth: int
    branching_factor: int
    request_timeout_seconds: int
    max_retries: int
    exploration_constant: float = DEFAULT_EXPLORATION_CONSTANT
    max_children: int | None = None
    max_evaluator_calls: int | None = None
    max_tokens: int | None = None
    max_cost: float | None = None
    runtime_config_semantic_version: str = RUNTIME_CONFIG_SEMANTIC_VERSION
    runtime_config_semantic_signature: str = RUNTIME_CONFIG_SEMANTIC_SIGNATURE
    idea_taste_modes: list[str] = field(default_factory=lambda: list(IDEA_TASTE_MODES))

    def __post_init__(self) -> None:
        sanitized_base_url = _validated_base_url(self.openai_base_url)
        if self.openai_base_url != sanitized_base_url:
            raise ValueError("openai_base_url must be a sanitized provider origin and path")
        if self.chat_completions_url != sanitized_base_url.rstrip("/") + "/chat/completions":
            raise ValueError("chat_completions_url must be derived from openai_base_url")
        if self.idea_taste_modes != IDEA_TASTE_MODES:
            raise ValueError("idea_taste_modes must equal the canonical modes in canonical order")
        if self.runtime_config_semantic_version != RUNTIME_CONFIG_SEMANTIC_VERSION:
            raise ValueError("runtime_config_semantic_version must match the current runtime semantics")
        if self.runtime_config_semantic_signature != RUNTIME_CONFIG_SEMANTIC_SIGNATURE:
            raise ValueError("runtime_config_semantic_signature must match the current runtime semantics")
        if self.exploration_constant < 0:
            raise ValueError("exploration_constant cannot be negative")
        if any(
            value is not None and value < 0
            for value in (self.max_children, self.max_evaluator_calls, self.max_tokens, self.max_cost)
        ):
            raise ValueError("runtime budget limits cannot be negative")

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def llm_enabled(self) -> bool:
        return _truthy("XLAB_RESEARCH_IDEA_ENABLE_LLM", default=True)

    @property
    def openai_api_key_set(self) -> bool:
        return bool(_api_key())

    @property
    def provider_available(self) -> bool:
        return self.llm_enabled and self.openai_api_key_set


def load_runtime_config() -> RuntimeConfig:
    base_url = _validated_base_url(os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    chat_url = base_url.rstrip("/") + "/chat/completions"
    return RuntimeConfig(
        agent_model=_first_env("XLAB_RESEARCH_IDEA_AGENT_MODEL", default=DEFAULT_AGENT_MODEL),
        generation_model=_first_env("XLAB_RESEARCH_IDEA_GENERATION_MODEL", default=DEFAULT_AGENT_MODEL),
        evaluation_model=_first_env("XLAB_RESEARCH_IDEA_EVALUATION_MODEL", default=DEFAULT_FUSION_MODEL),
        fusion_model=_first_env("XLAB_RESEARCH_IDEA_FUSION_MODEL", default=DEFAULT_FUSION_MODEL),
        openai_base_url=base_url,
        chat_completions_url=chat_url,
        max_iterations=DEFAULT_MAX_ITERATIONS,
        max_depth=DEFAULT_MAX_DEPTH,
        branching_factor=DEFAULT_BRANCHING_FACTOR,
        request_timeout_seconds=_bounded_int("XLAB_RESEARCH_IDEA_TIMEOUT_SECONDS", default=900, minimum=10, maximum=3600),
        max_retries=_bounded_int("XLAB_RESEARCH_IDEA_MAX_RETRIES", default=2, minimum=0, maximum=5),
    )


def runtime_config_from_json(value: dict[str, Any]) -> RuntimeConfig:
    base_url = _validated_base_url(str(value.get("openai_base_url") or DEFAULT_BASE_URL))
    defaults = RuntimeConfig(
        agent_model=DEFAULT_AGENT_MODEL,
        generation_model=DEFAULT_AGENT_MODEL,
        evaluation_model=DEFAULT_FUSION_MODEL,
        fusion_model=DEFAULT_FUSION_MODEL,
        openai_base_url=base_url,
        chat_completions_url=base_url.rstrip("/") + "/chat/completions",
        max_iterations=DEFAULT_MAX_ITERATIONS,
        max_depth=DEFAULT_MAX_DEPTH,
        branching_factor=DEFAULT_BRANCHING_FACTOR,
        request_timeout_seconds=900,
        max_retries=2,
    ).to_json()
    defaults.update({key: value[key] for key in defaults if key in value})
    return RuntimeConfig(**defaults)


def _api_key() -> str:
    return _first_env("OPENAI_API_KEY", default="")


def provider_api_key() -> str:
    """Return the provider key for immediate request use only; callers must never persist or log it."""
    return _api_key()


def _validated_base_url(raw_url: str) -> str:
    try:
        parsed = urlsplit(raw_url.strip())
        port = parsed.port
    except ValueError as error:
        raise ValueError("OPENAI_BASE_URL must be a valid provider URL.") from error
    host = parsed.hostname
    if not host or parsed.scheme not in {"https", "http"}:
        raise ValueError("OPENAI_BASE_URL must use HTTPS.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("OPENAI_BASE_URL must not contain user information.")
    if parsed.query or parsed.fragment:
        raise ValueError("OPENAI_BASE_URL must not contain a query or fragment.")
    if _SECRET_BEARING_PATH.search(unquote(parsed.path)):
        raise ValueError("OPENAI_BASE_URL path must not contain credential-like values.")
    if parsed.scheme == "http" and not _is_loopback_host(host):
        raise ValueError("OPENAI_BASE_URL must use HTTPS except for explicit loopback HTTP tests.")
    normalized_host = f"[{host}]" if ":" in host else host
    netloc = f"{normalized_host}:{port}" if port is not None else normalized_host
    path = parsed.path.rstrip("/") or "/v1"
    return urlunsplit((parsed.scheme, netloc, path, "", ""))


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


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
