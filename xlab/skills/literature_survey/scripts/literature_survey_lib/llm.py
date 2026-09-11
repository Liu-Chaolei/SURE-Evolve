from __future__ import annotations

from .config import RuntimeConfig


def llm_status(config: RuntimeConfig) -> dict[str, object]:
    enabled = config.llm_api_key_set and config.llm_refinement_enabled
    reason = None
    if not config.llm_refinement_enabled:
        reason = "LLM refinement is not enabled in this runtime; the Survey Agent path cannot run."
    elif not config.llm_api_key_set:
        reason = "OPENAI_API_KEY is not set; the Survey Agent path cannot run."
    return {
        "enabled": enabled,
        "model": config.model,
        "base_url": config.llm_base_url,
        "context_window": config.llm_context_window,
        "api_key_set": config.llm_api_key_set,
        "reason": reason,
    }
