from __future__ import annotations

from .config import RuntimeConfig


def llm_status(config: RuntimeConfig) -> dict[str, object]:
    reason = None
    if not config.llm_enabled:
        reason = "LLM generation is disabled in this runtime."
    elif not config.openai_api_key_set:
        reason = "OPENAI_API_KEY is unavailable in the runtime environment; package-native research idea generation will remain incomplete. Configure it as an environment secret, never as an argument."
    return {
        "enabled": config.provider_available,
        "agent_model": config.agent_model,
        "generation_model": config.generation_model,
        "evaluation_model": config.evaluation_model,
        "fusion_model": config.fusion_model,
        "base_url": config.openai_base_url,
        "api_key_set": config.openai_api_key_set,
        "reason": reason,
    }
