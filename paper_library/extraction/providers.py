from __future__ import annotations

from typing import Protocol, TypeVar, cast

from pydantic import BaseModel

from paper_library.config import ExtractionConfig
from paper_library.errors import ErrorCode, PaperLibraryError

T = TypeVar("T", bound=BaseModel)


class StructuredExtractionProvider(Protocol):
    name: str
    model: str

    def extract(self, *, prompt: str, response_model: type[T]) -> T:
        """Return a schema-validated candidate without executing source content."""


class DisabledProvider:
    name = "disabled"
    model = "disabled"

    def extract(self, *, prompt: str, response_model: type[T]) -> T:
        del prompt, response_model
        raise PaperLibraryError(
            "LLM extraction is disabled in this configuration",
            code=ErrorCode.EXTRACTION_FAILED,
        )


class OpenAIProvider:
    name = "openai"

    def __init__(self, config: ExtractionConfig) -> None:
        from openai import OpenAI

        self.model = config.model or ""
        self.temperature = config.temperature
        self.max_retries = config.max_retries
        self.client = OpenAI(
            api_key=config.api_key.get_secret_value() if config.api_key else None,
            base_url=config.base_url,
            max_retries=config.max_retries,
        )

    def extract(self, *, prompt: str, response_model: type[T]) -> T:
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format=response_model,
            )
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                raise ValueError("provider returned no structured response")
            return cast(T, parsed)
        except PaperLibraryError:
            raise
        except Exception as exc:
            raise PaperLibraryError(
                f"OpenAI extraction failed: {type(exc).__name__}: {exc}",
                code=ErrorCode.EXTRACTION_FAILED,
                retryable=_is_retryable(exc),
            ) from exc


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, config: ExtractionConfig) -> None:
        from anthropic import Anthropic

        self.model = config.model or ""
        self.temperature = config.temperature
        self.client = Anthropic(
            api_key=config.api_key.get_secret_value() if config.api_key else None,
            base_url=config.base_url,
            max_retries=config.max_retries,
        )

    def extract(self, *, prompt: str, response_model: type[T]) -> T:
        tool_name = "return_structured_extraction"
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
                tools=[
                    {
                        "name": tool_name,
                        "description": "Return the requested schema-validated extraction.",
                        "input_schema": response_model.model_json_schema(),
                    }
                ],
                tool_choice={"type": "tool", "name": tool_name},
            )
            for block in message.content:
                if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool_name:
                    return response_model.model_validate(block.input)
            raise ValueError("provider returned no structured tool result")
        except PaperLibraryError:
            raise
        except Exception as exc:
            raise PaperLibraryError(
                f"Anthropic extraction failed: {type(exc).__name__}: {exc}",
                code=ErrorCode.EXTRACTION_FAILED,
                retryable=_is_retryable(exc),
            ) from exc


def create_provider(config: ExtractionConfig) -> StructuredExtractionProvider:
    if config.provider == "disabled":
        return DisabledProvider()
    if config.api_key is None:
        raise PaperLibraryError(
            f"An API key is required for extraction provider {config.provider!r}",
            code=ErrorCode.CONFIG_INVALID,
        )
    if config.provider == "openai":
        return OpenAIProvider(config)
    if config.provider == "anthropic":
        return AnthropicProvider(config)
    raise PaperLibraryError(
        f"Unsupported extraction provider: {config.provider}",
        code=ErrorCode.CONFIG_INVALID,
    )


def _is_retryable(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code == 429 or status_code >= 500
    name = type(exc).__name__.casefold()
    return any(token in name for token in ("timeout", "connection", "ratelimit"))
