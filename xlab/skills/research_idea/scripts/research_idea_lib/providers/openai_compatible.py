from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from .contracts import (
    JsonValue,
    ProviderExhaustedError,
    ProviderRequest,
    ProviderResult,
    ProviderTrace,
    ProviderUsage,
)


HttpResponse: TypeAlias = tuple[int, Mapping[str, str], bytes]
Transport: TypeAlias = Callable[[str, Mapping[str, str], bytes, float], HttpResponse]


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    timeout_seconds: float = 60.0
    max_attempts: int = 3
    retry_delay_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")
        if self.retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds cannot be negative.")


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str,
        config: OpenAICompatibleConfig | None = None,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key is required.")
        if not endpoint.strip():
            raise ValueError("endpoint is required.")
        self._api_key = api_key.strip()
        self._endpoint = endpoint.strip()
        self._config = config or OpenAICompatibleConfig()
        self._transport = transport or _urllib_transport
        self._sleep = sleep

    def complete(self, request: ProviderRequest) -> ProviderResult:
        body: dict[str, object] = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.output_kind == "json":
            body["response_format"] = {"type": "json_object"}
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        last_code = "provider_error"
        last_message = "Provider request failed."

        for attempt in range(1, self._config.max_attempts + 1):
            try:
                status, _, raw = self._transport(
                    self._endpoint,
                    headers,
                    encoded,
                    self._config.timeout_seconds,
                )
                if status < 200 or status >= 300:
                    last_code = f"http_{status}"
                    last_message = f"Provider request failed with HTTP {status}."
                    if status not in {408, 425, 429, 500, 502, 503, 504}:
                        break
                    raise RuntimeError(last_message)
                text, usage = _decode_completion(raw)
                json_value = extract_json_object(text) if request.output_kind == "json" else None
                return ProviderResult(
                    text=text,
                    json_value=json_value,
                    usage=usage,
                    trace=self._trace(request, attempt, "success"),
                )
            except (TimeoutError, urllib.error.URLError) as error:
                last_code = "timeout" if isinstance(error, TimeoutError) else "transport_error"
                last_message = f"Provider request failed: {last_code}."
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, KeyError, IndexError, TypeError):
                last_code = "malformed_response"
                last_message = f"Provider returned a malformed {request.output_kind} response."
            except RuntimeError:
                pass

            if attempt < self._config.max_attempts:
                self._sleep(self._config.retry_delay_seconds)

        trace = self._trace(
            request,
            min(self._config.max_attempts, attempt),
            "error",
            last_code,
        )
        raise ProviderExhaustedError(last_message, trace=trace)

    @staticmethod
    def _trace(
        request: ProviderRequest,
        attempts: int,
        status: Literal["success", "error"],
        error_code: str | None = None,
    ) -> ProviderTrace:
        return ProviderTrace(
            provider="openai_compatible",
            operation=request.operation,
            input_digest=request.input_digest,
            output_kind=request.output_kind,
            model=request.model,
            attempts=attempts,
            status=status,
            error_code=error_code,
        )


def extract_json_object(text: str) -> dict[str, JsonValue]:
    stripped = text.strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, character in enumerate(stripped):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return cast(dict[str, JsonValue], value)
        raise ValueError("No JSON object found in provider response.") from None
    if not isinstance(value, dict):
        raise ValueError("Provider response JSON must be an object.")
    return cast(dict[str, JsonValue], value)


def _decode_completion(raw: bytes) -> tuple[str, ProviderUsage]:
    payload = json.loads(raw.decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Provider returned empty content.")
    usage = payload.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    input_tokens = _token_count(usage.get("prompt_tokens"))
    output_tokens = _token_count(usage.get("completion_tokens"))
    return content, ProviderUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=_token_count(usage.get("total_tokens")) or input_tokens + output_tokens,
    )


def _token_count(value: object) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def _urllib_transport(
    endpoint: str,
    headers: Mapping[str, str],
    body: bytes,
    timeout_seconds: float,
) -> HttpResponse:
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers=dict(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - injected provider endpoint.
            return int(response.getcode()), dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers.items()), error.read()
