from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from .common import JsonObject, append_jsonl, as_list, as_mapping, text, utc_now


CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class LlmServiceUnavailable(RuntimeError):
    """The provider is unavailable; preserve queued papers for a later resume."""


def completion_url(api_url: str) -> str:
    parsed = urlsplit(api_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        endpoint = path
    elif path.endswith("/v1"):
        endpoint = path + "/chat/completions"
    elif not path:
        endpoint = "/v1/chat/completions"
    else:
        endpoint = path + "/chat/completions"
    return urlunsplit((parsed.scheme, parsed.netloc, endpoint, "", ""))


def extract_json_object(content: str) -> JsonObject:
    stripped = CODE_FENCE.sub("", content.strip()).strip()
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return as_mapping(value)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return as_mapping(value)
    raise ValueError("LLM response does not contain a JSON object")


def _retry_delay(headers: object, attempt: int) -> float:
    retry_after = None
    if hasattr(headers, "get"):
        retry_after = headers.get("Retry-After")
    if retry_after:
        try:
            return min(max(float(retry_after), 0.0), 60.0)
        except ValueError:
            try:
                when = parsedate_to_datetime(retry_after).timestamp()
                return min(max(when - time.time(), 0.0), 60.0)
            except (TypeError, ValueError):
                pass
    return min((2**attempt) + random.random(), 20.0)


class LlmClient:
    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        model: str,
        timeout: int,
        retries: int,
        requests_per_minute: int,
        log_path: Path,
        enable_thinking: bool | None = None,
    ) -> None:
        self.endpoint = completion_url(api_url)
        self.api_key = api_key
        self.model = model
        self.enable_thinking = enable_thinking
        self.timeout = timeout
        self.retries = retries
        self.minimum_interval = (
            60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        )
        self.log_path = log_path
        self._rate_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._last_request = 0.0
        self._unavailable = threading.Event()

    def _redact(self, value: object) -> str:
        result = text(value)
        return result.replace(self.api_key, "<redacted>") if self.api_key else result

    def _wait_for_rate_limit(self) -> None:
        if self.minimum_interval <= 0:
            return
        with self._rate_lock:
            wait = self.minimum_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

    def _log(self, value: JsonObject) -> None:
        with self._log_lock:
            append_jsonl(self.log_path, value)

    def chat(
        self,
        *,
        paper_id: str,
        pass_name: str,
        system: str,
        user: str,
        max_tokens: int,
    ) -> tuple[JsonObject, JsonObject]:
        if self._unavailable.is_set():
            raise LlmServiceUnavailable("LLM service is paused after exhausted transport retries.")
        request_hash = hashlib.sha256(
            (self.model + "\x1f" + system + "\x1f" + user).encode("utf-8")
        ).hexdigest()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error = "unknown LLM error"
        service_failure = False
        for attempt in range(self.retries + 1):
            if self._unavailable.is_set():
                raise LlmServiceUnavailable("LLM service is paused after exhausted transport retries.")
            self._wait_for_rate_limit()
            started = time.monotonic()
            request = Request(
                self.endpoint,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "api-key": self.api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "xlab-knowledge-graph/3",
                },
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8", errors="replace")
                    status = response.status
                response_value = as_mapping(json.loads(raw))
                choices = [as_mapping(value) for value in as_list(response_value.get("choices"))]
                if not choices:
                    raise ValueError("LLM response has no choices")
                message = as_mapping(choices[0].get("message"))
                content_value = message.get("content")
                if isinstance(content_value, list):
                    content = "\n".join(
                        text(as_mapping(item).get("text"))
                        for item in content_value
                        if text(as_mapping(item).get("text"))
                    )
                else:
                    content = text(content_value)
                if not content:
                    raise ValueError("LLM response content is empty")
                result = extract_json_object(content)
                metadata: JsonObject = {
                    "model": response_value.get("model") or self.model,
                    "response_id": response_value.get("id"),
                    "usage": as_mapping(response_value.get("usage")),
                    "finish_reason": choices[0].get("finish_reason"),
                    "request_sha256": request_hash,
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "attempt": attempt + 1,
                }
                self._log(
                    {
                        "recorded_at": utc_now(),
                        "paper_id": paper_id,
                        "pass": pass_name,
                        "status": status,
                        **metadata,
                    }
                )
                return result, metadata
            except HTTPError as error:
                service_failure = error.code in {408, 429} or error.code >= 500
                detail = error.read().decode("utf-8", errors="replace")[:2000]
                last_error = self._redact(f"HTTP {error.code}: {detail}")
                lowered = detail.lower()
                payload_changed = False
                if (
                    error.code == 400
                    and "max_tokens" in lowered
                    and "max_tokens" in payload
                ):
                    payload["max_completion_tokens"] = payload.pop("max_tokens")
                    payload_changed = True
                elif (
                    error.code == 400
                    and "temperature" in lowered
                    and "temperature" in payload
                ):
                    payload.pop("temperature")
                    payload_changed = True
                if payload_changed:
                    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                retryable = (
                    payload_changed
                    or error.code in {408, 409, 425, 429}
                    or error.code >= 500
                )
                self._log(
                    {
                        "recorded_at": utc_now(),
                        "paper_id": paper_id,
                        "pass": pass_name,
                        "status": error.code,
                        "attempt": attempt + 1,
                        "request_sha256": request_hash,
                        "error": last_error,
                        "retryable": retryable,
                    }
                )
                if not retryable or attempt >= self.retries:
                    break
                if not payload_changed:
                    time.sleep(_retry_delay(error.headers, attempt))
            except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as error:
                service_failure = isinstance(error, (URLError, TimeoutError, OSError))
                last_error = self._redact(error)
                self._log(
                    {
                        "recorded_at": utc_now(),
                        "paper_id": paper_id,
                        "pass": pass_name,
                        "status": "error",
                        "attempt": attempt + 1,
                        "request_sha256": request_hash,
                        "error": last_error[:2000],
                        "retryable": attempt < self.retries,
                    }
                )
                if attempt >= self.retries:
                    break
                time.sleep(_retry_delay({}, attempt))
        if service_failure:
            self._unavailable.set()
            raise LlmServiceUnavailable(f"LLM service unavailable after {self.retries + 1} attempts: {last_error}")
        raise RuntimeError(
            f"LLM {pass_name} failed after {self.retries + 1} attempts: {last_error}"
        )
