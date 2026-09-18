from __future__ import annotations

import json
import hashlib
import os
import http.client
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Literal, TypeAlias, cast

from .contracts import (
    JsonValue,
    ProviderExhaustedError,
    ProviderContractError,
    ProviderRequest,
    ProviderResult,
    ProviderTrace,
    ProviderUsage,
    structured_input_digest,
)
from .native_recovery import bounded_request
from .routing import routing_credentials, routing_policy_from_environment
from .route_health import RouteHealth, cooldown_seconds


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
        responses: bool | None = None,
        enable_routing: bool = True,
        http_attempts: int | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key is required.")
        if not endpoint.strip():
            raise ValueError("endpoint is required.")
        self._api_key = api_key.strip()
        self._endpoint = endpoint.strip()
        self._responses = responses if responses is not None else os.environ.get('XLAB_SURE_RESPONSES') == '1'
        self._routing_policy = routing_policy_from_environment() if enable_routing else None
        if http_attempts is not None and (type(http_attempts) is not int or http_attempts < 1):
            raise ValueError('http_attempts must be a positive integer')
        self._http_attempts = http_attempts
        if self._responses:
            if not self._endpoint.endswith('/chat/completions'):
                raise ValueError('Responses transport requires a chat-completions base endpoint')
            self._endpoint = self._endpoint[:-len('/chat/completions')] + '/responses'
        self._config = config or OpenAICompatibleConfig()
        self._transport = transport or _urllib_transport
        self._sleep = sleep
        cache_root = os.environ.get("XLAB_SURE_PROVIDER_CACHE_ROOT")
        self._cache_root = Path(cache_root) if cache_root else None
        self._cache_counts: dict[str, int] = {}
        self._cache_lock = Lock()
        self._replay_files = set(self._cache_root.glob('*.json')) if self._cache_root and self._cache_root.exists() else set()
        health_root = os.environ.get('XLAB_API_ROUTING_STATE_DIR')
        self._health = RouteHealth(Path(health_root)) if health_root and enable_routing else None
        self._last_http_error = None

    def complete(self, request: ProviderRequest) -> ProviderResult:
        if self._cache_root is None:
            return self._complete_uncached(request)
        identity = hashlib.sha256(json.dumps(
            {"endpoint": self._endpoint, "request": request.cache_payload(),
             "wire_policy": self._wire_policy()}, sort_keys=True,
            ensure_ascii=False).encode()).hexdigest()
        with self._cache_lock:
            index = self._cache_counts.get(identity, 0)
            self._cache_counts[identity] = index + 1
        path = self._cache_root / f'{identity}-{index}.json'
        self._cache_root.mkdir(parents=True, exist_ok=True)
        progress = {'operation':request.operation,'request_digest':identity,'status':'running',
                    'started_at':time.time(),'requested_model':request.model}
        pending_progress = self._cache_root.parent / (self._cache_root.name + '.progress.pending')
        pending_progress.write_text(json.dumps(progress)+'\n')
        pending_progress.replace(pending_progress.with_suffix('.json'))
        if path in self._replay_files:
            document = json.loads(path.read_text())
            if document['request_digest'] != identity:
                raise ValueError('Provider replay identity mismatch')
            value = document['result']
            result = ProviderResult(text=value['text'], json_value=value['json_value'],
                                    usage=ProviderUsage(**value['usage']), trace=ProviderTrace(**value['trace']))
            if request.response_validator is not None:
                try:
                    request.response_validator(result)
                except (ValueError, TypeError) as error:
                    rejected = path.with_name(path.stem + '.rejected-cache.json')
                    if not rejected.exists():
                        rejected.write_bytes(path.read_bytes())
                    repair_digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    repaired_request = replace(request, validation_profile=request.validation_profile + ':repair:' + repair_digest)
                    return self.complete(repaired_request)
            with (self._cache_root/'replays.jsonl').open('a') as log:
                log.write(json.dumps({'operation': request.operation, 'cache_file': path.name, 'time': time.time()})+'\n')
            progress.update(status='replayed',finished_at=time.time(),actual_model=result.trace.model)
            pending_progress.write_text(json.dumps(progress)+'\n')
            pending_progress.replace(pending_progress.with_suffix('.json'))
            return result
        if os.environ.get('XLAB_NATIVE_RECOVERY') == '1':
            result = bounded_request(path.with_suffix('.operation.json'), identity,
                lambda deadline: self._complete_uncached(request, deadline=deadline), sleep=self._sleep)
        else:
            result = self._complete_uncached(request)
        self._cache_root.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix('.pending')
        fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as handle:
            json.dump({'request_digest': identity, 'result': asdict(result)}, handle, ensure_ascii=False)
        pending.replace(path)
        progress.update(status='completed',finished_at=time.time(),actual_model=result.trace.model)
        pending_progress.write_text(json.dumps(progress)+'\n')
        pending_progress.replace(pending_progress.with_suffix('.json'))
        return result

    def _wire_policy(self) -> dict:
        policy = {"max_output_tokens": os.environ.get("XLAB_SURE_MAX_OUTPUT_TOKENS"),
                "reasoning_effort": os.environ.get("XLAB_SURE_REASONING_EFFORT"),
                "responses": self._responses,
                "stream": os.environ.get("XLAB_RESEARCH_IDEA_STREAM") == "1"}
        if self._routing_policy is not None:
            policy["api_routing"] = self._routing_policy
        return policy

    def _complete_routed(self, request: ProviderRequest, *, deadline: float | None) -> ProviderResult:
        policy = self._routing_policy
        credentials = routing_credentials(policy)
        deadline = deadline or time.time() + self._config.timeout_seconds
        attempts = []
        last_error = None
        last_contract_error = None
        last_invalid_result = None
        last_validation_message = ""
        usage = ProviderUsage()
        deadline_expired = False
        retry_times = []
        for route in policy["routes"]:
            slot = route["slot"]
            if not credentials.get(slot) or not route["base_url"]:
                attempts.append({"slot": slot, "status": "unconfigured", "attempts": 0})
                continue
            if time.time() >= deadline:
                deadline_expired = True
                break
            if self._health:
                retry_at = self._health.acquire(slot, route['base_url'], lease_seconds=max(1, deadline-time.time()))
                if retry_at:
                    retry_times.append(retry_at)
                    attempts.append({'slot': slot, 'status': 'cooldown', 'attempts': 0, 'retry_at': retry_at})
                    self._record_route(request, attempts[-1])
                    continue
            client = OpenAICompatibleProvider(
                api_key=credentials[slot], endpoint=route["base_url"] + "/chat/completions",
                config=self._config, transport=self._transport, sleep=self._sleep,
                responses=route["responses"], enable_routing=False, http_attempts=1,
            )
            client._cache_root = self._cache_root / "routes" / slot if self._cache_root else None
            actual_request = replace(request, model=route["model"])
            started_at = time.time()
            try:
                result = client._complete_uncached(actual_request, deadline=deadline)
            except ProviderExhaustedError as error:
                last_error = error
                http = client._last_http_error
                if self._health:
                    code = error.trace.error_code or ''
                    transient = code in {'timeout', 'transport_error'} or code.startswith('http_5')
                    delay = cooldown_seconds(*http, time.time()) if http and (http[0] in (401,403,429) or http[0]>=500) else (15 if transient else 0)
                    self._health.finish(slot, route['base_url'], delay=delay)
                    if delay:
                        retry_times.append(time.time()+delay)
                attempts.append({"slot": slot, "model": route["model"], "status": "error",
                                 "error_code": error.trace.error_code, "attempts": error.trace.attempts,
                                 "seconds": time.time()-started_at})
                self._record_route(request, attempts[-1])
                continue
            if self._health:
                self._health.finish(slot, route['base_url'])
            usage = ProviderUsage(usage.input_tokens + result.usage.input_tokens,
                                  usage.output_tokens + result.usage.output_tokens,
                                  usage.total_tokens + result.usage.total_tokens)
            if request.response_validator is not None:
                try:
                    request.response_validator(result)
                except (ValueError, TypeError) as error:
                    last_contract_error, last_invalid_result = error, result
                    last_validation_message = str(error)
                    for key in credentials.values():
                        last_validation_message = last_validation_message.replace(key, "[REDACTED]")
                    attempt = {"slot": slot, "model": route["model"], "status": "error",
                               "error_code": "invalid_response_contract", "attempts": result.trace.attempts,
                               "seconds": time.time()-started_at,
                               "validation_error": last_validation_message[:500], "usage": asdict(result.usage)}
                    if self._cache_root is not None:
                        rejected = self._cache_root / "routes" / slot / f"rejected-{time.time_ns()}.json"
                        rejected.parent.mkdir(parents=True, exist_ok=True)
                        raw = json.dumps({"validation_error": str(error), "result": asdict(result)}, ensure_ascii=False)
                        for key in credentials.values():
                            raw = raw.replace(key, "[REDACTED]")
                        with os.fdopen(os.open(rejected, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
                            handle.write(raw + "\n")
                        attempt["raw_response_file"] = str(rejected.relative_to(self._cache_root))
                    attempts.append(attempt)
                    self._record_route(request, attempt)
                    continue
            attempts.append({"slot": slot, "model": route["model"], "status": "success",
                             "attempts": result.trace.attempts, "usage": asdict(result.usage),
                             "seconds": time.time()-started_at})
            self._record_route(request, attempts[-1])
            return replace(result, usage=usage, trace=replace(result.trace,
                attempts=sum(item["attempts"] for item in attempts),
                routing={"policy_digest": structured_input_digest(policy),
                         "requested_model": request.model, "selected_slot": slot,
                         "base_url": route["base_url"], "validation_profile": request.validation_profile,
                         "attempts": attempts}))
        if last_contract_error is not None:
            trace = self._trace(request, sum(item["attempts"] for item in attempts), "error", "invalid_response_contract")
            trace = replace(trace, model=last_invalid_result.trace.model, routing={
                "policy_digest": structured_input_digest(policy), "requested_model": request.model,
                "selected_slot": None, "validation_profile": request.validation_profile, "attempts": attempts})
            feedback = json.dumps({
                "previous_draft": getattr(last_contract_error, "previous_draft", None) or last_invalid_result.json_value,
                "validation_issues": getattr(last_contract_error, "validation_issues", [])}, ensure_ascii=False)
            for key in credentials.values():
                feedback = feedback.replace(key, "[REDACTED]")
            raise ProviderContractError(last_validation_message, trace=trace, **json.loads(feedback)) from None
        code = "timeout" if deadline_expired else (last_error.trace.error_code if last_error else ("http_429" if retry_times else "no_configured_route"))
        transient = {"timeout", "transport_error", "http_408", "http_425", "http_429",
                     "http_500", "http_502", "http_503", "http_504"}
        for item in reversed(attempts):
            if not deadline_expired and item.get("error_code") in transient:
                code = item["error_code"]
                break
        trace = self._trace(request, sum(item["attempts"] for item in attempts), "error", code)
        trace = replace(trace, model=last_error.trace.model if last_error else request.model,
            routing={"policy_digest": structured_input_digest(policy),
            "requested_model": request.model, "selected_slot": None, "attempts": attempts})
        error = ProviderExhaustedError("All configured API routes failed: " + ", ".join(
            item["slot"] + "=" + str(item.get("error_code", item["status"])) for item in attempts), trace=trace)
        error.retry_at = min(retry_times) if retry_times else None
        raise error

    def _record_route(self, request: ProviderRequest, record: dict) -> None:
        if self._cache_root is not None:
            self._cache_root.mkdir(parents=True, exist_ok=True)
            with (self._cache_root / "routing_attempts.jsonl").open("a") as handle:
                handle.write(json.dumps({"time": time.time(), "operation": request.operation,
                    "input_digest": request.input_digest, "requested_model": request.model, **record}) + "\n")

    def _complete_uncached(self, request: ProviderRequest, *, deadline: float | None = None) -> ProviderResult:
        if self._routing_policy is not None:
            return self._complete_routed(request, deadline=deadline)
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
        if os.environ.get("XLAB_RESEARCH_IDEA_STREAM") == "1":
            body.update(stream=True, stream_options={"include_usage": True})
        if self._responses:
            body = {'model': request.model, 'input': body['messages'], 'stream': True,
                    **({'text': {'format': {'type': 'json_object'}}} if request.output_kind == 'json' else {})}
            if request.temperature is not None:
                body['temperature'] = request.temperature
        policy = self._wire_policy()
        if os.environ.get('XLAB_NATIVE_RECOVERY') == '1' and request.output_kind == 'json':
            messages = body['input'] if self._responses else body['messages']
            messages[0]['content'] = 'Respond with a JSON object.\n' + messages[0]['content']
        if policy["max_output_tokens"]:
            body["max_output_tokens" if self._responses else "max_tokens"] = int(policy["max_output_tokens"])
        if policy["reasoning_effort"]:
            if self._responses:
                body["reasoning"] = {"effort": policy["reasoning_effort"]}
            else:
                body["reasoning_effort"] = policy["reasoning_effort"]
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": "XLab/1.0",
        }
        last_code = "provider_error"
        last_message = "Provider request failed."

        attempts = self._http_attempts or max(self._config.max_attempts, int(os.environ.get("XLAB_SURE_HTTP_ATTEMPTS", "1")))
        if os.environ.get('XLAB_NATIVE_RECOVERY') == '1':
            attempts = min(3, attempts)
        for attempt in range(1, attempts + 1):
            raw = b""
            try:
                remaining = deadline - time.time() if deadline else self._config.timeout_seconds
                if remaining <= 0:
                    last_code, last_message = 'timeout', 'Native request recovery deadline exhausted'
                    break
                status, response_headers, raw = self._transport(
                    self._endpoint,
                    headers,
                    encoded,
                    min(self._config.timeout_seconds, remaining),
                )
                if status < 200 or status >= 300:
                    self._last_http_error = (status, dict(response_headers), raw)
                    if self._cache_root:
                        self._cache_root.mkdir(parents=True, exist_ok=True)
                        try:
                            detail = json.loads(raw).get('error', {})
                            detail = detail if isinstance(detail, dict) else {}
                        except (ValueError, TypeError, AttributeError):
                            detail = {}
                        message = str(detail.get('message','')).replace(self._api_key,'[REDACTED]')[:500]
                        category = 'context_limit' if any(t in message.lower() for t in ('context length','context window','too many tokens')) else 'request_error'
                        if any(t in message.lower() for t in ('unsupported model','model_not_found','invalid model')):
                            category = 'unsupported_model'
                        report = {'status':status,'provider_error_code':str(detail.get('code',''))[:80],
                                  'category':category,'message':message,'operation':request.operation}
                        (self._cache_root / f'http-error-{time.time_ns()}.json').write_text(json.dumps(report)+'\n')
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
            except (TimeoutError, urllib.error.URLError, ConnectionError, http.client.HTTPException) as error:
                last_code = "timeout" if isinstance(error, TimeoutError) else "transport_error"
                last_message = f"Provider request failed: {last_code} ({type(error).__name__})."
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, KeyError, IndexError, TypeError) as error:
                last_code = "malformed_response"
                last_message = f"Malformed {request.output_kind}: {type(error).__name__}: {str(error)[:200]}".replace(self._api_key, "[REDACTED]")
                if self._cache_root is not None:
                    self._cache_root.mkdir(parents=True, exist_ok=True)
                    path = self._cache_root / f"malformed-{time.time_ns()}.txt"
                    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
                        handle.write(raw.decode(errors="replace").replace(self._api_key, "[REDACTED]"))
                if os.environ.get('XLAB_NATIVE_RECOVERY') == '1':
                    break
            except RuntimeError:
                pass

            if self._cache_root is not None:
                self._cache_root.mkdir(parents=True, exist_ok=True)
                with (self._cache_root / "transport_attempts.jsonl").open("a") as log:
                    log.write(json.dumps({"time": time.time(), "attempt": attempt,
                        "operation": request.operation, "error_code": last_code, "error": last_message}) + "\n")
            if attempt < attempts:
                delay = min(60, 15 * 2 ** (attempt - 1)) if os.environ.get("XLAB_SURE_HTTP_ATTEMPTS") else min(8, 2 ** attempt)
                delay = max(self._config.retry_delay_seconds, delay)
                self._sleep(min(delay, max(0, deadline-time.time())) if deadline else delay)

        trace = self._trace(
            request,
            attempt,
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
    text = raw.decode("utf-8")
    if text.lstrip().startswith(("data:", ":", "event:")):
        content_parts = []
        usage = {}
        complete = False
        for line in text.splitlines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                complete = True
                break
            if not data:
                continue
            chunk = json.loads(data)
            if chunk.get('type') == 'response.completed':
                return _decode_completion(json.dumps(chunk['response']).encode('utf-8'))
            if chunk.get('type') in {'response.failed', 'response.incomplete', 'error'}:
                _raise_response_failure(chunk.get('response', chunk))
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                if choice.get("index", 0) == 0:
                    part = (choice.get("delta") or {}).get("content")
                    if isinstance(part, str):
                        content_parts.append(part)
        if not complete:
            raise ValueError("Provider stream ended before DONE")
        payload = {"choices": [{"message": {"content": "".join(content_parts)}}], "usage": usage}
    else:
        payload = json.loads(text)
    if payload.get('object') == 'response':
        if payload.get('status') != 'completed':
            _raise_response_failure(payload)
        parts = [part['text'] for item in payload.get('output', [])
                 if item.get('type') == 'message' and item.get('role') == 'assistant'
                 for part in item.get('content', []) if part.get('type') == 'output_text']
        usage = payload.get('usage') or {}
        payload = {'choices': [{'message': {'content': ''.join(parts)}}],
                   'usage': {'prompt_tokens': usage.get('input_tokens', 0),
                             'completion_tokens': usage.get('output_tokens', 0),
                             'total_tokens': usage.get('total_tokens', 0)}}
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


def _raise_response_failure(payload: dict) -> None:
    error = payload.get('error') or {}
    code = error.get('code') if isinstance(error, dict) else None
    if payload.get('type') == 'error':
        code = code or payload.get('code')
    # These terminal events confirm a failed upstream operation. Retrying is
    # different from resending an unknown in-flight call or a truncated result.
    if code in {'server_error', 'overloaded', 'rate_limit_exceeded'}:
        raise ConnectionError('Retryable upstream response failure: ' + code)
    raise ValueError('Responses generation did not complete successfully')


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
            if json.loads(body).get("stream") and "text/event-stream" in response.headers.get("Content-Type", ""):
                chunks = []
                end = time.monotonic() + timeout_seconds
                while True:
                    remaining = end - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('Native response exceeded its request deadline')
                    raw_stream = getattr(getattr(response, 'fp', None), 'raw', None)
                    sock = getattr(raw_stream, '_sock', None)
                    if sock is not None:
                        sock.settimeout(remaining)
                    line = response.readline()
                    if not line:
                        raise ValueError('Native stream ended before a terminal event')
                    chunks.append(line)
                    data = line.strip()
                    if data == b"data: [DONE]":
                        break
                    if data.startswith(b"data:"):
                        try:
                            event = json.loads(data[5:])
                        except ValueError:
                            continue
                        if event.get("type") in {"response.completed", "response.failed", "response.incomplete"}:
                            break
                raw = b"".join(chunks)
            else:
                raw = response.read()
            return int(response.getcode()), dict(response.headers.items()), raw
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers.items()), error.read()
