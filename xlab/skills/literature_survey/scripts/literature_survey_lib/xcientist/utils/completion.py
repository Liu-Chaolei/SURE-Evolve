"""Read complete model responses and preserve request diagnostics without credentials."""
import hashlib
import json
import time
import uuid
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import requests
from .completion_cache import find_completed_response

_sessions = threading.local()
_request_start_lock = threading.Lock()
_last_request_start = 0.0


def complete_chat(url, headers, payload, timeout, run_dir, stream_timeout=None):
    global _last_request_start
    if os.environ.get("XLAB_LITERATURE_SURVEY_REUSE_COMPLETIONS") == "1":
        cached = find_completed_response(run_dir, url, payload)
        if cached is not None:
            restored = requests.Response()
            restored.status_code = 200
            restored.url = url
            restored._content = json.dumps({"choices": [{"message": {"content": cached["content"]}, "finish_reason": "stop"}], "usage": cached.get("usage", {})}).encode()
            restored.headers["X-XLab-Completion-Cache"] = "hit"
            return cached["content"], restored
    started = time.monotonic()
    content = []
    reasoning_chars = 0
    finish_reason = None
    usage = {}
    response = None
    error = None
    done = False
    try:
        effective_timeout = (min(30, timeout), stream_timeout or timeout) if payload["stream"] else timeout
        if not hasattr(_sessions, "session"):
            _sessions.session = requests.Session()
        interval = float(os.environ.get("XLAB_LITERATURE_SURVEY_REQUEST_START_INTERVAL_SECONDS", "0"))
        if not 0 <= interval <= 10:
            raise ValueError("Request start interval must be between 0 and 10 seconds")
        if interval:
            with _request_start_lock:
                delay = _last_request_start + interval - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                _last_request_start = time.monotonic()
        response = _sessions.session.post(url, headers=headers, json=payload, timeout=effective_timeout, stream=payload["stream"])
        response.raise_for_status()
        if payload["stream"]:
            for line in response.iter_lines():
                if not line or not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    done = True
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise requests.ConnectionError("Malformed JSON in completion stream") from exc
                if chunk.get("error"):
                    raise requests.ConnectionError("Provider error inside completion stream")
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices", []):
                    if choice.get("index", 0) != 0:
                        continue
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    content.append(delta.get("content") or "")
                    reasoning_chars += len(delta.get("reasoning_content") or "")
            if not done and not finish_reason:
                raise requests.ConnectionError("Completion stream ended without a terminal event")
        else:
            value = response.json()
            usage = value.get("usage") or {}
            choices = value.get("choices") or []
            if not choices:
                raise requests.ConnectionError("Completion response has no choices")
            finish_reason = choices[0].get("finish_reason")
            message = choices[0].get("message") or {}
            content.append(message.get("content") or "")
            reasoning_chars = len(message.get("reasoning_content") or "")
        if finish_reason == "length":
            raise ValueError("Model output truncated at token limit; reduce task size or increase output budget")
        if finish_reason not in {None, "stop"}:
            raise ValueError(f"Model did not finish normal text generation: {finish_reason}")
        result = "".join(content)
        if not result.strip():
            raise requests.ConnectionError("Model returned no final content")
        return result, response
    except Exception as exc:
        error = type(exc).__name__
        raise
    finally:
        if response is not None:
            response.close()
        directory = Path(run_dir) / "state" / "llm_responses"
        directory.mkdir(parents=True, exist_ok=True)
        record = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "request_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            "endpoint_sha256": hashlib.sha256(url.encode()).hexdigest(),
            "model": payload["model"], "stream": payload["stream"],
            "max_tokens": payload["max_tokens"], "finish_reason": finish_reason,
            "done": done, "reasoning_chars": reasoning_chars,
            "duration_seconds": round(time.monotonic() - started, 3),
            "http_status": response.status_code if response is not None else None,
            "stream_read_timeout": stream_timeout or timeout if payload["stream"] else None,
            "usage": usage, "error_type": error, "content": "".join(content),
        }
        path = directory / f"{time.time_ns()}-{uuid.uuid4().hex}.json"
        path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
