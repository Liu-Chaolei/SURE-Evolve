from __future__ import annotations

import json
from collections.abc import Mapping
import os
from pathlib import Path
import threading
import time

import httpx

from .common import atomic_write_json
from .corpus_store import fingerprint
from .llm import extract_json_object


class ServiceUnavailable(RuntimeError):
    pass


class OutputInvalid(ValueError):
    pass


class CorpusLlm:
    def __init__(self, run: Path, tokenizer: object, model_identity: dict, *, context: int = 32768) -> None:
        self.run = run
        self.tokenizer = tokenizer
        self.identity = model_identity
        self.context = context
        self.client = httpx.Client(timeout=httpx.Timeout(600, connect=15), trust_env=False)
        self.lock = threading.Lock()
        self.calls = 0
        self.benchmark_nonce = ''

    def messages(self, system: str, user: str) -> list[dict[str, str]]:
        return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]

    def count(self, messages: list[dict]) -> int:
        encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        if isinstance(encoded, Mapping):
            encoded = encoded['input_ids']
        if encoded and isinstance(encoded[0], (list, tuple)):
            if len(encoded) != 1:
                raise ValueError('Expected a single encoded conversation')
            encoded = encoded[0]
        return len(encoded)

    def chunks(self, text: str, *, reserve: int = 8000, maximum: int = 20000) -> list[dict]:
        budget = min(maximum, self.context - reserve)
        if budget < 512:
            raise ValueError('Context leaves no useful document budget')
        chunks = []
        start = 0
        overlap = min(512, budget // 8)
        while start < len(text):
            low, high = start + 1, len(text)
            while low < high:
                middle = (low + high + 1) // 2
                if len(self.tokenizer.encode(text[start:middle], add_special_tokens=False)) <= budget:
                    low = middle
                else:
                    high = middle - 1
            end = low
            if end < len(text):
                boundary = text.rfind('\n\n', start + (end - start) * 3 // 4, end)
                if boundary > start:
                    end = boundary
            chunks.append({'start': start, 'end': end, 'text': text[start:end]})
            if end == len(text):
                break
            low, high = start + 1, end
            while low < high:
                middle = (low + high) // 2
                if len(self.tokenizer.encode(text[middle:end], add_special_tokens=False)) > overlap:
                    low = middle + 1
                else:
                    high = middle
            start = max(start + 1, low)
        return chunks

    def complete(self, system: str, user: str, *, maximum: int, tag: str, json_output: bool = True) -> tuple[object, dict]:
        if self.benchmark_nonce:
            system = 'Benchmark isolation label: ' + self.benchmark_nonce + '\n' + system
        messages = self.messages(system, user)
        input_tokens = self.count(messages)
        if input_tokens + maximum + 1024 > self.context:
            raise OutputInvalid('Request exceeds the tokenizer-verified context budget')
        payload = {'model': 'Qwen3.8-27B-W8A8', 'messages': messages, 'max_tokens': maximum,
                   'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}}
        registry = json.loads((self.run / 'service.json').read_text())
        backends = [item for item in registry['backends'] if item.get('ready')]
        if not backends:
            raise ServiceUnavailable('No healthy local model backend')
        with self.lock:
            backend = backends[self.calls % len(backends)]
            self.calls += 1
        key = os.environ.get('SPEECH_PIPELINE_KEY', '')
        if not key:
            raise ServiceUnavailable('Local model authentication is not configured')
        error = ''
        for attempt in range(3):
            started = time.monotonic()
            try:
                response = self.client.post(backend['url'].rstrip('/') + '/chat/completions', json=payload, headers={'Authorization': f'Bearer {key}'})
                if response.status_code in (408, 429) or response.status_code >= 500:
                    error = f'HTTP {response.status_code}'
                    delay = response.headers.get('Retry-After', '5')
                    time.sleep(min(60, float(delay) if delay.replace('.', '', 1).isdigit() else 5))
                    continue
                if response.status_code != 200:
                    raise OutputInvalid(f'Non-retryable model HTTP {response.status_code}')
                body = response.json()
                choice = body['choices'][0]
                if choice.get('finish_reason') != 'stop':
                    raise OutputInvalid(f"Incomplete response: {choice.get('finish_reason')}")
                content = choice['message'].get('content') or ''
                if not content.strip():
                    raise OutputInvalid('Empty model response')
                value = extract_json_object(content) if json_output else content
                meta = {'tag': tag, 'input_tokens': input_tokens, 'usage': body.get('usage', {}),
                        'seconds': round(time.monotonic() - started, 3), 'backend': backend['id'],
                        'request_sha256': fingerprint({'identity': self.identity, 'payload': payload}), 'time': time.time()}
                with self.lock:
                    log = self.run / 'llm-calls.jsonl'
                    with log.open('a') as handle:
                        handle.write(json.dumps(meta) + '\n')
                return value, meta
            except (httpx.TransportError, KeyError, json.JSONDecodeError) as exc:
                error = type(exc).__name__
                time.sleep(min(20, 2 ** attempt))
        raise ServiceUnavailable(f'Local model unavailable after bounded retries: {error}')

    def validated(self, system: str, user: str, *, maximum: int, tag: str, validator: object, cache_path: Path, signature: str) -> dict:
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            if cached.get('signature') == signature and cached.get('validated'):
                try:
                    return validator(cached['value'])
                except ValueError:
                    cache_path.replace(cache_path.with_suffix('.stale.json'))
        correction = ''
        failures = []
        for attempt in range(2):
            value = None
            try:
                value, meta = self.complete(system, user + correction, maximum=maximum, tag=tag)
                validated = validator(value)
                atomic_write_json(cache_path, {'signature': signature, 'validated': True, 'value': validated, 'response': meta})
                return validated
            except (OutputInvalid, ValueError) as error:
                failures.append({'attempt': attempt + 1, 'error': str(error), 'value': value})
                correction = '\nYour response failed validation: ' + str(error)[:2000] + '\nReturn a complete corrected JSON. Copy evidence quotes verbatim from the supplied text. Remove unsupported facts; do not repeat invalid quotes.'
        atomic_write_json(cache_path.with_suffix('.failure.json'), {'signature': signature, 'tag': tag, 'failures': failures})
        raise OutputInvalid('Business validation failed after one correction: ' + failures[-1]['error'])
