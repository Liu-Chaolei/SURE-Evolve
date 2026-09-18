"""Loopback OpenAI-compatible gateway with an explicit user-owned API order."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlsplit
import uuid

from dotenv import dotenv_values

from playground.sure_master.tools.completion_transport import completion_transport

SUPPORTED = ('XI_API_KEY', 'ZAI_API_KEY', 'ZAI_API_KEY2', 'ZHOU_API_KEY', 'OPENAI_API_KEY', 'XCODE_API_KEY')
ORDER = ('ZAI_API_KEY', 'ZAI_API_KEY2')
ALIAS = 'sure-priority-router'
ROLE_ORDERS = {
    'sure-glm-router': ('ZAI_API_KEY', 'ZHOU_API_KEY', 'OPENAI_API_KEY', 'XCODE_API_KEY'),
    'sure-openai-router': ('OPENAI_API_KEY', 'XCODE_API_KEY'),
}


@dataclass(frozen=True)
class Upstream:
    name: str
    endpoint: str
    model: str
    key: str = field(repr=False)

    @property
    def identity(self) -> str:
        return hashlib.sha256((self.name + self.endpoint + self.model + self.key).encode()).hexdigest()


def upstreams(values: dict, order=None) -> list[Upstream]:
    order = ORDER if order is None else order
    if (not isinstance(order, (list, tuple)) or not order or
            len(order) != len(set(order)) or not set(order) <= set(SUPPORTED)):
        raise ValueError('Invalid API priority order')
    result = []
    for name in order:
        if not values.get(name):
            continue
        if name == 'XI_API_KEY':
            base, model = values.get('XI_BASE_URL'), values.get('XI_MODEL') or 'gpt-6-astra'
        elif name in ('ZAI_API_KEY', 'ZAI_API_KEY2'):
            second = name == 'ZAI_API_KEY2'
            base = (values.get('ZAI_BASE_URL2') if second else None) or values.get('ZAI_BASE_URL')
            model = (values.get('ZAI_MODEL2') if second else None) or values.get('ZAI_MODEL') or 'glm-5.3-flash'
        elif name == 'ZHOU_API_KEY':
            base, model = values.get('ZHOU_API_BASE_URL'), 'glm-5.3-flash'
        elif name == 'XCODE_API_KEY':
            base = values.get('XCODE_API_BASE_URL')
            model = values.get('XCODE_MODEL') or values.get('OPENAI_MODEL') or 'gpt-6-astra'
        else:
            base = values.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1'
            model = values.get('OPENAI_MODEL') or 'gpt-6-astra'
        if not base:
            raise ValueError(f'Missing endpoint for {name}')
        parsed = urlsplit(str(base))
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(f'Invalid HTTPS endpoint for {name}')
        base = str(base).rstrip('/') + ('/v1' if not parsed.path.strip('/') else '')
        result.append(Upstream(name, base + '/chat/completions', str(model), str(values[name])))
    if not result:
        raise ValueError('No configured API credentials')
    return result


def cooldown_seconds(status: int, headers: dict, body: bytes, now: float) -> float:
    if status in (401, 403):
        return 300
    if status == 429:
        retry = next((v for k, v in headers.items() if k.lower() == 'retry-after'), None)
        if retry and str(retry).isdigit():
            return max(1, int(retry))
        # ZAI code 1308 reports its reset time in local cluster time (CST).
        text = body.decode(errors='replace')
        try:
            error = json.loads(text).get('error', {})
            if str(error.get('code')) == '1113':
                return 300  # No balance/resource package; do not hammer it.
            text = json.dumps(error, ensure_ascii=False)
        except (ValueError, AttributeError):
            pass
        match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', text)
        if match and ('1308' in text or '重置' in text):
            try:
                return max(60, datetime.strptime(match[1], '%Y-%m-%d %H:%M:%S').timestamp() - now + 5)
            except ValueError:
                pass
        return 60
    return 15


class PriorityRouter:
    def __init__(self, load_values, audit_path: Path, *, transport=completion_transport, clock=time.time,
                 load_order=None, cooldown_file: Path | None = None, wait_timeout: float = 0,
                 max_in_flight_per_key: int = 1):
        self.load_values = load_values
        self.audit_path = audit_path
        self.transport = transport
        self.clock = clock
        self.load_order = load_order or (lambda: ORDER)
        self.cooldown_file = cooldown_file
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        if max_in_flight_per_key < 1:
            raise ValueError('API concurrency must be positive')
        self.busy: dict[str, int] = {}
        self.max_in_flight_per_key = max_in_flight_per_key
        self.active_requests = 0
        self.wait_timeout = wait_timeout
        self.blocked_until: dict[str, float] = {}
        if cooldown_file and cooldown_file.exists():
            self.blocked_until = {str(k): float(v) for k,v in json.loads(cooldown_file.read_text()).items()}

    def record(self, value: dict) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock, self.audit_path.open('a') as log:
            log.write(json.dumps({'time': self.clock(), **value}, ensure_ascii=False) + '\n')

    def health(self) -> dict:
        order = self.load_order()
        profiles = upstreams(self.load_values(), order)
        with self.lock:
            return {'status': 'ready', 'order': list(order), 'alias': ALIAS,
                    'upstreams': [{'name': p.name, 'model': p.model,
                        'retry_at': self.blocked_until.get(p.identity, 0),
                        'in_flight': self.busy.get(p.identity, 0)} for p in profiles],
                    'max_in_flight_per_key': self.max_in_flight_per_key, 'wait_timeout_seconds': self.wait_timeout,
                    'queued_requests': max(0, self.active_requests - sum(self.busy.values()))}

    def complete(self, payload: dict) -> tuple[int, dict, bytes]:
        deadline = time.monotonic() + self.wait_timeout
        with self.lock:
            self.active_requests += 1
        try:
            while True:
                result = self._complete_once(payload)
                if result[0] != 503 or time.monotonic() >= deadline:
                    if result[0] == 503 and self.wait_timeout:
                        self.record({'status': 'queue_timeout', 'wait_timeout_seconds': self.wait_timeout,
                                     'request_digest': hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()})
                    return result
                with self.condition:
                    self.condition.wait(timeout=min(1, max(0, deadline - time.monotonic())))
        finally:
            with self.condition:
                self.active_requests -= 1
                self.condition.notify_all()

    def _complete_once(self, payload: dict) -> tuple[int, dict, bytes]:
        enabled = self.load_order()
        role_order = ROLE_ORDERS.get(payload.get('model'))
        order = [name for name in role_order if name in enabled] if role_order else enabled
        profiles = upstreams(self.load_values(), order)
        route_id = uuid.uuid4().hex
        request_digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        attempts = []
        for profile in profiles:
            # A removed endpoint must not be attempted later in an in-flight
            # request's fallback chain after a live policy update.
            if profile.name not in self.load_order():
                continue
            with self.lock:
                retry_at = self.blocked_until.get(profile.identity, 0)
                if retry_at > self.clock():
                    attempts.append({'provider': profile.name, 'status': 'cooldown'})
                    continue
                if self.busy.get(profile.identity, 0) >= self.max_in_flight_per_key:
                    attempts.append({'provider': profile.name, 'status': 'busy'})
                    continue
                self.busy[profile.identity] = self.busy.get(profile.identity, 0) + 1
            try:
                result = self._call_profile(payload, profile, profiles, route_id, request_digest)
                if result[0] != 0:
                    return result
                attempts.append({'provider': profile.name, 'status': 'unavailable'})
            finally:
                with self.condition:
                    self.busy[profile.identity] -= 1
                    if not self.busy[profile.identity]:
                        del self.busy[profile.identity]
                    self.condition.notify_all()
        if not self.wait_timeout:
            self.record({'route_id': route_id, 'request_digest': request_digest, 'status': 'all_unavailable', 'attempts': attempts})
        body = json.dumps({'error': {'type': 'upstream_unavailable', 'message': 'No configured API is currently available.',
                                     'attempts': attempts}}).encode()
        return 503, {'Content-Type': 'application/json', 'X-SURE-Route-ID': route_id}, body

    def _call_profile(self, payload, profile, profiles, route_id, request_digest):
        outgoing = {**payload, 'model': profile.model}
        # GLM may return a provider-specific assistant field; ordinary
        # Chat Completions endpoints do not all accept it on input.
        if profile.name not in ('ZAI_API_KEY', 'ZAI_API_KEY2'):
            outgoing['messages'] = [{k: v for k, v in message.items() if k != 'reasoning_content'}
                                    for message in payload.get('messages', [])]
        body = json.dumps(outgoing, ensure_ascii=False).encode()
        start = self.clock()
        try:
            status, headers, response = self.transport(profile.endpoint,
                {'Authorization': 'Bearer ' + profile.key, 'Content-Type': 'application/json'}, body, 600)
        except (OSError, TimeoutError, ConnectionError):
            status, headers, response = 502, {}, b''
        item = {'route_id': route_id, 'request_digest': request_digest, 'request_model': payload.get('model'),
                'provider': profile.name, 'upstream_model': profile.model,
                'http_status': status, 'seconds': round(self.clock() - start, 3)}
        if status >= 400:
            try:
                error = json.loads(response).get('error', {})
                code = str(error.get('code', ''))
                if re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', code):
                    item['provider_error_code'] = code
            except (ValueError, AttributeError):
                pass
        if 200 <= status < 300:
            # Responses are buffered until the upstream completion marker;
            # no partial answer or tool call is exposed before failover.
            self.record({**item, 'status': 'success'})
            content_type = next((v for k, v in headers.items() if k.lower() == 'content-type'), 'application/json')
            return status, {'Content-Type': content_type, 'X-SURE-Route-ID': route_id,
                            'X-SURE-Upstream': profile.name, 'X-SURE-Model': profile.model}, response
        fallback = status in (401, 403, 408, 429) or status >= 500
        if status == 400 and any(token in response.lower() for token in (b'model_not_found', b'unsupported_model', b'invalid_model')):
            fallback = True
        if not fallback:
            # Preserve context/validation errors for the caller's own
            # compaction logic. Do not hide a malformed request by routing.
            for entry in profiles:
                response = response.replace(entry.key.encode(), b'[redacted]')
            self.record({**item, 'status': 'request_error'})
            return status, {'Content-Type': 'application/json', 'X-SURE-Route-ID': route_id}, response
        until = self.clock() + cooldown_seconds(status, headers, response, self.clock())
        with self.lock:
            self.blocked_until[profile.identity] = until
            if self.cooldown_file:
                temporary = self.cooldown_file.with_suffix('.pending')
                temporary.write_text(json.dumps(self.blocked_until) + '\n')
                temporary.replace(self.cooldown_file)
        self.record({**item, 'status': 'fallback', 'retry_at': until})
        return 0, {}, b''

def serve(env_file: Path, token_file: Path, audit_path: Path, port: int,
          policy_file: Path | None = None, cooldown_file: Path | None = None) -> None:
    token = token_file.read_text().strip()
    if not token or token_file.stat().st_mode & 0o077:
        raise ValueError('Local router token must be nonempty and mode 0600')
    router = PriorityRouter(lambda: dotenv_values(env_file, interpolate=False), audit_path,
        load_order=(lambda: json.loads(policy_file.read_text())['order']) if policy_file else None,
        cooldown_file=cooldown_file, wait_timeout=2400, max_in_flight_per_key=4)
    router.health()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def authorized(self):
            return hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + token)

        def reply(self, status, headers, body):
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if not self.authorized():
                return self.reply(401, {}, b'Unauthorized')
            if self.path == '/health':
                return self.reply(200, {'Content-Type': 'application/json'}, json.dumps(router.health()).encode())
            if self.path == '/v1/models':
                return self.reply(200, {'Content-Type': 'application/json'},
                                  json.dumps({'object':'list','data':[{'id':ALIAS,'object':'model','owned_by':'local-router'}]}).encode())
            self.reply(404, {}, b'Not found')

        def do_POST(self):
            if not self.authorized():
                return self.reply(401, {}, b'Unauthorized')
            if self.path != '/v1/chat/completions':
                return self.reply(404, {}, b'Not found')
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 64 * 1024 * 1024:
                    return self.reply(413, {}, b'Request too large')
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict) or not isinstance(payload.get('messages'), list):
                    return self.reply(400, {}, b'Invalid chat request')
                status, headers, body = router.complete(payload)
                self.reply(status, headers, body)
            except (ValueError, KeyError, TypeError):
                self.reply(503, {'Content-Type':'application/json'}, b'{"error":{"message":"Router configuration or request error"}}')

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    print(json.dumps({'status':'listening','host':'127.0.0.1','port':server.server_port,'order':router.health()['order']}), flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--token-file', type=Path, required=True)
    parser.add_argument('--audit-log', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18991)
    parser.add_argument('--policy-file', type=Path)
    parser.add_argument('--cooldown-file', type=Path)
    args = parser.parse_args()
    serve(args.env_file, args.token_file, args.audit_log, args.port, args.policy_file, args.cooldown_file)


if __name__ == '__main__':
    main()
