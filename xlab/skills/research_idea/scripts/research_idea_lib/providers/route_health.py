"""Deployment-local, process-safe cooldowns; no credentials are persisted."""
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re
import time


def cooldown_seconds(status, headers, body, now):
    if status in (401, 403):
        return 300.0
    if status == 429:
        retry = next((v for k, v in headers.items() if k.lower() == 'retry-after'), None)
        if retry:
            try:
                return max(1.0, float(retry))
            except ValueError:
                try:
                    return max(1.0, parsedate_to_datetime(retry).timestamp() - now)
                except (ValueError, TypeError, OverflowError):
                    pass
        try:
            error = json.loads(body).get('error', {})
            if str(error.get('code')) == '1113':
                return 300.0
            text = str(error.get('message', ''))
            match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', text)
            if match and (str(error.get('code')) == '1308' or '重置' in text):
                reset = datetime.fromisoformat(match[0] + '+08:00').timestamp()
                return max(60.0, reset - now + 5)
        except (ValueError, TypeError, AttributeError):
            pass
        return 60.0
    return 15.0


class RouteHealth:
    def __init__(self, root: Path, *, clock=time.time):
        self.root = root
        self.clock = clock

    @contextmanager
    def _record(self, slot, endpoint):
        self.root.mkdir(parents=True, exist_ok=True)
        identity = hashlib.sha256((slot + '\0' + endpoint).encode()).hexdigest()
        path = self.root / (identity + '.json')
        with (self.root / (identity + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            value = json.loads(path.read_text()) if path.exists() else {}
            yield value
            pending = path.with_suffix('.pending')
            pending.write_text(json.dumps(value, sort_keys=True) + '\n')
            pending.replace(path)

    def acquire(self, slot, endpoint, *, lease_seconds):
        with self._record(slot, endpoint) as value:
            until = max(value.get('retry_at', 0), value.get('probe_until', 0))
            if until > self.clock():
                return until
            if value.get('retry_at', 0):
                value['probe_until'] = self.clock() + lease_seconds
            return 0.0

    def finish(self, slot, endpoint, *, delay=0):
        with self._record(slot, endpoint) as value:
            value.update(retry_at=self.clock() + delay if delay else 0,
                         probe_until=0, updated_at=self.clock())
