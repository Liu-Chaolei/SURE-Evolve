"""Durable bounded retries for one confirmed native model operation."""
import json
import time
from pathlib import Path

from .contracts import ProviderExhaustedError, ProviderRecoveryBlockedError

TRANSIENT = {'timeout', 'transport_error', 'http_408', 'http_425', 'http_429',
             'http_500', 'http_502', 'http_503', 'http_504'}


def write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(record, sort_keys=True) + '\n')
    pending.replace(path)


def bounded_request(path, request_digest, callback, *, sleep=time.sleep, now=time.time):
    record = json.loads(path.read_text()) if path.exists() else {
        'request_digest': request_digest, 'cycles': 0, 'deadline': now() + 1800,
        'status': 'ready', 'events': [],
    }
    if record['request_digest'] != request_digest:
        raise ProviderRecoveryBlockedError('Native request recovery identity changed')
    if record['status'] in {'running', 'completed'}:
        raise ProviderRecoveryBlockedError('Unconfirmed native operation requires reconciliation; no automatic resend')
    if record['status'] == 'failed' and record.get('error_code') not in TRANSIENT:
        raise ProviderRecoveryBlockedError('Native operation failed with a non-retryable error')
    while record['cycles'] < 3 and now() < record['deadline']:
        record.update(cycles=record['cycles'] + 1, status='running')
        record['events'].append({'event': 'cycle_started', 'time': now(), 'cycle': record['cycles']})
        write_record(path, record)
        try:
            result = callback(record['deadline'])
        except ProviderExhaustedError as error:
            retry_at = getattr(error, 'retry_at', None)
            if error.trace.attempts == 0 and retry_at is not None:
                record.update(status='ready', cycles=record['cycles'] - 1)
                record['events'].append({'event': 'cooldown_wait', 'time': now(), 'retry_at': retry_at})
                write_record(path, record)
                if retry_at >= record['deadline']:
                    record.update(status='failed', error_code=error.trace.error_code)
                    write_record(path, record)
                    raise
                sleep(min(60, max(0.1, retry_at-now())))
                continue
            record.update(status='failed', error_code=error.trace.error_code)
            record['events'].append({'event': 'cycle_failed', 'time': now(), 'code': error.trace.error_code})
            write_record(path, record)
            if error.trace.error_code not in TRANSIENT or record['cycles'] >= 3 or now() >= record['deadline']:
                raise
            sleep(min(60 if record['cycles'] == 1 else 180, max(0, record['deadline'] - now())))
        except Exception:
            record.update(status='failed', error_code='contract_error')
            write_record(path, record)
            raise
        else:
            # The provider publishes its result cache immediately after returning.
            record.update(status='completed', finished_at=now())
            write_record(path, record)
            return result
    raise ProviderRecoveryBlockedError('Native request recovery exhausted its persisted 30-minute/3-cycle budget')
