import hashlib
import json
import threading
from pathlib import Path

_lock = threading.Lock()
_indices = {}


def find_completed_response(run_dir, url, payload):
    """Reuse only complete successful responses to an identical run-local request."""
    root = Path(run_dir).resolve()
    endpoint = hashlib.sha256(url.encode()).hexdigest()
    key = (str(root), endpoint)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    with _lock:
        if key not in _indices:
            runtime = root / "state/runtime_config.json"
            configuration = json.loads(runtime.read_text()) if runtime.exists() else {}
            base = str(configuration.get("llm_base_url") or "").rstrip("/")
            same_runtime_endpoint = base and (base if base.endswith("/chat/completions") else base + "/chat/completions") == url
            records = {}
            for path in sorted((root / "state/llm_responses").glob("*.json")):
                try:
                    record = json.loads(path.read_text())
                except (ValueError, OSError):
                    continue
                same_endpoint = record.get("endpoint_sha256") == endpoint or (record.get("endpoint_sha256") is None and same_runtime_endpoint)
                if (same_endpoint and record.get("http_status") == 200 and not record.get("error_type")
                        and record.get("finish_reason") == "stop" and str(record.get("content") or "").strip()):
                    records[record["request_sha256"]] = (record, str(path))
            _indices[key] = records
        found = _indices[key].get(digest)
        if found is not None:
            record, source = found
            with (root / "state/completion_cache_hits.jsonl").open("a") as log:
                log.write(json.dumps({"request_sha256": digest, "source_record": source}) + "\n")
            return record
    return None
