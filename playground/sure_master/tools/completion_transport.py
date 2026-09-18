"""Read completed SSE responses without waiting for a proxy to close the socket."""
from __future__ import annotations

import json
import urllib.error
import urllib.request


def completion_transport(endpoint, headers, body, timeout_seconds):
    request_headers = dict(headers)
    # Some compatible gateways reject urllib's default User-Agent (HTTP 403/1010).
    if not any(key.lower() == "user-agent" for key in request_headers):
        request_headers["User-Agent"] = "SURE-Evolve/1.0"
    request = urllib.request.Request(endpoint, data=body, headers=request_headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status, response_headers = int(response.getcode()), dict(response.headers.items())
            if "text/event-stream" not in response.headers.get("Content-Type", ""):
                return status, response_headers, response.read()
            chunks = []
            finished = False
            while True:
                line = response.readline()
                if not line:
                    if not finished:
                        raise ConnectionError("SSE stream ended before a completion marker")
                    return status, response_headers, b"".join(chunks)
                chunks.append(line)
                if line.strip() == b"data: [DONE]":
                    return status, response_headers, b"".join(chunks)
                if line.startswith(b"data:"):
                    try:
                        event = json.loads(line[5:].strip())
                    except (ValueError, UnicodeDecodeError):
                        continue
                    finished = finished or event.get("type") == "response.completed" or any(
                        choice.get("finish_reason") is not None for choice in event.get("choices", []))
                    if event.get("type") == "response.completed":
                        return status, response_headers, b"".join(chunks)
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers.items()), error.read()
