"""Unwrap explicit response envelopes without guessing or replacing fields."""
import json
from copy import deepcopy
from typing import Any, Mapping


def unwrap_answer_object(value: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    payload = deepcopy(dict(value))
    depth = 0
    while depth < 3 and set(payload) == {"answer"}:
        wrapped = payload["answer"]
        if isinstance(wrapped, str):
            try:
                wrapped = json.loads(wrapped)
            except json.JSONDecodeError:
                break
        if not isinstance(wrapped, dict):
            break
        payload = deepcopy(wrapped)
        depth += 1
    return payload, depth
