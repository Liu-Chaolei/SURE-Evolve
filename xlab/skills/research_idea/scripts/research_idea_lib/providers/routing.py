"""Explicit ordered API routes, with public policy and secret-free provenance."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

from dotenv import dotenv_values

from .contracts import ProviderTrace, structured_input_digest

API_KEY_ORDER = ("XI_API_KEY", "ZAI_API_KEY", "ZAI_API_KEY2", "OPENAI_API_KEY")
ROUTE_MODELS = ("gpt-6-astra", "glm-5.3-flash", "glm-5.3-flash", "gpt-6-astra")
ROUTE_RESPONSES = (True, False, False, True)


def build_routing_policy(values: Mapping[str, str | None]) -> dict:
    bases = (
        values.get("XI_BASE_URL"), values.get("ZAI_BASE_URL"),
        values.get("ZAI_BASE_URL2") or values.get("ZAI_BASE_URL"),
        values.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
    )
    routes = []
    for slot, model, responses, raw in zip(API_KEY_ORDER, ROUTE_MODELS, ROUTE_RESPONSES, bases):
        base = (raw or "").strip().rstrip("/")
        if base:
            url = urlsplit(base)
            if url.scheme != "https" or not url.netloc or url.username or url.password or url.query or url.fragment:
                raise ValueError(f"Invalid HTTPS API base for {slot}")
            if not url.path:
                base += "/v1"
        routes.append({"slot": slot, "model": model, "responses": responses, "base_url": base})
    return {"schema_version": "xlab.api_routing.v1", "routes": routes}


def routing_policy_from_environment() -> dict | None:
    raw = os.environ.get("XLAB_API_ROUTING_POLICY")
    if not raw:
        return None
    policy = json.loads(raw)
    if not isinstance(policy, dict) or set(policy) != {"schema_version", "routes"}:
        raise ValueError("Invalid API routing policy")
    if policy["schema_version"] != "xlab.api_routing.v1":
        raise ValueError("Unsupported API routing policy version")
    routes = policy["routes"]
    if (not isinstance(routes, list) or len(routes) != len(API_KEY_ORDER)
            or not all(isinstance(r, dict) for r in routes)
            or [r.get("slot") for r in routes] != list(API_KEY_ORDER)):
        raise ValueError("API routing order must be XI, ZAI, ZAI2, OPENAI")
    for route, model, responses in zip(routes, ROUTE_MODELS, ROUTE_RESPONSES):
        if set(route) != {"slot", "model", "responses", "base_url"}:
            raise ValueError("API routing policy must not contain credentials or unknown fields")
        if route["model"] != model or route["responses"] is not responses or not isinstance(route["base_url"], str):
            raise ValueError("API routing model or protocol differs from its declared slot")
        if route["base_url"]:
            url = urlsplit(route["base_url"])
            if url.scheme != "https" or not url.netloc or url.username or url.password or url.query or url.fragment:
                raise ValueError("Invalid API routing URL")
    return policy


def routing_credentials(policy: dict) -> dict[str, str]:
    env_file = os.environ.get("XLAB_API_ROUTING_ENV_FILE")
    values = dotenv_values(Path(env_file), interpolate=False) if env_file else os.environ
    if build_routing_policy(values) != policy:
        raise ValueError("API routing endpoints changed; freeze a new routing policy")
    return {slot: str(values[slot]) for slot in API_KEY_ORDER
            if values.get(slot) and not str(values[slot]).startswith("${")}


def trace_matches_requested_model(trace: ProviderTrace, requested_model: str) -> bool:
    if trace.routing is None:
        return trace.model == requested_model
    if not isinstance(trace.routing, dict):
        return False
    policy = routing_policy_from_environment()
    if policy is None or trace.routing.get("policy_digest") != structured_input_digest(policy):
        return False
    if trace.routing.get("requested_model") != requested_model:
        return False
    route = next((r for r in policy["routes"] if r["slot"] == trace.routing.get("selected_slot")), None)
    return route is not None and trace.model == route["model"]
