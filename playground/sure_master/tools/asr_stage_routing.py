"""Explicit per-operation ASR API policy, installed above immutable research code."""
from __future__ import annotations

from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import time
from urllib.parse import urlsplit

from dotenv import dotenv_values

SLOTS = ("ZAI_API_KEY", "ZHOU_API_KEY", "OPENAI_API_KEY", "XCODE_API_KEY")
MODELS = ("glm-5.3-flash", "glm-5.3-flash", "gpt-6-astra", "gpt-6-astra")
BASES = ("ZAI_BASE_URL", "ZHOU_API_BASE_URL", "OPENAI_BASE_URL", "XCODE_API_BASE_URL")
OPENAI_SLOTS = frozenset({"OPENAI_API_KEY", "XCODE_API_KEY"})
OPENAI_OPERATIONS = frozenset({
    "xlab.sure.parent_projection.v1",
    "xlab.research_idea.analysis.generate.v1",
    "xlab.research_idea.analysis.replan.v1",
    "xlab.research_idea.idea.generate.v1",
    "xlab.research_idea.fusion.generate.v1",
    "xlab.research_idea.fusion.referee.v1",
    "xlab.research_idea.fusion.repair.v1",
    "xlab.research_idea.idea.materialize.v1",
    "xlab.sure.candidate_review.v1",
})
GLM_OPERATIONS = frozenset({
    "xlab.research_idea.background.generate.v1",
    "xlab.research_idea.retrieval.query.v1",
    "xlab.research_idea.reference.rank.v1",
    "xlab.research_idea.keynote.score.v1",
    "xlab.research_idea.keynote.compress.v1",
    "xlab.research_idea.keynote.rollup.v1",
    "xlab.research_idea.operator.theory-transfer-query.v1",
    "xlab.research_idea.operator.mechanism-commit-query.v1",
    "xlab.research_idea.idea.diagnostic.v1",
    "xlab.research_idea.idea.evaluate.v1",
    "xlab.research_idea.component_novelty.evaluate.v1",
    "xlab.sure.executable_architecture.v1",
})
AGENT_ROUTES = {
    "prefetch": "glm", "draft": "glm", "debug": "glm", "improve": "glm",
    "reseach": "openai", "knowledge_promotion": "openai", "wisdom_promotion": "openai",
}


def build_policy(values):
    routes = []
    for slot, base_key, model in zip(SLOTS, BASES, MODELS):
        base = str(values.get(base_key) or "").strip().rstrip("/")
        url = urlsplit(base)
        if (url.scheme != "https" or not url.netloc or url.username or url.password
                or url.query or url.fragment):
            raise ValueError(f"Missing or invalid HTTPS endpoint: {base_key}")
        if not url.path:
            base += "/v1"
        routes.append(dict(slot=slot, base_url=base, model=model, responses=slot in OPENAI_SLOTS))
    return {"schema_version": "asr.api_stage_routing.v1", "routes": routes,
            "operations": {**{k: "glm" for k in sorted(GLM_OPERATIONS)},
                           **{k: "openai" for k in sorted(OPENAI_OPERATIONS)}},
            "agents": AGENT_ROUTES}


def routing_policy_from_environment():
    raw = os.environ.get("XLAB_API_ROUTING_POLICY")
    if not raw:
        return None
    policy = json.loads(raw)
    values = dotenv_values(os.environ["XLAB_API_ROUTING_ENV_FILE"], interpolate=False)
    if policy != build_policy(values):
        raise ValueError("Stage routing policy/endpoints changed; prepare a new revision")
    return policy


def routing_credentials(policy):
    if policy != routing_policy_from_environment():
        raise ValueError("Unexpected stage routing policy")
    values = dotenv_values(os.environ["XLAB_API_ROUTING_ENV_FILE"], interpolate=False)
    return {slot: str(values[slot]) for slot in SLOTS
            if values.get(slot) and not str(values[slot]).startswith("${")}


def select_routes(policy, operation):
    group = policy["operations"].get(operation)
    if group not in {"glm", "openai"}:
        raise ValueError(f"Unclassified ASR API operation: {operation}")
    return [r for r in policy["routes"] if group == "glm" or r["slot"] in OPENAI_SLOTS]


def trace_matches_requested_model(trace, requested_model):
    from research_idea_lib.providers.contracts import structured_input_digest
    if trace.routing is None:
        return trace.model == requested_model
    policy = routing_policy_from_environment()
    if (not isinstance(trace.routing, dict) or policy is None
            or trace.routing.get("policy_digest") != structured_input_digest(policy)
            or trace.routing.get("requested_model") != requested_model):
        return False
    return any(r["slot"] == trace.routing.get("selected_slot") and r["model"] == trace.model
               for r in select_routes(policy, trace.operation))


def install_native():
    """Keep the proven transport/recovery implementation; change route selection only."""
    from research_idea_lib.providers import routing
    from research_idea_lib.providers import openai_compatible as provider
    replacements = {
        routing.routing_policy_from_environment: routing_policy_from_environment,
        routing.routing_credentials: routing_credentials,
        routing.trace_matches_requested_model: trace_matches_requested_model,
    }
    for module_name, module in list(sys.modules.items()):
        if not module or not (module_name.startswith("research_idea_lib") or module_name == "native_sure"):
            continue
        for name, value in list(vars(module).items()):
            for old, new in replacements.items():
                if value is old:
                    setattr(module, name, new)
    source = textwrap.dedent(inspect.getsource(provider.OpenAICompatibleProvider._complete_routed))
    needle = 'for route in policy["routes"]:'
    if source.count(needle) != 1:
        raise ValueError("Native routing implementation changed; review the adapter")
    source = source.replace(needle, 'for route in select_routes(policy, request.operation):')
    namespace = dict(vars(provider), select_routes=select_routes)
    exec(compile(source, __file__ + ":native_route_selection", "exec"), namespace)
    provider.OpenAICompatibleProvider._complete_routed = namespace["_complete_routed"]
    routing.API_KEY_ORDER = SLOTS


def install_controller(env_file: Path, policy_path: Path, audit: Path):
    """Route SURE SDK requests by their configured model without a shared proxy."""
    from openai import OpenAI, APIConnectionError, APIStatusError
    from openai.resources.chat.completions import Completions
    policy = json.loads(policy_path.read_text())
    original = Completions.create

    def create(_self, *args, **kwargs):
        requested = kwargs.get("model")
        if requested not in set(MODELS):
            raise ValueError(f"Unclassified SURE controller model: {requested}")
        values = dotenv_values(env_file, interpolate=False)
        if build_policy(values) != policy:
            raise ValueError("Controller API policy identity changed")
        routes = [r for r in policy["routes"] if requested != "gpt-6-astra" or r["slot"] in OPENAI_SLOTS]
        last_error = None
        for route in routes:
            key = values.get(route["slot"])
            if not key:
                continue
            client = OpenAI(api_key=str(key), base_url=route["base_url"], max_retries=0, timeout=600)
            outgoing = {**kwargs, "model": route["model"]}
            if route["model"] == "glm-5.3-flash":
                outgoing.setdefault("reasoning_effort", "low")
            started = time.time()
            try:
                answer = original(client.chat.completions, *args, **outgoing)
            except (APIConnectionError, APIStatusError) as error:
                code = getattr(error, "status_code", None)
                if code is not None and code not in {401, 403, 408, 429} and code < 500:
                    raise
                last_error = error
                event = dict(status="unavailable", http_status=code)
            else:
                event = dict(status="stream_dispatched" if kwargs.get("stream") else "completed")
            with audit.open("a") as handle:
                handle.write(json.dumps(dict(time=time.time(), requested_model=requested,
                    selected_slot=route["slot"], actual_model=route["model"],
                    seconds=time.time()-started, **event)) + "\n")
            if event["status"] != "unavailable":
                return answer
            client.close()
        # Do not expose upstream error bodies or credentials in operational logs.
        raise RuntimeError("Configured ASR controller API routes unavailable") from None

    Completions.create = create
