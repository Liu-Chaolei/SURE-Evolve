"""Native SURE policy, grounded parent projection and deterministic observations."""
from __future__ import annotations
import os

import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

from research_idea_lib.algorithm.memory import CandidateOutcomeRecord, MemoryState
from research_idea_lib.algorithm.runtime_adapters import _idea_state
from research_idea_lib.algorithm.response_projection import unwrap_answer_object
from research_idea_lib.common import atomic_write_json, read_json
from research_idea_lib.config import provider_api_key
from research_idea_lib.providers import OpenAICompatibleConfig, OpenAICompatibleProvider, ProviderRequest
from research_idea_lib.research_policy import ResearchPolicy


def digest(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                               separators=(",", ":")).encode()).hexdigest()


def policy_for(payload):
    generation = payload.get("generation_policy", {})
    if generation.get("engine") != "native":
        return {}
    policy = ResearchPolicy(evidence_mode="local_literature" if generation.get("ablation", {}).get("use_literature", True)
                            else "task_only", feedback_kind=generation.get("feedback_kind", "candidate_outcome"),
                            refinement=generation.get("refinement", "current_best"),
                            allow_auxiliary_experiments=generation.get("allow_auxiliary_experiments", False))
    return asdict(ResearchPolicy.from_payload(asdict(policy)))


def native_runtime(runtime, payload):
    values = payload.get("generation_policy", {}).get("mcts", {})
    allowed = {"max_iterations", "max_depth", "branching_factor", "exploration_constant"}
    if set(values) - allowed:
        raise ValueError("Unknown MCTS configuration")
    for key, value in values.items():
        if key == "exploration_constant":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid exploration constant")
        elif type(value) is not int or value < 1:
            raise ValueError("MCTS limits must be positive integers")
    return replace(runtime, **values)


def feedback_records(payload):
    records = []
    scope = payload.get("generation_policy", {}).get("memory_scope")
    if payload.get("generation_policy", {}).get("ablation", {}).get("use_feedback") is False:
        if payload.get("prior_rounds"):
            raise ValueError("Feedback-disabled generation received history")
        return {"feedback_kind": "candidate_outcome", "records": []}
    for history in payload.get("prior_rounds", []):
        for raw in history.get("summary", {}).get("symbolic_memory", []):
            record = CandidateOutcomeRecord(**raw)
            if scope and record.scope == scope:
                records.append(record.to_payload())
    memory = MemoryState.from_experiment_feedback({"feedback_kind": "candidate_outcome", "records": records})
    return {"feedback_kind": "candidate_outcome", "records": [r.to_payload() for r in memory.candidate_records]}


def outcome_records(payload):
    parent = payload.get("parent_snapshot", {})
    context = payload.get("evaluation_context", {})
    scope = context.get("memory_scope")
    if not scope:
        return []
    metric = payload.get("metric", {})
    direction = {"lower_is_better": "lower", "higher_is_better": "higher"}.get(metric.get("direction"), metric.get("direction"))
    records = []
    for candidate in payload.get("candidates", []):
        score = next((r.get("score") for r in candidate.get("rungs", []) if r.get("name") == "search" and r.get("success")), None)
        parent_score = parent.get("score")
        finite = lambda value: not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        score = score if finite(score) else None
        parent_score = parent_score if finite(parent_score) else None
        failure = candidate.get("failure_category")
        outcome = "incomparable"
        if candidate.get("final_status") != "success" or failure:
            outcome = "execution_failed"
        elif (score is not None and parent_score is not None and direction in {"lower", "higher"}
              and parent.get("memory_scope") == scope and parent.get("solution_digest")):
            gain = parent_score - score if direction == "lower" else score - parent_score
            outcome = "improved" if gain > 0 else "worse" if gain < 0 else "tied"
        identity = {"run": payload.get("sure_run_id"), "result": payload.get("result_digest"),
                    "candidate": candidate.get("idea_id")}
        record = CandidateOutcomeRecord(
            record_id=digest(identity), candidate_id=candidate["idea_id"],
            parent_digest=parent.get("solution_digest", ""),
            implementation_digest=candidate.get("code_digest", ""), result_digest=payload["result_digest"],
            scope=scope, change_set=candidate.get("idea", {}).get("spec", {}).get("change_set", []),
            outcome=outcome, metric=metric.get("name", ""), direction=direction or "",
            parent_score=parent_score, candidate_score=score, failure_category=failure,
            source_artifacts=[candidate["idea_artifact_id"]] if candidate.get("idea_artifact_id") else [],
        )
        records.append(record.to_payload())
    return records


def prepare_parent(payload, run_dir: Path, runtime):
    """Project the measured implementation once; no invented baseline intervention."""
    parent = dict(payload.get("current_best", {}))
    if parent.get("native_idea"):
        if parent.get("native_idea_digest") != digest(parent["native_idea"]):
            raise ValueError("Mature idea digest mismatch")
        _idea_state(parent["native_idea"])
        return parent
    supplied = {"current_best": parent, "base_model_profile": payload.get("base_model_profile", {}),
                "task_card": payload.get("task_card", {}), "execution_contract": payload.get("execution_contract", {})}
    signature = digest({"input": supplied, "model": runtime.agent_model, "profile": "sure.parent_projection.v1",
        "endpoint": runtime.chat_completions_url,
        "wire_policy": {key: os.environ.get(key) for key in ("XLAB_SURE_RESPONSES", "XLAB_RESEARCH_IDEA_STREAM",
            "XLAB_SURE_MAX_OUTPUT_TOKENS", "XLAB_SURE_REASONING_EFFORT", "XLAB_API_ROUTING_POLICY")}})
    path = run_dir.parent / "parent_projections" / (signature.split(":", 1)[1] + ".json")
    if path.exists():
        cached = read_json(path)
        if cached.get("signature") != signature:
            raise ValueError("Parent projection identity changed")
        value = cached["idea"]
    else:
        def validate_parent(response):
            if not isinstance(response.json_value, dict):
                raise ValueError('Parent projection requires an object')
            projected, _ = unwrap_answer_object(response.json_value)
            for name in ['tags', 'root_domains']:
                if not isinstance(projected.get(name), list) or any(not isinstance(v, str) for v in projected[name]):
                    raise ValueError(f'Parent projection requires explicit {name} string list')
            _idea_state(projected)

        provider = OpenAICompatibleProvider(api_key=provider_api_key(), endpoint=runtime.chat_completions_url,
            config=OpenAICompatibleConfig(timeout_seconds=runtime.request_timeout_seconds, max_attempts=runtime.max_retries + 1))
        response = provider.complete(ProviderRequest(
            operation="xlab.sure.parent_projection.v1", model=runtime.agent_model, structured_input=supplied,
            system_prompt="Describe ONLY the supplied existing implementation and model configuration, not a new idea. "
            "No external literature or citations. Return JSON with nonempty title, abstract, core_contribution, method, "
            "risks (string), components (nonempty list of name/description objects), tags and root_domains (string lists). "
            "Use actual module names and capabilities supported by the supplied source. No proposed changes or invented metrics.",
            user_prompt=json.dumps(supplied, ensure_ascii=False), output_kind="json",
            validation_profile='xlab.sure.parent_projection.response.v1', response_validator=validate_parent))
        value = response.json_value
        projection = {}
        if isinstance(value, dict):
            value, depth = unwrap_answer_object(value)
            if depth:
                projection = {"wrapper_depth": depth, "raw_response_digest": digest(response.json_value)}
        _idea_state(value)
        atomic_write_json(path, {"signature": signature, "idea": value, "source_digest": parent.get("solution_digest"),
                                 "trace": response.trace.to_dict(), "usage": asdict(response.usage),
                                 "response_projection": projection})
    _idea_state(value)
    return {**parent, "native_idea": value, "native_idea_digest": digest(value)}
