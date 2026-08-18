"""Strict structured input contract for the model tool agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sure_eval.storage import validate_model_id


MODEL_TOOL_INPUT_SCHEMA = "sure.eval.model_tool_input.v2"
ALLOWED_DEPLOYMENT_TYPES = ("local", "api")


class ModelToolInputError(ValueError):
    """Raised when model onboarding input is incomplete or path-bearing."""


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ModelToolInputError(f"{location} must be a mapping")
    return value


def _only_keys(value: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ModelToolInputError(f"unsupported field(s) at {location}: {', '.join(unknown)}")


def _required_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelToolInputError(f"{location} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class ModelToolInput:
    model_id: str
    task_type: str
    deployment_type: str
    repository_url: str
    repository_commit: str
    weights_source: str
    weights_revision: str
    preferred_backend: str
    python_version: str
    schema: str = MODEL_TOOL_INPUT_SCHEMA


def parse_model_tool_input(payload: dict[str, Any]) -> ModelToolInput:
    if not isinstance(payload, dict):
        raise ModelToolInputError("MODEL_INPUT must be a mapping")
    _only_keys(
        payload,
        {
            "schema",
            "model_id",
            "task_type",
            "deployment_type",
            "repository",
            "weights",
            "environment",
        },
        "root",
    )
    if payload.get("schema", MODEL_TOOL_INPUT_SCHEMA) != MODEL_TOOL_INPUT_SCHEMA:
        raise ModelToolInputError("unsupported MODEL_INPUT schema")
    try:
        model_id = validate_model_id(str(payload.get("model_id", "")))
    except ValueError as exc:
        raise ModelToolInputError(str(exc)) from exc
    task_type = _required_string(payload.get("task_type"), "task_type").lower()
    deployment_type = _required_string(
        payload.get("deployment_type"), "deployment_type"
    ).lower()
    if deployment_type not in ALLOWED_DEPLOYMENT_TYPES:
        raise ModelToolInputError(
            f"deployment_type must be one of {', '.join(ALLOWED_DEPLOYMENT_TYPES)}"
        )

    repository = _mapping(payload.get("repository"), "repository")
    _only_keys(repository, {"url", "commit"}, "repository")
    weights = _mapping(payload.get("weights"), "weights")
    _only_keys(weights, {"source", "revision"}, "weights")
    environment = _mapping(payload.get("environment"), "environment")
    _only_keys(environment, {"preferred_backend", "python_version"}, "environment")
    return ModelToolInput(
        model_id=model_id,
        task_type=task_type,
        deployment_type=deployment_type,
        repository_url=_required_string(repository.get("url"), "repository.url"),
        repository_commit=_required_string(repository.get("commit"), "repository.commit"),
        weights_source=_required_string(weights.get("source"), "weights.source"),
        weights_revision=_required_string(weights.get("revision"), "weights.revision"),
        preferred_backend=_required_string(
            environment.get("preferred_backend"), "environment.preferred_backend"
        ),
        python_version=_required_string(
            environment.get("python_version"), "environment.python_version"
        ),
    )
