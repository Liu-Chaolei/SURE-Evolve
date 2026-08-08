from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


LOWER_IS_BETTER = {
    "cer",
    "wer",
    "mer",
    "der",
    "cpwer",
    "tts_cer",
    "tts_wer",
    "vc_cer",
    "vc_wer",
}

HIGHER_IS_BETTER = {"accuracy", "bleu"}


@dataclass(frozen=True)
class BaseModelProfile:
    """Base model/recipe contract that a SURE task must optimize from."""

    model_id: str
    model_type: str = ""
    framework: str = ""
    required_paths: dict[str, str] = field(default_factory=dict)
    source_paths: dict[str, str] = field(default_factory=dict)
    entrypoints: dict[str, str] = field(default_factory=dict)
    python: str | None = None
    usage_policy: str = "optional"
    prompt_guidance: str = ""

    @property
    def is_required(self) -> bool:
        return self.usage_policy.lower() == "required"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "framework": self.framework,
            "required_paths": self.required_paths,
            "source_paths": self.source_paths,
            "entrypoints": self.entrypoints,
            "python": self.python,
            "usage_policy": self.usage_policy,
            "prompt_guidance": self.prompt_guidance,
        }

    def to_prompt_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class SureTaskCard:
    """Structured SURE task/metric contract injected into agents and runner."""

    task_id: str
    canonical_task: str
    task_alias: str
    primary_metric: str
    language: str | None = None
    profile: str | None = None
    route: str = ""
    metric_direction: str = "lower"
    required_roles: list[str] = field(default_factory=list)
    optional_roles: list[str] = field(default_factory=list)
    artifact_contract: dict[str, str] = field(default_factory=dict)
    base_model: BaseModelProfile | None = None
    prompt_guidance: str = ""
    description: str = ""

    @property
    def is_lower_better(self) -> bool:
        return self.metric_direction.lower() == "lower"

    def to_prompt_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "canonical_task": self.canonical_task,
            "task_alias": self.task_alias,
            "language": self.language,
            "profile": self.profile,
            "primary_metric": self.primary_metric,
            "metric_direction": self.metric_direction,
            "route": self.route,
            "required_roles": self.required_roles,
            "optional_roles": self.optional_roles,
            "artifact_contract": self.artifact_contract,
            "base_model": self.base_model.to_dict() if self.base_model else None,
            "prompt_guidance": self.prompt_guidance,
            "description": self.description,
        }


def infer_metric_direction(metric: str) -> str:
    normalized = metric.strip().lower()
    if normalized in LOWER_IS_BETTER:
        return "lower"
    if normalized in HIGHER_IS_BETTER:
        return "higher"
    raise ValueError(f"Unknown metric direction for metric: {metric}")


def parse_base_model_profile(raw: dict[str, Any] | None) -> BaseModelProfile | None:
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ValueError("base_model must be a mapping")
    model_id = raw.get("model_id")
    if not model_id:
        raise ValueError("base_model.model_id is required when base_model is configured")
    return BaseModelProfile(
        model_id=str(model_id),
        model_type=str(raw.get("model_type", "")),
        framework=str(raw.get("framework", "")),
        required_paths=dict(raw.get("required_paths") or {}),
        source_paths=dict(raw.get("source_paths") or {}),
        entrypoints=dict(raw.get("entrypoints") or {}),
        python=raw.get("python"),
        usage_policy=str(raw.get("usage_policy", "optional")),
        prompt_guidance=str(raw.get("prompt_guidance", "")),
    )


def merge_base_model_profile(
    base: BaseModelProfile | None,
    override: dict[str, Any] | None,
) -> BaseModelProfile | None:
    if not override:
        return base
    if not isinstance(override, dict):
        raise ValueError("base model override must be a mapping")

    merged: dict[str, Any] = base.to_dict() if base else {}
    for key, value in override.items():
        if key in {"required_paths", "source_paths", "entrypoints"}:
            current = dict(merged.get(key) or {})
            current.update(value or {})
            merged[key] = current
        else:
            merged[key] = value
    return parse_base_model_profile(merged)


def validate_base_model_profile(profile: BaseModelProfile | None, task_id: str) -> None:
    if profile is None:
        return
    missing_sources = [
        name
        for name in profile.required_paths
        if name not in profile.source_paths or not str(profile.source_paths[name]).strip()
    ]
    if profile.is_required and missing_sources:
        missing = ", ".join(sorted(missing_sources))
        raise ValueError(
            f"SURE task {task_id!r} requires base_model source_paths for: {missing}"
        )
    missing_targets = [
        name
        for name, target in profile.required_paths.items()
        if not str(target).strip()
    ]
    if missing_targets:
        missing = ", ".join(sorted(missing_targets))
        raise ValueError(
            f"SURE task {task_id!r} has empty base_model required_paths for: {missing}"
        )


def load_task_cards(path: str | Path) -> dict[str, SureTaskCard]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cards = data.get("tasks", data)
    if not isinstance(cards, dict):
        raise ValueError("SURE task card file must contain a mapping or a top-level 'tasks' mapping")

    result: dict[str, SureTaskCard] = {}
    for task_id, raw in cards.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Task card {task_id!r} must be a mapping")
        metric = str(raw["primary_metric"])
        direction = raw.get("metric_direction") or infer_metric_direction(metric)
        card = SureTaskCard(
            task_id=str(raw.get("task_id") or task_id),
            canonical_task=str(raw["canonical_task"]),
            task_alias=str(raw.get("task_alias") or raw["canonical_task"]),
            language=raw.get("language"),
            profile=raw.get("profile"),
            primary_metric=metric,
            metric_direction=str(direction),
            route=str(raw.get("route", "")),
            required_roles=list(raw.get("required_roles") or []),
            optional_roles=list(raw.get("optional_roles") or []),
            artifact_contract=dict(raw.get("artifact_contract") or {}),
            base_model=parse_base_model_profile(raw.get("base_model")),
            prompt_guidance=str(raw.get("prompt_guidance", "")),
            description=str(raw.get("description", "")),
        )
        result[card.task_id] = card
    return result


def resolve_task_card(task_cards_path: str | Path, task_id: str) -> SureTaskCard:
    cards = load_task_cards(task_cards_path)
    if task_id not in cards:
        available = ", ".join(sorted(cards))
        raise KeyError(f"Unknown SURE task_id {task_id!r}. Available task cards: {available}")
    return cards[task_id]
