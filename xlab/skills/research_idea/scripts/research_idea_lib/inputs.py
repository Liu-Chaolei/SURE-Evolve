from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .common import resolve_project_path
from .config import RuntimeConfig, load_runtime_config


@dataclass
class IdeaRequest:
    survey_path: Path | None
    topic: str = ""
    mature_idea: str = ""
    refinement_scope: str = ""
    discussion: str = ""
    experiment_feedback: str = ""
    resume: bool = False
    raw_args: str = ""
    compatibility_warnings: list[str] = field(default_factory=list)
    research_policy: dict[str, Any] = field(default_factory=dict)
    task_context: dict[str, Any] = field(default_factory=dict)

    def to_json(self, cwd: Path) -> dict[str, Any]:
        return {
            "schema_version": "xlab.research_idea.request.v1",
            "survey_path": _relative(cwd, self.survey_path) if self.survey_path else None,
            "topic": self.topic,
            "mature_idea": self.mature_idea,
            "refinement_scope": self.refinement_scope,
            "discussion": self.discussion,
            "experiment_feedback": self.experiment_feedback,
            "resume": self.resume,
            "compatibility_warnings": self.compatibility_warnings,
            "research_policy": self.research_policy,
            "task_context": self.task_context,
        }


def _relative(cwd: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(cwd.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def normalize_key_value_tokens(tokens: list[str]) -> list[str]:
    value_flags = {"--survey", "--topic", "--mature-idea", "--refinement-scope",
                   "--discussion", "--experiment-feedback"}
    normalized: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            normalized.extend(tokens[index:])
            break
        if token in value_flags and index + 1 < len(tokens):
            # Attach the value so argparse preserves text beginning with '--'.
            normalized.append(token + "=" + tokens[index + 1])
            index += 2
            continue
        if not token.startswith("--") and "=" in token:
            key, value = token.split("=", 1)
            flag = "--" + key.strip().replace("_", "-")
            if flag in value_flags:
                normalized.append(flag + "=" + value)
            elif flag == "--resume":
                if value.strip().lower() in {"1", "true", "yes", "on"}:
                    normalized.append(flag)
            else:
                raise ValueError(f"Unsupported research_idea argument name: {key}")
        else:
            normalized.append(token)
        index += 1
    return normalized


def parse_request_args(raw_args: str, cwd: Path, runtime: RuntimeConfig | None = None) -> IdeaRequest:
    runtime = runtime or load_runtime_config()
    tokens = shlex.split(raw_args)
    if tokens and tokens[0] == "--":
        tokens = tokens[1:]
    tokens = normalize_key_value_tokens(tokens)
    parser = argparse.ArgumentParser(prog="/xlab generate-research-ideas", add_help=False)
    parser.add_argument("positional", nargs="*")
    parser.add_argument("--survey")
    parser.add_argument("--topic")
    parser.add_argument("--mature-idea")
    parser.add_argument("--refinement-scope")
    parser.add_argument("--discussion")
    parser.add_argument("--experiment-feedback")
    parser.add_argument("--resume", action="store_true")
    namespace, unknown = parser.parse_known_args(tokens)
    if unknown:
        raise ValueError(f"Unsupported research_idea arguments: {' '.join(unknown)}")
    survey_value = namespace.survey or (namespace.positional[0] if namespace.positional else "")
    if not survey_value:
        raise ValueError("Provide --survey <survey.json | literature_survey run dir | manifest.json>.")
    extra_positionals = namespace.positional[1:] if namespace.survey is None else namespace.positional
    if extra_positionals:
        namespace.topic = " ".join([part for part in [namespace.topic, *extra_positionals] if part]).strip()
    survey_path = resolve_project_path(cwd, survey_value)
    if survey_path is None:
        raise ValueError("Provide a valid --survey path.")
    warnings = compatibility_warnings(namespace, runtime, survey_path=survey_path)
    return IdeaRequest(
        survey_path=survey_path,
        topic=(namespace.topic or "").strip(),
        mature_idea=(namespace.mature_idea or "").strip(),
        refinement_scope=(namespace.refinement_scope or "").strip(),
        discussion=(namespace.discussion or "").strip(),
        experiment_feedback=(namespace.experiment_feedback or "").strip(),
        resume=bool(namespace.resume),
        raw_args=raw_args,
        compatibility_warnings=warnings,
    )


def compatibility_warnings(namespace: argparse.Namespace, runtime: RuntimeConfig, *, survey_path: Path) -> list[str]:
    warnings: list[str] = []
    if not survey_path.exists():
        warnings.append("Survey path does not exist.")
    if namespace.refinement_scope and not namespace.mature_idea:
        warnings.append("--refinement-scope is most effective with --mature-idea; without one it remains free-text generation guidance.")
    if not runtime.provider_available:
        warnings.append(
            "OPENAI_API_KEY is unavailable in the runtime environment; package-native research idea generation will remain incomplete until access is configured. Configure it as an environment secret, never as an argument."
        )
    return warnings


def request_from_json(value: dict[str, Any], cwd: Path) -> IdeaRequest:
    survey_raw = str(value.get("survey_path") or "")
    survey_path = resolve_project_path(cwd, survey_raw)
    if survey_path is None and value.get("research_policy", {}).get("evidence_mode") != "task_only":
        raise ValueError("request.json is missing survey_path.")
    return IdeaRequest(
        survey_path=survey_path,
        topic=str(value.get("topic") or ""),
        mature_idea=str(value.get("mature_idea") or ""),
        refinement_scope=str(value.get("refinement_scope") or ""),
        discussion=str(value.get("discussion") or ""),
        experiment_feedback=str(value.get("experiment_feedback") or ""),
        resume=bool(value.get("resume")),
        raw_args=str(value.get("raw_args") or ""),
        research_policy=dict(value.get("research_policy") or {}),
        task_context=dict(value.get("task_context") or {}),
        compatibility_warnings=[str(item) for item in value.get("compatibility_warnings", []) if item is not None]
        if isinstance(value.get("compatibility_warnings"), list)
        else [],
    )
