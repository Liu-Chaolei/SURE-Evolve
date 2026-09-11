"""Research scope, independent of the ordinary search scheduling strategy."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping
import json

ARCHITECTURE_ONLY = "architecture_only"
ALL = "all"
STRUCTURE_ARGUMENTS = {
    "--num-encoder-layers",
    "--encoder-dim",
    "--feedforward-dim",
    "--encoder-unmasked-dim",
}
GUIDANCE = (
    "Generate exactly four distinct architecture-only research candidates. "
    "Every candidate must have candidate_type=arch, change_domains=[arch], and requires_training=true. "
    "Change model structure only. Keep optimizer, learning rate, loss/objective, augmentation, "
    "sampling, training data, seed, training budget, and inference/decoding settings fixed. "
    "Training the changed architecture with the fixed recipe is required execution, not a train-domain intervention. "
    "Do not propose training-strategy or inference optimizations, including as part of an architecture candidate. "
    "Use prior results as evidence, but do not continue out-of-scope training/inference changes."
)


def search_scope(sure: dict) -> str:
    scope = sure.get("search_scope", ARCHITECTURE_ONLY)
    if scope not in {ARCHITECTURE_ONLY, ALL}:
        raise ValueError(f"Unsupported sure.search_scope: {scope}")
    return scope


def execution_contract(context: dict, sure: dict) -> dict:
    contract = deepcopy({**context, **(sure.get("execution_contract") or {})})
    scope = search_scope(sure)
    contract["search_scope"] = scope
    if scope == ALL:
        return contract
    contract.update(
        candidate_types=["arch"],
        allowed_change_domains=["arch"],
        research_guidance=GUIDANCE,
        requires_training=True,
    )
    contract["constraints"] = [*contract.get("constraints", []), GUIDANCE]
    parameters = contract.get("candidate_parameters", {})
    if "architecture" in parameters:
        contract["candidate_parameters"] = {
            "architecture": {
                key: value
                for key, value in parameters["architecture"].items()
                if key != "checkpoint_activations"
            }
        }
    elif "train_args_json" in parameters:
        contract["candidate_parameters"] = {
            "train_args_json": "Only the listed structure_arguments; no training or decoding overrides."
        }
    task = sure.get("task") or {}
    contract["fixed_training"] = deepcopy(task.get("training", {}))
    contract["fixed_inference"] = deepcopy(task.get("inference", {}))
    return contract


def restricted_search(env: Mapping[str, str]) -> bool:
    return env.get("SURE_SEARCH_SCOPE") == ARCHITECTURE_ONLY and env.get(
        "SURE_CANDIDATE_PHASE", "search"
    ) in {"search", "improve"}


def validate_structure_args(arguments: list[str], env: Mapping[str, str]) -> None:
    allowed = {
        "--" + name.lstrip("-")
        for name in json.loads(
            env.get("SURE_ARCH_ARGUMENTS_JSON", json.dumps(sorted(STRUCTURE_ARGUMENTS)))
        )
    }
    if not arguments:
        raise ValueError("Architecture-only search requires structure arguments")
    index = 0
    while index < len(arguments):
        flag, equal, value = arguments[index].partition("=")
        if flag not in allowed:
            raise ValueError(
                f"Architecture-only search forbids training/inference argument: {flag}"
            )
        if equal:
            if not value:
                raise ValueError(f"Missing structure value: {flag}")
        else:
            index += 1
            if index >= len(arguments) or arguments[index].startswith("--"):
                raise ValueError(f"Missing structure value: {flag}")
        index += 1


def validate_candidate_parameters(
    action: str, parameters: dict, env: Mapping[str, str], *, frozen: bool = False
) -> None:
    if not restricted_search(env) or action == "baseline" or frozen:
        return
    if action != "arch":
        raise ValueError("Architecture-only search requires an arch candidate")
    if env.get("SURE_TASK_ADAPTER") == "asr.zipformer":
        if set(parameters) - {"train_args_json"}:
            raise ValueError(
                "Architecture-only search forbids training/inference parameter overrides"
            )
        arguments = parameters.get("train_args_json", [])
        if not isinstance(arguments, list) or any(
            not isinstance(value, str) for value in arguments
        ):
            raise ValueError("train_args_json must be a list of structure arguments")
        if not arguments:
            from ..runtime.icefall import recipe_changes

            if not recipe_changes():
                raise ValueError(
                    "Architecture-only search requires structure arguments or model source changes"
                )
        else:
            validate_structure_args(arguments, env)
    else:
        if (
            set(parameters) - {"architecture"}
            or not isinstance(parameters.get("architecture"), dict)
            or not parameters["architecture"]
            or "checkpoint_activations" in parameters["architecture"]
        ):
            raise ValueError(
                "Architecture-only search accepts nonempty architecture parameters only"
            )


def validate_search_entrypoint(code: str, env: Mapping[str, str]) -> None:
    if not restricted_search(env):
        return
    if env.get("SURE_TASK_ADAPTER") in {"tts.f5tts", "sd.diarizen"} and not any(
        marker in code for marker in ("SURE_TASK_WRAPPER", "run_task_candidate.py")
    ):
        raise ValueError(
            "Architecture-only search must use SURE_TASK_WRAPPER to preserve fixed training/inference settings"
        )
