"""Agent contracts and legacy compatibility exports."""

from __future__ import annotations

from .main_flow_input import MainFlowInput, MainFlowInputError, parse_main_flow_input
from .model_tool_input import ModelToolInput, ModelToolInputError, parse_model_tool_input


__all__ = [
    "AutonomousEvaluator",
    "EvaluationResult",
    "MainFlowInput",
    "MainFlowInputError",
    "ModelToolInput",
    "ModelToolInputError",
    "parse_main_flow_input",
    "parse_model_tool_input",
]


def __getattr__(name: str):
    if name in {"AutonomousEvaluator", "EvaluationResult"}:
        from sure_eval.agent.evaluator import AutonomousEvaluator, EvaluationResult

        return {
            "AutonomousEvaluator": AutonomousEvaluator,
            "EvaluationResult": EvaluationResult,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
