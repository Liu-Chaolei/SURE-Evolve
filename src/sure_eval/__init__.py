"""SURE-EVAL: Tool and Model Evaluation Framework."""

__version__ = "0.1.0"

__all__ = [
    "Config",
    "configure_logging",
    "get_logger",
    "AutonomousEvaluator",
    "EvaluationResult",
    "DatasetManager",
    "SUREEvaluator",
    "RPSManager",
]


def __getattr__(name: str):
    """Load top-level exports lazily so task-local modules can stay lightweight."""
    if name == "Config":
        from sure_eval.core.config import Config

        return Config
    if name in {"configure_logging", "get_logger"}:
        from sure_eval.core.logging import configure_logging, get_logger

        return {"configure_logging": configure_logging, "get_logger": get_logger}[name]
    if name in {"AutonomousEvaluator", "EvaluationResult"}:
        from sure_eval.agent import AutonomousEvaluator, EvaluationResult

        return {
            "AutonomousEvaluator": AutonomousEvaluator,
            "EvaluationResult": EvaluationResult,
        }[name]
    if name == "DatasetManager":
        from sure_eval.datasets import DatasetManager

        return DatasetManager
    if name in {"SUREEvaluator", "RPSManager"}:
        from sure_eval.evaluation import RPSManager, SUREEvaluator

        return {"SUREEvaluator": SUREEvaluator, "RPSManager": RPSManager}[name]
    raise AttributeError(f"module 'sure_eval' has no attribute {name!r}")
