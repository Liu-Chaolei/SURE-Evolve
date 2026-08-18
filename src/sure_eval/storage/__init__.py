"""Canonical read-only registry and repository-local staging paths."""

from .registry import (
    DECLARED_MODELS_READ_ROOT,
    DECLARED_RESULTS_READ_ROOT,
    DEFAULT_MODELS_READ_ROOT,
    DEFAULT_RESULTS_READ_ROOT,
    PathPolicyError,
    StorageConfig,
    load_storage_config,
    validate_model_id,
)

__all__ = [
    "DECLARED_MODELS_READ_ROOT",
    "DECLARED_RESULTS_READ_ROOT",
    "DEFAULT_MODELS_READ_ROOT",
    "DEFAULT_RESULTS_READ_ROOT",
    "PathPolicyError",
    "StorageConfig",
    "load_storage_config",
    "validate_model_id",
]
