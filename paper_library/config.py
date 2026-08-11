from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import Field, SecretStr, field_validator, model_validator

from paper_library.errors import ConfigurationError
from paper_library.schemas.common import StrictModel
from paper_library.utils.hashing import hash_canonical

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class PathConfig(StrictModel):
    source_root: Path
    build_root: Path

    @model_validator(mode="after")
    def validate_separation(self) -> "PathConfig":
        source = self.source_root.resolve()
        build = self.build_root.resolve()
        if source == build or source in build.parents or build in source.parents:
            raise ValueError("source_root and build_root must not contain one another")
        return self


class ParserConfig(StrictModel):
    backend: Literal["pymupdf"] = "pymupdf"
    min_text_characters: int = Field(default=100, ge=0)
    extract_tables: bool = True
    extract_figures: bool = True
    extract_equations: bool = True


class ExtractionConfig(StrictModel):
    provider: Literal["openai", "anthropic", "disabled"] = "disabled"
    model: str | None = None
    api_key: SecretStr | None = None
    base_url: str | None = None
    prompt_version: str = "1.0.0"
    temperature: float = Field(default=0, ge=0, le=2)
    max_retries: int = Field(default=2, ge=0, le=10)

    @model_validator(mode="after")
    def validate_provider(self) -> "ExtractionConfig":
        if self.provider != "disabled" and not self.model:
            raise ValueError("an extraction model is required when extraction is enabled")
        return self


class ValidationConfig(StrictModel):
    minimum_grounding_completeness: float = Field(default=0.9, ge=0, le=1)
    require_exact_grounding: bool = True
    publish_review_required: bool = False


class IndexConfig(StrictModel):
    lexical_enabled: bool = True
    vector_enabled: bool = False
    embedding_provider: str | None = None
    embedding_model: str | None = None

    @model_validator(mode="after")
    def validate_vector_settings(self) -> "IndexConfig":
        if self.vector_enabled and (not self.embedding_provider or not self.embedding_model):
            raise ValueError("vector indexing requires an embedding provider and model")
        return self


class RuntimeConfig(StrictModel):
    workers: int = Field(default=1, ge=1)
    fail_fast: bool = False


class PaperLibraryConfig(StrictModel):
    paths: PathConfig
    parser: ParserConfig = Field(default_factory=ParserConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    indexing: IndexConfig = Field(default_factory=IndexConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    config_path: Path | None = Field(default=None, exclude=True)

    @classmethod
    def load(cls, path: str | Path) -> "PaperLibraryConfig":
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise ConfigurationError(f"Configuration file does not exist: {config_path}")
        _load_nearest_dotenv(config_path.parent)
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            substituted = _substitute_env(raw)
            paths = substituted.get("paths")
            if isinstance(paths, dict):
                for key in ("source_root", "build_root"):
                    if key in paths:
                        candidate = Path(paths[key]).expanduser()
                        if not candidate.is_absolute():
                            paths[key] = str((config_path.parent / candidate).resolve())
            return cls.model_validate({**substituted, "config_path": config_path})
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ConfigurationError(f"Invalid configuration {config_path}: {exc}") from exc

    def content_hash(self) -> str:
        value = self.model_dump(mode="json", exclude={"config_path"})
        value["extraction"].pop("api_key", None)
        return hash_canonical(value)

    def ensure_output_path(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser().resolve()
        build_root = self.paths.build_root.resolve()
        if candidate != build_root and build_root not in candidate.parents:
            raise ConfigurationError(f"Output path escapes build_root: {candidate}")
        return candidate

    @field_validator("config_path")
    @classmethod
    def normalize_config_path(cls, value: Path | None) -> Path | None:
        return value.resolve() if value else None


def _load_nearest_dotenv(start: Path) -> None:
    for directory in (start, *start.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return


def _substitute_env(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise ConfigurationError(f"Undefined environment variable: {name}")
            return os.environ[name]

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, dict):
        return {key: _substitute_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute_env(item) for item in value]
    return value
