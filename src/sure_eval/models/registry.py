"""Read verified model adapters from the single published NFS registry."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import yaml

from sure_eval.storage import StorageConfig, load_storage_config, validate_model_id


MODEL_ARTIFACT_MANIFEST_SCHEMA = "sure.eval.model_artifact_manifest.v1"
MODEL_PUBLICATION_SCHEMA = "sure.eval.model_publication.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_artifact_digest(files: list[dict[str, str]]) -> str:
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ModelInfo:
    """Canonical model metadata loaded from one published adapter directory."""

    name: str
    task: str
    path: Path
    config: dict[str, Any]

    @property
    def model_id(self) -> str:
        return self.name

    @property
    def provider_model_id(self) -> str:
        model = self.config.get("model")
        return str(model.get("id", "")) if isinstance(model, dict) else ""

    @property
    def description(self) -> str:
        return str(self.config.get("description", ""))

    @property
    def version(self) -> str:
        return str(self.config.get("version", ""))

    @property
    def languages(self) -> list[str]:
        value = self.config.get("languages") or []
        return [str(item) for item in value] if isinstance(value, list) else []

    @property
    def config_path(self) -> Path:
        return self.path / "config.yaml"

    @property
    def server_config(self) -> dict[str, Any]:
        value = self.config.get("server")
        return dict(value) if isinstance(value, dict) else {}

    @property
    def server_command(self) -> list[str]:
        value = self.server_config.get("command")
        if isinstance(value, list):
            return [str(part) for part in value]
        if isinstance(value, str) and value:
            return [value]
        return []

    @property
    def working_dir(self) -> Path:
        configured = self.server_config.get("working_dir") or "."
        return (self.path / str(configured)).resolve(strict=False)

    @property
    def env(self) -> dict[str, str]:
        value = self.server_config.get("env")
        if not isinstance(value, dict):
            return {}
        return {str(key): str(item) for key, item in value.items()}

    @property
    def timeout(self) -> int:
        try:
            return int(self.server_config.get("timeout", 300))
        except (TypeError, ValueError):
            return 300

    @property
    def protocols(self) -> dict[str, Any]:
        value = self.config.get("protocols")
        return dict(value) if isinstance(value, dict) else {}

    @property
    def publication(self) -> dict[str, Any]:
        path = self.path / "publication.json"
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return dict(value) if isinstance(value, dict) else {}

    @property
    def artifact_sha256(self) -> str:
        return str(self.publication.get("artifact_sha256", ""))

    @property
    def is_human_verified(self) -> bool:
        try:
            self.verify_publication()
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return True

    def verify_publication(self) -> None:
        publication = self.publication
        expected_publication_fields = {
            "schema",
            "status",
            "model_id",
            "artifact_sha256",
            "artifact_manifest_path",
            "artifact_manifest_sha256",
            "verified_by",
            "verified_at",
        }
        if set(publication) != expected_publication_fields:
            raise ValueError("model publication fields are incomplete or contain unknown values")
        if publication.get("schema") != MODEL_PUBLICATION_SCHEMA:
            raise ValueError("model publication schema is not v1")
        if publication.get("status") != "verified" or publication.get("model_id") != self.name:
            raise ValueError("model publication identity or status is invalid")
        if not _SHA256.fullmatch(self.artifact_sha256):
            raise ValueError("model publication artifact_sha256 is invalid")
        for field in ("verified_by", "verified_at"):
            if not isinstance(publication.get(field), str) or not publication[field].strip():
                raise ValueError(f"model publication {field} is required")
        manifest_name = publication.get("artifact_manifest_path")
        if manifest_name != "publication_artifacts.json":
            raise ValueError("model publication must reference publication_artifacts.json")
        manifest_path = (self.path / manifest_name).resolve(strict=True)
        if manifest_path.parent != self.path.resolve(strict=True):
            raise ValueError("model artifact manifest escapes the published model directory")
        if _file_sha256(manifest_path) != publication.get("artifact_manifest_sha256"):
            raise ValueError("model artifact manifest checksum mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if set(manifest) != {"schema", "model_id", "files"}:
            raise ValueError("model artifact manifest fields are incomplete or unknown")
        if manifest.get("schema") != MODEL_ARTIFACT_MANIFEST_SCHEMA:
            raise ValueError("model artifact manifest schema is not v1")
        if manifest.get("model_id") != self.name:
            raise ValueError("model artifact manifest identity mismatch")
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("model artifact manifest files must not be empty")
        for item in files:
            if not isinstance(item, dict):
                raise ValueError("model artifact manifest file entry must be a mapping")
            if set(item) != {"path", "sha256", "size_bytes"}:
                raise ValueError("model artifact file fields are incomplete or unknown")
            relative = Path(str(item.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("model artifact path must be relative")
            artifact = (self.path / relative).resolve(strict=True)
            try:
                artifact.relative_to(self.path.resolve(strict=True))
            except ValueError as exc:
                raise ValueError("model artifact escapes the published model directory") from exc
            if not artifact.is_file() or artifact.stat().st_size != item.get("size_bytes"):
                raise ValueError(f"model artifact size mismatch: {relative}")
            if _file_sha256(artifact) != item.get("sha256"):
                raise ValueError(f"model artifact checksum mismatch: {relative}")
        if model_artifact_digest(files) != self.artifact_sha256:
            raise ValueError("model artifact aggregate digest mismatch")

    @property
    def is_implemented(self) -> bool:
        return (self.path / "config.yaml").is_file() and (self.path / "server.py").is_file()

    def get_mcp_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": self.server_command,
            "working_dir": str(self.working_dir),
            "env": self.env,
            "timeout": self.timeout,
        }


class ModelRegistry:
    """Discover published adapters; local staging is deliberately excluded."""

    def __init__(self, *, storage: StorageConfig | None = None) -> None:
        self.storage = storage or load_storage_config()
        self.models_dir = self.storage.models_read_root
        self._models: dict[str, ModelInfo] = {}
        self._errors: dict[str, str] = {}
        self._discover_models()

    def _discover_models(self) -> None:
        if not self.models_dir.is_dir():
            return
        for item in sorted(self.models_dir.iterdir(), key=lambda path: path.name):
            if not item.is_dir() or item.name.startswith((".", "_")):
                continue
            try:
                model_id = validate_model_id(item.name)
                published_path = self.storage.published_model(model_id)
                config_path = published_path / "config.yaml"
                if not config_path.is_file():
                    continue
                config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                if not isinstance(config, dict):
                    raise TypeError("config.yaml must contain a mapping")
                configured_name = config.get("name")
                if configured_name and str(configured_name) != model_id:
                    raise ValueError(
                        f"config name {configured_name!r} does not match published model_id {model_id!r}"
                    )
                self._models[model_id] = ModelInfo(
                    name=model_id,
                    task=str(config.get("task", "unknown")),
                    path=published_path,
                    config=config,
                )
            except Exception as exc:
                self._errors[item.name] = str(exc)

    def list_models(self) -> list[str]:
        return sorted(self._models)

    def list_by_task(self, task: str) -> list[str]:
        expected = task.strip().lower().replace("-", "_")
        return sorted(
            name
            for name, info in self._models.items()
            if info.task.strip().lower().replace("-", "_") == expected
        )

    def get_model(self, model_id: str) -> ModelInfo | None:
        return self._models.get(validate_model_id(model_id))

    def require_verified(self, model_id: str) -> ModelInfo:
        model = self.get_model(model_id)
        if model is None:
            raise FileNotFoundError(f"published model is not available: {model_id}")
        try:
            model.verify_publication()
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"published model {model_id!r} has no valid verified publication record: {exc}"
            ) from exc
        return model

    @property
    def discovery_errors(self) -> dict[str, str]:
        return dict(self._errors)

    def get_mcp_configs(self) -> dict[str, Any]:
        return {
            model_id: info.get_mcp_config()
            for model_id, info in self._models.items()
            if info.is_implemented
        }
