from __future__ import annotations

from pathlib import Path
import json

import pytest

from sure_eval.models.registry import ModelRegistry, model_artifact_digest
from sure_eval.results import file_sha256
from sure_eval.storage import StorageConfig


def _storage(tmp_path: Path) -> StorageConfig:
    return StorageConfig(
        repo_root=tmp_path / "repo",
        models_read_root=tmp_path / "nfs" / "models",
        results_read_root=tmp_path / "nfs" / "results",
        models_write_root=tmp_path / "repo" / "src" / "sure_eval" / "models",
        results_write_root=tmp_path / "repo" / "results",
    )


def test_model_registry_reads_only_the_published_models_root(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    published = storage.models_read_root / "Org__Model"
    published.mkdir(parents=True)
    (published / "config.yaml").write_text(
        "name: Org__Model\ntask: ASR\nserver:\n  command: [python, server.py]\n",
        encoding="utf-8",
    )
    staged = storage.models_write_root / "Local__Unpublished"
    staged.mkdir(parents=True)
    (staged / "config.yaml").write_text("name: Local__Unpublished\ntask: ASR\n", encoding="utf-8")

    registry = ModelRegistry(storage=storage)

    assert registry.list_models() == ["Org__Model"]
    assert registry.get_model("Local__Unpublished") is None


def test_model_registry_rejects_config_identity_mismatch(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    published = storage.models_read_root / "Org__Model"
    published.mkdir(parents=True)
    (published / "config.yaml").write_text("name: Other__Model\ntask: ASR\n", encoding="utf-8")

    registry = ModelRegistry(storage=storage)

    assert registry.list_models() == []
    assert "does not match" in registry.discovery_errors["Org__Model"]


def test_verified_model_requires_human_publication_record(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    published = storage.models_read_root / "Org__Model"
    published.mkdir(parents=True)
    (published / "config.yaml").write_text("name: Org__Model\ntask: ASR\n", encoding="utf-8")
    registry = ModelRegistry(storage=storage)

    with pytest.raises(ValueError, match="publication record"):
        registry.require_verified("Org__Model")

    artifact_files = [
        {
            "path": "config.yaml",
            "sha256": file_sha256(published / "config.yaml"),
            "size_bytes": (published / "config.yaml").stat().st_size,
        }
    ]
    artifact_manifest = published / "publication_artifacts.json"
    artifact_manifest.write_text(
        json.dumps(
            {
                "schema": "sure.eval.model_artifact_manifest.v1",
                "model_id": "Org__Model",
                "files": artifact_files,
            }
        ),
        encoding="utf-8",
    )
    (published / "publication.json").write_text(
        json.dumps(
            {
                "schema": "sure.eval.model_publication.v1",
                "status": "verified",
                "model_id": "Org__Model",
                "artifact_sha256": model_artifact_digest(artifact_files),
                "artifact_manifest_path": "publication_artifacts.json",
                "artifact_manifest_sha256": file_sha256(artifact_manifest),
                "verified_by": "human",
                "verified_at": "2026-08-08T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    refreshed = ModelRegistry(storage=storage)

    assert refreshed.require_verified("Org__Model").artifact_sha256 == model_artifact_digest(
        artifact_files
    )

    (published / "config.yaml").write_text("name: Org__Model\ntask: TAMPERED\n", encoding="utf-8")
    with pytest.raises(ValueError, match="(size|checksum) mismatch"):
        ModelRegistry(storage=storage).require_verified("Org__Model")
