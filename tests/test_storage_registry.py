from __future__ import annotations

from pathlib import Path

import pytest

from sure_eval.storage import (
    DEFAULT_MODELS_READ_ROOT,
    DEFAULT_RESULTS_READ_ROOT,
    PathPolicyError,
    StorageConfig,
    load_storage_config,
    validate_model_id,
)


def _storage(tmp_path: Path) -> StorageConfig:
    return StorageConfig(
        repo_root=tmp_path / "repo",
        models_read_root=tmp_path / "nfs" / "models",
        results_read_root=tmp_path / "nfs" / "results",
        models_write_root=tmp_path / "repo" / "src" / "sure_eval" / "models",
        results_write_root=tmp_path / "repo" / "results",
    )


def test_repository_storage_config_has_the_only_published_roots() -> None:
    storage = load_storage_config()

    assert storage.models_read_root == DEFAULT_MODELS_READ_ROOT
    assert storage.results_read_root == DEFAULT_RESULTS_READ_ROOT
    assert storage.models_write_root == storage.repo_root / "src/sure_eval/models"
    assert storage.results_write_root == storage.repo_root / "results"


@pytest.mark.parametrize("value", ("", ".", "..", "a/b", "../model", "/abs", "a b"))
def test_model_id_rejects_path_like_values(value: str) -> None:
    with pytest.raises(PathPolicyError):
        validate_model_id(value)


def test_registry_reads_and_staging_writes_are_capability_separated(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    published_model = storage.models_read_root / "Org__Model"
    published_model.mkdir(parents=True)

    assert storage.published_model("Org__Model") == published_model.resolve()
    staged = storage.staged_results("Org__Model", create=True)
    assert staged == (storage.results_write_root / "Org__Model").resolve()

    with pytest.raises(PathPolicyError, match="never write"):
        storage.assert_staging_write_path(published_model / "config.yaml")
    with pytest.raises(PathPolicyError, match="NFS registries"):
        storage.assert_registry_read_path(staged / "report.json")


def test_staging_rejects_symbolic_link_escape(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    storage.models_write_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (storage.models_write_root / "Org__Model").symlink_to(outside, target_is_directory=True)

    with pytest.raises(PathPolicyError):
        storage.staged_model("Org__Model", create=True)


def test_loading_an_override_registry_root_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "storage.yaml"
    config.write_text(
        """schema: sure.eval.storage.v1
registry:
  models_read_root: /tmp/models
  results_read_root: /tmp/results
staging:
  models_write_root: src/sure_eval/models
  results_write_root: results
""",
        encoding="utf-8",
    )

    with pytest.raises(PathPolicyError, match="fixed"):
        load_storage_config(config)
