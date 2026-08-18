from __future__ import annotations

import pytest

from sure_eval.agent.model_tool_input import ModelToolInputError, parse_model_tool_input


def _payload() -> dict:
    return {
        "model_id": "Org__Model",
        "task_type": "asr",
        "deployment_type": "local",
        "repository": {"url": "https://example.invalid/model.git", "commit": "abc123"},
        "weights": {"source": "huggingface", "revision": "revision-1"},
        "environment": {"preferred_backend": "uv", "python_version": "3.11"},
    }


def test_model_tool_input_requires_pinned_identities() -> None:
    parsed = parse_model_tool_input(_payload())

    assert parsed.model_id == "Org__Model"
    assert parsed.repository_commit == "abc123"
    assert parsed.weights_revision == "revision-1"


@pytest.mark.parametrize("field", ("model_dir", "output_dir", "nfs_path", "models_root"))
def test_model_tool_input_rejects_path_fields(field: str) -> None:
    payload = _payload()
    payload[field] = "/tmp/bypass"

    with pytest.raises(ModelToolInputError, match="unsupported field"):
        parse_model_tool_input(payload)


def test_model_tool_input_rejects_unpinned_repository_or_weights() -> None:
    payload = _payload()
    payload["repository"]["commit"] = ""
    with pytest.raises(ModelToolInputError, match="repository.commit"):
        parse_model_tool_input(payload)

    payload = _payload()
    payload["weights"]["revision"] = ""
    with pytest.raises(ModelToolInputError, match="weights.revision"):
        parse_model_tool_input(payload)
