from __future__ import annotations

import pytest

from sure_eval.agent.main_flow_input import MainFlowInputError, parse_main_flow_input


def _payload() -> dict:
    return {
        "target": {"model_id": "Org__Model"},
        "datasets": [{"name": "aishell1", "version": "v1.0.2", "split": "test"}],
    }


def test_defaults_to_standard_system_and_auto() -> None:
    parsed = parse_main_flow_input(_payload())

    assert parsed.protocol_id == "standard_system"
    assert parsed.execution_mode == "auto"
    assert parsed.datasets[0].id == "aishell1__v1.0.2"


def test_explicit_strict_core_and_retest_are_allowed() -> None:
    payload = _payload()
    payload["inference"] = {"protocol_id": "strict_core"}
    payload["execution"] = {"mode": "retest"}

    parsed = parse_main_flow_input(payload)

    assert parsed.protocol_id == "strict_core"
    assert parsed.execution_mode == "retest"


@pytest.mark.parametrize("field", ("model_dir", "results_output_dir", "output_dir"))
def test_legacy_path_fields_are_rejected(field: str) -> None:
    payload = _payload()
    payload["target"][field] = "/tmp/bypass"

    with pytest.raises(MainFlowInputError, match="unsupported field"):
        parse_main_flow_input(payload)


def test_task_suffixed_dataset_identity_is_rejected() -> None:
    payload = _payload()
    payload["datasets"][0]["name"] = "aishell1__v1.0.2__asr"

    with pytest.raises(MainFlowInputError, match="source dataset name"):
        parse_main_flow_input(payload)


def test_custom_protocol_and_execution_mode_are_rejected() -> None:
    payload = _payload()
    payload["inference"] = {"protocol_id": "custom"}

    with pytest.raises(MainFlowInputError, match="protocol_id"):
        parse_main_flow_input(payload)

    payload = _payload()
    payload["execution"] = {"mode": "maybe"}
    with pytest.raises(MainFlowInputError, match="execution.mode"):
        parse_main_flow_input(payload)
