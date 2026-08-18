from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_FLOW_ROOT = REPO_ROOT / "docs/agents/main_flow_agent"


def test_main_flow_declares_standard_default_and_explicit_strict_policy() -> None:
    text = (MAIN_FLOW_ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert "[SYSTEM_CONSTRAINT: MAIN_FLOW_PROTOCOL_SELECTION]" in text
    assert "`standard_system` is the default" in text
    assert "`strict_core` is selected only when the user explicitly requests it" in text


def test_global_protocol_catalog_has_exactly_two_ids() -> None:
    payload = yaml.safe_load((REPO_ROOT / "config/protocols.yaml").read_text(encoding="utf-8"))

    assert list(payload["protocols"]) == ["standard_system", "strict_core"]
    assert payload["default_protocol_id"] == "standard_system"
    assert payload["protocols"]["standard_system"]["is_default"] is True
    assert payload["protocols"]["strict_core"]["is_default"] is False


def test_structured_templates_default_to_standard_system() -> None:
    templates = MAIN_FLOW_ROOT / "templates"
    for filename in (
        "main_agent_plan.json",
        "main_agent_script_routing.json",
        "main_agent_run_report.json",
        "model_eval_manifest.json",
    ):
        payload = json.loads((templates / filename).read_text(encoding="utf-8"))
        protocol = payload.get("protocol") or {}
        assert payload.get("protocol_id", protocol.get("id")) == "standard_system"


@pytest.mark.parametrize("protocol_id", ("standard_system", "strict_core"))
def test_shell_template_accepts_both_protocols_before_later_input_gate(
    tmp_path: Path,
    protocol_id: str,
) -> None:
    env = os.environ.copy()
    env.update(
        {
            "REPO_ROOT": str(REPO_ROOT),
            "MODEL_ID": "Org__Model",
            "ACTION": "reuse_result",
            "EVALUATION_ID": "unused",
            "EVALUATION_INPUT_MANIFEST": "unused",
            "PROTOCOL_ID": protocol_id,
        }
    )
    completed = subprocess.run(
        ["bash", str(MAIN_FLOW_ROOT / "templates/run_single_model_single_dataset.sh")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0


def test_shell_template_rejects_custom_protocol() -> None:
    env = os.environ.copy()
    env.update(
        {
            "REPO_ROOT": str(REPO_ROOT),
            "MODEL_ID": "Org__Model",
            "ACTION": "reuse_result",
            "EVALUATION_ID": "unused",
            "EVALUATION_INPUT_MANIFEST": "unused",
            "PROTOCOL_ID": "custom",
        }
    )
    completed = subprocess.run(
        ["bash", str(MAIN_FLOW_ROOT / "templates/run_single_model_single_dataset.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "invalid protocol_id" in completed.stderr
