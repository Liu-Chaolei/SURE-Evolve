from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

import run


FIXED_TIME = datetime(2026, 8, 8, 12, 34, 56)
TIMESTAMP = "20260808_123456"


def resolve(agent_name: str, run_dir_arg: str | None) -> Path:
    with patch.object(run, "datetime") as mock_datetime:
        mock_datetime.now.return_value = FIXED_TIME
        return run.resolve_run_dir(agent_name, run_dir_arg)


def test_default_run_dir_uses_configured_root_without_creating_it(tmp_path):
    run_root = tmp_path / "runs"
    expected = run_root / f"minimal_{TIMESTAMP}"

    with patch.object(run, "DEFAULT_RUN_ROOT", run_root):
        assert resolve("minimal", None) == expected
    assert not run_root.exists()


def test_default_run_dir_sanitizes_agent_name():
    assert resolve("my agent/test", None) == (
        run.DEFAULT_RUN_ROOT / f"my_agent_test_{TIMESTAMP}"
    )


def test_bare_run_name_uses_configured_root_and_timestamp():
    assert resolve("minimal", "my experiment") == (
        run.DEFAULT_RUN_ROOT / f"my_experiment_{TIMESTAMP}"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("./experiment", Path("experiment")),
        ("runs/experiment", Path("runs/experiment")),
        ("nested/experiment", Path("nested/experiment")),
        ("/tmp/experiment", Path("/tmp/experiment")),
    ],
)
def test_explicit_paths_are_used_exactly(value: str, expected: Path):
    assert resolve("minimal", value) == expected


@pytest.mark.parametrize(
    ("agent_name", "run_dir_arg", "message"),
    [
        ("///", None, "Agent name cannot be converted"),
        ("minimal", "   ", "--run-dir cannot be empty"),
    ],
)
def test_invalid_names_are_rejected(
    agent_name: str,
    run_dir_arg: str | None,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        resolve(agent_name, run_dir_arg)
