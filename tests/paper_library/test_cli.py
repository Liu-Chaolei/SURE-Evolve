from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_library.cli import ExitCode, build_parser, main
from paper_library.errors import BuildStageError


def test_parser_exposes_expected_commands() -> None:
    parser = build_parser()
    for argv in (
        ["validate-config", "--config", "config.yaml"],
        ["scan", "--config", "config.yaml"],
        ["extract", "--config", "config.yaml", "candidates"],
        ["extract", "--config", "config.yaml", "evidence"],
        ["extract", "--config", "config.yaml", "cards"],
        ["build", "--config", "config.yaml"],
        ["validate", "--config", "config.yaml"],
        ["publish", "--config", "config.yaml"],
        ["index", "--config", "config.yaml"],
        ["query", "--config", "config.yaml", "--kind", "evidence", "WER"],
    ):
        assert parser.parse_args(argv).command == argv[0]


def test_validate_config_emits_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "paths:\n"
        "  source_root: papers\n"
        "  build_root: output\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc_info:
        main(["validate-config", "--config", str(config)])
    assert exc_info.value.code == ExitCode.SUCCESS
    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is True
    assert output["source_root"] == str((tmp_path / "papers").resolve())


def test_extract_candidates_uses_structural_fallback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "papers"
    source.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text(
        "paths:\n"
        f"  source_root: {source}\n"
        f"  build_root: {tmp_path / 'output'}\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc_info:
        main(["extract", "--config", str(config), "candidates"])
    assert exc_info.value.code == ExitCode.SUCCESS
    assert json.loads(capsys.readouterr().out)["processed"] == 0



def test_build_stage_error_uses_partial_failure_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(_args: object) -> int:
        raise BuildStageError("one extraction stage failed")

    parser = build_parser()
    args = parser.parse_args(["status", "--config", "config.yaml"])
    args.func = fail
    monkeypatch.setattr(parser, "parse_args", lambda _argv: args)
    monkeypatch.setattr("paper_library.cli.build_parser", lambda: parser)

    with pytest.raises(SystemExit) as exc_info:
        main([])

    assert exc_info.value.code == ExitCode.PARTIAL_FAILURE
    assert "extraction_failed" in capsys.readouterr().err


def test_invalid_config_uses_config_exit_code(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["validate-config", "--config", str(tmp_path / "missing.yaml")])
    assert exc_info.value.code == ExitCode.CONFIG_ERROR
    assert "config_invalid" in capsys.readouterr().err
