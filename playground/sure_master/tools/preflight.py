"""Task-specific preflight before model/LLM work; no training is launched."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    document = yaml.safe_load(path.read_text())
    missing: set[str] = set()

    def expand(value):
        if isinstance(value, str):
            missing.update(name for name in re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", value)
                           if not os.environ.get(name))
            return os.path.expandvars(value)
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    document = expand(document)
    if missing:
        raise ValueError("Missing configuration environment variables: " + ", ".join(sorted(missing)))
    if not isinstance(document, dict):
        raise ValueError("Configuration must be a mapping")
    return document


def validate_xlab_survey(path: Path) -> None:
    root = path if path.is_dir() else path.parent
    if root.name == "artifacts":
        root = root.parent
    manifest = root / "manifest.json"
    report = root / "artifacts/survey_report.json"
    if not manifest.is_file() or not report.is_file():
        raise ValueError("XLab requires a completed literature_survey run with manifest and validation report")
    state, validation = json.loads(manifest.read_text()), json.loads(report.read_text())
    if state.get("status") != "success" or state.get("validation", {}).get("passed") is not True or validation.get("passed") is not True:
        raise ValueError("XLab source survey has not passed validation; finish it before starting model training")


def check_config(config: dict, *, check_model: bool = True) -> dict:
    from playground.sure_master.tasks import get_adapter
    from playground.sure_master.core.utils.task_cards import resolve_task_card
    from playground.sure_master.runtime.accelerator import runtime_environment

    sure = config["sure"]
    card = resolve_task_card(sure["task_cards_path"], sure["task_id"])
    adapter = get_adapter(card.canonical_task, sure.get("adapter"))
    report = adapter.preflight(sure)
    xlab = config.get("xlab", {})
    if xlab.get("enabled"):
        provider = xlab["idea_provider"]
        env = {**os.environ, **provider.get("environment", {})}
        if not env.get("OPENAI_API_KEY"):
            raise ValueError("XLab worker needs OPENAI_API_KEY via its deployment environment")
        survey = env.get("XLAB_SURE_SURVEY_PATH")
        if not survey or not Path(survey).exists():
            raise FileNotFoundError("XLab needs a real literature survey at XLAB_SURE_SURVEY_PATH")
        validate_xlab_survey(Path(survey))
        command = provider.get("command")
        if not isinstance(command, list) or not command:
            raise ValueError("XLab provider command must be an argv list")
        preflight_command = provider.get("preflight_command")
        if preflight_command:
            if not isinstance(preflight_command, list) or any(not isinstance(arg, str) or not arg for arg in preflight_command):
                raise ValueError("XLab provider preflight_command must be an argv list")
            checked = subprocess.run(preflight_command, env=env, text=True, capture_output=True, timeout=120)
            if checked.returncode:
                raise ValueError("XLab idea resources are not ready: " + checked.stdout[-4000:] + checked.stderr[-1000:])
    runtime = sure.get("runtime", {})
    if check_model:
        env = {**os.environ, **runtime_environment(runtime)}
        settings = sure.get("task", {})
        source = settings.get("resources", {}).get("source")
        source_paths = []
        if source:
            source_paths = [source, str(Path(source) / "src"), str(Path(source) / "pyannote-audio")]
        code = ("import sys, importlib, json; "
                + "sys.path[:0] = " + repr([str(Path(__file__).resolve().parents[3]), *source_paths]) + "; "
                + "from playground.sure_master.runtime.accelerator import check_accelerator; "
                + "result = check_accelerator(" + repr(runtime.get("accelerator", "cpu")) + "); "
                + "[importlib.import_module(m) for m in " + repr(list(adapter.modules)) + "]; print(json.dumps(result))")
        result = subprocess.run([runtime.get("python", sys.executable), "-c", code],
                                env=env, text=True, capture_output=True, timeout=120)
        if result.returncode:
            raise RuntimeError("Model runtime preflight failed:\n" + result.stderr[-4000:])
    return {"status": "ready", **report, "runtime": runtime.get("accelerator"), "model_checked": check_model}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()
    try:
        result = check_config(load_config(args.config), check_model=not args.skip_model)
    except Exception as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}))
        raise SystemExit(1)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
