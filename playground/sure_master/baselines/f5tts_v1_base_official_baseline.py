from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


WORKSPACE = Path.cwd()
ARTIFACTS_DIR = Path("artifacts")
WORKING_DIR = Path("working/f5tts_v1_base_official")
WAV_DIR = ARTIFACTS_DIR / "wavs"

OFFICIAL_MODEL_ID = "SWivid/F5-TTS/F5TTS_v1_Base"
OFFICIAL_MODEL_URL = "https://huggingface.co/SWivid/F5-TTS/tree/main/F5TTS_v1_Base"
DEFAULT_MODEL = "F5TTS_v1_Base"
DEFAULT_EVAL_DATA = "base_model/eval_data/prompts.jsonl"
DEFAULT_F5_ROOT = "base_model/root"
DEFAULT_IDEA_TEXT = "official F5-TTS v1 base inference baseline"


class CommandFailedError(RuntimeError):
    def __init__(self, name: str, return_code: int, log_path: Path):
        self.name = name
        self.return_code = return_code
        self.log_path = log_path
        super().__init__(
            f"{name} failed with exit code {return_code}; see {log_path}"
        )


class CommandTimedOutError(TimeoutError):
    def __init__(self, name: str, timeout: int, log_path: Path):
        self.name = name
        self.timeout = timeout
        self.log_path = log_path
        super().__init__(f"{name} timed out after {timeout}s; see {log_path}")


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def int_env(name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(float(os.environ.get(name, str(default))))
    except ValueError:
        value = default
    return max(minimum, value)


def workspace_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE / path


def ensure_workspace() -> None:
    for path in (ARTIFACTS_DIR, WAV_DIR, WORKING_DIR):
        path.mkdir(parents=True, exist_ok=True)


def require_existing_file(label: str, value: str) -> Path:
    path = workspace_path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    return path


def require_existing_dir(label: str, value: str) -> Path:
    path = workspace_path(value)
    if not path.is_dir():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    return path


def validate_local_model_file(label: str, value: str) -> None:
    if not value or value.startswith("hf://"):
        return
    require_existing_file(label, value)


def ensure_official_inputs() -> None:
    require_existing_dir("F5-TTS root", env_str("SURE_TTS_ROOT", DEFAULT_F5_ROOT))
    require_existing_file("F5-TTS eval prompts", env_str("SURE_TTS_EVAL_DATA", DEFAULT_EVAL_DATA))
    require_existing_file(
        "F5-TTS batch inference wrapper",
        env_str("SURE_TTS_BATCH_INFER_WRAPPER", ""),
    )
    validate_local_model_file("F5-TTS checkpoint", env_str("SURE_TTS_CKPT_FILE", ""))
    validate_local_model_file("F5-TTS vocab", env_str("SURE_TTS_VOCAB_FILE", ""))


def build_batch_infer_command() -> list[str]:
    timeout = int_env("SURE_RUN_TIMEOUT", 21600, minimum=60)
    command = [
        env_str("SURE_TTS_PYTHON", sys.executable),
        env_str("SURE_TTS_BATCH_INFER_WRAPPER", ""),
        "--f5-root",
        env_str("SURE_TTS_ROOT", DEFAULT_F5_ROOT),
        "--eval-data",
        env_str("SURE_TTS_EVAL_DATA", DEFAULT_EVAL_DATA),
        "--language",
        env_str("SURE_TTS_LANGUAGE", "en"),
        "--max-samples",
        env_str("SURE_TTS_MAX_SAMPLES", "4"),
        "--model",
        env_str("SURE_TTS_MODEL", DEFAULT_MODEL),
        "--ckpt-file",
        env_str("SURE_TTS_CKPT_FILE", ""),
        "--vocab-file",
        env_str("SURE_TTS_VOCAB_FILE", ""),
        "--nfe-step",
        env_str("SURE_TTS_NFE_STEP", "32"),
        "--cfg-strength",
        env_str("SURE_TTS_CFG_STRENGTH", "2.0"),
        "--sway-sampling-coef",
        env_str("SURE_TTS_SWAY_SAMPLING_COEF", "-1.0"),
        "--speed",
        env_str("SURE_TTS_SPEED", "1.0"),
        "--remove-silence",
        env_str("SURE_TTS_REMOVE_SILENCE", "0"),
        "--workers",
        env_str("SURE_TTS_INFER_WORKERS", "auto"),
        "--candidate-type",
        "inference",
        "--idea-text",
        DEFAULT_IDEA_TEXT,
        "--training-action",
        "no_train",
        "--arch-action",
        "no_arch",
        "--timeout",
        str(timeout),
    ]
    text_cleanup = env_str("SURE_TTS_TEXT_CLEANUP", "")
    if text_cleanup:
        command.extend(["--text-cleanup", text_cleanup])
    return command


def command_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def run_command(name: str, command: list[str], timeout: int | None = None) -> Path:
    log_path = WORKING_DIR / f"{name}.log"
    rendered = " ".join(shlex.quote(part) for part in command)
    print(f"[f5tts official baseline] running {name}: {rendered}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {rendered}\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=str(WORKSPACE),
            env=command_env(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise CommandTimedOutError(name, timeout, log_path)
    if return_code != 0:
        raise CommandFailedError(name, return_code, log_path)
    return log_path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} line {line_no} is not a JSON object")
            rows.append(value)
    return rows


def resolve_sample_audio(samples_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = samples_path.parent / path
    return path


def validate_outputs() -> list[dict[str, Any]]:
    samples_path = ARTIFACTS_DIR / "samples.jsonl"
    if not samples_path.is_file():
        raise FileNotFoundError(f"Missing required TTS samples artifact: {samples_path}")
    rows = read_jsonl(samples_path)
    if not rows:
        raise RuntimeError(f"{samples_path} contains no samples")
    for index, row in enumerate(rows, start=1):
        audio_value = str(row.get("prediction_audio") or "").strip()
        if not audio_value:
            raise RuntimeError(f"{samples_path} line {index} is missing prediction_audio")
        audio_path = resolve_sample_audio(samples_path, audio_value)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Missing generated audio for line {index}: {audio_path}")
        if audio_path.stat().st_size <= 1024:
            raise RuntimeError(f"Generated audio is implausibly small: {audio_path}")
    changes_path = ARTIFACTS_DIR / "candidate_changes.json"
    if not changes_path.is_file():
        raise FileNotFoundError(f"Missing candidate change record: {changes_path}")
    return rows


def write_official_baseline_record(
    *,
    command: list[str],
    sample_count: int,
    elapsed_seconds: float,
) -> None:
    payload = {
        "baseline_type": "official",
        "task_id": "tts_en_wer",
        "model_id": OFFICIAL_MODEL_ID,
        "source_url": OFFICIAL_MODEL_URL,
        "model": env_str("SURE_TTS_MODEL", DEFAULT_MODEL),
        "checkpoint_file": env_str("SURE_TTS_CKPT_FILE", ""),
        "vocab_file": env_str("SURE_TTS_VOCAB_FILE", ""),
        "f5_root": env_str("SURE_TTS_ROOT", DEFAULT_F5_ROOT),
        "eval_data": env_str("SURE_TTS_EVAL_DATA", DEFAULT_EVAL_DATA),
        "language": env_str("SURE_TTS_LANGUAGE", "en"),
        "sample_count": sample_count,
        "inference_config": {
            "nfe_step": env_str("SURE_TTS_NFE_STEP", "32"),
            "cfg_strength": env_str("SURE_TTS_CFG_STRENGTH", "2.0"),
            "sway_sampling_coef": env_str("SURE_TTS_SWAY_SAMPLING_COEF", "-1.0"),
            "speed": env_str("SURE_TTS_SPEED", "1.0"),
            "remove_silence": env_str("SURE_TTS_REMOVE_SILENCE", "0"),
            "workers": env_str("SURE_TTS_INFER_WORKERS", "auto"),
            "text_cleanup": env_str("SURE_TTS_TEXT_CLEANUP", "none"),
        },
        "wrapper": env_str("SURE_TTS_BATCH_INFER_WRAPPER", ""),
        "command": command,
        "samples_jsonl": "artifacts/samples.jsonl",
        "wav_dir": "artifacts/wavs",
        "elapsed_seconds": elapsed_seconds,
        "timestamp": time.time(),
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "official_baseline.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    start = time.time()
    ensure_workspace()
    ensure_official_inputs()
    command = build_batch_infer_command()
    run_timeout = int_env("SURE_RUN_TIMEOUT", 21600, minimum=60) + 300
    run_command("infer", command, timeout=run_timeout)
    rows = validate_outputs()
    write_official_baseline_record(
        command=command,
        sample_count=len(rows),
        elapsed_seconds=time.time() - start,
    )
    print(
        f"[f5tts official baseline] wrote {len(rows)} samples in "
        f"{time.time() - start:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
