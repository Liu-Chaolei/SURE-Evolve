#!/usr/bin/env python3
"""Guarded Zipformer ASR candidate runner for SURE Master.

Generated candidates call this wrapper instead of directly spawning icefall
train.py/decode.py. The wrapper keeps the process, duration, checkpoint, split,
and hyp-format protocol fixed while still allowing candidate-controlled
Zipformer train/decode argument changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from playground.sure_master.baselines import (  # noqa: E402
    zipformer_large_cr_ctc_rnnt_baseline as baseline,
)
from playground.sure_master.core.utils.candidate_changes import write_candidate_changes  # noqa: E402
from playground.sure_master.core.utils.candidate_type import (  # noqa: E402
    ARCH,
    FINE_TUNE,
    INFERENCE,
    normalize_candidate_type,
)


STRUCTURE_ARGS = {
    "--num-encoder-layers",
    "--encoder-dim",
    "--feedforward-dim",
    "--encoder-unmasked-dim",
}
TRAIN_ONLY_ARGS = {
    "--ctc-loss-scale",
    "--cr-loss-scale",
    "--enable-spec-aug",
    "--time-mask-ratio",
}
FORBIDDEN_TRAIN_ARGS = {"--lang-dir"}
DEFAULT_MODEL_DIR = Path("models/zipformer_candidate")
DEFAULT_WORKING_DIR = Path("working/zipformer_candidate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=["train_decode", "decode_only"],
        default="train_decode",
    )
    parser.add_argument(
        "--candidate-type",
        choices=[INFERENCE, FINE_TUNE, ARCH, "training", "finetune", "fine-tune"],
        required=True,
    )
    parser.add_argument("--idea-text", default="")
    parser.add_argument("--train-args-json", default="[]")
    parser.add_argument("--decode-args-json", default="[]")
    parser.add_argument("--changed-fields-json", default="[]")
    parser.add_argument("--exp-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--working-dir", default=str(DEFAULT_WORKING_DIR))
    parser.add_argument("--train-epochs", default=os.environ.get("SURE_MAX_TRAIN_EPOCHS", "1"))
    parser.add_argument("--train-timeout", default=os.environ.get("SURE_BASELINE_TRAIN_TIMEOUT", "43200"))
    parser.add_argument("--decode-timeout", default=os.environ.get("SURE_BASELINE_DECODE_TIMEOUT", "21600"))
    parser.add_argument("--decode-epoch", default="")
    parser.add_argument("--decode-avg", default="1")
    parser.add_argument("--use-averaged-model", default="0")
    parser.add_argument("--decode-method", default="modified_beam_search")
    parser.add_argument("--decode-max-duration", default=os.environ.get("SURE_BASELINE_DECODE_MAX_DURATION", "300"))
    return parser.parse_args()


def parse_list(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [str(item) for item in shlex.split(text)]
    if not isinstance(parsed, list):
        raise ValueError("argument JSON must be a list")
    return [str(item) for item in parsed]


def parse_int(value: Any, default: int) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def arg_names(args: list[str]) -> set[str]:
    return {item.split("=", 1)[0] for item in args if item.startswith("--")}


def merge_cli_args(default_args: list[str], extra_args: list[str]) -> list[str]:
    override_names = arg_names(extra_args)
    merged: list[str] = []
    index = 0
    while index < len(default_args):
        item = default_args[index]
        if item in override_names:
            index += 1
            if index < len(default_args) and not default_args[index].startswith("--"):
                index += 1
            continue
        merged.append(item)
        index += 1
    merged.extend(extra_args)
    return merged


def args_to_config(args: list[str]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    index = 0
    while index < len(args):
        key = args[index]
        if not key.startswith("--"):
            index += 1
            continue
        if "=" in key:
            name, value = key[2:].split("=", 1)
        else:
            name = key[2:]
            value: Any = True
            if index + 1 < len(args) and not args[index + 1].startswith("--"):
                value = args[index + 1]
                index += 1
        name = name.replace("-", "_")
        config[name] = value
        index += 1
    return config


def last_arg_value(args: list[str], option: str, default: str | Path | None = None) -> str | None:
    value = str(default) if default is not None else None
    inline_prefix = option + "="
    index = 0
    while index < len(args):
        item = args[index]
        if item == option:
            if index + 1 >= len(args) or args[index + 1].startswith("--"):
                raise ValueError(f"{option} requires a value")
            value = args[index + 1]
            index += 2
            continue
        if item.startswith(inline_prefix):
            inline_value = item[len(inline_prefix):]
            if not inline_value:
                raise ValueError(f"{option} requires a value")
            value = inline_value
        index += 1
    return value


def remove_cli_arg(args: list[str], option: str) -> list[str]:
    cleaned: list[str] = []
    inline_prefix = option + "="
    index = 0
    while index < len(args):
        item = args[index]
        if item == option:
            index += 1
            if index < len(args) and not args[index].startswith("--"):
                index += 1
            continue
        if item.startswith(inline_prefix):
            index += 1
            continue
        cleaned.append(item)
        index += 1
    return cleaned


def force_cli_arg(args: list[str], option: str, value: str) -> list[str]:
    return [*remove_cli_arg(args, option), option, value]


def selected_structure_args(args: list[str]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for option in sorted(STRUCTURE_ARGS):
        value = last_arg_value(args, option)
        if value is not None:
            selected[option] = value
    return selected


def validate_requested_decode_structure_args(
    *,
    train_structure: dict[str, str],
    decode_extra_args: list[str],
) -> None:
    decode_names = arg_names(decode_extra_args)
    for option in sorted(STRUCTURE_ARGS.intersection(decode_names)):
        decode_value = last_arg_value(decode_extra_args, option)
        train_value = train_structure.get(option)
        if train_value is None:
            continue
        if decode_value != train_value:
            raise ValueError(
                "Zipformer train/decode structure mismatch: "
                f"training uses {option}={train_value}, "
                f"but decoding explicitly requested {option}={decode_value}. "
                "train_decode candidates must decode with the same Zipformer "
                "structure used for training."
            )


def force_decode_structure_from_train(
    *,
    final_train_args: list[str],
    final_decode_args: list[str],
    decode_extra_args: list[str],
) -> list[str]:
    train_structure = selected_structure_args(final_train_args)
    validate_requested_decode_structure_args(
        train_structure=train_structure,
        decode_extra_args=decode_extra_args,
    )
    resolved_decode_args = final_decode_args
    for option, value in train_structure.items():
        resolved_decode_args = force_cli_arg(resolved_decode_args, option, value)
    return resolved_decode_args


def resolve_workspace_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bpe_fingerprint(path: str | Path, *, label: str) -> tuple[Path, str]:
    resolved = resolve_workspace_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing Zipformer BPE model for {label}: {resolved}")
    return resolved, file_sha256(resolved)


def selected_train_bpe_model(final_train_args: list[str]) -> Path:
    default_bpe_model = baseline.default_bpe_model()
    value = last_arg_value(
        ["--bpe-model", str(default_bpe_model), *final_train_args],
        "--bpe-model",
        default_bpe_model,
    )
    if value is None:
        raise ValueError("Zipformer training requires --bpe-model")
    return Path(value)


def selected_requested_decode_bpe_model(final_decode_args: list[str]) -> Path | None:
    value = last_arg_value(final_decode_args, "--bpe-model")
    return Path(value) if value is not None else None


def selected_decode_bpe_model(
    *,
    action: str,
    final_train_args: list[str],
) -> Path:
    if action == "train_decode":
        return selected_train_bpe_model(final_train_args)
    return baseline.decode_bpe_model_path()


def validate_train_decode_bpe_models(train_bpe_model: Path, decode_bpe_model: Path) -> None:
    train_path, train_sha256 = bpe_fingerprint(train_bpe_model, label="candidate training")
    decode_path, decode_sha256 = bpe_fingerprint(decode_bpe_model, label="candidate decoding")
    if train_sha256 != decode_sha256:
        raise RuntimeError(
            "Zipformer train/decode BPE mismatch: "
            f"training uses {train_path} (sha256={train_sha256}), "
            f"but decoding uses {decode_path} (sha256={decode_sha256}). "
            "train_decode candidates must decode with the same BPE model used for training."
        )


def validate_requested_decode_bpe_model(
    *,
    train_bpe_model: Path,
    final_decode_args: list[str],
) -> None:
    requested_decode_bpe = selected_requested_decode_bpe_model(final_decode_args)
    if requested_decode_bpe is None:
        return
    validate_train_decode_bpe_models(train_bpe_model, requested_decode_bpe)


def bpe_sha256_if_available(path: Path | None) -> str:
    if path is None:
        return ""
    resolved = resolve_workspace_path(path)
    if not resolved.is_file():
        return ""
    return file_sha256(resolved)


def validate_candidate_args(
    *,
    candidate_type: str,
    action: str,
    train_extra_args: list[str],
    decode_extra_args: list[str],
) -> None:
    train_names = arg_names(train_extra_args)
    decode_names = arg_names(decode_extra_args)
    if candidate_type == INFERENCE and action != "decode_only":
        raise ValueError("inference candidates must use --action decode_only")
    if candidate_type == FINE_TUNE and train_names.intersection(STRUCTURE_ARGS):
        raise ValueError("fine_tune candidates must not change Zipformer structure args")
    if candidate_type == ARCH and not train_names.intersection(STRUCTURE_ARGS):
        raise ValueError(
            "arch candidates must change at least one Zipformer structure arg: "
            + ", ".join(sorted(STRUCTURE_ARGS))
        )
    bad_train = sorted(train_names.intersection(FORBIDDEN_TRAIN_ARGS))
    if bad_train:
        raise ValueError("train.py does not accept these wrapper train args: " + ", ".join(bad_train))
    bad_decode = sorted(decode_names.intersection(TRAIN_ONLY_ARGS))
    if bad_decode:
        raise ValueError("decode.py must not receive training-only args: " + ", ".join(bad_decode))


def configure_baseline_paths(exp_dir: Path, working_dir: Path) -> None:
    baseline.MODELS_DIR = exp_dir
    baseline.WORKING_DIR = working_dir


def current_world_size() -> int:
    raw = os.environ.get("ASR_WORLD_SIZE") or os.environ.get("SURE_BASELINE_WORLD_SIZE")
    requested = parse_int(raw, 0)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        visible_count = len([item for item in visible.split(",") if item.strip()])
        if visible_count <= 0:
            return 1
        return max(1, min(requested or visible_count, visible_count))
    return max(1, requested or 1)


def selected_train_duration(final_train_args: list[str]) -> int:
    configured_default = parse_int(
        os.environ.get("SURE_BASELINE_TRAIN_MAX_DURATION"),
        baseline.BASELINE_TRAIN_MAX_DURATION,
    )
    duration = baseline.max_duration(
        configured_default,
        baseline.BASELINE_TRAIN_MAX_DURATION,
        final_train_args,
    )
    floor = max(
        1,
        parse_int(os.environ.get("SURE_TRAIN_DURATION_MIN"), 100),
        parse_int(os.environ.get("SURE_DURATION_AUTOTUNE_MIN"), 100),
    )
    if duration < floor:
        raise RuntimeError(
            f"selected max-duration {duration} is below configured floor {floor}"
        )
    return duration


def _link_or_copy_checkpoint(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        try:
            if target.resolve() == source.resolve():
                return
        except OSError:
            pass
        target.unlink()
    try:
        os.link(source, target)
        return
    except OSError:
        pass
    try:
        target.symlink_to(source)
        return
    except OSError:
        pass
    shutil.copy2(source, target)


def staged_resume_config(exp_dir: Path, requested_train_epochs: int) -> dict[str, Any]:
    target_epoch = max(
        1,
        parse_int(
            os.environ.get("SURE_STAGED_TARGET_EPOCH"),
            requested_train_epochs,
        ),
    )
    checkpoint = str(os.environ.get("SURE_STAGED_RESUME_CHECKPOINT") or "").strip()
    resume_enabled = baseline.truthy(
        os.environ.get("SURE_STAGED_RESUME_ENABLED"),
        default=bool(checkpoint),
    )
    if not resume_enabled and not checkpoint:
        return {
            "enabled": False,
            "start_epoch": 1,
            "target_epoch": target_epoch,
            "resume_epoch": None,
            "resume_checkpoint": "",
            "local_checkpoint": "",
            "source_rung": "",
        }
    if not checkpoint:
        raise RuntimeError("SURE_STAGED_RESUME_ENABLED is set but SURE_STAGED_RESUME_CHECKPOINT is empty")
    source = Path(checkpoint).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Missing staged resume checkpoint: {source}")
    resume_epoch = parse_int(os.environ.get("SURE_STAGED_RESUME_EPOCH"), 0)
    if resume_epoch <= 0:
        match = re.search(r"epoch[-_](\d+)\.pt$", str(source))
        if match:
            resume_epoch = int(match.group(1))
    if resume_epoch <= 0:
        raise RuntimeError(f"Could not infer staged resume epoch from {source}")
    if target_epoch <= resume_epoch:
        raise RuntimeError(
            "SURE_STAGED_TARGET_EPOCH must be greater than SURE_STAGED_RESUME_EPOCH "
            f"for staged resume; got target={target_epoch}, resume={resume_epoch}"
        )
    target = exp_dir / f"epoch-{resume_epoch}.pt"
    _link_or_copy_checkpoint(source, target)
    return {
        "enabled": True,
        "start_epoch": resume_epoch + 1,
        "target_epoch": target_epoch,
        "resume_epoch": resume_epoch,
        "resume_checkpoint": str(source),
        "local_checkpoint": str(target),
        "source_rung": str(os.environ.get("SURE_STAGED_RESUME_SOURCE_RUNG") or ""),
    }


def ensure_staged_resume_checkpoint(staged_resume: dict[str, Any] | None) -> None:
    if not staged_resume or not staged_resume.get("enabled"):
        return
    source = Path(str(staged_resume.get("resume_checkpoint") or "")).expanduser()
    local_checkpoint = str(staged_resume.get("local_checkpoint") or "").strip()
    if not source.is_file():
        raise FileNotFoundError(f"Missing staged resume checkpoint: {source}")
    if not local_checkpoint:
        raise RuntimeError("Staged resume local checkpoint path is empty")
    _link_or_copy_checkpoint(source, Path(local_checkpoint).expanduser())


def train_command(
    *,
    exp_dir: Path,
    train_epochs: int,
    start_epoch: int,
    train_duration: int,
    final_train_args: list[str],
    attempt_index: int,
) -> list[str]:
    fp16 = "1" if baseline.truthy(os.environ.get("SURE_USE_FP16"), default=True) else "0"
    command = [
        os.environ.get("SURE_ICEFALL_PYTHON", sys.executable),
        str(baseline.RECIPE_DIR / "train.py"),
        "--world-size",
        str(current_world_size()),
        "--num-epochs",
        str(train_epochs),
        "--start-epoch",
        str(start_epoch),
        "--use-fp16",
        fp16,
        "--exp-dir",
        str(exp_dir),
        "--bpe-model",
        str(baseline.default_bpe_model()),
        "--master-port",
        str(15000 + ((os.getpid() + train_duration + attempt_index) % 20000)),
        "--max-duration",
        str(train_duration),
        *final_train_args,
    ]
    if os.environ.get("SURE_ENABLE_MUSAN", "0").strip() == "0" and "--enable-musan" not in arg_names(final_train_args):
        command.extend(["--enable-musan", "0"])
    return command


def run_training(
    *,
    exp_dir: Path,
    train_epochs: int,
    start_epoch: int,
    final_train_args: list[str],
    staged_resume: dict[str, Any] | None = None,
) -> tuple[int, int, list[dict[str, Any]]]:
    duration = selected_train_duration(final_train_args)
    attempts: list[dict[str, Any]] = []
    timeout = baseline.timeout_env("SURE_BASELINE_TRAIN_TIMEOUT", 43200)
    retry_sequence = baseline.duration_retry_sequence(duration)
    for attempt_index, attempt_duration in enumerate(retry_sequence, start=1):
        if attempt_index > 1:
            baseline.clean_training_exp_dir_for_retry()
        ensure_staged_resume_checkpoint(staged_resume)
        command = train_command(
            exp_dir=exp_dir,
            train_epochs=train_epochs,
            start_epoch=start_epoch,
            train_duration=attempt_duration,
            final_train_args=final_train_args,
            attempt_index=attempt_index,
        )
        name = "train" if attempt_index == 1 else f"train_retry_{attempt_index}_duration_{attempt_duration}"
        attempt: dict[str, Any] = {
            "attempt": attempt_index,
            "max_duration": attempt_duration,
            "command": command,
        }
        try:
            log_path = baseline.run_command(name, command, timeout=timeout)
        except (baseline.CommandFailedError, baseline.CommandTimedOutError) as exc:
            oom = baseline.is_oom_failure(exc.log_path)
            attempt.update(
                {
                    "status": "failed",
                    "oom": oom,
                    "return_code": getattr(exc, "return_code", None),
                    "timed_out": isinstance(exc, baseline.CommandTimedOutError),
                    "log_path": str(exc.log_path),
                    "max_memory_allocated_mb": baseline.parse_max_memory_mb(exc.log_path),
                }
            )
            attempts.append(attempt)
            if oom and attempt_index < len(retry_sequence):
                continue
            raise
        attempt.update(
            {
                "status": "success",
                "oom": False,
                "log_path": str(log_path),
                "max_memory_allocated_mb": baseline.parse_max_memory_mb(log_path),
            }
        )
        attempts.append(attempt)
        return train_epochs, attempt_duration, attempts
    raise RuntimeError("no Zipformer training attempt was executed")


def decode_command(
    *,
    exp_dir: Path,
    epoch: int,
    avg: int,
    use_averaged_model: str,
    decode_method: str,
    decode_max_duration: int,
    bpe_model: Path,
    final_decode_args: list[str],
) -> list[str]:
    return [
        os.environ.get("SURE_ICEFALL_PYTHON", sys.executable),
        str(baseline.patched_decode_script()),
        "--epoch",
        str(epoch),
        "--avg",
        str(avg),
        "--use-averaged-model",
        str(use_averaged_model),
        "--exp-dir",
        str(exp_dir),
        "--bpe-model",
        str(bpe_model),
        "--max-duration",
        str(decode_max_duration),
        "--decoding-method",
        decode_method,
        *final_decode_args,
    ]


def write_candidate_record(
    *,
    candidate_type: str,
    idea_text: str,
    changed_fields: list[str],
    final_train_args: list[str],
    final_decode_args: list[str],
    train_extra_args: list[str],
    decode_extra_args: list[str],
    exp_dir: Path,
    trained_epoch: int | None,
    selected_duration: int | None,
    attempts: list[dict[str, Any]],
    train_bpe_model: Path | None,
    decode_bpe_model: Path,
    elapsed_seconds: float,
    staged_resume: dict[str, Any] | None = None,
) -> None:
    train_config = args_to_config(final_train_args)
    decode_config = args_to_config(final_decode_args)
    train_bpe_sha256 = bpe_sha256_if_available(train_bpe_model)
    decode_bpe_sha256 = bpe_sha256_if_available(decode_bpe_model)
    inferred_changed = []
    for option in arg_names(train_extra_args):
        prefix = "arch_config" if option in STRUCTURE_ARGS else "training_config"
        inferred_changed.append(f"{prefix}.{option[2:].replace('-', '_')}")
    for option in arg_names(decode_extra_args):
        inferred_changed.append(f"inference_config.{option[2:].replace('-', '_')}")
    payload = {
        "candidate_type": candidate_type,
        "idea_text": idea_text,
        "changed_fields": changed_fields or sorted(set(inferred_changed)),
        "arch_config": {
            key: value
            for key, value in train_config.items()
            if "--" + key.replace("_", "-") in STRUCTURE_ARGS
        },
        "training_config": {
            **train_config,
            "actual_train_epoch": trained_epoch,
            "actual_train_start_epoch": (staged_resume or {}).get("start_epoch", 1),
            "staged_resume": staged_resume or {"enabled": False},
            "actual_train_max_duration": selected_duration,
            "actual_bpe_model": str(train_bpe_model) if train_bpe_model else "",
            "actual_bpe_sha256": train_bpe_sha256,
            "attempts": attempts,
        },
        "inference_config": {
            **decode_config,
            "eval_splits": baseline.parse_eval_splits(),
            "actual_bpe_model": str(decode_bpe_model),
            "actual_bpe_sha256": decode_bpe_sha256,
        },
        "defaults": {
            "dataset_profile": baseline.dataset_profile().name,
            "recipe_profile": baseline.recipe_profile().name,
            "recipe": baseline.recipe_profile().recipe_family,
            "train_args": baseline.large_cr_ctc_rnnt_train_args(),
            "decode_args": baseline.large_cr_ctc_rnnt_decode_args(),
        },
        "diff_from_defaults": {
            "train_extra_args": train_extra_args,
            "decode_extra_args": decode_extra_args,
        },
        "produced_artifacts": {
            "hyp": "artifacts/hyp.txt",
            "checkpoint_dir": str(exp_dir),
            "candidate_checkpoint": str(exp_dir / f"epoch-{trained_epoch}.pt") if trained_epoch else "",
        },
        "elapsed_seconds": elapsed_seconds,
        "timestamp": time.time(),
    }
    write_candidate_changes(Path("artifacts") / "candidate_changes.json", payload)


def main() -> int:
    args = parse_args()
    started = time.time()
    candidate_type = normalize_candidate_type(args.candidate_type, default=INFERENCE)
    train_extra_args = parse_list(args.train_args_json)
    decode_extra_args = parse_list(args.decode_args_json)
    changed_fields = parse_list(args.changed_fields_json)
    validate_candidate_args(
        candidate_type=candidate_type,
        action=args.action,
        train_extra_args=train_extra_args,
        decode_extra_args=decode_extra_args,
    )
    if args.action == "train_decode":
        os.environ["SURE_BASELINE_USE_PRETRAINED"] = "0"

    exp_dir = Path(args.exp_dir)
    working_dir = Path(args.working_dir)
    configure_baseline_paths(exp_dir, working_dir)
    baseline.ensure_workspace()
    exp_dir.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)

    final_train_args = merge_cli_args(
        baseline.large_cr_ctc_rnnt_train_args(),
        train_extra_args,
    )
    final_decode_args = merge_cli_args(
        baseline.large_cr_ctc_rnnt_decode_args(),
        decode_extra_args,
    )
    if args.action == "train_decode":
        final_decode_args = force_decode_structure_from_train(
            final_train_args=final_train_args,
            final_decode_args=final_decode_args,
            decode_extra_args=decode_extra_args,
        )
    final_decode_command_args = final_decode_args
    train_bpe_model = (
        selected_train_bpe_model(final_train_args)
        if args.action == "train_decode"
        else None
    )
    decode_bpe_model = selected_decode_bpe_model(
        action=args.action,
        final_train_args=final_train_args,
    )
    if args.action == "train_decode":
        if train_bpe_model is None:
            raise RuntimeError("train_decode action did not select a training BPE model")
        validate_requested_decode_bpe_model(
            train_bpe_model=train_bpe_model,
            final_decode_args=final_decode_args,
        )
        validate_train_decode_bpe_models(train_bpe_model, decode_bpe_model)
        final_decode_command_args = remove_cli_arg(final_decode_args, "--bpe-model")
    else:
        baseline.validate_decode_bpe_model()

    trained_epoch: int | None = None
    selected_duration: int | None = None
    attempts: list[dict[str, Any]] = []
    staged_resume: dict[str, Any] = {"enabled": False, "start_epoch": 1}
    if args.action == "train_decode":
        requested_train_epochs = max(1, parse_int(args.train_epochs, 1))
        staged_resume = staged_resume_config(exp_dir, requested_train_epochs)
        trained_epoch, selected_duration, attempts = run_training(
            exp_dir=exp_dir,
            train_epochs=int(staged_resume["target_epoch"]),
            start_epoch=int(staged_resume["start_epoch"]),
            final_train_args=final_train_args,
            staged_resume=staged_resume,
        )

    epoch = parse_int(
        args.decode_epoch,
        trained_epoch or baseline.int_env("SURE_BASELINE_EPOCH", baseline.BASELINE_EPOCH),
    )
    avg = max(1, parse_int(args.decode_avg, 1))
    decode_max_duration = max(1, parse_int(args.decode_max_duration, baseline.BASELINE_DECODE_MAX_DURATION))
    baseline.validate_decode_checkpoints(epoch)
    decode_cmd = decode_command(
        exp_dir=exp_dir,
        epoch=epoch,
        avg=avg,
        use_averaged_model=args.use_averaged_model,
        decode_method=args.decode_method,
        decode_max_duration=decode_max_duration,
        bpe_model=decode_bpe_model,
        final_decode_args=final_decode_command_args,
    )
    print(
        "[zipformer-wrapper] decoding eval splits: "
        + ", ".join(baseline.parse_eval_splits()),
        flush=True,
    )
    baseline.run_command(
        "decode",
        decode_cmd,
        timeout=parse_int(args.decode_timeout, 21600),
    )
    baseline.write_sure_hyp()
    write_candidate_record(
        candidate_type=candidate_type,
        idea_text=args.idea_text,
        changed_fields=changed_fields,
        final_train_args=final_train_args,
        final_decode_args=final_decode_args,
        train_extra_args=train_extra_args,
        decode_extra_args=decode_extra_args,
        exp_dir=exp_dir,
        trained_epoch=trained_epoch,
        selected_duration=selected_duration,
        attempts=attempts,
        train_bpe_model=train_bpe_model,
        decode_bpe_model=decode_bpe_model,
        elapsed_seconds=time.time() - started,
        staged_resume=staged_resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
