#!/usr/bin/env python3
"""Controlled F5-TTS fine-tuning wrapper for SURE Master candidates.

This wrapper is the only supported training-level action surface for the
F5-TTS SURE task. It keeps training bounded and workspace-local:

- no model-structure changes
- no from-scratch training
- no eval/search/selection/holdout data as training input
- no writes to the external F5-TTS source tree or original checkpoint
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ALLOWED_ACTIONS = {"no_train", "finetune_short"}
ALLOWED_MANIFESTS = {
    "libritts_train_clean_100_1h",
    "libritts_train_clean_100_5h",
    "libritts_train_clean_100_10h",
}
ALLOWED_MAX_STEPS = {500, 1000, 2000, 5000}
ALLOWED_LEARNING_RATES = {1e-6, 3e-6, 5e-6, 1e-5}
ALLOWED_EFFECTIVE_BATCHES = {
    8: {"batch_size_per_gpu": 1, "grad_accumulation_steps": 8},
    16: {"batch_size_per_gpu": 1, "grad_accumulation_steps": 16},
    32: {"batch_size_per_gpu": 2, "grad_accumulation_steps": 16},
}
ALLOWED_FREEZE_POLICIES = {"none"}
DEFAULT_WORKSPACE = Path.cwd()
DEFAULT_REPO_TOOL_ROOT = Path("/hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster/playground/sure_master/tools")
DEFAULT_TRAINING_CONFIG = {
    "action": "no_train",
    "train_manifest": "libritts_train_clean_100_1h",
    "max_steps": 1000,
    "learning_rate": "3e-6",
    "effective_batch_size": 8,
    "freeze_policy": "none",
}


class UserError(ValueError):
    """Raised for invalid candidate-provided training actions."""


def env_value(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_path(name: str, default: str = "") -> Path | None:
    value = os.environ.get(name, default)
    return Path(value) if value else None


def parse_float_choice(value: str) -> float:
    parsed = float(value)
    for allowed in ALLOWED_LEARNING_RATES:
        if abs(parsed - allowed) <= allowed * 1e-9:
            return allowed
    raise UserError(f"learning_rate must be one of {sorted(ALLOWED_LEARNING_RATES)}, got {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", default=env_value("SURE_TTS_TRAIN_ACTION", "no_train"), choices=sorted(ALLOWED_ACTIONS))
    parser.add_argument("--train-manifest", default=env_value("SURE_TTS_TRAIN_MANIFEST", "libritts_train_clean_100_1h"))
    parser.add_argument("--max-steps", type=int, default=int(env_value("SURE_TTS_TRAIN_MAX_STEPS", "1000")))
    parser.add_argument("--learning-rate", default=env_value("SURE_TTS_TRAIN_LEARNING_RATE", "3e-6"))
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=int(env_value("SURE_TTS_TRAIN_EFFECTIVE_BATCH_SIZE", "8")),
    )
    parser.add_argument("--freeze-policy", default=env_value("SURE_TTS_TRAIN_FREEZE_POLICY", "none"))
    parser.add_argument("--seed", type=int, default=int(env_value("SURE_TTS_TRAIN_SEED", "20260714")))
    parser.add_argument("--f5-root", type=Path, default=Path(env_value("SURE_TTS_ROOT", "base_model/root")))
    parser.add_argument("--train-data-root", type=Path, default=Path(env_value("SURE_TTS_TRAIN_DATA", "base_model/train_data")))
    parser.add_argument("--base-ckpt", type=Path, default=env_path("SURE_TTS_BASE_CKPT_FILE", env_value("SURE_TTS_CKPT_FILE", "")))
    parser.add_argument("--vocab-file", type=Path, default=env_path("SURE_TTS_VOCAB_FILE"))
    parser.add_argument("--output-dir", type=Path, default=Path(env_value("SURE_TTS_TRAIN_OUTPUT", "models/f5tts_finetune")))
    parser.add_argument("--working-dir", type=Path, default=Path(env_value("SURE_TTS_TRAIN_WORKDIR", "working/f5tts_finetune")))
    parser.add_argument("--timeout", type=int, default=int(env_value("SURE_TTS_TRAIN_TIMEOUT", "14400")))
    parser.add_argument("--prepare-workers", type=int, default=int(env_value("SURE_TTS_TRAIN_PREPARE_WORKERS", "4")))
    parser.add_argument("--reuse-existing", default=env_value("SURE_TTS_TRAIN_REUSE_EXISTING", "1"))
    parser.add_argument("--dry-run", action="store_true", help="Validate action and paths without preparing or training.")
    return parser.parse_args()


def resolve_in_workspace(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def validate_args(args: argparse.Namespace) -> float:
    if args.action == "no_train":
        if not args.base_ckpt or not args.base_ckpt.is_file():
            raise UserError(f"base checkpoint does not exist: {args.base_ckpt}")
        return parse_float_choice(args.learning_rate)

    if args.train_manifest not in ALLOWED_MANIFESTS:
        raise UserError(f"train_manifest must be one of {sorted(ALLOWED_MANIFESTS)}, got {args.train_manifest}")
    if args.max_steps not in ALLOWED_MAX_STEPS:
        raise UserError(f"max_steps must be one of {sorted(ALLOWED_MAX_STEPS)}, got {args.max_steps}")
    if args.effective_batch_size not in ALLOWED_EFFECTIVE_BATCHES:
        raise UserError(
            f"effective_batch_size must be one of {sorted(ALLOWED_EFFECTIVE_BATCHES)}, got {args.effective_batch_size}"
        )
    if args.freeze_policy not in ALLOWED_FREEZE_POLICIES:
        raise UserError(f"freeze_policy must be one of {sorted(ALLOWED_FREEZE_POLICIES)}, got {args.freeze_policy}")
    if not args.base_ckpt or not args.base_ckpt.is_file():
        raise UserError(f"base checkpoint does not exist: {args.base_ckpt}")
    if args.vocab_file and not args.vocab_file.is_file():
        raise UserError(f"vocab file does not exist: {args.vocab_file}")

    f5_root = resolve_in_workspace(args.f5_root)
    train_data_root = resolve_in_workspace(args.train_data_root)
    manifest_csv = train_data_root / args.train_manifest / "metadata.csv"
    if not f5_root.exists():
        raise UserError(f"F5-TTS root does not exist: {f5_root}")
    if not manifest_csv.exists():
        raise UserError(f"training metadata.csv does not exist: {manifest_csv}")

    resolved_manifest = manifest_csv.resolve()
    if "f5tts_train_manifests" not in resolved_manifest.as_posix():
        raise UserError(f"training manifest must come from f5tts_train_manifests: {resolved_manifest}")
    if "f5tts_staged" in resolved_manifest.as_posix() or "f5tts_en_eval" in resolved_manifest.as_posix():
        raise UserError(f"eval prompt data must not be used for training: {resolved_manifest}")

    return parse_float_choice(args.learning_rate)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def candidate_changes_payload(
    *,
    args: argparse.Namespace,
    learning_rate: float,
    final_checkpoint: Path,
    output_dir: Path,
) -> dict[str, Any]:
    training_config = {
        "action": args.action,
        "train_manifest": args.train_manifest,
        "max_steps": args.max_steps,
        "learning_rate": str(learning_rate),
        "effective_batch_size": args.effective_batch_size,
        "freeze_policy": args.freeze_policy,
    }
    diff_training = {
        key: {"default": DEFAULT_TRAINING_CONFIG.get(key), "value": value}
        for key, value in training_config.items()
        if DEFAULT_TRAINING_CONFIG.get(key) != value
    }
    return {
        "candidate_type": "fine_tune",
        "idea_text": os.environ.get("SURE_CANDIDATE_IDEA_TEXT", "F5-TTS controlled fine-tuning candidate"),
        "changed_fields": [f"training_config.{key}" for key in diff_training],
        "arch_config": {},
        "training_config": training_config,
        "inference_config": {
            "ckpt_file": str(final_checkpoint),
            "vocab_file": str(args.vocab_file) if args.vocab_file else "",
        },
        "defaults": {"training_config": DEFAULT_TRAINING_CONFIG},
        "diff_from_defaults": {"training_config": diff_training},
        "produced_artifacts": {
            "final_checkpoint": str(final_checkpoint),
            "training_summary": str(output_dir / "training_summary.json"),
        },
    }


def copy_f5_root(source_root: Path, target_root: Path) -> None:
    if target_root.exists():
        shutil.rmtree(target_root)

    def ignore(_directory: str, names: list[str]) -> set[str]:
        ignored = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            "ckpts",
            "runs",
            "outputs",
            "wandb",
        }
        return {name for name in names if name in ignored or name.endswith(".pyc")}

    shutil.copytree(source_root, target_root, symlinks=False, ignore=ignore)


def patch_trainer_max_steps(local_root: Path) -> None:
    trainer_path = local_root / "src/f5_tts/model/trainer.py"
    text = trainer_path.read_text(encoding="utf-8")
    marker = "                    global_update += 1\n                    progress_bar.update(1)\n"
    patch = (
        "                    global_update += 1\n"
        "                    sure_max_steps = int(os.environ.get(\"SURE_TTS_TRAIN_MAX_STEPS\", \"0\") or 0)\n"
        "                    if sure_max_steps > 0 and global_update >= sure_max_steps:\n"
        "                        self.save_checkpoint(global_update, last=True)\n"
        "                        self.accelerator.end_training()\n"
        "                        return\n"
        "                    progress_bar.update(1)\n"
    )
    if patch in text:
        return
    if marker not in text:
        raise RuntimeError(f"Could not patch max-step guard into {trainer_path}")
    trainer_path.write_text(text.replace(marker, patch, 1), encoding="utf-8")


def patch_finetune_seed(local_root: Path) -> None:
    finetune_path = local_root / "src/f5_tts/train/finetune_cli.py"
    text = finetune_path.read_text(encoding="utf-8")
    old = "        resumable_with_seed=666,  # seed for shuffling dataset\n"
    new = (
        "        resumable_with_seed=int(os.environ.get(\"SURE_TTS_TRAIN_SEED\", \"666\")),"
        "  # seed for shuffling dataset\n"
    )
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"Could not patch training seed into {finetune_path}")
    finetune_path.write_text(text.replace(old, new, 1), encoding="utf-8")


def command_env(local_root: Path, args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{local_root / 'src'}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(local_root / "src")
    env["SURE_TTS_TRAIN_MAX_STEPS"] = str(args.max_steps)
    env["SURE_TTS_TRAIN_SEED"] = str(args.seed)
    env["WANDB_MODE"] = "offline"
    env["TOKENIZERS_PARALLELISM"] = "false"
    cache_dir = resolve_in_workspace(args.working_dir) / "cache"
    env.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
    env.setdefault("HF_HOME", os.environ.get("HF_HOME", "/hpc_stor03/sjtu_home/chaolei.liu/.cache/huggingface"))
    env.setdefault("HUGGINGFACE_HUB_CACHE", os.environ.get("HUGGINGFACE_HUB_CACHE", f"{env['HF_HOME']}/hub"))
    return env


def visible_cuda_world_size() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or visible.lower() in {"none", "null", "false", "-1"}:
        return 1
    devices = [part.strip() for part in visible.split(",") if part.strip()]
    return max(1, len(devices))


def scaled_batch_config(base_config: dict[str, int], world_size: int, requested_effective_batch: int) -> dict[str, int]:
    batch_size_per_gpu = max(1, int(base_config["batch_size_per_gpu"]))
    grad_accumulation_steps = max(
        1,
        math.ceil(requested_effective_batch / max(1, batch_size_per_gpu * world_size)),
    )
    return {
        "batch_size_per_gpu": batch_size_per_gpu,
        "grad_accumulation_steps": grad_accumulation_steps,
    }


def accelerate_port() -> str:
    configured = os.environ.get("SURE_TTS_ACCELERATE_PORT")
    if configured:
        return configured
    return str(29000 + (os.getpid() % 1000))


def build_train_command(
    *,
    local_root: Path,
    args: argparse.Namespace,
    learning_rate: float,
    batch_config: dict[str, int],
    world_size: int,
) -> list[str]:
    script = str(local_root / "src/f5_tts/train/finetune_cli.py")
    train_args = [
        script,
        "--exp_name",
        "F5TTS_v1_Base",
        "--dataset_name",
        f"suremaster_{args.train_manifest}",
        "--learning_rate",
        str(learning_rate),
        "--batch_size_per_gpu",
        str(batch_config["batch_size_per_gpu"]),
        "--batch_size_type",
        "sample",
        "--max_samples",
        "0",
        "--grad_accumulation_steps",
        str(batch_config["grad_accumulation_steps"]),
        "--epochs",
        "100000",
        "--num_warmup_updates",
        str(max(1, min(100, args.max_steps // 10))),
        "--save_per_updates",
        str(args.max_steps),
        "--keep_last_n_checkpoints",
        "0",
        "--last_per_updates",
        str(args.max_steps),
        "--finetune",
        "--pretrain",
        str(args.base_ckpt),
        "--tokenizer",
        "pinyin",
    ]
    if world_size <= 1:
        return [sys.executable, *train_args]

    mixed_precision = os.environ.get("SURE_TTS_ACCELERATE_MIXED_PRECISION", "fp16")
    return [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(world_size),
        "--mixed_precision",
        mixed_precision,
        "--main_process_port",
        accelerate_port(),
        *train_args,
    ]


def run_logged(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("+ " + " ".join(cmd) + "\n")
        log_file.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise TimeoutError(f"Command timed out after {timeout}s; see {log_path}") from exc
        if return_code != 0:
            raise RuntimeError(f"Command failed with exit code {return_code}; see {log_path}")


def write_no_train_summary(args: argparse.Namespace) -> None:
    output_dir = resolve_in_workspace(args.output_dir)
    learning_rate = parse_float_choice(args.learning_rate)
    payload = {
        "action": "no_train",
        "final_checkpoint": str(args.base_ckpt),
        "vocab_file": str(args.vocab_file) if args.vocab_file else "",
        "message": "No fine-tuning was run; use the base checkpoint for inference.",
    }
    changes = candidate_changes_payload(
        args=args,
        learning_rate=learning_rate,
        final_checkpoint=args.base_ckpt,
        output_dir=output_dir,
    )
    write_json(output_dir / "candidate_changes.json", changes)
    write_json(Path.cwd() / "artifacts" / "candidate_changes.json", changes)
    write_json(output_dir / "training_summary.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def truthy_text(value: str | bool) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def should_reuse_existing(
    args: argparse.Namespace,
    learning_rate: float,
    final_checkpoint: Path,
    output_dir: Path,
) -> bool:
    if not truthy_text(args.reuse_existing):
        return False
    summary_path = output_dir / "training_summary.json"
    if not final_checkpoint.is_file() or not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "action": args.action,
        "train_manifest": args.train_manifest,
        "max_steps": args.max_steps,
        "effective_batch_size": args.effective_batch_size,
        "freeze_policy": args.freeze_policy,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            return False
    try:
        if abs(float(summary.get("learning_rate")) - float(learning_rate)) > float(learning_rate) * 1e-9:
            return False
    except (TypeError, ValueError):
        return False
    return True


def run_finetune(args: argparse.Namespace, learning_rate: float) -> None:
    workspace = Path.cwd().resolve()
    f5_root = resolve_in_workspace(args.f5_root)
    train_data_root = resolve_in_workspace(args.train_data_root)
    manifest_csv = train_data_root / args.train_manifest / "metadata.csv"
    working_dir = resolve_in_workspace(args.working_dir)
    output_dir = resolve_in_workspace(args.output_dir)
    local_root = working_dir / "f5tts_root"
    dataset_name = f"suremaster_{args.train_manifest}"
    prepared_dataset = local_root / "data" / f"{dataset_name}_pinyin"
    ckpt_dir = local_root / "ckpts" / dataset_name
    final_source = ckpt_dir / "model_last.pt"
    final_target = output_dir / "final_checkpoint.pt"

    started_at = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)
    if should_reuse_existing(args, learning_rate, final_target, output_dir):
        changes = candidate_changes_payload(
            args=args,
            learning_rate=learning_rate,
            final_checkpoint=final_target,
            output_dir=output_dir,
        )
        changes["reused_existing"] = True
        write_json(output_dir / "candidate_changes.json", changes)
        write_json(workspace / "artifacts" / "candidate_changes.json", changes)
        summary = {
            "action": args.action,
            "reused_existing": True,
            "train_manifest": args.train_manifest,
            "max_steps": args.max_steps,
            "learning_rate": learning_rate,
            "effective_batch_size": args.effective_batch_size,
            "freeze_policy": args.freeze_policy,
            "final_checkpoint": str(final_target),
            "elapsed_seconds": round(time.time() - started_at, 3),
            "workspace": str(workspace),
        }
        write_json(output_dir / "training_summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    copy_f5_root(f5_root, local_root)
    patch_trainer_max_steps(local_root)
    patch_finetune_seed(local_root)

    env = command_env(local_root, args)
    world_size = visible_cuda_world_size()
    batch_config = scaled_batch_config(
        ALLOWED_EFFECTIVE_BATCHES[args.effective_batch_size],
        world_size,
        args.effective_batch_size,
    )

    prepare_cmd = [
        sys.executable,
        str(local_root / "src/f5_tts/train/datasets/prepare_csv_wavs.py"),
        str(manifest_csv),
        str(prepared_dataset),
        "--workers",
        str(max(1, args.prepare_workers)),
    ]
    run_logged(
        prepare_cmd,
        cwd=local_root,
        env=env,
        timeout=min(args.timeout, 3600),
        log_path=output_dir / "prepare_dataset.log",
    )

    train_cmd = build_train_command(
        local_root=local_root,
        args=args,
        learning_rate=learning_rate,
        batch_config=batch_config,
        world_size=world_size,
    )
    run_logged(train_cmd, cwd=local_root, env=env, timeout=args.timeout, log_path=output_dir / "finetune.log")

    if not final_source.exists():
        raise RuntimeError(f"F5-TTS did not produce expected checkpoint: {final_source}")
    shutil.copy2(final_source, final_target)

    payload = {
        "action": args.action,
        "train_manifest": args.train_manifest,
        "max_steps": args.max_steps,
        "learning_rate": learning_rate,
        "effective_batch_size": args.effective_batch_size,
        "world_size": world_size,
        "batch_size_per_gpu": batch_config["batch_size_per_gpu"],
        "grad_accumulation_steps": batch_config["grad_accumulation_steps"],
        "accelerate": world_size > 1,
        "freeze_policy": args.freeze_policy,
        "seed": args.seed,
        "base_checkpoint": str(args.base_ckpt),
        "vocab_file": str(args.vocab_file) if args.vocab_file else "",
        "prepared_dataset": str(prepared_dataset),
        "final_checkpoint": str(final_target),
        "elapsed_seconds": round(time.time() - started_at, 3),
        "workspace": str(workspace),
    }
    changes = candidate_changes_payload(
        args=args,
        learning_rate=learning_rate,
        final_checkpoint=final_target,
        output_dir=output_dir,
    )
    write_json(output_dir / "candidate_changes.json", changes)
    write_json(workspace / "artifacts" / "candidate_changes.json", changes)
    write_json(output_dir / "training_summary.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    try:
        learning_rate = validate_args(args)
        if args.dry_run:
            payload = {
                "dry_run": True,
                "action": args.action,
                "train_manifest": args.train_manifest,
                "max_steps": args.max_steps,
                "learning_rate": learning_rate,
                "effective_batch_size": args.effective_batch_size,
                "freeze_policy": args.freeze_policy,
            }
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return
        if args.action == "no_train":
            write_no_train_summary(args)
            return
        run_finetune(args, learning_rate)
    except UserError as exc:
        print(f"Invalid F5-TTS training action: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
