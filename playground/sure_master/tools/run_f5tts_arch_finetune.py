#!/usr/bin/env python3
"""Controlled F5-TTS architecture fine-tuning wrapper for SURE Master.

This is the only supported architecture-level action surface for the F5-TTS
SURE task. It keeps structure search bounded and reproducible:

- whitelist-only DiT architecture parameters
- no tokenizer, vocab, mel, or vocoder changes
- no direct writes to the external F5-TTS source tree
- no unbounded training
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

import yaml


ALLOWED_ACTIONS = {"no_arch", "arch_finetune_short"}
ALLOWED_INIT_MODES = {"scratch", "partial_load"}
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
ALLOWED_DEPTHS = {18, 20, 22, 24}
ALLOWED_FF_MULTS = {2, 3, 4}
ALLOWED_CONV_LAYERS = {2, 4, 6}
ALLOWED_QK_NORMS = {"none", "rms_norm"}
ALLOWED_BOOL_STRINGS = {"0", "1", "false", "true", "no", "yes", "off", "on"}
FIXED_ARCH = {
    "dim": 1024,
    "heads": 16,
    "text_dim": 512,
    "text_mask_padding": True,
    "pe_attn_head": None,
    "attn_backend": "torch",
}
DEFAULT_ARCH = {
    **FIXED_ARCH,
    "depth": 22,
    "ff_mult": 2,
    "conv_layers": 4,
    "qk_norm": None,
    "attn_mask_enabled": False,
    "checkpoint_activations": False,
}


class UserError(ValueError):
    """Raised for invalid candidate-provided architecture actions."""


def env_value(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_path(name: str, default: str = "") -> Path | None:
    value = os.environ.get(name, default)
    return Path(value) if value else None


def bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text not in ALLOWED_BOOL_STRINGS:
        raise argparse.ArgumentTypeError(f"expected boolean string, got {value!r}")
    return text in {"1", "true", "yes", "on"}


def parse_float_choice(value: str) -> float:
    parsed = float(value)
    for allowed in ALLOWED_LEARNING_RATES:
        if abs(parsed - allowed) <= allowed * 1e-9:
            return allowed
    raise UserError(f"learning_rate must be one of {sorted(ALLOWED_LEARNING_RATES)}, got {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", default=env_value("SURE_TTS_ARCH_ACTION", "no_arch"), choices=sorted(ALLOWED_ACTIONS))
    parser.add_argument(
        "--init-mode",
        default=env_value("SURE_TTS_ARCH_INIT_MODE", "partial_load"),
        choices=sorted(ALLOWED_INIT_MODES),
    )
    parser.add_argument("--train-manifest", default=env_value("SURE_TTS_TRAIN_MANIFEST", "libritts_train_clean_100_1h"))
    parser.add_argument("--max-steps", type=int, default=int(env_value("SURE_TTS_TRAIN_MAX_STEPS", "2000")))
    parser.add_argument("--learning-rate", default=env_value("SURE_TTS_TRAIN_LEARNING_RATE", "3e-6"))
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=int(env_value("SURE_TTS_TRAIN_EFFECTIVE_BATCH_SIZE", "8")),
    )
    parser.add_argument("--seed", type=int, default=int(env_value("SURE_TTS_TRAIN_SEED", "20260714")))
    parser.add_argument("--depth", type=int, default=int(env_value("SURE_TTS_ARCH_DEPTH", "22")))
    parser.add_argument("--ff-mult", type=int, default=int(env_value("SURE_TTS_ARCH_FF_MULT", "2")))
    parser.add_argument("--conv-layers", type=int, default=int(env_value("SURE_TTS_ARCH_CONV_LAYERS", "4")))
    parser.add_argument("--qk-norm", default=env_value("SURE_TTS_ARCH_QK_NORM", "none"), choices=sorted(ALLOWED_QK_NORMS))
    parser.add_argument(
        "--attn-mask-enabled",
        type=bool_arg,
        default=bool_arg(env_value("SURE_TTS_ARCH_ATTN_MASK_ENABLED", "0")),
    )
    parser.add_argument(
        "--checkpoint-activations",
        type=bool_arg,
        default=bool_arg(env_value("SURE_TTS_ARCH_CHECKPOINT_ACTIVATIONS", "0")),
    )
    parser.add_argument("--f5-root", type=Path, default=Path(env_value("SURE_TTS_ROOT", "base_model/root")))
    parser.add_argument("--train-data-root", type=Path, default=Path(env_value("SURE_TTS_TRAIN_DATA", "base_model/train_data")))
    parser.add_argument("--base-ckpt", type=Path, default=env_path("SURE_TTS_BASE_CKPT_FILE", env_value("SURE_TTS_CKPT_FILE", "")))
    parser.add_argument("--vocab-file", type=Path, default=env_path("SURE_TTS_VOCAB_FILE"))
    parser.add_argument("--output-dir", type=Path, default=Path(env_value("SURE_TTS_ARCH_OUTPUT", "models/f5tts_arch")))
    parser.add_argument("--working-dir", type=Path, default=Path(env_value("SURE_TTS_ARCH_WORKDIR", "working/f5tts_arch")))
    parser.add_argument("--timeout", type=int, default=int(env_value("SURE_TTS_TRAIN_TIMEOUT", "14400")))
    parser.add_argument("--prepare-workers", type=int, default=int(env_value("SURE_TTS_TRAIN_PREPARE_WORKERS", "4")))
    parser.add_argument("--match-threshold", type=float, default=float(env_value("SURE_TTS_ARCH_MATCH_THRESHOLD", "0.70")))
    parser.add_argument("--early-stop", type=bool_arg, default=bool_arg(env_value("SURE_TTS_ARCH_EARLY_STOP", "1")))
    parser.add_argument(
        "--early-stop-min-updates",
        type=int,
        default=int(env_value("SURE_TTS_ARCH_EARLY_STOP_MIN_UPDATES", "0")),
    )
    parser.add_argument("--early-stop-patience", type=int, default=int(env_value("SURE_TTS_ARCH_EARLY_STOP_PATIENCE", "0")))
    parser.add_argument("--early-stop-ema-alpha", type=float, default=float(env_value("SURE_TTS_ARCH_EARLY_STOP_EMA_ALPHA", "0.05")))
    parser.add_argument(
        "--early-stop-min-relative-delta",
        type=float,
        default=float(env_value("SURE_TTS_ARCH_EARLY_STOP_MIN_RELATIVE_DELTA", "0.002")),
    )
    parser.add_argument("--reuse-existing", type=bool_arg, default=bool_arg(env_value("SURE_TTS_ARCH_REUSE_EXISTING", "1")))
    parser.add_argument("--idea-text", default=env_value("SURE_CANDIDATE_IDEA_TEXT", "F5-TTS controlled architecture candidate"))
    parser.add_argument("--dry-run", action="store_true", help="Validate action and paths without preparing or training.")
    return parser.parse_args()


def resolve_in_workspace(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def arch_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        **FIXED_ARCH,
        "depth": args.depth,
        "ff_mult": args.ff_mult,
        "conv_layers": args.conv_layers,
        "qk_norm": None if args.qk_norm == "none" else args.qk_norm,
        "attn_mask_enabled": bool(args.attn_mask_enabled),
        "checkpoint_activations": bool(args.checkpoint_activations),
    }


def validate_args(args: argparse.Namespace) -> float:
    forced_init_mode = env_value("SURE_TTS_ARCH_FORCE_INIT_MODE", "").strip()
    if forced_init_mode:
        if forced_init_mode not in ALLOWED_INIT_MODES:
            raise UserError(
                f"SURE_TTS_ARCH_FORCE_INIT_MODE must be one of {sorted(ALLOWED_INIT_MODES)}, "
                f"got {forced_init_mode}"
            )
        args.init_mode = forced_init_mode
    if args.depth not in ALLOWED_DEPTHS:
        raise UserError(f"depth must be one of {sorted(ALLOWED_DEPTHS)}, got {args.depth}")
    if args.ff_mult not in ALLOWED_FF_MULTS:
        raise UserError(f"ff_mult must be one of {sorted(ALLOWED_FF_MULTS)}, got {args.ff_mult}")
    if args.conv_layers not in ALLOWED_CONV_LAYERS:
        raise UserError(f"conv_layers must be one of {sorted(ALLOWED_CONV_LAYERS)}, got {args.conv_layers}")
    if not 0.0 <= args.match_threshold <= 1.0:
        raise UserError(f"match_threshold must be in [0, 1], got {args.match_threshold}")
    if not 0.0 < args.early_stop_ema_alpha <= 1.0:
        raise UserError(f"early_stop_ema_alpha must be in (0, 1], got {args.early_stop_ema_alpha}")
    if args.early_stop_min_relative_delta < 0:
        raise UserError("early_stop_min_relative_delta must be non-negative")
    if args.action == "no_arch" and args.init_mode != "partial_load":
        raise UserError("no_arch requires init_mode=partial_load because it evaluates the base checkpoint")
    if args.init_mode == "partial_load" and (not args.base_ckpt or not args.base_ckpt.is_file()):
        raise UserError(f"base checkpoint does not exist: {args.base_ckpt}")
    if args.vocab_file and not args.vocab_file.is_file():
        raise UserError(f"vocab file does not exist: {args.vocab_file}")

    if args.action == "no_arch":
        return parse_float_choice(args.learning_rate)

    if args.train_manifest not in ALLOWED_MANIFESTS:
        raise UserError(f"train_manifest must be one of {sorted(ALLOWED_MANIFESTS)}, got {args.train_manifest}")
    if args.max_steps not in ALLOWED_MAX_STEPS:
        raise UserError(f"max_steps must be one of {sorted(ALLOWED_MAX_STEPS)}, got {args.max_steps}")
    if args.effective_batch_size not in ALLOWED_EFFECTIVE_BATCHES:
        raise UserError(
            f"effective_batch_size must be one of {sorted(ALLOWED_EFFECTIVE_BATCHES)}, got {args.effective_batch_size}"
        )

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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def patch_trainer_guards(local_root: Path) -> None:
    trainer_path = local_root / "src/f5_tts/model/trainer.py"
    text = trainer_path.read_text(encoding="utf-8")
    init_marker = "        for epoch in range(skipped_epoch, self.epochs):\n"
    init_patch = (
        "        sure_max_steps = int(os.environ.get(\"SURE_TTS_TRAIN_MAX_STEPS\", \"0\") or 0)\n"
        "        sure_state_path = os.environ.get(\"SURE_TTS_EARLY_STOP_STATE\", \"\")\n"
        "        sure_early_enabled = os.environ.get(\"SURE_TTS_EARLY_STOP\", \"1\").lower() not in {\"0\", \"false\", \"no\", \"off\"}\n"
        "        sure_early_min_updates = int(os.environ.get(\"SURE_TTS_EARLY_STOP_MIN_UPDATES\", \"0\") or 0)\n"
        "        sure_early_patience = int(os.environ.get(\"SURE_TTS_EARLY_STOP_PATIENCE\", \"0\") or 0)\n"
        "        sure_early_alpha = float(os.environ.get(\"SURE_TTS_EARLY_STOP_EMA_ALPHA\", \"0.05\") or 0.05)\n"
        "        sure_min_relative_delta = float(os.environ.get(\"SURE_TTS_EARLY_STOP_MIN_RELATIVE_DELTA\", \"0.002\") or 0.002)\n"
        "        sure_loss_ema = None\n"
        "        sure_best_loss_ema = None\n"
        "        sure_no_improve_updates = 0\n"
        "\n"
        "        def sure_write_stop_state(reason, update, loss_value, best_value, no_improve):\n"
        "            if not self.is_main or not sure_state_path:\n"
        "                return\n"
        "            import json as _sure_json\n"
        "            with open(sure_state_path, \"w\", encoding=\"utf-8\") as _sure_file:\n"
        "                _sure_json.dump({\n"
        "                    \"stop_reason\": reason,\n"
        "                    \"update\": int(update),\n"
        "                    \"loss\": float(loss_value),\n"
        "                    \"loss_ema\": float(best_value) if best_value is not None else None,\n"
        "                    \"no_improve_updates\": int(no_improve),\n"
        "                }, _sure_file, indent=2)\n"
        "\n"
        "        for epoch in range(skipped_epoch, self.epochs):\n"
    )
    if init_patch not in text:
        if init_marker not in text:
            raise RuntimeError(f"Could not patch early-stop state into {trainer_path}")
        text = text.replace(init_marker, init_patch, 1)

    update_marker = "                    global_update += 1\n                    progress_bar.update(1)\n"
    update_patch = (
        "                    global_update += 1\n"
        "                    sure_loss_value = float(loss.item())\n"
        "                    if sure_loss_ema is None:\n"
        "                        sure_loss_ema = sure_loss_value\n"
        "                    else:\n"
        "                        sure_loss_ema = sure_early_alpha * sure_loss_value + (1.0 - sure_early_alpha) * sure_loss_ema\n"
        "                    sure_improved = sure_best_loss_ema is None or sure_loss_ema < sure_best_loss_ema * (1.0 - sure_min_relative_delta)\n"
        "                    if sure_improved:\n"
        "                        sure_best_loss_ema = sure_loss_ema\n"
        "                        sure_no_improve_updates = 0\n"
        "                    else:\n"
        "                        sure_no_improve_updates += 1\n"
        "                    if sure_max_steps > 0 and global_update >= sure_max_steps:\n"
        "                        self.save_checkpoint(global_update, last=True)\n"
        "                        sure_write_stop_state(\"max_steps\", global_update, sure_loss_value, sure_best_loss_ema, sure_no_improve_updates)\n"
        "                        self.accelerator.end_training()\n"
        "                        return\n"
        "                    if sure_early_enabled and global_update >= sure_early_min_updates and sure_early_patience > 0 and sure_no_improve_updates >= sure_early_patience:\n"
        "                        self.save_checkpoint(global_update, last=True)\n"
        "                        sure_write_stop_state(\"early_stop\", global_update, sure_loss_value, sure_best_loss_ema, sure_no_improve_updates)\n"
        "                        self.accelerator.end_training()\n"
        "                        return\n"
        "                    progress_bar.update(1)\n"
    )
    if update_patch not in text:
        if update_marker not in text:
            raise RuntimeError(f"Could not patch max-step/early-stop guard into {trainer_path}")
        text = text.replace(update_marker, update_patch, 1)
    trainer_path.write_text(text, encoding="utf-8")


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


def command_env(local_root: Path, working_dir: Path, args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{local_root / 'src'}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(local_root / "src")
    env["SURE_TTS_TRAIN_MAX_STEPS"] = str(args.max_steps)
    env["SURE_TTS_EARLY_STOP"] = "1" if args.early_stop else "0"
    env["SURE_TTS_EARLY_STOP_STATE"] = str(working_dir / "early_stop_state.json")
    min_updates = args.early_stop_min_updates or min(500, max(1, args.max_steps // 2))
    patience = args.early_stop_patience or max(100, max(1, args.max_steps // 5))
    env["SURE_TTS_EARLY_STOP_MIN_UPDATES"] = str(min_updates)
    env["SURE_TTS_EARLY_STOP_PATIENCE"] = str(patience)
    env["SURE_TTS_EARLY_STOP_EMA_ALPHA"] = str(args.early_stop_ema_alpha)
    env["SURE_TTS_EARLY_STOP_MIN_RELATIVE_DELTA"] = str(args.early_stop_min_relative_delta)
    env["SURE_TTS_TRAIN_SEED"] = str(args.seed)
    env["WANDB_MODE"] = "offline"
    env["TOKENIZERS_PARALLELISM"] = "false"
    cache_dir = working_dir / "cache"
    env.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
    env.setdefault("HF_HOME", os.environ.get("HF_HOME", "/hpc_stor03/sjtu_home/chaolei.liu/.cache/huggingface"))
    env.setdefault("HUGGINGFACE_HUB_CACHE", os.environ.get("HUGGINGFACE_HUB_CACHE", f"{env['HF_HOME']}/hub"))
    return env


def accelerate_port() -> str:
    configured = os.environ.get("SURE_TTS_ACCELERATE_PORT")
    if configured:
        return configured
    return str(30000 + (os.getpid() % 1000))


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


def write_model_cfg(source_root: Path, output_path: Path, arch_config: dict[str, Any]) -> None:
    base_cfg_path = source_root / "src/f5_tts/configs/F5TTS_v1_Base.yaml"
    cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))
    cfg["model"]["name"] = "F5TTS_v1_Base"
    cfg["model"]["tokenizer"] = "pinyin"
    cfg["model"]["tokenizer_path"] = None
    cfg["model"]["backbone"] = "DiT"
    cfg["model"]["arch"] = arch_config
    cfg["model"]["mel_spec"]["mel_spec_type"] = "vocos"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def write_train_script(path: Path) -> None:
    path.write_text(
        r'''
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def normalize_checkpoint_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    if checkpoint_path.suffix == ".safetensors":
        raw = load_safetensors(str(checkpoint_path), device="cpu")
        state = raw
    else:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(raw, dict) and "ema_model_state_dict" in raw:
            state = raw["ema_model_state_dict"]
        elif isinstance(raw, dict) and "model_state_dict" in raw:
            state = raw["model_state_dict"]
        elif isinstance(raw, dict):
            state = raw
        else:
            raise ValueError(f"Unsupported checkpoint payload in {checkpoint_path}")

    normalized = {}
    for key, value in state.items():
        if key in {"initted", "step", "update"}:
            continue
        cleaned = key.replace("ema_model.", "")
        if cleaned.startswith("module."):
            cleaned = cleaned[len("module."):]
        if cleaned.startswith("model."):
            cleaned = cleaned[len("model."):]
        normalized[cleaned] = value
    return normalized


def partial_load(model, checkpoint_path: Path, stats_path: Path, threshold: float):
    source = normalize_checkpoint_state(checkpoint_path)
    target = model.state_dict()
    merged = dict(target)
    matched = {}
    skipped_shape = {}
    skipped_missing = {}
    matched_numel = 0
    total_numel = sum(tensor.numel() for tensor in target.values())

    for key, tensor in source.items():
        if key not in target:
            skipped_missing[key] = list(tensor.shape)
            continue
        if tuple(target[key].shape) != tuple(tensor.shape):
            skipped_shape[key] = {"source": list(tensor.shape), "target": list(target[key].shape)}
            continue
        merged[key] = tensor
        matched[key] = list(tensor.shape)
        matched_numel += tensor.numel()

    ratio = matched_numel / max(1, total_numel)
    stats = {
        "init_mode": "partial_load",
        "matched_keys": len(matched),
        "target_keys": len(target),
        "source_keys": len(source),
        "matched_numel": int(matched_numel),
        "target_numel": int(total_numel),
        "matched_param_ratio": ratio,
        "skipped_missing_keys": len(skipped_missing),
        "skipped_shape_keys": len(skipped_shape),
        "skipped_missing_examples": dict(list(skipped_missing.items())[:20]),
        "skipped_shape_examples": dict(list(skipped_shape.items())[:20]),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if ratio < threshold:
        raise RuntimeError(f"partial checkpoint match ratio {ratio:.4f} is below threshold {threshold:.4f}")
    model.load_state_dict(merged, strict=True)
    return stats


def write_scratch_init_stats(model, stats_path: Path):
    target = model.state_dict()
    total_numel = sum(tensor.numel() for tensor in target.values())
    stats = {
        "init_mode": "scratch",
        "matched_keys": 0,
        "target_keys": len(target),
        "source_keys": 0,
        "matched_numel": 0,
        "target_numel": int(total_numel),
        "matched_param_ratio": 0.0,
        "skipped_missing_keys": 0,
        "skipped_shape_keys": 0,
        "skipped_missing_examples": {},
        "skipped_shape_examples": {},
        "message": "scratch initialization; base checkpoint was not loaded",
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return stats


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    local_root = Path(cfg["local_root"])
    sys.path.insert(0, str(local_root / "src"))

    from f5_tts.model import CFM, DiT, Trainer
    from f5_tts.model.dataset import load_dataset
    from f5_tts.model.utils import get_tokenizer

    target_sample_rate = 24000
    n_mel_channels = 100
    hop_length = 256
    win_length = 1024
    n_fft = 1024
    mel_spec_type = "vocos"

    dataset_name = cfg["dataset_name"]
    tokenizer = "pinyin"
    vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)
    mel_spec_kwargs = {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": win_length,
        "n_mel_channels": n_mel_channels,
        "target_sample_rate": target_sample_rate,
        "mel_spec_type": mel_spec_type,
    }
    arch_config = dict(cfg["arch_config"])
    model = CFM(
        transformer=DiT(**arch_config, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )
    init_mode = str(cfg.get("init_mode", "partial_load"))
    if init_mode == "partial_load":
        partial_load(
            model,
            Path(cfg["base_ckpt"]),
            Path(cfg["partial_load_stats_path"]),
            float(cfg["match_threshold"]),
        )
    elif init_mode == "scratch":
        write_scratch_init_stats(model, Path(cfg["partial_load_stats_path"]))
    else:
        raise ValueError(f"Unsupported init_mode: {init_mode}")

    trainer = Trainer(
        model,
        int(cfg["epochs"]),
        float(cfg["learning_rate"]),
        num_warmup_updates=int(cfg["num_warmup_updates"]),
        save_per_updates=int(cfg["save_per_updates"]),
        keep_last_n_checkpoints=0,
        checkpoint_path=cfg["checkpoint_path"],
        batch_size_per_gpu=int(cfg["batch_size_per_gpu"]),
        batch_size_type="sample",
        max_samples=0,
        grad_accumulation_steps=int(cfg["grad_accumulation_steps"]),
        max_grad_norm=1.0,
        logger=None,
        wandb_project=dataset_name,
        wandb_run_name="suremaster_f5tts_arch",
        log_samples=False,
        last_per_updates=int(cfg["last_per_updates"]),
        model_cfg_dict={
            "sure_arch_config": arch_config,
            "init_mode": init_mode,
            "learning_rate": float(cfg["learning_rate"]),
            "max_steps": int(cfg["max_steps"]),
            "effective_batch_size": int(cfg["effective_batch_size"]),
        },
    )
    train_dataset = load_dataset(dataset_name, tokenizer, mel_spec_kwargs=mel_spec_kwargs)
    trainer.train(train_dataset, resumable_with_seed=int(cfg["seed"]))


if __name__ == "__main__":
    main()
'''.lstrip(),
        encoding="utf-8",
    )


def candidate_changes_payload(
    *,
    args: argparse.Namespace,
    arch_config: dict[str, Any],
    output_dir: Path,
    model_cfg_path: Path,
    final_checkpoint: Path,
) -> dict[str, Any]:
    defaults = {
        "arch_config": DEFAULT_ARCH,
        "training_config": {
            "action": "no_arch",
            "init_mode": "partial_load",
            "train_manifest": "libritts_train_clean_100_1h",
            "max_steps": 2000,
            "learning_rate": "3e-6",
            "effective_batch_size": 8,
        },
    }
    training_config = {
        "action": args.action,
        "init_mode": args.init_mode,
        "train_manifest": args.train_manifest,
        "max_steps": args.max_steps,
        "learning_rate": str(parse_float_choice(args.learning_rate)),
        "effective_batch_size": args.effective_batch_size,
        "early_stop": bool(args.early_stop),
    }
    diff_arch = {
        key: {"default": DEFAULT_ARCH.get(key), "value": value}
        for key, value in arch_config.items()
        if DEFAULT_ARCH.get(key) != value
    }
    diff_training = {
        key: {"default": defaults["training_config"].get(key), "value": value}
        for key, value in training_config.items()
        if defaults["training_config"].get(key) != value
    }
    changed_fields = [f"arch_config.{key}" for key in diff_arch] + [f"training_config.{key}" for key in diff_training]
    produced_artifacts = {
        "final_checkpoint": str(final_checkpoint),
        "model_cfg": str(model_cfg_path),
        "training_summary": str(output_dir / "training_summary.json"),
        "arch_summary": str(output_dir / "arch_summary.json"),
    }
    if args.action != "no_arch":
        produced_artifacts["partial_load_stats"] = str(output_dir / "partial_load_stats.json")
    return {
        "candidate_type": "arch",
        "idea_text": args.idea_text,
        "changed_fields": changed_fields,
        "arch_config": arch_config,
        "training_config": training_config,
        "inference_config": {
            "model_cfg": str(model_cfg_path),
            "ckpt_file": str(final_checkpoint),
            "vocab_file": str(args.vocab_file) if args.vocab_file else "",
        },
        "defaults": defaults,
        "diff_from_defaults": {
            "arch_config": diff_arch,
            "training_config": diff_training,
        },
        "produced_artifacts": produced_artifacts,
    }


def write_no_arch_summary(args: argparse.Namespace, arch_config: dict[str, Any]) -> None:
    output_dir = resolve_in_workspace(args.output_dir)
    model_cfg_path = output_dir / "model_cfg.yaml"
    final_checkpoint = args.base_ckpt
    output_dir.mkdir(parents=True, exist_ok=True)
    write_model_cfg(resolve_in_workspace(args.f5_root), model_cfg_path, arch_config)
    payload = candidate_changes_payload(
        args=args,
        arch_config=arch_config,
        output_dir=output_dir,
        model_cfg_path=model_cfg_path,
        final_checkpoint=final_checkpoint,
    )
    payload["training_config"]["action"] = "no_arch"
    payload["training_config"]["init_mode"] = "partial_load"
    write_json(output_dir / "candidate_changes.json", payload)
    write_json(Path.cwd() / "artifacts" / "candidate_changes.json", payload)
    write_json(output_dir / "arch_summary.json", {
        "init_mode": "partial_load",
        "arch_config": arch_config,
        "default_arch": DEFAULT_ARCH,
        "diff_from_defaults": payload["diff_from_defaults"]["arch_config"],
    })
    summary = {
        "action": "no_arch",
        "init_mode": "partial_load",
        "final_checkpoint": str(final_checkpoint),
        "model_cfg": str(model_cfg_path),
        "message": "No architecture training was run; use the base checkpoint and generated base-compatible config.",
    }
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def run_arch_finetune(args: argparse.Namespace, learning_rate: float, arch_config: dict[str, Any]) -> None:
    started_at = time.time()
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
    model_cfg_path = output_dir / "model_cfg.yaml"
    partial_stats_path = output_dir / "partial_load_stats.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_existing and final_target.is_file() and model_cfg_path.is_file():
        arch_summary_path = output_dir / "arch_summary.json"
        existing_arch: dict[str, Any] | None = None
        if arch_summary_path.is_file():
            try:
                arch_summary_payload = json.loads(arch_summary_path.read_text(encoding="utf-8"))
                existing_arch = arch_summary_payload.get("arch_config")
                existing_init_mode = str(arch_summary_payload.get("init_mode") or "partial_load")
            except json.JSONDecodeError:
                existing_arch = None
                existing_init_mode = "partial_load"
        else:
            existing_init_mode = "partial_load"
        if existing_arch == arch_config and existing_init_mode == args.init_mode:
            changes = candidate_changes_payload(
                args=args,
                arch_config=arch_config,
                output_dir=output_dir,
                model_cfg_path=model_cfg_path,
                final_checkpoint=final_target,
            )
            changes["reused_existing"] = True
            write_json(output_dir / "candidate_changes.json", changes)
            write_json(workspace / "artifacts" / "candidate_changes.json", changes)
            summary = {
                "action": args.action,
                "init_mode": args.init_mode,
                "reused_existing": True,
                "model_cfg": str(model_cfg_path),
                "final_checkpoint": str(final_target),
                "arch_config": arch_config,
                "elapsed_seconds": round(time.time() - started_at, 3),
                "workspace": str(workspace),
            }
            write_json(output_dir / "training_summary.json", summary)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return

    copy_f5_root(f5_root, local_root)
    patch_trainer_guards(local_root)
    write_model_cfg(local_root, model_cfg_path, arch_config)

    env = command_env(local_root, working_dir, args)
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

    train_script = working_dir / "train_arch_candidate.py"
    train_config_path = working_dir / "train_arch_config.json"
    write_train_script(train_script)
    train_config = {
        "local_root": str(local_root),
        "dataset_name": dataset_name,
        "arch_config": arch_config,
        "base_ckpt": str(args.base_ckpt),
        "init_mode": args.init_mode,
        "partial_load_stats_path": str(partial_stats_path),
        "match_threshold": args.match_threshold,
        "checkpoint_path": str(ckpt_dir),
        "epochs": 100000,
        "learning_rate": learning_rate,
        "num_warmup_updates": max(1, min(100, args.max_steps // 10)),
        "save_per_updates": args.max_steps,
        "last_per_updates": args.max_steps,
        "batch_size_per_gpu": batch_config["batch_size_per_gpu"],
        "grad_accumulation_steps": batch_config["grad_accumulation_steps"],
        "max_steps": args.max_steps,
        "effective_batch_size": args.effective_batch_size,
        "seed": args.seed,
    }
    write_json(train_config_path, train_config)

    train_cmd = [sys.executable, str(train_script), "--config", str(train_config_path)]
    if world_size > 1:
        train_cmd = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--num_processes",
            str(world_size),
            "--mixed_precision",
            os.environ.get("SURE_TTS_ACCELERATE_MIXED_PRECISION", "fp16"),
            "--main_process_port",
            accelerate_port(),
            str(train_script),
            "--config",
            str(train_config_path),
        ]
    run_logged(train_cmd, cwd=local_root, env=env, timeout=args.timeout, log_path=output_dir / "finetune.log")

    if not final_source.exists():
        raise RuntimeError(f"F5-TTS architecture training did not produce expected checkpoint: {final_source}")
    shutil.copy2(final_source, final_target)

    early_state_path = working_dir / "early_stop_state.json"
    if early_state_path.is_file():
        early_state = json.loads(early_state_path.read_text(encoding="utf-8"))
    else:
        early_state = {"stop_reason": "completed", "update": None}
    partial_stats = json.loads(partial_stats_path.read_text(encoding="utf-8"))
    changes = candidate_changes_payload(
        args=args,
        arch_config=arch_config,
        output_dir=output_dir,
        model_cfg_path=model_cfg_path,
        final_checkpoint=final_target,
    )
    write_json(output_dir / "candidate_changes.json", changes)
    write_json(workspace / "artifacts" / "candidate_changes.json", changes)
    write_json(output_dir / "arch_summary.json", {
        "init_mode": args.init_mode,
        "arch_config": arch_config,
        "default_arch": DEFAULT_ARCH,
        "diff_from_defaults": changes["diff_from_defaults"]["arch_config"],
        "partial_load": partial_stats,
    })
    summary = {
        "action": args.action,
        "init_mode": args.init_mode,
        "train_manifest": args.train_manifest,
        "max_steps": args.max_steps,
        "learning_rate": learning_rate,
        "effective_batch_size": args.effective_batch_size,
        "world_size": world_size,
        "batch_size_per_gpu": batch_config["batch_size_per_gpu"],
        "grad_accumulation_steps": batch_config["grad_accumulation_steps"],
        "accelerate": world_size > 1,
        "seed": args.seed,
        "early_stop": {
            "enabled": bool(args.early_stop),
            "min_updates": int(env["SURE_TTS_EARLY_STOP_MIN_UPDATES"]),
            "patience": int(env["SURE_TTS_EARLY_STOP_PATIENCE"]),
            "ema_alpha": args.early_stop_ema_alpha,
            "min_relative_delta": args.early_stop_min_relative_delta,
            "state": early_state,
        },
        "base_checkpoint": str(args.base_ckpt),
        "vocab_file": str(args.vocab_file) if args.vocab_file else "",
        "model_cfg": str(model_cfg_path),
        "final_checkpoint": str(final_target),
        "partial_load": partial_stats,
        "elapsed_seconds": round(time.time() - started_at, 3),
        "workspace": str(workspace),
    }
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    try:
        learning_rate = validate_args(args)
        arch_config = arch_config_from_args(args)
        if args.action == "no_arch" and arch_config != DEFAULT_ARCH:
            raise UserError("no_arch can only be used with the default architecture; use arch_finetune_short for structure changes")
        payload = {
            "action": args.action,
            "init_mode": args.init_mode,
            "arch_config": arch_config,
            "learning_rate": learning_rate,
            "train_manifest": args.train_manifest,
            "max_steps": args.max_steps,
        }
        if args.dry_run:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return
        if args.action == "no_arch":
            write_no_arch_summary(args, arch_config)
        else:
            run_arch_finetune(args, learning_rate, arch_config)
    except UserError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
