"""F5 execution/replay using the existing guarded training and batch wrappers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ..runtime.model_source import snapshot_source, prepare_f5_source

TOOLS = Path(__file__).resolve().parents[1] / "tools"
INFERENCE_KEYS = {
    "nfe_step",
    "cfg_strength",
    "sway_sampling_coef",
    "speed",
    "remove_silence",
    "text_cleanup",
    "max_chunk_chars",
    "min_chunk_chars",
    "target_rms",
    "cross_fade_duration",
}
TRAIN_KEYS = {"learning_rate", "effective_batch_size", "freeze_policy"}
ARCH_KEYS = {
    "depth",
    "ff_mult",
    "conv_layers",
    "qk_norm",
    "attn_mask_enabled",
    "checkpoint_activations",
}


def execute(action, parameters, settings, manifest, saved, backend, parent):
    import yaml

    if set(parameters) - {"inference", "training", "architecture"}:
        raise ValueError("Expected inference/training/architecture parameter sections")
    for section, allowed in (
        ("inference", INFERENCE_KEYS),
        ("training", TRAIN_KEYS),
        ("architecture", ARCH_KEYS),
    ):
        if (
            not isinstance(parameters.get(section, {}), dict)
            or set(parameters.get(section, {})) - allowed
        ):
            raise ValueError(f"Unsupported {section} parameters")
    if action in {"baseline", "infer"} and (
        parameters.get("training") or parameters.get("architecture")
    ):
        raise ValueError("Inference cannot request training or structural changes")
    if action == "fine_tune" and parameters.get("architecture"):
        raise ValueError("Structural changes require arch action")
    resources = {k: Path(v).resolve() for k, v in settings["resources"].items()}
    if saved:
        resources.update(saved)
    root = snapshot_source(resources["source"], Path("working/f5_source"))
    prepare_f5_source(root)
    model_cfg = resources.get(
        "model_cfg", root / "src/f5_tts/configs/F5TTS_v1_Base.yaml"
    )
    inference = {
        **settings.get("inference", {}),
        **manifest.get("inference_config", {}),
        **parameters.get("inference", {}),
    }
    if set(inference) - INFERENCE_KEYS:
        raise ValueError(
            f"Unsupported F5 inference configuration: {sorted(set(inference) - INFERENCE_KEYS)}"
        )
    train = {**settings.get("training", {}), **parameters.get("training", {})}
    env = os.environ.copy()
    env.update(
        SURE_TTS_ROOT=str(root),
        SURE_TTS_PYTHON=sys.executable,
        SURE_TTS_BASE_CKPT_FILE=str(resources["checkpoint"]),
        SURE_TTS_VOCAB_FILE=str(resources["vocab"]),
        SURE_TTS_ACCELERATE_MIXED_PRECISION="no",
        ACCELERATE_MIXED_PRECISION="no",
        ACCELERATE_TORCH_DEVICE=str(backend.device),
        SURE_TTS_TRAIN_SEED=str(train.get("seed", 42)),
    )
    # Training always starts from configured fixed resources, never from the parent.
    if action in {"fine_tune", "arch"}:
        cap = int(settings.get("training", {}).get("max_steps", 1000))
        train_manifest = Path(train["manifest"]).resolve()
        env["SURE_TRAIN_MANIFESTS_JSON"] = json.dumps(
            {train_manifest.parent.name: str(train_manifest)}
        )
        env["SURE_TRAIN_MAX_STEPS"] = str(cap)
        script = (
            "run_f5tts_finetune.py"
            if action == "fine_tune"
            else "run_f5tts_arch_finetune.py"
        )
        out = Path(
            "models/f5tts_finetune" if action == "fine_tune" else "models/f5tts_arch"
        )
        command = [
            sys.executable,
            str(TOOLS / script),
            "--action",
            "finetune_short" if action == "fine_tune" else "arch_finetune_short",
            "--f5-root",
            str(root),
            "--base-ckpt",
            str(resources["checkpoint"]),
            "--vocab-file",
            str(resources["vocab"]),
            "--train-data-root",
            str(train_manifest.parent.parent),
            "--train-manifest",
            train_manifest.parent.name,
            "--max-steps",
            str(cap),
            "--seed",
            str(train.get("seed", 42)),
            "--reuse-existing",
            "0",
            "--output-dir",
            str(out),
        ]
        for key in TRAIN_KEYS:
            if key in train and (key != "freeze_policy" or action == "fine_tune"):
                command.extend(["--" + key.replace("_", "-"), str(train[key])])
        if action == "arch":
            command.extend(
                [
                    "--init-mode",
                    "partial_load",
                    "--match-threshold",
                    "0.70",
                    "--early-stop",
                    "0",
                ]
            )
            if not parameters.get("architecture"):
                raise ValueError(
                    "Architecture candidate must declare structural changes"
                )
            for key, value in parameters["architecture"].items():
                command.extend(
                    [
                        "--" + key.replace("_", "-"),
                        str(int(value)) if isinstance(value, bool) else str(value),
                    ]
                )
        subprocess.run(command, env=env, check=True)
        resources["checkpoint"] = (out / "final_checkpoint.pt").resolve()
        if action == "arch":
            model_cfg = (out / "model_cfg.yaml").resolve()
    eval_manifest = os.environ["SURE_EVAL_MANIFEST"]
    language = settings.get("language", "zh")
    command = [
        sys.executable,
        str(TOOLS / "run_f5tts_batch_infer.py"),
        "--f5-root",
        str(root),
        "--eval-data",
        eval_manifest,
        "--language",
        language,
        "--max-samples",
        "0",
        "--resume",
        "0",
        "--workers",
        "1",
        "--device",
        str(backend.device),
        "--ckpt-file",
        str(resources["checkpoint"]),
        "--vocab-file",
        str(resources["vocab"]),
        "--model-cfg",
        str(model_cfg),
        "--load-vocoder-from-local",
        "1",
        "--vocoder-local-path",
        str(resources["vocoder"]),
        "--candidate-type",
        {
            "baseline": "inference",
            "infer": "inference",
            "fine_tune": "fine_tune",
            "arch": "arch",
        }[action],
        "--training-action",
        "finetune_short" if action == "fine_tune" else "no_train",
        "--arch-action",
        "arch_finetune_short" if action == "arch" else "no_arch",
    ]
    for key, value in inference.items():
        command.extend(
            [
                "--" + key.replace("_", "-"),
                str(int(value)) if isinstance(value, bool) else str(value),
            ]
        )
    subprocess.run(command, env=env, check=True)
    resources.update(source=root, model_cfg=model_cfg)
    report = {
        "resources": {k: str(v) for k, v in resources.items()},
        "model_config": yaml.safe_load(Path(model_cfg).read_text()),
        "inference_config": inference,
        "provenance": {
            "action": action,
            "parent": parent,
            "training": train if action in {"fine_tune", "arch"} else {},
            "accelerator": backend.name,
            "cpu_operations": ["vocoder", "mel_spectrogram"]
            if backend.name == "npu"
            else [],
        },
    }
    Path("artifacts/model_resources.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
