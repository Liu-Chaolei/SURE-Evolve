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
    "text_mask_padding",
    "text_embedding_average_upsampling",
    "pe_attn_head",
    "long_skip_connection",
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
    train = dict(settings.get("training", {}))
    if parameters.get("training"):
        raise ValueError("Official full-training settings are fixed across candidates")
    env = os.environ.copy()
    env.update(
        SURE_TTS_TRAIN_SEED=str(train.get("seed", 42)), SURE_TTS_PYTHON=sys.executable
    )
    completion = None
    if action in {"baseline", "fine_tune", "arch"}:
        from ..core.training import validate_training_config
        from ..runtime.training_sources import prepare_f5_training_source
        from .training_jobs import run_training

        validate_training_config("tts.f5tts", train)
        defaults = yaml.safe_load(model_cfg.read_text())
        model_config = yaml.safe_load(model_cfg.read_text())
        changes = parameters.get("architecture", {})
        if action == "arch" and (
            not changes
            or all(
                model_config["model"]["arch"].get(k) == v for k, v in changes.items()
            )
        ):
            raise ValueError("Architecture candidate must change model structure")
        if set(changes) - ARCH_KEYS:
            raise ValueError("Unsupported F5 architecture parameter")
        from ..core.search_scope import execution_contract
        from .adapters import TtsAdapter

        allowed = execution_contract(TtsAdapter().context(), {})[
            "candidate_parameters"
        ]["architecture"]
        for key, value in changes.items():
            if key not in allowed or value not in allowed[key]:
                raise ValueError(f"Unsupported F5 structure setting: {key}={value}")
            model_config["model"]["arch"][key] = (
                None if key == "qk_norm" and value == "none" else value
            )
        prepare_f5_training_source(root)
        completion, evidence = run_training(
            "tts.f5tts",
            settings,
            model_config["model"]["arch"],
            root,
            backend,
            Path("models/f5tts_training"),
            structural=action == "arch",
        )
        resources["checkpoint"] = evidence / "final_checkpoint.pt"
        model_cfg = evidence / "model_cfg.yaml"
        model_cfg.write_text(yaml.safe_dump(model_config, sort_keys=False))
        resources["training_evidence"] = evidence
        from ..runtime.official_trainers import checkpoint_identity
        from ..runtime.training_state import atomic_json

        completion["checkpoints"]["model_cfg"] = checkpoint_identity(
            evidence, model_cfg
        )
        atomic_json(evidence / "training_completion.json", completion)
        Path("artifacts").mkdir(exist_ok=True)
        record = {
            "candidate_type": "arch" if action == "arch" else "fine_tune",
            "idea_text": json.dumps(parameters),
            "changed_fields": [f"arch_config.{key}" for key in changes],
            "arch_config": model_config["model"]["arch"],
            "training_config": {
                **train,
                "epochs_completed": completion["epochs_completed"],
                "action": "finetune_full",
            },
            "inference_config": inference,
            "defaults": {"arch_config": defaults["model"]["arch"]},
            "diff_from_defaults": {
                "arch_config": {
                    key: {"default": defaults["model"]["arch"].get(key), "value": value}
                    for key, value in changes.items()
                }
            },
            "produced_artifacts": {"final_checkpoint": str(resources["checkpoint"])},
        }
        atomic_json(Path("artifacts/candidate_changes.json"), record)
    elif saved:
        from ..core.training import require_completion

        if "training_evidence" not in resources:
            raise ValueError(
                "Legacy short-trained F5 artifact has no full-training proof"
            )
        completion = require_completion(
            resources["training_evidence"] / "training_completion.json"
        )
    elif os.environ.get("SURE_COMPONENT_TEST") != "1":
        raise ValueError("F5 inference requires a completed training artifact")
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
            "baseline": "fine_tune",
            "infer": "inference",
            "fine_tune": "fine_tune",
            "arch": "arch",
        }[action],
        "--training-action",
        "finetune_full" if action in {"baseline", "fine_tune"} else "no_train",
        "--arch-action",
        "arch_finetune_full" if action == "arch" else "no_arch",
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
            "training": completion or {},
            "accelerator": backend.name,
            "cpu_operations": ["vocoder", "mel_spectrogram"]
            if backend.name == "npu"
            else [],
        },
    }
    Path("artifacts/model_resources.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
