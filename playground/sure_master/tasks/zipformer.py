"""Zipformer command translation behind the generic candidate worker."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def execute(action: str, parameters: dict, parent: str, frozen: str) -> None:
    allowed = {"train_args_json", "decode_args_json", "decode_method"}
    if set(parameters) - allowed:
        raise ValueError("Unsupported ASR parameter")
    env = os.environ.copy()
    if parent and not frozen:
        env["SURE_ASR_INITIAL_MODEL_ARTIFACT"] = parent
    if action == "baseline":
        command = [
            sys.executable,
            str(TOOLS.parent / "baselines/zipformer_large_cr_ctc_rnnt_baseline.py"),
        ]
    else:
        command = [
            sys.executable,
            env["SURE_ASR_ZIPFORMER_WRAPPER"],
            "--action",
            "decode_only" if action == "infer" else "train_decode",
            "--candidate-type",
            {"infer": "inference", "fine_tune": "fine_tune", "arch": "arch"}[action],
        ]
        if frozen:
            command.extend(["--model-artifact", frozen])
        for key, value in parameters.items():
            command.extend(
                [
                    "--" + key.replace("_", "-"),
                    json.dumps(value) if isinstance(value, list) else str(value),
                ]
            )
    subprocess.run(command, env=env, check=True)
