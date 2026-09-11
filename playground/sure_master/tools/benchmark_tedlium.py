"""Calibrate a common batch duration using the actual eight-NPU training recipe."""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--duration", type=int)
    args = p.parse_args()
    config = yaml.safe_load(os.path.expandvars(args.config.read_text()))
    sure = config["sure"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.duration:
        os.chdir(output)
        sources = sure["base_models"]["asr_en_wer"]["source_paths"]
        (output / "base_model").mkdir(exist_ok=True)
        for name, path in sources.items():
            target = output / "base_model" / name
            if not target.exists():
                target.symlink_to(path, target_is_directory=True)
        (output / "input").mkdir(exist_ok=True)
        import shutil

        shutil.copy2(sure["inputs"]["ref"], output / "input/ref.txt")
        os.environ.update({str(k): str(v) for k, v in sure["execution_env"].items()})
        os.environ.update(
            SURE_PROFILE_STEPS="200",
            SURE_PROFILE_OUTPUT=str(output / "profile.json"),
            SURE_MAX_DURATION=str(args.duration),
            SURE_BASELINE_TRAIN_MAX_DURATION=str(args.duration),
            SURE_CPU_THREADS="8",
            SURE_ACCELERATOR="npu",
            SURE_ICEFALL_PYTHON=sys.executable,
        )
        from playground.sure_master.baselines import (
            zipformer_large_cr_ctc_rnnt_baseline as baseline,
        )

        baseline.ensure_workspace()
        cmd = baseline.build_train_command(
            train_epochs=10, train_max_duration=args.duration, fp16="0", attempt_index=1
        )
        subprocess.run(cmd, check=True, env=baseline.command_env())
        if not (output / "profile.json").exists():
            raise RuntimeError(
                "Training ended before the 200-step measurement completed"
            )
        return
    trials = []
    for duration in (30, 60, 120, 240):
        folder = output / str(duration)
        folder.mkdir(exist_ok=True)
        with (folder / "benchmark.log").open("w") as log:
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "playground.sure_master.tools.benchmark_tedlium",
                        "--config",
                        str(args.config.resolve()),
                        "--output",
                        str(folder),
                        "--duration",
                        str(duration),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=3600,
                )
                report = (
                    json.loads((folder / "profile.json").read_text())
                    if result.returncode == 0
                    else {"status": "failed"}
                )
            except subprocess.TimeoutExpired:
                report = {"status": "timeout"}
        trials.append({"max_duration": duration, **report})
    eligible = [
        r
        for r in trials
        if r.get("peak_memory_fraction", 1) > 0
        and r.get("peak_memory_fraction", 1) <= 0.9
    ]
    if not eligible:
        raise RuntimeError("No stable training duration passed; inspect benchmark logs")
    winner = max(eligible, key=lambda r: r["audio_hours_per_hour"])
    q = winner["audio_hours_per_hour"]
    (output / "benchmark.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "trials": trials,
                "max_duration": winner["max_duration"],
                "search_candidate_hours": 1000 / q,
                "six_round_training_hours": 7000 / q,
                "ten_round_training_hours": 11000 / q,
                "full_model_hours_estimate": 453.8 * 30 / q,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
