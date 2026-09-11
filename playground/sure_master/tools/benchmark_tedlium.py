"""Calibrate a common batch duration using the actual eight-NPU training recipe."""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import yaml


def training_signature(sure: dict) -> dict:
    from playground.sure_master.core.artifacts import file_digest

    data = Path(sure["base_models"][sure["task_id"]]["source_paths"]["data"]).resolve()
    return {
        "data": str(data),
        "preparation_sha256": file_digest(data / "preparation.json"),
        "epochs": int(sure["execution_env"]["SURE_MAX_TRAIN_EPOCHS"]),
    }


def training_cost_estimates(hours: float, epochs: int, throughput: float) -> dict:
    import math

    if not all(math.isfinite(v) and v > 0 for v in (hours, epochs, throughput)):
        raise ValueError(
            "Training size, epochs and throughput must be positive and finite"
        )
    per_candidate = hours * epochs / throughput
    return {
        "search_candidate_hours": per_candidate,
        "six_round_training_hours": (1 + 4 * 6) * per_candidate,
        "ten_round_training_hours": (1 + 4 * 10) * per_candidate,
        "full_model_hours_estimate": per_candidate,
        "post_search_training_hours": 0,
        "estimate_unit": "aggregate training job-hours; excludes queueing and inference",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--duration", type=int)
    args = p.parse_args()
    config = yaml.safe_load(os.path.expandvars(args.config.read_text()))
    from playground.sure_master.core.full_training import (
        promote_full_training_to_search,
    )

    sure = promote_full_training_to_search(config["sure"])
    config["sure"] = sure
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
            train_epochs=int(sure["execution_env"]["SURE_MAX_TRAIN_EPOCHS"]),
            train_max_duration=args.duration,
            fp16="0",
            attempt_index=1,
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
    signature = training_signature(sure)
    preparation = json.loads((Path(signature["data"]) / "preparation.json").read_text())
    estimates = training_cost_estimates(
        preparation["splits"]["train"]["hours"], signature["epochs"], q
    )
    (output / "benchmark.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "trials": trials,
                "max_duration": winner["max_duration"],
                "training_signature": signature,
                **estimates,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
