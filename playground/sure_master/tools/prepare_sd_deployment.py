"""Freeze the official four-NPU AMI deployment; credentials stay unresolved."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import hashlib
from pathlib import Path

import yaml

from playground.sure_master.tools.freeze_tedlium_run import freeze
from playground.sure_master.tools.with_api_profile import apply_api_routing

PROJECT = Path(__file__).resolve().parents[3]
PERSONAL = PROJECT.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, default=PROJECT / "runs/sd_ami_official")
    parser.add_argument("--env-file", type=Path, default=PROJECT / ".env")
    args = parser.parse_args()
    if "@sha256:" not in args.image:
        parser.error("--image must be an immutable registry digest")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    resources = PERSONAL / "data/sure_sd_resources"
    env = {
        "SURE_EVOLVE_ROOT": str(PROJECT), "SURE_EVAL_ROOT": str(PERSONAL / "sure-eval"),
        "SURE_DIARIZEN_ROOT": str(PERSONAL / "SD/DiariZen"),
        "SURE_SD_DATA": str(PERSONAL / "data/sure_ami"),
        "SURE_WAVLM_INIT": str(resources / "wavlm-init/wavlm-base-plus-converted.bin"),
        "SURE_SD_EMBEDDING": str(resources / "pyannote/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin"),
        "SURE_WORKER_PYTHON": "python", "SURE_METRIC_PYTHON": "/opt/sure-metrics/bin/python",
        "XLAB_ROOT": str(PERSONAL / "XLab"),
        "XLAB_PYTHON": str(PERSONAL / "data/sure_xlab_runtime312/bin/python"),
        "XLAB_SURE_SURVEY_PATH": str(PERSONAL / "XLab/.xlab/runs/20260910-survey-sd-v2/artifacts/survey.json"),
        "SURE_SD_IMAGE": args.image,
    }
    raw = (PROJECT / "configs/sure_master/ordinary-sd-npu.yaml").read_text()
    for key, value in env.items():
        raw = raw.replace("${" + key + "}", value)
    config = yaml.safe_load(raw)
    sure = config["sure"]
    sure["execution_mode"] = "slurm"
    sure["task_cards_path"] = str(PROJECT / "playground/sure_master/task_cards/sure_tasks.yaml")
    sure["initial_solution_path"] = str(PROJECT / sure["initial_solution_path"])
    # User removed the 24-hour projection gate; Slurm limits are resumable slices.
    sure["execution_env"]["SURE_TRAIN_BUDGET_SECONDS"] = "0"
    sure["execution_env"]["SURE_RUN_TIMEOUT"] = "0"
    sure["execution_env"]["ASCEND_PROCESS_LOG_PATH"] = "/local/job/ascend/log"
    sure["execution_env"]["ASCEND_WORK_PATH"] = "/local/job/ascend/work"
    sure["slurm"] = {"shared_root": str(PERSONAL), "image": args.image,
                     "partition": "compute", "max_parallel": 1,
                     "time_limit": "1-00:00:00", "max_segments": 30, "poll_seconds": 20}
    for agent in config["agents"].values():
        for key in ("system_prompt_file", "user_prompt_file"):
            agent[key] = str(PROJECT / "playground/sure_master" / agent[key])
    config["session"]["local"]["working_dir"] = str(output / "search/workspace")
    config["session"]["local"]["timeout"] = 0
    config["xlab"]["idea_provider"]["environment"]["XLAB_SURE_RUN_ROOT"] = str(output / "xlab_ideas")
    python = PERSONAL / "data/sure_asr_controller/bin/python"
    config = apply_api_routing(config, args.env_file, str(python),
                               PROJECT / "playground/sure_master/tools/with_api_profile.py")
    candidate = output / "source-config.yaml"
    candidate.write_text(yaml.safe_dump(config, sort_keys=False))
    snapshot, deployment = freeze(candidate, output)
    shutil.copy2(PROJECT / "run.py", snapshot / "run.py")
    manifest_path = snapshot / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["run.py"] = hashlib.sha256((snapshot / "run.py").read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    environment = output / "runtime.env"
    environment.write_text("".join(f"export {key}={shlex.quote(value)}\n" for key, value in env.items()))
    environment.chmod(0o600)
    python = PERSONAL / "data/sure_asr_controller/bin/python"
    launch = output / "launch.sh"
    argv = [str(python), "-m", "playground.sure_master.tools.with_api_profile",
            "--env-file", str(args.env_file.resolve()), "--role", "controller", "--", str(python), str(snapshot / "run.py"),
            "--agent", "sure_master", "--config", str(deployment), "--run-dir", str(output / "search"),
            "--task", "使用官方 WavLM-updated 固定训练预算，在 AMI 上进行两轮结构自进化，以 SURE DER 选优。"]
    launch.write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(snapshot))
                      + "\nexport PYTHONPATH=" + shlex.quote(str(snapshot)) + "\nexec " + shlex.join(argv) + "\n")
    launch.chmod(0o700)
    print(json.dumps({"source": str(snapshot), "config": str(deployment), "launch": str(launch)}))


if __name__ == "__main__":
    main()
