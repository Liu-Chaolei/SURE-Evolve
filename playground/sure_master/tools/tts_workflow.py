"""Durable Premium preparation, eight-NPU acceptance and formal free evolution."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import time

import yaml

from playground.sure_master.core.training import canonical_digest
from playground.sure_master.core.utils.slurm import (
    atomic_json,
    command,
    job_state,
    ACTIVE,
    source_digest,
)

PROJECT = Path(__file__).resolve().parents[3]


def signature(sure: dict) -> str:
    return canonical_digest(
        {
            "source": source_digest(),
            "image": sure["slurm"]["image"],
            "training": sure["task"]["training"],
            "resources": sure["task"]["resources"],
            "preparation": sure["data_preparation"],
        }
    )


def prepared_ready(sure: dict) -> bool:
    from playground.sure_master.tasks.training_resources import validate_task_resources

    marker = Path(sure["data_preparation"])
    if not marker.exists():
        return False
    record = json.loads(marker.read_text())
    if record.get("training_selection") != "full_prepared_train":
        return False
    validate_task_resources("tts.f5tts", sure)
    from playground.sure_master.core.artifacts import file_digest

    for name, count in (("search", 400), ("selection", 800), ("holdout", 2020)):
        manifest = Path(sure["datasets"][name]["manifest"])
        if file_digest(manifest) != record["digests"][name]:
            raise ValueError(f"Prepared {name} changed")
        if (
            len([line for line in manifest.read_text().splitlines() if line.strip()])
            != count
        ):
            return False
    return True


def acceptance_signature(sure: dict) -> str:
    from playground.sure_master.core.artifacts import file_digest
    from playground.sure_master.runtime.f5_evolution import executable_identity

    resources = sure["task"]["resources"]
    files = [
        Path(sure["data_preparation"]),
        Path(resources["checkpoint"]),
        Path(resources["vocab"]),
        Path(resources["vocoder"]) / "config.yaml",
        Path(resources["vocoder"]) / "pytorch_model.bin",
    ]
    return canonical_digest(
        {
            "deployment": signature(sure),
            "files": {str(p): file_digest(p) for p in files},
            "f5_source": executable_identity(Path(resources["source"])),
        }
    )


def submit_stage(
    name: str, argv: list[str], config: dict, output: Path, devices: int = 0
) -> str:
    sure = config["sure"]
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / "job.json"
    identity = canonical_digest(
        {"argv": argv, "devices": devices, "signature": signature(sure)}
    )
    with (directory / "submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old["identity"] != identity:
                raise ValueError(
                    f"{name} belongs to another deployment; use a new output directory"
                )
            if not old.get("job_id"):
                raise RuntimeError(
                    f"Ambiguous submission: inspect {receipt} before retrying"
                )
            return old["job_id"]
        launcher = ["srun", "--ntasks=1"]
        if devices:
            launcher.append(f"--gres=gpu:ascend910b3:{devices}")
        launcher += [
            "sudo",
            "-n",
            "slurm-docker-run",
            "--pull",
            "missing",
            "--shm-size",
            "128g",
            "--mount",
            f"{sure['slurm']['shared_root']}:{sure['slurm']['shared_root']}:rw",
            "--env",
            f"PYTHONPATH={PROJECT}",
            "--workdir",
            str(PROJECT),
            sure["slurm"]["image"],
            *argv,
        ]
        script = directory / "stage.sbatch"
        script.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(launcher) + "\n"
        )
        args = [
            "sbatch",
            "--parsable",
            "--partition=compute",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={160 if devices else 32}",
            f"--mem={'1000G' if devices else '128G'}",
            "--tmp=1T",
            "--time=01:00:00" if devices else "--time=1-00:00:00",
            "--job-name",
            f"sure-tts-{name}",
            "--output",
            str(directory / "job-%j.log"),
        ]
        if devices:
            args.append(f"--gres=gpu:ascend910b3:{devices}")
        atomic_json(receipt, {"identity": identity, "status": "submitting"})
        job = command([*args, str(script)]).split(";")[0]
        atomic_json(
            receipt, {"identity": identity, "job_id": job, "status": "submitted"}
        )
        return job


def wait_stage(job: str, receipt: Path) -> None:
    record = json.loads(receipt.read_text())
    if record.get("status") == "completed":
        return
    while True:
        state = job_state(job)
        if state == "COMPLETED":
            atomic_json(receipt, {**record, "status": "completed"})
            return
        if state not in ACTIVE:
            raise RuntimeError(
                f"Stage job {job} ended with {state}; inspect {receipt.parent}"
            )
        time.sleep(20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=["prepare", "probe", "ideas", "search", "all"], required=True
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text())
    sure = config["sure"]
    if "@sha256:" not in sure["slurm"]["image"]:
        raise ValueError("Freeze a deployment with an immutable image before running")
    with (output / "workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {
            "stage": args.stage,
            "status": "running",
            "pid": os.getpid(),
            "signature": signature(sure),
        }
        atomic_json(output / "workflow_state.json", state)
        try:
            if args.stage in {"prepare", "all"} and not prepared_ready(sure):
                data = Path(sure["data_preparation"]).parent
                argv = [
                    "python",
                    "-m",
                    "playground.sure_master.tools.prepare_task_data",
                    "tts",
                    "--root",
                    str(
                        Path(sure["slurm"]["shared_root"])
                        / "data/WenetSpeech4TTS-Premium"
                    ),
                    "--seed-root",
                    str(Path(sure["slurm"]["shared_root"]) / "data/seed-tts-eval"),
                    "--output",
                    str(data),
                    "--search-count",
                    "400",
                    "--selection-count",
                    "800",
                ]
                job = submit_stage("prepare", argv, config, output)
                wait_stage(job, output / "prepare/job.json")
            if args.stage in {"probe", "all"}:
                if not prepared_ready(sure):
                    raise ValueError(
                        "Prepare the full Premium splits before acceptance"
                    )
                argv = [
                    "python",
                    "-m",
                    "playground.sure_master.tools.probe_f5tts",
                    "--config",
                    str(args.config.resolve()),
                    "--output",
                    str(output / "probe"),
                ]
                job = submit_stage("probe", argv, config, output, devices=8)
                wait_stage(job, output / "probe/job.json")
            if args.stage in {"ideas", "search", "all"}:
                if not prepared_ready(sure):
                    raise ValueError(
                        "The full prepared dataset is required for formal search"
                    )
                evidence = json.loads((output / "probe/probe_result.json").read_text())
                if evidence.get("status") != "passed" or evidence.get(
                    "signature"
                ) != acceptance_signature(sure):
                    raise ValueError(
                        "Eight-card acceptance does not match the frozen deployment"
                    )
                from playground.sure_master.tools.with_api_profile import (
                    profile_environment,
                )
                from playground.sure_master.core.playground import SureMasterPlayground

                os.environ.update(
                    profile_environment(
                        Path(config["api_profile"]["env_file"]), "controller"
                    )
                )
                playground = SureMasterPlayground(config_path=args.config.resolve())
                playground.set_run_dir(output / "search")
                task = (
                    PROJECT / "playground/sure_master/data/tts_zh_free_evolution.md"
                ).read_text()
                if args.stage == 'ideas':
                    result = playground.run(task, ideas_only=True)
                else:
                    for job, receipt_path in sure['slurm'].get('required_takeover_receipts', {}).items():
                        receipt = json.loads(Path(receipt_path).read_text())
                        if receipt.get('job_id') != job or receipt.get('status') != 'ready':
                            raise RuntimeError(f'Allocation {job} handoff is not ready')
                    result = playground.run(task)
                atomic_json(output / ('ideas_result.json' if args.stage == 'ideas' else 'result.json'), result)
                if result.get("status") != ('ideas_ready' if args.stage == 'ideas' else 'completed'):
                    raise RuntimeError(
                        f"Evolution is incomplete: {result.get('error', result.get('status'))}"
                    )
            atomic_json(
                output / "workflow_state.json", {**state, "status": "completed"}
            )
        except Exception as exc:
            atomic_json(
                output / "workflow_state.json",
                {**state, "status": "blocked", "error": str(exc)},
            )
            raise


if __name__ == "__main__":
    main()
