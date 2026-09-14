"""Submit the TEDLIUM preparation and device gates using cluster-standard jobs."""

from __future__ import annotations
import argparse
import fcntl
import json
import os
import shlex
import time
from pathlib import Path
import yaml
from playground.sure_master.core.utils.slurm import (
    atomic_json,
    command,
    job_state,
    ACTIVE,
)

PROJECT = Path(__file__).resolve().parents[3]
ACTIVE_OUTPUT = None


def submit_gate(
    name, argv, *, config, output, npu=0, cpu=32, memory="128G", limit="04:00:00"
):
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / "job.json"
    if receipt.exists():
        return json.loads(receipt.read_text())["job_id"]
    options = ["srun", "--ntasks=1"]
    if npu:
        options.append(f"--gres=gpu:ascend910b3:{npu}")
    options += [
        "sudo",
        "-n",
        "slurm-docker-run",
        "--pull",
        "missing",
        "--shm-size",
        "128g",
        "--mount",
        f"{config['sure']['slurm'].get('shared_root', PROJECT.parent)}:{config['sure']['slurm'].get('shared_root', PROJECT.parent)}:rw",
        "--workdir",
        str(PROJECT),
        "--env",
        f"PYTHONPATH={PROJECT}",
        config["sure"]["slurm"]["image"],
        *argv,
    ]
    script = directory / "run.sbatch"
    script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(options) + "\n"
    )
    submit = [
        "sbatch",
        "--parsable",
        "--partition",
        "compute",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={cpu}",
        f"--mem={memory}",
        "--tmp=1T",
        "--time",
        limit,
        "--job-name",
        f"sure-{name}",
        "--output",
        str(directory / "job-%j.log"),
    ]
    if npu:
        submit.append(f"--gres=gpu:ascend910b3:{npu}")
    job = command([*submit, str(script)]).split(";")[0]
    atomic_json(
        receipt,
        {"job_id": job, "argv": argv, "image": config["sure"]["slurm"]["image"]},
    )
    print(json.dumps({"gate": name, "job_id": job}), flush=True)
    return job


def wait_job(job, receipt=None):
    if receipt and json.loads(receipt.read_text()).get("status") == "completed":
        return
    while True:
        state = job_state(job)
        if state == "COMPLETED":
            if receipt:
                record = json.loads(receipt.read_text())
                atomic_json(receipt, {**record, "status": "completed"})
            return
        if state not in ACTIVE:
            raise RuntimeError(f"Gate {job} did not complete: {state}")
        time.sleep(20)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/sure_master/xlab-tedlium3-npu-evolution.yaml",
    )
    p.add_argument(
        "--stage",
        choices=("prepare", "probe", "benchmark", "search", "all"),
        required=True,
    )
    p.add_argument("--output", type=Path, default=PROJECT / "runs/tedlium3_evolution")
    args = p.parse_args()
    args.output = args.output.resolve()
    if (args.output / "SUPERSEDED.json").is_file():
        raise SystemExit("This ASR workflow was superseded; use the current ASR run pointer")
    args.output.mkdir(parents=True, exist_ok=True)
    global ACTIVE_OUTPUT
    ACTIVE_OUTPUT = args.output
    driver_lock = None
    global_driver_lock = None
    active_registry = None
    if args.stage in ("all", "search"):
        driver_lock = (args.output / "workflow.lock").open("a")
        fcntl.flock(driver_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def status(phase, state="running"):
        atomic_json(
            args.output / "workflow_state.json",
            {
                "phase": phase,
                "status": state,
                "pid": os.getpid(),
                "updated_at": time.time(),
            },
        )
        if active_registry is not None:
            atomic_json(active_registry, {"run_dir": str(args.output), "phase": phase,
                                         "status": state, "pid": os.getpid(), "updated_at": time.time()})

    status(args.stage)
    config = yaml.safe_load(args.config.read_text())
    if config["sure"].get("active_registry") and args.stage == "search":
        active_registry = Path(config["sure"]["active_registry"])
        active_registry.parent.mkdir(parents=True, exist_ok=True)
        global_driver_lock = active_registry.with_suffix(".lock").open("a")
        fcntl.flock(global_driver_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status(args.stage)
    from playground.sure_master.core.full_training import promote_full_training_to_search
    config["sure"] = promote_full_training_to_search(config["sure"])
    data = config["sure"]["base_models"][config["sure"]["task_id"]]["source_paths"]["data"]
    direct_formal = config["sure"].get("startup_mode") == "direct_formal"
    if direct_formal and args.stage != "search":
        raise ValueError("direct_formal starts with --stage search; no preparation/probe/benchmark jobs")
    if args.stage in ("prepare", "all"):
        status("prepare")
        job = submit_gate(
            "prepare",
            [
                "python",
                "-m",
                "playground.sure_master.tools.prepare_tedlium",
                "--corpus",
                "/shared/chaolei.liu/data/TEDLIUM_release-3",
                "--output",
                data,
                "--train-hours",
                "0",
                "--search-hours",
                "0",
                "--seed",
                "42",
                "--jobs",
                "16",
            ],
            config=config,
            output=args.output,
            limit="1-00:00:00",
        )
        wait_job(job, args.output / "prepare/job.json")
    if args.stage in ("probe", "all"):
        status("probe")
        for size in (1, 2, 8):
            target = args.output / f"probe-{size}"
            argv = [
                "python",
                "-m",
                f'playground.sure_master.tools.probe_zipformer{"_ddp" if size>1 else ""}',
                "--icefall",
                "/shared/chaolei.liu/ASR/icefall",
                "--output",
                str(target),
            ]
            argv += ["--world-size", str(size)] if size > 1 else ["--backend", "npu"]
            job = submit_gate(
                f"probe-{size}",
                argv,
                config=config,
                output=args.output,
                npu=size,
                cpu=20 * size,
                memory="1000G" if size == 8 else "128G",
                limit="01:00:00",
            )
            if not (target / "probe_result.json").exists():
                wait_job(job, target / "job.json")
            report = json.loads((target / "probe_result.json").read_text())
            if report["status"] != "passed":
                raise RuntimeError(f"{size}-NPU probe failed")
    if args.stage in ("benchmark", "all"):
        status("benchmark")
        from playground.sure_master.runtime.distributed import backend_digest

        probe = args.output / "probe-8/probe_result.json"
        if not probe.exists():
            raise RuntimeError("Eight-card probe must pass before calibration")
        evidence = json.loads(probe.read_text())
        if (
            evidence.get("status") != "passed"
            or evidence.get("backend_digest") != backend_digest()
        ):
            raise RuntimeError(
                "Eight-card probe does not validate this backend source snapshot"
            )
        job = submit_gate(
            "benchmark",
            [
                "python",
                "-m",
                "playground.sure_master.tools.benchmark_tedlium",
                "--config",
                str(args.config.resolve()),
                "--output",
                str(args.output / "benchmark"),
            ],
            config=config,
            output=args.output,
            npu=8,
            cpu=160,
            memory="1000G",
            limit="08:00:00",
        )
        wait_job(job, args.output / "benchmark/job.json")
    if args.stage in ("search", "all"):
        status("search")
        if not direct_formal:
            if not (args.output / "probe-8/probe_result.json").is_file():
                raise RuntimeError("Pass the eight-card probe before starting evolution")
            report = json.loads((args.output / "benchmark/benchmark.json").read_text())
            if report.get("status") != "passed":
                raise RuntimeError("Common training duration has not passed calibration")
            from playground.sure_master.tools.benchmark_tedlium import training_signature
            if report.get("training_signature") != training_signature(config["sure"]):
                raise RuntimeError("Calibration is from another training data/budget; rerun benchmark for full-budget search")
        from playground.sure_master.tools.with_xlab_environment import xlab_environment, zai_environment

        api_profile = config.get("api_profile") or {}
        if api_profile.get("provider") == "zai_controller_xi_xlab":
            from playground.sure_master.tools.with_api_profile import profile_environment
            os.environ.update(profile_environment(Path(api_profile["env_file"]), "controller"))
        elif api_profile.get("provider") == "zai":
            env = zai_environment(Path(api_profile["env_file"]))
            os.environ.update(env)
            if not direct_formal:
                from playground.sure_master.tools.zai_preflight import check_zai_api
                api_result = check_zai_api(env)
                atomic_json(args.output / "api_preflight.json", api_result)
                if api_result["status"] != "passed":
                    raise RuntimeError("ZAI model access/compatibility check failed; see api_preflight.json")
        else:
            os.environ.update(xlab_environment(Path("/shared/chaolei.liu/.pi/agent")))
        from playground.sure_master.core.playground import SureMasterPlayground

        # Freeze the measured batch budget in a run-local config before any LLM call.
        if not direct_formal:
            for key in ("SURE_MAX_DURATION", "SURE_BASELINE_TRAIN_MAX_DURATION"):
                config["sure"]["execution_env"][key] = str(report["max_duration"])
            config["sure"]["execution_contract"]["max_duration"] = report["max_duration"]
        resolved = args.output / "execution.yaml"
        # Write the original credential placeholders, never expanded key values.
        resolved.write_text(yaml.safe_dump(config, sort_keys=False))
        playground = SureMasterPlayground(config_path=resolved)
        playground.set_run_dir(args.output / "search")
        task = Path(config["sure"].get("task_description_path") or (
            PROJECT / "playground/sure_master/data/asr_tedlium3_evolution_description.md"
        )).read_text()
        result = playground.run(task)
        atomic_json(args.output / "result.json", result)
        if result.get("status") != "completed":
            raise RuntimeError(
                f'Evolution incomplete: {result.get("error", result.get("status"))}'
            )
    status(args.stage, "completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if ACTIVE_OUTPUT is not None and not isinstance(exc, BlockingIOError):
            path = ACTIVE_OUTPUT / "workflow_state.json"
            state = json.loads(path.read_text()) if path.exists() else {}
            atomic_json(
                path,
                {
                    **state,
                    "status": "blocked",
                    "error": str(exc),
                    "updated_at": time.time(),
                },
            )
        raise
