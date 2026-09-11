"""Durable Slurm candidate submission, without model imports or credentials."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[4]
ACTIVE = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "SUSPENDED",
    "REQUEUED",
    "RESIZING",
}


def source_digest(root: Path | None = None) -> str:
    root = root or PROJECT
    value = hashlib.sha256()
    for package in (root / "evomaster", root / "playground/sure_master"):
        for path in sorted(package.rglob("*.py")):
            if "__pycache__" not in path.parts:
                value.update(str(path.relative_to(root)).encode())
                value.update(path.read_bytes())
    return value.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    pending.replace(path)


def command(args: list[str]) -> str:
    return subprocess.run(
        args, check=True, text=True, capture_output=True, timeout=60
    ).stdout.strip()


def job_state(job: str) -> str:
    try:
        state = command(["squeue", "-h", "-j", job, "-o", "%T"])
        if state:
            return state.splitlines()[0]
    except subprocess.CalledProcessError:
        pass
    try:
        output = command(["scontrol", "show", "job", job, "-o"])
        match = re.search(r"JobState=(\S+)", output)
        if match:
            return match.group(1)
    except subprocess.CalledProcessError:
        pass
    try:
        rows = command(
            ["sacct", "-n", "-X", "-j", job, "--format=State", "--parsable2"]
        ).splitlines()
        if rows:
            return rows[0].split("|")[0].split()[0].rstrip("+")
    except subprocess.CalledProcessError:
        pass
    return "UNKNOWN"


def resource_profile(settings: dict, candidate_type: str) -> dict:
    training = candidate_type != "inference"
    defaults = (
        dict(npu=8, cpu=160, memory="1000G", temporary="1T", shm="128g")
        if training
        else dict(npu=1, cpu=20, memory="128G", temporary="100G", shm="32g")
    )
    defaults.update(
        settings.get("resource_profiles", {}).get(
            "training" if training else "inference", {}
        )
    )
    if training and int(defaults["npu"]) != 8:
        raise ValueError("Production TEDLIUM training requires eight allocated NPUs")
    return defaults


def batch_script(settings: dict, request: Path, workspace: Path, profile: dict) -> str:
    image = settings["image"]
    if "@sha256:" not in image:
        raise ValueError("Slurm execution requires an immutable image digest")
    argv = [
        "srun",
        "--ntasks=1",
        f"--gres=gpu:ascend910b3:{profile['npu']}",
        "sudo",
        "-n",
        "slurm-docker-run",
        "--pull",
        "missing",
        "--shm-size",
        profile["shm"],
        "--mount",
        f"{settings.get('shared_root', PROJECT.parent)}:{settings.get('shared_root', PROJECT.parent)}:ro",
        "--mount",
        f"{workspace}:{workspace}:rw",
        "--mount",
        f'{workspace.parent / "model_artifacts"}:{workspace.parent / "model_artifacts"}:rw',
        "--workdir",
        str(workspace),
        "--env",
        f"PYTHONPATH={PROJECT}",
        image,
        "python",
        "-m",
        "playground.sure_master.tools.slurm_candidate",
        "--request",
        str(request),
    ]
    return "#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(argv) + "\n"


def run_candidate(exp) -> dict:
    config = (
        exp.config.model_dump() if hasattr(exp.config, "model_dump") else exp.config
    )
    sure = config["sure"]
    settings = sure["slurm"]
    workspace = Path(exp.workspace_path).resolve()
    profile = resource_profile(settings, exp.candidate_type_hint)
    context = json.loads(
        (workspace / "metric/remote_candidate_context.json").read_text()
    )
    env = context["execution_env"]
    for key in list(env):
        if key in {
            "ASCEND_VISIBLE_DEVICES",
            "ASCEND_RT_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        }:
            env.pop(key)
        elif any(word in key.upper() for word in ("API_KEY", "PASSWORD", "SECRET")):
            raise ValueError(f"Credentials must not be passed to model workers: {key}")
    env.update(
        SURE_FRAMEWORK_DIGEST=source_digest(), SURE_RUNTIME_IMAGE=settings["image"]
    )
    env.update(
        ASR_WORLD_SIZE=str(profile["npu"]),
        SURE_BASELINE_WORLD_SIZE=str(profile["npu"]),
        SURE_WORKER_PYTHON="python",
        SURE_ICEFALL_PYTHON="python",
        SURE_CANDIDATE_PYTHON="python",
        SURE_CPU_THREADS="8",
        OMP_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
    )
    # No controller LLM settings or credentials in the worker request.
    worker_sure = {
        k: v
        for k, v in sure.items()
        if k
        not in {"slurm", "preflight", "full_training", "datasets", "final_evaluation"}
    }
    worker_sure.update(execution_mode="local", remote_training={"enabled": False})
    payload = {
        "sure": worker_sure,
        "context": context,
        "workspace": str(workspace),
        "code": exp.code,
        "image": settings["image"],
        "profile": profile,
        "source_digest": env["SURE_FRAMEWORK_DIGEST"],
    }
    # Timing is observational, not part of the operation identity.
    payload["context"].pop("candidate_started_at", None)
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    root = workspace / "metric/slurm" / identity
    root.mkdir(parents=True, exist_ok=True)
    (workspace.parent / "model_artifacts").mkdir(exist_ok=True)
    request, receipt, result = (
        root / "request.json",
        root / "job.json",
        root / "result.json",
    )
    payload["result"] = str(result)
    with (root / "lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if result.is_file():
            return json.loads(result.read_text())
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if not record.get("job_id") or record.get("next_submission_pending"):
                raise RuntimeError(
                    f"Ambiguous Slurm submission; reconcile {receipt} before retrying"
                )
        else:
            atomic_json(request, payload)
            script = root / "candidate.sbatch"
            script.write_text(batch_script(settings, request, workspace, profile))
            record = {
                "identity": identity,
                "status": "submitting",
                "submitted_at": time.time(),
            }
            atomic_json(receipt, record)
            job = command(
                [
                    "sbatch",
                    "--parsable",
                    "--partition",
                    settings.get("partition", "compute"),
                    "--nodes=1",
                    "--ntasks=1",
                    f"--cpus-per-task={profile['cpu']}",
                    f"--mem={profile['memory']}",
                    f"--tmp={profile['temporary']}",
                    f"--gres=gpu:ascend910b3:{profile['npu']}",
                    "--time",
                    settings.get("time_limit", "1-00:00:00"),
                    "--job-name",
                    "sure-" + identity[:12],
                    "--output",
                    str(root / "job-%j.log"),
                    str(script),
                ]
            ).split(";")[0]
            if not job.isdigit():
                raise RuntimeError(f"Invalid sbatch response: {job}")
            record.update(job_id=job, status="submitted")
            atomic_json(receipt, record)
    job = record["job_id"]
    while True:
        if result.exists():
            return json.loads(result.read_text())
        state = job_state(job)
        record.update(status=state or "accounting_pending", checked_at=time.time())
        atomic_json(receipt, record)
        if state and state not in ACTIVE:
            # Resume infrastructure interruptions, never a user cancellation.
            recoverable = state == "TIMEOUT" or (
                state in {"NODE_FAIL", "PREEMPTED"}
                and record.get("system_retries", 0) < 2
            )
            if recoverable and int(record.get("segments", 1)) < int(
                settings.get("max_segments", 30)
            ):
                record.setdefault("previous_jobs", []).append(
                    {"job_id": job, "state": state}
                )
                record["next_submission_pending"] = True
                if state != "TIMEOUT":
                    record["system_retries"] = record.get("system_retries", 0) + 1
                atomic_json(receipt, record)
                job = command(
                    [
                        "sbatch",
                        "--parsable",
                        "--partition",
                        settings.get("partition", "compute"),
                        "--nodes=1",
                        "--ntasks=1",
                        f"--cpus-per-task={profile['cpu']}",
                        f"--mem={profile['memory']}",
                        f"--tmp={profile['temporary']}",
                        f"--gres=gpu:ascend910b3:{profile['npu']}",
                        "--time",
                        settings.get("time_limit", "1-00:00:00"),
                        "--job-name",
                        "sure-" + identity[:12],
                        "--output",
                        str(root / "job-%j.log"),
                        str(root / "candidate.sbatch"),
                    ]
                ).split(";")[0]
                if not job.isdigit():
                    raise RuntimeError(
                        "Ambiguous Slurm resubmission; reconcile the receipt"
                    )
                record.update(
                    job_id=job,
                    segments=int(record.get("segments", 1)) + 1,
                    next_submission_pending=False,
                )
                atomic_json(receipt, record)
                continue
            raise RuntimeError(
                f"Slurm job {job} ended {state} without a published result; see {root}"
            )
        time.sleep(max(5, min(60, int(settings.get("poll_seconds", 20)))))
