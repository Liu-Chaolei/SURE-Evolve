"""Preserve an explicitly authorized service allocation for candidate steps."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

from .slurm import atomic_json, command


def eligible(info: str, job: str) -> bool:
    if job != "10028":
        raise ValueError("Only explicitly authorized Qwen allocation 10028 may be acquired")
    owner = re.search(r"UserId=\S+\((\d+)\)", info)
    node = re.search(r"\bNodeList=(\S+)", info)
    if not owner or int(owner[1]) != os.getuid():
        raise ValueError("Handoff allocation owner differs")
    if "JobState=RUNNING " not in info:
        return False
    if not node or not re.fullmatch(r"n\d+", node[1]) or node[1] in {"n11", "n13"}:
        raise ValueError("Handoff would affect a protected or unexpected node")
    if "NumCPUs=64 " not in info or "mem=256G" not in info or "gres/gpu:ascend910b3:8" not in info:
        raise ValueError("Handoff resource limits changed")
    return True


def remote(job: str, args: list[str]) -> str:
    return command(["srun", f"--jobid={job}", "--overlap", "--exact",
                    "--cpus-per-task=1", "--gres=none", "--ntasks=1", *args])


def ready_allocation(settings: dict, *, acquire: bool = False) -> str | None:
    target = settings.get("authorized_service_takeover")
    if not target:
        return None
    job = str(target["job_id"])
    directory = Path(target["receipt_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{job}.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        path = directory / f"{job}.json"
        record = json.loads(path.read_text()) if path.exists() else {}
        if not record and not acquire:
            return None
        try:
            info = command(["scontrol", "show", "job", job, "-o"])
        except (RuntimeError, subprocess.CalledProcessError):
            return None
        if not eligible(info, job):
            return None
        if record.get("status") == "ready":
            if "JobName=asr-v2-allocation " not in info:
                raise ValueError("Previously acquired allocation changed identity")
            return job
        if not acquire:
            return None
        if not record:
            if "JobName=qwen38-27b-w8a8-tp8 " not in info:
                raise ValueError("Allocation no longer runs the authorized Qwen service")
            steps = command(["scontrol", "show", "step", job, "-o"])
            step = next((line for line in steps.splitlines() if line.startswith(f"StepId={job}.0 ")), "")
            match = re.search(r"SrunHost:Pid=\S+:(\d+)", step)
            if not match:
                return None
            pid = int(match[1])
            if remote(job, ["ps", "-p", str(pid), "-o", "comm="]).strip() != "srun":
                raise ValueError("Unexpected service holder process")
            cgroup = remote(job, ["cat", f"/proc/{pid}/cgroup"])
            if f"/job_{job}/step_batch" not in cgroup:
                raise ValueError("Holder does not belong to this batch allocation")
            identity = remote(job, ["cat", f"/proc/{pid}/stat"]).rsplit(") ", 1)[1].split()[19]
            record = {"job_id": job, "node": re.search(r"\bNodeList=(\S+)", info)[1],
                      "holder_pid": pid, "holder_identity": identity,
                      "service_step": job + ".0", "status": "prepared", "created_at": time.time()}
            atomic_json(path, record)
        pid = record["holder_pid"]
        stat = remote(job, ["cat", f"/proc/{pid}/stat"]).rsplit(") ", 1)[1].split()
        if stat[19] != record["holder_identity"]:
            raise ValueError("Service holder PID was reused")
        remote(job, ["kill", "-STOP", str(pid)])
        if "T" not in remote(job, ["ps", "-p", str(pid), "-o", "stat="]):
            raise RuntimeError("Could not preserve the batch holder")
        record.update(status="holder_stopped")
        atomic_json(path, record)
        for attempt in range(30):
            steps = command(["scontrol", "show", "step", job, "-o"])
            if not any(line.startswith(f"StepId={job}.0 ") for line in steps.splitlines()):
                break
            if attempt in {0, 20}:
                command(["scancel", "--signal=" + ("TERM" if attempt == 0 else "KILL"), job + ".0"])
            time.sleep(2)
        else:
            raise RuntimeError("Qwen service did not release its step; allocation remains preserved")
        command(["scontrol", "update", f"JobId={job}", "JobName=asr-v2-allocation"])
        record.update(status="ready", ready_at=time.time())
        atomic_json(path, record)
        return job
