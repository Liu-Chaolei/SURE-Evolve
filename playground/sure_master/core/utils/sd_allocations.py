"""Lease only the explicitly authorized SD service allocations."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import time

AUTHORIZED = frozenset({"10085", "10086", "10087", "10088", "10094", "10095", "10096", "10097"})


def validate_jobs(settings: dict) -> list[str]:
    jobs = [str(job) for job in settings.get("existing_allocations", [])]
    if not jobs or len(jobs) != len(set(jobs)) or not set(jobs) <= AUTHORIZED:
        raise ValueError("SD allocation pool contains a job outside the authorized whitelist")
    if settings.get("allocation_slots_per_job", 1) != 1 or settings.get("allocation_overlap", False):
        raise ValueError("SD requires one exclusive candidate per allocation")
    return jobs


def job_info(job: str) -> dict:
    from .slurm import command
    if job not in AUTHORIZED:
        raise ValueError("Unauthorized SD allocation")
    raw = command(["scontrol", "show", "job", job, "-o"])
    values = dict(re.findall(r"(?:^|\s)(\w+)=([^\s]*)", raw))
    owner = re.search(r"\((\d+)\)$", values.get("UserId", ""))
    if not owner or int(owner[1]) != os.getuid():
        raise ValueError("SD allocation owner changed")
    if values.get("NumCPUs") != "64" or "mem=256G" not in raw or "gres/gpu:ascend910b3:8" not in raw:
        raise ValueError("SD allocation resources changed")
    return values


def acquire(job: str, directory: Path) -> bool:
    """Durable, identity-checked service-step handoff; never cancel an allocation."""
    from .allocation_handoff import remote
    from .slurm import atomic_json, command
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"handoff-{job}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        info = job_info(job)
        if info["JobState"] != "RUNNING":
            return False
        path = directory / f"handoff-{job}.json"
        record = json.loads(path.read_text()) if path.exists() else {}
        if record.get("status") == "ready":
            if record["node"] != info["NodeList"] or info["JobName"] != "sd-ablation-allocation":
                raise ValueError("Acquired SD allocation identity changed")
            return True
        if not record:
            if info["JobName"] != "qwen38-27b-w8a8-tp8":
                raise ValueError("SD takeover target is not the expected service")
            steps = command(["scontrol", "show", "step", job, "-o"])
            step = next((line for line in steps.splitlines() if line.startswith(f"StepId={job}.0 ")), "")
            found = re.search(r"SrunHost:Pid=\S+:(\d+)", step)
            if not found:
                return False
            pid = int(found[1])
            if remote(job, ["ps", "-p", str(pid), "-o", "comm="]).strip() != "srun":
                raise ValueError("Unexpected service holder")
            if f"/job_{job}/step_batch" not in remote(job, ["cat", f"/proc/{pid}/cgroup"]):
                raise ValueError("Service holder belongs to another allocation")
            stat = remote(job, ["cat", f"/proc/{pid}/stat"]).rsplit(") ", 1)[1].split()
            record = {"job_id": job, "node": info["NodeList"], "holder_pid": pid,
                      "holder_identity": stat[19], "status": "prepared"}
            atomic_json(path, record)
        pid = record["holder_pid"]
        stat = remote(job, ["cat", f"/proc/{pid}/stat"]).rsplit(") ", 1)[1].split()
        if stat[19] != record["holder_identity"]:
            raise ValueError("Holder process identity changed")
        remote(job, ["kill", "-STOP", str(pid)])
        if "T" not in remote(job, ["ps", "-p", str(pid), "-o", "stat="]):
            raise RuntimeError("Cannot preserve SD allocation holder")
        record["status"] = "holder_stopped"
        atomic_json(path, record)
        for attempt in range(30):
            steps = command(["scontrol", "show", "step", job, "-o"])
            if not any(line.startswith(f"StepId={job}.0 ") for line in steps.splitlines()):
                break
            if attempt in {0, 20}:
                command(["scancel", "--signal=" + ("TERM" if attempt == 0 else "KILL"), f"{job}.0"])
            time.sleep(2)
        else:
            raise RuntimeError("Service step did not exit; allocation retained")
        command(["scontrol", "update", f"JobId={job}", "JobName=sd-ablation-allocation"])
        record.update(status="ready", ready_at=time.time())
        atomic_json(path, record)
        return True
