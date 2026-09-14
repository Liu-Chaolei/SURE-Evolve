"""Run exclusive candidate steps inside explicitly configured owned allocations."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time


def process_identity(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except FileNotFoundError:
        return None


def owned_running(job: str) -> bool:
    from .slurm import command

    try:
        info = command(["scontrol", "show", "job", str(job), "-o"])
    except (RuntimeError, subprocess.CalledProcessError):
        return False
    owner = re.search(r"UserId=\S+\((\d+)\)", info)
    if not owner or int(owner[1]) != os.getuid():
        raise ValueError(f"Allocation {job} does not belong to this user")
    return "JobState=RUNNING " in info


def run_in_allocations(settings: dict, argv: list[str], root: Path, result: Path) -> bool:
    """Return False only when allocations expired, allowing standard resume submission.

    The srun child inherits the node lock, so controller restarts cannot double book it.
    A receipt reconnects to a surviving srun without launching a duplicate worker.
    """
    from .slurm import atomic_json

    jobs = [str(job) for job in settings["existing_allocations"]]
    restricted = settings.get("sd_restricted_pool", False)
    if restricted:
        from .sd_allocations import validate_jobs
        jobs = validate_jobs(settings)
    slots = int(settings.get("allocation_slots_per_job", 1))
    if slots < 1 or (slots > 1 and settings.get("allocation_overlap", True)):
        raise ValueError("Multiple allocation slots require exclusive steps")
    pool = Path(settings["allocation_lock_dir"])
    pool.mkdir(parents=True, exist_ok=True)
    receipt = root / "allocation.json"
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old.get("status") == "starting":
            raise RuntimeError(f"Ambiguous allocation launch; reconcile {receipt}")
        while old.get("pid") and process_identity(old["pid"]) == old.get("process_identity"):
            time.sleep(5)
        if result.exists():
            return True
        if owned_running(old["job_id"]):
            raise RuntimeError(f"Allocation step ended without a result; see {root}")
        if restricted:
            receipt.rename(root / f"expired-allocation-{time.time_ns()}.json")
        else:
            return False
    while True:
        from .allocation_handoff import ready_allocation

        acquired = ready_allocation(settings)
        available_jobs = jobs + ([acquired] if acquired and acquired not in jobs else [])
        active = False
        for job, slot in [(job, slot) for job in available_jobs for slot in range(slots)]:
            if not owned_running(job):
                continue
            if restricted:
                from .sd_allocations import acquire
                if not acquire(job, pool):
                    continue
            active = True
            if slots > 1:
                with (pool / f"{job}.lock").open("a") as legacy:
                    try:
                        fcntl.flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
            lock_name = f"{job}.lock" if slots == 1 else f"{job}.slot-{slot}.lock"
            with (pool / lock_name).open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if not owned_running(job):
                    continue
                sharing = "--overlap" if settings.get("allocation_overlap", True) else "--exclusive"
                command = [argv[0], f"--jobid={job}", sharing, "--exact",
                           f"--cpus-per-task={settings.get('allocation_cpus', 64)}"]
                if settings.get("allocation_step_memory"):
                    command.append(f"--mem={settings['allocation_step_memory']}")
                command.extend(argv[1:])
                record = {"job_id": job, "slot": slot, "status": "starting", "command": command,
                          "started_at": time.time()}
                atomic_json(receipt, record)
                with (root / f"allocation-{job}.log").open("ab") as log:
                    proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True,
                                            pass_fds=(lock.fileno(),))
                record.update(pid=proc.pid, process_identity=process_identity(proc.pid), status="running")
                atomic_json(receipt, record)
                rc = proc.wait()
                atomic_json(receipt, {**record, "status": "finished", "exit_code": rc})
                if rc != 0 and not owned_running(job):
                    if result.exists():
                        result.rename(root / f"allocation-{job}-interrupted-result.json")
                    if restricted:
                        receipt.rename(root / f"expired-allocation-{time.time_ns()}.json")
                        continue
                    return False
                if result.exists():
                    return True
                if not owned_running(job):
                    if restricted:
                        receipt.rename(root / f"expired-allocation-{time.time_ns()}.json")
                        continue
                    return False
                raise RuntimeError(f"Allocation {job} step exited {rc}; see {root}")
        if not active:
            if restricted:
                time.sleep(10)
                continue
            return False
        if settings.get("authorized_service_takeover") and not acquired:
            # All existing slots are leased and this candidate has a durable request.
            if ready_allocation(settings, acquire=True):
                continue
        if settings.get("allocation_fallback_when_busy") and not restricted:
            # Other ready candidates can queue normally while the lease is busy.
            return False
        time.sleep(5)
