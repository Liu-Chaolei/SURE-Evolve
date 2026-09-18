"""Explicit ASR deployment registration, checkpoint pause, resume and allocation handoff."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import yaml

from playground.sure_master.core.artifacts import file_digest
from playground.sure_master.core.utils.slurm import atomic_json
from playground.sure_master.core.utils.slurm_allocations import process_identity, owned_running
from playground.sure_master.core.utils.fingerprints import digest, canonical_json

PROJECT = Path(__file__).resolve().parents[3]
PYTHON = PROJECT.parent / "data/sure_asr_controller/bin/python"
REGISTRY = PROJECT / "runs/asr_deployments.json"
POINTER = PROJECT / "runs/current_asr_evolution.json"
JOBS = {"10023", "10028"}


class CheckpointNotReady(RuntimeError):
    """The old deployment must keep training until a complete epoch is available."""


def read(path):
    return json.loads(path.read_text())


def checkpoint_response(stdout):
    for line in reversed(stdout.splitlines()):
        if line.startswith("ASR_CHECKPOINT_JSON="):
            record = json.loads(line.split("=", 1)[1])
            if set(("checkpoint", "epoch", "next_epoch", "sha256", "batch_idx_train")) <= set(record):
                return record
    raise ValueError("Checkpoint probe did not return a validation record")


def register(root, name, status):
    lock = REGISTRY.with_suffix(".lock").open("a")
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = read(REGISTRY) if REGISTRY.exists() else {"deployments": {}}
        prior = value["deployments"].get(name)
        if prior and prior["run_dir"] != str(root):
            raise ValueError("Deployment name already belongs to a different directory")
        value["deployments"][name] = {"run_dir": str(root), "status": status, "updated_at": time.time()}
        atomic_json(REGISTRY, value)
    atomic_json(root / "deployment_name.json", {"name": name, "run_dir": str(root)})


def active_workers(root):
    result = []
    for workspace in (root / "search/workspace").glob("exp_*_improve"):
        owner = workspace / "metric/active_worker_request.json"
        if not owner.exists():
            continue
        identity = read(owner)["identity"]
        directory = workspace / "metric/slurm" / identity
        if (directory / "result.json").exists() and read(directory / "result.json").get("success"):
            continue
        allocation = directory / "allocation.json"
        if not allocation.exists():
            continue
        record = read(allocation)
        if record["job_id"] not in JOBS:
            raise ValueError("Unexpected ASR allocation")
        if process_identity(record.get("pid", 0)) == record.get("process_identity") and record.get("process_identity"):
            result.append({"workspace": str(workspace), "identity": identity, "directory": str(directory), **record})
    return result


def launch(root, ideas_only=False):
    if (root / "PAUSED.json").exists():
        raise ValueError("Explicit resume is required for a paused deployment")
    state = read(root / "workflow_state.json") if (root / "workflow_state.json").exists() else {}
    if state.get("status") == "running" and process_identity(state.get("pid", 0)):
        raise ValueError("Controller is already running")
    script = root / ("prepare_ideas.sh" if ideas_only else "launch.sh")
    with (root / "controller.log").open("ab") as log:
        process = subprocess.Popen(["bash", str(script)], cwd=PROJECT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    atomic_json(root / "launcher.json", {"pid": process.pid, "process_identity": process_identity(process.pid),
                                        "ideas_only": ideas_only, "started_at": time.time()})
    return process.pid


def pause(root):
    if (root / "PAUSED.json").exists():
        return read(root / "PAUSED.json")
    state = read(root / "workflow_state.json")
    config = yaml.safe_load((root / "execution.yaml").read_text())
    name = read(root / "deployment_name.json")["name"] if (root / "deployment_name.json").exists() else "ASR-Direct-v1"
    workers = active_workers(root)
    # Validate all checkpoints before stopping anything. No NPU computation is requested.
    for worker in workers:
        job = worker["job_id"]
        if not owned_running(job):
            raise ValueError("Owned allocation is not running")
        argv = ["srun", f"--jobid={job}", "--overlap", "--exact", "-c1", "--gres=none", "--ntasks=1",
            "sudo", "-n", "slurm-docker-run", "--pull", "missing", "--mount",
            f"{PROJECT.parent}:{PROJECT.parent}:ro", config["sure"]["slurm"]["image"],
            "python", "-P", str(Path(__file__).with_name("asr_checkpoint_probe.py")),
            "--workspace", worker["workspace"]]
        response = subprocess.run(argv, capture_output=True, text=True, timeout=600)
        if response.returncode:
            if "No complete resumable epoch checkpoint" in response.stdout + response.stderr:
                raise CheckpointNotReady("Waiting for a complete old epoch checkpoint")
            raise RuntimeError("Checkpoint validation failed: " + response.stderr[-1500:])
        worker["resume"] = checkpoint_response(response.stdout)
        worker["checkpoint_probe_command"] = argv
    known = {worker["identity"] for worker in workers}
    if any(worker["identity"] not in known for worker in active_workers(root)):
        raise CheckpointNotReady("Old round changed during checkpoint validation; retry against its new workers")
    directory = root / "preservation" / str(time.time_ns())
    directory.mkdir(parents=True)
    record = {"name": name, "run_dir": str(root), "paused_at": time.time(),
              "archive": str(directory), "workers": workers, "original_state": state}
    atomic_json(root / "PAUSING.json", record)
    pid = state.get("pid", 0)
    if process_identity(pid):
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
        if not any(entry in command for entry in (b"controller_entry.py", b"asr_native_controller")) or str(root).encode() not in command:
            raise ValueError("Old controller identity does not match its deployment")
        group = os.getpgid(pid)
        leader = Path(f"/proc/{group}/cmdline")
        if group <= 1 or group == os.getpgrp() or not leader.exists() or str(root).encode() not in leader.read_bytes():
            raise ValueError("Controller process group is not isolated to this deployment")
        os.killpg(group, signal.SIGTERM)
        for _ in range(60):
            if not process_identity(pid):
                break
            time.sleep(1)
        if process_identity(pid):
            raise RuntimeError("Old controller did not stop")
    for worker in workers:
        pid = worker["pid"]
        if process_identity(pid) != worker["process_identity"]:
            continue
        if os.getpgid(pid) != pid or b"srun" not in Path(f"/proc/{pid}/cmdline").read_bytes():
            raise ValueError("Candidate srun process identity changed")
        os.killpg(pid, signal.SIGTERM)
    for worker in workers:
        for _ in range(120):
            if process_identity(worker["pid"]) != worker["process_identity"]:
                break
            time.sleep(1)
        if process_identity(worker["pid"]) == worker["process_identity"]:
            raise RuntimeError("Candidate step did not stop; parent allocation preserved")
        finished = Path(worker["directory"]) / "result.json"
        if finished.exists() and read(finished).get("success"):
            worker["completed_during_pause"] = True
            continue
        response = subprocess.run(worker["checkpoint_probe_command"], capture_output=True,
                                  text=True, check=True, timeout=600)
        worker["resume"] = checkpoint_response(response.stdout)
        source = Path(worker["resume"]["checkpoint"])
        if file_digest(source) != worker["resume"]["sha256"]:
            raise ValueError("Selected checkpoint changed during pause")
        target = directory / Path(worker["workspace"]).name / source.name
        target.parent.mkdir(parents=True)
        shutil.copy2(source, target)
        worker["resume"]["backup"] = str(target)
    for name in ("deployment.yaml", "execution.yaml", "controller_entry.py", "resume_controller.py",
                 "source_manifest.json", "xlab_source_manifest.json", "controller_adapter_manifest.json",
                 "workflow_state.json", "recovery.json"):
        source = root / name
        if source.exists():
            shutil.copy2(source, directory / name)
    for name in ("metric", "artifacts"):
        source = root / "search/workspace" / name
        if source.exists():
            shutil.copytree(source, directory / name)
    receipts = root / "search/workspace/artifacts/xlab_operations.json"
    record["api_operations"] = ({key: value for key, value in read(receipts).get("operations", {}).items()
                                  if value.get("status") != "published"} if receipts.exists() else {})
    files = {str(p.relative_to(root)): {"size": p.stat().st_size,
             "sha256": file_digest(p) if p.suffix != ".pt" and p.stat().st_size < 100_000_000 else None}
             for p in root.rglob("*") if p.is_file() and "preservation" not in p.parts and "__pycache__" not in p.parts}
    atomic_json(directory / "inventory.json", files)
    atomic_json(root / "PAUSED.json", record)
    atomic_json(root / "workflow_state.json", {**state, "status": "paused", "updated_at": time.time()})
    if read(POINTER).get("run_dir") == str(root):
        atomic_json(POINTER, {**read(POINTER), "status": "paused"})
    register(root, name, "paused")
    return record


def resume_old(root):
    paused = read(root / "PAUSED.json")
    current = read(POINTER)
    if current.get("run_dir") != str(root) and current.get("status") not in {"paused", "completed"}:
        raise ValueError("Another deployment owns the active allocation; pause it before resuming the old run")
    for worker in paused["workers"]:
        directory = Path(worker["directory"])
        if (directory / "result.json").exists() and read(directory / "result.json").get("success"):
            continue
        if process_identity(worker["pid"]) == worker["process_identity"]:
            raise ValueError("Paused candidate is still running")
        checkpoint = Path(worker["resume"]["checkpoint"])
        if file_digest(checkpoint) != worker["resume"]["sha256"]:
            raise ValueError("Resume checkpoint identity changed")
        # This pause promises epoch-boundary recovery; keep later batch snapshots as evidence,
        # but do not let automatic checkpoint discovery select an unverified mid-epoch state.
        for batch in checkpoint.parent.glob("checkpoint-*.pt"):
            backup = Path(paused["archive"]) / Path(worker["workspace"]).name / batch.name
            if backup.exists():
                raise ValueError("Batch checkpoint archive collision")
            batch.rename(backup)
        for name in ("allocation.json", "result.json"):
            path = directory / name
            if path.exists():
                path.rename(directory / f"paused-{time.time_ns()}-{name}")
    if paused["name"] == "ASR-Direct-v1" and paused.get("api_operations"):
        receipts = root / "search/workspace/artifacts/xlab_operations.json"
        current_receipts = read(receipts)
        shutil.copy2(receipts, Path(paused["archive"]) / f"before-api-resume-{time.time_ns()}.json")
        for operation, saved in paused["api_operations"].items():
            present = current_receipts["operations"].get(operation)
            if present and present.get("status") == "published":
                continue
            if present != saved or Path(operation).name != operation:
                raise ValueError("Paused API receipt changed; explicit reconciliation required")
            state = root / "xlab_operations" / operation / "generation.json"
            if state.exists() and read(state).get("status") != "published":
                # Completed response caches remain in place and are replayed by the old adapter.
                state.rename(state.with_name(f"paused-{time.time_ns()}-generation.json"))
            del current_receipts["operations"][operation]
        current_receipts["document_digest"] = digest({k: v for k, v in current_receipts.items() if k != "document_digest"})
        receipts.write_text(canonical_json(current_receipts) + "\n")
    (root / "PAUSED.json").rename(root / f"resumed-{time.time_ns()}.json")
    atomic_json(POINTER, {"run_dir": str(root), "status": "resuming", "pid": None})
    if paused["name"].startswith("ASR-Native-MCTS-"):
        launch(root)
    else:
        subprocess.run([str(PYTHON), "-P", str(root / "resume_controller.py")], check=True)
    register(root, paused["name"], "resuming")


def handoff(root):
    metadata = read(root / "deployment.json")
    old = Path(metadata["old_run"])
    ready = read(root / "ideas_ready.json")
    if ready.get("status") != "ideas_ready" or ready.get("candidate_count") != 4:
        raise ValueError("Four fully reviewed native candidates are required before handoff")
    for job in JOBS:
        if not owned_running(job):
            raise ValueError("Expected allocation no longer available")
    if metadata["name"] == "ASR-Native-MCTS-v2":
        state = read(old / "workflow_state.json")
        if state.get("status") != "completed" or process_identity(state.get("pid", 0)):
            raise CheckpointNotReady("Waiting for old final selection/test evaluation to finish")
        register(old, "ASR-Direct-v1", "completed")
    else:
        pause(old)
    (root / "TRAINING_ENABLED").write_text("Owned allocations handed off from ASR-Direct-v1\n")
    pid = launch(root)
    register(root, metadata["name"], "starting")
    atomic_json(root / "handoff.json", {"old_run": str(old), "jobs": sorted(JOBS), "launcher_pid": pid,
                                       "completed_at": time.time()})
    return pid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("register", "status", "prepare-ideas", "pause", "resume", "handoff"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "register":
        register(root, "ASR-Direct-v1", "running")
    elif args.action == "status":
        print(json.dumps({"state": read(root / "workflow_state.json"), "active_workers": active_workers(root)}))
    elif args.action == "prepare-ideas":
        print(launch(root, True))
    elif args.action == "pause":
        print(json.dumps(pause(root)))
    elif args.action == "resume":
        resume_old(root)
    elif args.action == "handoff":
        print(handoff(root))


if __name__ == "__main__":
    main()
