"""Finish an already authorized ASR handoff after formal native ideas are ready."""
from __future__ import annotations

import argparse
import fcntl
import json
import time
from pathlib import Path

from playground.sure_master.core.utils.slurm import atomic_json
from playground.sure_master.core.utils.slurm_allocations import process_identity
from playground.sure_master.tools.asr_deployment_ops import handoff, read, register, CheckpointNotReady


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    lock = (root / "transition.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    name = read(root / "deployment.json")["name"]
    status = root / "transition_state.json"
    while True:
        if (root / "TRANSITION_STOP").exists():
            atomic_json(status, {"status": "stopped", "time": time.time()})
            return
        state_path = root / "workflow_state.json"
        state = read(state_path) if state_path.exists() else {}
        if state.get("status") == "blocked":
            atomic_json(status, {"status": "waiting_for_research_recovery", "error": state.get("error"), "time": time.time()})
        elif (root / "ideas_ready.json").exists() and state.get("status") == "ideas_ready":
            if process_identity(state.get("pid", 0)):
                time.sleep(5)
                continue
            break
        else:
            atomic_json(status, {"status": "waiting_for_formal_ideas", "time": time.time()})
        time.sleep(20)
    try:
        atomic_json(status, {"status": "handoff", "time": time.time()})
        checkpoint_deadline = time.time() + 3600
        while True:
            try:
                handoff(root)
                break
            except CheckpointNotReady as error:
                if time.time() >= checkpoint_deadline:
                    raise
                atomic_json(status, {"status": "waiting_for_complete_old_epoch", "reason": str(error), "time": time.time()})
                time.sleep(30)
        deadline = time.time() + 1800
        while time.time() < deadline:
            verified = []
            for workspace in (root / "search/workspace").glob("exp_*_improve"):
                log = workspace / "working/zipformer_candidate/train.log"
                if not log.exists():
                    continue
                lines = log.read_bytes().replace(b"\0", b"").decode(errors="replace").splitlines()
                first = next((line for line in lines if "(0/4) Epoch 1, batch 0," in line), None)
                if first and "loss=nan" not in first.lower() and "loss=inf" not in first.lower():
                    identity = read(workspace / "metric/active_worker_request.json")["identity"]
                    allocation = read(workspace / "metric/slurm" / identity / "allocation.json")
                    verified.append({"workspace": str(workspace), "job_id": allocation["job_id"],
                                     "slot": allocation["slot"], "first_training_line": first})
            if len(verified) == 4 and len({(v["job_id"], v["slot"]) for v in verified}) == 4:
                atomic_json(root / "training_verification.json", {"verified_at": time.time(), "candidates": verified})
                register(root, name, "running")
                monitor = Path("/shared/chaolei.liu/asr-monitor/formal-v2/task.md")
                with monitor.open("a") as handle:
                    handle.write(f"\nActive deployment has changed to {name}: {root}. "
                        "Monitor this run only; ASR-Direct-v1 is archived and MUST NOT be resumed automatically. "
                        "Native five-mode MCTS: 8 iterations per mode, independent 6x4 search from imported baseline. "
                        "No FLOW_FIRST fallback. Check training_verification.json and the current pointer.\n")
                atomic_json(status, {"status": "verified", "time": time.time()})
                return
            state = read(root / "workflow_state.json")
            if state.get("status") == "blocked":
                raise RuntimeError(state.get("error", "New training controller blocked"))
            time.sleep(20)
        raise TimeoutError("Four native training workers were not verified within 30 minutes; allocations retained")
    except Exception as error:
        atomic_json(status, {"status": "blocked", "error": str(error), "time": time.time()})
        raise


if __name__ == "__main__":
    main()
