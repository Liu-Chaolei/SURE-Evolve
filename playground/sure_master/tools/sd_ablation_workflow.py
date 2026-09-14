"""Run or resume the frozen SD ablation deployment and its holdout barrier."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from playground.sure_master.core.utils.slurm import atomic_json
from playground.sure_master.core.utils.slurm_allocations import process_identity


def run_arm(root: Path, stage: str) -> None:
    from playground.sure_master.core.playground import SureMasterPlayground
    directory = root / stage
    configuration = directory / "deployment.yaml"
    from playground.sure_master.core.utils.sd_allocations import job_info
    settings = yaml.safe_load(configuration.read_text())["sure"]["slurm"]
    while True:
        states = {}
        for job in settings["existing_allocations"]:
            try:
                info = job_info(str(job))
                states[str(job)] = {"state": info["JobState"], "node": info.get("NodeList", "")}
            except subprocess.CalledProcessError:
                states[str(job)] = {"state": "UNKNOWN", "node": ""}
        ready = any(value["state"] == "RUNNING" for value in states.values())
        atomic_json(directory / "resources.json", {"status": "available" if ready else "waiting",
                    "jobs": states, "updated_at": time.time()})
        if ready:
            break
        time.sleep(10)
    playground = SureMasterPlayground(config_path=configuration)
    playground.set_run_dir(directory / "search")
    if stage != "baseline":
        # The same frozen baseline is scored on the node/lease used by this arm
        # before any hypothesis is generated. This uses no research API calls.
        playground.setup()
        from playground.sure_master.core.sd_baseline_import import import_sd_baseline
        baseline, _ = import_sd_baseline(root / "baseline", playground.sure_config,
                                         Path(playground.session.config.workspace_path))
        check = directory / "baseline_replay.json"
        if not check.exists():
            exp = playground._create_run_exp("improve", 900000)
            exp.candidate_phase = "search"
            exp.execution_env["SURE_FROZEN_MODEL_ARTIFACT"] = baseline["model_artifact"]
            result = playground.execute_parallel_tasks([
                lambda: exp.run_existing_code(code=playground.task_adapter.frozen_code(baseline["model_artifact"]),
                    role_paths=playground._role_paths(), candidate_type_hint="inference")],
                max_workers=1, workspace_names=[exp.exp_name])[0]
            if isinstance(result, Exception):
                raise result
            if not result[0] or abs(result[1] - baseline["score"]) > 1e-6:
                raise ValueError("SD baseline replay differs on allocated node")
            atomic_json(check, {"score": result[1], "baseline_score": baseline["score"],
                               "workspace": exp.workspace_path})
    task = (root / "task.md").read_text()
    result = playground.run(task)
    expected = "baseline_ready" if stage == "baseline" else "completed"
    if result.get("status") != expected:
        raise RuntimeError(f"SD {stage} did not complete: {result.get('status')}")
    atomic_json(directory / "result.json", result)


def holdout(root: Path):
    from playground.sure_master.core.playground import SureMasterPlayground
    if not all((root / group / "result.json").is_file() for group in "ABCD"):
        raise ValueError("All four groups must freeze selection before holdout")
    for group in "ABCD":
        target = root / group / "holdout.json"
        if target.exists():
            continue
        playground = SureMasterPlayground(config_path=root / group / "deployment.yaml")
        playground.set_run_dir(root / group / "search")
        playground.setup()
        selection = json.loads((root / group / "result.json").read_text())["final_evaluation"]
        winner = next(item for item in selection["selection"] if item["idea_id"] == selection["winner"])
        baseline = json.loads((root / "baseline/result.json").read_text())["baseline"]
        from playground.sure_master.core.datasets import split_specs
        split = split_specs(playground.sure_config)["holdout"]
        records = [baseline] if winner["idea_id"] == "baseline" else [baseline, winner]
        results = []
        for i, record in enumerate(records):
            exp = playground._create_run_exp("improve", 900001 + i)
            exp.candidate_phase = "holdout"
            exp.execution_env.update(playground.task_adapter.phase_environment(split))
            exp.execution_env["SURE_FROZEN_MODEL_ARTIFACT"] = record["model_artifact"]
            result = playground.execute_parallel_tasks([
                lambda: exp.run_existing_code(code=playground.task_adapter.frozen_code(record["model_artifact"]),
                    role_paths={**playground._role_paths(), **split.roles}, candidate_type_hint="inference")],
                max_workers=1, workspace_names=[exp.exp_name])[0]
            if isinstance(result, Exception):
                raise result
            if not result[0]:
                raise ValueError(f"Frozen holdout failed: {group}/{record['idea_id']}")
            results.append({"idea_id": record["idea_id"], "score": result[1],
                            "model_artifact": record["model_artifact"], "workspace": exp.workspace_path})
        atomic_json(target, {"winner": winner["idea_id"], "records": results})


def supervise(root: Path):
    """Keep controllers alive across the SSH session; never resample failed stages."""
    with (root / "workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = root / "workflow.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {"stages": {}}
        def launch(stage):
            entry = state["stages"].get(stage, {})
            done = all((root / g / "holdout.json").exists() for g in "ABCD") if stage == "holdout" else (root / stage / "result.json").exists()
            if done:
                entry["status"] = "completed"
            elif entry.get("pid") and process_identity(entry["pid"]) == entry.get("identity"):
                entry["status"] = "running"
            elif entry.get("status") == "running":
                entry["status"] = "failed"
            elif entry.get("status") == "failed":
                pass
            else:
                log_dir = root if stage == "holdout" else root / stage
                with (log_dir / "controller.log").open("ab") as log:
                    proc = subprocess.Popen([sys.executable, "-m", "playground.sure_master.tools.sd_ablation_workflow", "--root", str(root),
                        "--stage", stage], stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                        start_new_session=True)
                entry = {"pid": proc.pid, "identity": process_identity(proc.pid), "status": "running"}
            state["stages"][stage] = entry
            state["updated_at"] = time.time()
            atomic_json(state_path, state)
            return entry["status"]
        while True:
            baseline = launch("baseline")
            if baseline == "failed":
                raise RuntimeError("Baseline controller failed; inspect baseline/controller.log")
            if baseline == "completed":
                statuses = [launch(group) for group in "ABCD"]
                if "failed" in statuses:
                    raise RuntimeError("An SD arm failed; inspect workflow.json and its controller.log")
                if all(status == "completed" for status in statuses):
                    final = launch("holdout")
                    if final == "completed":
                        experiment = json.loads((root / "experiment.json").read_text())
                        subprocess.run(["docker", "run", "--rm", "--network", "none", "--cpus", "2", "--memory", "2g",
                            "-e", "TORCH_DEVICE_BACKEND_AUTOLOAD=0", "-v", f"{root}:{root}:rw",
                            "-w", experiment["source"], experiment["image"], "python", "-m",
                            "playground.sure_master.tools.summarize_sd_ablation", "--root", str(root)], check=True)
                        state["status"] = "completed"
                        atomic_json(state_path, state)
                        return
                    if final == "failed":
                        raise RuntimeError("Holdout failed; inspect holdout controller log")
            time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=["supervise", "baseline", "A", "B", "C", "D", "holdout"], default="supervise")
    args = parser.parse_args()
    if args.stage == "supervise":
        supervise(args.root.resolve())
    elif args.stage == "holdout":
        holdout(args.root.resolve())
    else:
        run_arm(args.root.resolve(), args.stage)


if __name__ == "__main__":
    main()
