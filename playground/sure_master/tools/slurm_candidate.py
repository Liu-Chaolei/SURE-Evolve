"""Execute a trusted, immutable candidate request inside a Slurm container."""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from playground.sure_master.core.utils.slurm import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    workspace = Path(request["workspace"])
    os.chdir(workspace)
    context = request["context"]
    env = {str(k): str(v) for k, v in context["execution_env"].items()}
    env["SURE_SLURM_RESUME"] = "1"
    env["SURE_CANDIDATE_ORIGIN_TIME"] = str(args.request.stat().st_mtime)
    # Checkpoints live in a code-addressed directory, so another candidate cannot resume them.
    owner = workspace / "metric/active_worker_request.json"
    identity = args.request.parent.name
    if owner.exists() and json.loads(owner.read_text()).get("identity") != identity:
        import shutil

        for name in ("models", "working"):
            directory = workspace / name
            if directory.is_symlink():
                raise ValueError("Candidate output directories must not be symlinks")
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir()
    atomic_json(owner, {"identity": identity})
    os.environ.update(env)
    from playground.sure_master.runtime.accelerator import check_accelerator
    from playground.sure_master.core.exp.run_exp import SureRunExp
    from playground.sure_master.core.utils.metric import SureMetricRunner
    from playground.sure_master.core.utils.task_cards import (
        resolve_task_card,
        merge_base_model_profile,
    )

    sure = request["sure"]
    result_path = Path(request["result"])
    started = time.time()
    try:
        if request.get("source_digest"):
            from playground.sure_master.core.utils.slurm import source_digest

            if request["source_digest"] != source_digest():
                raise RuntimeError(
                    "Framework source changed after submission; reconcile this request before running it"
                )
        check_accelerator(env["SURE_ACCELERATOR"])
        import torch

        backend = env["SURE_ACCELERATOR"]
        expected_devices = int(env.get("SURE_ALLOCATED_DEVICES", env.get("ASR_WORLD_SIZE", "1")))
        if backend in {"cuda", "npu"} and getattr(torch, backend).device_count() != expected_devices:
            raise RuntimeError("Allocated accelerator count differs from the task resource contract")
        card = resolve_task_card(sure["task_cards_path"], sure["task_id"])
        profile = merge_base_model_profile(
            card.base_model, sure["base_models"][sure["task_id"]]
        )
        roles = context["role_paths"]

        def execute_tool(call):
            spec = json.loads(call.function.arguments)
            log = workspace / "metric/worker.log"
            with log.open("a") as output:
                proc = subprocess.run(
                    ["bash", "-c", spec["command"]],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=int(spec["timeout"]) or None,
                )
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 16000))
                tail = stream.read().decode(errors="replace")
            return tail, {"exit_code": proc.returncode}

        agent = SimpleNamespace(
            session=SimpleNamespace(
                config=SimpleNamespace(workspace_path=str(workspace.parent))
            ),
            _execute_tool=execute_tool,
        )
        exp = SureRunExp(
            context["stage"],
            agent,
            agent,
            {"sure": sure},
            workspace.name,
            card,
            base_model_profile=profile,
            metric_runner=SureMetricRunner(
                sure["root"], sure.get("pythonpath"), device="cpu",
                python=(sure.get("metric_runtime") or {}).get("python"),
            ),
            execution_env=env,
        )
        exp.workspace_path = str(workspace)
        exp.candidate_type_hint = context["candidate_type_hint"]
        exp.candidate_phase = context.get("phase", "")
        exp.candidate_idea_id = context.get("idea_id", "")
        exp.base_model_source_overrides = context.get("base_model_source_overrides", {})
        exp._prepare_base_model_source_overrides()
        # Restarting the same candidate must not apply its source edits twice.
        import shutil

        if card.canonical_task == "asr":
            recipe = workspace / "base_model/recipe"
            pristine = args.request.parent / "pristine_recipe"
            if not pristine.exists():
                shutil.copytree(
                    recipe,
                    pristine,
                    symlinks=False,
                    ignore=shutil.ignore_patterns("data", "exp", "__pycache__"),
                )
            if recipe.is_symlink():
                recipe.unlink()
            elif recipe.exists():
                shutil.rmtree(recipe)
            shutil.copytree(pristine, recipe, symlinks=False)
        exp._prepare_workspace_inputs(roles)
        success, score, details = exp._execute_and_score(request["code"], roles)
        pause = workspace / "models/diarizen_training/budget_pause.json"
        if pause.exists():
            atomic_json(result_path, {"success": False, "score": None,
                        "reason_code": "budget_paused", "pause": json.loads(pause.read_text()),
                        "job_id": os.environ.get("SLURM_JOB_ID")})
            return
        atomic_json(
            result_path,
            {
                "success": success,
                "score": score,
                "details": details,
                "produced_artifacts": details.get("produced_artifacts", {}),
                "reason_code": details.get("reason_code"),
                "metric_feedback": exp.metric_feedback,
                "terminal_output": exp.terminal_output,
                "runtime_seconds": time.time() - started,
                "image": request["image"],
                "job_id": os.environ.get("SLURM_JOB_ID"),
            },
        )
    except Exception as exc:
        atomic_json(
            result_path,
            {
                "success": False,
                "score": None,
                "reason_code": "worker_failed",
                "error": str(exc),
                "runtime_seconds": time.time() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
