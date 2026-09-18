"""Freeze a reproducible A/B SD experiment without submitting compute jobs."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import subprocess
from pathlib import Path
import shlex
import shutil

import yaml

from playground.sure_master.core.artifacts import file_digest
from playground.sure_master.core.utils.sd_allocations import AUTHORIZED
from playground.sure_master.core.utils.slurm import atomic_json
from playground.sure_master.tools.freeze_tedlium_run import freeze
from playground.sure_master.tools.with_api_profile import apply_api_routing

PROJECT = Path(__file__).resolve().parents[3]
PERSONAL = PROJECT.parent


def common_config(output: Path, image: str, env_file: Path) -> dict:
    resources = PERSONAL / "data/sure_sd_resources"
    variables = {
        "SURE_EVOLVE_ROOT": str(PROJECT), "SURE_EVAL_ROOT": str(output / "external/sure-eval"),
        "SURE_DIARIZEN_ROOT": str(output / "external/DiariZen"),
        "SURE_SD_DATA": str(PERSONAL / "data/sure_ami"),
        "SURE_WAVLM_INIT": str(resources / "wavlm-init/wavlm-base-plus-converted.bin"),
        "SURE_SD_EMBEDDING": str(resources / "pyannote/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin"),
        "SURE_WORKER_PYTHON": "python", "SURE_METRIC_PYTHON": "/opt/sure-metrics/bin/python",
        "XLAB_ROOT": str(output / "external/XLab"),
        "XLAB_PYTHON": str(PERSONAL / "data/sure_xlab_runtime312/bin/python"),
        "XLAB_SURE_SURVEY_PATH": str(PERSONAL / "XLab/.xlab/runs/20260910-survey-sd-v2/artifacts/survey.json"),
    }
    raw = (PROJECT / "configs/sure_master/ordinary-sd-npu.yaml").read_text()
    for name, value in variables.items():
        raw = raw.replace("${" + name + "}", value)
    config = yaml.safe_load(raw)
    sure = config["sure"]
    sure["research_topic"] = "Improve DiariZen speaker diarization on AMI through architecture, training and inference refinements evaluated by SURE DER."
    sure.update(search_scope="all", execution_mode="slurm", require_training_candidate=False,
        search_budget={"min_rounds": 6, "max_rounds": 6, "ideas_per_round": 4, "patience": 6},
        ablation={"use_literature": True, "use_feedback": True}, defer_holdout=True)
    sure["task"]["training"].update(recipe="diarizen.evolution.v1.bf16", batch_size=48, candidate_options={})
    sure["runtime"].update(precision="fp32", training_precision="bf16")
    sure["execution_contract"].update(training_recipe="diarizen.evolution.v1.bf16",
        precision="bf16", training_precision="bf16", master_parameter_precision="fp32", batch_size_per_rank=48,
        initialization="Training restarts from SSL backbone and seed 3407; inherit parent design only.")
    sure["execution_contract"]["constraints"] = [
        "Locally refine the current best architecture, training hooks or inference within the supplied interface; preserve the overall method.",
        "Fixed data splits, seed, initialization, BF16 autocast, batch 48 per rank, four ranks, epoch cap and validation protocol.",
        "No candidate may inspect selection/holdout or other experiment histories."]
    sure["task_cards_path"] = str(PROJECT / "playground/sure_master/task_cards/sure_tasks.yaml")
    sure["initial_solution_path"] = str(PROJECT / "playground/sure_master/baselines/task_baseline.py")
    sure["execution_env"].update(SURE_TRAIN_BUDGET_SECONDS="0", SURE_RUN_TIMEOUT="0",
        ASCEND_PROCESS_LOG_PATH="/local/job/ascend/log", ASCEND_WORK_PATH="/local/job/ascend/work")
    sure["slurm"] = {
        "shared_root": str(PERSONAL), "image": image, "partition": "compute", "max_parallel": 4,
        "time_limit": "1-00:00:00", "max_segments": 30, "poll_seconds": 20,
        "sd_restricted_pool": True, "existing_allocations": ["10085", "10086"],
        "allocation_slots_per_job": 2, "allocation_overlap": False, "allocation_cpus": 32,
        "allocation_step_memory": "120G", "allocation_fallback_when_busy": False,
        "allocation_lock_dir": str(output / "allocation_locks"),
        "resource_profiles": {
            "training": {"npu": 4, "cpu": 32, "memory": "120G", "temporary": "40G", "shm": "32g"},
            "inference": {"npu": 1, "cpu": 32, "memory": "120G", "temporary": "40G", "shm": "32g"}}}
    for agent in config["agents"].values():
        for key in ("system_prompt_file", "user_prompt_file"):
            agent[key] = str(PROJECT / "playground/sure_master" / agent[key])
        agent["tools"] = {"builtin": []}
    config["session"]["local"].update(working_dir=str(output / "search/workspace"), timeout=0)
    config["session"]["local"]["parallel"].update(max_parallel=4, split_workspace_for_exp=True)
    config["max_research_rounds"] = 6
    provider = config["xlab"]["idea_provider"]
    script = str(Path(variables["XLAB_ROOT"]) / "xlab/skills/sure_master/scripts/xlab_idea_client.py")
    provider["command"] = [variables["XLAB_PYTHON"], script]
    provider["preflight_command"] = [variables["XLAB_PYTHON"], script, "--check-survey", variables["XLAB_SURE_SURVEY_PATH"]]
    config["xlab"]["idea_generation"].update(engine="native", feedback_kind="candidate_outcome",
        refinement="current_best", allow_auxiliary_experiments=False,
        mcts={"max_iterations": 64, "max_depth": 3, "branching_factor": 3, "exploration_constant": 1.2})
    provider["environment"] = {"XLAB_ROOT": variables["XLAB_ROOT"],
        "XLAB_SURE_SURVEY_PATH": variables["XLAB_SURE_SURVEY_PATH"], "XLAB_SURE_RUN_ROOT": str(output / "xlab_ideas")}
    python = str(PERSONAL / "data/sure_asr_controller/bin/python")
    return apply_api_routing(config, env_file, python, PROJECT / "playground/sure_master/tools/with_api_profile.py")


def prepare(output: Path, image: str, env_file: Path, baseline_from: Path | None = None):
    if "@sha256:" not in image:
        raise ValueError("An immutable Registry image digest is required")
    if output.exists():
        raise ValueError("Use a new deployment directory; resume via its launch.sh")
    output.mkdir(parents=True)
    ignore = shutil.ignore_patterns(".git", "__pycache__", ".env", ".env.*", "node_modules", "*.log", "checkpoints")
    baseline_config = yaml.safe_load((baseline_from / "deployment.yaml").read_text())["sure"] if baseline_from else None
    model_source = Path(baseline_config["task"]["resources"]["source"]) if baseline_config else PERSONAL / "SD/DiariZen"
    scoring_source = Path(baseline_config["pythonpath"]) if baseline_config else PERSONAL / "sure-eval/src"
    for source, target in ((PERSONAL / "XLab/xlab", output / "external/XLab/xlab"),
                           (model_source, output / "external/DiariZen"),
                           (scoring_source, output / "external/sure-eval/src")):
        shutil.copytree(source, target, ignore=ignore, symlinks=False, ignore_dangling_symlinks=True)
    config = common_config(output, image, env_file)
    provider_env = config["xlab"]["idea_provider"]["environment"]
    frozen = subprocess.run([str(PERSONAL / "data/sure_xlab_runtime312/bin/python"),
        str(PERSONAL / "XLab/xlab/skills/sure_master/scripts/xlab_idea_client.py"),
        "--freeze-survey", provider_env["XLAB_SURE_SURVEY_PATH"], str(output / "external/XLab/.xlab/runs/sd-literature")],
        env={**os.environ, **provider_env}, text=True, capture_output=True, check=True)
    provider_env["XLAB_SURE_SURVEY_PATH"] = json.loads(frozen.stdout)["survey_path"]
    config["xlab"]["idea_provider"]["preflight_command"][-1] = provider_env["XLAB_SURE_SURVEY_PATH"]
    source_config = output / "common.yaml"
    source_config.write_text(yaml.safe_dump(config, sort_keys=False))
    snapshot, deployment = freeze(source_config, output)
    config = yaml.safe_load(deployment.read_text())
    fingerprints = {name: file_digest(Path(spec["manifest"])) for name, spec in config["sure"]["datasets"].items()}
    atomic_json(output / "data_fingerprints.json", fingerprints)
    arms = {"baseline": (True, True), "A": (True, True), "B": (False, True)}
    atomic_json(output / "control.json", {"active_groups": ["A", "B"], "cancelled_groups": ["C", "D"]})
    for name, (literature, feedback) in arms.items():
        directory = output / name
        directory.mkdir()
        arm = deepcopy(config)
        arm["sure"]["ablation"] = {"use_literature": literature, "use_feedback": feedback}
        arm["sure"]["baseline_only"] = name == "baseline"
        if name != "baseline":
            arm["sure"]["initial_baseline_run"] = str(output / "baseline")
            arm["sure"]["initial_baseline_config"] = str(output / "baseline/deployment.yaml")
            # Both arms share four slots across the two active SD allocations.
            arm["sure"]["slurm"]["existing_allocations"] = ["10085", "10086"] if name == "A" else ["10086", "10085"]
        arm["session"]["local"]["working_dir"] = str(directory / "search/workspace")
        provider = arm["xlab"]["idea_provider"]
        provider["environment"]["XLAB_SURE_RUN_ROOT"] = str(directory / "xlab_ideas")
        if not literature:
            provider.pop("preflight_command", None)
            for key in ("XLAB_SURE_SURVEY_PATH", "XLAB_SURE_EVIDENCE_JSON", "XLAB_SURE_EVIDENCE_SHA256"):
                provider["environment"].pop(key, None)
        (directory / "deployment.yaml").write_text(yaml.safe_dump(arm, sort_keys=False))
        atomic_json(directory / "data_fingerprints.json", fingerprints)
    task = (PROJECT / "playground/sure_master/data/sd_ami_free_ablation.md").read_text().replace("FP32,", "BF16 autocast with FP32 master parameters, per-rank batch 48,")
    task += "\nUpstream reference configuration below is for API context; the execution contract overrides its batch and precision.\n"
    for relative in ("recipes/diar_ssl/conf/wavlm_updated_conformer.toml",
                     "diarizen/models/eend/model_wavlm_conformer.py",
                     "diarizen/models/module/conformer.py", "recipes/diar_ssl/trainer_dual_opt.py"):
        task += "\n\n## Frozen implementation reference: " + relative + "\n\n```\n"
        task += (output / "external/DiariZen" / relative).read_text() + "\n```\n"
    (output / "task.md").write_text(task)
    if baseline_from is not None:
        from playground.sure_master.core.sd_baseline_import import copy_baseline
        copy_baseline(baseline_from.resolve(), output / "baseline", config["sure"])
    external = {str(p.relative_to(output)): file_digest(p) for p in (output / "external").rglob("*")
                if p.is_file() and p.suffix in {".py", ".toml", ".yaml", ".json"}}
    atomic_json(output / "external_manifest.json", external)
    atomic_json(output / "experiment.json", {"schema": "sure.sd_ablation.v1", "source": str(snapshot),
        "image": image, "groups": ["A", "B"], "precision": "bf16", "batch_size_per_rank": 48, "rounds": 6, "ideas_per_round": 4,
        "independent_runs_per_group": 1, "authorized_jobs": sorted(AUTHORIZED),
        "rules": "/shared/chaolei.liu/rules.md", "manual": "http://127.0.0.1:18088/"})
    python = str(PERSONAL / "data/sure_asr_controller/bin/python")
    argv = [python, "-m", "playground.sure_master.tools.with_api_profile", "--env-file", str(env_file),
            "--role", "controller", "--", python, "-m", "playground.sure_master.tools.sd_ablation_workflow",
            "--root", str(output)]
    script = output / "launch.sh"
    script.write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(snapshot)) +
                      "\nexport PYTHONPATH=" + shlex.quote(str(snapshot)) + "\nexec " + shlex.join(argv) + "\n")
    script.chmod(0o700)
    return {"root": str(output), "source": str(snapshot), "launch": str(script)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--baseline-from", type=Path, help="Verified completed baseline-only directory to import")
    parser.add_argument("--env-file", type=Path, default=PROJECT / ".env")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output.resolve(), args.image, args.env_file.resolve(), args.baseline_from)))


if __name__ == "__main__":
    main()
