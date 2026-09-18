"""Freeze a native ASR deployment while retaining the independently resumable direct run."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import yaml

from playground.sure_master.core.artifacts import file_digest
from playground.sure_master.core.utils.slurm import atomic_json
from playground.sure_master.tools.freeze_tedlium_run import freeze
from playground.sure_master.tools.with_api_profile import apply_api_routing

PROJECT = Path(__file__).resolve().parents[3]
PERSONAL = PROJECT.parent
PYTHON = PERSONAL / "data/sure_asr_controller/bin/python"
XLAB_PYTHON = PERSONAL / "data/sure_xlab_runtime312/bin/python"


def prepare(old: Path, output: Path, transport: str = "responses", name: str = "ASR-Native-MCTS-v2"):
    old, output = old.resolve(), output.resolve()
    if output.exists():
        raise ValueError("New deployment requires an unused directory")
    meta = json.loads((old / "recovery.json").read_text())
    old_source = old / "source" / meta["source_digest"]
    original = yaml.safe_load((old / "execution.yaml").read_text())
    baseline = Path(original["sure"]["initial_baseline_run"])
    baseline_state = json.loads((baseline / "search/workspace/metric/controller_state.json").read_text())
    if baseline_state["completed_rounds"] != 0 or baseline_state["candidates"]:
        raise ValueError("Use the original baseline-only source")
    output.mkdir(parents=True)
    def relocate(value):
        if isinstance(value, str):
            if value.startswith(str(old_source) + "/"):
                return str(PROJECT / Path(value).relative_to(old_source))
            if value.startswith(str(old) + "/"):
                return str(output / Path(value).relative_to(old))
        if isinstance(value, dict):
            return {k: relocate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relocate(v) for v in value]
        return value
    config = relocate(original)
    sure = config["sure"]
    sure.pop("active_registry", None)
    sure["initial_baseline_run"] = str(baseline)
    sure["search_budget"].update(min_rounds=6, max_rounds=6, ideas_per_round=4, patience=6)
    sure["ablation"] = {"use_literature": True, "use_feedback": True}
    sure["startup_mode"] = "direct_formal"
    sure["slurm"].pop("authorized_service_takeover", None)
    sure["slurm"].update(existing_allocations=["10023", "10028"],
        allocation_slots_per_job=2, allocation_overlap=False, allocation_cpus=32,
        allocation_step_memory="128G", allocation_lock_dir=str(PERSONAL / "data/sure_asr_allocations"),
        allocation_fallback_when_busy=False, max_parallel=4)
    config["freeze"] = {"reuse_icefall": True}
    config["deployment_registry"] = {"name": name, "old_run": str(old),
        "active_pointer": str(PROJECT / "runs/current_asr_evolution.json")}
    xlab = output / "external/XLab"
    shutil.copytree(PERSONAL / "XLab/xlab", xlab / "xlab", symlinks=False,
        ignore=shutil.ignore_patterns("__pycache__", ".git", "node_modules", ".env", ".env.*"))
    lib = xlab / "xlab/skills/research_idea/scripts/research_idea_lib"
    support = Path(__file__).with_name("asr_native_support")
    shutil.copy2(support / "materialization_validation.py", lib / "materialization_validation.py")
    shutil.copy2(support / "deferred_reference_audit.py", lib / "asr_deferred_reference_audit.py")
    routing = lib / "providers/routing.py"
    routing_text = routing.read_text()
    for before, after in [
        ('("XI_API_KEY", "ZAI_API_KEY", "ZAI_API_KEY2", "OPENAI_API_KEY")', '("ZAI_API_KEY", "ZAI_API_KEY2", "OPENAI_API_KEY")'),
        ('("gpt-6-astra", "glm-5.3-flash", "glm-5.3-flash", "gpt-6-astra")', '("glm-5.3-flash", "glm-5.3-flash", "gpt-6-astra")'),
        ('(True, False, False, True)', '(False, False, True)'),
        ('values.get("XI_BASE_URL"), values.get("ZAI_BASE_URL"),', 'values.get("ZAI_BASE_URL"),'),
        ('must be XI, ZAI, ZAI2, OPENAI', 'must be ZAI, ZAI2, OPENAI'),
    ]:
        if before not in routing_text:
            raise ValueError("Routing source changed; review the ASR slot projection")
        routing_text = routing_text.replace(before, after)
    routing.write_text(routing_text)
    adapter = lib / "algorithm/provider_adapter.py"
    adapter.write_text(adapter.read_text().replace(' or "{{" in text or "}}" in text', ''))
    survey = PERSONAL / "XLab/.xlab/runs/20260910-survey-asr-v2/artifacts/survey.json"
    script = xlab / "xlab/skills/sure_master/scripts/xlab_idea_client.py"
    freeze_script = PERSONAL / "XLab/xlab/skills/sure_master/scripts/xlab_idea_client.py"
    result = subprocess.run([str(XLAB_PYTHON), "-P", str(freeze_script), "--freeze-survey", str(survey),
        str(xlab / "literature_bundle")], capture_output=True, text=True)
    if result.returncode:
        (output / "resource_freeze_error.log").write_text(result.stderr)
        raise RuntimeError("Survey freezing failed; see resource_freeze_error.log")
    survey = Path(json.loads(result.stdout)["survey_path"])
    operations = xlab / ".xlab/runs/asr_native"
    operations.mkdir(parents=True)
    (output / "xlab_operations").symlink_to(operations, target_is_directory=True)
    config["xlab"]["idea_generation"] = {"engine": "native", "max_attempts": 8,
        "feedback_kind": "candidate_outcome", "refinement": "current_best", "allow_auxiliary_experiments": False,
        "mcts": {"max_iterations": 8, "max_depth": 3, "branching_factor": 3, "exploration_constant": 1.2}}
    provider = config["xlab"]["idea_provider"]
    provider.update(command=[str(XLAB_PYTHON), "-P", str(script)],
        preflight_command=[str(XLAB_PYTHON), "-P", str(script), "--check-survey", str(survey)],
        timeout_seconds=86400)
    provider["environment"] = {"XLAB_ROOT": str(xlab), "XLAB_SURE_SURVEY_PATH": str(survey),
        "XLAB_SURE_RUN_ROOT": str(operations),
        "XLAB_SURE_PROVIDER_CACHE_ROOT": str(output / "parent_provider_cache"),
        "XLAB_SURE_MAX_OUTPUT_TOKENS": "8192", "XLAB_SURE_REASONING_EFFORT": "low",
        "XLAB_SURE_HTTP_ATTEMPTS": "3", "XLAB_SURE_RESPONSES": "1" if transport == "responses" else "0"}
    config = apply_api_routing(config, PROJECT / ".env", str(PYTHON),
                               PROJECT / "playground/sure_master/tools/with_api_profile.py")
    config["llm"]["zai_flash"].update(max_tokens=8192, timeout=900)
    # Explicit three-slot frozen native policy; no inherited XI or nested proxy.
    from dotenv import dotenv_values
    values = dotenv_values(PROJECT / ".env", interpolate=False)
    routes = []
    for slot, base_key, model, responses in (
        ("ZAI_API_KEY", "ZAI_BASE_URL", "glm-5.3-flash", False),
        ("ZAI_API_KEY2", "ZAI_BASE_URL2", "glm-5.3-flash", False),
        ("OPENAI_API_KEY", "OPENAI_BASE_URL", "gpt-6-astra", True),
    ):
        base = (values.get(base_key) or (values.get("ZAI_BASE_URL") if slot == "ZAI_API_KEY2"
                else "https://api.openai.com/v1" if slot == "OPENAI_API_KEY" else "")).rstrip("/")
        from urllib.parse import urlsplit
        if base and not urlsplit(base).path:
            base += "/v1"
        routes.append(dict(slot=slot, model=model, responses=responses, base_url=base))
    policy = {"schema_version": "xlab.api_routing.v1", "routes": routes}
    atomic_json(output / "api_routing_policy.json", policy)
    provider = config["xlab"]["idea_provider"]
    entry = PROJECT / "playground/sure_master/tools/asr_native_reliability.py"
    provider["command"] = [str(XLAB_PYTHON), "-P", str(entry)]
    provider["preflight_command"] = [str(XLAB_PYTHON), "-P", str(entry), "--check-survey", str(survey)]
    provider["environment"].update(XLAB_API_ROUTING_POLICY=json.dumps(policy),
        XLAB_API_ROUTING_ENV_FILE=str(PROJECT / ".env"), XLAB_NATIVE_RECOVERY="1",
        XLAB_API_ROUTING_STATE_DIR=str(output / "route_health"))
    config["api_profile"]["provider"] = "asr_ordered_controller_native_direct"
    config["api_profile"]["router_config"] = str(PROJECT / "runs/services/asr_api_priority_v1/router.json")

    task = Path(original["sure"]["task_description_path"]).read_text()
    task = task.replace("Train the native regular Zipformer baseline once", "Reuse the verified original regular Zipformer baseline without retraining")
    task = task.replace("selection compares the new baseline and", "selection compares the imported baseline and")
    task += """
Baseline policy: import the verified original baseline (regular WER 5.6946006749%).
Do not train a new baseline.
"""
    task += """\nNative research execution boundary: refine Zipformer ONLY through these four six-stage
arguments: num-encoder-layers, encoder-dim, feedforward-dim, encoder-unmasked-dim.
All dimensions are multiples of32; unmasked<=encoder, total layers<=20,
encoder width<=512, feedforward width<=1536. Each native mode must propose mechanisms
expressible within this boundary. Full native retrieval/MCTS/fusion is mandatory.
Train each accepted architecture from scratch. Do not run auxiliary ablations.
This is a fresh independent six-round run: no previous direct-run candidate history is supplied.
"""
    (output / "task.md").write_text(task)
    config["sure"]["task_description_path"] = str(output / "task.md")
    (output / "input_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    snapshot, deployment = freeze(output / "input_config.yaml", output)
    atomic_json(output / "deployment.json", {"name": name, "source": str(snapshot),
        "old_run": str(old), "baseline_run": str(baseline), "deployment": str(deployment),
        "image": sure["slurm"]["image"], "training_enabled": False})
    deps = {"baseline_run": str(baseline), "icefall_sources": sure["base_models"][sure["task_id"]]["source_paths"],
            "baseline_model_artifact": baseline_state["baseline"]["model_artifact"],
            "baseline_score": baseline_state["baseline"]["score"], "policy": "retain immutable referenced resources"}
    atomic_json(output / "dependencies.json", deps)
    atomic_json(output / "external_manifest.json", {str(p.relative_to(output)): file_digest(p)
        for p in (output / "external").rglob("*") if p.is_file()})
    argv = [str(PYTHON), "-P", "-m", "playground.sure_master.tools.asr_native_controller", "--root", str(output)]
    for name, extra in (("prepare_ideas.sh", ["--ideas-only"]), ("launch.sh", [])):
        file = output / name
        file.write_text("#!/bin/bash\nset -euo pipefail\ncd " + shlex.quote(str(PROJECT)) +
            "\nexport PYTHONPATH=" + shlex.quote(str(snapshot)) + "\nexec " + shlex.join(argv + extra) + "\n")
        file.chmod(0o700)
    return {"root": str(output), "source": str(snapshot)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--name", default="ASR-Native-MCTS-v2")
    parser.add_argument("--transport", choices=("responses", "chat"), default="responses")
    args = parser.parse_args()
    print(json.dumps(prepare(args.old_run, args.output, args.transport, args.name)))


if __name__ == "__main__":
    main()
