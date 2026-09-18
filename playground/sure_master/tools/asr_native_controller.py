"""Run a frozen native ASR deployment, with a durable research-only preparation phase."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from pathlib import Path

import yaml

from playground.sure_master.core.utils.slurm import atomic_json


def compile_architecture(arguments: dict) -> str:
    keys = ("num-encoder-layers", "encoder-dim", "feedforward-dim", "encoder-unmasked-dim")
    if not isinstance(arguments, dict) or set(arguments) != set(keys):
        raise ValueError("Native ASR requires a reviewed four-argument projection")
    values = {}
    for key in keys:
        if not isinstance(arguments[key], str):
            raise ValueError("Architecture values must be strings")
        row = [int(x) for x in arguments[key].split(",")]
        if len(row) != 6 or min(row) < (0 if key == "encoder-unmasked-dim" else 1):
            raise ValueError("Invalid stage values")
        if key != "num-encoder-layers" and any(x % 32 for x in row):
            raise ValueError("Invalid dimension alignment")
        values[key] = row
    if any(a > b for a, b in zip(values[keys[3]], values[keys[1]])):
        raise ValueError("Unmasked dimension exceeds encoder width")
    if sum(values[keys[0]]) > 20 or max(values[keys[1]]) > 512 or max(values[keys[2]]) > 1536:
        raise ValueError("Candidate exceeds fixed architecture envelope")
    args = [part for key in keys for part in ("--" + key, ",".join(map(str, values[key])))]
    return ("import json, os, subprocess\nfrom pathlib import Path\n"
            "assert Path('base_model/recipe/train.py').is_file()\n"
            f"train_args = {args!r}\n"
            "subprocess.run([os.environ['SURE_ICEFALL_PYTHON'], os.environ['SURE_ASR_ZIPFORMER_WRAPPER'], "
            "'--action', 'train_decode', '--candidate-type', 'arch', '--train-args-json', json.dumps(train_args)], check=True)\n"
            "assert Path('artifacts/hyp.txt').is_file()\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ideas-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    lock = (root / "workflow.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "PAUSED.json").exists():
        raise SystemExit("Deployment is paused; use its explicit resume operation")
    config = yaml.safe_load((root / "deployment.yaml").read_text())
    from playground.sure_master.tools.with_api_profile import profile_environment
    from playground.sure_master.core.playground import SureMasterPlayground
    from playground.sure_master.core.exp.run_exp import SureRunExp
    from openai.resources.chat.completions import Completions

    if config["api_profile"].get("router_config"):
        router = json.loads(Path(config["api_profile"]["router_config"]).read_text())
        token = Path(router["token_file"]).read_text().strip()
        os.environ.update(ZAI_API_KEY=token, ZAI_BASE_URL=router["base_url"],
                          OPENAI_API_KEY=token, OPENAI_BASE_URL=router["base_url"])
    else:
        os.environ.update(profile_environment(Path(config["api_profile"]["env_file"]), "controller"))
    original_create = Completions.create
    def create(self, *a, **kw):
        if kw.get("model", "").lower() == "glm-5.3-flash":
            kw.setdefault("reasoning_effort", "low")
        return original_create(self, *a, **kw)
    Completions.create = create
    original_main = SureRunExp._run_main_agent
    def implementation(exp, *a, **kw):
        for file in (root / "xlab_operations").glob("*/batch.json"):
            batch = json.loads(file.read_text())
            for idea in batch.get("ideas", []):
                if idea["idea_id"] != exp.candidate_idea_id:
                    continue
                native = idea["native_artifact"]
                if native.get("research_policy", {}).get("profile") != "xlab.sure.native.v1":
                    raise ValueError("Native ASR refuses direct-generation artifacts")
                if native.get("execution_projection_source") != "native_fusion_review":
                    raise ValueError("Missing faithful fused-idea projection")
                code = compile_architecture(native["architecture_arguments"])
                atomic_json(Path(exp.workspace_path) / "metric/implementation_source.json", {
                    "source": "native_fusion_review_compiler", "idea_id": idea["idea_id"],
                    "batch_digest": batch["batch_digest"], "architecture_arguments": native["architecture_arguments"]})
                return "```python\n" + code + "\n```"
        return original_main(exp, *a, **kw)
    SureRunExp._run_main_agent = implementation
    phase = "preparing_ideas" if args.ideas_only else "search"
    pointer = Path(config["deployment_registry"]["active_pointer"])
    registry_lock = None
    if not args.ideas_only:
        if not (root / "TRAINING_ENABLED").exists():
            raise SystemExit("Training requires completed resource handoff")
        registry_lock = pointer.with_suffix(".lock").open("a")
        fcntl.flock(registry_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def status(value, **extra):
        record = {"run_dir": str(root), "name": config["deployment_registry"]["name"], "phase": phase,
                  "status": value, "pid": os.getpid(), "updated_at": time.time(), **extra}
        atomic_json(root / "workflow_state.json", record)
        if not args.ideas_only:
            atomic_json(pointer, record)
    status("running")
    try:
        (root / "execution.yaml").write_text((root / "deployment.yaml").read_text())
        playground = SureMasterPlayground(config_path=root / "deployment.yaml")
        playground.set_run_dir(root / "search")
        result = playground.run((root / "task.md").read_text(), ideas_only=args.ideas_only)
        atomic_json(root / ("ideas_ready.json" if args.ideas_only else "result.json"), result)
        expected = "ideas_ready" if args.ideas_only else "completed"
        if result.get("status") != expected:
            raise RuntimeError(result.get("error", str(result.get("status"))))
        status(expected)
    except Exception as error:
        status("blocked", error=str(error))
        raise


if __name__ == "__main__":
    main()
