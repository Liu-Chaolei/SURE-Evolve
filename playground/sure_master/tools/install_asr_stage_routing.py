"""Freeze and activate a routing-only amendment for an existing ASR deployment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import time

from dotenv import dotenv_values
import yaml

from playground.sure_master.tools.asr_stage_routing import build_policy, AGENT_ROUTES


def install(root: Path, env_file: Path, revision_name="api_stage_revision_20260917"):
    root = root.resolve()
    state = json.loads((root / "workflow_state.json").read_text())
    pid = state.get("pid", 0)
    proc = Path(f"/proc/{pid}/stat")
    if proc.exists() and proc.read_text().rsplit(") ", 1)[1].split()[0] != "Z":
        raise ValueError("Stop or finish the controller before amending its API configuration")
    values = dotenv_values(env_file, interpolate=False)
    policy = build_policy(values)
    missing = [r["slot"] for r in policy["routes"] if not values.get(r["slot"])]
    if missing:
        raise ValueError("Missing credentials: " + ", ".join(missing))
    if not revision_name.startswith("api_stage_revision_") or Path(revision_name).name != revision_name:
        raise ValueError("Invalid routing revision directory name")
    revision = root / revision_name
    if revision.exists():
        raise ValueError("Routing revision already exists; do not overwrite its audit history")
    revision.mkdir()
    backup = revision / "before"
    backup.mkdir()
    for name in ("deployment.yaml", "execution.yaml", "prepare_ideas.sh", "launch.sh", "api_routing_policy.json", "workflow_state.json"):
        source = root / name
        if source.exists():
            shutil.copy2(source, backup / name)
    shutil.copy2(Path(__file__).with_name("asr_stage_routing.py"), revision / "asr_stage_routing.py")
    config = yaml.safe_load((root / "deployment.yaml").read_text())
    original_command = config["xlab"]["idea_provider"]["command"]
    previous_entry = original_command[-1]
    source = Path(json.loads((root / "deployment.json").read_text())["source"])
    original_entry = str(source / "playground/sure_master/tools/asr_native_reliability.py")
    frozen_controller = source / "playground/sure_master/tools/asr_native_controller.py"
    native_entry = revision / "native_entry.py"
    native_entry.write_text(
        '"""Routing amendment; no cached research result is rewritten."""\n'
        'import os, sys, runpy\nfrom pathlib import Path\n'
        'sys.path.insert(0, str(Path(__file__).resolve().parent))\n'
        'root = Path(os.environ["XLAB_ROOT"])\n'
        'sys.path.insert(0, str(root / "xlab/skills/research_idea/scripts"))\n'
        'from asr_stage_routing import install_native\ninstall_native()\n'
        f'runpy.run_path({original_entry!r}, run_name="__main__")\n')
    controller_entry = revision / "controller_entry.py"
    controller_entry.write_text(
        '"""Keep frozen ASR execution; route controller agents by model."""\n'
        'import sys, runpy\nfrom pathlib import Path\n'
        'revision = Path(__file__).resolve().parent\n'
        'sys.path.insert(0, str(revision))\n'
        'from asr_stage_routing import install_controller\n'
        f'install_controller(Path({str(env_file.resolve())!r}), revision / "policy.json", revision / "controller_requests.jsonl")\n'
        f'runpy.run_path({str(frozen_controller)!r}, run_name="__main__")\n')
    (revision / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    config["llm"]["openai_research"] = {**config["llm"]["zai_flash"], "model": "gpt-6-astra"}
    for name, agent in config["agents"].items():
        if name not in AGENT_ROUTES:
            raise ValueError("Unclassified controller agent: " + name)
        agent["llm"] = "openai_research" if AGENT_ROUTES[name] == "openai" else "zai_flash"
    provider = config["xlab"]["idea_provider"]
    for field in ("command", "preflight_command"):
        provider[field] = [str(native_entry) if item == previous_entry else item for item in provider[field]]
    provider["environment"]["XLAB_API_ROUTING_POLICY"] = json.dumps(policy)
    # The wire policy includes this amendment, so old results cannot be mistaken
    # for new-route responses. All prior caches and operation budgets remain intact.
    previous_controller = config["api_profile"].get("stage_controller_entry")
    config["api_profile"]["stage_policy"] = str(revision / "policy.json")
    config["api_profile"]["stage_controller_entry"] = str(controller_entry)
    (root / "deployment.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (root / "api_routing_policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    for name in ("prepare_ideas.sh", "launch.sh"):
        path = root / name
        text = path.read_text()
        needle = shlex.quote(previous_controller) if previous_controller else "-m playground.sure_master.tools.asr_native_controller"
        if text.count(needle) != 1:
            raise ValueError("Unexpected launch entry")
        path.write_text(text.replace(needle, shlex.quote(str(controller_entry))))
    manifest = {"time": time.time(), "scope": "API routing only", "previous_status": state,
                "research_artifacts_rewritten": False, "automatic_restart": False,
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in revision.iterdir() if p.is_file()}}
    (revision / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"revision": str(revision), "status": "configured", "workflow_status": state["status"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--revision-name", default="api_stage_revision_20260917")
    args = parser.parse_args()
    print(json.dumps(install(args.root, args.env_file, args.revision_name)))
