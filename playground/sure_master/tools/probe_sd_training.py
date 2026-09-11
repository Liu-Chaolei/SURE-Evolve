"""Real four-rank SD training and full validation, isolated from search results."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml

from playground.sure_master.core.training import build_training_contract
from playground.sure_master.runtime.accelerator import RuntimeBackend, runtime_environment
from playground.sure_master.runtime.model_source import snapshot_source
from playground.sure_master.runtime.training_state import atomic_json
from playground.sure_master.tasks import get_adapter
from playground.sure_master.tasks.training_jobs import training_command, training_source_identity
from playground.sure_master.tools.probe_task import expand


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    sure = expand(yaml.safe_load(args.config.read_text())["sure"])
    get_adapter("sd.diarizen").preflight(sure)
    work = args.workspace.resolve()
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    os.environ.update(runtime_environment(sure["runtime"]))
    backend = RuntimeBackend("npu")
    if backend.torch.npu.device_count() != 4:
        raise ValueError("Probe requires exactly four Slurm-allocated devices")
    settings = sure["task"]
    source = snapshot_source(Path(settings["resources"]["source"]), work / "source")
    import toml

    architecture = toml.load(source / "recipes/diar_ssl/conf/wavlm_updated_conformer.toml")["model"]["args"]
    architecture.pop("wavlm_src", None)
    manifests = {k: Path(sure["datasets"][k]["manifest"]) for k in ("train", "train_validation")}
    manifests["preparation"] = Path(sure["data_preparation"])
    contract = build_training_contract(
        "sd.diarizen", settings["training"], architecture, manifests,
        Path(settings["resources"]["wavlm"]), training_source_identity("sd.diarizen", source),
        "npu", component_test=True,
    )
    job = {"adapter": "sd.diarizen", "source": str(source), "output": str(work),
           "settings": settings, "architecture": architecture, "contract": contract,
           "manifests": {k: str(v) for k, v in manifests.items()}}
    atomic_json(work / "job.json", job)
    # Second invocation must restore both optimizers, the cursor and all rank RNGs.
    results = []
    for updates in (120, 121):
        with (work / f"train-{updates}.log").open("a") as log:
            subprocess.run(training_command(work / "job.json", "sd.diarizen", settings),
                           env={**os.environ, "SURE_SD_PROBE_UPDATES": str(updates)},
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((work / "probe_result.json").read_text())
        if result["updates"] != updates or (updates == 121 and not result["restored"]):
            raise ValueError("Probe did not restore the requested training cursor")
        results.append(result)
    if (work / "evidence/training_completion.json").exists():
        raise ValueError("Component probe must not publish full-training completion")
    atomic_json(work / "acceptance.json", {"status": "passed", "component_test": True, "runs": results})


if __name__ == "__main__":
    main()
