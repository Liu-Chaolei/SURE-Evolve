"""Freeze the three-candidate, eight-NPU full Premium deployment."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import os

import yaml

from playground.sure_master.tools.freeze_tedlium_run import freeze

PROJECT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/sure_master/xlab-f5tts-premium-npu-evolution.yaml",
    )
    args = parser.parse_args()
    if "@sha256:" not in args.image or not args.image.startswith(
        "registry.cluster.local:5000/"
    ):
        parser.error("Provide an immutable internal registry image digest")
    output = args.output.resolve()
    if (output / "deployment.yaml").exists():
        raise ValueError(
            "Deployment exists; use launch.sh to resume or choose a new output directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text())
    # Freeze the F5 source as well as the coordinator. Checkpoints remain shared read-only assets.
    from playground.sure_master.runtime.model_source import snapshot_source

    original = Path(config["sure"]["task"]["resources"]["source"])
    before = Path.cwd()
    try:
        os.chdir(output)
        frozen_f5 = output / "base_f5_source"
        if not frozen_f5.exists():
            snapshot_source(original, frozen_f5)
    finally:
        os.chdir(before)
    config["sure"]["task"]["resources"]["source"] = str(frozen_f5)
    config["sure"]["base_models"]["tts_zh_cer"]["source_paths"]["root"] = str(frozen_f5)
    config["sure"]["slurm"]["image"] = args.image
    config["session"]["local"]["working_dir"] = str(output / "search/workspace")
    config["xlab"]["idea_provider"]["environment"]["XLAB_SURE_RUN_ROOT"] = str(
        output / "xlab_ideas"
    )
    candidate = output / "source-config.yaml"
    candidate.write_text(yaml.safe_dump(config, sort_keys=False))
    snapshot, deployment = freeze(candidate, output)
    python = PROJECT.parent / "data/sure_asr_controller/bin/python"
    argv = [
        str(python),
        "-m",
        "playground.sure_master.tools.tts_workflow",
        "--config",
        str(deployment),
        "--output",
        str(output),
        "--stage",
        "all",
    ]
    launch = output / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd "
        + shlex.quote(str(snapshot))
        + "\nexport PYTHONPATH="
        + shlex.quote(str(snapshot))
        + "\nexec "
        + shlex.join(argv)
        + "\n"
    )
    launch.chmod(0o700)
    print(launch)


if __name__ == "__main__":
    main()
