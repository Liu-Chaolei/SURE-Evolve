"""Freeze controller/worker source and unresolved configuration for a long run."""

from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import yaml

PROJECT = Path(__file__).resolve().parents[3]


def freeze(config_path: Path, output: Path) -> tuple[Path, Path]:
    from playground.sure_master.core.utils.slurm import source_digest, atomic_json

    identity = source_digest()
    snapshot = output / "source" / identity
    if not snapshot.exists():
        snapshot.mkdir(parents=True)
        ignore = shutil.ignore_patterns("__pycache__", "node_modules", ".git", ".cache")
        shutil.copytree(PROJECT / "evomaster", snapshot / "evomaster", ignore=ignore)
        (snapshot / "playground").mkdir()
        shutil.copy2(
            PROJECT / "playground/__init__.py", snapshot / "playground/__init__.py"
        )
        shutil.copytree(
            PROJECT / "playground/sure_master",
            snapshot / "playground/sure_master",
            ignore=ignore,
        )
        files = {
            str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in snapshot.rglob("*")
            if p.is_file()
        }
        atomic_json(
            snapshot / "source_manifest.json",
            {"source_digest": identity, "files": files},
        )
    if source_digest(snapshot) != identity:
        raise RuntimeError(
            "Source changed during freezing; discard this incomplete snapshot and retry"
        )
    config = yaml.safe_load(config_path.read_text())

    def relocate(value):
        if isinstance(value, str) and value.startswith(str(PROJECT) + "/"):
            relative = Path(value).relative_to(PROJECT)
            if relative.parts[0] in {"evomaster", "playground"}:
                return str(snapshot / relative)
        if isinstance(value, dict):
            return {k: relocate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relocate(v) for v in value]
        return value

    config = relocate(config)
    config["sure"]["execution_env"]["PYTHONPATH"] = str(snapshot)
    if config["sure"].get("task_id", "").startswith("asr_"):
        config["sure"]["execution_env"]["PYTHONPATH"] += ":/shared/chaolei.liu/ASR/icefall"
    config["sure"]["slurm"]["shared_root"] = "/shared/chaolei.liu"
    target = output / "deployment.yaml"
    if target.exists() and yaml.safe_load(target.read_text()) != config:
        raise ValueError(
            "Deployment already exists with different source/config; use a new output directory"
        )
    target.write_text(yaml.safe_dump(config, sort_keys=False))
    return snapshot, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/sure_master/xlab-tedlium3-npu-evolution.yaml",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "runs/tedlium3_evolution"
    )
    args = parser.parse_args()
    source, config = freeze(args.config.resolve(), args.output.resolve())
    print(json.dumps({"source": str(source), "config": str(config)}))


if __name__ == "__main__":
    main()
