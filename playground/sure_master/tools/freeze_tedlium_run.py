"""Freeze controller/worker source and unresolved configuration for a long run."""

from __future__ import annotations
import argparse
import hashlib
import json
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
    frozen_icefall = None
    if config["sure"].get("execution_env", {}).get("SURE_ASR_PROTOCOL"):
        from playground.sure_master.runtime.asr_protocol import validate_resources
        env = config["sure"]["execution_env"]
        marker = Path(env["SURE_ASR_PREPARATION"])
        prepared = json.loads(marker.read_text())
        env["SURE_ASR_DATA_FINGERPRINT"] = prepared["fingerprint"]
        validate_resources(env)
        config["sure"]["execution_contract"]["data_fingerprint"] = prepared["fingerprint"]
        sources = config["sure"]["base_models"][config["sure"]["task_id"]]["source_paths"]
        frozen_icefall = (Path(sources["root"]) if config.get("freeze", {}).get("reuse_icefall")
                          else output / "icefall_source")
        recipe = frozen_icefall / "egs/tedlium3/ASR/zipformer"
        if not frozen_icefall.exists():
            shutil.copytree(Path(sources["root"]) / "icefall", frozen_icefall / "icefall",
                            ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copytree(sources["recipe"], recipe, symlinks=False,
                            ignore=shutil.ignore_patterns("__pycache__", "exp", "log"))
            files = {str(p.relative_to(frozen_icefall)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in frozen_icefall.rglob("*") if p.is_file()}
            atomic_json(frozen_icefall / "source_manifest.json", files)
        manifest_path = frozen_icefall / "source_manifest.json"
        if not manifest_path.is_file():
            raise ValueError("Incomplete frozen Icefall source; choose a new run directory")
        for relative, sha in json.loads(manifest_path.read_text()).items():
            if hashlib.sha256((frozen_icefall / relative).read_bytes()).hexdigest() != sha:
                raise ValueError("Frozen Icefall source changed")
        sources.update(root=str(frozen_icefall), recipe=str(recipe))
        config["sure"]["execution_contract"]["icefall_source_digest"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    config["sure"]["execution_env"]["PYTHONPATH"] = str(snapshot)
    if config["sure"].get("task_id", "").startswith("asr_"):
        config["sure"]["execution_env"]["PYTHONPATH"] += ":" + str(frozen_icefall or "/shared/chaolei.liu/ASR/icefall")
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
