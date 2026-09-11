"""Retain a self-contained candidate payload before workspace compaction."""
from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path
from typing import Any


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def retain_model_artifact(workspace: str | Path) -> dict[str, Any]:
    root = Path(workspace).resolve()
    models = root / "models"
    checkpoints = sorted(models.rglob("epoch-*.pt")) if models.is_dir() else []
    if not checkpoints:
        return {}
    retained = root / "retained_model"
    pending = root / ".retained_model.pending"
    if pending.exists():
        shutil.rmtree(pending)
    pending.mkdir()
    try:
        changes = root / "artifacts/candidate_changes.json"
        baseline_record = root / "artifacts/official_baseline.json"
        metadata = json.loads(changes.read_text()) if changes.is_file() else (
            json.loads(baseline_record.read_text()) if baseline_record.is_file() else {})
        shutil.copytree(models, pending / "models", symlinks=False)
        recipe = Path(metadata.get("runtime", {}).get("recipe_source") or "base_model/recipe")
        recipe = recipe if recipe.is_absolute() else root / recipe
        if recipe.is_dir():
            shutil.copytree(recipe, pending / "recipe", symlinks=False,
                            ignore=shutil.ignore_patterns("data", "__pycache__"))
        if (root / "working").is_dir():
            shutil.copytree(root / "working", pending / "working", symlinks=False,
                            ignore=shutil.ignore_patterns("*.log", "__pycache__", "duration_probes", "decode_probes", "eval_data", "data"))
        for relative in ("run_sure.py", "artifacts/candidate_changes.json", "artifacts/official_baseline.json", "data/lang_bpe_500/bpe.model"):
            source = root / relative
            if source.is_file():
                target = pending / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        checkpoint = max(checkpoints, key=lambda p: int(p.stem.split("-")[-1]))
        if changes.is_file():
            selected = metadata.get("produced_artifacts", {}).get("candidate_checkpoint")
            if selected:
                candidate = Path(selected)
                candidate = candidate if candidate.is_absolute() else root / candidate
                if candidate in checkpoints:
                    checkpoint = candidate
        selected_bpe = metadata.get("inference_config", {}).get("actual_bpe_model") or metadata.get("bpe_model")
        bpe = Path(selected_bpe) if selected_bpe else Path("data/lang_bpe_500/bpe.model")
        bpe = bpe if bpe.is_absolute() else root / bpe
        bpe_relative = "data/lang_bpe_500/bpe.model"
        if selected_bpe and not bpe.is_file():
            raise FileNotFoundError(f"The model's selected BPE is missing: {bpe}")
        if bpe.is_file():
            bpe_relative = "tokenizer/bpe.model"
            (pending / "tokenizer").mkdir()
            shutil.copy2(bpe, pending / bpe_relative)
        relative = checkpoint.relative_to(root)
        manifest = {"schema_version": "sure.model_artifact.v1", "checkpoint": str(relative),
                    "code": "run_sure.py", "bpe_model": bpe_relative,
                    "runtime": metadata.get("runtime", {}),
                    "checkpoint_sha256": file_digest(pending / relative),
                    "bpe_sha256": file_digest(pending / bpe_relative) if bpe.is_file() else None}
        (pending / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if retained.exists():
            shutil.rmtree(retained)
        pending.rename(retained)
        return {"candidate_checkpoint": str(retained / relative),
                "checkpoint_dir": str((retained / relative).parent),
                "model_artifact": str(retained / "manifest.json")}
    except Exception:
        shutil.rmtree(pending, ignore_errors=True)
        raise


def restore_model_artifact(manifest_path: str | Path, exp_dir: Path, *, workspace: Path | None = None) -> tuple[int, Path, dict[str, Any]]:
    manifest_path = Path(manifest_path).resolve()
    from ..artifacts import load_bundle
    payload = load_bundle(manifest_path)
    root = manifest_path.parent
    relative = Path(payload["checkpoint"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid retained checkpoint path")
    checkpoint = root / relative
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if payload.get("checkpoint_sha256") and file_digest(checkpoint) != payload["checkpoint_sha256"]:
        raise ValueError("Retained checkpoint digest mismatch")
    bpe = root / payload["bpe_model"]
    if payload.get("bpe_sha256") and file_digest(bpe) != payload["bpe_sha256"]:
        raise ValueError("Retained tokenizer digest mismatch")
    exp_dir.mkdir(parents=True, exist_ok=True)
    if workspace is not None:
        workspace = workspace.resolve()
        for directory in ("models", "working"):
            if (root / directory).is_dir():
                for source in [root / directory, *(root / directory).rglob("*")]:
                    target = workspace / source.relative_to(root)
                    try:
                        target.resolve().relative_to(workspace)
                    except ValueError as exc:
                        raise ValueError("Model restoration cannot write through an external symlink") from exc
                shutil.copytree(root / directory, workspace / directory, dirs_exist_ok=True, symlinks=False)
    else:
        for source in checkpoint.parent.glob("*.pt"):
            shutil.copy2(source, exp_dir / source.name)
    metadata_path = root / "artifacts/candidate_changes.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    if not metadata and (root / "artifacts/official_baseline.json").is_file():
        baseline = json.loads((root / "artifacts/official_baseline.json").read_text())
        metadata = {"inference_config": {"decoding_method": baseline["decoding_method"],
                    "decode_avg": baseline["avg"], "use_averaged_model": baseline["use_averaged_model"]}}
    return int(checkpoint.stem.split("-")[-1]), bpe, metadata
