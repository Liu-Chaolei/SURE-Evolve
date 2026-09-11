"""Readiness gates for complete datasets and the original SSL initialization resource."""

from __future__ import annotations

import json
from pathlib import Path

from ..core.artifacts import file_digest


def validate_prepared_data(
    adapter: str,
    preparation: Path,
    manifests: dict[str, Path],
    *,
    allow_extracted_only: bool = False,
) -> dict:
    record = json.loads(preparation.read_text())
    if (
        record.get("schema_version") != "sure.prepared_training.v1"
        or record.get("adapter") != adapter
        or record.get("source_complete") is not True
    ):
        raise ValueError(
            "Full training needs a verified complete preparation, not partial downloaded data"
        )
    if record.get("source_mode") == "extracted_only" and not allow_extracted_only:
        raise ValueError(
            "extracted-only Premium preparation requires explicit data_provenance_mode=extracted_only"
        )
    for role, path in manifests.items():
        expected = record.get("digests", {}).get(role)
        if not expected or file_digest(path) != expected:
            raise ValueError(
                f"Training manifest does not match the verified preparation: {role}"
            )
    return record


def validate_wavlm_provenance(path: Path) -> dict:
    metadata = json.loads(
        path.with_suffix(path.suffix + ".provenance.json").read_text()
    )
    if (
        metadata.get("source_model") != "microsoft/wavlm-base-plus"
        or metadata.get("kind") != "ssl_backbone"
        or metadata.get("strict_state_match") is not True
        or metadata.get("sha256") != file_digest(path)
    ):
        raise ValueError(
            "DiariZen requires the verified original WavLM-Base+ SSL resource, not a trained diarization checkpoint"
        )
    return metadata


def validate_task_resources(adapter: str, sure: dict) -> None:
    settings = sure["task"]
    resources = settings["resources"]
    if adapter == "sd.diarizen":
        validate_wavlm_provenance(Path(resources["wavlm"]))
        if "model" in resources:
            raise ValueError(
                "Official DiariZen training must not load a finished DiariZen model"
            )
    datasets = sure.get("datasets") or {}
    manifests = {
        role: Path(datasets[role]["manifest"]) for role in ("train", "train_validation")
    }
    if adapter == "tts.f5tts":
        manifests["train_csv"] = Path(settings["training"]["manifest"])
    validate_prepared_data(
        adapter,
        Path(sure["data_preparation"]),
        manifests,
        allow_extracted_only=(sure.get("data_provenance_mode") == "extracted_only"),
    )
