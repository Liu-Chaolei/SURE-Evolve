"""Strict schemas and integrity checks for published evaluation results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from sure_eval.protocols.schema import ALLOWED_PROTOCOL_IDS
from sure_eval.storage import validate_model_id


RESULT_INDEX_SCHEMA = "sure.eval.result_index.v2"
PREDICTION_MANIFEST_SCHEMA = "sure.eval.prediction_manifest.v2"
PUBLISHED_REPORT_SCHEMA = "sure.eval.published_report.v2"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ResultIndexError(ValueError):
    """Raised when a published result cannot be trusted for reuse."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ResultIndexError(f"{location} must be a mapping")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], location: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ResultIndexError(
            f"{location} fields differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ResultIndexError(f"{location} must be a list")
    return value


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResultIndexError(f"{location} must be a non-empty string")
    return value


def _sha(value: Any, location: str) -> str:
    text = _string(value, location)
    if not _SHA256.fullmatch(text):
        raise ResultIndexError(f"{location} must be a lowercase SHA256")
    return text


def _artifact_path(model_root: Path, value: Any, location: str) -> Path:
    relative = Path(_string(value, location))
    if relative.is_absolute() or ".." in relative.parts:
        raise ResultIndexError(f"{location} must be relative to the model result root")
    root = model_root.resolve(strict=True)
    candidate = (root / relative).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ResultIndexError(f"{location} escapes the model result root") from exc
    if not candidate.is_file():
        raise ResultIndexError(f"{location} does not exist: {candidate}")
    return candidate


def _checked_artifact(
    model_root: Path,
    container: dict[str, Any],
    path_key: str,
    sha_key: str,
    location: str,
) -> Path:
    path = _artifact_path(model_root, container.get(path_key), f"{location}.{path_key}")
    expected = _sha(container.get(sha_key), f"{location}.{sha_key}")
    actual = file_sha256(path)
    if actual != expected:
        raise ResultIndexError(
            f"{location}.{path_key} checksum mismatch: expected {expected}, got {actual}"
        )
    return path


def validate_dataset_identity(value: Any, location: str = "dataset") -> dict[str, Any]:
    dataset = _mapping(value, location)
    _exact_keys(
        dataset,
        {"id", "name", "version", "split", "manifest_sha256", "sample_count"},
        location,
    )
    name = _string(dataset.get("name"), f"{location}.name")
    version = _string(dataset.get("version"), f"{location}.version")
    split = _string(dataset.get("split"), f"{location}.split")
    dataset_id = _string(dataset.get("id"), f"{location}.id")
    if "__" in name or dataset_id != f"{name}__{version}":
        raise ResultIndexError(
            f"{location}.id must equal '<source_name>__<version>' without a task suffix"
        )
    _sha(dataset.get("manifest_sha256"), f"{location}.manifest_sha256")
    sample_count = dataset.get("sample_count")
    if not isinstance(sample_count, int) or sample_count < 1:
        raise ResultIndexError(f"{location}.sample_count must be a positive integer")
    return dataset


def validate_protocol_identity(value: Any, location: str = "protocol") -> dict[str, Any]:
    protocol = _mapping(value, location)
    _exact_keys(protocol, {"id", "version", "effective_params_sha256"}, location)
    protocol_id = _string(protocol.get("id"), f"{location}.id")
    if protocol_id not in ALLOWED_PROTOCOL_IDS:
        raise ResultIndexError(f"{location}.id is not a comparable protocol: {protocol_id}")
    _string(protocol.get("version"), f"{location}.version")
    _sha(protocol.get("effective_params_sha256"), f"{location}.effective_params_sha256")
    return protocol


def _read_json(path: Path, location: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultIndexError(f"{location} is not valid JSON: {path}") from exc
    return _mapping(value, location)


def validate_prediction_manifest(
    model_root: Path,
    reference: dict[str, Any],
    *,
    expected_model: dict[str, Any],
    expected_dataset: dict[str, Any],
    expected_protocol: dict[str, Any],
) -> dict[str, Any]:
    path = _checked_artifact(
        model_root,
        reference,
        "manifest_path",
        "manifest_sha256",
        "prediction",
    )
    manifest = _read_json(path, "prediction_manifest")
    _exact_keys(
        manifest,
        {
            "schema",
            "model",
            "dataset",
            "protocol",
            "sample_ids_sha256",
            "sample_count",
            "files",
        },
        "prediction_manifest",
    )
    if manifest.get("schema") != PREDICTION_MANIFEST_SCHEMA:
        raise ResultIndexError("prediction manifest schema is not v2")
    if _mapping(manifest.get("model"), "prediction_manifest.model") != expected_model:
        raise ResultIndexError("prediction manifest model identity does not match result index")
    if validate_dataset_identity(
        manifest.get("dataset"), "prediction_manifest.dataset"
    ) != expected_dataset:
        raise ResultIndexError("prediction manifest dataset identity does not match result index")
    if validate_protocol_identity(
        manifest.get("protocol"), "prediction_manifest.protocol"
    ) != expected_protocol:
        raise ResultIndexError("prediction manifest protocol does not match result index")
    _sha(manifest.get("sample_ids_sha256"), "prediction_manifest.sample_ids_sha256")
    if manifest.get("sample_count") != expected_dataset["sample_count"]:
        raise ResultIndexError("prediction manifest sample_count does not match dataset")
    files = _list(manifest.get("files"), "prediction_manifest.files")
    if not files:
        raise ResultIndexError("prediction_manifest.files must not be empty")
    seen: set[str] = set()
    for index, raw_file in enumerate(files):
        item = _mapping(raw_file, f"prediction_manifest.files[{index}]")
        _exact_keys(
            item,
            {"role", "path", "sha256"},
            f"prediction_manifest.files[{index}]",
        )
        role = _string(item.get("role"), f"prediction_manifest.files[{index}].role")
        if role in seen:
            raise ResultIndexError(f"duplicate prediction file role: {role}")
        seen.add(role)
        _checked_artifact(
            model_root,
            item,
            "path",
            "sha256",
            f"prediction_manifest.files[{index}]",
        )
    return manifest


def _validate_published_report(
    report: dict[str, Any],
    *,
    expected_model: dict[str, Any],
    expected_dataset: dict[str, Any],
    expected_protocol: dict[str, Any],
    pipeline_id: str,
    engine: dict[str, Any],
    inference_id: str,
    model_root: Path,
) -> None:
    _exact_keys(
        report,
        {
            "schema",
            "status",
            "model",
            "dataset",
            "protocol",
            "evaluation",
            "metric",
            "lineage",
            "completed_at",
        },
        "report",
    )
    if report.get("schema") != PUBLISHED_REPORT_SCHEMA or report.get("status") != "success":
        raise ResultIndexError("published report must be a successful v2 report")
    if _mapping(report.get("model"), "report.model") != expected_model:
        raise ResultIndexError("published report model identity does not match")
    if validate_dataset_identity(report.get("dataset"), "report.dataset") != expected_dataset:
        raise ResultIndexError("published report dataset identity does not match")
    if validate_protocol_identity(report.get("protocol"), "report.protocol") != expected_protocol:
        raise ResultIndexError("published report protocol identity does not match")
    evaluation = _mapping(report.get("evaluation"), "report.evaluation")
    _exact_keys(
        evaluation,
        {"pipeline_id", "engine_commit", "engine_tree_sha256"},
        "report.evaluation",
    )
    if evaluation.get("pipeline_id") != pipeline_id:
        raise ResultIndexError("published report pipeline_id does not match")
    if evaluation.get("engine_commit") != engine["commit"]:
        raise ResultIndexError("published report engine commit does not match")
    if evaluation.get("engine_tree_sha256") != engine["tree_sha256"]:
        raise ResultIndexError("published report engine tree does not match")
    metric = _mapping(report.get("metric"), "report.metric")
    _exact_keys(metric, {"name", "score"}, "report.metric")
    _string(metric.get("name"), "report.metric.name")
    if not isinstance(metric.get("score"), (int, float)):
        raise ResultIndexError("report.metric.score must be numeric")
    _string(report.get("completed_at"), "report.completed_at")
    lineage = _mapping(report.get("lineage"), "report.lineage")
    _exact_keys(
        lineage,
        {
            "inference_id",
            "inference_source",
            "evaluation_input_path",
            "evaluation_input_sha256",
        },
        "report.lineage",
    )
    if lineage.get("inference_id") != inference_id:
        raise ResultIndexError("published report inference lineage does not match")
    if lineage.get("inference_source") not in {"published", "staged"}:
        raise ResultIndexError("published report inference_source is invalid")
    _checked_artifact(
        model_root,
        lineage,
        "evaluation_input_path",
        "evaluation_input_sha256",
        "report.lineage",
    )


def validate_result_index(
    model_root: str | Path,
    *,
    expected_model_id: str | None = None,
    require_verified: bool = True,
) -> dict[str, Any]:
    root = Path(model_root).resolve(strict=True)
    model_id = validate_model_id(expected_model_id or root.name)
    if root.name != model_id:
        raise ResultIndexError("result directory name does not match requested model_id")
    index_path = root / "result_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    payload = _read_json(index_path, "result_index")
    _exact_keys(
        payload,
        {"schema", "publication", "model", "inference_runs", "evaluation_runs"},
        "result_index",
    )
    if payload.get("schema") != RESULT_INDEX_SCHEMA:
        raise ResultIndexError("result_index schema is not v2")

    publication = _mapping(payload.get("publication"), "publication")
    status = publication.get("status")
    if require_verified and status != "verified":
        raise ResultIndexError("published result index is not human-verified")
    if status not in {"pending_human_review", "verified"}:
        raise ResultIndexError(f"invalid publication status: {status!r}")
    if status == "verified":
        _exact_keys(publication, {"status", "verified_by", "verified_at"}, "publication")
        _string(publication.get("verified_by"), "publication.verified_by")
        _string(publication.get("verified_at"), "publication.verified_at")
    else:
        _exact_keys(publication, {"status"}, "publication")

    model = _mapping(payload.get("model"), "model")
    _exact_keys(model, {"id", "artifact_sha256"}, "model")
    if model.get("id") != model_id:
        raise ResultIndexError("result index model.id does not match directory")
    _sha(model.get("artifact_sha256"), "model.artifact_sha256")

    inference_runs = _list(payload.get("inference_runs"), "inference_runs")
    inference_ids: set[str] = set()
    inference_identity: dict[str, dict[str, Any]] = {}
    for run_index, raw_run in enumerate(inference_runs):
        location = f"inference_runs[{run_index}]"
        run = _mapping(raw_run, location)
        _exact_keys(run, {"id", "created_at", "protocol", "datasets"}, location)
        run_id = _string(run.get("id"), f"{location}.id")
        _string(run.get("created_at"), f"{location}.created_at")
        if run_id in inference_ids:
            raise ResultIndexError(f"duplicate inference run id: {run_id}")
        inference_ids.add(run_id)
        protocol = validate_protocol_identity(run.get("protocol"), f"{location}.protocol")
        datasets = _list(run.get("datasets"), f"{location}.datasets")
        if not datasets:
            raise ResultIndexError(f"{location}.datasets must not be empty")
        dataset_keys: set[str] = set()
        for dataset_index, raw_dataset in enumerate(datasets):
            item_location = f"{location}.datasets[{dataset_index}]"
            item = _mapping(raw_dataset, item_location)
            _exact_keys(item, {"dataset", "prediction"}, item_location)
            dataset = validate_dataset_identity(item.get("dataset"), f"{item_location}.dataset")
            dataset_key = f"{dataset['id']}@{dataset['split']}"
            if dataset_key in dataset_keys:
                raise ResultIndexError(f"duplicate inference dataset: {dataset_key}")
            dataset_keys.add(dataset_key)
            prediction_reference = _mapping(item.get("prediction"), f"{item_location}.prediction")
            _exact_keys(
                prediction_reference,
                {"manifest_path", "manifest_sha256"},
                f"{item_location}.prediction",
            )
            validate_prediction_manifest(
                root,
                prediction_reference,
                expected_model=model,
                expected_dataset=dataset,
                expected_protocol=protocol,
            )
        inference_identity[run_id] = {
            "protocol": protocol,
            "datasets": {
                _dataset_key: _dataset
                for _dataset_key, _dataset in (
                    (
                        f"{item['dataset']['id']}@{item['dataset']['split']}",
                        item["dataset"],
                    )
                    for item in datasets
                )
            },
        }

    evaluation_runs = _list(payload.get("evaluation_runs"), "evaluation_runs")
    evaluation_ids: set[str] = set()
    for run_index, raw_run in enumerate(evaluation_runs):
        location = f"evaluation_runs[{run_index}]"
        run = _mapping(raw_run, location)
        _exact_keys(
            run,
            {"id", "created_at", "inference_id", "engine", "datasets"},
            location,
        )
        run_id = _string(run.get("id"), f"{location}.id")
        _string(run.get("created_at"), f"{location}.created_at")
        if run_id in evaluation_ids:
            raise ResultIndexError(f"duplicate evaluation run id: {run_id}")
        evaluation_ids.add(run_id)
        inference_id = _string(run.get("inference_id"), f"{location}.inference_id")
        if inference_id not in inference_ids:
            raise ResultIndexError(f"evaluation references unknown inference run: {inference_id}")
        engine = _mapping(run.get("engine"), f"{location}.engine")
        _exact_keys(engine, {"repository", "commit", "tree_sha256"}, f"{location}.engine")
        _string(engine.get("repository"), f"{location}.engine.repository")
        _string(engine.get("commit"), f"{location}.engine.commit")
        _sha(engine.get("tree_sha256"), f"{location}.engine.tree_sha256")
        datasets = _list(run.get("datasets"), f"{location}.datasets")
        evaluation_dataset_keys: set[str] = set()
        for dataset_index, raw_dataset in enumerate(datasets):
            item_location = f"{location}.datasets[{dataset_index}]"
            item = _mapping(raw_dataset, item_location)
            _exact_keys(item, {"dataset", "protocol", "pipelines"}, item_location)
            dataset = validate_dataset_identity(item.get("dataset"), f"{item_location}.dataset")
            protocol = validate_protocol_identity(item.get("protocol"), f"{item_location}.protocol")
            dataset_key = f"{dataset['id']}@{dataset['split']}"
            if dataset_key in evaluation_dataset_keys:
                raise ResultIndexError(f"duplicate evaluation dataset: {dataset_key}")
            evaluation_dataset_keys.add(dataset_key)
            source_identity = inference_identity[inference_id]
            if source_identity["datasets"].get(dataset_key) != dataset:
                raise ResultIndexError(
                    f"evaluation dataset {dataset_key} does not match its inference run"
                )
            if source_identity["protocol"] != protocol:
                raise ResultIndexError("evaluation protocol does not match its inference run")
            pipelines = _list(item.get("pipelines"), f"{item_location}.pipelines")
            if not pipelines:
                raise ResultIndexError(f"{item_location}.pipelines must not be empty")
            pipeline_ids: set[str] = set()
            for pipeline_index, raw_pipeline in enumerate(pipelines):
                pipeline_location = f"{item_location}.pipelines[{pipeline_index}]"
                pipeline = _mapping(raw_pipeline, pipeline_location)
                _exact_keys(
                    pipeline,
                    {
                        "pipeline_id",
                        "report_path",
                        "report_sha256",
                        "engine_report_path",
                        "engine_report_sha256",
                        "pipeline_description_path",
                        "pipeline_description_sha256",
                    },
                    pipeline_location,
                )
                pipeline_id = _string(pipeline.get("pipeline_id"), f"{pipeline_location}.pipeline_id")
                if pipeline_id in pipeline_ids:
                    raise ResultIndexError(f"duplicate pipeline_id: {pipeline_id}")
                pipeline_ids.add(pipeline_id)
                report_path = _checked_artifact(
                    root, pipeline, "report_path", "report_sha256", pipeline_location
                )
                engine_report_path = _checked_artifact(
                    root,
                    pipeline,
                    "engine_report_path",
                    "engine_report_sha256",
                    pipeline_location,
                )
                description_path = _checked_artifact(
                    root,
                    pipeline,
                    "pipeline_description_path",
                    "pipeline_description_sha256",
                    pipeline_location,
                )
                report = _read_json(report_path, f"{pipeline_location}.report")
                _validate_published_report(
                    report,
                    expected_model=model,
                    expected_dataset=dataset,
                    expected_protocol=protocol,
                    pipeline_id=pipeline_id,
                    engine=engine,
                    inference_id=inference_id,
                    model_root=root,
                )
                engine_report = _read_json(engine_report_path, f"{pipeline_location}.engine_report")
                description = _read_json(description_path, f"{pipeline_location}.pipeline_description")
                if engine_report.get("pipeline_id") != pipeline_id:
                    raise ResultIndexError("engine report pipeline_id does not match index")
                if description.get("pipeline_id") != pipeline_id:
                    raise ResultIndexError("pipeline description pipeline_id does not match index")
    return payload
