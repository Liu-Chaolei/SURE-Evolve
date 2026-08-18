"""Deterministic human-readable views over the append-only v2 result state."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterator

from sure_eval.storage import StorageConfig

from .schema import (
    PUBLISHED_REPORT_SCHEMA,
    file_sha256,
    validate_dataset_identity,
    validate_protocol_identity,
    validate_result_index,
)


REPORT_ROW_SCHEMA = "sure.eval.report_row.v2"
RESULT_DELTA_SCHEMA = "sure.eval.result_delta.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PIPELINE_KEYS = {
    "pipeline_id",
    "report_path",
    "report_sha256",
    "engine_report_path",
    "engine_report_sha256",
    "pipeline_description_path",
    "pipeline_description_sha256",
}


class DerivedReportError(ValueError):
    """Raised when a report view cannot be proved from the canonical state."""


def refresh_derived_report_views(
    *,
    storage: StorageConfig,
    model_id: str,
    model_artifact_sha256: str,
    model_root: Path,
    result_delta_path: Path | None,
) -> dict[str, Any]:
    """Atomically rebuild JSONL and Markdown views from the effective result state."""

    model = {"id": model_id, "artifact_sha256": model_artifact_sha256}
    with _report_lock(model_root):
        local_index = validate_result_index(
            model_root,
            expected_model_id=model_id,
            require_verified=False,
        )
        if local_index["model"] != model:
            raise DerivedReportError("staged result index belongs to a different model artifact")

        if result_delta_path is None:
            runs = [(item, model_root) for item in local_index["evaluation_runs"]]
            source = {
                "kind": "staged_index",
                "staged_result_index_sha256": file_sha256(model_root / "result_index.json"),
                "result_delta_sha256": None,
                "base_published_index": None,
            }
        else:
            published_root = storage.published_results(model_id)
            published_index_path = published_root / "result_index.json"
            published_index = validate_result_index(
                published_root,
                expected_model_id=model_id,
                require_verified=True,
            )
            if published_index["model"] != model:
                raise DerivedReportError(
                    "published result index belongs to a different model artifact"
                )
            delta = _load_and_validate_delta(
                result_delta_path,
                storage=storage,
                model=model,
                local_index=local_index,
                published_index=published_index,
                published_index_path=published_index_path,
            )
            runs = [
                *((item, published_root) for item in published_index["evaluation_runs"]),
                *((item, model_root) for item in delta["additions"]["evaluation_runs"]),
            ]
            source = {
                "kind": "verified_index_plus_delta",
                "staged_result_index_sha256": file_sha256(model_root / "result_index.json"),
                "result_delta_sha256": file_sha256(result_delta_path),
                "base_published_index": dict(delta["base_published_index"]),
            }

        rows = _build_rows(runs, expected_model=model)
        report_path = model_root / "report.jsonl"
        snapshot_path = model_root / "report_snapshot.md"
        report_text = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
        _atomic_text_write(report_path, report_text, storage=storage)
        _atomic_text_write(
            snapshot_path,
            _render_snapshot(model=model, rows=rows, source_kind=source["kind"]),
            storage=storage,
        )
        return {
            "report_jsonl": {
                "path": str(report_path.relative_to(model_root)),
                "sha256": file_sha256(report_path),
                "row_count": len(rows),
            },
            "report_snapshot": {
                "path": str(snapshot_path.relative_to(model_root)),
                "sha256": file_sha256(snapshot_path),
            },
            "view_source": source,
        }


def _load_and_validate_delta(
    path: Path,
    *,
    storage: StorageConfig,
    model: dict[str, str],
    local_index: dict[str, Any],
    published_index: dict[str, Any],
    published_index_path: Path,
) -> dict[str, Any]:
    target = storage.assert_staging_write_path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DerivedReportError(f"result delta is not valid JSON: {target}") from exc
    if not isinstance(payload, dict):
        raise DerivedReportError("result delta must be a mapping")
    _exact_keys(
        payload,
        {
            "schema",
            "status",
            "model",
            "base_published_index",
            "additions",
            "created_at",
            "automatic_publish_allowed",
        },
        "result_delta",
    )
    if payload.get("schema") != RESULT_DELTA_SCHEMA:
        raise DerivedReportError("unsupported result delta schema")
    if payload.get("status") != "pending_human_review":
        raise DerivedReportError("result delta must be pending human review")
    if payload.get("automatic_publish_allowed") is not False:
        raise DerivedReportError("result delta cannot allow automatic publication")
    if payload.get("model") != model:
        raise DerivedReportError("result delta belongs to a different model artifact")
    if not isinstance(payload.get("created_at"), str) or not payload["created_at"].strip():
        raise DerivedReportError("result_delta.created_at must be a non-empty string")

    base = _mapping(payload.get("base_published_index"), "result_delta.base_published_index")
    _exact_keys(base, {"path", "sha256"}, "result_delta.base_published_index")
    expected_path = str(storage.declared_published_results(model["id"]) / "result_index.json")
    if base.get("path") != expected_path:
        raise DerivedReportError("result delta does not reference the canonical NFS result index")
    if base.get("sha256") != file_sha256(published_index_path):
        raise DerivedReportError("published result index changed after the delta was created")

    additions = _mapping(payload.get("additions"), "result_delta.additions")
    _exact_keys(additions, {"inference_runs", "evaluation_runs"}, "result_delta.additions")
    inference_runs = _list(additions.get("inference_runs"), "result_delta.additions.inference_runs")
    evaluation_runs = _list(
        additions.get("evaluation_runs"), "result_delta.additions.evaluation_runs"
    )
    if inference_runs != local_index["inference_runs"]:
        raise DerivedReportError("result delta inference additions differ from the staged index")

    published_inference = {item["id"]: item for item in published_index["inference_runs"]}
    local_inference = {item["id"]: item for item in local_index["inference_runs"]}
    if set(published_inference) & set(local_inference):
        raise DerivedReportError("staged inference id collides with a published inference id")
    valid_inference_ids = set(published_inference) | set(local_inference)
    inference_identity = {
        inference_id: _inference_identity(run)
        for inference_id, run in {**published_inference, **local_inference}.items()
    }

    published_evaluation_ids = {item["id"] for item in published_index["evaluation_runs"]}
    local_evaluation_ids = {item["id"] for item in local_index["evaluation_runs"]}
    delta_ids: set[str] = set()
    delta_by_id: dict[str, dict[str, Any]] = {}
    for raw in evaluation_runs:
        item = _mapping(raw, "result_delta.additions.evaluation_runs[]")
        evaluation_id = _required_string(item.get("id"), "evaluation.id")
        if evaluation_id in delta_ids or evaluation_id in published_evaluation_ids:
            raise DerivedReportError(f"duplicate effective evaluation id: {evaluation_id}")
        if item.get("inference_id") not in valid_inference_ids:
            raise DerivedReportError(
                f"delta evaluation references an unknown inference: {item.get('inference_id')}"
            )
        if (
            evaluation_id not in local_evaluation_ids
            and item["inference_id"] not in published_inference
        ):
            raise DerivedReportError(
                "delta-only evaluations may reference only published inference runs"
            )
        _validate_evaluation_identity(item, inference_identity[item["inference_id"]])
        delta_ids.add(evaluation_id)
        delta_by_id[evaluation_id] = item

    for local_evaluation in local_index["evaluation_runs"]:
        if delta_by_id.get(local_evaluation["id"]) != local_evaluation:
            raise DerivedReportError(
                "result delta evaluation additions differ from the staged result index"
            )
    return payload


def _inference_identity(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": run["protocol"],
        "datasets": {
            f"{item['dataset']['id']}@{item['dataset']['split']}": item["dataset"]
            for item in run["datasets"]
        },
    }


def _validate_evaluation_identity(
    evaluation: dict[str, Any],
    inference: dict[str, Any],
) -> None:
    datasets = _list(evaluation.get("datasets"), "evaluation.datasets")
    if not datasets:
        raise DerivedReportError("evaluation.datasets must not be empty")
    seen: set[str] = set()
    for item in datasets:
        mapping = _mapping(item, "evaluation.datasets[]")
        _exact_keys(mapping, {"dataset", "protocol", "pipelines"}, "evaluation.datasets[]")
        dataset = validate_dataset_identity(mapping.get("dataset"), "evaluation.dataset")
        protocol = validate_protocol_identity(mapping.get("protocol"), "evaluation.protocol")
        key = f"{dataset['id']}@{dataset['split']}"
        if key in seen:
            raise DerivedReportError(f"duplicate evaluation dataset: {key}")
        seen.add(key)
        if inference["datasets"].get(key) != dataset:
            raise DerivedReportError(
                f"evaluation dataset {key} differs from its source inference"
            )
        if inference["protocol"] != protocol:
            raise DerivedReportError("evaluation protocol differs from its source inference")


def _build_rows(
    runs: list[tuple[dict[str, Any], Path]],
    *,
    expected_model: dict[str, str],
) -> list[dict[str, Any]]:
    evaluation_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    ordered_runs = sorted(
        runs,
        key=lambda item: (str(item[0].get("created_at", "")), item[0]["id"]),
    )
    for run, artifact_root in ordered_runs:
        _exact_keys(run, {"id", "created_at", "inference_id", "engine", "datasets"}, "evaluation")
        evaluation_id = _required_string(run.get("id"), "evaluation.id")
        if evaluation_id in evaluation_ids:
            raise DerivedReportError(f"duplicate effective evaluation id: {evaluation_id}")
        evaluation_ids.add(evaluation_id)
        created_at = _required_string(run.get("created_at"), "evaluation.created_at")
        inference_id = _required_string(run.get("inference_id"), "evaluation.inference_id")
        engine = _mapping(run.get("engine"), "evaluation.engine")
        _exact_keys(engine, {"repository", "commit", "tree_sha256"}, "evaluation.engine")
        _required_string(engine.get("repository"), "evaluation.engine.repository")
        _required_string(engine.get("commit"), "evaluation.engine.commit")
        _required_sha(engine.get("tree_sha256"), "evaluation.engine.tree_sha256")

        datasets = _list(run.get("datasets"), "evaluation.datasets")
        if not datasets:
            raise DerivedReportError("evaluation.datasets must not be empty")
        for dataset_item in sorted(
            datasets,
            key=lambda item: (item["dataset"]["id"], item["dataset"]["split"]),
        ):
            _exact_keys(dataset_item, {"dataset", "protocol", "pipelines"}, "evaluation.dataset")
            dataset = validate_dataset_identity(dataset_item.get("dataset"), "evaluation.dataset")
            protocol = validate_protocol_identity(
                dataset_item.get("protocol"), "evaluation.protocol"
            )
            pipelines = _list(dataset_item.get("pipelines"), "evaluation.pipelines")
            if not pipelines:
                raise DerivedReportError("evaluation.pipelines must not be empty")
            for pipeline in sorted(pipelines, key=lambda item: item["pipeline_id"]):
                _exact_keys(pipeline, _PIPELINE_KEYS, "evaluation.pipeline")
                report = _validate_pipeline_artifacts(
                    artifact_root,
                    pipeline,
                    expected_model=expected_model,
                    expected_dataset=dataset,
                    expected_protocol=protocol,
                    expected_engine=engine,
                    expected_inference_id=inference_id,
                )
                rows.append(
                    {
                        "schema": REPORT_ROW_SCHEMA,
                        "status": "success",
                        "model": dict(expected_model),
                        "evaluation": {
                            "id": evaluation_id,
                            "created_at": created_at,
                            "inference_id": inference_id,
                            "engine": dict(engine),
                        },
                        "dataset": dict(dataset),
                        "protocol": dict(protocol),
                        "pipeline": dict(pipeline),
                        "metric": dict(report["metric"]),
                        "lineage": dict(report["lineage"]),
                        "completed_at": report["completed_at"],
                    }
                )
    return rows


def _validate_pipeline_artifacts(
    root: Path,
    pipeline: dict[str, Any],
    *,
    expected_model: dict[str, str],
    expected_dataset: dict[str, Any],
    expected_protocol: dict[str, Any],
    expected_engine: dict[str, Any],
    expected_inference_id: str,
) -> dict[str, Any]:
    pipeline_id = _required_string(pipeline.get("pipeline_id"), "pipeline.pipeline_id")
    report_path = _checked_artifact(root, pipeline, "report_path", "report_sha256")
    engine_report_path = _checked_artifact(
        root, pipeline, "engine_report_path", "engine_report_sha256"
    )
    description_path = _checked_artifact(
        root,
        pipeline,
        "pipeline_description_path",
        "pipeline_description_sha256",
    )
    report = _load_mapping(report_path, "pipeline report")
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
        "pipeline report",
    )
    if report.get("schema") != PUBLISHED_REPORT_SCHEMA or report.get("status") != "success":
        raise DerivedReportError("pipeline report must be a successful v2 report")
    if report.get("model") != expected_model:
        raise DerivedReportError("pipeline report model identity does not match")
    if validate_dataset_identity(report.get("dataset"), "report.dataset") != expected_dataset:
        raise DerivedReportError("pipeline report dataset identity does not match")
    if validate_protocol_identity(report.get("protocol"), "report.protocol") != expected_protocol:
        raise DerivedReportError("pipeline report protocol identity does not match")
    evaluation = _mapping(report.get("evaluation"), "report.evaluation")
    _exact_keys(
        evaluation,
        {"pipeline_id", "engine_commit", "engine_tree_sha256"},
        "report.evaluation",
    )
    if evaluation != {
        "pipeline_id": pipeline_id,
        "engine_commit": expected_engine["commit"],
        "engine_tree_sha256": expected_engine["tree_sha256"],
    }:
        raise DerivedReportError("pipeline report evaluation identity does not match")
    metric = _mapping(report.get("metric"), "report.metric")
    _exact_keys(metric, {"name", "score"}, "report.metric")
    _required_string(metric.get("name"), "report.metric.name")
    score = metric.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise DerivedReportError("report.metric.score must be numeric")
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
    if lineage.get("inference_id") != expected_inference_id:
        raise DerivedReportError("pipeline report inference identity does not match")
    if lineage.get("inference_source") not in {"published", "staged"}:
        raise DerivedReportError("pipeline report inference source is invalid")
    _checked_artifact(
        root,
        lineage,
        "evaluation_input_path",
        "evaluation_input_sha256",
    )
    _required_string(report.get("completed_at"), "report.completed_at")

    engine_report = _load_mapping(engine_report_path, "engine report")
    description = _load_mapping(description_path, "pipeline description")
    if engine_report.get("pipeline_id") != pipeline_id:
        raise DerivedReportError("engine report pipeline identity does not match")
    if description.get("pipeline_id") != pipeline_id:
        raise DerivedReportError("pipeline description identity does not match")
    return report


def _render_snapshot(
    *,
    model: dict[str, str],
    rows: list[dict[str, Any]],
    source_kind: str,
) -> str:
    evaluation_count = len({row["evaluation"]["id"] for row in rows})
    dataset_count = len(
        {(row["dataset"]["id"], row["dataset"]["split"]) for row in rows}
    )
    lines = [
        f"# {_markdown(model['id'])} Evaluation Snapshot",
        "",
        "> Derived review view. The canonical source of truth is result_index.json "
        "and, when present, result_delta.json.",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Model ID | `{_markdown(model['id'])}` |",
        f"| Model artifact SHA-256 | `{model['artifact_sha256']}` |",
        f"| Effective source | `{source_kind}` |",
        f"| Evaluation runs | {evaluation_count} |",
        f"| Dataset splits | {dataset_count} |",
        f"| Report rows | {len(rows)} |",
        "",
        "## Evaluation History",
        "",
    ]
    if not rows:
        lines.extend(["No completed evaluation pipelines are staged.", ""])
        return "\n".join(lines)
    lines.extend(
        [
            "| Evaluation | Inference | Dataset | Split | Protocol | Pipeline | "
            "Metric | Score | Completed |",
            "|---|---|---|---|---|---|---|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"`{_markdown(row['evaluation']['id'])}` | "
            f"`{_markdown(row['evaluation']['inference_id'])}` | "
            f"`{_markdown(row['dataset']['id'])}` | "
            f"`{_markdown(row['dataset']['split'])}` | "
            f"`{_markdown(row['protocol']['id'])}` | "
            f"`{_markdown(row['pipeline']['pipeline_id'])}` | "
            f"`{_markdown(row['metric']['name'])}` | "
            f"{row['metric']['score']:.12g} | "
            f"`{_markdown(row['completed_at'])}` |"
        )
    lines.extend(
        [
            "",
            "## Artifact Contract",
            "",
            "- `report.jsonl` contains one deterministic row per evaluation pipeline.",
            "- Every row preserves exact model, dataset, protocol, engine, and lineage identities.",
            "- Pipeline artifact paths are relative to the logical model result root "
            "and are checksum-bound.",
            "- This snapshot and `report.jsonl` are rebuilt views; immutable run "
            "directories and the index/delta remain authoritative.",
            "",
        ]
    )
    return "\n".join(lines)


def _checked_artifact(
    root: Path,
    container: dict[str, Any],
    path_key: str,
    sha_key: str,
) -> Path:
    relative_text = _required_string(container.get(path_key), path_key)
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise DerivedReportError(f"{path_key} must be relative to the model result root")
    resolved_root = root.resolve(strict=True)
    candidate = (resolved_root / relative).resolve(strict=False)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise DerivedReportError(f"{path_key} escapes the model result root") from exc
    if not candidate.is_file():
        raise DerivedReportError(f"referenced artifact does not exist: {candidate}")
    expected_sha = _required_sha(container.get(sha_key), sha_key)
    actual_sha = file_sha256(candidate)
    if actual_sha != expected_sha:
        raise DerivedReportError(
            f"{path_key} checksum mismatch: expected {expected_sha}, got {actual_sha}"
        )
    return candidate


def _load_mapping(path: Path, location: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DerivedReportError(f"{location} is not valid JSON: {path}") from exc
    return _mapping(payload, location)


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DerivedReportError(f"{location} must be a mapping")
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise DerivedReportError(f"{location} must be a list")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], location: str) -> None:
    actual = set(value)
    if actual != expected:
        raise DerivedReportError(
            f"{location} fields differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _required_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DerivedReportError(f"{location} must be a non-empty string")
    return value


def _required_sha(value: Any, location: str) -> str:
    text = _required_string(value, location)
    if not _SHA256.fullmatch(text):
        raise DerivedReportError(f"{location} must be a lowercase SHA256")
    return text


def _markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("`", "\\`").replace("\n", " ")


@contextmanager
def _report_lock(model_root: Path) -> Iterator[None]:
    lock_path = model_root / ".report_views.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_text_write(path: Path, content: str, *, storage: StorageConfig) -> None:
    target = storage.assert_staging_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
