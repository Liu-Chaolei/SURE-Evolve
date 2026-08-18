"""Create reviewable evaluation runs without writing to the published registries."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Literal

from sure_eval.models.registry import ModelRegistry
from sure_eval.results import AppendOnlyResultStore, file_sha256, validate_result_index
from sure_eval.storage import StorageConfig, load_storage_config, validate_model_id

from .bridge import EvaluationEngine, EvaluationInput, load_evaluation_input


PUBLISHED_REPORT_SCHEMA = "sure.eval.published_report.v2"
InferenceSource = Literal["published", "staged"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _dataset_key(dataset: dict[str, Any]) -> str:
    return f"{dataset['id']}@{dataset['split']}"


class EvaluationOrchestrationError(RuntimeError):
    """Raised when evaluation lineage cannot be proved before execution."""


class EvaluationRunOrchestrator:
    def __init__(
        self,
        *,
        storage: StorageConfig | None = None,
        engine: EvaluationEngine | None = None,
    ) -> None:
        self.storage = storage or load_storage_config()
        self.engine = engine or EvaluationEngine()

    def run(
        self,
        *,
        model_id: str,
        evaluation_id: str,
        inference_id: str,
        inference_source: InferenceSource,
        input_manifests: list[str | Path],
    ) -> dict[str, Any]:
        model_id = validate_model_id(model_id)
        evaluation_id = validate_model_id(evaluation_id)
        inference_id = validate_model_id(inference_id)
        if inference_source not in {"published", "staged"}:
            raise EvaluationOrchestrationError(
                "inference_source must be exactly 'published' or 'staged'"
            )
        if not input_manifests:
            raise EvaluationOrchestrationError("at least one evaluation input manifest is required")

        model = ModelRegistry(storage=self.storage).require_verified(model_id)
        source_root, source_index = self._load_source_index(model_id, inference_source)
        if source_index["model"] != {
            "id": model_id,
            "artifact_sha256": model.artifact_sha256,
        }:
            raise EvaluationOrchestrationError(
                "inference result model identity does not match the verified model adapter"
            )
        inference = next(
            (item for item in source_index["inference_runs"] if item["id"] == inference_id),
            None,
        )
        if inference is None:
            raise EvaluationOrchestrationError(f"inference run does not exist: {inference_id}")

        loaded = [(Path(path).resolve(strict=True), load_evaluation_input(path)) for path in input_manifests]
        requested_dataset_keys = {_dataset_key(item.dataset) for _, item in loaded}
        source_datasets = {
            _dataset_key(item["dataset"]): item["dataset"] for item in inference["datasets"]
        }
        if requested_dataset_keys != set(source_datasets):
            raise EvaluationOrchestrationError(
                "evaluation input dataset set must exactly match the source inference dataset set"
            )
        for _, evaluation_input in loaded:
            key = _dataset_key(evaluation_input.dataset)
            if evaluation_input.dataset != source_datasets[key]:
                raise EvaluationOrchestrationError(
                    f"evaluation dataset identity differs from source inference: {key}"
                )

        store = AppendOnlyResultStore(model_id, model.artifact_sha256, storage=self.storage)
        run_root = store.reserve_evaluation_run(evaluation_id)
        dataset_outputs: dict[str, dict[str, Any]] = {}
        for ordinal, (manifest_path, evaluation_input) in enumerate(loaded, start=1):
            dataset_key = _dataset_key(evaluation_input.dataset)
            item_root = run_root / f"item-{ordinal:03d}"
            item_root.mkdir()
            input_copy = item_root / "evaluation_input.json"
            store.write_json_once(
                input_copy,
                json.loads(manifest_path.read_text(encoding="utf-8")),
            )
            bridge_result = self.engine.evaluate(
                evaluation_input,
                output_dir=item_root / "engine",
                trace_dir=item_root / "trace",
            )
            pipeline_id = str(bridge_result.get("pipeline_id", ""))
            metric_name = bridge_result.get("metric")
            score = bridge_result.get("score")
            if not pipeline_id or not isinstance(metric_name, str) or not metric_name:
                raise EvaluationOrchestrationError("engine result is missing pipeline or metric identity")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise EvaluationOrchestrationError("engine result score must be numeric")

            engine_report = Path(bridge_result["engine_report_path"])
            pipeline_description = Path(bridge_result["pipeline_description_path"])
            report_path = item_root / "report.json"
            report = {
                "schema": PUBLISHED_REPORT_SCHEMA,
                "status": "success",
                "model": {"id": model_id, "artifact_sha256": model.artifact_sha256},
                "dataset": evaluation_input.dataset,
                "protocol": dict(inference["protocol"]),
                "evaluation": {
                    "pipeline_id": pipeline_id,
                    "engine_commit": self.engine.identity["commit"],
                    "engine_tree_sha256": self.engine.identity["tree_sha256"],
                },
                "metric": {"name": metric_name, "score": score},
                "lineage": {
                    "inference_id": inference_id,
                    "inference_source": inference_source,
                    "evaluation_input_path": str(input_copy.relative_to(store.model_root)),
                    "evaluation_input_sha256": file_sha256(input_copy),
                },
                "completed_at": _utc_now(),
            }
            store.write_json_once(report_path, report)
            pipeline = {
                "pipeline_id": pipeline_id,
                "report_path": str(report_path.relative_to(store.model_root)),
                "report_sha256": file_sha256(report_path),
                "engine_report_path": str(engine_report.relative_to(store.model_root)),
                "engine_report_sha256": file_sha256(engine_report),
                "pipeline_description_path": str(
                    pipeline_description.relative_to(store.model_root)
                ),
                "pipeline_description_sha256": file_sha256(pipeline_description),
            }
            output = dataset_outputs.setdefault(
                dataset_key,
                {
                    "dataset": evaluation_input.dataset,
                    "protocol": dict(inference["protocol"]),
                    "pipelines": [],
                },
            )
            if any(item["pipeline_id"] == pipeline_id for item in output["pipelines"]):
                raise EvaluationOrchestrationError(
                    f"duplicate pipeline for dataset {dataset_key}: {pipeline_id}"
                )
            output["pipelines"].append(pipeline)

        entry = {
            "id": evaluation_id,
            "inference_id": inference_id,
            "engine": self.engine.identity,
            "datasets": [dataset_outputs[key] for key in sorted(dataset_outputs)],
        }
        if inference_source == "published":
            store.append_published_inference_evaluation(
                entry,
                published_index_path=source_root / "result_index.json",
            )
        else:
            store.append_evaluation_entry(entry)
            validate_result_index(
                store.model_root,
                expected_model_id=model_id,
                require_verified=False,
            )
        publication_manifest = store.write_publication_manifest()
        return {
            "status": "staged_for_human_review",
            "model_id": model_id,
            "inference_id": inference_id,
            "evaluation_id": evaluation_id,
            "inference_source": inference_source,
            "evaluation_root": str(run_root),
            "publication_manifest": str(publication_manifest),
            "automatic_publish_allowed": False,
        }

    def _load_source_index(
        self,
        model_id: str,
        source: InferenceSource,
    ) -> tuple[Path, dict[str, Any]]:
        if source == "published":
            root = self.storage.published_results(model_id)
            self.storage.assert_registry_read_path(root / "result_index.json")
            return root, validate_result_index(
                root,
                expected_model_id=model_id,
                require_verified=True,
            )
        root = self.storage.staged_results(model_id)
        self.storage.assert_staging_write_path(root / "result_index.json")
        return root, validate_result_index(
            root,
            expected_model_id=model_id,
            require_verified=False,
        )
