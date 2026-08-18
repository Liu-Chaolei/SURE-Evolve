"""Select verified NFS results for reuse with all-or-nothing dataset matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

from sure_eval.agent.main_flow_input import ALLOWED_EXECUTION_MODES
from sure_eval.storage import StorageConfig, load_storage_config, validate_model_id

from .schema import (
    ResultIndexError,
    validate_dataset_identity,
    validate_protocol_identity,
    validate_result_index,
)


REUSE_DECISION_SCHEMA = "sure.eval.result_reuse_decision.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ResultRegistryError(ValueError):
    """Raised for invalid reuse requests or unavailable mandatory artifacts."""


def _dataset_key(dataset: dict[str, Any]) -> str:
    return f"{dataset['id']}@{dataset['split']}"


@dataclass(frozen=True)
class ReuseRequest:
    model_id: str
    model_artifact_sha256: str
    protocol_id: str
    protocol_version: str
    effective_params_sha256: str
    datasets: tuple[dict[str, Any], ...]
    mode: str = "auto"
    engine_commit: str | None = None
    engine_tree_sha256: str | None = None
    pipeline_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_model_id(self.model_id)
        if not _SHA256.fullmatch(self.model_artifact_sha256):
            raise ResultRegistryError("model_artifact_sha256 must be a lowercase SHA256")
        validate_protocol_identity(
            {
                "id": self.protocol_id,
                "version": self.protocol_version,
                "effective_params_sha256": self.effective_params_sha256,
            },
            "request.protocol",
        )
        if self.mode not in ALLOWED_EXECUTION_MODES:
            raise ResultRegistryError(f"unsupported execution mode: {self.mode}")
        if not self.datasets:
            raise ResultRegistryError("reuse request requires at least one dataset")
        keys: set[str] = set()
        for index, dataset in enumerate(self.datasets):
            validated = validate_dataset_identity(dataset, f"request.datasets[{index}]")
            key = _dataset_key(validated)
            if key in keys:
                raise ResultRegistryError(f"duplicate requested dataset: {key}")
            keys.add(key)
        if self.mode == "reuse_result" and not self.pipeline_ids:
            raise ResultRegistryError("reuse_result requires the current evaluation pipeline identities")
        if self.pipeline_ids:
            if set(self.pipeline_ids) != keys:
                raise ResultRegistryError(
                    "pipeline_ids keys must exactly match the requested dataset identity set"
                )
            for dataset_key, values in self.pipeline_ids.items():
                if not values or len(set(values)) != len(values):
                    raise ResultRegistryError(
                        f"pipeline_ids[{dataset_key!r}] must be a non-empty unique tuple"
                    )


@dataclass(frozen=True)
class ReuseDecision:
    status: str
    action: str
    model_id: str
    mode: str
    source_index: str | None
    inference_id: str | None
    evaluation_id: str | None
    checks: tuple[dict[str, Any], ...]
    reason_codes: tuple[str, ...]
    schema: str = REUSE_DECISION_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status,
            "action": self.action,
            "model_id": self.model_id,
            "mode": self.mode,
            "source_index": self.source_index,
            "inference_id": self.inference_id,
            "evaluation_id": self.evaluation_id,
            "checks": list(self.checks),
            "reason_codes": list(self.reason_codes),
        }


class ResultRegistry:
    """Read only human-verified results from the canonical NFS root."""

    def __init__(self, *, storage: StorageConfig | None = None) -> None:
        self.storage = storage or load_storage_config()

    def decide(self, request: ReuseRequest) -> ReuseDecision:
        if request.mode == "retest":
            return self._decision(
                request,
                action="infer_and_evaluate",
                checks=({"field": "execution.mode", "passed": True, "actual": "retest"},),
                reasons=("forced_retest",),
            )

        model_root = self.storage.published_results(request.model_id, require_exists=False)
        index_path = model_root / "result_index.json"
        declared_index_path = (
            self.storage.declared_published_results(request.model_id) / "result_index.json"
        )
        if not index_path.is_file():
            return self._missing_candidate(request, "published_index_missing")
        try:
            index = validate_result_index(
                model_root,
                expected_model_id=request.model_id,
                require_verified=True,
            )
        except (ResultIndexError, OSError, ValueError) as exc:
            return self._decision(
                request,
                status="blocked",
                action="blocked",
                source_index=str(declared_index_path),
                checks=({"field": "result_index", "passed": False, "error": str(exc)},),
                reasons=("published_index_invalid",),
            )

        checks: list[dict[str, Any]] = []
        model = index["model"]
        model_matches = model.get("id") == request.model_id
        digest_matches = model.get("artifact_sha256") == request.model_artifact_sha256
        checks.extend(
            (
                {
                    "field": "model.id",
                    "expected": request.model_id,
                    "actual": model.get("id"),
                    "passed": model_matches,
                },
                {
                    "field": "model.artifact_sha256",
                    "expected": request.model_artifact_sha256,
                    "actual": model.get("artifact_sha256"),
                    "passed": digest_matches,
                },
            )
        )
        if not model_matches or not digest_matches:
            return self._no_match(request, declared_index_path, checks, "model_identity_mismatch")

        inference = self._select_inference(index, request, checks)
        if inference is None:
            return self._no_match(
                request, declared_index_path, checks, "inference_identity_mismatch"
            )

        if request.mode == "reevaluate":
            return self._decision(
                request,
                action="reevaluate",
                source_index=str(declared_index_path),
                inference_id=inference["id"],
                checks=tuple(checks),
                reasons=("verified_predictions_reused", "evaluation_forced"),
            )

        evaluation = self._select_evaluation(index, inference, request, checks)
        if evaluation is not None:
            return self._decision(
                request,
                action="reuse_result",
                source_index=str(declared_index_path),
                inference_id=inference["id"],
                evaluation_id=evaluation["id"],
                checks=tuple(checks),
                reasons=("verified_result_exact_match",),
            )
        if request.mode == "reuse_result":
            return self._decision(
                request,
                status="blocked",
                action="blocked",
                source_index=str(declared_index_path),
                inference_id=inference["id"],
                checks=tuple(checks),
                reasons=("matching_evaluation_not_found",),
            )
        return self._decision(
            request,
            action="reevaluate",
            source_index=str(declared_index_path),
            inference_id=inference["id"],
            checks=tuple(checks),
            reasons=("verified_predictions_reused", "evaluation_identity_changed"),
        )

    def _select_inference(
        self,
        index: dict[str, Any],
        request: ReuseRequest,
        checks: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        requested_datasets = {_dataset_key(item): item for item in request.datasets}
        candidates: list[dict[str, Any]] = []
        for run in index["inference_runs"]:
            protocol = run["protocol"]
            protocol_matches = (
                protocol["id"] == request.protocol_id
                and protocol["version"] == request.protocol_version
                and protocol["effective_params_sha256"] == request.effective_params_sha256
            )
            run_datasets = {
                _dataset_key(item["dataset"]): item["dataset"] for item in run["datasets"]
            }
            dataset_matches = run_datasets == requested_datasets
            if protocol_matches and dataset_matches:
                candidates.append(run)
        selected = sorted(candidates, key=lambda item: (str(item.get("created_at", "")), item["id"]))[-1] if candidates else None
        checks.append(
            {
                "field": "inference.protocol_and_dataset_set",
                "expected": {
                    "protocol_id": request.protocol_id,
                    "protocol_version": request.protocol_version,
                    "effective_params_sha256": request.effective_params_sha256,
                    "datasets": sorted(requested_datasets),
                },
                "actual": selected["id"] if selected else None,
                "passed": selected is not None,
                "candidate_count": len(candidates),
            }
        )
        return selected

    def _select_evaluation(
        self,
        index: dict[str, Any],
        inference: dict[str, Any],
        request: ReuseRequest,
        checks: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not request.engine_commit or not request.engine_tree_sha256 or not request.pipeline_ids:
            checks.append(
                {
                    "field": "evaluation.identity",
                    "passed": False,
                    "actual": None,
                    "reason": "current engine and pipeline identities are required",
                }
            )
            return None
        expected_pipelines = {
            key: tuple(sorted(values)) for key, values in request.pipeline_ids.items()
        }
        candidates: list[dict[str, Any]] = []
        for run in index["evaluation_runs"]:
            if run["inference_id"] != inference["id"]:
                continue
            engine = run["engine"]
            if (
                engine.get("commit") != request.engine_commit
                or engine.get("tree_sha256") != request.engine_tree_sha256
            ):
                continue
            actual_pipelines = {
                _dataset_key(item["dataset"]): tuple(
                    sorted(pipeline["pipeline_id"] for pipeline in item["pipelines"])
                )
                for item in run["datasets"]
            }
            if actual_pipelines == expected_pipelines:
                candidates.append(run)
        selected = sorted(candidates, key=lambda item: (str(item.get("created_at", "")), item["id"]))[-1] if candidates else None
        checks.append(
            {
                "field": "evaluation.engine_and_pipeline_set",
                "expected": {
                    "commit": request.engine_commit,
                    "tree_sha256": request.engine_tree_sha256,
                    "pipelines": expected_pipelines,
                },
                "actual": selected["id"] if selected else None,
                "passed": selected is not None,
                "candidate_count": len(candidates),
            }
        )
        return selected

    def _missing_candidate(self, request: ReuseRequest, reason: str) -> ReuseDecision:
        if request.mode in {"reuse_result", "reevaluate"}:
            return self._decision(
                request,
                status="blocked",
                action="blocked",
                checks=({"field": "result_index", "passed": False, "actual": None},),
                reasons=(reason,),
            )
        return self._decision(
            request,
            action="infer_and_evaluate",
            checks=({"field": "result_index", "passed": False, "actual": None},),
            reasons=(reason, "new_inference_required"),
        )

    def _no_match(
        self,
        request: ReuseRequest,
        index_path: Path,
        checks: list[dict[str, Any]],
        reason: str,
    ) -> ReuseDecision:
        if request.mode in {"reuse_result", "reevaluate"}:
            return self._decision(
                request,
                status="blocked",
                action="blocked",
                source_index=str(index_path),
                checks=tuple(checks),
                reasons=(reason,),
            )
        return self._decision(
            request,
            action="infer_and_evaluate",
            source_index=str(index_path),
            checks=tuple(checks),
            reasons=(reason, "new_inference_required"),
        )

    @staticmethod
    def _decision(
        request: ReuseRequest,
        *,
        action: str,
        checks: tuple[dict[str, Any], ...],
        reasons: tuple[str, ...],
        status: str = "ready",
        source_index: str | None = None,
        inference_id: str | None = None,
        evaluation_id: str | None = None,
    ) -> ReuseDecision:
        return ReuseDecision(
            status=status,
            action=action,
            model_id=request.model_id,
            mode=request.mode,
            source_index=source_index,
            inference_id=inference_id,
            evaluation_id=evaluation_id,
            checks=checks,
            reason_codes=reasons,
        )
