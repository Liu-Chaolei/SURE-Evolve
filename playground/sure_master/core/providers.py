"""Idea provider boundary used by the SURE-Master round loop."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .contracts import (
    IdeaBatch,
    IdeaItem,
    IdeaRequest,
    IdeaSpec,
    RoundResult,
    RoundSummary,
    validate_idea_batch,
)
from .utils.fingerprints import digest


class XlabIdeaProvider(Protocol):
    def generate(self, request: IdeaRequest) -> IdeaBatch: ...

    def summarize(self, result: RoundResult) -> RoundSummary: ...

    def reconcile(self, operation_id: str) -> dict[str, Any]: ...

    def close(self) -> None: ...


@dataclass
class FakeXlabIdeaProvider:
    """Deterministic provider for contract and control-flow tests."""

    xlab_run_id: str = "fake-xlab-run"
    operation_state: dict[str, dict[str, Any]] | None = None
    generated_requests: list[IdeaRequest] | None = None
    summarized_results: list[RoundResult] | None = None

    def __post_init__(self) -> None:
        if self.operation_state is None:
            self.operation_state = {}
        if self.generated_requests is None:
            self.generated_requests = []
        if self.summarized_results is None:
            self.summarized_results = []

    def generate(self, request: IdeaRequest) -> IdeaBatch:
        self.generated_requests.append(request)
        axis = request.axis
        candidate_type = {"arch": "arch", "train": "fine_tune", "inference": "inference"}.get(axis or "", "inference")
        ideas = []
        for index in range(request.requested_idea_count):
            idea_id = f"fake-{axis or 'ordinary'}-r{request.round_index}-i{index + 1}"
            spec = {
                "change_set": [f"parameter_{index + 1}"],
                "implementation_instructions": f"Apply deterministic change {index + 1}.",
                "invariants": ["preserve evaluator and artifact contract"],
                "success_criteria": ["improve the configured metric"],
                "ablation": [],
                "expected_effect": "controlled improvement",
                "risks": [],
                "resource_class": "test",
            }
            if request.execution_contract.get("allowed_change_domains") == ["arch"]:
                candidate_type = "arch"
                spec.update(change_domains=["arch"], requires_training=True,
                            change_set=[{"domain": "arch", "target": f"component_{index}",
                                         "description": "Change model structure"}])
            artifact = {"idea_id": idea_id, "spec": spec, "candidate_type": candidate_type}
            ideas.append(
                IdeaItem(
                    idea_id=idea_id,
                    artifact_id=f"fake-artifact-{index + 1}",
                    artifact_digest=digest(artifact),
                    title=idea_id,
                    axis=axis,
                    candidate_type=candidate_type,
                    hypothesis=f"Change {index + 1} improves the metric.",
                    mechanism="deterministic test mechanism",
                    evidence_refs=[],
                    novelty={"risk_level": "unknown", "confidence": "low"},
                    spec=IdeaSpec(**spec),
                )
            )
        batch_data = {
            "request_digest": request.input_digest,
            "xlab_run_id": self.xlab_run_id,
            "workspace": "fake",
            "ideas": [asdict(idea) for idea in ideas],
        }
        batch = IdeaBatch(
            status="success",
            request_digest=request.input_digest,
            xlab_run_id=self.xlab_run_id,
            workspace="fake",
            ideas=ideas,
            batch_digest=digest(batch_data),
        )
        validate_idea_batch(batch, request)
        self.operation_state[f"generate:{request.request_id}"] = {"status": "published", "batch_digest": batch.batch_digest}
        return batch

    def summarize(self, result: RoundResult) -> RoundSummary:
        self.summarized_results.append(result)
        failures: dict[str, int] = {}
        for candidate in result.candidates:
            if candidate.failure_category:
                failures[candidate.failure_category] = failures.get(candidate.failure_category, 0) + 1
        summary_data = {
            "sure_run_id": result.sure_run_id,
            "round_index": result.round_index,
            "ranking": result.ranking,
            "failures": failures,
            "history_digest": result.history_digest,
            "feedback_digest": result.feedback_digest,
        }
        summary = RoundSummary(
            sure_run_id=result.sure_run_id,
            search_mode=result.search_mode,
            axis=result.axis,
            round_index=result.round_index,
            supported_mechanisms=["deterministic test mechanism"],
            rejected_mechanisms=[],
            failure_counts=failures,
            next_directions=["continue deterministic validation"],
            attempted_idea_fingerprints=[candidate.idea_id for candidate in result.candidates],
            source_artifacts=list(result.artifact_refs),
            history_digest=result.history_digest,
            parent_artifacts=list(result.parent_artifacts),
            feedback_digest=result.feedback_digest,
            phase=result.phase,
            summary_digest=digest(summary_data),
        )
        self.operation_state[f"summarize:{result.sure_run_id}:{result.round_index}"] = {"status": "published", "summary_digest": summary.summary_digest}
        return summary

    def reconcile(self, operation_id: str) -> dict[str, Any]:
        return dict(self.operation_state.get(operation_id, {"status": "unknown"}))

    def close(self) -> None:
        return None
