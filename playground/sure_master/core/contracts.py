"""Versioned contracts exchanged between SURE-Master and XLab."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .utils.fingerprints import digest, portable_path

SCHEMA_VERSIONS = {
    "idea_request": "sure.idea_request.v1",
    "idea_batch": "sure.idea_batch.v1",
    "round_result": "sure.round_result.v1",
    "round_summary": "sure.round_summary.v1",
}

SearchMode = Literal["ordinary", "staged_axes"]
Axis = Literal["arch", "train", "inference"]
FailureCategory = Literal[
    "candidate_failure", "system_failure", "contract_failure", "metric_failure"
]


def _required(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: Literal["lower", "higher", "lower_is_better", "higher_is_better"]

    def __post_init__(self) -> None:
        _required(self.name, "metric.name")
        if self.direction not in {"lower", "higher", "lower_is_better", "higher_is_better"}:
            raise ValueError("unsupported metric direction")


@dataclass(frozen=True)
class IdeaRequest:
    request_id: str
    sure_run_id: str
    task_id: str
    task_description: str
    search_mode: SearchMode
    round_index: int
    requested_idea_count: int
    metric: MetricSpec
    input_digest: str
    schema_version: str = SCHEMA_VERSIONS["idea_request"]
    axis: Axis | None = None
    axis_index: int | None = None
    phase: str = "research"
    current_best: dict[str, Any] = field(default_factory=dict)
    task_card: dict[str, Any] = field(default_factory=dict)
    execution_contract: dict[str, Any] = field(default_factory=dict)
    history_artifacts: list[str] = field(default_factory=list)
    prior_rounds: list[dict[str, Any]] = field(default_factory=list)
    history_digest: str = ""
    parent_lineage: list[str] = field(default_factory=list)
    base_model_profile: dict[str, Any] = field(default_factory=dict)
    generation_policy: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("request_id", "sure_run_id", "task_id", "task_description", "input_digest"):
            _required(getattr(self, name), name)
        if self.schema_version != SCHEMA_VERSIONS["idea_request"]:
            raise ValueError("unsupported idea request schema")
        if self.search_mode not in {"ordinary", "staged_axes"}:
            raise ValueError("unsupported search mode")
        if self.round_index < 1 or self.requested_idea_count != 4:
            raise ValueError("research rounds must request exactly four ideas")
        if not isinstance(self.generation_policy, dict):
            raise ValueError("generation_policy must be an object")
        attempts = self.generation_policy.get("max_attempts", 8)
        if type(attempts) is not int or attempts < 4:
            raise ValueError("generation max_attempts must be an integer >= 4")
        if self.axis_index is not None and self.axis_index < 0:
            raise ValueError("axis_index must not be negative")
        if not self.phase.strip():
            raise ValueError("phase must not be empty")
        if self.search_mode == "ordinary" and self.axis is not None:
            raise ValueError("ordinary requests must not specify axis")
        if self.search_mode == "staged_axes" and self.axis not in {"arch", "train", "inference"}:
            raise ValueError("staged requests require axis")


@dataclass(frozen=True)
class IdeaSpec:
    implementation_instructions: str
    change_set: list[Any] = field(default_factory=list)
    invariants: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    ablation: list[str] = field(default_factory=list)
    expected_effect: str = ""
    risks: list[str] = field(default_factory=list)
    resource_class: str = ""
    change_domains: list[Axis] = field(default_factory=list)
    requires_training: bool | None = None

    def __post_init__(self) -> None:
        _required(self.implementation_instructions, "implementation_instructions")
        if any(domain not in {"arch", "train", "inference"} for domain in self.change_domains):
            raise ValueError("unsupported change domain")
        if self.requires_training is not None and not self.change_domains:
            raise ValueError("training declaration requires change domains")
        if self.change_domains:
            training = bool(set(self.change_domains) & {"arch", "train"})
            if type(self.requires_training) is not bool or self.requires_training != training:
                raise ValueError("training declaration conflicts with change domains")
            if {change.get("domain") for change in self.change_set if isinstance(change, dict)} != set(self.change_domains):
                raise ValueError("change domains must match concrete changes")


@dataclass(frozen=True)
class IdeaItem:
    idea_id: str
    artifact_id: str
    artifact_digest: str
    title: str
    candidate_type: Literal["inference", "fine_tune", "arch"]
    hypothesis: str
    mechanism: str
    spec: IdeaSpec
    axis: Axis | None = None
    evidence_refs: list[str] = field(default_factory=list)
    novelty: dict[str, Any] = field(default_factory=dict)
    native_artifact: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("idea_id", "artifact_id", "artifact_digest", "title", "hypothesis", "mechanism"):
            _required(getattr(self, name), name)
        if self.candidate_type not in {"inference", "fine_tune", "arch"}:
            raise ValueError("unsupported candidate type")
        if self.spec.change_domains:
            expected = "arch" if "arch" in self.spec.change_domains else "fine_tune" if self.spec.requires_training else "inference"
            if self.candidate_type != expected:
                raise ValueError("candidate type conflicts with execution requirements")


@dataclass(frozen=True)
class IdeaBatch:
    status: Literal["success", "incomplete", "failed"]
    request_digest: str
    xlab_run_id: str
    workspace: str
    ideas: list[IdeaItem]
    batch_digest: str
    schema_version: str = SCHEMA_VERSIONS["idea_batch"]
    blockers: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSIONS["idea_batch"]:
            raise ValueError("unsupported idea batch schema")
        if len({idea.idea_id for idea in self.ideas}) != len(self.ideas):
            raise ValueError("idea_id values must be unique")


@dataclass(frozen=True)
class RungResult:
    name: str
    success: bool
    score: float | None = None
    runtime_seconds: float | None = None
    checkpoint_artifact: str | None = None
    remote_result_artifact: str | None = None
    metric_feedback: str = ""
    promoted: bool = False
    reason_code: str | None = None
    failure_category: FailureCategory | None = None


@dataclass(frozen=True)
class CandidateResult:
    idea_id: str
    idea_artifact_id: str
    final_status: str
    code_digest: str | None = None
    workspace_ref: str | None = None
    candidate_type: str | None = None
    metric_feedback: str = ""
    rungs: list[RungResult] = field(default_factory=list)
    failure_category: FailureCategory | None = None
    reason_code: str | None = None
    improved: bool = False
    idea: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoundResult:
    sure_run_id: str
    search_mode: SearchMode
    round_index: int
    baseline_digest: str
    idea_batch_digest: str
    result_digest: str
    candidates: list[CandidateResult] = field(default_factory=list)
    ranking: list[str] = field(default_factory=list)
    phase: str = "research"
    rung_names: list[str] = field(default_factory=list)
    current_best: dict[str, Any] = field(default_factory=dict)
    parent_artifacts: list[str] = field(default_factory=list)
    feedback_digest: str = ""
    history_digest: str = ""
    artifact_refs: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSIONS["round_result"]
    axis: Axis | None = None

    metric: dict[str, str] = field(default_factory=dict)
    baseline_score: float | None = None


@dataclass(frozen=True)
class RoundSummary:
    sure_run_id: str
    search_mode: SearchMode
    round_index: int
    summary_digest: str
    supported_mechanisms: list[str] = field(default_factory=list)
    rejected_mechanisms: list[str] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    failure_counts: dict[str, int] = field(default_factory=dict)
    next_directions: list[str] = field(default_factory=list)
    attempted_idea_fingerprints: list[str] = field(default_factory=list)
    source_artifacts: list[str] = field(default_factory=list)
    history_digest: str = ""
    parent_artifacts: list[str] = field(default_factory=list)
    feedback_digest: str = ""
    phase: str = "research"
    schema_version: str = SCHEMA_VERSIONS["round_summary"]
    axis: Axis | None = None


def validate_idea_batch(batch: IdeaBatch, request: IdeaRequest) -> None:
    if request.requested_idea_count != 4:
        raise ValueError("research rounds require exactly four ideas")
    if batch.status != "success":
        raise ValueError(f"idea batch is not successful: {batch.status}")
    if batch.request_digest != request.input_digest:
        raise ValueError("idea batch request digest does not match request")
    if len(batch.ideas) != request.requested_idea_count:
        raise ValueError("idea batch count does not match request")
    portable_path(batch.workspace, ".")
    expected_candidate_type = {
        "arch": "arch",
        "train": "fine_tune",
        "inference": "inference",
    }
    for idea in batch.ideas:
        if request.execution_contract.get("allowed_change_domains") == ["arch"]:
            if (idea.candidate_type != "arch" or idea.spec.requires_training is not True
                    or set(idea.spec.change_domains) != {"arch"}
                    or not idea.spec.change_set
                    or any(not isinstance(change, dict) or change.get("domain") != "arch"
                           for change in idea.spec.change_set)):
                raise ValueError("Architecture-only search requires arch candidates with only arch changes and fixed training/inference")
        if request.search_mode == "ordinary" and idea.axis is not None:
            raise ValueError("ordinary ideas must not specify axis")
        if request.axis is not None and idea.axis != request.axis:
            raise ValueError("idea axis does not match request axis")
        if request.axis is not None and idea.candidate_type != expected_candidate_type[request.axis]:
            raise ValueError("idea candidate type does not match request axis")
        if not idea.spec.implementation_instructions.strip():
            raise ValueError("idea implementation instructions must not be empty")
        portable_path(idea.artifact_id, ".")

    artifact_ids = [idea.artifact_id for idea in batch.ideas]
    if len(set(artifact_ids)) != len(artifact_ids):
        raise ValueError("idea artifact_id values must be unique")
    artifact_digests = [idea.artifact_digest for idea in batch.ideas]
    if len(set(artifact_digests)) != len(artifact_digests):
        raise ValueError("idea artifact digests must be unique")
    scientific_fingerprints = [
        digest(
            {
                "hypothesis": idea.hypothesis.strip(),
                "mechanism": idea.mechanism.strip(),
                "change_set": idea.spec.change_set,
            }
        )
        for idea in batch.ideas
    ]
    if len(set(scientific_fingerprints)) != len(scientific_fingerprints):
        raise ValueError("idea batch contains duplicate scientific candidates")


__all__ = [
    "CandidateResult", "IdeaBatch", "IdeaItem", "IdeaRequest", "IdeaSpec",
    "MetricSpec", "RoundResult", "RoundSummary", "RungResult", "validate_idea_batch",
]
