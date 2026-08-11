"""Data models used by the EvoMaster evolution pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class RunMetrics(BaseModel):
    """Compact metrics used to compare baseline and evolved runs."""

    status: str = "unknown"
    step_count: int = 0
    llm_call_count: int = 0
    tool_call_count: int = 0
    tool_error_count: int = 0
    issue_count: int = 0
    score: float | None = None


class TraceDigest(BaseModel):
    """A compact, analysis-friendly view of one completed run directory."""

    run_dir: str
    config_path: str
    known_agents: list[str] = Field(default_factory=list)
    task_description: str | None = None
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    tool_call_counts: dict[str, int] = Field(default_factory=dict)
    tool_error_counts: dict[str, int] = Field(default_factory=dict)
    step_summaries: list[dict[str, Any]] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    workspace_manifest: list[str] = Field(default_factory=list)
    log_excerpt: str = ""


class SkillCandidate(BaseModel):
    """A skill that can be materialized as ``SKILL.md``."""

    name: str
    agent_names: list[str] = Field(default_factory=list)
    description: str
    body: str
    evidence: list[str] = Field(default_factory=list)
    importance: Literal["low", "medium", "high"] = "medium"
    source: str = "llm"


class PromptPatchCandidate(BaseModel):
    """A prompt overlay patch for one configured agent."""

    agent_name: str
    prompt_type: Literal["system", "user"] = "system"
    patch_text: str
    rationale: str = ""
    evidence: list[str] = Field(default_factory=list)
    source: str = "llm"


class ToolProposal(BaseModel):
    """A proposed tool. It is never auto-enabled by the first evolution pass."""

    name: str
    description: str
    params_schema: dict[str, Any] = Field(default_factory=dict)
    implementation_notes: str = ""
    rationale: str = ""
    evidence: list[str] = Field(default_factory=list)
    source: str = "llm"


class EvolutionCandidates(BaseModel):
    """All candidate outputs produced from one run analysis."""

    skills: list[SkillCandidate] = Field(default_factory=list)
    prompt_patches: list[PromptPatchCandidate] = Field(default_factory=list)
    tool_proposals: list[ToolProposal] = Field(default_factory=list)
    analysis_summary: str = ""


class EvolutionOverlay(BaseModel):
    """Files generated for an evolved rerun."""

    config_path: Path
    skills_dir: Path
    prompts_dir: Path
    proposals_path: Path
    summary_path: Path
    applied_skills: list[str] = Field(default_factory=list)
    applied_prompt_patches: list[str] = Field(default_factory=list)
    tool_proposals: list[str] = Field(default_factory=list)


class ExperienceSettings(BaseModel):
    """Configuration for lightweight cross-run experience evidence."""

    enabled: bool = False
    snapshot_path: str = ".evomaster/evolution_experience.v1.json"
    min_recurring_runs: int = Field(default=2, ge=2)
    min_stable_runs: int = Field(default=3, ge=2)
    min_directional_runs: int = Field(default=2, ge=1)
    stable_ratio: float = Field(default=0.67, ge=0.5, le=1.0)
    merge_similarity: float = Field(default=0.72, ge=0.0, le=1.0)
    max_records: int = Field(default=200, ge=1)
    max_events_per_record: int = Field(default=12, ge=1)
    max_hints: int = Field(default=4, ge=0)
    max_hint_chars: int = Field(default=4000, ge=0)
    score_higher_is_better: bool | None = None


class ExperienceOutcome(BaseModel):
    """Direction assigned to one applied overlay comparison."""

    direction: Literal["positive", "negative", "neutral"] = "neutral"
    reasons: list[str] = Field(default_factory=list)


class ExperienceObservation(BaseModel):
    """Compact evidence about one candidate seen in a top-level run."""

    run_id: str
    candidate_kind: Literal["skill", "prompt_patch"]
    scope: str
    fingerprint: str
    normalized_text: str
    title: str
    summary: str
    outcome: Literal["positive", "negative", "neutral"] = "neutral"
    outcome_reasons: list[str] = Field(default_factory=list)
    evidence_relation: Literal["overlay_correlated"] = "overlay_correlated"


class ExperienceRecord(BaseModel):
    """Bounded aggregate of semantically matching observations."""

    evidence_id: str
    candidate_kind: Literal["skill", "prompt_patch"]
    scope: str
    fingerprint: str
    normalized_text: str
    title: str
    summary: str
    state: Literal["observed", "recurring", "stable_positive", "stable_negative"] = "observed"
    occurrence_run_ids: list[str] = Field(default_factory=list)
    positive_run_ids: list[str] = Field(default_factory=list)
    negative_run_ids: list[str] = Field(default_factory=list)
    neutral_run_ids: list[str] = Field(default_factory=list)
    events: list[ExperienceObservation] = Field(default_factory=list)


class ExperienceSnapshot(BaseModel):
    """Versioned bounded cross-run experience snapshot."""

    schema_version: Literal[1] = 1
    records: list[ExperienceRecord] = Field(default_factory=list)


class AdvisoryEvidenceItem(BaseModel):
    """Generic bounded evidence supplied only to an analyzer."""

    source: str
    evidence_id: str
    title: str
    summary: str
    direction: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalyzerAdvisoryContext(BaseModel):
    """Shared bounded advisory channel for experience and future sources."""

    items: list[AdvisoryEvidenceItem] = Field(default_factory=list)
    max_chars: int = Field(default=4000, ge=0)

