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

