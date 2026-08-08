"""LLM-assisted and heuristic analysis for EvoMaster evolution."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from evomaster.config import ConfigManager
from evomaster.utils import Dialog, LLMConfig, SystemMessage, UserMessage, create_llm

from .models import (
    EvolutionCandidates,
    PromptPatchCandidate,
    SkillCandidate,
    ToolProposal,
    TraceDigest,
)

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are the EvoMaster self-evolution analyzer.

Analyze one completed agent run and extract reusable improvements.
Be aggressive about turning important or effective single-run knowledge into agent-local skills, but do not include secrets, private gold labels, or one-off file paths that would overfit the next task.

Generate:
1. skills: reusable methods, debugging workflows, domain facts, successful strategies, or failure-avoidance checklists.
2. prompt_patches: concise instructions that should be appended to an agent prompt.
3. tool_proposals: proposed tools only. Do not assume tools will be enabled automatically.

Return JSON only with this schema:
{
  "analysis_summary": "short summary",
  "skills": [
    {
      "name": "filesystem-safe-skill-name",
      "agent_names": ["configured_agent_name"],
      "description": "when to use this skill",
      "body": "markdown skill body with procedure and evidence",
      "evidence": ["specific observed evidence"],
      "importance": "low|medium|high"
    }
  ],
  "prompt_patches": [
    {
      "agent_name": "configured_agent_name",
      "prompt_type": "system|user",
      "patch_text": "text to append to that prompt",
      "rationale": "why this helps",
      "evidence": ["specific observed evidence"]
    }
  ],
  "tool_proposals": [
    {
      "name": "proposed_tool_name",
      "description": "what the tool should do",
      "params_schema": {"type": "object", "properties": {}},
      "implementation_notes": "how to implement safely",
      "rationale": "why a tool is warranted",
      "evidence": ["specific observed evidence"]
    }
  ]
}

Use only agent_names from known_agents. If unsure and there is a single known agent, use that agent.
"""


def _slug(value: str, max_len: int = 64) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9_-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-_")
    return value[:max_len].strip("-_") or "evolved-skill"


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(stripped[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _model_dump_json(model: TraceDigest, limit: int = 120_000) -> str:
    raw = json.dumps(model.model_dump(), ensure_ascii=False, indent=2, default=str)
    if len(raw) <= limit:
        return raw
    return raw[: limit // 2] + "\n...[trace digest truncated]...\n" + raw[-limit // 2 :]


class EvolutionAnalyzer:
    """Analyze a collected trace digest and propose evolution candidates."""

    def __init__(self, config_path: str | Path, use_llm: bool = True):
        self.config_path = Path(config_path)
        self.use_llm = use_llm

    def analyze(self, digest: TraceDigest) -> EvolutionCandidates:
        """Return LLM-generated candidates, falling back to heuristic ones."""

        logger.info(
            "Evolution analyzer started: run_dir=%s status=%s steps=%s issues=%s use_llm=%s",
            digest.run_dir,
            digest.metrics.status,
            digest.metrics.step_count,
            digest.metrics.issue_count,
            self.use_llm,
        )
        candidates: EvolutionCandidates | None = None
        if self.use_llm:
            candidates = self._analyze_with_llm(digest)

        heuristic = self._heuristic_candidates(digest)
        if candidates is None:
            logger.info("Using heuristic evolution candidates")
            self._log_candidate_summary(heuristic)
            return heuristic

        candidates = self._normalize_candidates(candidates, digest)
        if not candidates.skills:
            logger.info("LLM produced no skills; adding heuristic skill candidates")
            candidates.skills.extend(heuristic.skills)
        if not candidates.prompt_patches:
            logger.info("LLM produced no prompt patches; adding heuristic prompt candidates")
            candidates.prompt_patches.extend(heuristic.prompt_patches)
        if not candidates.tool_proposals:
            logger.info("LLM produced no tool proposals; adding heuristic tool proposals")
            candidates.tool_proposals.extend(heuristic.tool_proposals)
        self._log_candidate_summary(candidates)
        return candidates

    def _analyze_with_llm(self, digest: TraceDigest) -> EvolutionCandidates | None:
        try:
            manager = ConfigManager(
                config_dir=self.config_path.parent,
                config_file=self.config_path.name,
            )
            llm_config = manager.get_llm_config()
            logger.info(
                "Evolution analyzer LLM config: provider=%s model=%s base_url=%s",
                llm_config.get("provider"),
                llm_config.get("model"),
                llm_config.get("base_url"),
            )
            llm = create_llm(
                LLMConfig(**llm_config),
                output_config={"show_in_console": False, "log_to_file": True},
            )
            user_prompt = (
                "Known configured agents: "
                f"{digest.known_agents}\n\n"
                "Trace digest:\n"
                f"{_model_dump_json(digest)}"
            )
            logger.info("=" * 80)
            logger.info("Evolution Analyzer LLM System Prompt:")
            logger.info(_SYSTEM_PROMPT)
            logger.info("=" * 80)
            logger.info("Evolution Analyzer LLM User Prompt:")
            logger.info(user_prompt)
            logger.info("=" * 80)
            response = llm.query(
                Dialog(
                    messages=[
                        SystemMessage(content=_SYSTEM_PROMPT),
                        UserMessage(content=user_prompt),
                    ],
                    tools=[],
                ),
                temperature=0.2,
            )
            logger.info("=" * 80)
            logger.info("Evolution Analyzer LLM Raw Response:")
            logger.info(response.content or "")
            logger.info("=" * 80)
            parsed = _extract_json_object(response.content or "")
            if parsed is None:
                logger.warning("Evolution analyzer LLM did not return valid JSON")
                return None
            candidates = EvolutionCandidates.model_validate(parsed)
            logger.info(
                "Evolution analyzer LLM parsed candidates: skills=%d prompt_patches=%d tool_proposals=%d",
                len(candidates.skills),
                len(candidates.prompt_patches),
                len(candidates.tool_proposals),
            )
            return candidates
        except Exception as exc:
            logger.warning("LLM evolution analysis failed, using heuristic fallback: %s", exc)
            return None

    @staticmethod
    def _log_candidate_summary(candidates: EvolutionCandidates) -> None:
        logger.info("Evolution analysis summary: %s", candidates.analysis_summary)
        logger.info("Skill candidates (%d):", len(candidates.skills))
        for skill in candidates.skills:
            logger.info(
                "  skill name=%s agents=%s importance=%s description=%s",
                skill.name,
                skill.agent_names,
                skill.importance,
                skill.description,
            )
            logger.info("  skill evidence=%s", skill.evidence)
            logger.info("  skill body:\n%s", skill.body)
        logger.info("Prompt patch candidates (%d):", len(candidates.prompt_patches))
        for patch in candidates.prompt_patches:
            logger.info(
                "  prompt_patch agent=%s type=%s rationale=%s patch=%s",
                patch.agent_name,
                patch.prompt_type,
                patch.rationale,
                patch.patch_text,
            )
            logger.info("  prompt_patch evidence=%s", patch.evidence)
        logger.info("Tool proposals (%d):", len(candidates.tool_proposals))
        for proposal in candidates.tool_proposals:
            logger.info(
                "  tool_proposal name=%s description=%s rationale=%s",
                proposal.name,
                proposal.description,
                proposal.rationale,
            )
            logger.info("  tool_proposal schema=%s", proposal.params_schema)
            logger.info("  tool_proposal implementation_notes=%s", proposal.implementation_notes)
            logger.info("  tool_proposal evidence=%s", proposal.evidence)

    def _normalize_candidates(
        self,
        candidates: EvolutionCandidates,
        digest: TraceDigest,
    ) -> EvolutionCandidates:
        known = set(digest.known_agents)
        default_agents = digest.known_agents[:1]
        for skill in candidates.skills:
            skill.name = _slug(skill.name)
            skill.agent_names = [a for a in skill.agent_names if a in known] or default_agents
            skill.description = skill.description.strip()[:600]
            skill.body = skill.body.strip()[:8000]
            skill.evidence = [str(e)[:1000] for e in skill.evidence[:8]]
        for patch in candidates.prompt_patches:
            if patch.agent_name not in known and default_agents:
                patch.agent_name = default_agents[0]
            patch.patch_text = patch.patch_text.strip()[:4000]
            patch.rationale = patch.rationale.strip()[:1000]
            patch.evidence = [str(e)[:1000] for e in patch.evidence[:8]]
        for proposal in candidates.tool_proposals:
            proposal.name = _slug(proposal.name, max_len=48).replace("-", "_")
            proposal.description = proposal.description.strip()[:800]
            proposal.implementation_notes = proposal.implementation_notes.strip()[:4000]
            proposal.rationale = proposal.rationale.strip()[:1000]
            proposal.evidence = [str(e)[:1000] for e in proposal.evidence[:8]]
        return candidates

    def _heuristic_candidates(self, digest: TraceDigest) -> EvolutionCandidates:
        known_agents = digest.known_agents or ["general"]
        agent = known_agents[0]
        run_hash = hashlib.sha1(digest.run_dir.encode("utf-8")).hexdigest()[:8]

        issue_text = "\n".join(f"- {issue}" for issue in digest.issues[:12])
        tool_text = "\n".join(
            f"- {name}: {count}" for name, count in sorted(digest.tool_call_counts.items())
        ) or "- No tool calls recorded."
        artifact_text = "\n".join(f"- {name}" for name in list(digest.artifacts)[:12]) or "- No score artifacts found."

        if digest.issues:
            title = "Run Feedback Debugging Lessons"
            description = (
                "Use after a similar run shows tool errors, execution failures, "
                "invalid outputs, or missing artifacts."
            )
            body = f"""# {title}

## Observed Signals
{issue_text}

## Reusable Procedure
1. Start from the first failing tool response or log line, not the final answer.
2. Reproduce the smallest failing command in the workspace.
3. Inspect stderr, missing files, schema mismatches, and timeout limits before changing the main approach.
4. After a fix, rerun the same validation command and confirm the expected artifact exists.
5. Only finish after the validation result is visible in the run feedback.

## Tool Usage Seen
{tool_text}
"""
        else:
            title = "Successful Run Method"
            description = (
                "Use when a task resembles this run and the agent needs a compact "
                "record of the effective workflow."
            )
            body = f"""# {title}

## Reusable Procedure
1. Reuse the successful high-level workflow from this trace.
2. Prefer the tools and validation sequence that produced a completed run.
3. Preserve output-format constraints observed in the final successful response.

## Tool Usage Seen
{tool_text}

## Artifacts
{artifact_text}
"""

        skill = SkillCandidate(
            name=f"evo-{agent}-{_slug(title)}-{run_hash}",
            agent_names=[agent],
            description=description,
            body=body,
            evidence=digest.issues[:6] or [f"Run completed with status={digest.metrics.status}"],
            importance="high" if digest.issues else "medium",
            source="heuristic",
        )

        patches: list[PromptPatchCandidate] = []
        if digest.issues:
            patches.append(
                PromptPatchCandidate(
                    agent_name=agent,
                    prompt_type="system",
                    patch_text=(
                        "When environment feedback contains an error, timeout, missing file, "
                        "invalid output, or nonzero exit code, diagnose that concrete signal first. "
                        "Rerun a minimal validation command after fixing it, and do not call finish "
                        "until the expected artifact or answer format has been verified."
                    ),
                    rationale="The previous run contained concrete failure feedback that should drive the next action.",
                    evidence=digest.issues[:5],
                    source="heuristic",
                )
            )

        proposals: list[ToolProposal] = []
        if digest.tool_error_counts or digest.tool_call_counts.get("execute_bash", 0) >= 3:
            proposals.append(
                ToolProposal(
                    name="run_and_validate_artifact",
                    description=(
                        "Run a command and verify that expected output files or patterns exist, "
                        "returning a structured success/failure report."
                    ),
                    params_schema={
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "expected_paths": {"type": "array", "items": {"type": "string"}},
                            "timeout": {"type": "integer"},
                        },
                        "required": ["command"],
                    },
                    implementation_notes=(
                        "Wrap the existing session bash execution, then check expected paths relative "
                        "to the active workspace. Do not auto-enable until reviewed."
                    ),
                    rationale="Repeated command execution and validation appeared in the trace.",
                    evidence=digest.issues[:5] or [f"Tool counts: {digest.tool_call_counts}"],
                    source="heuristic",
                )
            )

        return EvolutionCandidates(
            skills=[skill],
            prompt_patches=patches,
            tool_proposals=proposals,
            analysis_summary=(
                f"Heuristic analysis of {digest.run_dir}: status={digest.metrics.status}, "
                f"issues={digest.metrics.issue_count}, tools={digest.tool_call_counts}"
            ),
        )
