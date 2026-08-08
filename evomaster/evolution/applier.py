"""Apply evolution candidates into a reversible run-local overlay."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from .models import EvolutionCandidates, EvolutionOverlay, PromptPatchCandidate, SkillCandidate

logger = logging.getLogger(__name__)


def _slug(value: str, max_len: int = 64) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9_-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-_")
    return value[:max_len].strip("-_") or "evolved"


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _append_unique(items: list[Any], value: Any) -> list[Any]:
    if value not in items:
        items.append(value)
    return items


class EvolutionApplier:
    """Materialize skills, prompt overlays, tool proposals, and overlay config."""

    def __init__(self, project_root: str | Path, config_path: str | Path, output_dir: str | Path):
        self.project_root = Path(project_root).resolve()
        self.config_path = Path(config_path).resolve()
        self.config_dir = self.config_path.parent
        self.output_dir = Path(output_dir).resolve()

    def apply(
        self,
        candidates: EvolutionCandidates,
        source_config_path: str | Path,
        iteration: int,
    ) -> EvolutionOverlay:
        source_config_path = Path(source_config_path).resolve()
        config = _read_yaml(source_config_path)
        evolution_cfg = config.setdefault("evolution", {})
        if isinstance(evolution_cfg, dict):
            evolution_cfg.setdefault("base_config_dir", str(self.config_dir))
        else:
            config["evolution"] = {"base_config_dir": str(self.config_dir)}
        self._pin_session_config_dir(config)

        agents_cfg = config.get("agents") or {}
        if not isinstance(agents_cfg, dict):
            raise ValueError("Config field 'agents' must be a mapping for evolution overlays")

        iter_dir = self.output_dir / f"iter_{iteration:03d}"
        skills_dir = iter_dir / "skills"
        prompts_dir = iter_dir / "prompts"
        proposals_path = iter_dir / "tool_proposals.jsonl"
        summary_path = iter_dir / "overlay_summary.json"
        config_path = iter_dir / "config.yaml"
        skills_dir.mkdir(parents=True, exist_ok=True)
        prompts_dir.mkdir(parents=True, exist_ok=True)

        known_agents = list(agents_cfg.keys())
        applied_skills = self._write_skills(candidates.skills, skills_dir)
        self._enable_skills(config, candidates.skills, applied_skills, known_agents, skills_dir)

        applied_patches = self._write_prompt_overlays(
            candidates.prompt_patches,
            config,
            agents_cfg,
            prompts_dir,
            known_agents,
        )

        tool_proposals = self._write_tool_proposals(candidates, proposals_path)

        _write_yaml(config_path, config)
        summary = {
            "source_config": str(source_config_path),
            "overlay_config": str(config_path),
            "skills_dir": str(skills_dir),
            "prompts_dir": str(prompts_dir),
            "applied_skills": applied_skills,
            "applied_prompt_patches": applied_patches,
            "tool_proposals": tool_proposals,
            "analysis_summary": candidates.analysis_summary,
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        return EvolutionOverlay(
            config_path=config_path,
            skills_dir=skills_dir,
            prompts_dir=prompts_dir,
            proposals_path=proposals_path,
            summary_path=summary_path,
            applied_skills=applied_skills,
            applied_prompt_patches=applied_patches,
            tool_proposals=tool_proposals,
        )

    def _write_skills(self, skills: list[SkillCandidate], skills_dir: Path) -> list[str]:
        applied: list[str] = []
        seen: set[str] = set()
        for skill in skills:
            name = _slug(skill.name)
            if name in seen:
                continue
            if not skill.description.strip() or not skill.body.strip():
                continue
            seen.add(name)
            skill_path = skills_dir / name
            skill_path.mkdir(parents=True, exist_ok=True)
            description_lines = skill.description.strip().splitlines() or ["Generated evolution skill."]
            description_block = "\n".join(f"  {line}" for line in description_lines)
            frontmatter = (
                f"name: {name}\n"
                "description: |\n"
                f"{description_block}\n"
                "license: null"
            )
            evidence = "\n".join(f"- {item}" for item in skill.evidence[:8])
            body = skill.body.strip()
            content = (
                f"---\n{frontmatter}\n---\n\n"
                f"{body}\n\n"
                "## Evolution Evidence\n"
                f"{evidence or '- Generated from a completed EvoMaster run.'}\n"
            )
            (skill_path / "SKILL.md").write_text(content, encoding="utf-8")
            applied.append(name)
        return applied

    def _pin_session_config_dir(self, config: dict[str, Any]) -> None:
        """Keep session-relative symlink/volume paths tied to the original config."""
        session_cfg = config.get("session")
        if not isinstance(session_cfg, dict):
            return
        for key in ("local", "docker"):
            sub_cfg = session_cfg.get(key)
            if isinstance(sub_cfg, dict):
                sub_cfg.setdefault("config_dir", str(self.config_dir))

    def _enable_skills(
        self,
        config: dict[str, Any],
        skill_candidates: list[SkillCandidate],
        applied_skills: list[str],
        known_agents: list[str],
        skills_dir: Path,
    ) -> None:
        if not applied_skills:
            return

        skills_cfg = config.setdefault("skills", {})
        if not isinstance(skills_cfg, dict):
            skills_cfg = {}
            config["skills"] = skills_cfg
        skills_cfg.setdefault("enabled", True)
        skills_cfg.setdefault("skills_root", "evomaster/skills")
        extra_roots = skills_cfg.get("extra_roots") or []
        if isinstance(extra_roots, str):
            extra_roots = [extra_roots]
        if not isinstance(extra_roots, list):
            extra_roots = []
        _append_unique(extra_roots, str(skills_dir))
        skills_cfg["extra_roots"] = extra_roots

        agents_cfg = config.get("agents") or {}
        if not isinstance(agents_cfg, dict):
            return

        fallback_agents = known_agents[:1]
        for candidate in skill_candidates:
            skill_name = _slug(candidate.name)
            if skill_name not in applied_skills:
                continue
            target_agents = [a for a in candidate.agent_names if a in agents_cfg]
            if not target_agents:
                target_agents = fallback_agents
            for agent in target_agents:
                agent_cfg = agents_cfg.get(agent)
                if not isinstance(agent_cfg, dict):
                    continue
                current = agent_cfg.get("skills")
                if current == "*":
                    continue
                if current is None:
                    current = []
                if isinstance(current, str):
                    current = [current]
                if not isinstance(current, list):
                    current = []
                _append_unique(current, skill_name)
                agent_cfg["skills"] = current

    def _write_prompt_overlays(
        self,
        patches: list[PromptPatchCandidate],
        config: dict[str, Any],
        agents_cfg: dict[str, Any],
        prompts_dir: Path,
        known_agents: list[str],
    ) -> list[str]:
        applied: list[str] = []
        grouped: dict[tuple[str, str], list[PromptPatchCandidate]] = {}
        fallback_agent = known_agents[0] if known_agents else None
        for patch in patches:
            agent = patch.agent_name if patch.agent_name in agents_cfg else fallback_agent
            if not agent or not patch.patch_text.strip():
                continue
            grouped.setdefault((agent, patch.prompt_type), []).append(patch)

        for (agent, prompt_type), group in grouped.items():
            agent_cfg = agents_cfg.get(agent)
            if not isinstance(agent_cfg, dict):
                continue
            field = "system_prompt_file" if prompt_type == "system" else "user_prompt_file"
            original_path = self._resolve_prompt_path(agent_cfg.get(field))
            if original_path is None or not original_path.exists():
                logger.warning("Cannot apply prompt patch for %s: %s not found", agent, field)
                continue
            original = original_path.read_text(encoding="utf-8")
            patch_blocks = []
            for idx, patch in enumerate(group, 1):
                evidence = "\n".join(f"- {item}" for item in patch.evidence[:5])
                patch_blocks.append(
                    f"## Evolution Patch {idx}\n"
                    f"Rationale: {patch.rationale.strip() or 'Generated from run feedback.'}\n\n"
                    f"{patch.patch_text.strip()}\n\n"
                    f"Evidence:\n{evidence or '- See evolution overlay summary.'}"
                )

            evolved = (
                original.rstrip()
                + "\n\n# EvoMaster Evolution Notes\n"
                + "\n\n".join(patch_blocks)
                + "\n"
            )
            out_path = prompts_dir / agent / f"{prompt_type}_prompt.evolved.txt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(evolved, encoding="utf-8")
            agent_cfg[field] = str(out_path)
            applied.append(f"{agent}:{prompt_type}")
        return applied

    def _resolve_prompt_path(self, raw_path: Any) -> Path | None:
        if not raw_path or not isinstance(raw_path, str):
            return None
        path = Path(raw_path)
        if path.is_absolute():
            return path
        playground_base = Path(str(self.config_dir).replace("configs", "playground"))
        candidate = playground_base / path
        if candidate.exists():
            return candidate.resolve()
        return (self.config_dir / path).resolve()

    def _write_tool_proposals(
        self,
        candidates: EvolutionCandidates,
        proposals_path: Path,
    ) -> list[str]:
        proposals_path.parent.mkdir(parents=True, exist_ok=True)
        names: list[str] = []
        with proposals_path.open("w", encoding="utf-8") as f:
            for proposal in candidates.tool_proposals:
                if not proposal.name.strip() or not proposal.description.strip():
                    continue
                names.append(proposal.name)
                f.write(json.dumps(proposal.model_dump(), ensure_ascii=False, default=str) + "\n")
        return names
