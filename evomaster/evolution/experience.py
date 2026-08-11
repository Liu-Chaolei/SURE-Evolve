"""Lightweight cross-run experience evidence for EvoMaster evolution."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from .models import (
    AdvisoryEvidenceItem,
    EvolutionCandidates,
    EvolutionOverlay,
    ExperienceObservation,
    ExperienceOutcome,
    ExperienceRecord,
    ExperienceSettings,
    ExperienceSnapshot,
    RunMetrics,
)

logger = logging.getLogger(__name__)

_PATH_RE = re.compile(r"(?:[A-Za-z]:)?/(?:[^\s/]+/)+[^\s]*")
_RUN_FRAGMENT_RE = re.compile(r"\b(?:run|iter)[-_]?[0-9a-f]{4,}\b", re.IGNORECASE)
_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|secret|password)\b\s*[:=]\s*\S+"
)
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def load_experience_settings(
    config_path: str | Path,
    project_root: str | Path,
) -> tuple[ExperienceSettings, Path]:
    """Load optional experience settings and resolve the snapshot path."""

    path = Path(config_path).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    evolution = data.get("evolution") or {}
    raw = evolution.get("experience") or {} if isinstance(evolution, dict) else {}
    if not isinstance(raw, dict):
        raise ValueError("Config field 'evolution.experience' must be a mapping")
    settings = ExperienceSettings.model_validate(raw)
    snapshot_path = Path(settings.snapshot_path).expanduser()
    if not snapshot_path.is_absolute():
        snapshot_path = Path(project_root).resolve() / snapshot_path
    return settings, snapshot_path.resolve()


def _sanitize_text(text: str, limit: int) -> str:
    text = _SECRET_RE.sub(r"\1=<redacted>", str(text))
    text = _PATH_RE.sub("<path>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def normalize_candidate_text(text: str) -> str:
    """Normalize candidate content for deterministic cross-run matching."""

    text = _sanitize_text(text, 12_000).lower()
    text = _RUN_FRAGMENT_RE.sub(" ", text)
    return " ".join(_TOKEN_RE.findall(text))


def candidate_fingerprint(candidate_kind: str, scope: str, normalized_text: str) -> str:
    """Return a stable candidate fingerprint."""

    payload = f"{candidate_kind}\0{scope}\0{normalized_text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def candidate_similarity(left: str, right: str) -> float:
    """Compute token-set Jaccard similarity for normalized candidate text."""

    left_tokens = set(_TOKEN_RE.findall(left))
    right_tokens = set(_TOKEN_RE.findall(right))
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def classify_overlay_outcome(
    before: RunMetrics,
    after: RunMetrics,
    *,
    score_higher_is_better: bool | None,
) -> ExperienceOutcome:
    """Classify an evolved run using explicit, deterministic signals."""

    successful = {"success", "successful", "completed", "complete", "passed", "ok"}
    before_success = before.status.lower() in successful
    after_success = after.status.lower() in successful
    if before_success != after_success:
        direction: Literal["positive", "negative"] = "positive" if after_success else "negative"
        return ExperienceOutcome(
            direction=direction,
            reasons=[f"status:{before.status}->{after.status}"],
        )

    votes: list[tuple[Literal["positive", "negative"], str]] = []
    for label, old, new in (
        ("tool_errors", before.tool_error_count, after.tool_error_count),
        ("issues", before.issue_count, after.issue_count),
    ):
        if new < old:
            votes.append(("positive", f"{label}:{old}->{new}"))
        elif new > old:
            votes.append(("negative", f"{label}:{old}->{new}"))

    if (
        score_higher_is_better is not None
        and before.score is not None
        and after.score is not None
        and before.score != after.score
    ):
        improved = after.score > before.score
        if not score_higher_is_better:
            improved = not improved
        votes.append(
            (
                "positive" if improved else "negative",
                f"score:{before.score}->{after.score}",
            )
        )

    directions = {direction for direction, _ in votes}
    if len(directions) == 1:
        direction = votes[0][0]
        return ExperienceOutcome(direction=direction, reasons=[reason for _, reason in votes])
    reasons = [reason for _, reason in votes]
    if len(directions) > 1:
        reasons.append("conflicting_signals")
    elif not reasons:
        reasons.append("no_directional_signal")
    return ExperienceOutcome(direction="neutral", reasons=reasons)


def _slug(value: str, max_len: int = 64) -> str:
    value = re.sub(r"[^a-z0-9_-]+", "-", value.lower())
    value = re.sub(r"-+", "-", value).strip("-_")
    return value[:max_len].strip("-_") or "evolved"


def extract_experience_observations(
    *,
    run_id: str,
    candidates: EvolutionCandidates,
    overlay: EvolutionOverlay,
    outcome: ExperienceOutcome,
) -> list[ExperienceObservation]:
    """Extract observations only for candidates confirmed as applied."""

    observations: list[ExperienceObservation] = []
    try:
        overlay_config = yaml.safe_load(overlay.config_path.read_text(encoding="utf-8")) or {}
        agents_config = overlay_config.get("agents") or {}
        known_agents = list(agents_config) if isinstance(agents_config, dict) else []
    except (OSError, yaml.YAMLError):
        known_agents = []
    fallback_agent = known_agents[0] if known_agents else None

    applied_skills = set(overlay.applied_skills)
    seen_skills: set[str] = set()
    for skill in candidates.skills:
        skill_name = _slug(skill.name)
        if skill_name in seen_skills or skill_name not in applied_skills:
            continue
        seen_skills.add(skill_name)
        target_agents = [agent for agent in skill.agent_names if agent in known_agents]
        if not target_agents and fallback_agent:
            target_agents = [fallback_agent]
        scope = ",".join(sorted(set(target_agents))) or "general"
        raw = f"{skill.description}\n{skill.body}"
        normalized = normalize_candidate_text(raw)
        if not normalized:
            continue
        observations.append(
            ExperienceObservation(
                run_id=run_id,
                candidate_kind="skill",
                scope=scope,
                fingerprint=candidate_fingerprint("skill", scope, normalized),
                normalized_text=normalized,
                title=_sanitize_text(skill.name, 120),
                summary=_sanitize_text(skill.description, 800),
                outcome=outcome.direction,
                outcome_reasons=outcome.reasons,
            )
        )

    applied_patches = set(overlay.applied_prompt_patches)
    for patch in candidates.prompt_patches:
        target_agent = patch.agent_name if patch.agent_name in known_agents else fallback_agent
        if not target_agent:
            continue
        applied_key = f"{target_agent}:{patch.prompt_type}"
        if applied_key not in applied_patches:
            continue
        scope = applied_key
        raw = f"{patch.rationale}\n{patch.patch_text}"
        normalized = normalize_candidate_text(raw)
        if not normalized:
            continue
        observations.append(
            ExperienceObservation(
                run_id=run_id,
                candidate_kind="prompt_patch",
                scope=scope,
                fingerprint=candidate_fingerprint("prompt_patch", scope, normalized),
                normalized_text=normalized,
                title=f"Prompt guidance for {scope}",
                summary=_sanitize_text(patch.patch_text, 800),
                outcome=outcome.direction,
                outcome_reasons=outcome.reasons,
            )
        )
    return observations


class ExperienceStore:
    """Bounded JSON snapshot store for weak cross-run advisory evidence."""

    def __init__(self, snapshot_path: str | Path, settings: ExperienceSettings):
        self.snapshot_path = Path(snapshot_path)
        self.lock_path = self.snapshot_path.with_suffix(self.snapshot_path.suffix + ".lock")
        self.settings = settings

    def load_advisory_items(self) -> list[AdvisoryEvidenceItem]:
        """Load stable records as bounded analyzer advisory items."""

        if not self.settings.enabled or self.settings.max_hints == 0:
            return []
        try:
            with self._lock(shared=True):
                snapshot = self._read_snapshot()
        except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
            logger.warning("Cross-run experience retrieval disabled for this run: %s", exc)
            return []

        records = [
            record
            for record in snapshot.records
            if record.state in {"stable_positive", "stable_negative"}
        ]
        records.sort(
            key=lambda record: (
                -max(len(record.positive_run_ids), len(record.negative_run_ids)),
                -len(record.occurrence_run_ids),
                record.evidence_id,
            )
        )
        items: list[AdvisoryEvidenceItem] = []
        used_chars = 0
        for record in records:
            if len(items) >= self.settings.max_hints:
                break
            direction = "positive" if record.state == "stable_positive" else "negative"
            summary = _sanitize_text(record.summary, 800)
            item_chars = len(record.title) + len(summary)
            remaining = self.settings.max_hint_chars - used_chars
            if remaining <= 0:
                break
            if item_chars > remaining:
                summary = summary[: max(0, remaining - len(record.title))]
                item_chars = len(record.title) + len(summary)
            directional = (
                len(record.positive_run_ids) if direction == "positive" else len(record.negative_run_ids)
            )
            confidence = directional / max(1, len(record.occurrence_run_ids))
            items.append(
                AdvisoryEvidenceItem(
                    source="cross_run_experience",
                    evidence_id=record.evidence_id,
                    title=record.title,
                    summary=summary,
                    direction=direction,
                    confidence=confidence,
                    metadata={
                        "state": record.state,
                        "candidate_kind": record.candidate_kind,
                        "scope": record.scope,
                        "distinct_runs": len(record.occurrence_run_ids),
                    },
                )
            )
            used_chars += item_chars
        return items

    def update(self, observations: list[ExperienceObservation]) -> bool:
        """Merge observations atomically; return whether persistence succeeded."""

        if not self.settings.enabled or not observations:
            return False
        try:
            with self._lock(shared=False):
                snapshot = self._read_snapshot()
                for observation in observations:
                    self._merge_observation(snapshot, observation)
                self._prune(snapshot)
                self._atomic_write(snapshot)
            return True
        except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
            logger.warning("Cross-run experience update skipped: %s", exc)
            return False

    @contextmanager
    def _lock(self, *, shared: bool) -> Iterator[None]:
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_file:
            operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
            fcntl.flock(lock_file.fileno(), operation)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _read_snapshot(self) -> ExperienceSnapshot:
        if not self.snapshot_path.exists():
            return ExperienceSnapshot()
        raw = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise ValueError(f"Unsupported experience snapshot schema: {self.snapshot_path}")
        return ExperienceSnapshot.model_validate(raw)

    def _merge_observation(
        self,
        snapshot: ExperienceSnapshot,
        observation: ExperienceObservation,
    ) -> None:
        record = self._find_record(snapshot, observation)
        if record is None:
            record = ExperienceRecord(
                evidence_id=f"exp-{observation.fingerprint[:16]}",
                candidate_kind=observation.candidate_kind,
                scope=observation.scope,
                fingerprint=observation.fingerprint,
                normalized_text=observation.normalized_text,
                title=observation.title,
                summary=observation.summary,
            )
            snapshot.records.append(record)

        if observation.run_id not in record.occurrence_run_ids:
            record.occurrence_run_ids.append(observation.run_id)
        outcome_lists = {
            "positive": record.positive_run_ids,
            "negative": record.negative_run_ids,
            "neutral": record.neutral_run_ids,
        }
        previous_outcomes = {
            direction
            for direction, run_ids in outcome_lists.items()
            if observation.run_id in run_ids
        }
        effective_outcome = observation.outcome
        if previous_outcomes and observation.outcome not in previous_outcomes:
            effective_outcome = "neutral"
        for run_ids in outcome_lists.values():
            if observation.run_id in run_ids:
                run_ids.remove(observation.run_id)
        outcome_lists[effective_outcome].append(observation.run_id)
        existing_event = next(
            (event for event in record.events if event.run_id == observation.run_id),
            None,
        )
        if existing_event is None:
            record.events.append(observation)
        elif effective_outcome == "neutral" and existing_event.outcome != observation.outcome:
            existing_event.outcome = "neutral"
            existing_event.outcome_reasons = sorted(
                set(existing_event.outcome_reasons + observation.outcome_reasons + ["within_run_conflict"])
            )
        limit = self.settings.max_events_per_record
        record.events = record.events[-limit:]
        retained_runs = [event.run_id for event in record.events]
        record.occurrence_run_ids = [run_id for run_id in record.occurrence_run_ids if run_id in retained_runs]
        record.positive_run_ids = [run_id for run_id in record.positive_run_ids if run_id in retained_runs]
        record.negative_run_ids = [run_id for run_id in record.negative_run_ids if run_id in retained_runs]
        record.neutral_run_ids = [run_id for run_id in record.neutral_run_ids if run_id in retained_runs]
        self._recompute_state(record)

    def _find_record(
        self,
        snapshot: ExperienceSnapshot,
        observation: ExperienceObservation,
    ) -> ExperienceRecord | None:
        exact = next(
            (record for record in snapshot.records if record.fingerprint == observation.fingerprint),
            None,
        )
        if exact is not None:
            return exact
        eligible = [
            record
            for record in snapshot.records
            if record.candidate_kind == observation.candidate_kind and record.scope == observation.scope
        ]
        matches = [
            (candidate_similarity(record.normalized_text, observation.normalized_text), record)
            for record in eligible
        ]
        matches = [match for match in matches if match[0] >= self.settings.merge_similarity]
        return max(matches, key=lambda match: (match[0], match[1].evidence_id))[1] if matches else None

    def _recompute_state(self, record: ExperienceRecord) -> None:
        occurrences = len(set(record.occurrence_run_ids))
        positive = len(set(record.positive_run_ids))
        negative = len(set(record.negative_run_ids))
        directional = positive + negative
        record.state = "recurring" if occurrences >= self.settings.min_recurring_runs else "observed"
        if occurrences < self.settings.min_stable_runs or directional == 0:
            return
        if positive >= self.settings.min_directional_runs and positive / directional >= self.settings.stable_ratio:
            record.state = "stable_positive"
        elif negative >= self.settings.min_directional_runs and negative / directional >= self.settings.stable_ratio:
            record.state = "stable_negative"

    def _prune(self, snapshot: ExperienceSnapshot) -> None:
        rank = {"stable_positive": 3, "stable_negative": 3, "recurring": 2, "observed": 1}
        snapshot.records.sort(
            key=lambda record: (
                -rank[record.state],
                -len(record.occurrence_run_ids),
                record.evidence_id,
            )
        )
        snapshot.records = snapshot.records[: self.settings.max_records]

    def _atomic_write(self, snapshot: ExperienceSnapshot) -> None:
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.snapshot_path.name}.",
            suffix=".tmp",
            dir=self.snapshot_path.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
                json.dump(snapshot.model_dump(), temp_file, indent=2, ensure_ascii=False)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_name, self.snapshot_path)
            try:
                directory_fd = os.open(self.snapshot_path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                logger.debug("Could not fsync experience snapshot directory", exc_info=True)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
