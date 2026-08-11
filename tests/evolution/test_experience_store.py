from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import evomaster.evolution.experience as experience_module

from evomaster.evolution.experience import (
    ExperienceStore,
    candidate_fingerprint,
    classify_overlay_outcome,
    extract_experience_observations,
    normalize_candidate_text,
)
from evomaster.evolution.models import (
    EvolutionCandidates,
    EvolutionOverlay,
    ExperienceObservation,
    ExperienceSettings,
    PromptPatchCandidate,
    RunMetrics,
    SkillCandidate,
    ToolProposal,
)


def _observation(run_id: str, outcome: str = "positive", scope: str = "agent") -> ExperienceObservation:
    normalized = normalize_candidate_text("Validate the produced artifact before finishing")
    return ExperienceObservation(
        run_id=run_id,
        candidate_kind="skill",
        scope=scope,
        fingerprint=candidate_fingerprint("skill", scope, normalized),
        normalized_text=normalized,
        title="validate-artifact",
        summary="Validate the produced artifact before finishing.",
        outcome=outcome,
    )


def _write_observation(snapshot: str, run_id: str, scope: str) -> None:
    settings = ExperienceSettings(enabled=True)
    ExperienceStore(snapshot, settings).update([_observation(run_id, scope=scope)])


def test_disabled_store_creates_no_files(tmp_path: Path) -> None:
    snapshot = tmp_path / "experience.json"
    store = ExperienceStore(snapshot, ExperienceSettings())

    assert store.load_advisory_items() == []
    assert store.update([_observation("run-1")]) is False
    assert not snapshot.exists()
    assert not snapshot.with_suffix(".json.lock").exists()


def test_distinct_runs_reach_stability_and_same_run_is_deduplicated(tmp_path: Path) -> None:
    settings = ExperienceSettings(enabled=True, max_hints=4)
    snapshot = tmp_path / "experience.json"
    store = ExperienceStore(snapshot, settings)

    assert store.update([_observation("run-1"), _observation("run-1")])
    assert store.load_advisory_items() == []
    assert store.update([_observation("run-2")])
    assert store.load_advisory_items() == []
    assert store.update([_observation("run-3")])

    items = store.load_advisory_items()
    assert len(items) == 1
    assert items[0].direction == "positive"
    assert items[0].metadata["distinct_runs"] == 3
    raw = json.loads(snapshot.read_text(encoding="utf-8"))
    assert raw["records"][0]["occurrence_run_ids"] == ["run-1", "run-2", "run-3"]


def test_scope_prevents_merge(tmp_path: Path) -> None:
    settings = ExperienceSettings(enabled=True)
    snapshot = tmp_path / "experience.json"
    store = ExperienceStore(snapshot, settings)

    store.update([_observation("run-1", scope="agent-a"), _observation("run-1", scope="agent-b")])

    raw = json.loads(snapshot.read_text(encoding="utf-8"))
    assert len(raw["records"]) == 2


def test_outcome_classification_handles_score_direction_and_conflicts() -> None:
    before = RunMetrics(status="success", score=10.0, issue_count=2)
    after = RunMetrics(status="success", score=8.0, issue_count=1)

    lower_better = classify_overlay_outcome(before, after, score_higher_is_better=False)
    assert lower_better.direction == "positive"

    higher_better = classify_overlay_outcome(before, after, score_higher_is_better=True)
    assert higher_better.direction == "neutral"
    assert "conflicting_signals" in higher_better.reasons


def test_extraction_includes_only_applied_assets_and_not_tools(tmp_path: Path) -> None:
    candidates = EvolutionCandidates(
        skills=[
            SkillCandidate(
                name="applied-skill",
                agent_names=["agent"],
                description="Validate output.",
                body="Run validation before finish.",
            ),
            SkillCandidate(
                name="unused-skill",
                agent_names=["agent"],
                description="Unused.",
                body="Unused body.",
            ),
        ],
        prompt_patches=[
            PromptPatchCandidate(
                agent_name="agent",
                prompt_type="system",
                patch_text="Check outputs.",
            )
        ],
        tool_proposals=[ToolProposal(name="tool", description="Not executed")],
    )
    overlay_config = tmp_path / "config.yaml"
    overlay_config.write_text("agents:\n  agent: {}\n", encoding="utf-8")
    overlay = EvolutionOverlay(
        config_path=overlay_config,
        skills_dir=tmp_path / "skills",
        prompts_dir=tmp_path / "prompts",
        proposals_path=tmp_path / "tools.jsonl",
        summary_path=tmp_path / "summary.json",
        applied_skills=["applied-skill"],
        applied_prompt_patches=["agent:system"],
        tool_proposals=["tool"],
    )

    observations = extract_experience_observations(
        run_id="run-1",
        candidates=candidates,
        overlay=overlay,
        outcome=classify_overlay_outcome(
            RunMetrics(issue_count=1),
            RunMetrics(issue_count=0),
            score_higher_is_better=None,
        ),
    )

    assert [item.candidate_kind for item in observations] == ["skill", "prompt_patch"]
    assert all(item.evidence_relation == "overlay_correlated" for item in observations)


def test_same_run_conflicting_outcomes_are_neutral_and_history_is_bounded(tmp_path: Path) -> None:
    settings = ExperienceSettings(enabled=True, max_events_per_record=3)
    snapshot = tmp_path / "experience.json"
    store = ExperienceStore(snapshot, settings)

    store.update([_observation("run-1", "positive")])
    store.update([_observation("run-1", "negative")])
    for run_number in range(2, 8):
        store.update([_observation(f"run-{run_number}", "positive")])

    raw = json.loads(snapshot.read_text(encoding="utf-8"))
    record = raw["records"][0]
    assert len(record["events"]) == 3
    assert len(record["occurrence_run_ids"]) == 3
    assert set(record["positive_run_ids"]).isdisjoint(record["negative_run_ids"])


def test_unknown_schema_fails_open_without_overwrite(tmp_path: Path) -> None:
    snapshot = tmp_path / "experience.json"
    original = '{"schema_version": 99, "records": []}'
    snapshot.write_text(original, encoding="utf-8")
    store = ExperienceStore(snapshot, ExperienceSettings(enabled=True))

    assert store.update([_observation("run-1")]) is False
    assert snapshot.read_text(encoding="utf-8") == original


def test_atomic_replace_failure_preserves_previous_snapshot(tmp_path: Path, monkeypatch) -> None:
    snapshot = tmp_path / "experience.json"
    store = ExperienceStore(snapshot, ExperienceSettings(enabled=True))
    assert store.update([_observation("run-1")])
    original = snapshot.read_text(encoding="utf-8")

    def fail_replace(source: str, destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(experience_module.os, "replace", fail_replace)

    assert store.update([_observation("run-2")]) is False
    assert snapshot.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(".experience.json.*.tmp")) == []


def test_concurrent_writers_retain_both_updates(tmp_path: Path) -> None:
    snapshot = tmp_path / "experience.json"
    context = multiprocessing.get_context("fork")
    processes = [
        context.Process(target=_write_observation, args=(str(snapshot), "run-a", "agent-a")),
        context.Process(target=_write_observation, args=(str(snapshot), "run-b", "agent-b")),
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)

    assert all(process.exitcode == 0 for process in processes)
    raw = json.loads(snapshot.read_text(encoding="utf-8"))
    assert {record["scope"] for record in raw["records"]} == {"agent-a", "agent-b"}
