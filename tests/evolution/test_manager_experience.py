from __future__ import annotations

from pathlib import Path

from evomaster.evolution.experience import load_experience_settings
from evomaster.evolution.manager import EvolutionManager, EvolutionRunConfig
from evomaster.evolution.models import AdvisoryEvidenceItem


def _config(tmp_path: Path, experience: str = "") -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        "llm: {}\nagents: {}\n" + experience,
        encoding="utf-8",
    )
    return path


def test_manager_experience_is_disabled_by_default(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    manager = EvolutionManager(
        EvolutionRunConfig(
            agent_name="minimal",
            config_path=config_path,
            task_description="unchanged task",
            run_dir=tmp_path / "run",
            project_root=tmp_path,
        )
    )
    try:
        assert manager.experience_store is None
        assert manager.config.task_description == "unchanged task"
        assert manager._load_frozen_advisory_context().items == []
    finally:
        manager._teardown_logging()


def test_settings_resolve_snapshot_against_project_root(tmp_path: Path) -> None:
    config_path = _config(
        tmp_path,
        "evolution:\n  experience:\n    enabled: true\n    snapshot_path: state/experience.json\n",
    )

    settings, snapshot = load_experience_settings(config_path, tmp_path)

    assert settings.enabled is True
    assert snapshot == (tmp_path / "state" / "experience.json").resolve()


def test_manager_loads_advisory_items_once_per_request(tmp_path: Path) -> None:
    config_path = _config(
        tmp_path,
        "evolution:\n  experience:\n    enabled: true\n    snapshot_path: state/experience.json\n",
    )
    manager = EvolutionManager(
        EvolutionRunConfig(
            agent_name="minimal",
            config_path=config_path,
            task_description="unchanged task",
            run_dir=tmp_path / "run",
            project_root=tmp_path,
        )
    )
    calls = 0

    def load_items():
        nonlocal calls
        calls += 1
        return [
            AdvisoryEvidenceItem(
                source="cross_run_experience",
                evidence_id="exp-frozen",
                title="Frozen",
                summary="Frozen advisory.",
                direction="positive",
            )
        ]

    try:
        assert manager.experience_store is not None
        manager.experience_store.load_advisory_items = load_items
        frozen = manager._load_frozen_advisory_context()

        assert calls == 1
        assert [item.evidence_id for item in frozen.items] == ["exp-frozen"]
        assert [item.evidence_id for item in frozen.items] == ["exp-frozen"]
        assert calls == 1
    finally:
        manager._teardown_logging()


def test_manager_advisory_failure_fails_open(tmp_path: Path) -> None:
    config_path = _config(
        tmp_path,
        "evolution:\n  experience:\n    enabled: true\n    snapshot_path: state/experience.json\n",
    )
    manager = EvolutionManager(
        EvolutionRunConfig(
            agent_name="minimal",
            config_path=config_path,
            task_description="unchanged task",
            run_dir=tmp_path / "run",
            project_root=tmp_path,
        )
    )

    def fail_load():
        raise RuntimeError("unavailable")

    try:
        assert manager.experience_store is not None
        manager.experience_store.load_advisory_items = fail_load
        assert manager._load_frozen_advisory_context().items == []
    finally:
        manager._teardown_logging()
