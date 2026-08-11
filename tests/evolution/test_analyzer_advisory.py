from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import evomaster.evolution.analyzer as analyzer_module
from evomaster.evolution.analyzer import EvolutionAnalyzer
from evomaster.evolution.models import (
    AdvisoryEvidenceItem,
    AnalyzerAdvisoryContext,
    TraceDigest,
)


def _context() -> AnalyzerAdvisoryContext:
    return AnalyzerAdvisoryContext(
        items=[
            AdvisoryEvidenceItem(
                source="cross_run_experience",
                evidence_id="exp-123",
                title="Validate artifacts",
                summary="Confirm required artifacts exist before finishing.",
                direction="positive",
                metadata={"state": "stable_positive"},
            )
        ],
        max_chars=1000,
    )


def test_advisory_serialization_is_bounded() -> None:
    context = _context()
    context.max_chars = 40

    serialized = EvolutionAnalyzer._serialize_advisory_context(context)

    assert len(serialized) <= 40


def test_heuristic_mode_does_not_create_candidates_from_advisory() -> None:
    analyzer = EvolutionAnalyzer("unused.yaml", use_llm=False)
    digest = TraceDigest(run_dir="run-a", config_path="config.yaml", known_agents=["agent"])

    without_advisory = analyzer.analyze(digest)
    with_advisory = analyzer.analyze(digest, advisory_context=_context())

    assert with_advisory == without_advisory
    assert "exp-123" not in with_advisory.model_dump_json()


def test_advisory_uses_one_existing_llm_request_and_is_omitted_from_logs(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("llm: {}\nagents: {}\n", encoding="utf-8")
    calls = []
    llm_output_configs = []

    class FakeConfigManager:
        def __init__(self, **kwargs):
            pass

        def get_llm_config(self):
            return {"provider": "openai", "model": "fake", "api_key": "fake"}

    class FakeLLM:
        def query(self, dialog, temperature):
            calls.append((dialog, temperature))
            return SimpleNamespace(
                content=(
                    '{"analysis_summary":"ok","skills":[],"prompt_patches":[],"tool_proposals":[]}'
                )
            )

    monkeypatch.setattr(analyzer_module, "ConfigManager", FakeConfigManager)

    def create_fake_llm(*args, **kwargs):
        llm_output_configs.append(kwargs["output_config"])
        return FakeLLM()

    monkeypatch.setattr(analyzer_module, "create_llm", create_fake_llm)
    caplog.set_level(logging.INFO, logger=analyzer_module.__name__)
    digest = TraceDigest(run_dir="run-a", config_path=str(config), known_agents=["agent"])

    EvolutionAnalyzer(config).analyze(digest, advisory_context=_context())

    assert len(calls) == 1
    assert llm_output_configs == [{"show_in_console": False, "log_to_file": False}]
    prompt = calls[0][0].messages[-1].content
    assert "exp-123" in prompt
    assert "Confirm required artifacts exist before finishing." in prompt
    assert "exp-123" in caplog.text
    assert "Confirm required artifacts exist before finishing." not in caplog.text
