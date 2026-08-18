from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_runner():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_speech_understanding_metric_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_speech_understanding_metric_pipeline", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_speech_understanding_asr_uses_route_backed_run_task(tmp_path, monkeypatch) -> None:
    runner = _load_runner()
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "ref_asr.txt").write_text("utt1\thello world\n", encoding="utf-8")
    (artifacts_dir / "hyp_asr.txt").write_text("utt1\thello word\n", encoding="utf-8")

    calls = []

    def fake_run_task(task: str, **kwargs):
        calls.append((task, kwargs))
        return SimpleNamespace(
            metric="wer",
            score=0.5,
            pipeline_id="asr.en.wer.whisper_norm_english_v1.wenet_wer_v1",
            pipeline_kind="atomic",
            member_pipeline_ids=(),
            computation_node_ids=("normalization/whisper_norm", "scoring/wenet_wer"),
            details={"scoring_result": {"wer": 0.5}},
            pipeline_trace=(
                SimpleNamespace(
                    stage="scoring",
                    node_id="scoring/wenet_wer",
                    version="v1",
                    internal_stages=("wer",),
                ),
            ),
        )

    monkeypatch.setattr(runner, "run_task", fake_run_task, raising=False)

    result = runner.evaluate_task("ASR", artifacts_dir, "en", "zh")

    assert calls == [
        (
            "asr",
            {
                "ref_file": str(artifacts_dir / "ref_asr.txt"),
                "hyp_file": str(artifacts_dir / "hyp_asr.txt"),
                "language": "en",
                "metric": "wer",
                "output_dir": str(artifacts_dir / ".sure_eval_metrics" / "asr"),
            },
        )
    ]
    assert result["metric"]["backend"] == "sure_eval.evaluation.scripts.run_task"
    assert result["metric"]["pipeline_id"] == "asr.en.wer.whisper_norm_english_v1.wenet_wer_v1"
    assert result["metric"]["details"]["route_config_path"] == "src/sure_eval/evaluation/tasks/asr/routes.yaml"
