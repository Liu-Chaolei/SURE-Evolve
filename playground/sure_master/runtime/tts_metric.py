"""Use SURE's public TTS API with the scoring image's Paraformer environment."""

from __future__ import annotations

from pathlib import Path


def run_tts_metric(
    pipeline: dict, *, output_dir: str, samples_jsonl: str, device: str, cache_dir: str
) -> dict:
    from sure_eval.evaluation.cli_adapters import validate_pipeline_selection
    from sure_eval.evaluation.audio_samples import load_tts_samples_jsonl
    from sure_eval.evaluation.nodes.transcription.common.providers import (
        ParaformerZHTranscriber,
    )
    from sure_eval.evaluation.scripts.tts import run

    validate_pipeline_selection(pipeline)
    if (
        pipeline["task"] != "tts"
        or "transcription/paraformer_zh" not in pipeline["computation_node_ids"]
    ):
        raise ValueError("The worker TTS runtime requires the SURE Paraformer route")
    if not cache_dir:
        raise ValueError("The worker TTS runtime requires an explicit model cache")
    samples = load_tts_samples_jsonl(samples_jsonl, metrics=tuple(pipeline["metrics"]))
    report = run(
        samples,
        output_dir=output_dir,
        pipeline_id=pipeline["pipeline_id"],
        transcribers={
            "zh": ParaformerZHTranscriber(device=device, cache_dir=cache_dir)
        },
    )
    return {
        "status": "ok",
        "task": report.task,
        "metric": report.metric,
        "score": report.score,
        "pipeline_id": report.pipeline_id,
        "report_path": str(Path(output_dir) / "report.json"),
        "execution_runtime": "isolated_worker_paraformer",
    }
