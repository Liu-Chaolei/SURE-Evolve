from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from sure_eval.core.config import Config


def _extract_json_payload(stdout: str) -> dict:
    start = stdout.rfind("\n{")
    if start == -1:
        start = stdout.find("{")
    else:
        start += 1
    return json.loads(stdout[start:])


def _write_config(tmp_path: Path) -> Path:
    config = Config.from_env().model_dump()
    config["data"]["datasets"] = str(tmp_path / "datasets")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return config_path


def _write_dataset(tmp_path: Path, dataset_name: str = "aishell1") -> None:
    jsonl_dir = tmp_path / "datasets" / "sure_benchmark" / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = jsonl_dir / f"{dataset_name}.jsonl"
    rows = [
        {"key": "utt1", "path": "a.wav", "target": "你好", "task": "ASR", "language": "zh", "dataset": dataset_name},
        {"key": "utt2", "path": "b.wav", "target": "世界", "task": "ASR", "language": "zh", "dataset": dataset_name},
    ]
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_materialize_predictions_template_generates_manifest_and_template(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    _write_dataset(tmp_path)
    output_dir = tmp_path / "templates"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/materialize_predictions_template.py",
            "--dataset",
            "aishell1",
            "--output-dir",
            str(output_dir),
            "--config",
            str(config_path),
        ],
        cwd="/mnt/cloudstorfs/sjtu_home/junhao.du/sure-eval-sandbox",
        check=True,
        capture_output=True,
        text=True,
    )

    payload = _extract_json_payload(result.stdout)
    assert payload["templates"][0]["dataset"] == "aishell1"
    assert (output_dir / "aishell1.txt").read_text(encoding="utf-8").splitlines() == ["utt1\t", "utt2\t"]
    assert (output_dir / "manifest.json").exists()


def test_materialize_predictions_template_preserves_existing_predictions(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    _write_dataset(tmp_path)
    output_dir = tmp_path / "templates"
    output_dir.mkdir()
    prediction_path = output_dir / "aishell1.txt"
    prediction_path.write_text("utt1\talready done\nutt2\t\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/materialize_predictions_template.py",
            "--dataset",
            "aishell1",
            "--output-dir",
            str(output_dir),
            "--config",
            str(config_path),
        ],
        cwd="/mnt/cloudstorfs/sjtu_home/junhao.du/sure-eval-sandbox",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert prediction_path.read_text(encoding="utf-8") == "utt1\talready done\nutt2\t\n"
    payload = _extract_json_payload(result.stdout)
    assert payload["templates"][0]["template_path"] == str(prediction_path)


def test_multi_dataset_run_template_delegates_to_v2_single_dataset_surface() -> None:
    template = Path("docs/agents/main_flow_agent/templates/run_single_model.sh").read_text(
        encoding="utf-8"
    )

    assert "run_single_model_single_dataset.sh" in template
    assert "--overwrite" not in template


def test_validate_prediction_files_reports_missing_extra_and_empty(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    _write_dataset(tmp_path)
    pred_path = tmp_path / "aishell1.txt"
    pred_path.write_text("utt1\t你好\nutt3\t额外项\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_prediction_files.py",
            "--dataset",
            "aishell1",
            "--pred",
            "aishell1",
            str(pred_path),
            "--config",
            str(config_path),
            "--require-nonempty",
        ],
        cwd="/mnt/cloudstorfs/sjtu_home/junhao.du/sure-eval-sandbox",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["is_valid"] is False
    report = payload["results"][0]
    assert report["missing_keys"] == ["utt2"]
    assert report["extra_keys"] == ["utt3"]
    assert report["empty_prediction_keys"] == []


def test_validate_prediction_files_remaps_workspace_audio_paths(tmp_path: Path) -> None:
    from scripts.validate_prediction_files import validate_prediction_file

    dataset_name = "toy_tts"
    repo_root = Path.cwd()
    generated_audio = repo_root / "tmp_validate_prediction_audio.wav"
    generated_audio.write_bytes(b"fake")
    try:
        jsonl_path = tmp_path / f"{dataset_name}.jsonl"
        jsonl_path.write_text(
            json.dumps(
                {
                    "key": "utt1",
                    "target": "hello",
                    "task": "TTS",
                    "language": "en",
                    "reference_audio": str(generated_audio),
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        pred_path = tmp_path / f"{dataset_name}.txt"
        container_audio = f"/workspace/sure-eval/{generated_audio.relative_to(repo_root)}"
        pred_path.write_text(f"utt1\t{container_audio}\n", encoding="utf-8")
        pred_path.with_suffix(".jsonl").write_text(
            json.dumps(
                {
                    "key": "utt1",
                    "dataset": dataset_name,
                    "task": "TTS",
                    "language": "en",
                    "prediction": {"audio_path": container_audio},
                    "normalized_prediction": container_audio,
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        class DatasetManagerStub:
            def normalize_dataset_name(self, name: str) -> str:
                return name

            def get_jsonl_path(self, name: str) -> Path:
                return jsonl_path

            def download_and_convert(self, name: str) -> Path:
                raise AssertionError("dataset should already exist")

        result = validate_prediction_file(
            DatasetManagerStub(),
            dataset_name,
            pred_path,
            require_nonempty=True,
        )

        assert result["is_valid"] is True
        assert result["contract_violation_keys"] == []
    finally:
        generated_audio.unlink(missing_ok=True)


def test_tts_server_arguments_include_prompt_text_to_avoid_asr_fallback(tmp_path: Path) -> None:
    from scripts.generate_predictions_via_server import _build_tool_arguments

    prompt_audio = tmp_path / "prompt.wav"
    prompt_audio.write_bytes(b"RIFF")
    sample = {
        "key": "tts1",
        "task": "TTS",
        "language": "en",
        "path": str(prompt_audio),
        "reference_audio": str(prompt_audio),
        "target": "generate this text",
        "reference_text": "generate this text",
        "prompt_text": "speaker prompt text",
    }

    arguments = _build_tool_arguments(
        repo_root=tmp_path,
        sample=sample,
        task="TTS",
        language="en",
        argument_name="audio_path",
        audio_path=prompt_audio,
        output_audio_dir=tmp_path / "audio",
    )

    assert arguments["text"] == "generate this text"
    assert arguments["prompt_audio_path"] == str(prompt_audio)
    assert arguments["prompt_wav_path"] == str(prompt_audio)
    assert arguments["audio_path"] == str(tmp_path / "audio" / "tts1.wav")
    assert arguments["output_path"] == str(tmp_path / "audio" / "tts1.wav")
    assert arguments["language"] == "English"
    assert arguments["prompt_text"] == "speaker prompt text"
    assert arguments["ref_text"] == "speaker prompt text"


def test_tts_server_arguments_normalize_chinese_language_code(tmp_path: Path) -> None:
    from scripts.generate_predictions_via_server import _build_tool_arguments

    prompt_audio = tmp_path / "prompt.wav"
    prompt_audio.write_bytes(b"RIFF")
    sample = {
        "key": "tts-zh",
        "task": "TTS",
        "language": "zh",
        "path": str(prompt_audio),
        "reference_audio": str(prompt_audio),
        "target": "生成这句话",
    }

    arguments = _build_tool_arguments(
        repo_root=tmp_path,
        sample=sample,
        task="TTS",
        language="zh",
        argument_name="audio_path",
        audio_path=prompt_audio,
        output_audio_dir=tmp_path / "audio",
    )

    assert arguments["language"] == "Chinese"


def test_tts_server_arguments_include_explicit_tool_args(tmp_path: Path) -> None:
    from scripts.generate_predictions_via_server import _build_tool_arguments

    prompt_audio = tmp_path / "prompt.wav"
    prompt_audio.write_bytes(b"RIFF")
    sample = {
        "key": "tts-zh-hard",
        "task": "TTS",
        "language": "zh",
        "path": str(prompt_audio),
        "reference_audio": str(prompt_audio),
        "target": "生成这句话",
    }

    arguments = _build_tool_arguments(
        repo_root=tmp_path,
        sample=sample,
        task="TTS",
        language="zh",
        argument_name="audio_path",
        audio_path=prompt_audio,
        output_audio_dir=tmp_path / "audio",
        tool_args={"max_new_tokens": 2048},
    )

    assert arguments["max_new_tokens"] == 2048


def test_extract_response_payload_accepts_direct_json_rpc_result() -> None:
    from scripts.generate_predictions_via_server import _extract_response_payload

    response = {
        "jsonrpc": "2.0",
        "id": 3,
        "result": {
            "text": "hello",
            "audio_path": "/tmp/generated.wav",
            "language": "en",
            "raw": {"sample_rate": 48000},
        },
    }

    assert _extract_response_payload(response) == response["result"]


def test_prediction_status_upsert_preserves_other_datasets(tmp_path: Path) -> None:
    from scripts.generate_predictions_via_server import _upsert_dataset_status

    status_path = tmp_path / "prediction_generation_status.json"
    payload = {
        "run_id": "run1",
        "model_name": "model1",
        "execution_path": "direct_server_use",
        "protocol_id": "strict_core",
        "tool_name": "synthesize_speech",
        "datasets": [
            {"dataset": "seedtts_test_eval_en", "status": "completed", "num_generated_samples": 1088},
        ],
    }
    status_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    next_dataset = {"dataset": "seedtts_test_eval_zh", "status": "running", "num_generated_samples": 0}

    merged, current = _upsert_dataset_status(status_path, payload, next_dataset)

    assert current is merged["datasets"][1]
    assert merged["datasets"] == [
        {"dataset": "seedtts_test_eval_en", "status": "completed", "num_generated_samples": 1088},
        {"dataset": "seedtts_test_eval_zh", "status": "running", "num_generated_samples": 0},
    ]


def test_resume_predictions_can_load_nonempty_result_log(tmp_path: Path) -> None:
    from scripts.generate_predictions_via_server import _load_existing_predictions

    prediction_path = tmp_path / "seedtts_test_eval_en.txt"
    result_log_path = tmp_path / "seedtts_test_eval_en_results.log"
    prediction_path.write_text("utt1\t\nutt2\told.wav\n", encoding="utf-8")
    result_log_path.write_text("utt1\tnew.wav\nutt3\tthird.wav\n", encoding="utf-8")

    merged = _load_existing_predictions(prediction_path)
    merged.update(_load_existing_predictions(result_log_path))

    assert merged == {"utt1": "new.wav", "utt2": "old.wav", "utt3": "third.wav"}


def test_resume_predictions_can_exclude_invalid_repair_keys(tmp_path: Path) -> None:
    from scripts.generate_predictions_via_server import _load_existing_predictions

    prediction_path = tmp_path / "seedtts_test_eval_zh.txt"
    result_log_path = tmp_path / "seedtts_test_eval_zh_results.log"
    prediction_path.write_text(
        "utt1\tError: old language failure\nutt2\told-valid.wav\n",
        encoding="utf-8",
    )
    result_log_path.write_text(
        "utt1\tError: old language failure\nutt3\told-log-only.wav\n",
        encoding="utf-8",
    )

    merged = _load_existing_predictions(prediction_path, exclude_keys={"utt1"})
    merged.update(_load_existing_predictions(result_log_path, exclude_keys={"utt1"}))

    assert merged == {"utt2": "old-valid.wav", "utt3": "old-log-only.wav"}


def test_write_prediction_snapshots_preserves_dataset_layout(tmp_path: Path) -> None:
    from scripts.generate_predictions_via_server import _write_prediction_snapshots

    samples = [
        {"key": "utt1", "language": "en"},
        {"key": "utt2", "language": "en"},
    ]
    prediction_path = tmp_path / "predictions" / "seedtts_test_eval_en.txt"
    structured_prediction_path = tmp_path / "predictions" / "seedtts_test_eval_en.jsonl"
    prediction_path.parent.mkdir()

    _write_prediction_snapshots(
        samples=samples,
        prediction_path=prediction_path,
        structured_prediction_path=structured_prediction_path,
        prediction_map={"utt1": "audio/seedtts_test_eval_en/utt1.wav"},
        structured_map={
            "utt1": {
                "key": "utt1",
                "dataset": "seedtts_test_eval_en",
                "task": "TTS",
                "language": "en",
                "prediction": {"audio_path": "audio/seedtts_test_eval_en/utt1.wav"},
                "normalized_prediction": "audio/seedtts_test_eval_en/utt1.wav",
                "raw_response": {"audio_path": "audio/seedtts_test_eval_en/utt1.wav"},
            }
        },
        canonical_dataset="seedtts_test_eval_en",
        sample_task="TTS",
        sample_language="en",
    )

    assert prediction_path.read_text(encoding="utf-8").splitlines() == [
        "utt1\taudio/seedtts_test_eval_en/utt1.wav",
        "utt2\t",
    ]
    rows = [json.loads(line) for line in structured_prediction_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["dataset"] == "seedtts_test_eval_en"
    assert rows[0]["normalized_prediction"] == "audio/seedtts_test_eval_en/utt1.wav"
    assert rows[1]["dataset"] == "seedtts_test_eval_en"
    assert rows[1]["normalized_prediction"] == ""


def test_existing_result_log_entries_keep_out_of_subset_predictions(tmp_path: Path) -> None:
    from scripts.generate_predictions_via_server import _write_existing_result_log_entries

    log_path = tmp_path / "results.log"
    samples = [{"key": "utt1"}, {"key": "utt2"}]
    predictions = {"utt1": "one.wav", "utt3": "three.wav"}

    with log_path.open("w", encoding="utf-8") as handle:
        _write_existing_result_log_entries(handle, samples, predictions)

    assert log_path.read_text(encoding="utf-8").splitlines() == [
        "utt1\tone.wav",
        "utt3\tthree.wav",
    ]


def test_resume_pending_prediction_keys_detects_complete_resume() -> None:
    from scripts.generate_predictions_via_server import _pending_prediction_keys

    samples = [{"key": "utt1"}, {"key": "utt2"}, {"key": "utt3"}]

    assert _pending_prediction_keys(samples, {"utt1": "one", "utt2": "two", "utt3": "three"}) == []
    assert _pending_prediction_keys(samples, {"utt1": "one", "utt3": "three"}) == ["utt2"]


def test_extract_response_payload_rejects_tool_error_result() -> None:
    import pytest

    from scripts.generate_predictions_via_server import _extract_response_payload

    response = {
        "jsonrpc": "2.0",
        "id": 3,
        "result": {
            "content": [{"type": "text", "text": "Error: unsupported language"}],
            "isError": True,
        },
    }

    with pytest.raises(RuntimeError, match="unsupported language"):
        _extract_response_payload(response)


def test_evaluate_predictions_filters_language_routed_tts_semantic_metrics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import scripts.evaluate_predictions as evaluate_predictions

    jsonl_dir = tmp_path / "jsonl"
    pred_dir = tmp_path / "predictions"
    jsonl_dir.mkdir()
    pred_dir.mkdir()
    datasets = {
        "toy_tts_en": {
            "key": "en1",
            "task": "TTS",
            "language": "en",
            "target": "hello",
            "reference_audio": str(tmp_path / "ref_en.wav"),
        },
        "toy_tts_zh": {
            "key": "zh1",
            "task": "TTS",
            "language": "zh",
            "target": "你好",
            "reference_audio": str(tmp_path / "ref_zh.wav"),
        },
    }
    for dataset, row in datasets.items():
        (jsonl_dir / f"{dataset}.jsonl").write_text(
            json.dumps(row, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (pred_dir / f"{dataset}.txt").write_text(f"{row['key']}\t{tmp_path / (dataset + '.wav')}\n", encoding="utf-8")

    class DatasetManagerStub:
        def expand_dataset_names(self, names: list[str]) -> list[str]:
            return names

        def normalize_dataset_name(self, name: str) -> str:
            return name

        def get_jsonl_path(self, name: str) -> Path:
            return jsonl_dir / f"{name}.jsonl"

        def download_and_convert(self, name: str) -> Path:
            raise AssertionError("dataset should already exist")

    class RPSManagerStub:
        database = None

    calls: list[tuple[str, str | None]] = []

    def fake_evaluate_prediction_file(
        dataset_manager,
        sota_manager,
        dataset_name: str,
        prediction_path: Path,
        metric_override: str | None = None,
        metrics_root: Path | None = None,
        device: str = "cuda",
    ) -> dict:
        calls.append((dataset_name, metric_override))
        return {
            "dataset": dataset_name,
            "task": "TTS",
            "metric": metric_override,
            "score": 0.0,
            "result": {"score": 0.0},
        }

    monkeypatch.setattr(evaluate_predictions, "_build_dataset_and_rps_managers", lambda _config: (DatasetManagerStub(), RPSManagerStub()))
    monkeypatch.setattr(evaluate_predictions, "evaluate_prediction_file", fake_evaluate_prediction_file)
    monkeypatch.setattr(evaluate_predictions, "write_standard_evaluation_artifacts", lambda **_kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_predictions.py",
            "--dataset",
            "toy_tts_en",
            "toy_tts_zh",
            "--pred-dir",
            str(pred_dir),
            "--metric",
            "tts_wer",
            "--metric",
            "tts_cer",
            "--metric",
            "dnsmos",
        ],
    )

    assert evaluate_predictions.main() == 0
    assert calls == [
        ("toy_tts_en", "tts_wer"),
        ("toy_tts_en", "dnsmos"),
        ("toy_tts_zh", "tts_cer"),
        ("toy_tts_zh", "dnsmos"),
    ]


def test_evaluate_prediction_file_routes_asr_through_canonical_pipeline(tmp_path: Path) -> None:
    from scripts.evaluate_predictions import evaluate_prediction_file
    from sure_eval.evaluation.sure_evaluator import SUREEvaluator

    dataset_name = "toy_asr"
    jsonl_path = tmp_path / "toy_asr.jsonl"
    rows = [
        {"key": "utt1", "target": "hello world", "task": "ASR", "language": "en"},
        {"key": "utt2", "target": "I have 2 apples.", "task": "ASR", "language": "en"},
    ]
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    pred_path = tmp_path / "pred.txt"
    pred_path.write_text("utt1\thello brave world\nutt2\tI have two apples\n", encoding="utf-8")

    class DatasetManagerStub:
        def normalize_dataset_name(self, name: str) -> str:
            return name

        def get_jsonl_path(self, name: str) -> Path:
            return jsonl_path

        def download_and_convert(self, name: str) -> Path:
            raise AssertionError("dataset should already exist")

    class SOTAManagerStub:
        def get_metric(self, name: str) -> str:
            return "wer"

        def get_baseline(self, name: str):
            return None

        def calculate_rps(self, name: str, score: float) -> None:
            return None

    result = evaluate_prediction_file(
        DatasetManagerStub(),
        SOTAManagerStub(),
        dataset_name,
        pred_path,
    )

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    ref_file.write_text("utt1\thello world\nutt2\tI have 2 apples.\n", encoding="utf-8")
    hyp_file.write_text("utt1\thello brave world\nutt2\tI have two apples\n", encoding="utf-8")
    legacy = SUREEvaluator(language="en").evaluate("ASR", str(ref_file), str(hyp_file))

    assert result["task"] == "ASR"
    assert result["metric"] == "wer"
    assert result["score"] == legacy["score"]
    assert result["details"] == legacy


def test_evaluate_prediction_file_injects_tts_runtime_for_audio_metrics(monkeypatch, tmp_path: Path) -> None:
    from scripts import evaluate_predictions
    from sure_eval.evaluation.core.types import EvaluationReport

    dataset_name = "toy_tts"
    jsonl_path = tmp_path / "toy_tts.jsonl"
    generated_audio = tmp_path / "generated.wav"
    reference_audio = tmp_path / "reference.wav"
    generated_audio.write_bytes(b"fake")
    reference_audio.write_bytes(b"fake")
    jsonl_path.write_text(
        json.dumps(
            {
                "key": "utt1",
                "target": "hello world",
                "reference_audio": str(reference_audio),
                "task": "TTS",
                "language": "en",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    pred_path = tmp_path / "pred.txt"
    pred_path.write_text(f"utt1\t{generated_audio}\n", encoding="utf-8")

    class DatasetManagerStub:
        def normalize_dataset_name(self, name: str) -> str:
            return name

        def get_jsonl_path(self, name: str) -> Path:
            return jsonl_path

        def download_and_convert(self, name: str) -> Path:
            raise AssertionError("dataset should already exist")

    class SOTAManagerStub:
        def get_metric(self, name: str) -> str | None:
            return None

        def get_baseline(self, name: str):
            return None

        def calculate_rps(self, name: str, score: float):
            return {"status": "missing_baseline", "dataset": name, "score": score}

    calls: dict[str, object] = {}
    speaker_provider = object()
    mos_provider = object()

    def fake_build_tts_runtime(**kwargs):
        calls["runtime_kwargs"] = kwargs
        return {
            "transcribers": {},
            "speaker_providers": {"wavlm-large": speaker_provider},
            "mos_providers": {"dnsmos": mos_provider},
        }

    def fake_run_task(task: str, **kwargs):
        calls["run_task"] = {"task": task, "kwargs": kwargs}
        return EvaluationReport(
            task="TTS",
            language="en",
            metric="sim/wavlm-large",
            score=0.7,
            pipeline_id="tts.en.multi.audio_metric_nodes",
            pipeline_trace=(),
            details={"results": {"sim/wavlm-large": {"score": 0.7}}},
        )

    monkeypatch.setattr(evaluate_predictions, "build_tts_runtime", fake_build_tts_runtime)
    monkeypatch.setattr(evaluate_predictions, "run_task", fake_run_task)

    result = evaluate_predictions.evaluate_prediction_file(
        DatasetManagerStub(),
        SOTAManagerStub(),
        dataset_name,
        pred_path,
        metric_override="sim/wavlm-large",
    )

    assert result["metric"] == "sim/wavlm-large"
    assert calls["runtime_kwargs"] == {
        "metrics": ("sim/wavlm-large",),
        "language": "en",
        "device": "cuda",
        "cache_dir": None,
    }
    run_kwargs = calls["run_task"]["kwargs"]
    assert run_kwargs["transcribers"] == {}
    assert run_kwargs["speaker_providers"] == {"wavlm-large": speaker_provider}
    assert run_kwargs["mos_providers"] == {"dnsmos": mos_provider}


def test_evaluate_prediction_file_routes_s2tt_through_canonical_pipeline(tmp_path: Path) -> None:
    from scripts.evaluate_predictions import evaluate_prediction_file
    from sure_eval.evaluation.sure_evaluator import SUREEvaluator

    dataset_name = "toy_s2tt"
    jsonl_path = tmp_path / "toy_s2tt.jsonl"
    rows = [
        {"key": "utt1", "target": "你好世界。", "task": "S2TT", "language": "zh"},
        {"key": "utt2", "target": "今天天气很好。", "task": "S2TT", "language": "zh"},
    ]
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    pred_path = tmp_path / "pred.txt"
    pred_path.write_text("utt1\t你好世界。\nutt2\t今天的天气很好。\n", encoding="utf-8")

    class DatasetManagerStub:
        def normalize_dataset_name(self, name: str) -> str:
            return name

        def get_jsonl_path(self, name: str) -> Path:
            return jsonl_path

        def download_and_convert(self, name: str) -> Path:
            raise AssertionError("dataset should already exist")

    class SOTAManagerStub:
        def get_metric(self, name: str) -> str:
            return "bleu"

        def calculate_rps(self, name: str, score: float) -> None:
            return None

    result = evaluate_prediction_file(
        DatasetManagerStub(),
        SOTAManagerStub(),
        dataset_name,
        pred_path,
    )

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    ref_file.write_text("utt1\t你好世界。\nutt2\t今天天气很好。\n", encoding="utf-8")
    hyp_file.write_text("utt1\t你好世界。\nutt2\t今天的天气很好。\n", encoding="utf-8")
    legacy = SUREEvaluator(language="zh").evaluate("S2TT", str(ref_file), str(hyp_file))

    assert result["task"] == "S2TT"
    assert result["metric"] == "bleu"
    assert result["score"] == legacy["score"]
    assert result["details"] == legacy


def test_evaluate_prediction_file_routes_sd_without_key_text_materialization(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import scripts.evaluate_predictions as evaluate_predictions
    from sure_eval.evaluation.core.types import EvaluationReport

    dataset_name = "toy_sd"
    jsonl_path = tmp_path / "toy_sd.jsonl"
    rows = [
        {
            "key": "utt1",
            "target": "SPEAKER rec1 1 0.00 1.00 <NA> <NA> spk1 <NA> <NA>",
            "task": "SD",
            "language": "n/a",
        }
    ]
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    pred_path = tmp_path / "pred.txt"
    pred_path.write_text(
        "utt1\tSPEAKER rec1 1 0.00 1.00 <NA> <NA> hyp1 <NA> <NA>\n",
        encoding="utf-8",
    )
    captured: dict[str, str] = {}

    def fake_run_task(task: str, **kwargs):
        captured["task"] = task
        captured["ref_text"] = Path(kwargs["ref_file"]).read_text(encoding="utf-8")
        captured["hyp_text"] = Path(kwargs["hyp_file"]).read_text(encoding="utf-8")
        return EvaluationReport(
            task="SD",
            language="n/a",
            metric="der",
            score=0.2,
            pipeline_id="sd.der.meeteval",
            pipeline_trace=(),
            details={"scoring_result": {"der": 0.2, "num_sessions": 1}},
        )

    monkeypatch.setattr(evaluate_predictions, "run_task", fake_run_task)

    class DatasetManagerStub:
        def normalize_dataset_name(self, name: str) -> str:
            return name

        def get_jsonl_path(self, name: str) -> Path:
            return jsonl_path

        def download_and_convert(self, name: str) -> Path:
            raise AssertionError("dataset should already exist")

    class SOTAManagerStub:
        def get_metric(self, name: str) -> str:
            return "der"

        def calculate_rps(self, name: str, score: float) -> None:
            return None

    result = evaluate_predictions.evaluate_prediction_file(
        DatasetManagerStub(),
        SOTAManagerStub(),
        dataset_name,
        pred_path,
    )

    assert captured["task"] == "sd"
    assert captured["ref_text"] == "SPEAKER rec1 1 0.00 1.00 <NA> <NA> spk1 <NA> <NA>\n"
    assert captured["hyp_text"] == "SPEAKER rec1 1 0.00 1.00 <NA> <NA> hyp1 <NA> <NA>\n"
    assert result["score"] == 0.2
    assert result["evaluation_context"]["task_route"] == "src/sure_eval/evaluation/tasks/sd/routes.yaml"


def test_evaluate_prediction_file_routes_sa_asr_without_key_text_materialization(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import scripts.evaluate_predictions as evaluate_predictions
    from sure_eval.evaluation.core.types import EvaluationReport

    dataset_name = "toy_sa_asr"
    jsonl_path = tmp_path / "toy_sa_asr.jsonl"
    rows = [
        {
            "key": "utt1",
            "target": "rec1 1 spk1 0.00 1.00 hello world",
            "task": "SA-ASR",
            "language": "en",
        }
    ]
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    pred_path = tmp_path / "pred.txt"
    pred_path.write_text("utt1\trec1 1 hyp1 0.00 1.00 hello there\n", encoding="utf-8")
    captured: dict[str, str] = {}

    def fake_run_task(task: str, **kwargs):
        captured["task"] = task
        captured["ref_text"] = Path(kwargs["ref_file"]).read_text(encoding="utf-8")
        captured["hyp_text"] = Path(kwargs["hyp_file"]).read_text(encoding="utf-8")
        return EvaluationReport(
            task="SA-ASR",
            language="n/a",
            metric="cpwer",
            score=0.375,
            pipeline_id="sa_asr.cpwer.gstar_norm.meeteval",
            pipeline_trace=(),
            details={"scoring_result": {"cpwer": 0.375, "der": 0.2, "num_sessions": 1}},
        )

    monkeypatch.setattr(evaluate_predictions, "run_task", fake_run_task)

    class DatasetManagerStub:
        def normalize_dataset_name(self, name: str) -> str:
            return name

        def get_jsonl_path(self, name: str) -> Path:
            return jsonl_path

        def download_and_convert(self, name: str) -> Path:
            raise AssertionError("dataset should already exist")

    class SOTAManagerStub:
        def get_metric(self, name: str) -> str:
            return "cpwer"

        def calculate_rps(self, name: str, score: float) -> None:
            return None

    result = evaluate_predictions.evaluate_prediction_file(
        DatasetManagerStub(),
        SOTAManagerStub(),
        dataset_name,
        pred_path,
    )

    assert captured["task"] == "sa-asr"
    assert captured["ref_text"] == "rec1 1 spk1 0.00 1.00 hello world\n"
    assert captured["hyp_text"] == "rec1 1 hyp1 0.00 1.00 hello there\n"
    assert result["score"] == 0.375
    assert result["evaluation_context"]["task_route"] == "src/sure_eval/evaluation/tasks/sa_asr/routes.yaml"
