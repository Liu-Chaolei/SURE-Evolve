from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

import pytest

from scripts.prepare_sure_dataset import prepare_dataset
from sure_eval.core.config import Config
from sure_eval.datasets.dataset_manager import DatasetManager


def _make_config(tmp_path: Path) -> Config:
    config = Config.from_env()
    config.data.datasets = str(tmp_path / "datasets")
    return config


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 160)


def _write_source_version(source_root: Path, version_id: str) -> None:
    raw_dir = source_root / "raws" / "sample"
    wav_path = raw_dir / "utt1.wav"
    if not wav_path.exists():
        _write_wav(wav_path)

    sample_dir = source_root / "sample_files" / version_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "ds.jsonl").write_text(
        json.dumps(
            {
                "supported_tasks": ["ASR"],
                "type": ["audio"],
                "audio": {"speech": {"language": "en"}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (sample_dir / "sample.jsonl").write_text(
        json.dumps(
            {
                "annotation": [{"transcription": {"language": "en", "text": ["hello world"]}}],
                "attribute": {
                    "path": "utt1.wav",
                    "size": wav_path.stat().st_size,
                    "sample_rate": 16000,
                    "duration": 10,
                    "channels": 1,
                    "raw_data_format": "wav",
                    "raw_data_md5": "not_checked",
                },
                "sample_id": "utt1",
                "parent_sample_id": "utt1_parent",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _make_aispeech_source(
    tmp_path: Path,
    *,
    dataset_name: str = "aispeech_phy_demo-test",
    version_id: str = "v1.2.3",
) -> Path:
    source_root = tmp_path / "external_ds" / "aispeech" / "g001" / "store002" / "ds_pool" / dataset_name
    _write_source_version(source_root, version_id)
    return source_root


def test_aispeech_source_root_drives_prepared_dataset_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    allowed_root = tmp_path / "external_ds" / "aispeech"
    monkeypatch.setenv("SURE_EVAL_AISPEECH_SOURCE_ROOTS", str(allowed_root))
    source_root = _make_aispeech_source(tmp_path)
    manager = DatasetManager(_make_config(tmp_path))

    source_ref = manager.resolve_aispeech_source_entry(str(source_root))
    jsonl_path = manager.download_and_convert(str(source_root))

    assert source_ref.source_dataset_name == "aispeech_phy_demo-test"
    assert source_ref.version_id == "v1.2.3"
    assert source_ref.dataset_id == "aispeech_phy_demo-test__v1.2.3"
    assert jsonl_path.name == "aispeech_phy_demo-test__v1.2.3.jsonl"

    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["dataset"] == "aispeech_phy_demo-test__v1.2.3"
    assert rows[0]["metadata"]["source"] == "aispeech_ds_pool"
    assert rows[0]["metadata"]["source_dataset_root"] == str(source_root.resolve())
    assert rows[0]["metadata"]["source_dataset_name"] == "aispeech_phy_demo-test"
    assert rows[0]["metadata"]["version_id"] == "v1.2.3"

    info = manager.get_info("aispeech_phy_demo-test__v1.2.3")
    assert info is not None
    assert info["source"] == "aispeech_ds_pool"
    assert info["task"] == "ASR"
    assert info["language"] == "en"
    assert info["num_samples"] == 1
    assert info["naming_policy"] == "source_dataset_name__version_id"

    source_info = manager.get_source_info("aispeech_phy_demo-test__v1.2.3")
    assert source_info["source_dataset_name"] == "aispeech_phy_demo-test"
    assert source_info["version_id"] == "v1.2.3"


def test_prepare_dataset_rejects_legacy_name_by_default(tmp_path: Path) -> None:
    manager = DatasetManager(_make_config(tmp_path))

    with pytest.raises(ValueError, match="AiSpeech source root"):
        prepare_dataset(manager, "aishell1")


def test_prepare_dataset_summary_uses_source_name_and_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    allowed_root = tmp_path / "external_ds" / "aispeech"
    monkeypatch.setenv("SURE_EVAL_AISPEECH_SOURCE_ROOTS", str(allowed_root))
    source_root = _make_aispeech_source(tmp_path)
    manager = DatasetManager(_make_config(tmp_path))

    summary = prepare_dataset(manager, str(source_root))

    assert summary["dataset"] == "aispeech_phy_demo-test__v1.2.3"
    assert summary["dataset_id"] == "aispeech_phy_demo-test__v1.2.3"
    assert summary["source_dataset_root"] == str(source_root.resolve())
    assert summary["source_dataset_name"] == "aispeech_phy_demo-test"
    assert summary["version_id"] == "v1.2.3"
    assert summary["naming_policy"] == "source_dataset_name__version_id"


def test_aispeech_source_root_selects_latest_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    allowed_root = tmp_path / "external_ds" / "aispeech"
    monkeypatch.setenv("SURE_EVAL_AISPEECH_SOURCE_ROOTS", str(allowed_root))
    source_root = _make_aispeech_source(tmp_path)
    _write_source_version(source_root, "v2.0.0")
    manager = DatasetManager(_make_config(tmp_path))

    source_ref = manager.resolve_aispeech_source_entry(str(source_root))

    assert source_ref.version_id == "v2.0.0"
    assert source_ref.dataset_id == "aispeech_phy_demo-test__v2.0.0"


def test_audio_only_source_uses_explicit_annotation_overlay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    allowed_root = tmp_path / "external_ds" / "aispeech"
    monkeypatch.setenv("SURE_EVAL_AISPEECH_SOURCE_ROOTS", str(allowed_root))
    annotation_root = _make_aispeech_source(
        tmp_path,
        dataset_name="aispeech_phy_librispeech_test-clean",
        version_id="v1.0.1",
    )
    source_root = annotation_root.parent / "librispeech_test-clean"
    _write_wav(source_root / "raws" / "sample" / "utt1.wav")
    manager = DatasetManager(_make_config(tmp_path))

    with pytest.raises(FileNotFoundError, match="sample_files"):
        manager.resolve_aispeech_source_entry(str(source_root))

    source_ref = manager.resolve_aispeech_source_entry(
        str(source_root),
        annotation_source_root=str(annotation_root),
    )
    jsonl_path = manager.convert_aispeech_source_to_jsonl(source_ref)

    assert source_ref.dataset_id == "librispeech_test-clean__v1.0.1"
    assert source_ref.source_dataset_name == "librispeech_test-clean"
    assert source_ref.annotation_source_root == annotation_root.resolve()
    row = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["dataset"] == "librispeech_test-clean__v1.0.1"
    assert row["path"] == str((source_root / "raws" / "sample" / "utt1.wav").resolve())
    assert row["metadata"]["source_dataset_root"] == str(source_root.resolve())
    assert row["metadata"]["annotation_source_root"] == str(annotation_root.resolve())


def test_annotation_overlay_rejects_audio_basename_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    allowed_root = tmp_path / "external_ds" / "aispeech"
    monkeypatch.setenv("SURE_EVAL_AISPEECH_SOURCE_ROOTS", str(allowed_root))
    annotation_root = _make_aispeech_source(
        tmp_path,
        dataset_name="aispeech_phy_librispeech_test-clean",
        version_id="v1.0.1",
    )
    source_root = annotation_root.parent / "librispeech_test-clean"
    _write_wav(source_root / "raws" / "sample" / "utt1.wav")
    _write_wav(source_root / "raws" / "sample" / "extra.wav")
    manager = DatasetManager(_make_config(tmp_path))

    with pytest.raises(ValueError, match="audio basename sets differ"):
        manager.resolve_aispeech_source_entry(
            str(source_root),
            annotation_source_root=str(annotation_root),
        )


def test_prepare_summary_records_explicit_annotation_overlay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    allowed_root = tmp_path / "external_ds" / "aispeech"
    monkeypatch.setenv("SURE_EVAL_AISPEECH_SOURCE_ROOTS", str(allowed_root))
    annotation_root = _make_aispeech_source(
        tmp_path,
        dataset_name="aispeech_phy_librispeech_test-clean",
        version_id="v1.0.1",
    )
    source_root = annotation_root.parent / "librispeech_test-clean"
    _write_wav(source_root / "raws" / "sample" / "utt1.wav")
    manager = DatasetManager(_make_config(tmp_path))

    summary = prepare_dataset(
        manager,
        str(source_root),
        annotation_source_root=str(annotation_root),
    )

    assert summary["dataset"] == "librispeech_test-clean__v1.0.1"
    assert summary["source_dataset_root"] == str(source_root.resolve())
    assert summary["annotation_source_root"] == str(annotation_root.resolve())


def test_report_payload_preserves_aispeech_source_identity(tmp_path: Path) -> None:
    from scripts import evaluate_predictions

    dataset_id = "aispeech_phy_demo-test__v1.2.3"
    result: dict[str, Any] = {
        "dataset": dataset_id,
        "dataset_source": {
            "source": "aispeech_ds_pool",
            "dataset_id": dataset_id,
            "source_dataset_root": "/hpc_stor08/external_ds/aispeech/g001/store002/ds_pool/aispeech_phy_demo-test",
            "source_dataset_name": "aispeech_phy_demo-test",
            "version_id": "v1.2.3",
            "naming_policy": "source_dataset_name__version_id",
        },
        "jsonl_path": str(tmp_path / f"{dataset_id}.jsonl"),
        "prediction_path": str(tmp_path / "predictions" / f"{dataset_id}.txt"),
        "task": "ASR",
        "language": "en",
        "metric": "wer",
        "baseline_dataset": dataset_id,
        "score": 0.0,
        "rps": {"status": "missing_baseline"},
        "num_samples": 1,
        "evaluation_context": {},
        "pipeline_id": "asr.en.wer",
        "metric_artifact_dir": str(tmp_path / "metrics" / dataset_id / "wer"),
        "metric_report_path": str(tmp_path / "metrics" / dataset_id / "wer" / "report.json"),
        "pipeline_description_path": str(tmp_path / "metrics" / dataset_id / "wer" / "pipeline_description.json"),
        "pipeline_description": {"nodes": []},
        "details": {"score": 0.0, "wer": 0.0},
    }

    class SOTAManagerStub:
        def get_baseline(self, _name: str) -> None:
            return None

    payload = evaluate_predictions._build_evaluation_payload([result])
    report_line = evaluate_predictions._report_jsonl_line(
        result=result,
        tool_name="transcribe_audio",
        protocol_id="strict_core",
        run_dir=tmp_path / "run",
        model_dir=tmp_path / "model",
        sota_manager=SOTAManagerStub(),
        validation_payload={"results": []},
    )

    payload_row = payload["results"][0]
    assert payload_row["dataset"] == dataset_id
    assert payload_row["dataset_source"]["source_dataset_name"] == "aispeech_phy_demo-test"
    assert payload_row["dataset_source"]["version_id"] == "v1.2.3"
    assert report_line["dataset"]["name"] == dataset_id
    assert report_line["dataset"]["source_dataset_name"] == "aispeech_phy_demo-test"
    assert report_line["dataset"]["version_id"] == "v1.2.3"
    assert report_line["dataset"]["naming_policy"] == "source_dataset_name__version_id"
