from __future__ import annotations

import json
import wave
from pathlib import Path

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


def test_oref_platform_conversion_writes_projection_package(tmp_path: Path) -> None:
    manager = DatasetManager(_make_config(tmp_path))

    platform_root = tmp_path / "platform" / "demo_oref"
    raw_dir = platform_root / "raws" / "sample"
    wav_path = raw_dir / "utt1.wav"
    _write_wav(wav_path)

    sample_dir = platform_root / "sample_files" / "v1.2.3"
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
                    "path": str(wav_path),
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

    manager.oref_local_datasets["demo_platform"] = {
        "source": "oref_platform",
        "config_name": "demo_sure",
        "platform_dataset_name": "demo_oref",
        "version_id": "v1.2.3",
        "dataset_root": str(platform_root),
        "task": "ASR",
        "language": "en",
        "validation": {"require_audio_exists": True, "check_size": True},
    }

    jsonl_path = manager.download_and_convert("demo_platform")

    assert jsonl_path.name == "demo_sure__v1.2.3__asr.jsonl"
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["key"] == "utt1"
    assert rows[0]["path"] == str(wav_path)
    assert rows[0]["target"] == "hello world"
    assert rows[0]["dataset"] == "demo_sure__v1.2.3__asr"
    assert rows[0]["metadata"]["oref_dataset"] == "demo_oref"

    package_dir = manager.sure_dir / "demo_sure"
    report = json.loads(
        (package_dir / "projections" / "asr_transcription_v1" / "conversion_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["num_input_records"] == 1
    assert report["num_output_records"] == 1
    assert report["output_jsonl"] == str(jsonl_path)
    assert (package_dir / "oref" / "sample.jsonl").exists()
    assert manager.normalize_dataset_name("demo_sure") == "demo_sure__v1.2.3__asr"
