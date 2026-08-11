from __future__ import annotations

import json
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from playground.sure_master.tools import run_f5tts_batch_infer as batch


def write_valid_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 4000)


def test_missing_reference_audio_remains_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"target_text": "hello", "language": "en"}) + "\n")
    monkeypatch.setattr(batch, "WAV_DIR", tmp_path / "wavs")

    rows = batch.load_prompts(prompts, 1, "en", "none")

    assert rows[0]["reference_audio"] == ""


@pytest.mark.parametrize(
    ("reference_audio", "reference_text", "expected"),
    [
        ("", "reference", "reference_audio is empty"),
        ("missing.wav", "reference", "invalid reference_audio"),
        ("valid.wav", "", "reference_text is empty"),
    ],
)
def test_prompt_reference_preflight_rejects_invalid_conditioning(
    tmp_path: Path,
    reference_audio: str,
    reference_text: str,
    expected: str,
) -> None:
    valid_wav = tmp_path / "valid.wav"
    write_valid_wav(valid_wav)
    resolved_audio = str(tmp_path / reference_audio) if reference_audio else ""
    rows = [
        {
            "sample_id": "sample",
            "reference_audio": resolved_audio,
            "reference_text": reference_text,
        }
    ]

    with pytest.raises(RuntimeError, match=expected):
        batch.validate_prompt_references(rows, tmp_path / "prompts.jsonl")


def test_prompt_reference_preflight_accepts_valid_conditioning(tmp_path: Path) -> None:
    reference_audio = tmp_path / "reference.wav"
    write_valid_wav(reference_audio)

    batch.validate_prompt_references(
        [
            {
                "sample_id": "sample",
                "reference_audio": str(reference_audio),
                "reference_text": "Some call me nature.",
            }
        ],
        tmp_path / "prompts.jsonl",
    )


def test_run_main_fails_reference_preflight_before_worker_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "language": "en",
                "target_text": "hello",
                "reference_audio": "missing.wav",
                "reference_text": "reference",
            }
        )
        + "\n"
    )
    monkeypatch.setattr(batch, "WORKSPACE", tmp_path)
    monkeypatch.setattr(batch, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(batch, "WAV_DIR", tmp_path / "artifacts" / "wavs")
    monkeypatch.setattr(batch, "WORKING", tmp_path / "working")
    monkeypatch.setattr(batch, "BATCH_DIR", tmp_path / "working" / "f5tts_batch_infer")
    monkeypatch.setattr(batch, "LOG_DIR", tmp_path / "working" / "logs")
    args = batch.build_arg_parser().parse_args(
        ["--f5-root", str(tmp_path), "--eval-data", str(prompts), "--device", "cpu"]
    )

    with patch.object(batch, "launch_workers") as launch_workers, patch.object(
        batch, "load_f5_runtime"
    ) as load_runtime:
        with pytest.raises(RuntimeError, match="prompt reference preflight failed"):
            batch.run_main(args)

    launch_workers.assert_not_called()
    load_runtime.assert_not_called()
    assert not (tmp_path / "artifacts" / "samples.jsonl").exists()


def test_resume_requires_matching_synthesis_fingerprint(tmp_path: Path) -> None:
    wav = tmp_path / "sample.wav"
    write_valid_wav(wav)
    row = {"output_wav": str(wav), "gen_text": "hello", "reference_audio": "ref.wav", "reference_text": "ref"}
    config = {"model": "F5", "ckpt_file": "model.safetensors", "nfe_step": 32}

    assert not batch.resume_matches(row, config)
    batch.fingerprint_path(wav).write_text(batch.synthesis_fingerprint(row, config) + "\n")
    assert batch.resume_matches(row, config)
    assert not batch.resume_matches({**row, "gen_text": "changed"}, config)
    assert not batch.resume_matches(row, {**config, "nfe_step": 16})


def test_cuda_worker_pinning_is_case_insensitive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(batch, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(batch, "WORKSPACE", tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7")
    proc = MagicMock()
    proc.wait.return_value = 0
    proc.poll.return_value = 0

    with patch.object(batch.subprocess, "Popen", return_value=proc) as popen:
        batch.launch_workers(
            config_path=tmp_path / "config.json",
            shard_paths=[tmp_path / "shard.jsonl"],
            timeout=10,
            device="CUDA:0",
        )

    assert popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "4"


def test_split_text_chunks_merges_undersized_tail() -> None:
    text = "one two three four five six seven"
    chunks = batch.split_text_chunks(text, max_chars=20, min_chars=10)

    assert " ".join(chunks) == text
    assert len(chunks[-1]) >= 10


@pytest.mark.parametrize(
    ("shape", "expected"),
    [((16,), (16,)), ((1, 16), (16,)), ((16, 1), (16,))],
)
def test_normalize_audio_segment_accepts_mono_shapes(shape: tuple[int, ...], expected: tuple[int, ...]) -> None:
    assert batch.normalize_audio_segment(np.zeros(shape)).shape == expected


def test_normalize_audio_segment_rejects_unsupported_shape() -> None:
    with pytest.raises(ValueError, match="Unsupported F5-TTS waveform shape"):
        batch.normalize_audio_segment(np.zeros((2, 16)))


def test_write_jsonl_atomically_replaces_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(batch, "WORKSPACE", tmp_path)
    destination = tmp_path / "samples.jsonl"
    destination.write_text("old\n")

    batch.write_jsonl(destination, [{"sample_id": "new"}], atomic=True)

    assert json.loads(destination.read_text()) == {"sample_id": "new"}
    assert list(tmp_path.glob(".*.tmp")) == []
