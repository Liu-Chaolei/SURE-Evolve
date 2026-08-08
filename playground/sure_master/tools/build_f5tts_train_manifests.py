#!/usr/bin/env python3
"""Build whitelisted F5-TTS fine-tuning manifests for SURE Master.

The output is intentionally simple: each manifest directory contains a
metadata.csv with the format expected by F5-TTS prepare_csv_wavs.py:

    audio_file|text
    /abs/path/to/audio.wav|transcript

Only LibriTTS train-clean-100 is used here so training data stays separate from
the staged dev/selection/holdout evaluation prompts.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LIBRITTS_ROOT = Path("/hpc_stor03/sjtu_home/chaolei.liu/data/tts/LibriTTS")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "playground/sure_master/data/f5tts_train_manifests"
DEFAULT_SEED = 20260714

MANIFEST_TARGET_HOURS = {
    "libritts_train_clean_100_1h": 1.0,
    "libritts_train_clean_100_5h": 5.0,
    "libritts_train_clean_100_10h": 10.0,
}


@dataclass(frozen=True)
class TrainRow:
    split: str
    speaker: str
    utt_id: str
    audio_path: Path
    text: str
    duration: float | None = None


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def is_short_heading(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    if len(letters) < 3:
        return True
    return len(text) < 80 and "".join(letters).isupper()


def has_text_noise(text: str) -> bool:
    lower = text.lower()
    return (
        "illustration:" in lower
        or lower.startswith("chapter ")
        or lower.startswith("book ")
        or (text[:1].isdigit() and word_count(text) <= 6)
        or (text.startswith("[") and text.endswith("]"))
    )


def usable_train_text(text: str) -> bool:
    return 20 <= len(text) <= 280 and word_count(text) >= 5 and not is_short_heading(text) and not has_text_noise(text)


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        frame_rate = wav_file.getframerate()
        if frame_rate <= 0:
            raise ValueError(f"Invalid frame rate for {path}")
        return wav_file.getnframes() / float(frame_rate)


def read_librispeech_tts(root: Path, split: str) -> list[TrainRow]:
    split_root = root / split
    if not split_root.exists():
        raise FileNotFoundError(f"LibriTTS split does not exist: {split_root}")

    rows: list[TrainRow] = []
    for trans_path in sorted(split_root.rglob("*.trans.tsv")):
        for line_no, line in enumerate(trans_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) < 2:
                raise ValueError(f"Bad LibriTTS transcription at {trans_path}:{line_no}")
            utt_id = fields[0].strip()
            text = clean_text(fields[1])
            audio_path = trans_path.parent / f"{utt_id}.wav"
            if not utt_id or not text or not audio_path.exists() or not usable_train_text(text):
                continue
            speaker = utt_id.split("_", 1)[0]
            rows.append(
                TrainRow(
                    split=split,
                    speaker=speaker,
                    utt_id=utt_id,
                    audio_path=audio_path,
                    text=text,
                )
            )
    if not rows:
        raise RuntimeError(f"No usable LibriTTS rows found in {split_root}")
    return rows


def balanced_order(rows: Iterable[TrainRow], seed: int) -> list[TrainRow]:
    per_speaker: dict[str, list[TrainRow]] = defaultdict(list)
    for row in rows:
        per_speaker[row.speaker].append(row)

    rng = random.Random(seed)
    speakers = sorted(per_speaker)
    rng.shuffle(speakers)
    for speaker in speakers:
        per_speaker[speaker].sort(key=lambda row: row.utt_id)
        rng.shuffle(per_speaker[speaker])

    ordered: list[TrainRow] = []
    while True:
        progressed = False
        for speaker in speakers:
            bucket = per_speaker[speaker]
            if not bucket:
                continue
            ordered.append(bucket.pop())
            progressed = True
        if not progressed:
            break
    return ordered


def select_until_hours(ordered_rows: list[TrainRow], target_hours: float) -> list[TrainRow]:
    target_seconds = target_hours * 3600.0
    selected: list[TrainRow] = []
    total = 0.0
    for row in ordered_rows:
        try:
            duration = wav_duration(row.audio_path)
        except Exception:
            continue
        if not 1.0 <= duration <= 15.0:
            continue
        selected.append(replace(row, duration=duration))
        total += duration
        if total >= target_seconds:
            break
    if total < target_seconds:
        raise RuntimeError(f"Only found {total / 3600.0:.2f} hours, need {target_hours:.2f} hours")
    return selected


def write_manifest(output_dir: Path, name: str, rows: list[TrainRow], source_root: Path, seed: int) -> dict[str, object]:
    manifest_dir = output_dir / name
    manifest_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = manifest_dir / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|")
        writer.writerow(["audio_file", "text"])
        for row in rows:
            writer.writerow([str(row.audio_path), row.text])

    summary = {
        "name": name,
        "metadata_csv": str(metadata_path),
        "source": "libritts",
        "source_root": str(source_root),
        "splits": dict(Counter(row.split for row in rows)),
        "sample_count": len(rows),
        "speaker_count": len({row.speaker for row in rows}),
        "duration_hours": round(sum(row.duration or 0.0 for row in rows) / 3600.0, 4),
        "min_duration": round(min(row.duration or 0.0 for row in rows), 4),
        "max_duration": round(max(row.duration or 0.0 for row in rows), 4),
        "seed": seed,
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libritts-root", type=Path, default=DEFAULT_LIBRITTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_librispeech_tts(args.libritts_root, "train-clean-100")
    ordered = balanced_order(rows, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for name, hours in MANIFEST_TARGET_HOURS.items():
        selected = select_until_hours(ordered, hours)
        summaries.append(write_manifest(args.output_dir, name, selected, args.libritts_root, args.seed))

    root_summary = {
        "description": "Whitelisted F5-TTS fine-tuning manifests for SURE Master training actions.",
        "manifests": summaries,
        "available_source_rows": len(rows),
    }
    summary_path = args.output_dir / "manifest_summary.json"
    summary_path.write_text(json.dumps(root_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote F5-TTS train manifests under {args.output_dir}")
    for summary in summaries:
        print(f"- {summary['name']}: {summary['sample_count']} rows, {summary['duration_hours']} hours")
    print(f"Manifest summary: {summary_path}")


if __name__ == "__main__":
    main()
