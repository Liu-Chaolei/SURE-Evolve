#!/usr/bin/env python3
"""Prepare reproducible Premium/Seed and AMI splits without running models."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import wave
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from playground.sure_master.core.artifacts import file_digest
from playground.sure_master.tasks.diarization import read_rttm


def audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def grouped_split(
    rows: list[dict], proportions: dict[str, float], seed: int = 42
) -> dict[str, list[dict]]:
    groups = sorted({r["group_id"] for r in rows})
    if len(groups) < len(proportions):
        raise ValueError("Too few independent groups for the requested splits")
    random.Random(seed).shuffle(groups)
    names = list(proportions)
    if abs(sum(proportions.values()) - 1) > 1e-8:
        raise ValueError("Split proportions must sum to one")
    counts = {
        name: max(1, int(len(groups) * ratio)) for name, ratio in proportions.items()
    }
    counts[names[0]] += len(groups) - sum(counts.values())
    if counts[names[0]] < 1:
        raise ValueError("Insufficient groups for nonempty splits")
    assignment, offset = {}, 0
    for name in names:
        for group in groups[offset : offset + counts[name]]:
            assignment[group] = name
        offset += counts[name]
    return {
        name: [r for r in rows if assignment[r["group_id"]] == name] for name in names
    }


def write_jsonl(path: Path, data: list[dict]) -> None:
    if not data:
        raise ValueError(f"Refusing to create empty split: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data))


def premium_rows(root: Path, groups: Path | None) -> list[dict]:
    transcripts = sorted(root.glob("**/txts/*.txt"))
    if not transcripts:
        raise FileNotFoundError(
            f"Premium download/extraction is not ready: no txts/*.txt under {root}"
        )
    mapping = json.loads(groups.read_text()) if groups else {}
    result, seen = [], set()
    pending: list[tuple[str, str, str, Path]] = []
    for transcript in transcripts:
        fields = transcript.read_text().splitlines()[0].split("\t")
        if len(fields) < 2 or not fields[1].strip():
            raise ValueError(f"Invalid Premium transcript: {transcript}")
        sid, text = fields[0], fields[1].strip()
        if sid in seen:
            raise ValueError(f"Duplicate Premium utterance: {sid}")
        seen.add(sid)
        if groups is None:
            match = re.fullmatch(r"(.+)_S\d+(?:-S\d+)?", sid)
            if match:
                mapping[sid] = match.group(1)
        if sid not in mapping or not str(mapping[sid]).strip():
            raise ValueError(f"Missing speaker/original-recording group mapping: {sid}")
        audio = transcript.parent.parent / "wavs" / f"{sid}.wav"
        pending.append((sid, str(mapping[sid]), text, audio))
    # WAV metadata reads are latency-bound on the shared filesystem. Keep a
    # bounded pool while preserving transcript order for deterministic splits.
    with ThreadPoolExecutor(max_workers=32, thread_name_prefix="premium-wav") as pool:
        durations = list(pool.map(lambda item: audio_duration(item[3]), pending))
    for (sid, group, text, audio), duration in zip(pending, durations):
        if 1 <= duration <= 30:
            result.append(
                {
                    "sample_id": sid,
                    "group_id": "premium:" + group,
                    # Preserve the logical shared-filesystem prefix. Resolving
                    # symlinks here can produce a host-only /mnt path that is
                    # not visible inside VC child containers.
                    "audio": str(audio.absolute()),
                    "text": text,
                    "duration": duration,
                }
            )
    if not result:
        raise ValueError("No usable Premium audio")
    return result


def tts_prompts(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    output = []
    for group, items in sorted(groups.items()):
        references = sorted(
            (r for r in items if 3 <= r["duration"] <= 10), key=lambda r: r["sample_id"]
        )
        if not references:
            continue
        reference = references[0]
        for target in items:
            if target["sample_id"] == reference["sample_id"]:
                continue
            output.append(
                {
                    "sample_id": target["sample_id"],
                    "group_id": group,
                    "language": "zh",
                    "reference_audio": reference["audio"],
                    "reference_text": reference["text"],
                    "target_text": target["text"],
                }
            )
    return output


def seed_rows(root: Path, language: str) -> list[dict]:
    result = []
    for line in (root / language / "meta.lst").read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) != 4:
            raise ValueError("Seed standard meta.lst must have four columns")
        sid, text, raw, target = fields
        audio = (root / language / raw).resolve()
        if not audio.is_file():
            raise FileNotFoundError(audio)
        result.append(
            {
                "sample_id": sid,
                "group_id": f"seed:{language}:{Path(raw).stem}",
                "language": language,
                "reference_audio": str(audio.absolute()),
                "reference_text": text,
                "target_text": target,
            }
        )
    return result


def prepare_tts(
    root: Path,
    groups: Path | None,
    seed_root: Path,
    output: Path,
    *,
    source_mode: str = "archive_verified",
) -> dict:
    from playground.sure_master.tools.training_data_integrity import (
        verify_extracted_premium,
        verify_premium_source,
        write_preparation,
    )

    if source_mode == "extracted_only":
        integrity = verify_extracted_premium(root)
    elif source_mode == "archive_verified":
        integrity = verify_premium_source(root, output / "source_integrity.json")
    else:
        raise ValueError(f"Unsupported Premium source mode: {source_mode}")
    raw = premium_rows(root, groups)
    splits = grouped_split(
        raw,
        {"train": 0.90, "train_validation": 0.02, "search": 0.04, "selection": 0.04},
    )
    prepared = {}
    for name, data in splits.items():
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        if name in {"train", "train_validation"}:
            write_jsonl(directory / "manifest.jsonl", data)
            with (directory / "metadata.csv").open("w", newline="") as stream:
                writer = csv.writer(stream, delimiter="|")
                writer.writerow(["audio_file", "text"])
                writer.writerows((r["audio"], r["text"]) for r in data)
        else:
            write_jsonl(directory / "manifest.jsonl", tts_prompts(data))
        prepared[name] = {"manifest": str((directory / "manifest.jsonl").resolve())}
    write_jsonl(output / "holdout/manifest.jsonl", seed_rows(seed_root, "zh"))
    prepared["holdout"] = {
        "manifest": str((output / "holdout/manifest.jsonl").resolve())
    }
    write_preparation(output, "tts.f5tts", prepared, integrity)
    return prepared


def ami_rows(corpus: Path, recipe_data: Path, split: str) -> list[dict]:
    location = recipe_data / ("test/AMI" if split == "test" else split)
    reference = defaultdict(list)
    for row in read_rttm(location / "rttm"):
        # AMI references in the combined upstream recipe are unambiguous by recording ID.
        if re.fullmatch(r"(?:ES|IS|TS)\d{4}[a-z]", row[1]):
            reference[row[1]].append(" ".join(row))
    uems = defaultdict(list)
    for line in (location / "all.uem").read_text().splitlines():
        fields = line.split()
        if fields and fields[0] in reference:
            uems[fields[0]].append([float(fields[2]), float(fields[3])])
    result = []
    for sid, segments in sorted(reference.items()):
        audio = corpus / sid / "audio" / f"{sid}.Array1-01.wav"
        duration = audio_duration(audio)
        regions = sorted(uems[sid])
        if not regions or any(
            a < 0 or b <= a or b > duration + 0.02 for a, b in regions
        ):
            raise ValueError(f"Invalid/missing AMI UEM: {sid}")
        if any(regions[i][0] < regions[i - 1][1] for i in range(1, len(regions))):
            raise ValueError(f"Overlapping UEM regions: {sid}")
        result.append(
            {
                "session_id": sid,
                "group_id": "ami:" + sid[:-1],
                "audio": str(audio.resolve()),
                "duration": duration,
                "uem": regions,
                "reference_rttm": "\n".join(segments),
            }
        )
    if not result:
        raise ValueError(f"No AMI {split} annotations")
    return result


def prepare_sd(corpus: Path, recipe_data: Path, output: Path) -> dict:
    train, dev, test = (
        ami_rows(corpus, recipe_data, split) for split in ("train", "dev", "test")
    )
    splits = {
        **grouped_split(train, {"train": 0.95, "train_validation": 0.05}),
        **grouped_split(dev, {"search": 0.5, "selection": 0.5}),
        "holdout": test,
    }
    owners = {}
    for name, data in splits.items():
        for row in data:
            previous = owners.setdefault(row["group_id"], name)
            if previous != name:
                raise ValueError(
                    f"AMI meeting overlaps {previous}/{name}: {row['group_id']}"
                )
    prepared = {}
    for name, data in splits.items():
        directory = output / name
        write_jsonl(directory / "manifest.jsonl", data)
        (directory / "wav.scp").write_text(
            "".join(f"{r['session_id']} {r['audio']}\n" for r in data)
        )
        (directory / "ref.rttm").write_text(
            "".join(r["reference_rttm"] + "\n" for r in data)
        )
        (directory / "all.uem").write_text(
            "".join(f"{r['session_id']} 1 {a} {b}\n" for r in data for a, b in r["uem"])
        )
        # Evaluation manifests expose audio/UEM only, not reference speaker labels.
        if name not in {"train", "train_validation"}:
            write_jsonl(
                directory / "manifest.jsonl",
                [{k: v for k, v in r.items() if k != "reference_rttm"} for r in data],
            )
        prepared[name] = {
            "manifest": str((directory / "manifest.jsonl").resolve()),
            "roles": {"ref": str((directory / "ref.rttm").resolve())},
        }
    from playground.sure_master.tools.training_data_integrity import write_preparation
    write_preparation(output, "sd.diarizen", prepared, {
        "corpus": str(corpus.resolve()), "recipe_data": str(recipe_data.resolve()),
        "sessions": {name: len(data) for name, data in splits.items()}})
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["tts", "sd", "seed"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--groups",
        type=Path,
        help="JSON map of Premium utterance ID to speaker/original recording ID",
    )
    parser.add_argument(
        "--seed-root", type=Path, default=Path("/shared/chaolei.liu/data/seed-tts-eval")
    )
    parser.add_argument(
        "--source-mode",
        choices=["archive_verified", "extracted_only"],
        default="archive_verified",
        help="Premium provenance mode; extracted_only does not claim archive MD5 verification.",
    )
    parser.add_argument("--recipe-data", type=Path)
    parser.add_argument("--language", choices=["zh", "en"], default="zh")
    args = parser.parse_args()
    if args.task == "tts":
        prepared = prepare_tts(
            args.root,
            args.groups,
            args.seed_root,
            args.output,
            source_mode=args.source_mode,
        )
    elif args.task == "sd":
        if not args.recipe_data:
            parser.error("AMI requires --recipe-data from DiariZen")
        prepared = prepare_sd(args.root, args.recipe_data, args.output)
    else:
        path = args.output / "manifest.jsonl"
        write_jsonl(path, seed_rows(args.root, args.language))
        prepared = {"holdout": {"manifest": str(path.resolve())}}
    report = {
        "seed": 42,
        "datasets": prepared,
        "digests": {k: file_digest(Path(v["manifest"])) for k, v in prepared.items()},
    }
    (args.output / "splits.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
