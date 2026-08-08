#!/usr/bin/env python3
"""Build staged F5-TTS prompt sets for SURE Master evaluation.

The generated prompt files follow the existing F5-TTS SURE task contract:

    {"sample_id", "language", "reference_audio", "reference_text", "target_text"}

This script intentionally prepares evaluation/search data only. It does not
change SURE Master's control flow and it does not create a training curriculum.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LIBRITTS_ROOT = Path("/hpc_stor03/sjtu_home/chaolei.liu/data/tts/LibriTTS")
DEFAULT_VCTK_ROOT = Path("/hpc_stor03/sjtu_home/chaolei.liu/data/tts/VCTK-Corpus")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "playground/sure_master/data/f5tts_staged"
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs/sure_master/gpt-5-f5tts-docker.yaml"
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs/sure_master"
DEFAULT_RUNTIME_REPO_ROOT = Path("/hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster")

STAGE_LIMITS = {
    "smoke": 20,
    "early_search": 200,
    "regular_search": 800,
    "selection": 500,
    "holdout": 1000,
}

CONFIG_STAGE_FILES = {
    "smoke": "gpt-5-f5tts-smoke.yaml",
    "early_search": "gpt-5-f5tts-early-search.yaml",
    "regular_search": "gpt-5-f5tts-regular-search.yaml",
    "selection": "gpt-5-f5tts-selection.yaml",
}


@dataclass(frozen=True)
class PromptSourceRow:
    source: str
    split: str
    speaker: str
    utt_id: str
    audio_path: Path
    text: str

    @property
    def target_key(self) -> str:
        return f"{self.source}:{self.split}:{self.utt_id}"

    @property
    def group_key(self) -> tuple[str, str, str]:
        return (self.source, self.split, self.speaker)


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


def usable_target_text(text: str) -> bool:
    return 20 <= len(text) <= 280 and word_count(text) >= 5 and not is_short_heading(text) and not has_text_noise(text)


def usable_reference_text(text: str) -> bool:
    return 12 <= len(text) <= 280 and word_count(text) >= 5 and not is_short_heading(text) and not has_text_noise(text)


def safe_sample_id(row: PromptSourceRow) -> str:
    raw = f"{row.source}_{row.split}_{row.utt_id}"
    return re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")


def read_librispeech_tts(root: Path, splits: Iterable[str]) -> list[PromptSourceRow]:
    rows: list[PromptSourceRow] = []
    for split in splits:
        split_root = root / split
        if not split_root.exists():
            raise FileNotFoundError(f"LibriTTS split does not exist: {split_root}")

        for trans_path in sorted(split_root.rglob("*.trans.tsv")):
            for line_no, line in enumerate(trans_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split("\t")
                if len(fields) < 2:
                    raise ValueError(f"Bad LibriTTS transcription at {trans_path}:{line_no}")

                utt_id = fields[0].strip()
                text = clean_text(fields[1] if len(fields) > 1 else fields[-1])
                audio_path = trans_path.parent / f"{utt_id}.wav"
                if not utt_id or not text or not audio_path.exists():
                    continue

                speaker = utt_id.split("_", 1)[0]
                rows.append(
                    PromptSourceRow(
                        source="libritts",
                        split=split,
                        speaker=speaker,
                        utt_id=utt_id,
                        audio_path=audio_path,
                        text=text,
                    )
                )
    return rows


def read_vctk(root: Path) -> list[PromptSourceRow]:
    txt_root = root / "txt"
    wav_root = root / "wav"
    if not txt_root.exists() or not wav_root.exists():
        raise FileNotFoundError(f"VCTK txt/wav roots are incomplete under: {root}")

    rows: list[PromptSourceRow] = []
    for txt_path in sorted(txt_root.glob("p*/*.txt")):
        speaker = txt_path.parent.name
        utt_id = txt_path.stem
        audio_path = wav_root / speaker / f"{utt_id}.wav"
        if not audio_path.exists():
            continue

        text = clean_text(txt_path.read_text(encoding="utf-8", errors="replace"))
        if not text:
            continue

        rows.append(
            PromptSourceRow(
                source="vctk",
                split="heldout_speakers",
                speaker=speaker,
                utt_id=utt_id,
                audio_path=audio_path,
                text=text,
            )
        )
    return rows


def build_group_map(rows: Iterable[PromptSourceRow]) -> dict[tuple[str, str, str], list[PromptSourceRow]]:
    grouped: dict[tuple[str, str, str], list[PromptSourceRow]] = defaultdict(list)
    for row in rows:
        grouped[row.group_key].append(row)
    for key, values in grouped.items():
        grouped[key] = sorted(values, key=lambda item: item.utt_id)
    return grouped


def has_reference(row: PromptSourceRow, grouped: dict[tuple[str, str, str], list[PromptSourceRow]]) -> bool:
    return any(
        candidate.utt_id != row.utt_id and usable_reference_text(candidate.text)
        for candidate in grouped.get(row.group_key, [])
    )


def reference_for(
    row: PromptSourceRow,
    grouped: dict[tuple[str, str, str], list[PromptSourceRow]],
) -> PromptSourceRow:
    candidates = grouped[row.group_key]
    for candidate in candidates:
        if candidate.utt_id != row.utt_id and usable_reference_text(candidate.text):
            return candidate
    raise ValueError(f"No reference utterance found for {row.target_key}")


def select_balanced(
    rows: Iterable[PromptSourceRow],
    count: int,
    used_target_keys: set[str],
    grouped: dict[tuple[str, str, str], list[PromptSourceRow]],
    seed: int,
    label: str,
) -> list[PromptSourceRow]:
    per_speaker: dict[str, list[PromptSourceRow]] = defaultdict(list)
    for row in rows:
        if row.target_key in used_target_keys:
            continue
        if not usable_target_text(row.text):
            continue
        if not has_reference(row, grouped):
            continue
        per_speaker[row.speaker].append(row)

    rng = random.Random(f"{seed}:{label}")
    speakers = sorted(per_speaker)
    rng.shuffle(speakers)
    for speaker in speakers:
        per_speaker[speaker].sort(key=lambda item: (item.split, item.utt_id))
        rng.shuffle(per_speaker[speaker])

    selected: list[PromptSourceRow] = []
    while len(selected) < count:
        progressed = False
        for speaker in speakers:
            bucket = per_speaker[speaker]
            while bucket and bucket[-1].target_key in used_target_keys:
                bucket.pop()
            if not bucket:
                continue

            row = bucket.pop()
            used_target_keys.add(row.target_key)
            selected.append(row)
            progressed = True
            if len(selected) == count:
                break

        if not progressed:
            break

    if len(selected) != count:
        raise RuntimeError(f"Could only select {len(selected)} of {count} rows for {label}")
    return selected


def prompt_record(row: PromptSourceRow, grouped: dict[tuple[str, str, str], list[PromptSourceRow]]) -> dict[str, str]:
    reference = reference_for(row, grouped)
    return {
        "sample_id": safe_sample_id(row),
        "language": "en",
        "reference_audio": str(reference.audio_path),
        "reference_text": reference.text,
        "target_text": row.text,
    }


def validate_stage(
    stage: str,
    rows: list[PromptSourceRow],
    grouped: dict[tuple[str, str, str], list[PromptSourceRow]],
) -> None:
    sample_ids: set[str] = set()
    for row in rows:
        if not row.audio_path.exists():
            raise FileNotFoundError(f"{stage}: missing target audio {row.audio_path}")
        reference = reference_for(row, grouped)
        if not reference.audio_path.exists():
            raise FileNotFoundError(f"{stage}: missing reference audio {reference.audio_path}")
        if reference.utt_id == row.utt_id:
            raise ValueError(f"{stage}: reference equals target for {row.target_key}")
        if not row.text or not reference.text:
            raise ValueError(f"{stage}: empty text for {row.target_key}")

        sample_id = safe_sample_id(row)
        if sample_id in sample_ids:
            raise ValueError(f"{stage}: duplicate sample_id {sample_id}")
        sample_ids.add(sample_id)


def write_stage(
    output_dir: Path,
    stage: str,
    rows: list[PromptSourceRow],
    grouped: dict[tuple[str, str, str], list[PromptSourceRow]],
) -> dict[str, object]:
    validate_stage(stage, rows, grouped)
    stage_dir = output_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = stage_dir / "prompts.jsonl"
    with prompts_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(prompt_record(row, grouped), ensure_ascii=False) + "\n")

    return {
        "stage": stage,
        "count": len(rows),
        "prompts_jsonl": str(prompts_path),
        "source_counts": dict(Counter(row.source for row in rows)),
        "split_counts": dict(Counter(f"{row.source}:{row.split}" for row in rows)),
    }


def build_stages(libritts_root: Path, vctk_root: Path, seed: int) -> tuple[dict[str, list[PromptSourceRow]], dict]:
    libritts_dev_clean = read_librispeech_tts(libritts_root, ["dev-clean"])
    libritts_dev_other = read_librispeech_tts(libritts_root, ["dev-other"])
    libritts_test = read_librispeech_tts(libritts_root, ["test-clean", "test-other"])
    vctk_rows = read_vctk(vctk_root)

    all_rows = libritts_dev_clean + libritts_dev_other + libritts_test + vctk_rows
    grouped = build_group_map(all_rows)
    used: set[str] = set()

    dev_rows = libritts_dev_clean + libritts_dev_other
    stages = {
        "smoke": select_balanced(libritts_dev_clean, STAGE_LIMITS["smoke"], used, grouped, seed, "smoke"),
        "early_search": select_balanced(dev_rows, STAGE_LIMITS["early_search"], used, grouped, seed, "early_search"),
        "regular_search": select_balanced(
            dev_rows,
            STAGE_LIMITS["regular_search"],
            used,
            grouped,
            seed,
            "regular_search",
        ),
    }

    selection_vctk_count = STAGE_LIMITS["selection"] // 2
    selection_libritts_count = STAGE_LIMITS["selection"] - selection_vctk_count
    stages["selection"] = (
        select_balanced(vctk_rows, selection_vctk_count, used, grouped, seed, "selection_vctk")
        + select_balanced(dev_rows, selection_libritts_count, used, grouped, seed, "selection_libritts")
    )
    stages["holdout"] = select_balanced(
        libritts_test,
        STAGE_LIMITS["holdout"],
        used,
        grouped,
        seed,
        "holdout",
    )

    inventory = {
        "libritts_root": str(libritts_root),
        "vctk_root": str(vctk_root),
        "seed": seed,
        "available_counts": {
            "libritts:dev-clean": len(libritts_dev_clean),
            "libritts:dev-other": len(libritts_dev_other),
            "libritts:test-clean+test-other": len(libritts_test),
            "vctk:heldout_speakers": len(vctk_rows),
        },
    }
    return stages, {"grouped": grouped, "inventory": inventory}


def runtime_output_dir(output_dir: Path, runtime_repo_root: Path) -> Path:
    try:
        relative = output_dir.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return output_dir
    return runtime_repo_root / relative


def write_configs(output_dir: Path, base_config: Path, config_dir: Path, runtime_repo_root: Path) -> list[str]:
    text = base_config.read_text(encoding="utf-8")
    eval_data_pattern = re.compile(
        r'^(\s*eval_data:\s*)"/hpc_stor03/sjtu_home/chaolei\.liu/Agent/EvoMaster/'
        r'playground/sure_master/data/f5tts_en_eval"$',
        re.MULTILINE,
    )
    max_samples_pattern = re.compile(r'^(\s*SURE_TTS_MAX_SAMPLES:\s*)".*"$', re.MULTILINE)
    if not eval_data_pattern.search(text):
        raise ValueError(f"Could not find expected eval_data line in {base_config}")
    if not max_samples_pattern.search(text):
        raise ValueError(f"Could not find expected SURE_TTS_MAX_SAMPLES line in {base_config}")

    written: list[str] = []
    config_eval_root = runtime_output_dir(output_dir, runtime_repo_root)
    for stage, filename in CONFIG_STAGE_FILES.items():
        stage_text = text
        stage_text = stage_text.replace(
            "# SURE Master Docker configuration for F5-TTS.",
            f"# SURE Master Docker configuration for F5-TTS ({stage} stage).",
            1,
        )
        stage_text = stage_text.replace(
            "# /hpc_stor03/sjtu_home/chaolei.liu:/hpc_stor03/sjtu_home/chaolei.liu\n",
            "# /hpc_stor03/sjtu_home/chaolei.liu:/hpc_stor03/sjtu_home/chaolei.liu\n"
            "# /hpc_stor03/public/shared/data:/hpc_stor03/public/shared/data\n",
            1,
        )
        stage_text = max_samples_pattern.sub(
            lambda match: f'{match.group(1)}"{STAGE_LIMITS[stage]}"',
            stage_text,
            count=1,
        )
        stage_text = eval_data_pattern.sub(
            lambda match: f'{match.group(1)}"{(config_eval_root / stage).as_posix()}"',
            stage_text,
            count=1,
        )
        config_path = config_dir / filename
        config_path.write_text(stage_text, encoding="utf-8")
        written.append(str(config_path))
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libritts-root", type=Path, default=DEFAULT_LIBRITTS_ROOT)
    parser.add_argument("--vctk-root", type=Path, default=DEFAULT_VCTK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--runtime-repo-root", type=Path, default=DEFAULT_RUNTIME_REPO_ROOT)
    parser.add_argument("--skip-configs", action="store_true", help="Only write staged prompt files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages, context = build_stages(args.libritts_root, args.vctk_root, args.seed)
    grouped = context["grouped"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_summaries = [
        write_stage(args.output_dir, stage, rows, grouped)
        for stage, rows in stages.items()
    ]
    config_paths = (
        []
        if args.skip_configs
        else write_configs(args.output_dir, args.base_config, args.config_dir, args.runtime_repo_root)
    )

    manifest = {
        "description": "Staged F5-TTS prompt sets for the existing SURE Master self-optimization flow.",
        "stages": stage_summaries,
        "config_paths": config_paths,
        **context["inventory"],
    }
    manifest_path = args.output_dir / "manifest_summary.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote staged data under {args.output_dir}")
    for summary in stage_summaries:
        print(f"- {summary['stage']}: {summary['count']} rows -> {summary['prompts_jsonl']}")
    if config_paths:
        print("Wrote stage configs:")
        for path in config_paths:
            print(f"- {path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
