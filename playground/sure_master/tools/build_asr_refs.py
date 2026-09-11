from __future__ import annotations

import argparse
import gzip
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from playground.sure_master.baselines.asr_profiles import (
    AsrDatasetProfile,
    get_asr_dataset_profile,
    normalize_asr_cut_id,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "playground" / "sure_master" / "data"
DEFAULT_SEED = 20260714
TIER_SIZES_PER_SPLIT = {
    "smoke": 10,
    "early": 150,
    "regular": 500,
}
TEDLIUM_TIER_SIZES = {"smoke": 10, "early": 50, "regular": 200}


@dataclass(frozen=True)
class RefRow:
    split: str
    key: str
    text: str
    group: str


def output_files_for_profile(profile: AsrDatasetProfile) -> dict[str, str]:
    return {
        "smoke": f"{profile.ref_file_prefix}_smoke_ref.txt",
        "early": f"{profile.ref_file_prefix}_early_ref.txt",
        "regular": f"{profile.ref_file_prefix}_regular_ref.txt",
        "selection": f"{profile.ref_file_prefix}_selection_ref.txt",
    }


def _recording_id(cut: dict) -> str:
    recording = cut.get("recording") or {}
    if isinstance(recording, dict) and recording.get("id"):
        return str(recording["id"])
    if cut.get("recording_id"):
        return str(cut["recording_id"])
    return str(cut.get("id") or "")


def grouping_value(cut: dict, supervision: dict, profile: AsrDatasetProfile, key: str) -> str:
    if profile.grouping_key == "speaker":
        return str(supervision.get("speaker") or key.split("-", 1)[0] or key)
    if profile.grouping_key == "speaker_or_recording":
        return str(supervision.get("speaker") or _recording_id(cut) or key)
    if profile.grouping_key == "recording":
        return str(_recording_id(cut) or key)
    return str(supervision.get("speaker") or _recording_id(cut) or key)


def manifest_path(manifest_dir: Path, split: str, profile: AsrDatasetProfile) -> Path:
    try:
        pattern = profile.manifest_patterns[split]
    except KeyError as exc:
        allowed = ", ".join(sorted(profile.manifest_patterns))
        raise ValueError(
            f"No manifest pattern for split {split!r} in dataset profile "
            f"{profile.name!r}; known splits: {allowed}"
        ) from exc
    return manifest_dir / pattern


def load_ref_rows(
    manifest_dir: Path,
    split: str,
    profile: AsrDatasetProfile,
) -> list[RefRow]:
    path = manifest_path(manifest_dir, split, profile)
    rows: list[RefRow] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            cut = json.loads(line)
            supervisions = cut.get("supervisions") or []
            for supervision in supervisions:
                raw_key = supervision.get("id") or cut.get("id")
                if not raw_key:
                    continue
                key = normalize_asr_cut_id(str(raw_key), profile)
                text = (supervision.get("text") or "").strip()
                if not key or not text:
                    continue
                group = grouping_value(cut, supervision, profile, key)
                rows.append(RefRow(split=split, key=key, text=text, group=group))
    rows.sort(key=lambda row: (row.group, row.key))
    return rows


def select_tiered_rows(
    rows: list[RefRow],
    tier_sizes: dict[str, int],
    *,
    seed: int,
    split: str,
) -> dict[str, list[RefRow]]:
    groups_by_key: dict[str, list[RefRow]] = {}
    for row in rows:
        groups_by_key.setdefault(row.group, []).append(row)

    rng = random.Random(f"{seed}:{split}")
    groups = [
        (group, sorted(group_rows, key=lambda row: row.key))
        for group, group_rows in groups_by_key.items()
    ]
    rng.shuffle(groups)
    groups.sort(key=lambda item: len(item[1]))

    selected: dict[str, list[RefRow]] = {}
    remaining_groups = list(groups)
    for tier, size in tier_sizes.items():
        tier_rows: list[RefRow] = []
        while len(tier_rows) < size:
            if not remaining_groups:
                raise ValueError(
                    f"Not enough rows in {split} to fill {tier} with {size} rows"
                )
            _, group_rows = remaining_groups.pop(0)
            needed = size - len(tier_rows)
            tier_rows.extend(group_rows[:needed])
        selected[tier] = sorted(tier_rows, key=lambda row: row.key)

    selection_rows = [row for _, group_rows in remaining_groups for row in group_rows]
    selected["selection"] = sorted(selection_rows, key=lambda row: row.key)
    return selected


def write_ref(path: Path, rows: Iterable[RefRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            if row.key in seen:
                raise ValueError(f"Duplicate key while writing {path}: {row.key}")
            seen.add(row.key)
            f.write(f"{row.key}\t{row.text}\n")


def read_ref_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    if not path.is_file():
        return keys
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if "\t" not in line:
                continue
            key, _ = line.rstrip("\n").split("\t", 1)
            if key:
                keys.add(key)
    return keys


def validate_refs(
    output_dir: Path,
    profile: AsrDatasetProfile,
    final_ref: Path | None = None,
) -> None:
    seen: dict[str, str] = {}
    for tier, filename in output_files_for_profile(profile).items():
        path = output_dir / filename
        keys = read_ref_keys(path)
        with path.open("r", encoding="utf-8") as handle:
            line_count = sum(1 for _ in handle)
        if not keys:
            raise ValueError(f"{path} contains no reference utterances")
        if len(keys) != line_count:
            raise ValueError(f"{path} contains duplicate or malformed keys")
        overlap = sorted(key for key in keys if key in seen)
        if overlap:
            raise ValueError(
                f"{path} overlaps with {seen[overlap[0]]}: {', '.join(overlap[:10])}"
            )
        for key in keys:
            seen[key] = tier

    if final_ref and final_ref.is_file():
        final_keys = read_ref_keys(final_ref)
        overlap = sorted(set(seen).intersection(final_keys))
        if overlap:
            raise ValueError(
                f"Generated search refs overlap with {final_ref}: {', '.join(overlap[:10])}"
            )


def build_refs(
    dataset: str,
    manifest_dir: Path,
    output_dir: Path,
    *,
    seed: int,
    tier_sizes: dict[str, int] | None = None,
) -> dict[str, int]:
    profile = get_asr_dataset_profile(dataset)
    sizes = tier_sizes or (TEDLIUM_TIER_SIZES if profile.name == "tedlium3" else TIER_SIZES_PER_SPLIT)
    combined: dict[str, list[RefRow]] = {
        tier: [] for tier in [*sizes.keys(), "selection"]
    }
    for split in profile.search_splits:
        rows = load_ref_rows(manifest_dir, split, profile)
        selected = select_tiered_rows(rows, sizes, seed=seed, split=split)
        for tier, tier_rows in selected.items():
            combined[tier].extend(tier_rows)

    counts: dict[str, int] = {}
    split_order = {split: index for index, split in enumerate(profile.search_splits)}
    for tier, filename in output_files_for_profile(profile).items():
        rows = sorted(
            combined[tier],
            key=lambda row: (split_order.get(row.split, 999), row.key),
        )
        write_ref(output_dir / filename, rows)
        counts[tier] = len(rows)
    return counts


def parse_tier_sizes(raw: str | None) -> dict[str, int] | None:
    if not raw:
        return None
    values: dict[str, int] = {}
    for item in raw.replace(",", " ").split():
        if "=" not in item:
            raise ValueError(f"Expected tier=size item, got {item!r}")
        tier, value = item.split("=", 1)
        values[tier.strip()] = int(value)
    return values or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build tiered ASR refs for SURE ASR WER from icefall manifests."
    )
    parser.add_argument("--dataset", default="librispeech")
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--final-ref", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--tier-sizes",
        default=None,
        help="Optional space/comma separated tier sizes, e.g. 'smoke=10 early=150 regular=500'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = get_asr_dataset_profile(args.dataset)
    tier_sizes = parse_tier_sizes(args.tier_sizes)
    counts = build_refs(
        profile.name,
        args.manifest_dir,
        args.output_dir,
        seed=args.seed,
        tier_sizes=tier_sizes,
    )
    validate_refs(args.output_dir, profile, args.final_ref)
    for tier, filename in output_files_for_profile(profile).items():
        print(f"{tier}: {counts[tier]} -> {args.output_dir / filename}")


if __name__ == "__main__":
    main()
