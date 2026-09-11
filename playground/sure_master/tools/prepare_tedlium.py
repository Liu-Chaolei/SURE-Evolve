"""Prepare TEDLIUM STM/audio for SURE without modifying the source corpus.

Run with the Icefall Python environment. --refs-only needs only the standard
library and builds the same IDs and evaluation split as feature preparation.
"""
from __future__ import annotations

import argparse
import hashlib
import gzip
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from playground.sure_master.tools.build_asr_refs import (
    DEFAULT_SEED, TEDLIUM_TIER_SIZES, RefRow, output_files_for_profile,
    select_tiered_rows, validate_refs, write_ref,
)
from playground.sure_master.baselines.asr_profiles import get_asr_dataset_profile


@dataclass(frozen=True)
class Segment:
    key: str
    recording_id: str
    audio: str
    channel: int
    start: float
    duration: float
    text: str
    speaker: str


def corpus_directory(root: Path) -> Path:
    for candidate in (root, root / "TEDLIUM_release3", root / "TEDLIUM_release-3"):
        if (candidate / "legacy" / "dev").is_dir():
            return candidate.resolve()
    raise ValueError(f"No TEDLIUM corpus with legacy/dev under {root}")


def read_stm_split(root: Path, split: str) -> list[Segment]:
    """Support the official layout and the flattened legacy/{split} layout."""
    directory = root / "legacy" / split
    if split == "train" and not directory.is_dir():
        directory = root / "data"
    # The shipped corpus uses legacy/train/stm as a directory symlink to
    # data/stm. pathlib.rglob() does not recurse through that symlink, so use
    # the flattened data layout when the selected split has no STM files.
    stms = sorted(directory.rglob("*.stm"))
    if not stms and split == "train":
        directory = root / "data"
        stms = sorted(directory.rglob("*.stm"))
    audio = {path.stem: path for path in directory.rglob("*.sph")}
    if not stms:
        raise ValueError(f"No STM files for {split}: {directory}")
    result: list[Segment] = []
    seen: set[str] = set()
    for stm in stms:
        for index, line in enumerate(stm.read_text(encoding="utf-8").splitlines()):
            if not line.strip() or line.lstrip().startswith(";;"):
                continue
            fields = line.split(maxsplit=6)
            if len(fields) != 7:
                raise ValueError(f"Malformed STM row: {stm}:{index + 1}")
            recording, channel, speaker, start, end, _label, transcript = fields
            if transcript.strip().lower() == "ignore_time_segment_in_scoring":
                continue
            start_time, end_time = float(start), float(end)
            if not (math.isfinite(start_time) and math.isfinite(end_time) and 0 <= start_time < end_time):
                raise ValueError(f"Invalid STM interval: {stm}:{index + 1}")
            if recording not in audio:
                raise FileNotFoundError(f"Missing audio for {recording} in {directory}")
            key = f"{recording}-{index}"
            if key in seen:
                raise ValueError(f"Duplicate utterance ID: {key}")
            seen.add(key)
            result.append(Segment(key, recording, str(audio[recording].resolve()),
                                  int(channel) - 1, start_time, end_time - start_time,
                                  transcript.strip(), speaker))
    if not result:
        raise ValueError(f"No valid STM utterances for {split}")
    return sorted(result, key=lambda item: item.key)


def training_subset(segments: list[Segment], hours: float, seed: int) -> list[Segment]:
    if hours <= 0:
        return segments
    eligible = [s for s in segments if 1.0 <= s.duration <= 20]
    random.Random(seed).shuffle(eligible)
    selected: list[Segment] = []
    seconds = 0.0
    for segment in eligible:
        if seconds + segment.duration <= hours * 3600:
            selected.append(segment)
            seconds += segment.duration
    if not selected:
        raise ValueError("The training duration budget selected no utterances")
    return sorted(selected, key=lambda item: item.key)


def prepare_refs(splits: dict[str, list[Segment]], output: Path) -> dict[str, int]:
    profile = get_asr_dataset_profile("tedlium3")
    rows = [RefRow("dev", s.key, s.text, s.recording_id) for s in splits["dev"]]
    tiers = select_tiered_rows(rows, TEDLIUM_TIER_SIZES, seed=DEFAULT_SEED, split="dev")
    if not tiers["selection"]:
        raise ValueError("TEDLIUM selection has no remaining independent recordings")
    for tier, filename in output_files_for_profile(profile).items():
        write_ref(output / filename, tiers[tier])
    final_ref = output / "asr_tedlium3_test_ref.txt"
    write_ref(final_ref, [RefRow("test", s.key, s.text, s.recording_id) for s in splits["test"]])
    validate_refs(output, profile, final_ref)
    return {tier: len(rows) for tier, rows in tiers.items()}


def prepare_features(splits: dict[str, list[Segment]], output: Path, jobs: int) -> None:
    import torch
    torch.set_num_threads(1)
    from lhotse import CutSet, Fbank, FbankConfig, Recording, SupervisionSegment
    from lhotse.cut import MonoCut
    from lhotse.features.io import LilcomChunkyWriter
    import sentencepiece as spm

    lang = output / "lang_bpe_500"
    lang.mkdir(parents=True, exist_ok=True)
    transcript = lang / "train.txt"
    transcript.write_text("\n".join(s.text.upper() for s in splits["train"]) + "\n", encoding="utf-8")
    if not (lang / "bpe.model").is_file():
        spm.SentencePieceTrainer.train(
            input=str(transcript), model_prefix=str(lang / "bpe"), vocab_size=500,
            model_type="bpe", character_coverage=1.0, unk_id=2, bos_id=-1, eos_id=-1,
            user_defined_symbols=["<blk>", "<sos/eos>"],
            shuffle_input_sentence=False, num_threads=1,
        )
    extractor = Fbank(FbankConfig(num_mel_bins=80))
    for split, segments in splits.items():
        target = output / "fbank" / f"tedlium_cuts_{split}.jsonl.gz"
        if target.is_file():
            continue
        recordings = {}
        cuts = []
        for s in segments:
            if s.recording_id not in recordings:
                recordings[s.recording_id] = Recording.from_file(s.audio, recording_id=s.recording_id)
            recording = recordings[s.recording_id]
            if s.channel not in recording.channel_ids or s.start + s.duration > recording.duration + 0.03:
                raise ValueError(f"STM interval/channel exceeds audio: {s.key}")
            supervision = SupervisionSegment(
                id=s.key, recording_id=s.recording_id, start=0.0, duration=s.duration,
                channel=s.channel, text=s.text.upper(), language="English", speaker=s.speaker,
            )
            cuts.append(MonoCut(id=s.key, start=s.start, duration=s.duration, channel=s.channel,
                                recording=recording, supervisions=[supervision]))
        target.parent.mkdir(parents=True, exist_ok=True)
        result = CutSet.from_cuts(cuts).compute_and_store_features(
            extractor=extractor, storage_path=str(target.parent / f"tedlium_feats_{split}"),
            num_jobs=max(1, jobs), storage_type=LilcomChunkyWriter,
        )
        result.to_file(target)


def _ensure_symlink(target: Path, source: Path) -> None:
    source = source.expanduser().absolute()
    if target.is_symlink() and target.resolve() == source.resolve():
        return
    if target.exists() or target.is_symlink():
        raise ValueError(f"Cannot replace existing prepared-data path: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=source.is_dir())


def reuse_precomputed_features(
    feature_data: Path,
    bpe_dir: Path,
    output: Path,
) -> dict[str, object]:
    """Create a SURE data view over complete Icefall TEDLIUM3 features."""
    data = feature_data.expanduser().absolute()
    if (data / "data/fbank").is_dir():
        data = data / "data"
    fbank = data / "fbank"
    if not fbank.is_dir():
        raise FileNotFoundError(f"Missing reusable Icefall fbank directory: {fbank}")

    manifest_rows: dict[str, int] = {}
    manifest_variant: dict[str, str] = {}
    for split in ("train", "dev", "test"):
        lowercase = fbank / f"tedlium_cuts_{split}_lowercase.jsonl.gz"
        source = lowercase if lowercase.is_file() else fbank / f"tedlium_cuts_{split}.jsonl.gz"
        if not source.is_file():
            raise FileNotFoundError(f"Missing reusable {split} manifest: {source}")
        with gzip.open(source, "rt", encoding="utf-8") as stream:
            first = stream.readline()
            if not first.strip():
                raise ValueError(f"Reusable {split} manifest is empty: {source}")
            sample = json.loads(first)
        features = sample.get("features") or {}
        if features.get("num_features") != 80 or not features.get("storage_path"):
            raise ValueError(f"Reusable {split} manifest has incompatible features: {source}")
        feature_dir = fbank / f"tedlium_feats_{split}"
        if not feature_dir.is_dir() or not any(feature_dir.glob("*.lca")):
            raise FileNotFoundError(f"Missing reusable {split} feature archives: {feature_dir}")
        with gzip.open(source, "rt", encoding="utf-8") as stream:
            manifest_rows[split] = sum(1 for line in stream if line.strip())
        manifest_variant[split] = "lowercase" if source == lowercase else "default"
        _ensure_symlink(output / "fbank" / f"tedlium_cuts_{split}.jsonl.gz", source)
        _ensure_symlink(output / "fbank" / feature_dir.name, feature_dir)

    bpe = bpe_dir.expanduser().absolute()
    if bpe.is_file() and bpe.name == "bpe.model":
        bpe = bpe.parent
    if not (bpe / "bpe.model").is_file():
        raise FileNotFoundError(f"Missing reusable TEDLIUM3 BPE model: {bpe / 'bpe.model'}")
    _ensure_symlink(output / "lang_bpe_500", bpe)
    return {
        "feature_reuse": True,
        "feature_source": str(data),
        "feature_manifest_rows": manifest_rows,
        "feature_manifest_variant": manifest_variant,
        "bpe_source": str(bpe),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-hours", type=float, default=1.0, help="0 prepares the full training split")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42, help="Training subset seed; dev tier partition stays fixed")
    parser.add_argument("--search-hours", type=float, default=0, help="Create a feature-sharing search subset after full preparation")
    parser.add_argument("--refs-only", action="store_true")
    parser.add_argument(
        "--reuse-feature-data",
        type=Path,
        help="Reuse an existing Icefall TEDLIUM3 data/fbank tree instead of extracting features",
    )
    parser.add_argument(
        "--reuse-bpe-dir",
        type=Path,
        help="BPE directory paired with --reuse-feature-data (must contain bpe.model)",
    )
    args = parser.parse_args()
    if bool(args.reuse_feature_data) != bool(args.reuse_bpe_dir):
        parser.error("--reuse-feature-data and --reuse-bpe-dir must be set together")
    if args.refs_only and args.reuse_feature_data:
        parser.error("--refs-only cannot be combined with reusable feature data")
    root, output = corpus_directory(args.corpus), args.output.resolve()
    if output == root or root in output.parents:
        raise ValueError("Output must be outside the source corpus")
    splits = {split: read_stm_split(root, split) for split in ("train", "dev", "test")}
    splits["train"] = training_subset(splits["train"], args.train_hours, args.seed)
    identity = {split: [asdict(s) for s in rows] for split, rows in splits.items()}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "preparation.json"
    if marker.exists() and json.loads(marker.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Existing output belongs to different data; use a new output directory")
    previous_ready = marker.exists() and json.loads(marker.read_text()).get("features_ready", False)
    marker.write_text(json.dumps({"fingerprint": fingerprint, "features_ready": previous_ready}) + "\n")
    counts = prepare_refs(splits, output / "refs")
    feature_metadata: dict[str, object] = {}
    if not args.refs_only:
        if args.reuse_feature_data:
            feature_metadata = reuse_precomputed_features(
                args.reuse_feature_data,
                args.reuse_bpe_dir,
                output,
            )
        else:
            prepare_features(splits, output, args.jobs)
    summary = {"fingerprint": fingerprint, "corpus": str(root), "refs": counts,
               "training_selection": "full" if args.train_hours == 0 else "subset",
               "requested_train_hours": args.train_hours,
               "features_ready": previous_ready or not args.refs_only,
               "splits": {k: {"utterances": len(v), "hours": sum(s.duration for s in v) / 3600} for k, v in splits.items()},
               **feature_metadata}
    marker.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.search_hours:
        if args.train_hours != 0 or args.refs_only:
            raise ValueError("--search-hours requires full feature preparation (--train-hours 0)")
        subset = training_subset(splits["train"], args.search_hours, args.seed)
        search = output / f"search_{args.search_hours:g}h"
        keys = {row.key for row in subset}
        search.mkdir(exist_ok=True)
        for name in ("lang_bpe_500", "refs"):
            if not (search / name).exists():
                (search / name).symlink_to((output / name).resolve(), target_is_directory=True)
        (search / "fbank").mkdir(exist_ok=True)
        for source in (output / "fbank").iterdir():
            target = search / "fbank" / source.name
            if source.name == "tedlium_cuts_train.jsonl.gz":
                with gzip.open(source, "rt") as stream, gzip.open(target, "wt") as dest:
                    for line in stream:
                        if json.loads(line)["id"] in keys:
                            dest.write(line)
            elif not target.exists():
                target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
        (search / "train_ids.txt").write_text("\n".join(sorted(keys)) + "\n")
        (search / "preparation.json").write_text(json.dumps({
            "features_ready": True, "parent_fingerprint": fingerprint, "seed": args.seed,
            "utterances": len(keys), "hours": sum(row.duration for row in subset) / 3600,
            "ids_sha256": hashlib.sha256((search / "train_ids.txt").read_bytes()).hexdigest(),
            "bpe_sha256": hashlib.sha256((output / "lang_bpe_500/bpe.model").read_bytes()).hexdigest(),
        }, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
