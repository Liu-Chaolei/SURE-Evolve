from __future__ import annotations

import argparse
from pathlib import Path

from playground.sure_master.tools.build_asr_refs import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    build_refs,
    get_asr_dataset_profile,
    output_files_for_profile,
    validate_refs,
)


DEFAULT_MANIFEST_DIR = Path(
    "/hpc_stor03/sjtu_home/chaolei.liu/ASR/icefall/egs/librispeech/ASR/data/fbank"
)
DEFAULT_TEST_REF = DEFAULT_OUTPUT_DIR / "asr_en_wer_ref.txt"
DATASET = "librispeech"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build tiered LibriSpeech dev refs for SURE ASR WER."
    )
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-ref", type=Path, default=DEFAULT_TEST_REF)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = get_asr_dataset_profile(DATASET)
    counts = build_refs(DATASET, args.manifest_dir, args.output_dir, seed=args.seed)
    validate_refs(args.output_dir, profile, args.test_ref)
    for tier, filename in output_files_for_profile(profile).items():
        print(f"{tier}: {counts[tier]} -> {args.output_dir / filename}")


if __name__ == "__main__":
    main()
