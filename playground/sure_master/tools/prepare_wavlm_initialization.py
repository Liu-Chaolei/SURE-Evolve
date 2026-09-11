#!/usr/bin/env python3
"""Convert the official SSL backbone with DiariZen's converter and verify exact loading."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from playground.sure_master.core.artifacts import file_digest
from playground.sure_master.runtime.training_state import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diarizen-root", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Local microsoft/wavlm-base-plus snapshot",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the official repository into --source first",
    )
    args = parser.parse_args()
    if args.download:
        from huggingface_hub import snapshot_download

        snapshot_download("microsoft/wavlm-base-plus", local_dir=str(args.source))
    config = json.loads((args.source / "config.json").read_text())
    if config.get("hidden_size") != 768 or config.get("num_hidden_layers") != 12:
        raise ValueError("Expected the unpruned WavLM-Base+ architecture")
    sys.path[:0] = [
        str(args.diarizen_root.resolve()),
        str((args.diarizen_root / "pyannote-audio").resolve()),
    ]
    import torch
    from diarizen.models.pruning.utils import convert_wavlm
    from diarizen.models.module.wav2vec2.model import wav2vec2_model

    args.output.mkdir(parents=True, exist_ok=True)
    # Upstream converter uses the folder name to select a model-sized output name.
    alias = args.output.resolve() / "wavlm-base-plus"
    if alias.exists() and alias.resolve() != args.source.resolve():
        raise ValueError("Existing conversion source differs from requested source")
    if not alias.exists():
        alias.symlink_to(args.source.resolve(), target_is_directory=True)
    convert_wavlm(str(alias), str(args.output.resolve()))
    checkpoint = args.output / "wavlm-base-plus-converted.bin"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = wav2vec2_model(**payload["config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    source_files = {
        p.name: file_digest(p)
        for p in args.source.iterdir()
        if p.is_file()
        and (p.suffix in {".bin", ".safetensors"} or p.name == "config.json")
    }
    if len(source_files) < 2:
        raise ValueError("The SSL source snapshot is missing model weights")
    atomic_json(
        checkpoint.with_suffix(checkpoint.suffix + ".provenance.json"),
        {
            "kind": "ssl_backbone",
            "source_model": "microsoft/wavlm-base-plus",
            "source_files": source_files,
            "sha256": file_digest(checkpoint),
            "strict_state_match": True,
        },
    )
    print(json.dumps({"checkpoint": str(checkpoint.resolve()), "verified": True}))


if __name__ == "__main__":
    main()
