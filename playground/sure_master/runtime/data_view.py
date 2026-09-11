"""Workspace-local evaluation manifests containing only requested utterances."""
from __future__ import annotations

import gzip
import json
from pathlib import Path


def evaluation_data_view(source: Path, destination: Path, reference: Path, splits: list[str], *, include_train: bool = True) -> Path:
    keys = {line.split("\t", 1)[0] for line in reference.read_text().splitlines() if "\t" in line}
    if not keys:
        raise ValueError("Cannot filter evaluation data with an empty reference")
    if destination.is_symlink() or (destination / "fbank").is_symlink():
        raise ValueError("Evaluation view directories must not be symlinks")
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name not in {"fbank", "refs"} and not (destination / item.name).exists():
            (destination / item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())
    fbank = destination / "fbank"
    fbank.mkdir(exist_ok=True)
    selected_names = {f"tedlium_cuts_{split}.jsonl.gz" for split in splits}
    found: set[str] = set()
    for item in (source / "fbank").iterdir():
        target = fbank / item.name
        if item.name in selected_names:
            if target.exists() or target.is_symlink():
                target.unlink()
            with gzip.open(item, "rt") as input_file, gzip.open(target, "wt") as output:
                for line in input_file:
                    cut = json.loads(line)
                    cut_keys = {str(cut.get("id"))} | {str(s.get("id")) for s in cut.get("supervisions", [])}
                    if cut_keys & keys:
                        output.write(line)
                        found.update(cut_keys & keys)
        elif item.name.startswith("tedlium_cuts_") and (item.name != "tedlium_cuts_train.jsonl.gz" or not include_train):
            continue
        elif not target.exists():
            target.symlink_to(item.resolve(), target_is_directory=item.is_dir())
    if found != keys:
        raise ValueError(f"{len(keys - found)} reference IDs are missing from the selected evaluation manifests")
    return destination
