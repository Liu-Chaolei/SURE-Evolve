"""Stage only the selected data view once per candidate job, including feature paths."""

from __future__ import annotations
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path


def stage_view(source: Path) -> Path:
    local = Path("/local/job")
    if not os.environ.get("SLURM_JOB_ID") or not local.is_dir():
        raise RuntimeError("Data staging requires the Slurm private local directory")
    manifests = sorted((source / "fbank").glob("tedlium_cuts_*.jsonl.gz"))
    if os.environ.get("SURE_ENABLE_MUSAN") == "1" and any("_train." in p.name for p in manifests):
        musan = source / "fbank/musan_cuts.jsonl.gz"
        if not musan.is_file():
            raise FileNotFoundError("MUSAN is enabled but its feature manifest is missing")
        manifests.append(musan)
    digest = hashlib.sha256()
    digest.update((source / "lang_bpe_500/bpe.model").read_bytes())
    for manifest in manifests:
        digest.update(manifest.name.encode())
        digest.update(manifest.read_bytes())
    target = local / "sure-data" / digest.hexdigest()
    marker = target / "ready"
    if marker.exists():
        return target
    (target / "fbank").mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source / "lang_bpe_500", target / "lang_bpe_500", dirs_exist_ok=True
    )
    copied = {}
    for manifest in manifests:
        with (
            gzip.open(manifest, "rt") as stream,
            gzip.open(target / "fbank" / manifest.name, "wt") as output,
        ):
            for line in stream:
                row = json.loads(line)
                features = row.get("features", {})
                raw = features.get("storage_path")
                if raw:
                    original = Path(raw)
                    if not original.is_absolute():
                        original = (Path.cwd() / original).resolve()
                    key = str(original)
                    if key not in copied:
                        destination = (
                            target
                            / "features"
                            / hashlib.sha256(key.encode()).hexdigest()[:16]
                            / original.name
                        )
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if original.is_dir():
                            shutil.copytree(original, destination, dirs_exist_ok=True)
                        else:
                            shutil.copy2(original, destination)
                        copied[key] = str(destination)
                    features["storage_path"] = copied[key]
                output.write(json.dumps(row) + "\n")
    marker.write_text("complete\n")
    return target
