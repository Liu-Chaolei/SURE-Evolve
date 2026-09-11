"""One-time source completeness verification for prepared full-training datasets."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from playground.sure_master.core.artifacts import file_digest
from playground.sure_master.runtime.training_state import atomic_json


def verify_premium_source(root: Path, receipt: Path) -> dict:
    checksum = root / "Premium_md5check.txt"
    expected = {}
    for line in checksum.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.strip().lstrip("*")
        if len(digest) != 32 or Path(name).name != name or not name.endswith(".tar.gz"):
            raise ValueError("Invalid Premium archive checksum manifest")
        expected[name] = digest
    if not expected:
        raise ValueError("Empty Premium archive manifest")
    previous = json.loads(receipt.read_text()) if receipt.exists() else {}
    identity = file_digest(checksum)
    verified = (
        previous.get("verified_archives", {})
        if previous.get("manifest_sha256") == identity
        else {}
    )
    for name, md5 in expected.items():
        archive = root / name
        old = verified.get(name, {})
        if archive.is_file():
            stat = archive.stat()
            current = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "md5": md5}
            if current != {key: old.get(key) for key in current}:
                value = hashlib.md5()
                with archive.open("rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        value.update(block)
                if value.hexdigest() != md5:
                    raise ValueError(
                        f"Premium archive download is incomplete/corrupt: {name}"
                    )
                verified[name] = current
                atomic_json(
                    receipt,
                    {
                        "manifest_sha256": identity,
                        "verified_archives": verified,
                        "source_complete": False,
                    },
                )
        elif old.get("md5") != md5:
            raise FileNotFoundError(
                f"Unverified/missing Premium archive: {name}; retain archives for the first verification"
            )
        inventory = verified.get(name, {}).get("files")
        if inventory is None:
            if not archive.is_file():
                raise FileNotFoundError(
                    f"Missing extraction inventory for {name}; verify before removing the archive"
                )
            inventory = {}
            with tarfile.open(archive, mode="r|gz") as members:
                for member in members:
                    relative = Path(member.name)
                    if member.isfile() and relative.parent.name in {"txts", "wavs"}:
                        key = f"{relative.parent.name}/{relative.name}"
                        if key in inventory:
                            raise ValueError(f"Duplicate archive member: {key}")
                        inventory[key] = member.size
            if not inventory:
                raise ValueError(f"No audio/transcript inventory in {name}")
            verified[name]["files"] = inventory
            atomic_json(
                receipt,
                {
                    "manifest_sha256": identity,
                    "verified_archives": verified,
                    "source_complete": False,
                },
            )
        directory = root / name.removesuffix(".tar.gz")
        for relative, size in inventory.items():
            target = directory / relative
            if not target.is_file() or target.stat().st_size != size:
                raise ValueError(f"Premium archive extraction is incomplete: {target}")
        texts = {p.stem for p in (directory / "txts").glob("*.txt")}
        waves = {p.stem for p in (directory / "wavs").glob("*.wav")}
        if not texts or texts != waves:
            raise ValueError(f"Premium archive extraction is incomplete: {directory}")
    result = {
        "manifest_sha256": identity,
        "verified_archives": verified,
        "source_complete": True,
    }
    atomic_json(receipt, result)
    return result


def write_preparation(output: Path, adapter: str, splits: dict, source: dict) -> None:
    digests = {
        name: file_digest(Path(spec["manifest"])) for name, spec in splits.items()
    }
    if adapter == "tts.f5tts":
        digests["train_csv"] = file_digest(output / "train/metadata.csv")
    atomic_json(
        output / "preparation.json",
        {
            "schema_version": "sure.prepared_training.v1",
            "adapter": adapter,
            "source_complete": True,
            "training_selection": "full_prepared_train",
            "digests": digests,
            "source": source,
        },
    )
