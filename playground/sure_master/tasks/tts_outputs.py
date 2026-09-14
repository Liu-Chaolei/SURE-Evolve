"""Check generated speech against the immutable evaluation manifest."""

from __future__ import annotations

import json
from pathlib import Path


def validate_samples(manifest: Path, samples: Path) -> None:
    expected = [
        json.loads(line) for line in manifest.read_text().splitlines() if line.strip()
    ]
    actual = [
        json.loads(line) for line in samples.read_text().splitlines() if line.strip()
    ]
    refs = {r["sample_id"]: r for r in expected}
    if len(refs) != len(expected) or len(actual) != len(refs):
        raise ValueError("TTS output count differs from the evaluation manifest")
    seen = set()
    for row in actual:
        sid = row.get("sample_id")
        if sid not in refs or sid in seen:
            raise ValueError("TTS output contains a duplicate or unknown sample ID")
        seen.add(sid)
        ref = refs[sid]
        if row.get("reference_text") != ref["target_text"]:
            raise ValueError("TTS scoring text differs from the frozen target text")
        if ref.get("language") and row.get("language") != ref["language"]:
            raise ValueError("TTS output language differs from the evaluation manifest")
        audio = Path(row.get("prediction_audio", ""))
        audio = audio if audio.is_absolute() else samples.parent / audio
        audio.resolve().relative_to(samples.parent.resolve())
        if not audio.is_file():
            raise ValueError("Missing generated TTS audio")
