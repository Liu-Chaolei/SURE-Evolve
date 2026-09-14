"""Stage only this job's frozen training audio on its private local disk."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil

from .training_state import atomic_json


def stage_audio(paths: list[str], identity: dict) -> dict[str, str]:
    temporary = Path(os.environ["SLURM_TMPDIR"]).resolve()
    if temporary == Path("/shared") or "/shared/" in str(temporary):
        raise ValueError("TTS staging requires the job-private local filesystem")
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = temporary / "f5-audio" / key
    root.mkdir(parents=True, exist_ok=True)
    targets = {
        name: str(root / (hashlib.sha256(name.encode()).hexdigest() + ".wav"))
        for name in paths
    }
    with (root / "staging.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ready = root / "ready.json"
        if ready.exists() and json.loads(ready.read_text()) == identity:
            if all(Path(path).is_file() for path in targets.values()):
                return targets

        def copy(name):
            source, target = Path(name), Path(targets[name])
            if target.is_file() and target.stat().st_size == source.stat().st_size:
                return
            pending = target.with_suffix(".pending")
            shutil.copyfile(source, pending)
            pending.replace(target)

        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(copy, paths))
        atomic_json(ready, identity)
    return targets
