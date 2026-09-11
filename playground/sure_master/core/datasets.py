"""Explicit split identities shared by all tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import file_digest

SPLITS = {"train", "train_validation", "search", "selection", "holdout"}


@dataclass(frozen=True)
class DatasetSplitSpec:
    name: str
    manifest: str = ""
    roles: dict[str, str] = field(default_factory=dict)
    execution_env: dict[str, str] = field(default_factory=dict)
    base_model_source_paths: dict[str, str] = field(default_factory=dict)

    def validate(self) -> dict[str, Any]:
        if self.name not in SPLITS:
            raise ValueError(f"Unknown data split: {self.name}")
        paths = {**self.roles, **({"manifest": self.manifest} if self.manifest else {})}
        hashes = {}
        for role, raw in paths.items():
            path = Path(raw)
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing/empty {self.name}.{role}: {path}")
            hashes[role] = file_digest(path)
        return {"split": self.name, "digests": hashes}


def split_specs(sure: dict) -> dict[str, DatasetSplitSpec]:
    raw = dict(sure.get("datasets") or {})
    legacy = sure.get("final_evaluation") or {}
    for name, key in (("selection", "selection_ref"), ("holdout", "test_ref")):
        if name not in raw and legacy.get(key):
            raw[name] = {"roles": {"ref": legacy[key]}}
    result = {}
    for name, value in raw.items():
        if name not in SPLITS:
            raise ValueError(f"Unknown data split: {name}")
        result[name] = DatasetSplitSpec(name=name, **value)
    return result


def validate_split_groups(specs: dict[str, DatasetSplitSpec]) -> None:
    owners: dict[str, str] = {}
    for name, spec in specs.items():
        if not spec.manifest or Path(spec.manifest).suffix != ".jsonl":
            continue
        ids: set[str] = set()
        for line in Path(spec.manifest).read_text().splitlines():
            row = json.loads(line)
            sid = str(
                row.get("sample_id") or row.get("session_id") or row.get("id") or ""
            )
            group = str(row.get("group_id") or "")
            if not sid or sid in ids:
                raise ValueError(f"Missing or duplicate sample ID in {name}: {sid}")
            ids.add(sid)
            if not group:
                raise ValueError(f"Missing group_id in {name}: {sid}")
            if group:
                previous = owners.setdefault(group, name)
                if previous != name:
                    raise ValueError(f"Group {group} overlaps {previous} and {name}")
