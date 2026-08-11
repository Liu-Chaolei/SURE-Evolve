from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BuildLayout:
    root: Path

    @property
    def state_dir(self) -> Path:
        return self.root / "state"

    @property
    def state_database(self) -> Path:
        return self.state_dir / "build.sqlite3"

    @property
    def parsed_objects(self) -> Path:
        return self.root / "objects" / "parsed"

    @property
    def candidate_objects(self) -> Path:
        return self.root / "objects" / "candidates"

    @property
    def evidence_objects(self) -> Path:
        return self.root / "objects" / "evidence"

    @property
    def card_objects(self) -> Path:
        return self.root / "objects" / "cards"

    @property
    def records(self) -> Path:
        return self.root / "records"

    @property
    def indexes(self) -> Path:
        return self.root / "indexes"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    def lexical_index(self, corpus_version: str) -> Path:
        return self.indexes / corpus_version / "lexical.sqlite3"

    @property
    def reviews(self) -> Path:
        return self.root / "reviews"

    @property
    def failures(self) -> Path:
        return self.root / "failures"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def staging(self) -> Path:
        return self.root / "staging"

    def ensure(self) -> None:
        for path in (
            self.state_dir,
            self.parsed_objects,
            self.candidate_objects,
            self.evidence_objects,
            self.card_objects,
            self.records,
            self.indexes,
            self.manifests,
            self.reviews,
            self.failures,
            self.reports,
            self.staging,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def resolve_inside(self, relative_path: str | Path) -> Path:
        candidate = (self.root / relative_path).resolve()
        root = self.root.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"Path escapes build root: {candidate}")
        return candidate
