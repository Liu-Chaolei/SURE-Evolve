from __future__ import annotations

import re
from pathlib import Path

from paper_library.storage.layout import BuildLayout
from paper_library.storage.manifest import CorpusManifest, CurrentManifest
from paper_library.utils.hashing import sha256_file

_CORPUS_VERSION_PATTERN = re.compile(r"^corpus_[0-9a-f]{24}$")


def load_corpus_manifest(
    layout: BuildLayout,
    corpus_version: str | None = None,
) -> tuple[CorpusManifest, Path]:
    if corpus_version is None:
        current_path = layout.manifests / "current.json"
        if not current_path.is_file():
            raise ValueError("No published corpus is available")
        current = CurrentManifest.model_validate_json(
            current_path.read_text(encoding="utf-8")
        )
        corpus_version = current.corpus_version
        manifest_path = layout.resolve_inside(current.manifest_path)
    else:
        if not _CORPUS_VERSION_PATTERN.fullmatch(corpus_version):
            raise ValueError(f"Invalid corpus version: {corpus_version}")
        manifest_path = layout.manifests / f"{corpus_version}.json"

    if not manifest_path.is_file():
        raise ValueError(f"Corpus manifest is missing: {corpus_version}")
    manifest = CorpusManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.corpus_version != corpus_version:
        raise ValueError("Corpus manifest version does not match the requested corpus")

    snapshot = (layout.records / corpus_version).resolve()
    for artifact in (
        manifest.parsed_documents,
        manifest.evidence_records,
        manifest.research_design_cards,
    ):
        path = layout.resolve_inside(artifact.path)
        if path != snapshot and snapshot not in path.parents:
            raise ValueError("Corpus manifest references an invalid artifact path")
        if not path.is_file():
            raise ValueError(f"Corpus artifact is missing: {artifact.path}")
        if sha256_file(path) != artifact.sha256:
            raise ValueError(f"Corpus artifact hash does not match: {artifact.path}")
        with path.open("r", encoding="utf-8") as stream:
            count = sum(bool(line.strip()) for line in stream)
        if count != artifact.record_count:
            raise ValueError(f"Corpus artifact count does not match: {artifact.path}")
    return manifest, manifest_path
