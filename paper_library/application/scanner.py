from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from paper_library.utils.hashing import sha256_file
from paper_library.utils.ids import document_id, stable_id


@dataclass(frozen=True)
class SourceDocument:
    paper_id: str
    document_id: str
    pdf_sha256: str
    primary_path: Path
    aliases: tuple[str, ...]
    domain: str
    size: int
    mtime_ns: int


def infer_domain(relative_path: Path) -> str:
    parts = {part.casefold() for part in relative_path.parts}
    if "asr" in parts:
        return "asr"
    if "tts" in parts:
        return "tts"
    return "speech"


def paper_id_for_document(pdf_sha256: str) -> str:
    return stable_id("paper", pdf_sha256)


def scan_pdfs(source_root: Path) -> list[SourceDocument]:
    root = source_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"PDF source root does not exist: {root}")
    paths = sorted(
        (path for path in root.rglob("*.pdf") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )
    by_hash: dict[str, list[Path]] = {}
    for path in paths:
        by_hash.setdefault(sha256_file(path), []).append(path)

    documents: list[SourceDocument] = []
    for digest, duplicates in sorted(by_hash.items(), key=lambda item: item[0]):
        primary = duplicates[0]
        relative = primary.relative_to(root)
        stat = primary.stat()
        documents.append(
            SourceDocument(
                paper_id=paper_id_for_document(digest),
                document_id=document_id(digest),
                pdf_sha256=digest,
                primary_path=primary,
                aliases=tuple(path.relative_to(root).as_posix() for path in duplicates),
                domain=infer_domain(relative),
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
        )
    return sorted(documents, key=lambda document: document.aliases[0].casefold())
