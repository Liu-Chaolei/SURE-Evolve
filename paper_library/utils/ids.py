from __future__ import annotations

from paper_library.utils.hashing import hash_canonical


def stable_id(namespace: str, *parts: object, digest_length: int = 24) -> str:
    if not namespace or not namespace.replace("_", "").replace("-", "").isalnum():
        raise ValueError("namespace must be a non-empty identifier")
    digest = hash_canonical({"namespace": namespace, "parts": parts})
    return f"{namespace}_{digest[:digest_length]}"


def document_id(pdf_sha256: str) -> str:
    if len(pdf_sha256) != 64:
        raise ValueError("PDF SHA-256 must contain 64 hexadecimal characters")
    int(pdf_sha256, 16)
    return f"doc_{pdf_sha256}"
