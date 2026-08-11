from __future__ import annotations

import re

from paper_library.schemas.parsed_document import NormalizationSegment

_WHITESPACE = re.compile(r"\s+")
_HYPHENATED_LINE_BREAK = re.compile(r"(?<=\w)-\s*\n\s*(?=\w)")


def normalize_text(raw_text: str) -> tuple[str, list[NormalizationSegment]]:
    if not raw_text:
        return "", []
    dehyphenated = _HYPHENATED_LINE_BREAK.sub("", raw_text)
    normalized = _WHITESPACE.sub(" ", dehyphenated).strip()
    if not normalized:
        return "", []
    return normalized, [
        NormalizationSegment(
            normalized_start=0,
            normalized_end=len(normalized),
            raw_start=0,
            raw_end=len(raw_text),
        )
    ]


def grounding_normal_form(value: str) -> str:
    dehyphenated = _HYPHENATED_LINE_BREAK.sub("", value)
    return _WHITESPACE.sub(" ", dehyphenated).strip()
