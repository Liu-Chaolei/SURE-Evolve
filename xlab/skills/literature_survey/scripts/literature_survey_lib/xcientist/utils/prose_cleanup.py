import re


def remove_repeated_prose(text: str) -> tuple[str, int]:
    """Collapse exact long prose repetitions within a subsection, retaining evidence."""
    parts = re.split(r"(\n\s*\n)", text)
    output = []
    seen = set()
    removed = 0
    fenced = False
    for index in range(0, len(parts), 2):
        paragraph = parts[index]
        separator = parts[index + 1] if index + 1 < len(parts) else ""
        has_fence = bool(re.search(r"(?m)^\s*(?:```|~~~)", paragraph))
        if re.search(r"(?m)^#{1,6}\s", paragraph):
            seen.clear()
        key = " ".join(paragraph.split())
        prose = (not fenced and not has_fence and len(key) > 200
                 and not re.search(r"(?m)^\s*(?:#|\||\$|>|[-*+]\s|\d+[.)]\s)", paragraph))
        if prose and key in seen:
            removed += 1
            continue
        if prose:
            seen.add(key)
        output.extend([paragraph, separator])
        if len(re.findall(r"(?m)^\s*(?:```|~~~)", paragraph)) % 2:
            fenced = not fenced
    return "".join(output), removed
