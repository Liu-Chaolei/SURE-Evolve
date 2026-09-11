import re


def apply_editorial_corrections(text: str, corrections: list[dict], known_papers: set[str]) -> tuple[str, list[dict]]:
    """Apply reviewed, exact-text factual corrections during native checkpoint export."""
    report = []
    for correction in corrections:
        before = correction["before"]
        after = correction["after"]
        evidence = correction["evidence_paper_ids"]
        expected = correction["occurrences"]
        if (not before or not after or not correction.get("reason") or not evidence
                or not set(evidence).issubset(known_papers) or not isinstance(expected, int) or expected < 1):
            raise ValueError("Editorial correction requires exact text, reason and known source evidence")
        if re.findall(r"(?m)^#{1,6}\s.*$", before) != re.findall(r"(?m)^#{1,6}\s.*$", after):
            raise ValueError("Editorial corrections must preserve section headings")
        found = text.count(before)
        if found != expected:
            raise ValueError(f"Editorial correction expected {expected} matches, found {found}")
        text = text.replace(before, after)
        report.append({"reason": correction["reason"], "evidence_paper_ids": evidence, "occurrences": found})
    return text, report
