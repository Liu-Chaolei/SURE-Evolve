import math
import re
from typing import Tuple


_BOXED_RE = re.compile(r"\\boxed\{+\s*([^{}]+?)\s*\}+")


def normalize_wer_score(value: float | int | str | None) -> float | None:
    """Normalize WER to percentage-number scale, e.g. 15.66 means 15.66%."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().rstrip("%")
    try:
        wer = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(wer) or math.isinf(wer) or wer < 0:
        return None
    if 0 < wer < 1:
        wer *= 100.0
    if wer > 100:
        return None
    return wer


def extract_wer_from_metric(metric_result: str) -> Tuple[bool, float | None]:
    """Extract boxed WER and normalize it to percentage-number scale."""
    if not metric_result:
        return False, None
    matches = _BOXED_RE.findall(metric_result)
    if not matches:
        return False, None
    wer = normalize_wer_score(matches[-1].strip())
    return (wer is not None), wer
