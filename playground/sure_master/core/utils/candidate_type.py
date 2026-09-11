from __future__ import annotations

import ast
import json
import re
from typing import Any


INFERENCE = "inference"
FINE_TUNE = "fine_tune"
ARCH = "arch"
TRAINING = "training"
TRAINING_TYPES = {FINE_TUNE, ARCH}
VALID_CANDIDATE_TYPES = {INFERENCE, FINE_TUNE, ARCH}
LEGACY_TRAINING_TYPES = {TRAINING, "finetune", "fine-tune", "fine tune"}

_TYPE_PREFIX_RE = re.compile(
    r"^\s*\[(inference|training|fine_tune|fine-tune|finetune|arch|architecture|structure)\]\s*",
    re.IGNORECASE,
)
_ARCH_TEXT_RE = re.compile(
    r"\b(arch(?:itecture)?|structure|model structure|parameter count|"
    r"depth|ff[_ -]?mult|conv[_ -]?layers|qk[_ -]?norm|"
    r"num[_ -]?encoder[_ -]?layers|encoder[_ -]?dim|"
    r"feedforward[_ -]?dim|encoder[_ -]?unmasked[_ -]?dim)\b",
    re.IGNORECASE,
)
_TRAINING_TEXT_RE = re.compile(
    r"\b(fine[-_ ]?tune|finetune|training|train checkpoint|checkpoint training)\b",
    re.IGNORECASE,
)
_ARCH_CODE_MARKERS = (
    "SURE_TTS_ARCH_WRAPPER",
    "arch_finetune_short",
    "run_f5tts_arch_finetune.py",
    "--depth",
    "--ff-mult",
    "--ff_mult",
    "--conv-layers",
    "--conv_layers",
    "--qk-norm",
    "--qk_norm",
    "--num-encoder-layers",
    "--encoder-dim",
    "--feedforward-dim",
    "--encoder-unmasked-dim",
)
_FINE_TUNE_CODE_MARKERS = (
    "SURE_TTS_FINETUNE_WRAPPER",
    "SURE_TTS_TRAIN_ACTION",
    "finetune_short",
    "run_f5tts_finetune.py",
    "base_model/recipe/train.py",
    "recipe/train.py",
    "SURE_ICEFALL_PYTHON",
    "SURE_MAX_TRAIN_EPOCHS",
    "SURE_BASELINE_TRAIN_MAX_DURATION",
)


def normalize_candidate_type(value: Any, default: str = INFERENCE) -> str:
    """Normalize user/model provided candidate type labels."""
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"architecture", "structure"}:
        return ARCH
    if text in {"finetune", "fine_tune"}:
        return FINE_TUNE
    if text in {item.replace("-", "_").replace(" ", "_") for item in LEGACY_TRAINING_TYPES}:
        return FINE_TUNE
    if text in VALID_CANDIDATE_TYPES:
        return text
    return default if default in VALID_CANDIDATE_TYPES else INFERENCE


def candidate_type_from_idea(idea: Any, default: str = INFERENCE) -> str:
    """Infer the intended execution type from a research idea.

    Research plans historically used tuple pairs like ("1", "idea text"). Newer
    mixed F5-TTS plans may prefix idea text with [training] or [inference], or
    use dict fields if an LLM ignores the requested compact string shape.
    """
    if isinstance(idea, tuple) and len(idea) >= 2:
        return candidate_type_from_idea(idea[1], default=default)

    if isinstance(idea, dict):
        for key in ("idea_type", "candidate_type", "type"):
            if key in idea:
                return normalize_candidate_type(idea.get(key), default=default)
        try:
            text = json.dumps(idea, ensure_ascii=False)
        except TypeError:
            text = str(idea)
    else:
        text = str(idea or "")

    prefix = _TYPE_PREFIX_RE.match(text)
    if prefix:
        raw_label = prefix.group(1)
        normalized = normalize_candidate_type(raw_label, default=default)
        legacy_training_label = raw_label.strip().lower().replace("-", "_").replace(" ", "_") == TRAINING
        if legacy_training_label and _ARCH_TEXT_RE.search(text):
            return ARCH
        return normalized
    if _ARCH_TEXT_RE.search(text):
        return ARCH
    if _TRAINING_TEXT_RE.search(text):
        return FINE_TUNE
    return normalize_candidate_type(default)


def candidate_type_from_code(code: str, default: str = INFERENCE) -> str:
    """Classify explicit wrapper actions before inherited architecture arguments."""
    if "SURE_TASK_WRAPPER" in code or "SURE_ASR_ZIPFORMER_WRAPPER" in code:
        try:
            tree = ast.parse(code)
            actions = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.List, ast.Tuple)):
                    values = [item.value if isinstance(item, ast.Constant) else None for item in node.elts]
                    actions.update(values[i + 1] for i, value in enumerate(values[:-1]) if value == "--action" and isinstance(values[i + 1], str))
            if actions and actions <= {"infer", "decode_only"}:
                return INFERENCE
            if "arch" in actions:
                return ARCH
            if actions & {"fine_tune", "train_decode"}:
                return ARCH if any(marker in code for marker in _ARCH_CODE_MARKERS) else FINE_TUNE
        except SyntaxError:
            pass
    if any(marker in code for marker in _ARCH_CODE_MARKERS):
        return ARCH
    if any(marker in code for marker in _FINE_TUNE_CODE_MARKERS):
        return FINE_TUNE
    return normalize_candidate_type(default)


def is_training_candidate(code: str, idea: Any = None) -> bool:
    hinted = candidate_type_from_idea(idea)
    return candidate_type_from_code(code, default=hinted) in TRAINING_TYPES


def is_arch_candidate_type(value: Any) -> bool:
    return normalize_candidate_type(value) == ARCH
