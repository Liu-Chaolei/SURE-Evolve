#!/usr/bin/env python3
"""Batch F5-TTS inference wrapper for SURE Master candidates.

The wrapper keeps the F5-TTS model and vocoder loaded inside long-lived worker
processes. It replaces candidate scripts that start ``infer_cli.py`` once per
prompt, which repeatedly reloads the same checkpoint and wastes most runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from playground.sure_master.core.utils.candidate_changes import (  # noqa: E402
    validate_arch_candidate_changes,
)


WORKSPACE = Path.cwd()
ARTIFACTS = WORKSPACE / "artifacts"
WAV_DIR = ARTIFACTS / "wavs"
WORKING = WORKSPACE / "working"
BATCH_DIR = WORKING / "f5tts_batch_infer"
LOG_DIR = WORKING / "logs"

DEFAULTS = {
    "root": "base_model/root",
    "eval_data": "base_model/eval_data/prompts.jsonl",
    "language": "en",
    "max_samples": 4,
    "model": "F5TTS_v1_Base",
    "nfe_step": 32,
    "cfg_strength": 2.0,
    "sway_sampling_coef": -1.0,
    "speed": 1.0,
    "remove_silence": False,
    "run_timeout": 21600,
    "workers": "auto",
    "max_chunk_chars": 300,
    "min_chunk_chars": 40,
}


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def parse_int(value: Any, default: int, min_value: int | None = None) -> int:
    try:
        parsed = int(float(str(value).strip()))
    except (TypeError, ValueError):
        parsed = default
    if min_value is not None:
        parsed = max(min_value, parsed)
    return parsed


def parse_float(value: Any, default: float) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text == "":
        return default
    return text in {"1", "true", "yes", "on"}


def workspace_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else WORKSPACE / p


def resolve_read_path(raw: str | Path, base_dir: Path | None = None) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    if base_dir is not None and (base_dir / p).exists():
        return (base_dir / p).absolute()
    return (WORKSPACE / p).absolute()


def ensure_workspace_write_path(path: Path) -> None:
    base = WORKSPACE.resolve()
    parent = path.parent.resolve() if path.parent.exists() else path.parent.absolute()
    try:
        parent.relative_to(base)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to write outside workspace: {path}") from exc


def mkdir_inside(path: Path) -> None:
    ensure_workspace_write_path(path)
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_workspace_write_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, atomic: bool = False) -> None:
    ensure_workspace_write_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_path = path.with_name(f".{path.name}.{os.getpid()}.tmp") if atomic else path
    try:
        with write_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if atomic:
            os.replace(write_path, path)
    finally:
        if atomic:
            write_path.unlink(missing_ok=True)


def file_identity(value: Any) -> dict[str, Any] | str:
    text = str(value or "").strip()
    if not text or text.startswith("hf://"):
        return text
    path = Path(text).expanduser()
    try:
        stat = path.stat()
    except OSError:
        return text
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def synthesis_fingerprint(row: dict[str, Any], config: dict[str, Any]) -> str:
    reference_audio = Path(str(row.get("reference_audio") or ""))
    identity = {
        "gen_text": row.get("gen_text", ""),
        "reference_audio": file_identity(reference_audio),
        "reference_text": row.get("reference_text", ""),
        "model": config.get("model", ""),
        "ckpt_file": file_identity(config.get("ckpt_file", "")),
        "vocab_file": file_identity(config.get("vocab_file", "")),
        "model_cfg": file_identity(config.get("model_cfg", "")),
        "vocoder_name": config.get("vocoder_name", ""),
        "load_vocoder_from_local": config.get("load_vocoder_from_local"),
        "target_rms": config.get("target_rms"),
        "cross_fade_duration": config.get("cross_fade_duration"),
        "nfe_step": config.get("nfe_step"),
        "cfg_strength": config.get("cfg_strength"),
        "sway_sampling_coef": config.get("sway_sampling_coef"),
        "speed": config.get("speed"),
        "fix_duration": config.get("fix_duration"),
        "remove_silence": config.get("remove_silence"),
        "max_chunk_chars": config.get("max_chunk_chars"),
        "min_chunk_chars": config.get("min_chunk_chars"),
    }
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def fingerprint_path(wav_path: Path) -> Path:
    return wav_path.with_suffix(wav_path.suffix + ".fingerprint")


def resume_matches(row: dict[str, Any], config: dict[str, Any]) -> bool:
    wav_path = Path(str(row["output_wav"]))
    valid, _reason = validate_wav(wav_path)
    if not valid:
        return False
    try:
        stored = fingerprint_path(wav_path).read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return stored == synthesis_fingerprint(row, config)


def read_existing_candidate_changes() -> dict[str, Any]:
    path = ARTIFACTS / "candidate_changes.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} line {line_no} is not a JSON object")
            rows.append(value)
    return rows


def visible_cuda_devices() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return []
    if visible.lower() in {"none", "no", "-1"}:
        return []
    return [part.strip() for part in visible.split(",") if part.strip()]


def choose_worker_count(raw: str, row_count: int, device: str) -> int:
    text = str(raw or "auto").strip().lower()
    visible = visible_cuda_devices()
    uses_cuda = "cuda" in device.lower()
    if text == "auto":
        count = len(visible) if uses_cuda and visible else 1
    else:
        count = parse_int(text, 1, min_value=1)
        if uses_cuda:
            if not visible:
                count = 1
            else:
                count = min(count, len(visible))
    return max(1, min(count, max(row_count, 1)))


def sanitize_sample_id(sample_id: str, index: int) -> str:
    sid = str(sample_id or "").strip() or f"f5_en_{index:04d}"
    sid = re.sub(r"[^A-Za-z0-9_.-]+", "_", sid)
    return sid or f"f5_en_{index:04d}"


def unique_wav_name(sample_id: str, used: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._") or "sample"
    name = f"{base}.wav"
    suffix = 2
    while name in used:
        name = f"{base}_{suffix}.wav"
        suffix += 1
    used.add(name)
    return name


def normalize_text(text: str, mode: str) -> str:
    text = str(text)
    if mode in {"typography", "conservative"}:
        replacements = {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": ", ",
            "\u2014": ", ",
            "\u2026": "...",
            "\xa0": " ",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
    text = re.sub(r"\s+", " ", text).strip()
    if mode in {"typography", "conservative"} and text and text[-1] not in ".!?;:":
        text += "."
    return text


def validate_wav(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        if path.stat().st_size < 2048:
            return False, f"too small: {path.stat().st_size} bytes"
    except OSError as exc:
        return False, f"stat failed: {exc}"

    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            sr = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            if sr < 8000 or sr > 96000:
                return False, f"bad sample rate {sr}"
            if channels < 1 or channels > 8:
                return False, f"bad channels {channels}"
            if sampwidth not in (1, 2, 3, 4):
                return False, f"bad sample width {sampwidth}"
            if frames < sr * 0.20:
                return False, f"too short: {frames / max(sr, 1):.3f}s"
    except Exception as exc:
        return False, f"wav validation failed: {exc}"
    return True, "ok"


def rel_to_artifacts(path: Path) -> str:
    try:
        return os.path.relpath(str(path.absolute()), str(ARTIFACTS.absolute()))
    except ValueError:
        return str(path)


def load_prompts(eval_jsonl: Path, max_samples: int, language: str, text_cleanup: str) -> list[dict[str, Any]]:
    if not eval_jsonl.exists():
        raise FileNotFoundError(f"Prompt file not found: {eval_jsonl}")

    prompt_dir = eval_jsonl.parent
    rows: list[dict[str, Any]] = []
    used_wav_names: set[str] = set()
    with eval_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{eval_jsonl} line {line_no} is not a JSON object")
            if obj.get("language", language) != language:
                continue
            target_text = str(obj.get("target_text", "")).strip()
            if not target_text:
                continue

            index = len(rows) + 1
            sid = sanitize_sample_id(str(obj.get("sample_id", "")), index)
            wav_name = unique_wav_name(sid, used_wav_names)
            row_ref_audio = str(obj.get("reference_audio", "")).strip()
            ref_audio = resolve_read_path(row_ref_audio, prompt_dir) if row_ref_audio else None
            rows.append(
                {
                    "index": index,
                    "sample_id": sid,
                    "language": language,
                    "target_text": target_text,
                    "gen_text": normalize_text(target_text, text_cleanup),
                    "reference_text": str(obj.get("reference_text", "")).strip(),
                    "reference_audio": str(ref_audio) if ref_audio else "",
                    "wav_name": wav_name,
                    "output_wav": str((WAV_DIR / wav_name).absolute()),
                }
            )
            if max_samples > 0 and len(rows) >= max_samples:
                break
    if not rows:
        raise RuntimeError(f"No usable {language!r} prompt rows in {eval_jsonl}")
    return rows


def validate_prompt_references(rows: list[dict[str, Any]], eval_jsonl: Path) -> None:
    """Validate all conditioning inputs before starting any F5 runtime."""
    errors: list[str] = []
    for row in rows:
        sample_id = str(row.get("sample_id") or "<unknown>")
        reference_audio_value = str(row.get("reference_audio") or "").strip()
        reference_text = str(row.get("reference_text") or "").strip()
        if not reference_audio_value:
            errors.append(f"{sample_id}: reference_audio is empty")
        else:
            reference_audio = Path(reference_audio_value)
            valid, reason = validate_wav(reference_audio)
            if not valid:
                errors.append(
                    f"{sample_id}: invalid reference_audio {reference_audio} ({reason})"
                )
        if not reference_text:
            errors.append(f"{sample_id}: reference_text is empty")

    if errors:
        preview_limit = 20
        preview = errors[:preview_limit]
        if len(errors) > preview_limit:
            preview.append(f"... and {len(errors) - preview_limit} more error(s)")
        raise RuntimeError(
            f"F5-TTS prompt reference preflight failed for {eval_jsonl}: "
            + "; ".join(preview)
        )


def row_cost(row: dict[str, Any]) -> int:
    """Estimate synthesis work without relying on runtime-only tokenizers."""
    return max(1, len(str(row.get("gen_text") or "").strip()))


def split_shards(rows: list[dict[str, Any]], workers: int) -> list[list[dict[str, Any]]]:
    """Deterministically apply longest-processing-time greedy balancing."""
    worker_count = max(1, min(workers, max(len(rows), 1)))
    shards: list[list[dict[str, Any]]] = [[] for _ in range(worker_count)]
    costs = [0] * worker_count
    ranked = sorted(enumerate(rows), key=lambda item: (-row_cost(item[1]), item[0]))
    for _original_index, row in ranked:
        shard_index = min(range(worker_count), key=lambda index: (costs[index], index))
        shards[shard_index].append(row)
        costs[shard_index] += row_cost(row)
    for shard in shards:
        shard.sort(key=lambda row: int(row.get("index", 0)))
    return [shard for shard in shards if shard]


def split_text_chunks(text: str, max_chars: int, min_chars: int = 1) -> list[str]:
    """Split text within a character budget, preferring sentence punctuation."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text or max_chars <= 0 or len(text) <= max_chars:
        return [text] if text else []

    chunks: list[str] = []
    remaining = text
    punctuation = re.compile(r"[.!?;:,](?:[\"')\]]+)?\s+")
    while len(remaining) > max_chars:
        window = remaining[: max_chars + 1]
        candidates = [match.end() for match in punctuation.finditer(window) if match.end() <= max_chars]
        cut = max((value for value in candidates if value >= min_chars), default=0)
        if not cut:
            spaces = [match.start() for match in re.finditer(r"\s+", window)]
            cut = max((value for value in spaces if value >= min_chars), default=max_chars)
        chunk = remaining[:cut].strip()
        if not chunk:
            cut = max_chars
            chunk = remaining[:cut].strip()
        chunks.append(chunk)
        remaining = remaining[cut:].strip()
    if remaining:
        if chunks and len(remaining) < min_chars:
            chunks[-1] = f"{chunks[-1]} {remaining}"
        else:
            chunks.append(remaining)
    return chunks


def normalize_audio_segment(audio: Any) -> Any:
    """Return mono audio with time on its only axis."""
    import numpy as np  # type: ignore

    segment = np.asarray(audio)
    if segment.ndim == 1:
        return segment
    if segment.ndim == 2 and 1 in segment.shape:
        return segment.reshape(-1)
    raise ValueError(f"Unsupported F5-TTS waveform shape: {segment.shape}; expected mono audio")


def is_oom_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "out of memory",
        "cuda error: memory allocation",
        "cuda out of memory",
        "cublas_status_alloc_failed",
        "hip out of memory",
    )
    return any(marker in text for marker in markers)


def atomic_write_audio(
    path: Path,
    audio: Any,
    sample_rate: int,
    postprocess: Any | None = None,
) -> None:
    import soundfile as sf  # type: ignore

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.wav")
    try:
        sf.write(str(temporary), audio, sample_rate)
        if postprocess is not None:
            postprocess(str(temporary))
        valid, reason = validate_wav(temporary)
        if not valid:
            raise RuntimeError(f"Invalid temporary WAV {temporary}: {reason}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def add_f5_to_path(f5_root: Path) -> None:
    src = str((f5_root / "src").absolute())
    if src not in sys.path:
        sys.path.insert(0, src)


def load_f5_runtime(config: dict[str, Any]) -> dict[str, Any]:
    f5_root = Path(config["f5_root"])
    add_f5_to_path(f5_root)

    from cached_path import cached_path  # type: ignore
    from hydra.utils import get_class  # type: ignore
    from importlib.resources import files
    from omegaconf import OmegaConf  # type: ignore

    from f5_tts.infer.utils_infer import (  # type: ignore
        cfg_strength as default_cfg_strength,
        cross_fade_duration as default_cross_fade_duration,
        device as default_device,
        fix_duration as default_fix_duration,
        infer_process,
        load_model,
        load_vocoder,
        mel_spec_type as default_mel_spec_type,
        nfe_step as default_nfe_step,
        preprocess_ref_audio_text,
        remove_silence_for_generated_wav,
        speed as default_speed,
        sway_sampling_coef as default_sway_sampling_coef,
        target_rms as default_target_rms,
    )

    model_name = str(config["model"])
    model_cfg_path = str(config.get("model_cfg") or files("f5_tts").joinpath(f"configs/{model_name}.yaml"))
    model_cfg = OmegaConf.load(model_cfg_path)
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch

    vocoder_name = str(config.get("vocoder_name") or model_cfg.model.mel_spec.get("mel_spec_type", default_mel_spec_type))
    device = str(config.get("device") or default_device)

    if vocoder_name == "vocos":
        vocoder_local_path = "../checkpoints/vocos-mel-24khz"
    elif vocoder_name == "bigvgan":
        vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"
    else:
        raise ValueError(f"Unsupported vocoder: {vocoder_name}")

    vocoder = load_vocoder(
        vocoder_name=vocoder_name,
        is_local=parse_bool(config.get("load_vocoder_from_local"), False),
        local_path=vocoder_local_path,
        device=device,
    )

    repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"
    if model_name == "F5TTS_Base":
        if vocoder_name == "vocos":
            ckpt_step = 1200000
        elif vocoder_name == "bigvgan":
            model_name = "F5TTS_Base_bigvgan"
            ckpt_type = "pt"
    elif model_name == "E2TTS_Base":
        repo_name = "E2-TTS"
        ckpt_step = 1200000

    ckpt_file = str(config.get("ckpt_file") or "")
    vocab_file = str(config.get("vocab_file") or "")
    if not ckpt_file:
        ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{model_name}/model_{ckpt_step}.{ckpt_type}"))
    elif ckpt_file.startswith("hf://"):
        ckpt_file = str(cached_path(ckpt_file))
    if vocab_file.startswith("hf://"):
        vocab_file = str(cached_path(vocab_file))

    print(f"Using {model_name} on {device} with {vocoder_name}", flush=True)
    model = load_model(
        model_cls,
        model_arc,
        ckpt_file,
        mel_spec_type=vocoder_name,
        vocab_file=vocab_file,
        device=device,
    )

    return {
        "infer_process": infer_process,
        "preprocess_ref_audio_text": preprocess_ref_audio_text,
        "remove_silence_for_generated_wav": remove_silence_for_generated_wav,
        "model": model,
        "vocoder": vocoder,
        "mel_spec_type": vocoder_name,
        "target_rms": parse_float(config.get("target_rms"), default_target_rms),
        "cross_fade_duration": parse_float(config.get("cross_fade_duration"), default_cross_fade_duration),
        "nfe_step": parse_int(config.get("nfe_step"), default_nfe_step, min_value=1),
        "cfg_strength": parse_float(config.get("cfg_strength"), default_cfg_strength),
        "sway_sampling_coef": parse_float(config.get("sway_sampling_coef"), default_sway_sampling_coef),
        "speed": parse_float(config.get("speed"), default_speed),
        "fix_duration": config.get("fix_duration", default_fix_duration),
        "device": device,
    }


def run_worker(args: argparse.Namespace) -> int:
    config = json.loads(Path(args.worker_config).read_text(encoding="utf-8"))
    rows = read_jsonl(Path(args.shard_file))
    if not rows:
        return 0

    mkdir_inside(WAV_DIR)
    if "cuda" in str(config.get("device", "")).lower() and len(visible_cuda_devices()) == 1:
        # CUDA_VISIBLE_DEVICES remaps the assigned physical GPU to local index zero.
        config["device"] = "cuda"
    runtime = load_f5_runtime(config)
    resume = parse_bool(config.get("resume"), True)
    remove_silence = parse_bool(config.get("remove_silence"), False)

    generated = 0
    reused = 0
    total_chunks = 0
    oom_retries = 0
    worker_started = time.monotonic()
    for local_idx, row in enumerate(rows, 1):
        row_started = time.monotonic()
        out_wav = Path(str(row["output_wav"]))
        if resume and resume_matches(row, config):
            print(
                f"[worker {args.worker_index}] reuse {row['sample_id']} "
                f"({local_idx}/{len(rows)}; cost={row_cost(row)})",
                flush=True,
            )
            reused += 1
            continue
        if out_wav.exists():
            out_wav.unlink()
        fingerprint_path(out_wav).unlink(missing_ok=True)

        ref_audio = Path(str(row["reference_audio"]))
        ref_text = str(row["reference_text"])

        prepared_audio, prepared_text = runtime["preprocess_ref_audio_text"](str(ref_audio), ref_text)
        budget = parse_int(config.get("max_chunk_chars"), int(DEFAULTS["max_chunk_chars"]), min_value=1)
        minimum_budget = min(
            budget,
            parse_int(config.get("min_chunk_chars"), int(DEFAULTS["min_chunk_chars"]), min_value=1),
        )
        retries = 0
        while True:
            chunks = split_text_chunks(str(row["gen_text"]), budget, minimum_budget)
            print(
                f"[worker {args.worker_index}] synth {row['sample_id']} "
                f"({local_idx}/{len(rows)}; cost={row_cost(row)}; chunks={len(chunks)}; budget={budget})",
                flush=True,
            )
            segments: list[Any] = []
            sample_rate: int | None = None
            try:
                for chunk_index, chunk in enumerate(chunks, 1):
                    audio_segment, chunk_sample_rate, _spectrogram = runtime["infer_process"](
                        prepared_audio,
                        prepared_text,
                        chunk,
                        runtime["model"],
                        runtime["vocoder"],
                        mel_spec_type=runtime["mel_spec_type"],
                        target_rms=runtime["target_rms"],
                        cross_fade_duration=runtime["cross_fade_duration"],
                        nfe_step=runtime["nfe_step"],
                        cfg_strength=runtime["cfg_strength"],
                        sway_sampling_coef=runtime["sway_sampling_coef"],
                        speed=runtime["speed"],
                        fix_duration=runtime["fix_duration"],
                        device=runtime["device"],
                    )
                    if audio_segment is None:
                        raise RuntimeError(
                            f"F5-TTS returned no audio for {row['sample_id']} chunk {chunk_index}"
                        )
                    if sample_rate is not None and int(chunk_sample_rate) != sample_rate:
                        raise RuntimeError(
                            f"F5-TTS sample rate changed for {row['sample_id']}: "
                            f"{sample_rate} -> {chunk_sample_rate}"
                        )
                    sample_rate = int(chunk_sample_rate)
                    segments.append(normalize_audio_segment(audio_segment))
                break
            except Exception as exc:
                if not is_oom_error(exc) or budget <= minimum_budget:
                    raise
                retries += 1
                oom_retries += 1
                budget = max(minimum_budget, budget // 2)
                print(
                    f"[worker {args.worker_index}] OOM for {row['sample_id']}; "
                    f"retry={retries} smaller_budget={budget}",
                    flush=True,
                )
                segments.clear()
                audio_segment = None
                _spectrogram = None
                try:
                    import torch  # type: ignore

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except (ImportError, RuntimeError):
                    pass

        import numpy as np  # type: ignore

        final_audio = np.concatenate(segments) if len(segments) > 1 else segments[0]
        postprocess = runtime["remove_silence_for_generated_wav"] if remove_silence else None
        atomic_write_audio(out_wav, final_audio, int(sample_rate), postprocess)
        fingerprint_path(out_wav).write_text(synthesis_fingerprint(row, config) + "\n", encoding="utf-8")

        valid, reason = validate_wav(out_wav)
        if not valid:
            raise RuntimeError(f"Generated invalid WAV for {row['sample_id']}: {out_wav} ({reason})")
        generated += 1
        total_chunks += len(chunks)
        print(
            f"[worker {args.worker_index}] done {row['sample_id']} "
            f"chunks={len(chunks)} retries={retries} elapsed={time.monotonic() - row_started:.2f}s",
            flush=True,
        )

    done_path = BATCH_DIR / f"worker_{args.worker_index}.done.json"
    write_json(
        done_path,
        {
            "worker_index": args.worker_index,
            "generated": generated,
            "reused": reused,
            "chunks": total_chunks,
            "oom_retries": oom_retries,
            "estimated_cost": sum(row_cost(row) for row in rows),
            "elapsed_seconds": round(time.monotonic() - worker_started, 3),
        },
    )
    return 0


def terminate_process_group(proc: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        time.sleep(5)
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def launch_workers(
    *,
    config_path: Path,
    shard_paths: list[Path],
    timeout: int,
    device: str,
) -> None:
    visible = visible_cuda_devices()
    script = Path(__file__).resolve()
    procs: list[tuple[int, subprocess.Popen[Any], Path]] = []

    for worker_index, shard_path in enumerate(shard_paths):
        log_path = LOG_DIR / f"f5tts_batch_worker_{worker_index}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        if "cuda" in device.lower() and visible:
            env["CUDA_VISIBLE_DEVICES"] = visible[worker_index % len(visible)]
        cmd = [
            sys.executable,
            str(script),
            "--worker",
            "--worker-config",
            str(config_path),
            "--shard-file",
            str(shard_path),
            "--worker-index",
            str(worker_index),
        ]
        log = log_path.open("ab")
        log.write(("\n\n===== WORKER %d %s =====\n" % (worker_index, time.strftime("%Y-%m-%d %H:%M:%S"))).encode())
        log.write((" ".join(cmd) + "\n").encode(errors="replace"))
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(WORKSPACE),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
        )
        log.close()
        procs.append((worker_index, proc, log_path))

    deadline = time.monotonic() + timeout
    failed: list[str] = []
    try:
        for worker_index, proc, log_path in procs:
            remaining = max(1, int(deadline - time.monotonic()))
            try:
                rc = proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                failed.append(f"worker {worker_index} timed out; log={log_path}")
                continue
            if rc != 0:
                failed.append(f"worker {worker_index} exited {rc}; log={log_path}")
    finally:
        for _worker_index, proc, _log_path in procs:
            if proc.poll() is None:
                terminate_process_group(proc)

    if failed:
        raise RuntimeError("; ".join(failed))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f5-root", default=env_str("SURE_TTS_ROOT", str(DEFAULTS["root"])))
    parser.add_argument("--eval-data", default=env_str("SURE_TTS_EVAL_DATA", str(DEFAULTS["eval_data"])))
    parser.add_argument("--language", default=env_str("SURE_TTS_LANGUAGE", str(DEFAULTS["language"])))
    parser.add_argument("--max-samples", type=int, default=parse_int(env_str("SURE_TTS_MAX_SAMPLES", str(DEFAULTS["max_samples"])), 4, 1))
    parser.add_argument("--model", default=env_str("SURE_TTS_MODEL", str(DEFAULTS["model"])))
    parser.add_argument("--ckpt-file", default=env_str("SURE_TTS_CKPT_FILE", ""))
    parser.add_argument("--vocab-file", default=env_str("SURE_TTS_VOCAB_FILE", ""))
    parser.add_argument("--model-cfg", default=env_str("SURE_TTS_MODEL_CFG", ""))
    parser.add_argument("--vocoder-name", default=env_str("SURE_TTS_VOCODER_NAME", ""))
    parser.add_argument("--load-vocoder-from-local", default=env_str("SURE_TTS_LOAD_VOCODER_FROM_LOCAL", "0"))
    parser.add_argument("--device", default=env_str("SURE_TTS_DEVICE", "cuda"))
    parser.add_argument("--workers", default=env_str("SURE_TTS_INFER_WORKERS", str(DEFAULTS["workers"])))
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=parse_int(
            env_str("SURE_TTS_MAX_CHUNK_CHARS", str(DEFAULTS["max_chunk_chars"])),
            int(DEFAULTS["max_chunk_chars"]),
            1,
        ),
    )
    parser.add_argument(
        "--min-chunk-chars",
        type=int,
        default=parse_int(
            env_str("SURE_TTS_MIN_CHUNK_CHARS", str(DEFAULTS["min_chunk_chars"])),
            int(DEFAULTS["min_chunk_chars"]),
            1,
        ),
    )
    parser.add_argument("--nfe-step", type=int, default=parse_int(env_str("SURE_TTS_NFE_STEP", str(DEFAULTS["nfe_step"])), 32, 1))
    parser.add_argument("--cfg-strength", type=float, default=parse_float(env_str("SURE_TTS_CFG_STRENGTH", str(DEFAULTS["cfg_strength"])), 2.0))
    parser.add_argument("--sway-sampling-coef", type=float, default=parse_float(env_str("SURE_TTS_SWAY_SAMPLING_COEF", str(DEFAULTS["sway_sampling_coef"])), -1.0))
    parser.add_argument("--speed", type=float, default=parse_float(env_str("SURE_TTS_SPEED", str(DEFAULTS["speed"])), 1.0))
    parser.add_argument("--remove-silence", default=env_str("SURE_TTS_REMOVE_SILENCE", "0"))
    parser.add_argument("--text-cleanup", choices=["none", "typography", "conservative"], default=env_str("SURE_TTS_TEXT_CLEANUP", "none"))
    parser.add_argument(
        "--candidate-type",
        choices=["auto", "inference", "fine_tune", "arch", "training", "architecture"],
        default="auto",
    )
    parser.add_argument("--idea-text", default=env_str("SURE_CANDIDATE_IDEA_TEXT", "F5-TTS batch inference candidate"))
    parser.add_argument("--training-action", default=env_str("SURE_TTS_TRAIN_ACTION", "no_train"))
    parser.add_argument("--arch-action", default=env_str("SURE_TTS_ARCH_ACTION", "no_arch"))
    parser.add_argument("--resume", default=env_str("SURE_TTS_INFER_RESUME", "1"))
    parser.add_argument("--timeout", type=int, default=parse_int(env_str("SURE_RUN_TIMEOUT", str(DEFAULTS["run_timeout"])), 21600, 60))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-config", help=argparse.SUPPRESS)
    parser.add_argument("--shard-file", help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def infer_candidate_type(args: argparse.Namespace) -> str:
    if args.candidate_type != "auto":
        if args.candidate_type == "architecture":
            return "arch"
        if args.candidate_type == "training":
            return "fine_tune"
        return args.candidate_type
    if str(args.arch_action).strip() not in {"", "no_arch"}:
        return "arch"
    if str(args.training_action).strip() not in {"", "no_train"}:
        return "fine_tune"
    return "inference"


def validate_arch_change_record_before_infer(args: argparse.Namespace) -> None:
    uses_arch_artifact = any(
        "f5tts_arch" in str(value).replace("\\", "/")
        for value in (args.model_cfg, args.ckpt_file)
    )
    existing = read_existing_candidate_changes()
    existing_type = str(existing.get("candidate_type", "")).strip().lower()
    should_validate = (
        infer_candidate_type(args) == "arch"
        or existing_type in {"arch", "architecture", "structure"}
        or uses_arch_artifact
    )
    if not should_validate:
        return
    errors = validate_arch_candidate_changes(WORKSPACE)
    if errors:
        raise RuntimeError(
            "F5-TTS arch candidate change record is invalid before inference: "
            + "; ".join(errors)
        )


def write_candidate_changes(args: argparse.Namespace, rows: list[dict[str, Any]], workers: int) -> None:
    existing = read_existing_candidate_changes()
    existing_defaults = existing.get("defaults") if isinstance(existing.get("defaults"), dict) else {}
    existing_diff = existing.get("diff_from_defaults") if isinstance(existing.get("diff_from_defaults"), dict) else {}
    existing_arch_config = existing.get("arch_config") if isinstance(existing.get("arch_config"), dict) else {}
    existing_training_config = (
        existing.get("training_config") if isinstance(existing.get("training_config"), dict) else {}
    )
    existing_arch_diff = (
        existing_diff.get("arch_config") if isinstance(existing_diff.get("arch_config"), dict) else {}
    )
    uses_arch_artifact = any(
        "f5tts_arch" in str(value).replace("\\", "/")
        for value in (args.model_cfg, args.ckpt_file)
    )
    preserve_arch = (
        str(existing.get("candidate_type", "")).strip().lower() in {"arch", "architecture"}
        or bool(existing_arch_diff)
        or uses_arch_artifact
    )

    defaults = {
        "nfe_step": parse_int(env_str("SURE_TTS_NFE_STEP", str(DEFAULTS["nfe_step"])), 32, 1),
        "cfg_strength": parse_float(env_str("SURE_TTS_CFG_STRENGTH", str(DEFAULTS["cfg_strength"])), 2.0),
        "sway_sampling_coef": parse_float(env_str("SURE_TTS_SWAY_SAMPLING_COEF", str(DEFAULTS["sway_sampling_coef"])), -1.0),
        "speed": parse_float(env_str("SURE_TTS_SPEED", str(DEFAULTS["speed"])), 1.0),
        "remove_silence": parse_bool(env_str("SURE_TTS_REMOVE_SILENCE", "0"), False),
        "text_cleanup": env_str("SURE_TTS_TEXT_CLEANUP", "none"),
        "max_chunk_chars": parse_int(
            env_str("SURE_TTS_MAX_CHUNK_CHARS", str(DEFAULTS["max_chunk_chars"])),
            int(DEFAULTS["max_chunk_chars"]),
            1,
        ),
        "min_chunk_chars": parse_int(
            env_str("SURE_TTS_MIN_CHUNK_CHARS", str(DEFAULTS["min_chunk_chars"])),
            int(DEFAULTS["min_chunk_chars"]),
            1,
        ),
    }
    used = {
        "nfe_step": int(args.nfe_step),
        "cfg_strength": float(args.cfg_strength),
        "sway_sampling_coef": float(args.sway_sampling_coef),
        "speed": float(args.speed),
        "remove_silence": parse_bool(args.remove_silence, False),
        "text_cleanup": args.text_cleanup,
        "max_chunk_chars": int(getattr(args, "max_chunk_chars", DEFAULTS["max_chunk_chars"])),
        "min_chunk_chars": int(getattr(args, "min_chunk_chars", DEFAULTS["min_chunk_chars"])),
    }
    diff = {
        key: {"default": defaults[key], "used": value}
        for key, value in used.items()
        if defaults.get(key) != value
    }
    changed_fields = [f"inference_config.{key}" for key in diff]
    if rows and rows[0].get("reference_audio"):
        changed_fields.extend(["conditioning.reference_audio_source", "conditioning.reference_text_source"])
    if preserve_arch and isinstance(existing.get("changed_fields"), list):
        for field in existing["changed_fields"]:
            field_text = str(field)
            if field_text.startswith("arch_config.") and field_text not in changed_fields:
                changed_fields.append(field_text)

    candidate_type = "arch" if preserve_arch else infer_candidate_type(args)
    arch_config = existing_arch_config if preserve_arch and existing_arch_config else {"action": args.arch_action}
    training_config = (
        existing_training_config
        if preserve_arch and existing_training_config
        else {"action": args.training_action}
    )
    defaults_payload: dict[str, Any] = {"inference_config": defaults}
    existing_arch_defaults = (
        existing_defaults.get("arch_config") if isinstance(existing_defaults.get("arch_config"), dict) else {}
    )
    if preserve_arch and existing_arch_defaults:
        defaults_payload["arch_config"] = existing_arch_defaults
    diff_payload: dict[str, Any] = {"inference_config": diff}
    if preserve_arch:
        diff_payload["arch_config"] = existing_arch_diff

    produced_artifacts = (
        dict(existing.get("produced_artifacts"))
        if isinstance(existing.get("produced_artifacts"), dict)
        else {}
    )
    produced_artifacts.update(
        {
            "samples_jsonl": "artifacts/samples.jsonl",
            "wav_dir": "artifacts/wavs",
            "worker_logs": "working/logs/f5tts_batch_worker_*.log",
        }
    )

    payload = {
        "candidate_type": candidate_type,
        "idea_text": args.idea_text,
        "changed_fields": changed_fields,
        "arch_config": arch_config,
        "training_config": training_config,
        "inference_config": {
            **used,
            "model": args.model,
            "ckpt_file": args.ckpt_file,
            "vocab_file": args.vocab_file,
            "model_cfg": args.model_cfg,
            "workers": workers,
            "max_samples": len(rows),
            "conditioning": "row_reference_audio_and_text",
        },
        "defaults": defaults_payload,
        "diff_from_defaults": diff_payload,
        "produced_artifacts": produced_artifacts,
    }
    write_json(ARTIFACTS / "candidate_changes.json", payload)


def run_main(args: argparse.Namespace) -> int:
    mkdir_inside(ARTIFACTS)
    mkdir_inside(WAV_DIR)
    mkdir_inside(WORKING)
    mkdir_inside(BATCH_DIR)
    mkdir_inside(LOG_DIR)
    validate_arch_change_record_before_infer(args)

    f5_root = resolve_read_path(args.f5_root)
    eval_data = resolve_read_path(args.eval_data)
    rows = load_prompts(eval_data, args.max_samples, args.language, args.text_cleanup)
    validate_prompt_references(rows, eval_data)
    config = {
        "f5_root": str(f5_root),
        "model": args.model,
        "ckpt_file": args.ckpt_file,
        "vocab_file": args.vocab_file,
        "model_cfg": args.model_cfg,
        "vocoder_name": args.vocoder_name,
        "load_vocoder_from_local": args.load_vocoder_from_local,
        "device": args.device,
        "nfe_step": args.nfe_step,
        "cfg_strength": args.cfg_strength,
        "sway_sampling_coef": args.sway_sampling_coef,
        "speed": args.speed,
        "remove_silence": args.remove_silence,
        "resume": args.resume,
        "max_chunk_chars": args.max_chunk_chars,
        "min_chunk_chars": args.min_chunk_chars,
    }
    resume = parse_bool(args.resume, True)
    pending_rows: list[dict[str, Any]] = []
    reused_rows = 0
    for row in rows:
        if resume and resume_matches(row, config):
            reused_rows += 1
        else:
            pending_rows.append(row)
    workers = choose_worker_count(args.workers, len(pending_rows), args.device)

    if args.dry_run:
        print(
            f"Would synthesize {len(pending_rows)} of {len(rows)} prompts with {workers} worker(s); "
            f"resume_reused={reused_rows} estimated_cost={sum(row_cost(row) for row in pending_rows)}"
        )
        return 0

    config_path = BATCH_DIR / "config.json"
    write_json(config_path, config)

    shards = split_shards(pending_rows, workers) if pending_rows else []
    shard_paths: list[Path] = []
    shard_diagnostics: list[dict[str, Any]] = []
    for index, shard in enumerate(shards):
        shard_path = BATCH_DIR / f"shard_{index}.jsonl"
        write_jsonl(shard_path, shard)
        shard_paths.append(shard_path)
        shard_diagnostics.append(
            {
                "worker_index": index,
                "row_count": len(shard),
                "estimated_cost": sum(row_cost(row) for row in shard),
                "sample_ids": [row["sample_id"] for row in shard],
            }
        )
    write_json(
        BATCH_DIR / "diagnostics.json",
        {
            "total_rows": len(rows),
            "pending_rows": len(pending_rows),
            "resume_reused": reused_rows,
            "visible_cuda_devices": visible_cuda_devices(),
            "requested_workers": args.workers,
            "launched_workers": len(shard_paths),
            "max_chunk_chars": args.max_chunk_chars,
            "min_chunk_chars": args.min_chunk_chars,
            "shards": shard_diagnostics,
        },
    )

    if shard_paths:
        launch_workers(config_path=config_path, shard_paths=shard_paths, timeout=args.timeout, device=args.device)

    sample_rows: list[dict[str, Any]] = []
    for row in rows:
        wav_path = Path(row["output_wav"])
        valid, reason = validate_wav(wav_path)
        if not valid:
            raise RuntimeError(f"Invalid artifact WAV {wav_path}: {reason}")
        ref_audio = Path(str(row["reference_audio"]))
        sample_rows.append(
            {
                "sample_id": row["sample_id"],
                "prediction_audio": "wavs/" + row["wav_name"],
                "reference_text": row["target_text"],
                "reference_audio": rel_to_artifacts(ref_audio),
                "language": row["language"],
            }
        )

    write_jsonl(ARTIFACTS / "samples.jsonl", sample_rows, atomic=True)
    write_candidate_changes(args, rows, len(shard_paths))
    print(f"Generated {len(sample_rows)} TTS samples with {len(shard_paths)} worker(s)")
    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.worker:
        if not args.worker_config or not args.shard_file:
            parser.error("--worker requires --worker-config and --shard-file")
        return run_worker(args)
    return run_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
