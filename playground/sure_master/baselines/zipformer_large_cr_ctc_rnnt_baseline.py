from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from playground.sure_master.baselines.asr_profiles import (
    AsrDatasetProfile,
    AsrRecipeProfile,
    current_asr_dataset_profile,
    current_asr_recipe_profile,
    normalize_asr_cut_id,
    parse_eval_splits_for_profile,
    recipe_arch_args,
    recipe_decode_data_args,
    recipe_train_data_args,
    recipe_train_only_args,
)


WORKSPACE = Path.cwd()
RECIPE_DIR = Path("base_model/recipe")
ICEFALL_ROOT = Path("base_model/root")
DATA_DIR = Path("base_model/data")
ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = Path("models/zipformer_large_cr_ctc_rnnt")
WORKING_DIR = Path("working/zipformer_large_cr_ctc_rnnt")

BASELINE_EPOCH = 50
BASELINE_AVG = 26
BASELINE_TRAIN_MAX_DURATION = 1400
BASELINE_DECODE_MAX_DURATION = 300
BASELINE_WORLD_SIZE = 2
DEFAULT_ASR_EVAL_SPLITS = "test-clean,test-other"
PRETRAINED_CHECKPOINT_NAME = "pretrained.pt"
DEFAULT_BPE_MODEL = Path("data/lang_bpe_500/bpe.model")
OFFICIAL_LANG_BPE_MODEL = Path("data/lang_bpe_500/bpe.model")
OFFICIAL_MODEL_ID = (
    "Zengwei/icefall-asr-librispeech-zipformer-large-transducer-with-CR-CTC-20241019"
)
OFFICIAL_MODEL_DIR_NAME = OFFICIAL_MODEL_ID.rsplit("/", 1)[1]
OFFICIAL_MODEL_URL = (
    "https://huggingface.co/"
    "Zengwei/icefall-asr-librispeech-zipformer-large-transducer-with-CR-CTC-20241019"
)

LARGE_CR_CTC_RNNT_ARCH_ARGS = [
    "--use-cr-ctc",
    "1",
    "--use-ctc",
    "1",
    "--use-transducer",
    "1",
    "--use-attention-decoder",
    "0",
    "--num-encoder-layers",
    "2,2,4,5,4,2",
    "--feedforward-dim",
    "512,768,1536,2048,1536,768",
    "--encoder-dim",
    "192,256,512,768,512,256",
    "--encoder-unmasked-dim",
    "192,192,256,320,256,192",
]

LARGE_CR_CTC_RNNT_TRAIN_ONLY_ARGS = [
    "--ctc-loss-scale",
    "0.1",
    "--enable-spec-aug",
    "0",
    "--cr-loss-scale",
    "0.02",
    "--time-mask-ratio",
    "2.5",
]

LARGE_CR_CTC_RNNT_DATA_ARGS = [
    "--full-libri",
    "1",
]

LARGE_CR_CTC_RNNT_ARGS = (
    LARGE_CR_CTC_RNNT_ARCH_ARGS
    + LARGE_CR_CTC_RNNT_TRAIN_ONLY_ARGS
    + LARGE_CR_CTC_RNNT_DATA_ARGS
)
LARGE_CR_CTC_RNNT_DECODE_ARGS = (
    LARGE_CR_CTC_RNNT_ARCH_ARGS
    + LARGE_CR_CTC_RNNT_DATA_ARGS
)


def dataset_profile() -> AsrDatasetProfile:
    return current_asr_dataset_profile()


def recipe_profile() -> AsrRecipeProfile:
    return current_asr_recipe_profile(dataset_profile())


def default_eval_splits_csv() -> str:
    return ",".join(dataset_profile().default_eval_splits)


def default_bpe_model() -> Path:
    return recipe_profile().default_bpe_model


def official_bpe_model_path() -> Path | None:
    return recipe_profile().official_bpe_model


def official_model_id() -> str | None:
    return recipe_profile().official_model_id


def official_model_url() -> str | None:
    return recipe_profile().official_model_url


def official_model_dir_name() -> str | None:
    model_id = official_model_id()
    if not model_id:
        return None
    return model_id.rsplit("/", 1)[1]


def default_checkpoint_dir() -> Path:
    return Path(recipe_profile().default_checkpoint_dir_env or "base_model/checkpoints/zipformer")


def large_cr_ctc_rnnt_train_args() -> list[str]:
    profile = recipe_profile()
    return recipe_arch_args(profile) + recipe_train_only_args(profile) + recipe_train_data_args(profile)


def large_cr_ctc_rnnt_decode_args() -> list[str]:
    profile = recipe_profile()
    return recipe_arch_args(profile) + recipe_decode_data_args(profile)


class CommandFailedError(RuntimeError):
    def __init__(self, name: str, return_code: int, log_path: Path):
        self.name = name
        self.return_code = return_code
        self.log_path = log_path
        super().__init__(
            f"{name} failed with exit code {return_code}; see {log_path}"
        )


class CommandTimedOutError(TimeoutError):
    def __init__(self, name: str, timeout: int, log_path: Path):
        self.name = name
        self.timeout = timeout
        self.log_path = log_path
        super().__init__(f"{name} timed out after {timeout}s; see {log_path}")


def truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def int_env(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, str(default))))
    except ValueError:
        return default


def timeout_env(name: str, default: int) -> int | None:
    value = int_env(name, default)
    return None if value == 0 else value


def train_duration_floor() -> int:
    return max(
        1,
        int_env("SURE_TRAIN_DURATION_MIN", int_env("SURE_DURATION_AUTOTUNE_MIN", 100)),
        int_env("SURE_DURATION_AUTOTUNE_MIN", 100),
    )


def max_duration(
    default: int,
    cap: int | None = None,
    train_args: list[str] | None = None,
    enforce_floor: bool = False,
) -> int:
    raw = os.environ.get("SURE_MAX_DURATION", str(default)).strip()
    if raw.lower() == "auto":
        value = resolve_auto_max_duration(train_args or [], default)
    else:
        try:
            value = int(float(raw))
        except ValueError:
            value = default
    if cap is not None:
        value = min(value, cap)
    floor = train_duration_floor() if enforce_floor else 1
    return max(floor, value)


def visible_gpu_count() -> int | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return None
    devices = [item for item in visible.split(",") if item.strip()]
    return len(devices)


def world_size() -> int:
    requested = int_env(
        "SURE_BASELINE_WORLD_SIZE",
        int_env("ASR_WORLD_SIZE", BASELINE_WORLD_SIZE),
    )
    visible = visible_gpu_count()
    if visible is None:
        return max(1, requested)
    if visible <= 0:
        return 1
    return max(1, min(requested, visible))


def bounded_train_max_duration() -> int:
    resolved_max_duration = max_duration(
        BASELINE_TRAIN_MAX_DURATION,
        BASELINE_TRAIN_MAX_DURATION,
        LARGE_CR_CTC_RNNT_ARCH_ARGS + LARGE_CR_CTC_RNNT_TRAIN_ONLY_ARGS,
        enforce_floor=True,
    )
    configured_max_duration = int_env(
        "SURE_BASELINE_TRAIN_MAX_DURATION",
        resolved_max_duration,
    )
    return max(train_duration_floor(), min(configured_max_duration, resolved_max_duration))


def ensure_workspace() -> None:
    for path in (ARTIFACTS_DIR, MODELS_DIR, WORKING_DIR):
        path.mkdir(parents=True, exist_ok=True)
    if not RECIPE_DIR.exists():
        raise FileNotFoundError(f"Missing required Zipformer recipe symlink: {RECIPE_DIR}")
    if not ICEFALL_ROOT.exists():
        raise FileNotFoundError(f"Missing required icefall root symlink: {ICEFALL_ROOT}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Missing required icefall data symlink: {DATA_DIR}")
    data_link = Path("data")
    if not data_link.exists():
        data_link.symlink_to(DATA_DIR, target_is_directory=True)


def command_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    workspace_pythonpath = os.pathsep.join(
        [str(ICEFALL_ROOT), str(RECIPE_DIR), existing_pythonpath]
    ).strip(os.pathsep)
    env["PYTHONPATH"] = workspace_pythonpath
    env.setdefault("PYTHONUNBUFFERED", "1")
    if extra_env:
        env.update(extra_env)
    return env


def run_command(
    name: str,
    command: list[str],
    timeout: int | None = None,
    extra_env: dict[str, str] | None = None,
) -> Path:
    log_path = WORKING_DIR / f"{name}.log"
    rendered = " ".join(shlex.quote(part) for part in command)
    print(f"[baseline] running {name}: {rendered}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {rendered}\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=str(WORKSPACE),
            env=command_env(extra_env),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise CommandTimedOutError(name, timeout, log_path)
    if return_code != 0:
        raise CommandFailedError(name, return_code, log_path)
    return log_path


OOM_LOG_PATTERNS = (
    "cuda out of memory",
    "torch.cuda.outofmemoryerror",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
)


def is_oom_failure(log_path: Path) -> bool:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False
    return any(pattern in text for pattern in OOM_LOG_PATTERNS)


def duration_retry_sequence(start: int) -> list[int]:
    floor = train_duration_floor()
    start = max(floor, start)
    if not truthy(os.environ.get("SURE_TRAIN_DURATION_RETRY"), default=True):
        return [start]

    raw_values = os.environ.get("SURE_TRAIN_DURATION_RETRY_VALUES", "").strip()
    if raw_values:
        values: list[int] = [start]
        for item in raw_values.replace(",", " ").split():
            try:
                duration = int(float(item))
            except ValueError:
                continue
            if floor <= duration < start:
                values.append(duration)
        return sorted(set(values), reverse=True)

    step = int_env(
        "SURE_TRAIN_DURATION_RETRY_STEP",
        int_env("SURE_DURATION_AUTOTUNE_STEP", 100),
    )
    step = max(1, step)

    values = list(range(start, floor - 1, -step))
    if values[-1] != floor:
        values.append(floor)
    return values


def resolve_auto_max_duration(train_args: list[str], default: int) -> int:
    helper = Path(os.environ.get("SURE_RUNTIME_ENV_HELPER", ""))
    if not helper.is_file():
        raise RuntimeError(
            "SURE_MAX_DURATION=auto but SURE_RUNTIME_ENV_HELPER is not set "
            "to a readable runtime_env.py"
        )

    recipe_dir = (WORKSPACE / RECIPE_DIR).resolve()
    extra_env = {
        "SURE_DURATION_TRAIN_ARGS_JSON": json.dumps([str(item) for item in train_args]),
        "SURE_DURATION_RECIPE_DIR": os.environ.get("SURE_DURATION_RECIPE_DIR", str(RECIPE_DIR)),
        "SURE_DURATION_RECIPE_KEY": os.environ.get("SURE_DURATION_RECIPE_KEY", str(recipe_dir)),
        "SURE_DURATION_CANDIDATE_KEY": os.environ.get(
            "SURE_DURATION_CANDIDATE_KEY",
            architecture_signature(),
        ),
    }
    log_path = run_command(
        "duration_autotune",
        [sys.executable, str(helper), "resolve-max-duration"],
        timeout=int_env("SURE_DURATION_AUTOTUNE_TIMEOUT", 7200),
        extra_env=extra_env,
    )
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as exc:
        raise RuntimeError(f"Could not read duration autotune log: {log_path}") from exc
    for line in reversed(lines):
        value = line.strip()
        if re.fullmatch(r"\d+", value):
            parsed = int(value)
            if parsed > 0:
                return parsed
    raise RuntimeError(
        "Duration helper succeeded but did not finish with a standalone "
        f"positive integer; see {log_path}"
    )


def architecture_signature() -> str:
    profile = recipe_profile()
    payload = {
        "dataset": profile.dataset,
        "recipe": profile.recipe_family,
        "recipe_profile": profile.name,
        "train_args": large_cr_ctc_rnnt_train_args(),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def parse_max_memory_mb(log_path: Path) -> int | None:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    matches = re.findall(
        r"(?:max(?:imum)? memory allocated(?: so far)?(?: is)?|max_memory_allocated)"
        r"[^0-9]{0,32}([0-9]+(?:\.[0-9]+)?)\s*(?:mib|mb)?",
        text,
        flags=re.IGNORECASE,
    )
    if not matches:
        return None
    return int(float(matches[-1]))


def write_resource_profile(
    *,
    status: str,
    attempts: list[dict[str, object]],
    selected_duration: int | None,
    train_epochs: int,
    fp16: str,
) -> None:
    profile = {
        "status": status,
        "timestamp": time.time(),
        "recipe": "zipformer_large_cr_ctc_rnnt",
        "architecture_signature": architecture_signature(),
        "world_size": world_size(),
        "use_fp16": fp16 == "1",
        "train_epochs": train_epochs,
        "selected_max_duration": selected_duration,
        "attempts": attempts,
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = ARTIFACTS_DIR / "resource_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    with (WORKING_DIR / "resource_profile.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(profile, sort_keys=True) + "\n")


def checkpoint_source_dir() -> Path:
    source = os.environ.get("SURE_BASELINE_CHECKPOINT_DIR", "").strip()
    source_dir = Path(source) if source else default_checkpoint_dir()
    if source_dir.exists() and not any(source_dir.glob("*.pt")):
        exp_dir = source_dir / "exp"
        if exp_dir.exists() and any(exp_dir.glob("*.pt")):
            return exp_dir
    return source_dir


def has_configured_checkpoint_source() -> bool:
    return bool(os.environ.get("SURE_BASELINE_CHECKPOINT_DIR", "").strip())


def pretrained_checkpoint_source() -> Path:
    source = os.environ.get("SURE_BASELINE_PRETRAINED_CHECKPOINT", "").strip()
    if source:
        return Path(source)
    return checkpoint_source_dir() / PRETRAINED_CHECKPOINT_NAME


def use_pretrained_checkpoint() -> bool:
    raw = os.environ.get("SURE_BASELINE_USE_PRETRAINED")
    if raw is not None:
        return truthy(raw, default=False)
    return pretrained_checkpoint_source().is_file() or (MODELS_DIR / PRETRAINED_CHECKPOINT_NAME).is_file()


def checkpoint_package_candidates() -> list[Path]:
    raw_sources = [
        os.environ.get("SURE_BASELINE_CHECKPOINT_DIR", "").strip(),
        os.environ.get("SURE_BASELINE_PRETRAINED_CHECKPOINT", "").strip(),
    ]
    candidates: list[Path] = []

    for raw in raw_sources:
        if raw:
            path = Path(raw)
            candidates.extend([path, path.parent])
            if path.name == "exp":
                candidates.append(path.parent)

    source_dir = checkpoint_source_dir()
    candidates.extend([source_dir, source_dir.parent])

    pretrained_source = pretrained_checkpoint_source()
    candidates.extend([pretrained_source.parent, pretrained_source.parent.parent])

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def official_package_bpe_model() -> Path | None:
    official_bpe = official_bpe_model_path()
    if official_bpe is None:
        return None
    for package_dir in checkpoint_package_candidates():
        candidate = package_dir / official_bpe
        if candidate.is_file():
            return candidate
    return None


def checkpoint_source_looks_official() -> bool:
    dir_name = official_model_dir_name()
    if not dir_name:
        return False
    paths = checkpoint_package_candidates()
    return any(dir_name in path.parts for path in paths)


def decode_bpe_model_path() -> Path:
    configured = os.environ.get("SURE_BASELINE_BPE_MODEL", "").strip()
    if configured:
        return Path(configured)

    official_bpe = official_package_bpe_model()
    if official_bpe is not None:
        return official_bpe

    return default_bpe_model()


def validate_decode_bpe_model() -> None:
    bpe_model = decode_bpe_model_path()
    configured_bpe_model = os.environ.get("SURE_BASELINE_BPE_MODEL", "").strip()
    official_bpe = official_bpe_model_path()
    if checkpoint_source_looks_official() and official_bpe is not None and not configured_bpe_model:
        packaged_bpe = official_package_bpe_model()
        if packaged_bpe is None:
            raise FileNotFoundError(
                f"Recipe profile {recipe_profile().name!r} uses an official checkpoint "
                f"source that requires the matching {official_bpe}. Download the "
                "complete package, not only exp/*.pt, or set SURE_BASELINE_BPE_MODEL "
                "explicitly."
            )

    if not bpe_model.is_file():
        raise FileNotFoundError(
            "Missing Zipformer BPE model for decoding: "
            f"{bpe_model}. Dataset profile={dataset_profile().name}, "
            f"recipe profile={recipe_profile().name}. Set SURE_BASELINE_BPE_MODEL "
            "or provide the recipe's data/lang directory in the workspace."
        )


def decode_uses_averaged_model(epoch: int) -> bool:
    if use_pretrained_checkpoint():
        return False
    raw = os.environ.get("SURE_BASELINE_USE_AVERAGED_MODEL")
    if raw is not None:
        return truthy(raw, default=False)
    target_epoch = int_env("SURE_BASELINE_EPOCH", BASELINE_EPOCH)
    return epoch >= target_epoch


def decode_avg(epoch: int) -> int:
    if use_pretrained_checkpoint():
        return 1
    if decode_uses_averaged_model(epoch):
        return int_env("SURE_BASELINE_AVG", BASELINE_AVG)
    return 1


def required_decode_checkpoints(epoch: int) -> list[Path]:
    avg = max(1, decode_avg(epoch))
    if decode_uses_averaged_model(epoch):
        start = epoch - avg
        if start < 1:
            raise ValueError(
                f"Invalid averaged-model decode range: epoch={epoch}, avg={avg}; "
                "icefall requires epoch - avg >= 1."
            )
        return [MODELS_DIR / f"epoch-{start}.pt", MODELS_DIR / f"epoch-{epoch}.pt"]
    if avg == 1:
        return [MODELS_DIR / f"epoch-{epoch}.pt"]
    start = max(1, epoch - avg + 1)
    return [MODELS_DIR / f"epoch-{index}.pt" for index in range(start, epoch + 1)]


def missing_decode_checkpoints(epoch: int) -> list[Path]:
    return [path for path in required_decode_checkpoints(epoch) if not path.is_file()]


def validate_decode_checkpoints(epoch: int) -> None:
    missing = missing_decode_checkpoints(epoch)
    if not missing:
        validate_pretrained_checkpoint_mapping(epoch)
        return
    missing_text = ", ".join(str(path) for path in missing)
    raise FileNotFoundError(
        "Missing checkpoint(s) required for Zipformer baseline decoding: "
        f"{missing_text}. For the official epoch-50 avg-26 averaged model, "
        "download the full Hugging Face checkpoint package and point "
        "SURE_BASELINE_CHECKPOINT_DIR to its exp/ directory."
    )


def validate_pretrained_checkpoint_mapping(epoch: int) -> None:
    if not use_pretrained_checkpoint():
        return

    target = MODELS_DIR / f"epoch-{epoch}.pt"
    if target.is_symlink():
        raise RuntimeError(
            "Pretrained Zipformer decode requires a local copied checkpoint, "
            f"but {target} is a symlink. Remove it and copy {PRETRAINED_CHECKPOINT_NAME} "
            "to the target epoch checkpoint."
        )

    source = pretrained_checkpoint_source()
    if not source.is_file():
        source = MODELS_DIR / PRETRAINED_CHECKPOINT_NAME
    if not source.is_file() or not target.is_file():
        return

    source_size = source.stat().st_size
    target_size = target.stat().st_size
    if source_size != target_size:
        raise RuntimeError(
            "Pretrained Zipformer decode checkpoint mismatch: "
            f"{target} has size {target_size}, but {source} has size {source_size}. "
            f"In SURE_BASELINE_USE_PRETRAINED=1 mode, {target.name} must be a "
            f"copy of {PRETRAINED_CHECKPOINT_NAME}, not the raw official epoch checkpoint."
        )


def write_official_baseline_record(epoch: int, elapsed_seconds: float) -> None:
    official_checkpoint = epoch >= int_env("SURE_BASELINE_EPOCH", BASELINE_EPOCH)
    recipe = recipe_profile()
    payload = {
        "baseline_type": "official" if official_checkpoint else "training_fallback",
        "task_id": "asr_en_wer",
        "model_id": official_model_id(),
        "source_url": official_model_url(),
        "dataset_profile": dataset_profile().name,
        "recipe_profile": recipe.name,
        "recipe": recipe.recipe_family,
        "architecture_signature": architecture_signature(),
        "checkpoint_source_dir": str(checkpoint_source_dir()),
        "pretrained_checkpoint_source": str(pretrained_checkpoint_source()),
        "use_pretrained_checkpoint": use_pretrained_checkpoint(),
        "workspace_checkpoint_dir": str(MODELS_DIR),
        "required_checkpoints": [str(path) for path in required_decode_checkpoints(epoch)],
        "epoch": epoch,
        "avg": decode_avg(epoch),
        "use_averaged_model": decode_uses_averaged_model(epoch),
        "bpe_model": str(decode_bpe_model_path()),
        "decoding_method": "modified_beam_search",
        "eval_splits": parse_eval_splits(),
        "train_args": large_cr_ctc_rnnt_train_args(),
        "elapsed_seconds": elapsed_seconds,
        "timestamp": time.time(),
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "official_baseline.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def clean_training_exp_dir_for_retry() -> None:
    if MODELS_DIR.is_symlink():
        raise RuntimeError(f"Refusing to clean symlinked training exp dir: {MODELS_DIR}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for item in MODELS_DIR.iterdir():
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        else:
            item.unlink()


def parse_eval_splits() -> list[str]:
    return parse_eval_splits_for_profile(
        os.environ.get("SURE_ASR_EVAL_SPLITS"),
        dataset_profile(),
    )


def _patched_librispeech_decode_script(source: str, source_path: Path) -> Path:
    old = """    test_clean_cuts = librispeech.test_clean_cuts()
    test_other_cuts = librispeech.test_other_cuts()

    test_clean_dl = librispeech.test_dataloaders(test_clean_cuts)
    test_other_dl = librispeech.test_dataloaders(test_other_cuts)

    test_sets = ["test-clean", "test-other"]
    test_dl = [test_clean_dl, test_other_dl]
"""
    new = """    split_to_cuts = {
        "dev-clean": librispeech.dev_clean_cuts,
        "dev-other": librispeech.dev_other_cuts,
        "test-clean": librispeech.test_clean_cuts,
        "test-other": librispeech.test_other_cuts,
    }
    raw_eval_splits = os.environ.get("SURE_ASR_EVAL_SPLITS", "test-clean,test-other")
    test_sets = [
        item.strip().replace("_", "-")
        for item in raw_eval_splits.replace(",", " ").split()
        if item.strip()
    ]
    if not test_sets:
        test_sets = ["test-clean", "test-other"]
    invalid_splits = [name for name in test_sets if name not in split_to_cuts]
    if invalid_splits:
        raise ValueError(
            f"Unsupported SURE_ASR_EVAL_SPLITS values: {invalid_splits}; "
            f"allowed: {sorted(split_to_cuts)}"
        )
    test_dl = [
        librispeech.test_dataloaders(split_to_cuts[test_set]())
        for test_set in test_sets
    ]
"""
    if old not in source:
        raise RuntimeError(
            f"Could not patch eval splits in {source_path} for recipe profile "
            f"{recipe_profile().name!r}"
        )
    target = WORKING_DIR / "decode_sure_eval.py"
    target.write_text(source.replace(old, new), encoding="utf-8")
    return target


def _patched_tedlium3_decode_script(source: str, source_path: Path) -> Path:
    old = """    dev_cuts = tedlium.dev_cuts()
    test_cuts = tedlium.test_cuts()

    dev_dl = tedlium.test_dataloaders(dev_cuts)
    test_dl = tedlium.test_dataloaders(test_cuts)

    test_sets = ["dev", "test"]
    test_dls = [dev_dl, test_dl]
"""
    new = """    import os

    split_to_cuts = {
        "dev": tedlium.dev_cuts,
        "test": tedlium.test_cuts,
    }
    raw_eval_splits = os.environ.get("SURE_ASR_EVAL_SPLITS", "dev")
    test_sets = [
        item.strip().replace("_", "-")
        for item in raw_eval_splits.replace(",", " ").split()
        if item.strip()
    ]
    if not test_sets:
        test_sets = ["dev"]
    invalid_splits = [name for name in test_sets if name not in split_to_cuts]
    if invalid_splits:
        raise ValueError(
            f"Unsupported SURE_ASR_EVAL_SPLITS values: {invalid_splits}; "
            f"allowed: {sorted(split_to_cuts)}"
        )
    test_dls = [
        tedlium.test_dataloaders(split_to_cuts[test_set]())
        for test_set in test_sets
    ]
"""
    if old not in source:
        raise RuntimeError(
            f"Could not patch eval splits in {source_path} for recipe profile "
            f"{recipe_profile().name!r}"
        )
    target = WORKING_DIR / "decode_sure_eval.py"
    target.write_text(source.replace(old, new), encoding="utf-8")
    return target


def patched_decode_script() -> Path:
    source_path = RECIPE_DIR / "decode.py"
    source = source_path.read_text(encoding="utf-8")

    # Newer local Zipformer recipes already honor SURE_ASR_EVAL_SPLITS.
    # Use them directly; only patch older hardcoded eval split layouts.
    if "SURE_ASR_EVAL_SPLITS" in source and "split_to_cuts" in source:
        return source_path

    strategy = recipe_profile().decode_patch_strategy
    if strategy == "librispeech_legacy_eval_splits":
        return _patched_librispeech_decode_script(source, source_path)
    if strategy == "tedlium3_eval_splits":
        return _patched_tedlium3_decode_script(source, source_path)
    if strategy == "native_sure_eval_splits":
        return source_path
    raise RuntimeError(
        f"Unsupported decode patch strategy {strategy!r} for recipe profile "
        f"{recipe_profile().name!r}"
    )


def copy_checkpoint_file(source: Path, target: Path, *, replace: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not replace:
            return
        if source.resolve() == target.resolve() and not target.is_symlink():
            return
        target.unlink()
    print(f"[baseline] copying checkpoint {source} -> {target}", flush=True)
    shutil.copy2(source, target)


def copy_pretrained_checkpoint_source() -> None:
    source = pretrained_checkpoint_source()
    if not source.is_file():
        source = MODELS_DIR / PRETRAINED_CHECKPOINT_NAME
    if not source.is_file():
        raise FileNotFoundError(
            "SURE_BASELINE_USE_PRETRAINED=1 but pretrained checkpoint was not found: "
            f"{pretrained_checkpoint_source()}"
        )
    target_epoch = int_env("SURE_BASELINE_EPOCH", BASELINE_EPOCH)
    target = MODELS_DIR / f"epoch-{target_epoch}.pt"
    print(
        f"[baseline] using pretrained checkpoint {source} as {target}",
        flush=True,
    )
    copy_checkpoint_file(source, target, replace=True)
    validate_pretrained_checkpoint_mapping(target_epoch)


def copy_checkpoint_source() -> None:
    if use_pretrained_checkpoint():
        copy_pretrained_checkpoint_source()
        return

    source_dir = checkpoint_source_dir()
    if source_dir.exists():
        for pattern in ("epoch-*.pt", "checkpoint-*.pt", "model*.pt", "*.pt"):
            for item in source_dir.glob(pattern):
                target = MODELS_DIR / item.name
                copy_checkpoint_file(item, target)


def has_baseline_checkpoint() -> bool:
    target_epoch = int_env("SURE_BASELINE_EPOCH", BASELINE_EPOCH)
    try:
        return not missing_decode_checkpoints(target_epoch)
    except ValueError:
        return False


def build_train_command(
    *,
    train_epochs: int,
    train_max_duration: int,
    fp16: str,
    attempt_index: int,
) -> list[str]:
    command = [
        os.environ.get("SURE_ICEFALL_PYTHON", sys.executable),
        str(RECIPE_DIR / "train.py"),
        "--world-size",
        str(world_size()),
        "--num-epochs",
        str(train_epochs),
        "--start-epoch",
        "1",
        "--use-fp16",
        fp16,
        "--exp-dir",
        str(MODELS_DIR),
        "--bpe-model",
        str(default_bpe_model()),
        "--master-port",
        str(15000 + ((os.getpid() + train_max_duration + attempt_index) % 20000)),
        "--max-duration",
        str(train_max_duration),
        *large_cr_ctc_rnnt_train_args(),
    ]
    if os.environ.get("SURE_ENABLE_MUSAN", "0").strip() == "0":
        command.extend(["--enable-musan", "0"])
    return command


def train_if_needed() -> int:
    copy_checkpoint_source()
    target_epoch = int_env("SURE_BASELINE_EPOCH", BASELINE_EPOCH)
    if has_baseline_checkpoint():
        validate_decode_checkpoints(target_epoch)
        print(
            f"[baseline] found required epoch-{target_epoch} decode checkpoint(s); "
            "using decode-only baseline",
            flush=True,
        )
        return target_epoch
    if has_configured_checkpoint_source():
        validate_decode_checkpoints(target_epoch)
    if truthy(os.environ.get("SURE_DECODE_ONLY"), default=False):
        validate_decode_checkpoints(target_epoch)

    budget_epochs = int_env("SURE_MAX_TRAIN_EPOCHS", 1)
    train_epochs = max(1, min(budget_epochs, target_epoch))
    train_max_duration = bounded_train_max_duration()
    fp16 = "1" if truthy(os.environ.get("SURE_USE_FP16"), default=True) else "0"
    attempts: list[dict[str, object]] = []
    durations = duration_retry_sequence(train_max_duration)
    timeout = timeout_env("SURE_BASELINE_TRAIN_TIMEOUT", 43200)

    for attempt_index, duration in enumerate(durations, start=1):
        if attempt_index > 1:
            clean_training_exp_dir_for_retry()

        command = build_train_command(
            train_epochs=train_epochs,
            train_max_duration=duration,
            fp16=fp16,
            attempt_index=attempt_index,
        )
        log_name = "train" if attempt_index == 1 else f"train_retry_{attempt_index}_duration_{duration}"
        log_path = WORKING_DIR / f"{log_name}.log"
        attempt: dict[str, object] = {
            "attempt": attempt_index,
            "max_duration": duration,
            "log_path": str(log_path),
        }
        try:
            completed_log_path = run_command(log_name, command, timeout=timeout)
        except (CommandFailedError, CommandTimedOutError) as exc:
            oom = is_oom_failure(exc.log_path)
            attempt.update(
                {
                    "status": "failed",
                    "oom": oom,
                    "return_code": getattr(exc, "return_code", None),
                    "timed_out": isinstance(exc, CommandTimedOutError),
                    "max_memory_allocated_mb": parse_max_memory_mb(exc.log_path),
                }
            )
            attempts.append(attempt)
            can_retry = oom and attempt_index < len(durations)
            write_resource_profile(
                status="retrying" if can_retry else "failed",
                attempts=attempts,
                selected_duration=None,
                train_epochs=train_epochs,
                fp16=fp16,
            )
            if can_retry:
                next_duration = durations[attempt_index]
                print(
                    "[baseline] detected OOM at "
                    f"max-duration={duration}; retrying with {next_duration}",
                    flush=True,
                )
                continue
            raise

        attempt.update(
            {
                "status": "success",
                "oom": False,
                "log_path": str(completed_log_path),
                "max_memory_allocated_mb": parse_max_memory_mb(completed_log_path),
            }
        )
        attempts.append(attempt)
        write_resource_profile(
            status="success",
            attempts=attempts,
            selected_duration=duration,
            train_epochs=train_epochs,
            fp16=fp16,
        )
        return train_epochs

    raise RuntimeError("No Zipformer training attempt was executed")


def build_decode_command(epoch: int, decode_script: Path) -> list[str]:
    avg = decode_avg(epoch)
    use_averaged_model = "1" if decode_uses_averaged_model(epoch) else "0"
    decode_max_duration = int_env(
        "SURE_BASELINE_DECODE_MAX_DURATION",
        BASELINE_DECODE_MAX_DURATION,
    )
    return [
        os.environ.get("SURE_ICEFALL_PYTHON", sys.executable),
        str(decode_script),
        "--epoch",
        str(epoch),
        "--avg",
        str(avg),
        "--use-averaged-model",
        use_averaged_model,
        "--exp-dir",
        str(MODELS_DIR),
        "--bpe-model",
        str(decode_bpe_model_path()),
        "--max-duration",
        str(decode_max_duration),
        "--decoding-method",
        "modified_beam_search",
        *large_cr_ctc_rnnt_decode_args(),
    ]


def decode(epoch: int) -> None:
    eval_splits = parse_eval_splits()
    validate_decode_checkpoints(epoch)
    validate_decode_bpe_model()
    command = build_decode_command(epoch, patched_decode_script())
    print(f"[baseline] decoding eval splits: {', '.join(eval_splits)}", flush=True)
    run_command("decode", command, timeout=int_env("SURE_BASELINE_DECODE_TIMEOUT", 21600))


def parse_hyp_value(value: str) -> str:
    value = value.strip()
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value
    if isinstance(parsed, (list, tuple)):
        return " ".join(str(item) for item in parsed)
    return str(parsed)


def load_ref_order() -> list[str]:
    ref_path = Path("input/ref.txt")
    if not ref_path.is_file():
        return []
    keys: list[str] = []
    with ref_path.open("r", encoding="utf-8") as f:
        for line in f:
            if "\t" not in line:
                continue
            key, _ = line.rstrip("\n").split("\t", 1)
            if key:
                keys.append(key)
    return keys


def collect_hypotheses() -> dict[str, str]:
    pattern = re.compile(r"^(.*?):\thyp=(.*)$")
    hyps: dict[str, str] = {}
    profile = dataset_profile()
    for recogs_path in sorted(MODELS_DIR.rglob("recogs-*.txt")):
        with recogs_path.open("r", encoding="utf-8") as f:
            for line in f:
                match = pattern.match(line.rstrip("\n"))
                if not match:
                    continue
                key, hyp = match.groups()
                value = parse_hyp_value(hyp)
                hyps[key] = value
                normalized_key = normalize_asr_cut_id(key, profile)
                hyps.setdefault(normalized_key, value)
    return hyps


def normalize_librispeech_cut_id(key: str) -> str:
    return normalize_asr_cut_id(key)


def write_sure_hyp() -> None:
    ref_order = load_ref_order()
    ref_keys = set(ref_order)
    hyps = collect_hypotheses()
    if ref_keys:
        missing = [key for key in ref_order if key not in hyps]
        if missing:
            preview = ", ".join(missing[:10])
            raise RuntimeError(f"Decode output is missing {len(missing)} reference keys: {preview}")
        ordered = [(key, hyps[key]) for key in ref_order]
    else:
        ordered = sorted(hyps.items())
    if not ordered:
        raise RuntimeError(f"No hypotheses found under {MODELS_DIR}")
    hyp_path = ARTIFACTS_DIR / "hyp.txt"
    with hyp_path.open("w", encoding="utf-8") as f:
        for key, hyp in ordered:
            f.write(f"{key}\t{hyp}\n")
    print(f"[baseline] wrote {len(ordered)} hypotheses to {hyp_path}", flush=True)


def main() -> None:
    start = time.time()
    ensure_workspace()
    epoch = train_if_needed()
    decode(epoch)
    write_sure_hyp()
    write_official_baseline_record(epoch, time.time() - start)
    print(f"[baseline] completed in {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
