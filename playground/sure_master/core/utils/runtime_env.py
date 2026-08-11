from __future__ import annotations

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


_DURATION_CACHE_SCHEMA_VERSION = 2
_DURATION_PROBE_POLICY_VERSION = "recipe-cli-contract-v2"
_DEPRECATED_FULL_LIBRI_WARNED = False


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() not in {"", "0", "false", "no", "off"}


def _positive_int_env(name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return max(minimum, default)
    return max(minimum, value)


def _nonnegative_int_env(name: str, default: int) -> int:
    try:
        value = int(float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return max(0, default)
    return max(0, value)


def descending_duration_values(
    start: int,
    *,
    minimum: int,
    step: int,
    configured_values: str = "",
) -> list[int]:
    """Build a deterministic, descending duration fallback sequence.

    This is shared by exact-workload tuners.  It deliberately contains no
    retry policy: callers decide which failures are safe to retry.
    """
    minimum = max(1, int(minimum))
    start = max(minimum, int(start))
    step = max(1, int(step))
    if configured_values.strip():
        values = [start]
        for item in configured_values.replace(",", " ").split():
            try:
                value = int(float(item))
            except ValueError:
                continue
            if minimum <= value < start:
                values.append(value)
        return sorted(set(values), reverse=True)

    values = list(range(start, minimum - 1, -step))
    if values[-1] != minimum:
        values.append(minimum)
    return values


def write_json_atomic(path: Path, payload: object) -> None:
    """Atomically replace a JSON evidence/cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _log(message: str) -> None:
    print(f"[sure_runtime] {message}", file=sys.stderr, flush=True)


def _visible_cuda_indices(value: str | None) -> list[int]:
    if value is None:
        return []
    raw = value.strip()
    if not raw or raw.lower() in {"all", "none", "void", "-1"}:
        return []

    indices: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            indices.append(int(part))
        except ValueError:
            return []
    return indices


def query_gpu_memory_mb() -> list[int]:
    """Return visible GPU total memory in MB using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    if result.returncode != 0:
        return []

    memories: list[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            memories.append(int(float(line)))
        except ValueError:
            continue
    return memories


def visible_gpu_memory_mb() -> list[int]:
    memories = query_gpu_memory_mb()
    if not memories:
        return []

    visible_indices = _visible_cuda_indices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if not visible_indices:
        return memories

    if len(memories) == len(visible_indices):
        return memories

    selected = [memories[i] for i in visible_indices if 0 <= i < len(memories)]
    return selected or memories


def duration_from_gpu_memory(memory_mb: int, use_fp16: bool = True) -> int:
    if memory_mb < 20_000:
        duration = 600
    elif memory_mb < 32_000:
        duration = 1000
    elif memory_mb < 48_000:
        duration = 1600
    else:
        duration = 2400

    if not use_fp16:
        duration = max(600, int(duration * 0.6))
        duration = (duration // 100) * 100
    return duration


def autotune_max_duration(
    baseline: int,
    memories: list[int] | None = None,
    use_fp16: bool = True,
) -> int:
    """Probe a safe max-duration and cache the result across parallel experiments."""
    memories = memories or []
    step = _positive_int_env("SURE_DURATION_AUTOTUNE_STEP", 100)
    lower_limit = _positive_int_env("SURE_DURATION_AUTOTUNE_MIN", 100)
    upper_limit = _positive_int_env("SURE_DURATION_AUTOTUNE_MAX", max(baseline, lower_limit))
    upper_limit = max(lower_limit, upper_limit)
    start = min(max(baseline, lower_limit), upper_limit)
    timeout = _positive_int_env("SURE_DURATION_PROBE_TIMEOUT", 1800)
    world_size = _resolve_world_size()
    probe_args = _duration_probe_args()

    cache_dir = _duration_cache_dir().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _duration_cache_key(
        baseline=start,
        memories=memories,
        use_fp16=use_fp16,
        world_size=world_size,
        step=step,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        probe_args=probe_args,
    )
    cache_path = cache_dir / f"{cache_key}.json"
    cached = _read_cached_duration(cache_path)
    if cached is not None:
        _log(f"duration auto-tune cache hit: {cached}")
        return cached

    lock_dir = cache_dir / f"{cache_key}.lock"
    lock_timeout = max(300, timeout + 60)
    if not _acquire_lock(lock_dir, cache_path, lock_timeout):
        cached = _read_cached_duration(cache_path)
        if cached is not None:
            return cached
        raise RuntimeError(
            "duration auto-tune lock timed out without a cached safe value; "
            f"refusing to use unverified max-duration={start}"
        )

    try:
        cached = _read_cached_duration(cache_path)
        if cached is not None:
            return cached
        result = _autotune_uncached(
            start=start,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            step=step,
            timeout=timeout,
            world_size=world_size,
            use_fp16=use_fp16,
            cache_dir=cache_dir,
            cache_key=cache_key,
            probe_args=probe_args,
        )
        _write_cached_duration(cache_path, result)
        return result
    finally:
        shutil.rmtree(lock_dir, ignore_errors=True)


def _autotune_uncached(
    start: int,
    lower_limit: int,
    upper_limit: int,
    step: int,
    timeout: int,
    world_size: int,
    use_fp16: bool,
    cache_dir: Path,
    cache_key: str,
    probe_args: list[str],
) -> int:
    best: int | None = None
    probe_root = cache_dir / "probes" / cache_key

    for duration in range(start, upper_limit + 1, step):
        if _probe_duration(duration, world_size, use_fp16, timeout, probe_root, probe_args):
            best = duration
            _log(f"duration probe succeeded: {duration}")
            continue
        _log(f"duration probe failed: {duration}")
        if best is not None:
            return best
        break

    if best is not None:
        return best

    for duration in range(start - step, lower_limit - 1, -step):
        if _probe_duration(duration, world_size, use_fp16, timeout, probe_root, probe_args):
            _log(f"duration fallback probe succeeded: {duration}")
            return duration
        _log(f"duration fallback probe failed: {duration}")

    message = (
        f"all duration probes failed down to {lower_limit}; "
        "refusing to use an unverified max-duration"
    )
    _log(message)
    raise RuntimeError(message)


def _duration_cache_dir() -> Path:
    configured = os.environ.get("SURE_DURATION_CACHE_DIR")
    if configured:
        return Path(configured)
    return Path.cwd() / ".sure_runtime" / "duration_autotune"


def _train_script_identity() -> dict[str, object]:
    train_py = _recipe_dir().resolve() / "train.py"
    try:
        stat = train_py.stat()
        digest = hashlib.sha256(train_py.read_bytes()).hexdigest()
    except OSError:
        return {"path": str(train_py), "exists": False}
    return {
        "path": str(train_py),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest,
    }


def _duration_cache_key(
    baseline: int,
    memories: list[int],
    use_fp16: bool,
    world_size: int,
    step: int,
    lower_limit: int,
    upper_limit: int,
    probe_args: list[str],
) -> str:
    payload = {
        "baseline": baseline,
        "gpu_memory_mb": memories,
        "use_fp16": use_fp16,
        "world_size": world_size,
        "candidate_key": os.environ.get("SURE_DURATION_CANDIDATE_KEY", ""),
        "recipe_key": os.environ.get("SURE_DURATION_RECIPE_KEY", str(_recipe_dir().resolve())),
        "python": os.environ.get("SURE_ICEFALL_PYTHON", sys.executable),
        "manifest_dir": os.environ.get("SURE_DURATION_MANIFEST_DIR", "data/fbank"),
        "bpe_model": os.environ.get("SURE_DURATION_BPE_MODEL", "data/lang_bpe_500/bpe.model"),
        "enable_musan": os.environ.get("SURE_ENABLE_MUSAN", "0"),
        "probe_policy": _DURATION_PROBE_POLICY_VERSION,
        "train_script": _train_script_identity(),
        "probe_success_batches": os.environ.get("SURE_DURATION_PROBE_SUCCESS_BATCHES", "2"),
        "probe_log_interval": os.environ.get("SURE_DURATION_PROBE_LOG_INTERVAL", "1"),
        "probe_headroom_mb": os.environ.get("SURE_DURATION_HEADROOM_MB", "0"),
        "probe_args": probe_args,
        "step": step,
        "lower_limit": lower_limit,
        "upper_limit": upper_limit,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _read_cached_duration(path: Path) -> int | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != _DURATION_CACHE_SCHEMA_VERSION:
            return None
        if payload.get("probe_policy") != _DURATION_PROBE_POLICY_VERSION:
            return None
        duration = int(payload["duration"])
    except (OSError, AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return duration if duration > 0 else None


def _write_cached_duration(path: Path, duration: int) -> None:
    write_json_atomic(
        path,
        {
            "schema_version": _DURATION_CACHE_SCHEMA_VERSION,
            "probe_policy": _DURATION_PROBE_POLICY_VERSION,
            "duration": duration,
        },
    )


def _acquire_lock(lock_dir: Path, cache_path: Path, timeout: int) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            lock_dir.mkdir(parents=True)
            return True
        except FileExistsError:
            if _read_cached_duration(cache_path) is not None:
                return False
            time.sleep(5)
    return False


def _resolve_world_size() -> int:
    raw_world_size = os.environ.get("ASR_WORLD_SIZE")
    if raw_world_size:
        try:
            return max(1, int(raw_world_size))
        except ValueError:
            pass

    visible = _visible_cuda_indices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if visible:
        return len(visible)
    memories = visible_gpu_memory_mb()
    return max(1, len(memories))


def _recipe_dir() -> Path:
    return Path(os.environ.get("SURE_DURATION_RECIPE_DIR", "base_model/recipe"))


def _dedupe_paths(paths: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for path in paths:
        text = str(path or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _infer_icefall_root_from_recipe_dir(recipe_dir: Path) -> Path | None:
    for parent in (recipe_dir, *recipe_dir.parents):
        if (parent / "icefall").exists() and (parent / "egs").exists():
            return parent
    return None


def _probe_command_env(recipe_dir: Path, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return an env that can launch icefall recipe scripts from a workspace.

    Generated candidates sometimes invoke this helper directly.  The helper must
    therefore not rely on the caller to add the workspace icefall root/recipe to
    PYTHONPATH; otherwise import failures look like failed duration probes.
    """
    env = os.environ.copy()
    pythonpath_entries: list[str] = []

    configured_root = (
        env.get("SURE_DURATION_ICEFALL_ROOT", "").strip()
        or env.get("SURE_ICEFALL_ROOT", "").strip()
    )
    if configured_root:
        pythonpath_entries.append(str(Path(configured_root)))

    if recipe_dir.name == "recipe" and recipe_dir.parent.name == "base_model":
        workspace_root = recipe_dir.parent.parent
        pythonpath_entries.append(str(workspace_root / "base_model" / "root"))

    inferred_root = _infer_icefall_root_from_recipe_dir(recipe_dir)
    if inferred_root is not None:
        pythonpath_entries.append(str(inferred_root))
    pythonpath_entries.append(str(recipe_dir))

    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_entries.extend(existing_pythonpath.split(os.pathsep))

    env["PYTHONPATH"] = os.pathsep.join(_dedupe_paths(pythonpath_entries))
    env.setdefault("PYTHONUNBUFFERED", "1")
    if extra_env:
        env.update(extra_env)
    return env


def _duration_probe_args() -> list[str]:
    raw_json = os.environ.get("SURE_DURATION_TRAIN_ARGS_JSON", "").strip()
    if raw_json:
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Invalid SURE_DURATION_TRAIN_ARGS_JSON") from exc
        if not isinstance(parsed, list):
            raise RuntimeError("SURE_DURATION_TRAIN_ARGS_JSON must be a JSON list")
        return [str(item) for item in parsed]

    raw = os.environ.get("SURE_DURATION_TRAIN_ARGS", "").strip()
    if not raw:
        return []
    return shlex.split(raw)


_PROBE_CONTROLLED_OPTIONS = {
    "--world-size": 1,
    "--master-port": 1,
    "--num-epochs": 1,
    "--start-epoch": 1,
    "--tensorboard": 1,
    "--use-fp16": 1,
    "--exp-dir": 1,
    "--max-duration": 1,
    "--print-diagnostics": 1,
    "--enable-musan": 1,
    "--manifest-dir": 1,
    "--bpe-model": 1,
    "--log-interval": 1,
}

_PROBE_REQUIRED_OPTIONS = frozenset(
    {
        "--world-size",
        "--master-port",
        "--num-epochs",
        "--start-epoch",
        "--tensorboard",
        "--use-fp16",
        "--exp-dir",
        "--max-duration",
        "--enable-musan",
        "--manifest-dir",
        "--bpe-model",
    }
)
_PROBE_OPTIONAL_OPTIONS = frozenset({"--log-interval", "--print-diagnostics"})


def _strip_probe_controlled_args(args: list[str]) -> list[str]:
    stripped: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        matched = False
        for option, arity in _PROBE_CONTROLLED_OPTIONS.items():
            if arg == option:
                index += arity + 1
                matched = True
                break
            if arg.startswith(f"{option}="):
                index += 1
                matched = True
                break
        if matched:
            continue
        stripped.append(arg)
        index += 1
    return stripped


_TRAIN_HELP_OPTION_CACHE: dict[tuple[str, str, str], frozenset[str]] = {}


def _duration_probe_contract_error(marker: str, train_py: Path, **details: object) -> RuntimeError:
    payload = {"train_py": str(train_py), **details}
    message = f"{marker}: {json.dumps(payload, sort_keys=True)}"
    _log(message)
    return RuntimeError(message)


def _warn_deprecated_full_libri() -> None:
    global _DEPRECATED_FULL_LIBRI_WARNED
    if "SURE_DURATION_PROBE_FULL_LIBRI" not in os.environ or _DEPRECATED_FULL_LIBRI_WARNED:
        return
    _DEPRECATED_FULL_LIBRI_WARNED = True
    _log(
        "SURE_DURATION_PROBE_FULL_LIBRI is deprecated and ignored; "
        "recipe profiles own recipe-specific train arguments"
    )


def _train_supported_options(python_bin: str, train_py: Path, recipe_dir: Path) -> frozenset[str]:
    try:
        script_digest = hashlib.sha256(train_py.read_bytes()).hexdigest()
    except OSError as exc:
        raise _duration_probe_contract_error(
            "duration_probe_help_discovery_failed",
            train_py,
            error="script_read_failed",
            exception_type=type(exc).__name__,
            errno=exc.errno,
        ) from exc
    cache_key = (str(Path(python_bin)), str(train_py.resolve()), script_digest)
    cached = _TRAIN_HELP_OPTION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        result = subprocess.run(
            [python_bin, str(train_py), "--help"],
            cwd=str(recipe_dir),
            env=_probe_command_env(recipe_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise _duration_probe_contract_error(
            "duration_probe_help_discovery_failed",
            train_py,
            error="timeout",
            timeout_seconds=30,
        ) from exc
    except OSError as exc:
        raise _duration_probe_contract_error(
            "duration_probe_help_discovery_failed",
            train_py,
            error="launch_failed",
            exception_type=type(exc).__name__,
            errno=exc.errno,
        ) from exc

    if result.returncode != 0:
        raise _duration_probe_contract_error(
            "duration_probe_help_discovery_failed",
            train_py,
            error="nonzero_exit",
            returncode=result.returncode,
        )
    options = frozenset(
        re.findall(r"(?<!\w)--[A-Za-z0-9][A-Za-z0-9-]*", result.stdout or "")
    )
    if not options:
        raise _duration_probe_contract_error(
            "duration_probe_help_discovery_failed",
            train_py,
            error="no_long_options",
        )

    _TRAIN_HELP_OPTION_CACHE[cache_key] = options
    return options


def _argument_option_names(args: list[str]) -> frozenset[str]:
    return frozenset(arg.split("=", 1)[0] for arg in args if arg.startswith("--") and len(arg) > 2)


def _validate_probe_cli_contract(
    supported_options: frozenset[str],
    extra_args: list[str],
    train_py: Path,
) -> None:
    missing_helper = sorted(_PROBE_REQUIRED_OPTIONS - supported_options)
    if missing_helper:
        raise _duration_probe_contract_error(
            "duration_probe_helper_cli_incompatible",
            train_py,
            unsupported_options=missing_helper,
        )

    missing_candidate = sorted(_argument_option_names(extra_args) - supported_options)
    if missing_candidate:
        raise _duration_probe_contract_error(
            "duration_probe_candidate_cli_incompatible",
            train_py,
            unsupported_options=missing_candidate,
        )


def _append_supported_option(
    command: list[str],
    supported_options: frozenset[str],
    option: str,
    value: str,
) -> None:
    if option in supported_options:
        command.extend([option, value])


def _probe_duration(
    duration: int,
    world_size: int,
    use_fp16: bool,
    timeout: int,
    probe_root: Path,
    extra_train_args: list[str] | None = None,
) -> bool:
    recipe_dir = _recipe_dir().resolve()
    train_py = recipe_dir / "train.py"
    if not train_py.is_file():
        _log(f"duration probe skipped; train.py not found at {train_py}")
        return False

    exp_dir = probe_root / f"duration_{duration}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "probe.log"

    python_bin = os.environ.get("SURE_ICEFALL_PYTHON", sys.executable)
    manifest_dir = os.environ.get("SURE_DURATION_MANIFEST_DIR", "data/fbank")
    bpe_model = os.environ.get("SURE_DURATION_BPE_MODEL", "data/lang_bpe_500/bpe.model")
    enable_musan = os.environ.get("SURE_ENABLE_MUSAN", "0")
    print_diagnostics = "true" if _truthy(os.environ.get("SURE_DURATION_PRINT_DIAGNOSTICS")) else "false"
    success_batch_count = _nonnegative_int_env("SURE_DURATION_PROBE_SUCCESS_BATCHES", 2)
    log_interval = _positive_int_env("SURE_DURATION_PROBE_LOG_INTERVAL", 1)
    headroom_mb = _nonnegative_int_env("SURE_DURATION_HEADROOM_MB", 0)
    gpu_memory_mb = visible_gpu_memory_mb() if headroom_mb > 0 else []
    extra_args = _strip_probe_controlled_args(extra_train_args or [])
    master_port = 15000 + ((os.getpid() + duration) % 20000)
    _warn_deprecated_full_libri()
    supported_options = _train_supported_options(python_bin, train_py, recipe_dir)
    _validate_probe_cli_contract(supported_options, extra_args, train_py)
    command = [
        python_bin,
        str(train_py),
        "--world-size",
        str(world_size),
        "--master-port",
        str(master_port),
        "--num-epochs",
        "1",
        "--start-epoch",
        "1",
        "--tensorboard",
        "false",
        "--use-fp16",
        "1" if use_fp16 else "0",
        "--exp-dir",
        str(exp_dir),
        "--max-duration",
        str(duration),
        "--enable-musan",
        str(enable_musan),
        "--manifest-dir",
        str(manifest_dir),
        "--bpe-model",
        str(bpe_model),
    ]
    _append_supported_option(
        command,
        supported_options,
        "--print-diagnostics",
        print_diagnostics,
    )
    _append_supported_option(command, supported_options, "--log-interval", str(log_interval))
    command.extend(extra_args)
    _log(f"probing max-duration={duration}, world_size={world_size}, log={log_path}")
    return _run_logged_command(
        command,
        cwd=recipe_dir,
        timeout=timeout,
        log_path=log_path,
        success_batch_count=success_batch_count,
        memory_headroom_mb=headroom_mb,
        gpu_memory_mb=gpu_memory_mb,
    )


def _run_logged_command(
    command: list[str],
    cwd: Path,
    timeout: int,
    log_path: Path,
    success_batch_count: int = 0,
    memory_headroom_mb: int = 0,
    gpu_memory_mb: list[int] | None = None,
) -> bool:
    env = _probe_command_env(cwd)
    gpu_memory_mb = gpu_memory_mb or []
    with log_path.open("a", encoding="utf-8", errors="ignore") as log:
        log.write("[sure_runtime] command: " + " ".join(command) + "\n")
        log.write("[sure_runtime] cwd: " + str(cwd) + "\n")
        log.write("[sure_runtime] PYTHONPATH: " + env.get("PYTHONPATH", "") + "\n")
        log.flush()
        read_offset = log_path.stat().st_size
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=(sys.platform != "win32"),
        )
        deadline = time.monotonic() + timeout
        observed_batches: set[tuple[int, int]] = set()
        max_memory_mb: int | None = None
        while True:
            chunk, read_offset = _read_new_log_text(log_path, read_offset)
            if chunk:
                if _probe_has_incompatible_cli_failure(chunk):
                    log.write("[sure_runtime] probe command is incompatible with train.py CLI\n")
                    log.flush()
                    _terminate_process_group(proc, force=True)
                    raise RuntimeError(f"duration probe command is incompatible with train.py; see {log_path}")
                if _probe_has_fatal_failure(chunk):
                    log.write("[sure_runtime] probe observed fatal resource failure\n")
                    log.flush()
                    _terminate_process_group(proc, force=True)
                    return False
                if _probe_has_non_resource_failure(chunk):
                    log.write(
                        "[sure_runtime] probe observed non-resource startup/candidate failure\n"
                    )
                    log.flush()
                    _terminate_process_group(proc, force=True)
                    raise RuntimeError(
                        "duration probe failed before proving a resource limit; "
                        f"see {log_path}"
                    )
                parsed_memory_mb = _parse_max_memory_mb(chunk)
                if parsed_memory_mb is not None:
                    max_memory_mb = max(max_memory_mb or 0, parsed_memory_mb)
                    if not _probe_has_memory_headroom(
                        max_memory_mb,
                        memory_headroom_mb,
                        gpu_memory_mb,
                    ):
                        log.write(
                            "[sure_runtime] probe rejected because peak memory "
                            f"{max_memory_mb}MB leaves less than "
                            f"{memory_headroom_mb}MB headroom on visible GPU(s) "
                            f"{gpu_memory_mb}\n"
                        )
                        log.flush()
                        _terminate_process_group(proc, force=True)
                        return False
                if success_batch_count > 0:
                    for match in _TRAIN_BATCH_RE.finditer(chunk):
                        observed_batches.add((int(match.group(1)), int(match.group(2))))
                    if len(observed_batches) >= success_batch_count:
                        log.write(
                            "[sure_runtime] probe accepted after "
                            f"{len(observed_batches)} unique training batch log(s)\n"
                        )
                        log.flush()
                        _terminate_process_group(proc)
                        return True

            return_code = proc.poll()
            if return_code is not None:
                chunk, read_offset = _read_new_log_text(log_path, read_offset)
                if chunk and _probe_has_incompatible_cli_failure(chunk):
                    _terminate_process_group(proc, force=True)
                    raise RuntimeError(f"duration probe command is incompatible with train.py; see {log_path}")
                if chunk and _probe_has_fatal_failure(chunk):
                    return False
                if chunk and _probe_has_non_resource_failure(chunk):
                    raise RuntimeError(
                        "duration probe failed before proving a resource limit; "
                        f"see {log_path}"
                    )
                if chunk:
                    parsed_memory_mb = _parse_max_memory_mb(chunk)
                    if parsed_memory_mb is not None:
                        max_memory_mb = max(max_memory_mb or 0, parsed_memory_mb)
                if not _probe_has_memory_headroom(
                    max_memory_mb,
                    memory_headroom_mb,
                    gpu_memory_mb,
                ):
                    return False
                if return_code != 0:
                    _terminate_process_group(proc, force=True)
                    log.write(
                        "[sure_runtime] probe exited non-zero without a recognized "
                        "OOM/resource signal; treating this as a startup or candidate "
                        "error rather than lowering max-duration\n"
                    )
                    log.flush()
                    raise RuntimeError(
                        "duration probe exited non-zero without a recognized "
                        f"resource failure; see {log_path}"
                    )
                return True

            if time.monotonic() >= deadline:
                log.write(f"[sure_runtime] probe timed out after {timeout}s\n")
                log.flush()
                _terminate_process_group(proc)
                return False

            time.sleep(1)


_TRAIN_BATCH_RE = re.compile(r"\bEpoch\s+(\d+),\s+batch\s+(\d+)\b", re.IGNORECASE)
_MAX_MEMORY_RE = re.compile(
    r"(?:max(?:imum)? memory allocated(?: so far)?(?: is)?|max_memory_allocated)"
    r"[^0-9]{0,32}([0-9]+(?:\.[0-9]+)?)\s*(?:mib|mb)?",
    re.IGNORECASE,
)
_FATAL_PROBE_PATTERNS = (
    "cuda out of memory",
    "torch.cuda.outofmemoryerror",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
)

_INCOMPATIBLE_CLI_PROBE_PATTERNS = (
    "unrecognized arguments:",
    "no such option:",
    "unknown option:",
)

_NON_RESOURCE_PROBE_PATTERNS = (
    "modulenotfounderror:",
    "importerror:",
    "filenotfounderror:",
    "permissionerror:",
    "assertionerror:",
    "valueerror:",
    "runtimeerror:",
    "no module named",
)


def _read_new_log_text(log_path: Path, offset: int) -> tuple[str, int]:
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as reader:
            reader.seek(offset)
            chunk = reader.read()
            return chunk, reader.tell()
    except OSError:
        return "", offset


def _probe_has_fatal_failure(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in _FATAL_PROBE_PATTERNS)


def _probe_has_incompatible_cli_failure(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in _INCOMPATIBLE_CLI_PROBE_PATTERNS)


def _probe_has_non_resource_failure(text: str) -> bool:
    lowered = text.lower()
    if _probe_has_fatal_failure(lowered):
        return False
    return any(pattern in lowered for pattern in _NON_RESOURCE_PROBE_PATTERNS)


def _parse_max_memory_mb(text: str) -> int | None:
    matches = _MAX_MEMORY_RE.findall(text)
    if not matches:
        return None
    return int(float(matches[-1]))


def _probe_has_memory_headroom(
    max_memory_mb: int | None,
    headroom_mb: int,
    gpu_memory_mb: list[int],
) -> bool:
    if headroom_mb <= 0 or max_memory_mb is None or not gpu_memory_mb:
        return True
    return min(gpu_memory_mb) - max_memory_mb >= headroom_mb


def _terminate_process_group(
    proc: subprocess.Popen,
    grace_seconds: int = 10,
    force: bool = False,
) -> None:
    if proc.poll() is not None and not force:
        return
    if sys.platform == "win32":
        if proc.poll() is None:
            proc.kill()
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        proc.terminate()
    try:
        proc.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        proc.kill()


def resolve_max_duration(default: int = 600) -> int:
    memories = visible_gpu_memory_mb()
    use_fp16 = _truthy(os.environ.get("SURE_USE_FP16", "1"))
    baseline = duration_from_gpu_memory(min(memories), use_fp16=use_fp16) if memories else default
    if _truthy(os.environ.get("SURE_DURATION_AUTOTUNE")):
        return autotune_max_duration(baseline, memories=memories, use_fp16=use_fp16)
    return baseline


def main(argv: list[str] | None = None) -> int:
    argv = argv or []
    command = argv[0] if argv else "resolve-max-duration"
    if command != "resolve-max-duration":
        raise SystemExit(f"Unknown runtime_env command: {command}")

    print(resolve_max_duration())
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
