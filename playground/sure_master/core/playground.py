from __future__ import annotations

import json
import logging
import math
import os
import py_compile
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any, Callable

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evomaster.agent.session import LocalSessionConfig
from evomaster.core import BasePlayground, register_playground

from ..agent.session.local import SureMasterLocalSession
from .exp.knowledge_promotion_exp import KnowledgePromotionExp
from .exp.prefetch_exp import PrefetchExp
from .exp.research_exp import ResearchExp
from .exp.run_exp import SureRunExp
from .exp.wisdom_promotion_exp import WisdomPromotionExp
from .utils.code import save_code_to_file
from .utils.candidate_type import (
    ARCH,
    FINE_TUNE,
    INFERENCE,
    TRAINING_TYPES,
    candidate_type_from_code,
    candidate_type_from_idea,
    normalize_candidate_type,
)
from .utils.metric import SureMetricRunner
from .utils.task_cards import (
    BaseModelProfile,
    SureTaskCard,
    merge_base_model_profile,
    resolve_task_card,
    validate_base_model_profile,
)
from .utils.vc_remote import (
    candidate_limits_from,
    draft_runs_remotely,
    complete_remote_coverage_from,
    mixed_execution_enabled,
    remote_candidate_types_from,
    remote_training_max_parallel,
    sure_config_from,
)
from .utils.watch_dog import (
    GlobalTimeoutInterrupt,
    RUN_TIMEOUT_SECONDS,
    TimeoutWatchdog,
    _async_raise,
)


_NO_GPU_SENTINELS = {"", "none", "null", "false", "cpu", "-1"}
_AUTO_GPU_SENTINELS = {"auto", "all"}
_IDLE_GPU_SENTINELS = {"idle", "free", "available"}
_IDLE_GPU_DEFAULT_MIN_FREE_MIB = 9000
_IDLE_GPU_DEFAULT_MAX_UTILIZATION = 20


def _is_auto_gpu_value(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().lower() in _AUTO_GPU_SENTINELS
    )


def _is_idle_gpu_value(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in _IDLE_GPU_SENTINELS


def _parse_gpu_devices(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for item in value:
            parts.extend(str(item).split(","))
    else:
        raw = str(value).strip()
        if (
            raw.lower() in _NO_GPU_SENTINELS
            or raw.lower() in _AUTO_GPU_SENTINELS
            or raw.lower() in _IDLE_GPU_SENTINELS
        ):
            return []
        parts = raw.split(",")
    return [
        part.strip()
        for part in parts
        if part.strip() and part.strip().lower() not in _NO_GPU_SENTINELS
    ]


def _discover_gpu_devices(logger: logging.Logger) -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip().lower() not in _NO_GPU_SENTINELS:
        devices = _parse_gpu_devices(visible)
        if devices:
            return devices

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if devices:
                return devices
        elif result.stderr.strip():
            logger.debug("nvidia-smi GPU detection failed: %s", result.stderr.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug("nvidia-smi GPU detection unavailable: %s", e)

    try:
        import torch  # type: ignore

        count = torch.cuda.device_count()
        if count > 0:
            return [str(i) for i in range(count)]
    except Exception as e:
        logger.debug("torch GPU detection unavailable: %s", e)

    return []


def _visible_gpu_filter_from_env() -> set[str] | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.strip() == "":
        return None
    if visible.strip().lower() in _NO_GPU_SENTINELS:
        return set()
    devices = _parse_gpu_devices(visible)
    return set(devices)


def _parse_nvidia_smi_number(value: str) -> float:
    text = str(value).strip()
    number = "".join(ch for ch in text if ch.isdigit() or ch in ".-")
    if number in {"", ".", "-", "-."}:
        raise ValueError(f"not a number: {value!r}")
    return float(number)


def _env_or_config_int(
    env_name: str,
    config: dict[str, Any],
    config_name: str,
    default: int,
) -> int:
    if env_name in os.environ:
        return _non_negative_int(os.environ.get(env_name), default)
    return _non_negative_int(config.get(config_name), default)


def _env_or_config_bool(
    env_name: str,
    config: dict[str, Any],
    config_name: str,
    default: bool = False,
) -> bool:
    raw = os.environ.get(env_name, config.get(config_name))
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _discover_idle_gpu_devices(
    logger: logging.Logger,
    *,
    min_free_mib: int,
    max_utilization: int,
    max_count: int = 0,
) -> list[str]:
    allowed = _visible_gpu_filter_from_env()
    if allowed == set():
        return []

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug("nvidia-smi idle GPU detection unavailable: %s", e)
        return []

    if result.returncode != 0:
        if result.stderr.strip():
            logger.debug("nvidia-smi idle GPU detection failed: %s", result.stderr.strip())
        return []

    idle: list[tuple[str, float, float, float, float]] = []
    skipped: list[tuple[str, float, float, str]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            logger.debug("Skipping malformed nvidia-smi GPU row: %s", line)
            continue
        index = parts[0]
        if allowed is not None and index not in allowed:
            continue
        try:
            total_mib = _parse_nvidia_smi_number(parts[1])
            used_mib = _parse_nvidia_smi_number(parts[2])
            free_mib = _parse_nvidia_smi_number(parts[3])
            utilization = _parse_nvidia_smi_number(parts[4])
        except ValueError:
            logger.debug("Skipping unparsable nvidia-smi GPU row: %s", line)
            continue
        if free_mib >= min_free_mib and utilization <= max_utilization:
            idle.append((index, free_mib, utilization, total_mib, used_mib))
        else:
            skipped.append((index, free_mib, utilization, "busy"))

    idle.sort(
        key=lambda item: (
            -item[1],
            item[2],
            int(item[0]) if item[0].isdigit() else 10**9,
            item[0],
        )
    )
    if max_count > 0:
        idle = idle[:max_count]

    devices = [item[0] for item in idle]
    logger.info(
        "Resolved SURE idle GPU devices: %s (min_free_mib=%s, max_utilization=%s, visible_filter=%s)",
        devices,
        min_free_mib,
        max_utilization,
        sorted(allowed) if allowed is not None else "all",
    )
    if skipped:
        logger.debug("Skipped busy GPUs: %s", skipped)
    return devices


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _non_negative_int(value: Any, default: int) -> int:
    try:
        parsed = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


def _coordinator_local_gpu_required(config: Any) -> tuple[bool, str]:
    """Resolve the shared coordinator GPU policy before session discovery."""
    sure_config = sure_config_from(config)
    coordinator = sure_config.get("coordinator") or {}
    if not isinstance(coordinator, dict):
        coordinator = {}
    policy = str(coordinator.get("local_gpu_policy") or "auto").strip().lower()
    if policy in {"disabled", "none", "off"}:
        complete, diagnostics = complete_remote_coverage_from(config)
        if not complete:
            raise ValueError(
                "sure.coordinator.local_gpu_policy=disabled requires complete remote "
                f"coverage; diagnostics={diagnostics}"
            )
        return False, "disabled"
    if policy in {"required", "local", "on"}:
        return True, "required"
    if policy != "auto":
        raise ValueError(
            "sure.coordinator.local_gpu_policy must be auto, required, or disabled"
        )
    complete, _diagnostics = complete_remote_coverage_from(config)
    return not complete, "auto"


def _apply_coordinator_gpu_policy(
    config: Any,
    local_config: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, Any]:
    required, policy = _coordinator_local_gpu_required(config)
    if required:
        return _apply_auto_gpu_config(local_config, logger)
    local_config["gpu_devices"] = None
    disabled_top_level = {
        "refresh_idle_gpu_before_exec": False,
        "gpu_lock_enabled": False,
        "set_asr_world_size": False,
        "gpus_per_exp": 1,
        "serial_gpus_per_exp": None,
    }
    local_config.update(disabled_top_level)
    parallel = local_config.get("parallel")
    if isinstance(parallel, dict):
        parallel.update(disabled_top_level)
    logger.info(
        "SURE coordinator-only mode (local_gpu_policy=%s): skipping local GPU discovery",
        policy,
    )
    return local_config


def _apply_auto_gpu_config(
    local_config: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, Any]:
    """Resolve SURE local GPU config.

    ``auto``/``all`` means all visible GPUs. ``idle``/``free`` means GPUs that
    pass a conservative nvidia-smi free-memory and utilization filter.
    """
    parallel_config = local_config.setdefault("parallel", {})
    if not isinstance(parallel_config, dict):
        parallel_config = {}
        local_config["parallel"] = parallel_config

    raw_gpu_devices = local_config.get("gpu_devices")
    idle_requested = _is_idle_gpu_value(raw_gpu_devices)
    if not idle_requested and not _is_auto_gpu_value(raw_gpu_devices):
        return local_config

    if idle_requested:
        min_free_mib = _env_or_config_int(
            "SURE_IDLE_GPU_MIN_FREE_MIB",
            local_config,
            "idle_gpu_min_free_mib",
            _IDLE_GPU_DEFAULT_MIN_FREE_MIB,
        )
        max_utilization = _env_or_config_int(
            "SURE_IDLE_GPU_MAX_UTILIZATION",
            local_config,
            "idle_gpu_max_utilization",
            _IDLE_GPU_DEFAULT_MAX_UTILIZATION,
        )
        max_count = _env_or_config_int(
            "SURE_IDLE_GPU_MAX_COUNT",
            local_config,
            "idle_gpu_max_count",
            0,
        )
        min_count = max(
            1,
            _env_or_config_int(
                "SURE_IDLE_GPU_MIN_COUNT",
                local_config,
                "idle_gpu_min_count",
                1,
            ),
        )
        gpu_devices = _discover_idle_gpu_devices(
            logger,
            min_free_mib=min_free_mib,
            max_utilization=max_utilization,
            max_count=max_count,
        )
        parallel_config["refresh_idle_gpu_before_exec"] = _env_or_config_bool(
            "SURE_REFRESH_IDLE_GPU_BEFORE_EXEC",
            local_config,
            "refresh_idle_gpu_before_exec",
            True,
        )
        parallel_config["idle_gpu_min_free_mib"] = min_free_mib
        parallel_config["idle_gpu_max_utilization"] = max_utilization
        parallel_config["idle_gpu_allow_busy_fallback"] = _env_or_config_bool(
            "SURE_IDLE_GPU_ALLOW_BUSY_FALLBACK",
            local_config,
            "idle_gpu_allow_busy_fallback",
            False,
        )
        parallel_config["gpu_lock_enabled"] = _env_or_config_bool(
            "SURE_GPU_LOCK_ENABLED",
            local_config,
            "gpu_lock_enabled",
            True,
        )
        parallel_config["gpu_lock_dir"] = os.environ.get(
            "SURE_GPU_LOCK_DIR",
            str(local_config.get("gpu_lock_dir") or "/tmp/sure_master_gpu_locks"),
        )
        parallel_config["gpu_lock_wait_seconds"] = _env_or_config_int(
            "SURE_GPU_LOCK_WAIT_SECONDS",
            local_config,
            "gpu_lock_wait_seconds",
            30,
        )
        parallel_config["gpu_lock_poll_seconds"] = max(
            1,
            _env_or_config_int(
                "SURE_GPU_LOCK_POLL_SECONDS",
                local_config,
                "gpu_lock_poll_seconds",
                2,
            ),
        )
        if len(gpu_devices) < min_count:
            allow_fallback = _env_or_config_bool(
                "SURE_IDLE_GPU_ALLOW_BUSY_FALLBACK",
                local_config,
                "idle_gpu_allow_busy_fallback",
            )
            if allow_fallback:
                logger.warning(
                    "SURE idle GPU config found %s device(s), below min_count=%s; falling back to all visible GPUs",
                    len(gpu_devices),
                    min_count,
                )
                gpu_devices = _discover_gpu_devices(logger)
            else:
                raise RuntimeError(
                    "SURE idle GPU config found "
                    f"{len(gpu_devices)} GPU(s), below min_count={min_count}. "
                    f"Thresholds: free>={min_free_mib}MiB, utilization<={max_utilization}%. "
                    "Free a GPU, lower SURE_IDLE_GPU_MIN_FREE_MIB, or set "
                    "SURE_IDLE_GPU_ALLOW_BUSY_FALLBACK=1 to permit fallback."
                )
    else:
        gpu_devices = _discover_gpu_devices(logger)

    if not gpu_devices:
        local_config["gpu_devices"] = None
        logger.info("SURE auto GPU config found no visible GPUs; CUDA_VISIBLE_DEVICES is left unset")
        return local_config

    local_config["gpu_devices"] = gpu_devices
    if parallel_config.get("enabled", False):
        raw_max_parallel = parallel_config.get("max_parallel", 1)
        raw_gpus_per_exp = parallel_config.get("gpus_per_exp")

        if isinstance(raw_max_parallel, str) and raw_max_parallel.strip().lower() == "auto":
            gpus_per_exp = _positive_int(raw_gpus_per_exp, len(gpu_devices))
            max_parallel = max(1, len(gpu_devices) // gpus_per_exp)
        else:
            max_parallel = _positive_int(raw_max_parallel, 1)
            gpus_per_exp = _positive_int(
                raw_gpus_per_exp,
                max(1, len(gpu_devices) // max_parallel),
            )
            if idle_requested:
                max_parallel = min(
                    max_parallel,
                    max(1, len(gpu_devices) // max(1, gpus_per_exp)),
                )

        parallel_config["max_parallel"] = max_parallel
        parallel_config["gpus_per_exp"] = min(gpus_per_exp, len(gpu_devices))
        parallel_config.setdefault("set_asr_world_size", True)

    logger.info(
        "Resolved SURE GPU config: gpu_devices=%s, max_parallel=%s, gpus_per_exp=%s, set_asr_world_size=%s",
        local_config.get("gpu_devices"),
        parallel_config.get("max_parallel"),
        parallel_config.get("gpus_per_exp"),
        parallel_config.get("set_asr_world_size"),
    )
    return local_config


def _use_all_gpus_for_serial_tasks(parallel_config: dict[str, Any], max_workers: int) -> bool:
    if max_workers != 1:
        return False
    value = str(parallel_config.get("serial_gpus_per_exp", "")).strip().lower()
    return value == "all"


@register_playground("sure_master")
class SureMasterPlayground(BasePlayground):
    """SURE-backed multi-task speech self-evolution playground."""

    def __init__(self, config_dir: Path | None = None, config_path: Path | None = None):
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "sure_master"
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents.declare(
            "draft_agent",
            "debug_agent",
            "improve_agent",
            "reseach_agent",
            "knowledge_promotion_agent",
            "prefetch_agent",
            "wisdom_promotion_agent",
        )
        self.exp_index = 0
        self.initial_code: str | None = None
        self.best_score: float | None = None
        self.best_solution: str | None = None
        self.real_time_best_solution: str | None = None
        self.research_plan_and_result: list[str] = []
        self.prefetch_descriptor: str | None = None

        self.sure_config = self.config_manager.get("sure", {}) or {}
        self._source_snapshot_path: Path | None = None
        self.task_card = self._load_task_card()
        self.base_model_profile = self._resolve_base_model_profile()
        self.max_research_rounds = _non_negative_int(
            os.environ.get(
                "SURE_MAX_RESEARCH_ROUNDS",
                self.config_manager.get("max_research_rounds", 3),
            ),
            3,
        )
        self.max_improve_directions_per_round = _non_negative_int(
            os.environ.get(
                "SURE_MAX_IMPROVE_DIRECTIONS_PER_ROUND",
                self.config_manager.get("max_improve_directions_per_round", 0),
            ),
            0,
        )
        self.max_ideas_per_direction = _non_negative_int(
            os.environ.get(
                "SURE_MAX_IDEAS_PER_DIRECTION",
                self.config_manager.get("max_ideas_per_direction", 0),
            ),
            0,
        )
        self.metric_runner = self._create_metric_runner()

    def setup(self) -> None:
        self.logger.info("Setting up SURE Master playground...")
        self._setup_session()
        self._setup_agents()
        self._setup_workspace()
        self._prepare_source_snapshot_if_enabled()
        self._run_non_disk_preflight_if_enabled()
        self.logger.info("SURE Master playground setup complete")

    def _setup_session(self) -> None:
        if self.session is None:
            session_type = self.config.session.get("type", "local")
            if session_type == "docker":
                raise ValueError("Docker session is not supported for SURE Master v1")
            session_config_dict = self.config.session.get("local", {}).copy()
            session_config_dict = _apply_coordinator_gpu_policy(
                self.config,
                session_config_dict,
                self.logger,
            )
            self._inject_base_model_symlinks(session_config_dict)
            if "working_dir" in session_config_dict and "workspace_path" not in session_config_dict:
                session_config_dict["workspace_path"] = session_config_dict["working_dir"]
            elif "workspace_path" in session_config_dict and "working_dir" not in session_config_dict:
                session_config_dict["working_dir"] = session_config_dict["workspace_path"]
            if "config_dir" not in session_config_dict:
                session_config_dict["config_dir"] = str(self.config_dir)
            session_config = LocalSessionConfig(**session_config_dict)
            self.session = SureMasterLocalSession(session_config)
            self.logger.info("Using SURE Master local session")

        if not self.session.is_open:
            self.session.open()

    def _setup_workspace(self) -> None:
        for name in ("best_solution", "artifacts", "models", "metric", "working"):
            os.makedirs(os.path.join(self.session.config.workspace_path, name), exist_ok=True)
        self.logger.info("working_dir: %s", self.session.config.workspace_path)

    def _draft_candidate_type_hint(self) -> str:
        return FINE_TUNE if draft_runs_remotely(self.config) else INFERENCE

    def _run_root_dir(self) -> Path:
        workspace = Path(self.session.config.workspace_path).resolve()
        if workspace.parent.name == "workspaces":
            return workspace.parent.parent
        return workspace.parent

    def _prepare_source_snapshot_if_enabled(self) -> None:
        snapshot_config = self.sure_config.get("source_snapshot") or {}
        if not isinstance(snapshot_config, dict) or not self._truthy(snapshot_config.get("enabled")):
            return
        snapshot_root = Path(str(snapshot_config.get("path") or self._run_root_dir() / "source_snapshot"))
        snapshot_root = snapshot_root.expanduser().resolve()
        snapshot_root.mkdir(parents=True, exist_ok=True)

        def ignore(_dir: str, names: list[str]) -> set[str]:
            ignored = {
                ".git",
                ".pytest_cache",
                ".mypy_cache",
                ".ruff_cache",
                "__pycache__",
                "runs",
                "workspace",
                "workspaces",
            }
            return {name for name in names if name in ignored or name.endswith((".pyc", ".pyo"))}

        for rel in ("evomaster", "configs", "playground/sure_master"):
            source = project_root / rel
            if source.is_dir():
                shutil.copytree(
                    source,
                    snapshot_root / rel,
                    ignore=ignore,
                    dirs_exist_ok=True,
                )
        playground_init = project_root / "playground" / "__init__.py"
        if playground_init.is_file():
            target = snapshot_root / "playground" / "__init__.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(playground_init, target)
        run_py = project_root / "run.py"
        if run_py.is_file():
            shutil.copy2(run_py, snapshot_root / "run.py")

        config_path = Path(self.config_path)
        if config_path.is_file():
            try:
                rel_config = config_path.resolve().relative_to(project_root.resolve())
            except ValueError:
                rel_config = Path(config_path.name)
            target = snapshot_root / rel_config
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config_path, target)

        snapshot_config["enabled"] = True
        snapshot_config["path"] = str(snapshot_root)
        self.sure_config["source_snapshot"] = snapshot_config
        self._sync_sure_config()
        self._source_snapshot_path = snapshot_root
        self.logger.info("Prepared SURE source snapshot: %s", snapshot_root)

    def _run_non_disk_preflight_if_enabled(self) -> None:
        preflight = self.sure_config.get("preflight") or {}
        if not isinstance(preflight, dict) or not self._truthy(preflight.get("enabled")):
            return
        errors = self._non_disk_preflight_errors(preflight)
        if errors:
            payload = {"success": False, "reason_code": "preflight_failed", "errors": errors}
            self._write_staged_json("preflight_failure.json", payload)
            raise RuntimeError("SURE preflight failed: " + "; ".join(errors))
        self._write_staged_json("preflight_ok.json", {"success": True, "reason_code": "preflight_ok"})

    def _non_disk_preflight_errors(self, preflight: dict[str, Any]) -> list[str]:
        root = self._source_snapshot_path or project_root
        required_files = [
            root / "playground/sure_master/core/exp/run_exp.py",
            root / "playground/sure_master/core/playground.py",
            root / "playground/sure_master/core/utils/vc_remote.py",
            root / "playground/sure_master/tools/run_vc_sure_candidate.py",
        ]
        errors: list[str] = []
        for path in required_files:
            if not path.is_file():
                errors.append(f"required source file is missing: {path}")
                continue
            if path.stat().st_size <= 0:
                errors.append(f"required source file is empty: {path}")
                continue
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as exc:
                errors.append(f"required source file does not compile: {path}: {exc.msg}")
        if self._mixed_execution_enabled() and shutil.which("vc") is None:
            errors.append("vc command is not available on coordinator host")
        missing_env = self._missing_preflight_env_vars(preflight)
        if missing_env:
            errors.append("required runtime env var(s) are missing: " + ", ".join(missing_env))
        return errors

    def _missing_preflight_env_vars(self, preflight: dict[str, Any]) -> list[str]:
        raw_values = preflight.get("required_env_vars")
        if raw_values is None:
            return []
        if isinstance(raw_values, (list, tuple)):
            names = [str(value).strip() for value in raw_values if str(value).strip()]
        else:
            names = [part.strip() for part in str(raw_values).split(",") if part.strip()]
        env_values = dict(os.environ)
        env_file = Path(str(preflight.get("env_file") or project_root / ".env")).expanduser()
        if env_file.is_file():
            env_values.update(self._read_env_file(env_file))
        return [name for name in names if not str(env_values.get(name, "")).strip()]

    @staticmethod
    def _read_env_file(path: Path) -> dict[str, str]:
        values: dict[str, str] = {}
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return values
        for line in lines:
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            key, value = text.split("=", 1)
            key = key.strip()
            if key:
                values[key] = value.strip().strip("'\"")
        return values

    def _sync_sure_config(self) -> None:
        try:
            setattr(self.config, "sure", self.sure_config)
        except Exception:
            if isinstance(self.config, dict):
                self.config["sure"] = self.sure_config

    @staticmethod
    def _truthy(value: Any) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _load_task_card(self) -> SureTaskCard:
        task_cards_path = self.sure_config.get(
            "task_cards_path",
            str(project_root / "playground" / "sure_master" / "task_cards" / "sure_tasks.yaml"),
        )
        task_id = self.sure_config.get("task_id", "asr_en_wer")
        return resolve_task_card(task_cards_path, task_id)

    def _resolve_base_model_profile(self) -> BaseModelProfile | None:
        overrides = self.sure_config.get("base_models", {}) or {}
        override = overrides.get(self.task_card.task_id) or overrides.get(self.task_card.canonical_task)
        profile = merge_base_model_profile(self.task_card.base_model, override)
        if profile is None and bool(self.sure_config.get("require_base_model", True)):
            raise ValueError(
                f"SURE task {self.task_card.task_id!r} requires a base_model profile. "
                "Add one in the task card or configure sure.base_models.<task_id>."
            )
        validate_base_model_profile(profile, self.task_card.task_id)
        return profile

    def _inject_base_model_symlinks(self, session_config_dict: dict[str, Any]) -> None:
        profile = self.base_model_profile
        if profile is None:
            return
        symlinks = dict(session_config_dict.get("symlinks") or {})
        for name, source_path in profile.source_paths.items():
            target_path = profile.required_paths.get(name)
            if target_path:
                symlinks[str(source_path)] = str(target_path)
        session_config_dict["symlinks"] = symlinks

    def _create_metric_runner(self) -> SureMetricRunner:
        sure_root = self.sure_config.get("root", "/hpc_stor03/sjtu_home/chaolei.liu/sure")
        pythonpath = self.sure_config.get("pythonpath") or str(Path(sure_root) / "src")
        return SureMetricRunner(
            sure_root=sure_root,
            pythonpath=pythonpath,
            device=self.sure_config.get("device", "cuda"),
            cache_dir=self.sure_config.get("cache_dir"),
            validate_env=bool(self.sure_config.get("validate_env", False)),
            metric_gpu=self.sure_config.get("metric_gpu"),
        )

    def _role_paths(self) -> dict[str, str | None]:
        configured = self.sure_config.get("inputs", {}) or {}
        result = dict(self.task_card.artifact_contract)
        result.update(configured)
        return result

    def _is_valid_score(self, score: Any) -> bool:
        if score is None:
            return False
        if isinstance(score, float) and math.isnan(score):
            return False
        return True

    def compare_score(self, old_score: float | None, new_score: float | None) -> bool:
        if not self._is_valid_score(new_score):
            return False
        if not self._is_valid_score(old_score):
            return True
        if self.task_card.is_lower_better:
            return float(new_score) < float(old_score)
        return float(new_score) > float(old_score)

    def _mixed_execution_enabled(self) -> bool:
        return mixed_execution_enabled(self.config, task_id=self.task_card.task_id)

    def _candidate_limits(self) -> dict[str, int]:
        if not self._mixed_execution_enabled():
            return {INFERENCE: 0, FINE_TUNE: 0, ARCH: 0}
        limits = candidate_limits_from(self.config)
        if limits[INFERENCE] <= 0:
            limits[INFERENCE] = 4
        if limits[FINE_TUNE] <= 0:
            limits[FINE_TUNE] = 2
        if limits[ARCH] <= 0:
            limits[ARCH] = 2
        return limits

    def _idea_max_workers(self, parallel_config: dict[str, Any]) -> int:
        try:
            local_workers = max(1, int(parallel_config.get("max_parallel", 1) or 1))
        except (TypeError, ValueError):
            local_workers = 1
        if not self._mixed_execution_enabled():
            return local_workers
        remote_types = remote_candidate_types_from(self.config)
        remote_workers = remote_training_max_parallel(self.config, default=1)
        if {INFERENCE, *TRAINING_TYPES}.issubset(remote_types):
            return remote_workers
        return local_workers + remote_workers

    def _filter_and_order_ideas(
        self,
        ideas: list[Any],
        *,
        mixed_enabled: bool,
        round_candidate_counts: dict[str, int],
        round_candidate_limits: dict[str, int],
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for idea in ideas:
            candidate_type = (
                candidate_type_from_idea(idea)
                if mixed_enabled
                else INFERENCE
            )
            if mixed_enabled:
                limit = round_candidate_limits.get(candidate_type, 0)
                if limit > 0 and round_candidate_counts.get(candidate_type, 0) >= limit:
                    self.logger.info(
                        "Skipping SURE idea because %s per-round limit %s was reached: %s",
                        candidate_type,
                        limit,
                        idea,
                    )
                    continue
                round_candidate_counts[candidate_type] = (
                    round_candidate_counts.get(candidate_type, 0) + 1
                )
            entries.append({"idea": idea, "candidate_type": candidate_type})

        if mixed_enabled:
            order = {INFERENCE: 0, FINE_TUNE: 1, ARCH: 2}
            entries.sort(key=lambda item: order.get(item["candidate_type"], 99))
        return entries

    @staticmethod
    def _idea_result_key(idea: Any) -> Any:
        try:
            hash(idea)
            return idea
        except TypeError:
            return json.dumps(idea, ensure_ascii=False, sort_keys=True, default=str)

    def _create_run_exp(self, stage: str, exp_index: int) -> SureRunExp:
        if stage == "draft":
            main_agent = self.agents.draft_agent
        elif stage == "improve":
            main_agent = self.copy_agent(
                self.agents.improve_agent,
                new_agent_name=f"improve_exp_{exp_index}",
            )
        else:
            raise ValueError(f"Unknown SURE run stage: {stage}")
        debug_agent = self.copy_agent(
            self.agents.debug_agent,
            new_agent_name=f"debug_exp_{exp_index}",
        )
        return SureRunExp(
            stage=stage,
            main_agent=main_agent,
            debug_agent=debug_agent,
            config=self.config,
            exp_name=f"exp_{exp_index}_{stage}",
            task_card=self.task_card,
            base_model_profile=self.base_model_profile,
            metric_runner=self.metric_runner,
            execution_env=self._execution_env(),
            config_path=self.config_path,
        )

    def _execution_env(self) -> dict[str, str]:
        execution_env = {
            str(key): str(value)
            for key, value in (self.sure_config.get("execution_env", {}) or {}).items()
        }
        remote_icefall_python = os.environ.get("SURE_REMOTE_ICEFALL_PYTHON")
        for key in list(execution_env):
            if key == "SURE_ICEFALL_PYTHON":
                if remote_icefall_python:
                    execution_env[key] = remote_icefall_python
                    execution_env["SURE_REMOTE_ICEFALL_PYTHON"] = remote_icefall_python
                continue
            if key in os.environ and self._is_runtime_env_override_key(key):
                execution_env[key] = os.environ[key]
        local_icefall_python = os.environ.get("SURE_LOCAL_ICEFALL_PYTHON")
        if local_icefall_python:
            execution_env["SURE_LOCAL_ICEFALL_PYTHON"] = local_icefall_python
        return execution_env

    @staticmethod
    def _is_runtime_env_override_key(key: str) -> bool:
        return key.startswith(
            (
                "SURE_",
                "ASR_",
                "HF_",
                "HUGGINGFACE_",
                "MODELSCOPE_",
                "TOKENIZERS_",
                "WANDB_",
            )
        ) or key == "PYTHONPATH"

    def _staged_axes_config(self) -> dict[str, Any]:
        raw = self.sure_config.get("staged_axes") or {}
        return raw if isinstance(raw, dict) else {}

    def _staged_axes_enabled(self) -> bool:
        staged = self._staged_axes_config()
        strategy = str(self.sure_config.get("search_strategy", "")).strip().lower()
        return bool(staged.get("enabled", False)) or strategy in {
            "staged_axes",
            "axis_staged",
            "three_axis",
        }

    def _staged_start_phase(self) -> str:
        start_phase = str(
            self._staged_axes_config().get("start_phase", "draft")
        ).strip().lower()
        if start_phase not in {"draft", "arch"}:
            raise ValueError(
                "sure.staged_axes.start_phase must be 'draft' or 'arch', "
                f"got {start_phase!r}"
            )
        return start_phase

    def _staged_initial_source(self) -> tuple[str, str, Path]:
        raw_path = self.sure_config.get("initial_solution_path")
        if not raw_path or not str(raw_path).strip():
            raise ValueError(
                "sure.initial_solution_path is required when "
                "sure.staged_axes.start_phase is 'arch'"
            )
        configured_path = str(raw_path).strip()
        source_path = Path(configured_path).expanduser()
        if not source_path.is_absolute():
            source_path = project_root / source_path
        if not source_path.is_file():
            raise FileNotFoundError(
                "Configured initial solution for staged arch entry is not a "
                f"readable file: {source_path}"
            )
        try:
            source = source_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise OSError(
                "Cannot read configured initial solution for staged arch entry: "
                f"{source_path}"
            ) from exc
        if not source:
            raise ValueError(
                "Configured initial solution for staged arch entry is empty: "
                f"{source_path}"
            )
        return configured_path, source, source_path.resolve()

    @staticmethod
    def _staged_int(value: Any, default: int, minimum: int = 0) -> int:
        try:
            parsed = int(float(str(value).strip()))
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, parsed)

    @staticmethod
    def _string_dict(value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        return {str(key): str(item) for key, item in value.items() if item is not None}

    def _staged_axis_candidate_type(self, axis: str) -> str:
        if axis == "arch":
            return ARCH
        if axis == "train":
            return FINE_TUNE
        if axis == "inference":
            return INFERENCE
        raise ValueError(f"Unknown staged axis: {axis}")

    def _staged_axis_key(self, axis: str) -> str:
        return "fine_tune" if axis == "train" else axis

    def _staged_phase_config(self, phase: str) -> dict[str, Any]:
        raw = self._staged_axes_config().get(phase) or {}
        return raw if isinstance(raw, dict) else {}

    def _staged_axis_config(self, axis: str) -> dict[str, Any]:
        axes = self._staged_axes_config().get("axes") or {}
        if isinstance(axes, dict) and isinstance(axes.get(axis), dict):
            return axes[axis]
        raw = self._staged_axes_config().get(axis) or {}
        return raw if isinstance(raw, dict) else {}

    def _staged_role_paths(self, phase: str = "search") -> dict[str, str | None]:
        role_paths = self._role_paths()
        phase_config = self._staged_phase_config(phase)
        inputs = phase_config.get("inputs") or {}
        if isinstance(inputs, dict):
            role_paths.update({str(key): (None if value is None else str(value)) for key, value in inputs.items()})
        return role_paths

    def _staged_base_model_overrides(self, phase: str = "search") -> dict[str, str]:
        phase_config = self._staged_phase_config(phase)
        return self._string_dict(phase_config.get("base_model_source_paths"))

    def _staged_execution_env(
        self,
        *,
        phase: str = "search",
        stage_name: str = "",
        rung: dict[str, Any] | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> dict[str, str]:
        execution_env = self._execution_env()
        phase_config = self._staged_phase_config(phase)
        execution_env.update(self._string_dict(phase_config.get("execution_env")))
        if rung:
            execution_env.update(self._string_dict(rung.get("execution_env")))
            execution_env["SURE_STAGED_RUNG"] = str(rung.get("name") or "")
        if stage_name:
            execution_env["SURE_STAGED_STAGE"] = stage_name
        execution_env["SURE_STAGED_PHASE"] = phase
        if extra_env:
            execution_env.update({str(key): str(value) for key, value in extra_env.items()})
        return execution_env

    def _staged_draft_execution_env(self) -> dict[str, str]:
        execution_env = self._staged_execution_env(
            phase="draft",
            stage_name="stage0_draft",
        )
        use_pretrained = str(
            execution_env.get("SURE_BASELINE_USE_PRETRAINED", "")
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        if use_pretrained:
            execution_env["SURE_DURATION_AUTOTUNE"] = "0"
            max_duration = str(execution_env.get("SURE_MAX_DURATION", "")).strip().lower()
            if max_duration == "auto":
                execution_env["SURE_MAX_DURATION"] = str(
                    execution_env.get("SURE_BASELINE_DECODE_MAX_DURATION")
                    or "300"
                )
        return execution_env

    def _staged_axis_rungs(self, axis: str, final_keep: int) -> list[dict[str, Any]]:
        axis_config = self._staged_axis_config(axis)
        configured = axis_config.get("rungs")
        if isinstance(configured, list) and configured:
            rungs = [item for item in configured if isinstance(item, dict)]
        else:
            rungs = [
                {"name": "short", "keep": 8},
                {"name": "medium", "keep": final_keep},
                {"name": "final", "keep": final_keep},
            ]
        normalized: list[dict[str, Any]] = []
        for index, rung in enumerate(rungs):
            name = str(rung.get("name") or f"rung_{index + 1}")
            default_keep = final_keep if index == len(rungs) - 1 else 8
            keep = self._staged_int(rung.get("keep"), default_keep, minimum=1)
            normalized.append({**rung, "name": name, "keep": keep})
        return normalized

    def _staged_rounds_per_axis(self) -> int:
        staged = self._staged_axes_config()
        return self._staged_int(staged.get("rounds_per_axis"), 4, minimum=1)

    def _staged_ideas_per_round(self) -> int:
        staged = self._staged_axes_config()
        return self._staged_int(staged.get("ideas_per_round"), 4, minimum=1)

    def _staged_final_keep(self, axis: str) -> int:
        staged = self._staged_axes_config()
        key = "top_inference" if axis == "inference" else f"top_{axis}"
        default = 3 if axis == "inference" else 2
        return self._staged_int(staged.get(key), default, minimum=1)

    def _staged_runner_up_count(self) -> int:
        return self._staged_int(self._staged_axes_config().get("runner_up_count"), 5, minimum=0)

    def _staged_records_max_workers(self, records: list[dict[str, Any]]) -> int:
        task_count = len(records)
        if task_count <= 0:
            return 1
        if isinstance(self.config, dict):
            session_root = self.config.get("session", {}) or {}
        else:
            session_root = getattr(self.config, "session", {}) or {}
        session_config = session_root.get("local", {}) if isinstance(session_root, dict) else {}
        parallel_config = session_config.get("parallel", {}) or {}
        try:
            local_workers = max(1, int(parallel_config.get("max_parallel", 1) or 1))
        except (TypeError, ValueError):
            local_workers = 1
        if not self._mixed_execution_enabled():
            return min(local_workers, task_count)

        remote_types = remote_candidate_types_from(self.config)
        remote_workers = remote_training_max_parallel(self.config, default=1)
        candidate_types = {
            normalize_candidate_type(record.get("candidate_type"), default=INFERENCE)
            for record in records
        }
        has_remote = any(candidate_type in remote_types for candidate_type in candidate_types)
        has_local = any(candidate_type not in remote_types for candidate_type in candidate_types)
        if has_remote and not has_local:
            return min(remote_workers, task_count)
        if has_local and not has_remote:
            return min(local_workers, task_count)
        return min(local_workers + remote_workers, task_count)

    def _staged_output_dir(self) -> Path:
        path = Path(self.session.config.workspace_path) / "staged_axes"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_staged_json(self, name: str, payload: Any) -> None:
        path = self._staged_output_dir() / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    def _staged_checkpoint_dir(self) -> Path:
        path = self._staged_output_dir() / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _staged_training_candidate(record: dict[str, Any], axis: str) -> bool:
        if axis not in {"arch", "train"}:
            return False
        candidate_type = normalize_candidate_type(record.get("candidate_type"), default=INFERENCE)
        return candidate_type in {ARCH, FINE_TUNE}

    @staticmethod
    def _epoch_from_checkpoint_path(path: str | Path) -> int | None:
        match = re.search(r"epoch[-_](\d+)\.pt$", str(path))
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    @classmethod
    def _checkpoint_from_produced_artifacts(cls, payload: Any) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        artifacts = payload.get("produced_artifacts")
        if not isinstance(artifacts, dict):
            return None
        checkpoint = artifacts.get("candidate_checkpoint")
        checkpoint_dir = artifacts.get("checkpoint_dir")
        if not checkpoint:
            return None
        epoch = cls._epoch_from_checkpoint_path(str(checkpoint))
        return {
            "path": str(checkpoint),
            "checkpoint_dir": str(checkpoint_dir or Path(str(checkpoint)).parent),
            "epoch": epoch,
        }

    @classmethod
    def _checkpoint_from_nested_details(cls, payload: Any) -> dict[str, Any] | None:
        found = cls._checkpoint_from_produced_artifacts(payload)
        if found:
            return found
        if isinstance(payload, dict):
            for value in payload.values():
                found = cls._checkpoint_from_nested_details(value)
                if found:
                    return found
        elif isinstance(payload, list):
            for value in payload:
                found = cls._checkpoint_from_nested_details(value)
                if found:
                    return found
        return None

    def _staged_extract_checkpoint(self, record: dict[str, Any]) -> dict[str, Any] | None:
        staged_checkpoint = record.get("staged_checkpoint")
        if isinstance(staged_checkpoint, dict) and staged_checkpoint.get("path"):
            return dict(staged_checkpoint)

        found = self._checkpoint_from_nested_details(record.get("details"))
        if found:
            return found

        workspace = record.get("workspace")
        if not workspace:
            return None
        changes_path = Path(str(workspace)) / "artifacts" / "candidate_changes.json"
        if not changes_path.is_file():
            return None
        try:
            payload = json.loads(changes_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.debug("Could not read staged checkpoint metadata from %s: %s", changes_path, exc)
            return None
        return self._checkpoint_from_produced_artifacts(payload)

    def _staged_promote_checkpoint(
        self,
        record: dict[str, Any],
        *,
        axis: str,
        rung: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not record.get("success") or not self._staged_training_candidate(record, axis):
            return None
        checkpoint = self._staged_extract_checkpoint(record)
        if not checkpoint or not checkpoint.get("path"):
            return None
        source_path = Path(str(checkpoint["path"])).expanduser()
        if not source_path.is_file():
            self.logger.warning("Staged checkpoint source is missing for %s: %s", record.get("idea_id"), source_path)
            return None
        epoch = checkpoint.get("epoch") or self._epoch_from_checkpoint_path(source_path)
        if epoch is None:
            self.logger.warning("Could not infer checkpoint epoch for staged candidate %s: %s", record.get("idea_id"), source_path)
            return None
        idea_id = str(record.get("idea_id") or record.get("combo_id") or "candidate")
        safe_idea_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", idea_id).strip("_") or "candidate"
        rung_name = str(rung.get("name") or record.get("rung") or "rung")
        target_dir = self._staged_checkpoint_dir() / axis / safe_idea_id / rung_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"epoch-{epoch}.pt"
        if source_path.resolve() != target_path.resolve():
            shutil.copy2(source_path, target_path)
        metadata = {
            "path": str(target_path),
            "epoch": int(epoch),
            "source_path": str(source_path),
            "source_workspace": str(record.get("workspace") or ""),
            "axis": axis,
            "rung": rung_name,
            "idea_id": idea_id,
            "candidate_type": str(record.get("candidate_type") or ""),
            "checkpoint_dir": str(target_dir),
        }
        record["staged_checkpoint"] = metadata
        return metadata

    def _staged_promote_checkpoints(
        self,
        records: list[dict[str, Any]],
        *,
        axis: str,
        rung: dict[str, Any],
    ) -> list[dict[str, Any]]:
        promoted: list[dict[str, Any]] = []
        for record in records:
            updated = dict(record)
            self._staged_promote_checkpoint(updated, axis=axis, rung=rung)
            promoted.append(updated)
        return promoted

    def _staged_resume_env(
        self,
        previous_record: dict[str, Any],
        *,
        axis: str,
        current_rung: dict[str, Any],
    ) -> dict[str, str]:
        if not self._staged_training_candidate(previous_record, axis):
            return {}
        checkpoint = previous_record.get("staged_checkpoint")
        if not isinstance(checkpoint, dict):
            checkpoint = self._staged_extract_checkpoint(previous_record)
        if not checkpoint or not checkpoint.get("path") or checkpoint.get("epoch") is None:
            return {}
        path = Path(str(checkpoint["path"])).expanduser()
        if not path.is_file():
            self.logger.warning("Skipping staged resume for %s because checkpoint is missing: %s", previous_record.get("idea_id"), path)
            return {}
        target_epoch = self._string_dict(current_rung.get("execution_env")).get("SURE_STAGED_TARGET_EPOCH")
        if not target_epoch:
            target_epoch = self._string_dict(current_rung.get("execution_env")).get("SURE_MAX_TRAIN_EPOCHS")
        env = {
            "SURE_STAGED_RESUME_ENABLED": "1",
            "SURE_STAGED_RESUME_CHECKPOINT": str(path),
            "SURE_STAGED_RESUME_CHECKPOINT_DIR": str(path.parent),
            "SURE_STAGED_RESUME_EPOCH": str(checkpoint["epoch"]),
            "SURE_STAGED_RESUME_SOURCE_RUNG": str(checkpoint.get("rung") or previous_record.get("rung") or ""),
        }
        if target_epoch:
            env["SURE_STAGED_TARGET_EPOCH"] = str(target_epoch)
        return env

    def _rank_staged_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        valid = [
            record
            for record in records
            if record.get("success") and self._is_valid_score(record.get("score"))
        ]
        return sorted(
            valid,
            key=lambda record: float(record["score"]),
            reverse=not self.task_card.is_lower_better,
        )

    def _axis_research_request(self, axis: str) -> str:
        count = self._staged_ideas_per_round()
        key = self._staged_axis_key(axis)
        label = "fine_tune" if axis == "train" else axis
        task_card = getattr(self, "task_card", None)
        canonical_task = str(
            getattr(task_card, "canonical_task", "")
            or getattr(task_card, "task_id", "")
        ).lower()
        if canonical_task == "tts" and axis == "arch":
            scope = (
                "Every idea must be `[arch]`, must use `SURE_TTS_ARCH_WRAPPER`, "
                "and must truly change model structure or parameter count through "
                "the F5-TTS architecture whitelist, for example depth, ff_mult, "
                "conv_layers, qk_norm, attn_mask_enabled, or checkpoint_activations. "
                "Use the current `SURE_TTS_ARCH_INIT_MODE`; scratch is only for "
                "fair architecture screening, while final combinations use partial_load. "
                "Do not suggest training-only or inference-only changes."
            )
        elif canonical_task == "tts" and axis == "train":
            scope = (
                "Every idea must be `[fine_tune]`, must use "
                "`SURE_TTS_FINETUNE_WRAPPER`, and must change training strategy "
                "without changing model structure or parameter count."
            )
        elif canonical_task == "tts":
            scope = (
                "Every idea must be `[inference]` and must change only F5-TTS "
                "batch inference, text cleanup, chunking, sampling, speed, silence "
                "removal, or post-processing. It must not train."
            )
        elif axis == "arch":
            scope = (
                "Every idea must be `[arch]` and must truly change model structure or "
                "parameter count. Do not suggest training-only or inference-only changes."
            )
        elif axis == "train":
            scope = (
                "Every idea must be `[fine_tune]` and must change training strategy "
                "without changing model structure or parameter count."
            )
        else:
            scope = (
                "Every idea must be `[inference]` and must change only decoding, "
                "inference, sampling, normalization, or post-processing. It must not train."
            )
        examples = ",\n".join(
            f'    "{i}": "[{label}] specific {axis} idea {i}"'
            for i in range(1, count + 1)
        )
        return (
            f"Propose exactly {count} {axis} ideas for the staged SURE search.\n"
            f"{scope}\n\n"
            "Return JSON only in this shape:\n"
            "{\n"
            f'  "{key}": {{\n'
            f"{examples}\n"
            "  }\n"
            "}"
        )

    def _extract_axis_ideas(
        self,
        plan: dict[str, Any],
        axis: str,
        *,
        round_index: int,
    ) -> list[dict[str, Any]]:
        keys = [self._staged_axis_key(axis), axis]
        if axis == "train":
            keys.extend(["training", "train_strategy"])
        raw_ideas: Any = None
        for key in keys:
            if key in plan:
                raw_ideas = plan[key]
                break
        if raw_ideas is None:
            raw_ideas = plan

        pairs: list[tuple[str, Any]] = []
        if isinstance(raw_ideas, dict):
            pairs = [(str(key), value) for key, value in raw_ideas.items()]
        elif isinstance(raw_ideas, list):
            pairs = [(str(index + 1), value) for index, value in enumerate(raw_ideas)]
        else:
            pairs = [("1", raw_ideas)]

        result = []
        for idea_id, idea in pairs[: self._staged_ideas_per_round()]:
            result.append(
                {
                    "idea_id": f"{axis}_r{round_index}_{idea_id}",
                    "idea": idea,
                    "axis": axis,
                    "candidate_type": self._staged_axis_candidate_type(axis),
                }
            )
        return result

    def _generate_staged_axis_ideas(
        self,
        *,
        axis: str,
        task_description: str,
        data_preview: str,
    ) -> list[dict[str, Any]]:
        generated: list[dict[str, Any]] = []
        axis_history: list[str] = []
        for round_index in range(1, self._staged_rounds_per_axis() + 1):
            research_exp = ResearchExp(
                self.agents.reseach_agent,
                self.config,
                self.initial_code or "",
                f"exp_{self.exp_index}_research_{axis}_r{round_index}",
                self.task_card,
                self.base_model_profile,
                research_request=self._axis_research_request(axis),
            )
            self.exp_index += 1
            plan = self.execute_parallel_tasks(
                [
                    partial(
                        research_exp.run,
                        task_description=task_description,
                        data_preview=data_preview,
                        best_solution=self.best_solution or self.initial_code or "",
                        research_plan_and_result=axis_history,
                    )
                ],
                max_workers=1,
                workspace_names=[research_exp.exp_name],
            )[0]
            if isinstance(plan, Exception):
                raise plan
            ideas = self._extract_axis_ideas(plan, axis, round_index=round_index)
            generated.extend(ideas)
            axis_history.extend(
                [
                    json.dumps(plan, ensure_ascii=False, indent=2),
                    "Generated for staged axis screening; evaluation happens after all axis ideas are collected.",
                ]
            )
        self._write_staged_json(f"ideas_{axis}.json", generated)
        return generated

    def _record_from_result(
        self,
        *,
        base_record: dict[str, Any],
        result: Any,
        exp: SureRunExp,
        stage_name: str,
        rung_name: str,
    ) -> dict[str, Any]:
        record = dict(base_record)
        record["stage"] = stage_name
        record["rung"] = rung_name
        record["workspace"] = exp.workspace_path
        record["metric_feedback"] = getattr(exp, "metric_feedback", "")
        record["terminal_output"] = getattr(exp, "terminal_output", "")
        if isinstance(result, Exception):
            record.update(
                {
                    "success": False,
                    "score": None,
                    "code": base_record.get("code", ""),
                    "details": {"error": str(result)},
                    "reason_code": "exception",
                    "failure_category": "system_failure",
                }
            )
            return record
        is_success, score, uid, code, details = result
        reason_code = self._record_reason_code(details)
        record.update(
            {
                "success": bool(is_success),
                "score": score,
                "uid": str(uid),
                "code": code,
                "details": details,
                "reason_code": reason_code,
                "failure_category": "none" if is_success else self._failure_category_from_reason(reason_code),
            }
        )
        return record

    @staticmethod
    def _record_reason_code(details: Any) -> str:
        if isinstance(details, dict):
            reason = details.get("reason_code")
            if reason:
                return str(reason)
            nested = details.get("details")
            if isinstance(nested, dict) and nested.get("reason_code"):
                return str(nested["reason_code"])
        return "success"

    @staticmethod
    def _failure_category_from_reason(reason_code: str) -> str:
        system_reasons = {
            "exception",
            "preflight_failed",
            "remote_command_build_failed",
            "remote_result_missing",
            "remote_result_invalid_json",
            "remote_submit_timeout",
            "vc_command_not_found",
            "remote_exception",
            "remote_setup_failed",
            "remote_candidate_failed",
            "candidate_execution_timeout",
            "duration_probe_help_discovery_failed",
            "duration_probe_helper_cli_incompatible",
        }
        return "system_failure" if str(reason_code) in system_reasons else "candidate_failure"

    def _staged_failure_policy(self) -> str:
        policy = self._staged_axes_config().get("failure_policy", "fail_fast")
        return str(policy or "fail_fast").strip().lower()

    def _staged_rung_failure_summary(
        self,
        *,
        axis: str,
        rung: dict[str, Any],
        evaluated: list[dict[str, Any]],
        ranked: list[dict[str, Any]],
    ) -> dict[str, Any]:
        failures = [record for record in evaluated if not record.get("success")]
        counts: dict[str, int] = {}
        for record in failures:
            category = str(record.get("failure_category") or "candidate_failure")
            counts[category] = counts.get(category, 0) + 1
        return {
            "axis": axis,
            "rung": rung.get("name"),
            "success_count": len(ranked),
            "failure_count": len(failures),
            "failure_counts": counts,
            "failure_policy": self._staged_failure_policy(),
            "records": [
                {
                    "idea_id": record.get("idea_id"),
                    "workspace": record.get("workspace"),
                    "candidate_type": record.get("candidate_type"),
                    "success": bool(record.get("success")),
                    "score": record.get("score"),
                    "reason_code": record.get("reason_code"),
                    "failure_category": record.get("failure_category"),
                    "metric_feedback": record.get("metric_feedback"),
                }
                for record in evaluated
            ],
        }

    def _should_fallback_to_previous_rung(
        self,
        *,
        evaluated: list[dict[str, Any]],
        previous_ranked: list[dict[str, Any]],
    ) -> bool:
        if not previous_ranked:
            return False
        if self._staged_failure_policy() not in {
            "fallback_previous_rung_on_system_failure",
            "continue_on_system_failure",
        }:
            return False
        if not evaluated:
            return False
        return all(
            not record.get("success")
            and str(record.get("failure_category") or "") == "system_failure"
            for record in evaluated
        )

    def _should_fallback_final_candidate_failure(
        self,
        *,
        rung: dict[str, Any],
        previous_ranked: list[dict[str, Any]],
        evaluated: list[dict[str, Any]],
    ) -> bool:
        if not previous_ranked or str(rung.get("name") or "") != "final":
            return False
        policy = str(
            self._staged_axes_config().get("final_rung_candidate_failure_fallback") or ""
        ).strip().lower()
        if policy != "previous_rung":
            return False
        return bool(evaluated) and all(not record.get("success") for record in evaluated)

    def _allow_neutral_axis_fallback(self) -> bool:
        return str(self._staged_axes_config().get("allow_neutral_axis_fallback") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

    def _neutral_axis_record(self, axis: str, baseline_code: str, reason: str) -> dict[str, Any]:
        return {
            "idea_id": f"neutral_{axis}",
            "idea": f"No-op/baseline fallback for {axis}; keep the baseline behavior for this axis.",
            "candidate_type": self._staged_axis_candidate_type(axis),
            "code": baseline_code,
            "success": True,
            "score": None,
            "is_axis_fallback": True,
            "fallback_axis": axis,
            "fallback_reason": reason,
            "reason_code": "neutral_axis_fallback",
            "failure_category": "none",
        }

    def _staged_late_result_recovery_config(self) -> dict[str, Any]:
        staged_recovery = self._staged_axes_config().get("late_result_recovery")
        if isinstance(staged_recovery, dict):
            return staged_recovery
        remote = self.sure_config.get("remote_training") or {}
        if isinstance(remote, dict):
            remote_recovery = remote.get("result_recovery") or {}
            if isinstance(remote_recovery, dict):
                return remote_recovery
        return {}

    def _staged_late_result_recovery_enabled(self) -> bool:
        recovery = self._staged_late_result_recovery_config()
        if "enabled" not in recovery:
            return False
        return str(recovery.get("enabled") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

    def _remote_result_path_from_record(self, record: dict[str, Any]) -> Path | None:
        details = record.get("details")
        candidates: list[Any] = []
        if isinstance(details, dict):
            execution_info = details.get("execution_info")
            if isinstance(execution_info, dict):
                candidates.extend(
                    [
                        execution_info.get("remote_result"),
                        execution_info.get("result_path"),
                    ]
                )
            nested = details.get("details")
            if isinstance(nested, dict):
                nested_info = nested.get("execution_info")
                if isinstance(nested_info, dict):
                    candidates.extend(
                        [
                            nested_info.get("remote_result"),
                            nested_info.get("result_path"),
                        ]
                    )
        workspace = record.get("workspace")
        if workspace:
            candidates.append(Path(str(workspace)) / "metric" / "remote_training_result.json")
        for candidate in candidates:
            if candidate:
                return Path(str(candidate)).expanduser()
        return None

    def _recover_late_remote_record(self, record: dict[str, Any]) -> dict[str, Any]:
        if not self._staged_late_result_recovery_enabled():
            return record
        if record.get("success"):
            return record
        if str(record.get("failure_category") or "") != "system_failure":
            return record
        if str(record.get("reason_code") or "") not in {
            "remote_submit_timeout",
            "remote_result_missing",
            "remote_result_invalid_json",
        }:
            return record
        result_path = self._remote_result_path_from_record(record)
        if result_path is None or not result_path.is_file():
            return record
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.debug("Late remote result recovery failed for %s: %s", result_path, exc)
            return record
        if not isinstance(payload, dict):
            return record
        recovered = dict(record)
        details = payload.get("details") if isinstance(payload.get("details"), dict) else payload
        success = bool(payload.get("success"))
        reason_code = self._record_reason_code(payload) if not success else "success"
        recovered.update(
            {
                "success": success,
                "score": payload.get("score"),
                "code": payload.get("code") or record.get("code", ""),
                "details": details,
                "reason_code": reason_code,
                "failure_category": "none" if success else self._failure_category_from_reason(reason_code),
                "metric_feedback": payload.get("metric_feedback", record.get("metric_feedback", "")),
                "terminal_output": payload.get("terminal_output", record.get("terminal_output", "")),
                "recovered_late_remote_result": True,
                "late_remote_result_path": str(result_path),
                "original_reason_code": record.get("reason_code"),
            }
        )
        return recovered

    def _recover_late_remote_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self._recover_late_remote_record(record) for record in records]

    def _run_staged_records(
        self,
        *,
        records: list[dict[str, Any]],
        task_description: str,
        data_preview: str,
        previous_solution: str,
        role_paths: dict[str, str | None],
        stage_name: str,
        rung: dict[str, Any] | None = None,
        phase: str = "search",
        base_model_source_overrides: dict[str, str] | None = None,
        reuse_code: bool = False,
        data_knowledge: str = "",
        model_knowledge: str = "",
    ) -> list[dict[str, Any]]:
        tasks = []
        workspace_names = []
        exps: list[SureRunExp] = []
        rung_name = str((rung or {}).get("name") or "single")
        for i, record in enumerate(records):
            exp_idx = self.exp_index + i
            exp = self._create_run_exp("improve", exp_idx)
            exp.execution_env = self._staged_execution_env(
                phase=phase,
                stage_name=stage_name,
                rung=rung,
                extra_env=record.get("execution_env") if isinstance(record.get("execution_env"), dict) else None,
            )
            candidate_type = str(record.get("candidate_type") or INFERENCE)
            exp.candidate_stage_name = stage_name
            exp.candidate_phase = phase
            exp.candidate_rung_name = rung_name
            exp.candidate_idea_id = str(record.get("idea_id") or record.get("combo_id") or "")
            exp.execution_env.update(
                {
                    "SURE_STAGE_NAME": stage_name,
                    "SURE_PHASE_NAME": phase,
                    "SURE_RUNG_NAME": rung_name,
                    "SURE_IDEA_ID": exp.candidate_idea_id,
                    "SURE_CANDIDATE_TYPE_HINT": candidate_type,
                }
            )
            exp.enforce_candidate_type = True
            exps.append(exp)
            workspace_names.append(exp.exp_name)
            if reuse_code:
                tasks.append(
                    partial(
                        exp.run_existing_code,
                        code=str(record.get("code") or ""),
                        role_paths=role_paths,
                        candidate_type_hint=candidate_type,
                        base_model_source_overrides=base_model_source_overrides,
                    )
                )
            else:
                tasks.append(
                    partial(
                        exp.run,
                        task_description=task_description,
                        data_preview=data_preview,
                        data_knowledge=data_knowledge,
                        model_knowledge=model_knowledge,
                        previous_solution=previous_solution,
                        improve_idea=record.get("idea"),
                        role_paths=role_paths,
                        candidate_type_hint=candidate_type,
                        base_model_source_overrides=base_model_source_overrides,
                    )
                )
        self.exp_index += len(records)
        results = self.execute_parallel_tasks(
            tasks,
            max_workers=self._staged_records_max_workers(records),
            workspace_names=workspace_names,
        )
        return [
            self._record_from_result(
                base_record=record,
                result=result,
                exp=exp,
                stage_name=stage_name,
                rung_name=rung_name,
            )
            for record, result, exp in zip(records, results, exps)
        ]

    def _run_axis_screening(
        self,
        *,
        axis: str,
        task_description: str,
        data_preview: str,
        baseline_code: str,
        data_knowledge: str,
        model_knowledge: str,
    ) -> list[dict[str, Any]]:
        final_keep = self._staged_final_keep(axis)
        ideas = self._generate_staged_axis_ideas(
            axis=axis,
            task_description=task_description,
            data_preview=data_preview,
        )
        if not ideas:
            raise RuntimeError(f"Staged {axis} search produced no ideas")

        records = ideas
        previous_ranked: list[dict[str, Any]] = []
        for rung_index, rung in enumerate(self._staged_axis_rungs(axis, final_keep)):
            run_records = [dict(record) for record in records]
            if rung_index > 0:
                for record in run_records:
                    resume_env = self._staged_resume_env(
                        record,
                        axis=axis,
                        current_rung=rung,
                    )
                    if resume_env:
                        merged_env = dict(record.get("execution_env") or {})
                        merged_env.update(resume_env)
                        record["execution_env"] = merged_env
            evaluated = self._run_staged_records(
                records=run_records,
                task_description=task_description,
                data_preview=data_preview,
                previous_solution=baseline_code,
                role_paths=self._staged_role_paths("search"),
                stage_name=f"stage_{axis}",
                rung=rung,
                phase="search",
                base_model_source_overrides=self._staged_base_model_overrides("search"),
                reuse_code=rung_index > 0,
                data_knowledge=data_knowledge,
                model_knowledge=model_knowledge,
            )
            evaluated = self._recover_late_remote_records(evaluated)
            evaluated = self._staged_promote_checkpoints(evaluated, axis=axis, rung=rung)
            ranked = self._rank_staged_records(evaluated)
            self._write_staged_json(
                f"leaderboard_{axis}_{rung['name']}.json",
                {"axis": axis, "rung": rung, "records": ranked, "all_records": evaluated},
            )
            keep = min(self._staged_int(rung.get("keep"), final_keep, minimum=1), len(ranked))
            records = ranked[:keep]
            if not records:
                summary = self._staged_rung_failure_summary(
                    axis=axis,
                    rung=rung,
                    evaluated=evaluated,
                    ranked=ranked,
                )
                if self._should_fallback_to_previous_rung(
                    evaluated=evaluated,
                    previous_ranked=previous_ranked,
                ):
                    fallback_keep = min(
                        self._staged_int(rung.get("keep"), final_keep, minimum=1),
                        len(previous_ranked),
                    )
                    records = [dict(record) for record in previous_ranked[:fallback_keep]]
                    summary["fallback_used"] = True
                    summary["fallback_source"] = "previous_rung"
                    summary["fallback_records"] = [
                        {"idea_id": record.get("idea_id"), "score": record.get("score")}
                        for record in records
                    ]
                    self._write_staged_json(
                        f"{axis}_{rung['name']}_failure_summary.json",
                        summary,
                    )
                    self.logger.warning(
                        "Staged %s rung %s had only system failures; falling back to %s previous-rung candidate(s)",
                        axis,
                        rung["name"],
                        len(records),
                    )
                    continue
                if self._should_fallback_final_candidate_failure(
                    rung=rung,
                    previous_ranked=previous_ranked,
                    evaluated=evaluated,
                ):
                    fallback_keep = min(
                        self._staged_int(rung.get("keep"), final_keep, minimum=1),
                        len(previous_ranked),
                    )
                    records = [dict(record) for record in previous_ranked[:fallback_keep]]
                    summary["fallback_used"] = True
                    summary["fallback_source"] = "previous_rung_after_final_candidate_failure"
                    summary["fallback_reason_code"] = evaluated[0].get("reason_code") if evaluated else None
                    summary["fallback_records"] = [
                        {"idea_id": record.get("idea_id"), "score": record.get("score")}
                        for record in records
                    ]
                    self._write_staged_json(
                        f"{axis}_{rung['name']}_failure_summary.json",
                        summary,
                    )
                    self.logger.warning(
                        "Staged %s final rung failed; falling back to %s previous-rung candidate(s)",
                        axis,
                        len(records),
                    )
                    continue
                if (
                    self._allow_neutral_axis_fallback()
                    and self._staged_failure_policy() == "continue_on_system_failure"
                    and evaluated
                    and all(
                        not record.get("success")
                        and str(record.get("failure_category") or "") == "system_failure"
                        for record in evaluated
                    )
                ):
                    records = [
                        self._neutral_axis_record(
                            axis,
                            baseline_code,
                            reason=f"all_{rung['name']}_records_system_failure",
                        )
                    ]
                    summary["fallback_used"] = True
                    summary["fallback_source"] = "neutral_axis"
                    summary["fallback_records"] = [
                        {"idea_id": record.get("idea_id"), "score": record.get("score")}
                        for record in records
                    ]
                    self._write_staged_json(
                        f"{axis}_{rung['name']}_failure_summary.json",
                        summary,
                    )
                    self.logger.warning(
                        "Staged %s rung %s had only system failures; using neutral axis fallback",
                        axis,
                        rung["name"],
                    )
                    continue
                summary["fallback_used"] = False
                self._write_staged_json(
                    f"{axis}_{rung['name']}_failure_summary.json",
                    summary,
                )
                raise RuntimeError(f"Staged {axis} search had no successful candidates at rung {rung['name']}")
            if any(not record.get("success") for record in evaluated):
                self._write_staged_json(
                    f"{axis}_{rung['name']}_failure_summary.json",
                    self._staged_rung_failure_summary(
                        axis=axis,
                        rung=rung,
                        evaluated=evaluated,
                        ranked=ranked,
                    ),
                )
            previous_ranked = [dict(record) for record in records]
        if records and all(record.get("is_axis_fallback") for record in records):
            top_records = [dict(record) for record in records[:final_keep]]
        else:
            top_records = self._rank_staged_records(records)[:final_keep]
        self._write_staged_json(f"top_{axis}.json", top_records)
        return top_records

    def _combination_idea(
        self,
        *,
        arch_record: dict[str, Any],
        train_record: dict[str, Any],
        inference_record: dict[str, Any],
        combo_id: str,
    ) -> str:
        def _axis_line(label: str, record: dict[str, Any]) -> str:
            if record.get("is_axis_fallback"):
                return f"{label}: baseline/no-op fallback ({record.get('fallback_reason')})."
            return f"{label}: {record.get('idea')}"

        task_card = getattr(self, "task_card", None)
        canonical_task = str(
            getattr(task_card, "canonical_task", "")
            or getattr(task_card, "task_id", "")
        ).lower()
        if canonical_task == "tts":
            return (
                "[arch] Limited F5-TTS combination candidate for staged SURE search.\n"
                f"Combination id: {combo_id}\n"
                "Implement a single run_sure.py that combines these selected factors:\n"
                f"{_axis_line('Architecture idea', arch_record)}\n"
                f"{_axis_line('Training strategy idea', train_record)}\n"
                f"{_axis_line('Inference/decoding idea', inference_record)}\n\n"
                "Important: call `SURE_TTS_ARCH_WRAPPER` with "
                "`action=arch_finetune_short`, apply the selected architecture "
                "fields and selected training strategy in that wrapper invocation, "
                "use `--init-mode partial_load` or `SURE_TTS_ARCH_INIT_MODE=partial_load`, "
                "then evaluate its `final_checkpoint.pt` and `model_cfg.yaml` with "
                "`SURE_TTS_BATCH_INFER_WRAPPER` using the selected inference settings. "
                "Scratch architecture checkpoints are screening-only artifacts; do "
                "not reuse or evaluate them as final combination checkpoints. "
                "Do not patch raw F5-TTS model source files, do not call raw F5-TTS "
                "training or infer_cli entrypoints, and do not concatenate or reuse "
                "incompatible arch-only and fine-tune-only checkpoints."
            )
        return (
            "[arch] Limited combination candidate for staged SURE search.\n"
            f"Combination id: {combo_id}\n"
            "Implement a single run_sure.py that combines these selected factors:\n"
            f"Architecture idea: {arch_record.get('idea')}\n"
            f"Training strategy idea: {train_record.get('idea')}\n"
            f"Inference/decoding idea: {inference_record.get('idea')}\n\n"
            "Important: train the selected architecture using the selected training "
            "strategy to produce a fresh checkpoint, then evaluate that checkpoint "
            "with the selected inference/decoding method. Do not concatenate or "
            "reuse incompatible arch-only and train-only checkpoints."
        )

    def _combination_candidate_type(
        self,
        arch_record: dict[str, Any],
        train_record: dict[str, Any],
        inference_record: dict[str, Any],
        baseline_code: str,
    ) -> str:
        if not arch_record.get("is_axis_fallback"):
            return ARCH
        if not train_record.get("is_axis_fallback"):
            return FINE_TUNE
        if not inference_record.get("is_axis_fallback"):
            return INFERENCE
        return candidate_type_from_code(baseline_code, default=INFERENCE)

    def _run_staged_combinations(
        self,
        *,
        task_description: str,
        data_preview: str,
        baseline_code: str,
        top_arch: list[dict[str, Any]],
        top_train: list[dict[str, Any]],
        top_inference: list[dict[str, Any]],
        data_knowledge: str,
        model_knowledge: str,
    ) -> list[dict[str, Any]]:
        combo_records: list[dict[str, Any]] = []
        for combo_index, (arch_record, train_record, inference_record) in enumerate(
            product(top_arch, top_train, top_inference),
            start=1,
        ):
            combo_id = f"combo_{combo_index:02d}"
            combo_records.append(
                {
                    "combo_id": combo_id,
                    "idea_id": combo_id,
                    "idea": self._combination_idea(
                        arch_record=arch_record,
                        train_record=train_record,
                        inference_record=inference_record,
                        combo_id=combo_id,
                    ),
                    "candidate_type": self._combination_candidate_type(
                        arch_record,
                        train_record,
                        inference_record,
                        baseline_code,
                    ),
                    "arch_idea_id": arch_record.get("idea_id"),
                    "train_idea_id": train_record.get("idea_id"),
                    "inference_idea_id": inference_record.get("idea_id"),
                    "arch_idea": arch_record.get("idea"),
                    "train_idea": train_record.get("idea"),
                    "inference_idea": inference_record.get("idea"),
                }
            )
        if not combo_records:
            raise RuntimeError("No staged combinations were created")

        phase_config = self._staged_phase_config("combination")
        combination_rung = {
            "name": "combination",
            "execution_env": self._string_dict(phase_config.get("execution_env")),
        }
        evaluated = self._run_staged_records(
            records=combo_records,
            task_description=task_description,
            data_preview=data_preview,
            previous_solution=baseline_code,
            role_paths=self._staged_role_paths("combination"),
            stage_name="stage_combination",
            rung=combination_rung,
            phase="combination",
            base_model_source_overrides=self._staged_base_model_overrides("combination"),
            reuse_code=False,
            data_knowledge=data_knowledge,
            model_knowledge=model_knowledge,
        )
        evaluated = self._recover_late_remote_records(evaluated)
        ranked = self._rank_staged_records(evaluated)
        for rank, record in enumerate(ranked, start=1):
            record["search_rank"] = rank
        self._write_staged_json(
            "leaderboard_combination_search.json",
            {"records": ranked, "all_records": evaluated},
        )
        return ranked

    def _baseline_record(self, baseline_code: str) -> dict[str, Any]:
        return {
            "idea_id": "baseline_draft",
            "idea": "baseline/draft",
            "candidate_type": candidate_type_from_code(baseline_code, default=INFERENCE),
            "code": baseline_code,
            "is_baseline": True,
        }

    def _run_staged_rerank(
        self,
        *,
        phase: str,
        records: list[dict[str, Any]],
        task_description: str,
        data_preview: str,
        baseline_code: str,
    ) -> list[dict[str, Any]]:
        phase_config = self._staged_phase_config(phase)
        rung = {
            "name": phase,
            "execution_env": self._string_dict(phase_config.get("execution_env")),
        }
        evaluated = self._run_staged_records(
            records=records,
            task_description=task_description,
            data_preview=data_preview,
            previous_solution=baseline_code,
            role_paths=self._staged_role_paths(phase),
            stage_name=f"stage_{phase}",
            rung=rung,
            phase=phase,
            base_model_source_overrides=self._staged_base_model_overrides(phase),
            reuse_code=True,
        )
        evaluated = self._recover_late_remote_records(evaluated)
        ranked = self._rank_staged_records(evaluated)
        for rank, record in enumerate(ranked, start=1):
            record[f"{phase}_rank"] = rank
        self._write_staged_json(
            f"leaderboard_{phase}.json",
            {"records": ranked, "all_records": evaluated},
        )
        return ranked

    def _holdout_enabled(self) -> bool:
        holdout = self._staged_phase_config("holdout")
        if "enabled" in holdout:
            return bool(holdout.get("enabled"))
        return any(
            key in holdout
            for key in ("inputs", "execution_env", "base_model_source_paths")
        )

    def _run_staged_axes(
        self,
        *,
        task_description: str,
        data_preview: str,
        role_paths: dict[str, str | None],
    ) -> dict[str, Any]:
        self.logger.info("Running SURE staged_axes search strategy")
        start_phase = self._staged_start_phase()
        data_knowledge = ""
        model_knowledge = ""
        prefetch_exp = PrefetchExp(
            self.agents.prefetch_agent,
            self.config,
            f"exp_{self.exp_index}_prefetch",
            self.task_card,
            self.base_model_profile,
        )
        self.exp_index += 1
        prefetch_result = self.execute_parallel_tasks(
            [partial(prefetch_exp.run, task_description=task_description)],
            max_workers=1,
            workspace_names=[prefetch_exp.exp_name],
        )[0]
        if isinstance(prefetch_result, Exception):
            self.logger.warning("SURE prefetch failed non-fatally: %s", prefetch_result)
        else:
            data_knowledge, model_knowledge, self.prefetch_descriptor = prefetch_result

        baseline_executed = start_phase == "draft"
        if baseline_executed:
            draft_exp = self._create_run_exp("draft", self.exp_index)
            draft_candidate_type = self._draft_candidate_type_hint()
            draft_exp.execution_env = self._staged_draft_execution_env()
            draft_exp.candidate_stage_name = "stage0_draft"
            draft_exp.candidate_phase = "draft"
            draft_exp.candidate_rung_name = "draft"
            draft_exp.candidate_idea_id = "baseline_draft"
            draft_exp.execution_env.update(
                {
                    "SURE_STAGE_NAME": "stage0_draft",
                    "SURE_PHASE_NAME": "draft",
                    "SURE_RUNG_NAME": "draft",
                    "SURE_IDEA_ID": "baseline_draft",
                    "SURE_CANDIDATE_TYPE_HINT": draft_candidate_type,
                }
            )
            self.exp_index += 1
            draft_result = self.execute_parallel_tasks(
                [
                    partial(
                        draft_exp.run,
                        task_description=task_description,
                        data_preview=data_preview,
                        data_knowledge=data_knowledge,
                        model_knowledge=model_knowledge,
                        role_paths=self._staged_role_paths("draft"),
                        candidate_type_hint=draft_candidate_type,
                        base_model_source_overrides=self._staged_base_model_overrides("draft"),
                    )
                ],
                max_workers=1,
                workspace_names=[draft_exp.exp_name],
            )[0]
            if isinstance(draft_result, Exception):
                raise draft_result
            is_success, baseline_score, _uid, baseline_code, baseline_details = draft_result
            self.initial_code = baseline_code
            self.best_solution = baseline_code
            self.best_score = baseline_score
            self.real_time_best_solution = baseline_code
            if not is_success:
                return {
                    "status": "failed",
                    "steps": 0,
                    "search_strategy": "staged_axes",
                    "start_phase": start_phase,
                    "best_score": None,
                    "metric": self.task_card.primary_metric,
                    "error": "Draft phase failed to produce a SURE-scored solution",
                }
            baseline_payload = {
                "start_phase": start_phase,
                "execution_status": "executed",
                "score": baseline_score,
                "code": baseline_code,
                "details": baseline_details,
                "workspace": draft_exp.workspace_path,
            }
        else:
            configured_path, baseline_code, source_path = self._staged_initial_source()
            baseline_score = None
            self.initial_code = baseline_code
            self.best_solution = baseline_code
            self.best_score = None
            self.real_time_best_solution = baseline_code
            baseline_payload = {
                "start_phase": start_phase,
                "execution_status": "not_executed",
                "configured_source_path": configured_path,
                "resolved_source_path": str(source_path),
                "score": None,
                "code": baseline_code,
                "details": None,
                "workspace": None,
            }
            self.logger.info(
                "Starting staged search at arch with unexecuted initial source: %s",
                source_path,
            )

        self._write_staged_json("baseline_draft.json", baseline_payload)
        save_code_to_file(
            os.path.join(self.session.config.workspace_path, "best_solution"),
            "best_solution.py",
            baseline_code or "",
        )

        top_arch = self._run_axis_screening(
            axis="arch",
            task_description=task_description,
            data_preview=data_preview,
            baseline_code=baseline_code,
            data_knowledge=data_knowledge,
            model_knowledge=model_knowledge,
        )
        top_train = self._run_axis_screening(
            axis="train",
            task_description=task_description,
            data_preview=data_preview,
            baseline_code=baseline_code,
            data_knowledge=data_knowledge,
            model_knowledge=model_knowledge,
        )
        top_inference = self._run_axis_screening(
            axis="inference",
            task_description=task_description,
            data_preview=data_preview,
            baseline_code=baseline_code,
            data_knowledge=data_knowledge,
            model_knowledge=model_knowledge,
        )

        search_ranked = self._run_staged_combinations(
            task_description=task_description,
            data_preview=data_preview,
            baseline_code=baseline_code,
            top_arch=top_arch,
            top_train=top_train,
            top_inference=top_inference,
            data_knowledge=data_knowledge,
            model_knowledge=model_knowledge,
        )
        if not search_ranked:
            raise RuntimeError("Stage4 combination produced no successful candidates")

        selection_count = min(1 + self._staged_runner_up_count(), len(search_ranked))
        selection_inputs = [dict(record) for record in search_ranked[:selection_count]]
        if baseline_executed:
            selection_inputs.append(self._baseline_record(baseline_code))
        selection_ranked = self._run_staged_rerank(
            phase="selection",
            records=selection_inputs,
            task_description=task_description,
            data_preview=data_preview,
            baseline_code=baseline_code,
        )
        final_record = selection_ranked[0] if selection_ranked else search_ranked[0]
        self.best_solution = str(final_record.get("code") or baseline_code)
        self.best_score = final_record.get("score")
        self.real_time_best_solution = self.best_solution
        save_code_to_file(
            os.path.join(self.session.config.workspace_path, "best_solution"),
            "best_solution.py",
            self.best_solution,
        )

        holdout_ranked: list[dict[str, Any]] = []
        if self._holdout_enabled():
            holdout_inputs = [final_record]
            if baseline_executed:
                holdout_inputs.append(self._baseline_record(baseline_code))
            holdout_ranked = self._run_staged_rerank(
                phase="holdout",
                records=holdout_inputs,
                task_description=task_description,
                data_preview=data_preview,
                baseline_code=baseline_code,
            )

        summary = {
            "status": "completed",
            "steps": 0,
            "search_strategy": "staged_axes",
            "start_phase": start_phase,
            "task_id": self.task_card.task_id,
            "metric": self.task_card.primary_metric,
            "is_lower_better": self.task_card.is_lower_better,
            "baseline_score": baseline_score,
            "search_best_score": search_ranked[0].get("score"),
            "selection_best_score": final_record.get("score"),
            "holdout_scores": [
                {
                    "idea_id": record.get("idea_id"),
                    "score": record.get("score"),
                    "is_baseline": record.get("is_baseline", False),
                }
                for record in holdout_ranked
            ],
            "best_score": self.best_score,
            "best_idea_id": final_record.get("idea_id"),
            "best_combo_id": final_record.get("combo_id"),
            "staged_axes_dir": str(self._staged_output_dir()),
        }
        self._write_staged_json("summary.json", summary)
        return summary

    def run(self, task_description: str, output_file: str | None = None) -> dict:
        watchdog = TimeoutWatchdog(RUN_TIMEOUT_SECONDS)
        watchdog.start()
        self.logger.info("Watchdog started (%s seconds)", RUN_TIMEOUT_SECONDS)
        try:
            self.setup()
            self._setup_trajectory_file(output_file)
            data_preview = self._build_data_preview()
            role_paths = self._role_paths()
            self.logger.info(
                "SURE search limits: max_research_rounds=%s, max_improve_directions_per_round=%s, max_ideas_per_direction=%s",
                self.max_research_rounds,
                self.max_improve_directions_per_round or "unlimited",
                self.max_ideas_per_direction or "unlimited",
            )
            if self._staged_axes_enabled():
                return self._run_staged_axes(
                    task_description=task_description,
                    data_preview=data_preview,
                    role_paths=role_paths,
                )

            data_knowledge = ""
            model_knowledge = ""
            prefetch_exp = PrefetchExp(
                self.agents.prefetch_agent,
                self.config,
                f"exp_{self.exp_index}_prefetch",
                self.task_card,
                self.base_model_profile,
            )
            self.exp_index += 1
            prefetch_result = self.execute_parallel_tasks(
                [partial(prefetch_exp.run, task_description=task_description)],
                max_workers=1,
                workspace_names=[prefetch_exp.exp_name],
            )[0]
            if isinstance(prefetch_result, Exception):
                self.logger.warning("SURE prefetch failed non-fatally: %s", prefetch_result)
            else:
                data_knowledge, model_knowledge, self.prefetch_descriptor = prefetch_result

            draft_exp = self._create_run_exp("draft", self.exp_index)
            draft_candidate_type = self._draft_candidate_type_hint()
            self.exp_index += 1
            draft_result = self.execute_parallel_tasks(
                [
                    partial(
                        draft_exp.run,
                        task_description=task_description,
                        data_preview=data_preview,
                        data_knowledge=data_knowledge,
                        model_knowledge=model_knowledge,
                        role_paths=role_paths,
                        candidate_type_hint=draft_candidate_type,
                    )
                ],
                max_workers=1,
                workspace_names=[draft_exp.exp_name],
            )[0]
            if isinstance(draft_result, Exception):
                raise draft_result
            is_success, validation_score, _uid, self.best_solution, _details = draft_result
            self.initial_code = self.best_solution
            if not is_success:
                return {
                    "status": "failed",
                    "steps": 0,
                    "best_score": None,
                    "metric": self.task_card.primary_metric,
                    "error": "Draft phase failed to produce a SURE-scored solution",
                }

            self.best_score = validation_score
            self.real_time_best_solution = self.best_solution
            save_code_to_file(
                os.path.join(self.session.config.workspace_path, "best_solution"),
                "best_solution.py",
                self.best_solution or "",
            )

            for research_round in range(self.max_research_rounds):
                base_solution = self.best_solution or ""
                round_results: dict[str, dict[tuple, dict]] = {}

                research_exp = ResearchExp(
                    self.agents.reseach_agent,
                    self.config,
                    self.initial_code or "",
                    f"exp_{self.exp_index}_research",
                    self.task_card,
                    self.base_model_profile,
                )
                self.exp_index += 1
                research_plan = self.execute_parallel_tasks(
                    [
                        partial(
                            research_exp.run,
                            task_description=task_description,
                            data_preview=data_preview,
                            best_solution=self.best_solution or "",
                            research_plan_and_result=self.research_plan_and_result,
                        )
                    ],
                    max_workers=1,
                    workspace_names=[research_exp.exp_name],
                )[0]
                if isinstance(research_plan, Exception):
                    raise research_plan

                session_config = self.config.session.get("local", {})
                parallel_config = session_config.get("parallel", {}) or {}
                idea_max_workers = self._idea_max_workers(parallel_config)
                mixed_enabled = self._mixed_execution_enabled()
                round_candidate_counts = {INFERENCE: 0, FINE_TUNE: 0, ARCH: 0}
                round_candidate_limits = self._candidate_limits()

                direction_items = list(research_plan.items())
                if self.max_improve_directions_per_round > 0:
                    direction_items = direction_items[: self.max_improve_directions_per_round]

                for direction, direction_plan in direction_items:
                    direction_best_solution = self.best_solution
                    direction_best_score = self.best_score
                    direction_baseline_score = self.best_score
                    direction_best_idea = None
                    round_results[direction] = {}
                    ideas = list((direction_plan or {}).items())
                    if self.max_ideas_per_direction > 0:
                        ideas = ideas[: self.max_ideas_per_direction]
                    idea_entries = self._filter_and_order_ideas(
                        ideas,
                        mixed_enabled=mixed_enabled,
                        round_candidate_counts=round_candidate_counts,
                        round_candidate_limits=round_candidate_limits,
                    )
                    if not idea_entries:
                        continue

                    tasks = []
                    workspace_names = []
                    improve_exps = []
                    for i, entry in enumerate(idea_entries):
                        idea = entry["idea"]
                        exp_idx = self.exp_index + i
                        improve_exp = self._create_run_exp("improve", exp_idx)
                        improve_exps.append(improve_exp)
                        workspace_names.append(improve_exp.exp_name)
                        tasks.append(
                            partial(
                                improve_exp.run,
                                task_description=task_description,
                                data_preview=data_preview,
                                previous_solution=direction_best_solution or "",
                                improve_idea=idea,
                                role_paths=role_paths,
                                candidate_type_hint=entry["candidate_type"],
                            )
                        )
                    self.exp_index += len(idea_entries)
                    results = self.execute_parallel_tasks(
                        tasks,
                        max_workers=min(max(1, idea_max_workers), len(tasks)),
                        workspace_names=workspace_names,
                    )

                    for entry, improve_exp, result in zip(idea_entries, improve_exps, results):
                        idea = entry["idea"]
                        idea_key = self._idea_result_key(idea)
                        if isinstance(result, Exception):
                            is_success = False
                            validation_score = None
                            solution = None
                        else:
                            is_success, validation_score, _uid, solution, _details = result

                        improved = self.compare_score(direction_baseline_score, validation_score)
                        round_results[direction][idea_key] = {
                            "improved": improved,
                            "is_best_in_direction": False,
                            "score": validation_score,
                            "candidate_type": entry["candidate_type"],
                        }
                        if (
                            improved
                            and is_success
                            and solution is not None
                            and self.compare_score(direction_best_score, validation_score)
                        ):
                            direction_best_score = validation_score
                            direction_best_solution = solution
                            direction_best_idea = idea
                            save_code_to_file(
                                os.path.join(self.session.config.workspace_path, "best_solution"),
                                "best_solution.py",
                                direction_best_solution,
                            )
                            self.real_time_best_solution = direction_best_solution

                    if direction_best_idea is not None:
                        round_results[direction][self._idea_result_key(direction_best_idea)][
                            "is_best_in_direction"
                        ] = True
                    self.best_solution = direction_best_solution
                    self.best_score = direction_best_score

                self.research_plan_and_result.append(
                    json.dumps(research_plan, ensure_ascii=False, indent=2)
                )

                knowledge_exp = KnowledgePromotionExp(
                    self.agents.knowledge_promotion_agent,
                    self.config,
                    f"exp_{self.exp_index}_knowledge_promotion",
                    self.task_card,
                    self.base_model_profile,
                )
                self.exp_index += 1
                knowledge_result = self.execute_parallel_tasks(
                    [
                        partial(
                            knowledge_exp.run,
                            task_description=task_description,
                            data_preview=data_preview,
                            base_solution=base_solution,
                            best_solution=self.best_solution or "",
                            research_plan=research_plan,
                            research_round_idea_results=round_results,
                        )
                    ],
                    max_workers=1,
                    workspace_names=[knowledge_exp.exp_name],
                )[0]
                if isinstance(knowledge_result, Exception):
                    raise knowledge_result
                self.research_plan_and_result.append(knowledge_result)

            return {
                "status": "completed",
                "steps": 0,
                "task_id": self.task_card.task_id,
                "metric": self.task_card.primary_metric,
                "best_score": self.best_score,
                "is_lower_better": self.task_card.is_lower_better,
            }
        except GlobalTimeoutInterrupt:
            wisdom_exp = WisdomPromotionExp(
                self.agents.wisdom_promotion_agent,
                self.config,
                f"exp_{self.exp_index}_wisdom_promotion",
                self.task_card,
                self.base_model_profile,
            )
            wisdom_result = self.execute_parallel_tasks(
                [
                    partial(
                        wisdom_exp.run,
                        task_description=task_description,
                        best_solution=self.real_time_best_solution or "",
                    )
                ],
                max_workers=1,
                workspace_names=[wisdom_exp.exp_name],
            )
            return {
                "status": "timeout",
                "steps": 0,
                "task_id": self.task_card.task_id,
                "metric": self.task_card.primary_metric,
                "best_score": self.best_score,
                "timeout_seconds": RUN_TIMEOUT_SECONDS,
                "wisdom_promotion_result": wisdom_result,
            }
        except Exception as e:
            self.logger.error("SURE Master task execution failed: %s", e, exc_info=True)
            return {"status": "failed", "steps": 0, "error": str(e)}
        finally:
            watchdog.stop()
            self.cleanup()

    def _build_data_preview(self) -> str:
        role_paths = self._role_paths()
        lines = [
            "SURE task card:",
            self.task_card.to_prompt_json(),
            "",
            "Base model profile:",
            (
                self.base_model_profile.to_prompt_json()
                if self.base_model_profile
                else "No base model profile configured for this task."
            ),
            "",
            "Configured role paths:",
            json.dumps(role_paths, ensure_ascii=False, indent=2),
        ]
        for role, value in role_paths.items():
            if not value or value.startswith("literal:"):
                continue
            path = Path(value)
            if not path.is_absolute() and self.session is not None:
                path = Path(self.session.config.workspace_path) / path
            if path.exists() and path.is_file():
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                    lines.append(f"\nPreview for {role} ({path}):\n{text[:4000]}")
                except Exception as e:
                    lines.append(f"\nPreview for {role} unavailable: {e}")
        return "\n".join(lines)

    def execute_parallel_tasks(
        self,
        tasks: list[Callable],
        max_workers: int = 1,
        workspace_names: list[str] | None = None,
    ) -> list[Any]:
        self.logger.info("Starting execution of %d tasks with %d workers.", len(tasks), max_workers)
        results: list[Any] = [None] * len(tasks)
        session_config = self.config.session.get("local", {})
        parallel_config = session_config.get("parallel", {}) or {}
        parallel_enabled = parallel_config.get("enabled", False)
        split_workspace = parallel_config.get("split_workspace_for_exp", False)
        serial_full_gpu = _use_all_gpus_for_serial_tasks(parallel_config, max_workers)
        active_worker_tids: set[int] = set()
        tids_lock = threading.Lock()

        def wrap_task(task_func, parallel_index: int):
            def wrapped():
                current_tid = threading.get_ident()
                with tids_lock:
                    active_worker_tids.add(current_tid)
                try:
                    if parallel_enabled and self.session is not None:
                        from evomaster.agent.session.local import LocalSession

                        if isinstance(self.session, LocalSession):
                            resource_index = None if serial_full_gpu else parallel_index
                            self.session.set_parallel_index(resource_index)
                            if split_workspace:
                                main_workspace = self.session.config.workspace_path
                                exp_name = (
                                    workspace_names[parallel_index]
                                    if workspace_names and parallel_index < len(workspace_names)
                                    else f"exp_{parallel_index}"
                                )
                                exp_workspace = os.path.join(main_workspace, exp_name)
                                self.session._env.setup_exp_workspace(exp_workspace)
                                for name in ("artifacts", "models", "metric", "working"):
                                    os.makedirs(os.path.join(exp_workspace, name), exist_ok=True)
                                self.session.set_workspace_path(exp_workspace)
                                self.logger.info(
                                    "Exp %s using workspace: %s; resource_index=%s",
                                    parallel_index,
                                    exp_workspace,
                                    resource_index,
                                )
                    return task_func()
                except GlobalTimeoutInterrupt:
                    raise
                finally:
                    if parallel_enabled and self.session is not None:
                        from evomaster.agent.session.local import LocalSession

                        if isinstance(self.session, LocalSession):
                            self.session.set_parallel_index(None)
                            if split_workspace:
                                self.session.set_workspace_path(None)
                    with tids_lock:
                        active_worker_tids.discard(current_tid)

            return wrapped

        executor = ThreadPoolExecutor(max_workers=max_workers)
        future_to_index = {
            executor.submit(wrap_task(task, i)): i for i, task in enumerate(tasks)
        }
        try:
            not_done = set(future_to_index)
            while not_done:
                done, not_done = wait(not_done, timeout=0.5, return_when=FIRST_COMPLETED)
                for future in done:
                    index = future_to_index[future]
                    try:
                        results[index] = future.result()
                    except Exception as exc:
                        self.logger.error("Task %s generated an exception: %s", index, exc)
                        results[index] = exc
            return results
        finally:
            for future in future_to_index:
                future.cancel()
            with tids_lock:
                for tid in active_worker_tids:
                    try:
                        _async_raise(tid, GlobalTimeoutInterrupt)
                    except Exception as e:
                        self.logger.error("Unable to interrupt child thread %s: %s", tid, e)
            if sys.version_info >= (3, 9):
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=False)
