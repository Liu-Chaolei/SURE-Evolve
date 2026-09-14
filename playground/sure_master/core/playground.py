from __future__ import annotations

import json
import logging
import math
import os
import py_compile
import shutil
import subprocess
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from functools import partial
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
from .exp.run_exp import SureRunExp
from .exp.wisdom_promotion_exp import WisdomPromotionExp
from .utils.code import save_code_to_file
from .utils.candidate_type import (
    ARCH,
    FINE_TUNE,
    INFERENCE,
    TRAINING_TYPES,
    candidate_type_from_idea,
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
from .contracts import CandidateResult, IdeaRequest, MetricSpec, RoundResult, RungResult
from .providers import XlabIdeaProvider
from .xlab_client import XlabIdeaClient, XlabIdeaClientError
from .history import XlabHistoryJournal
from .utils.fingerprints import digest
from ..runtime.accelerator import runtime_environment
from ..tasks import get_adapter
from .datasets import split_specs
from .artifacts import load_bundle
from .search_scope import execution_contract as scoped_execution_contract
from ..runtime.training_budget import TrainingBudgetPaused, check_run_pause


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
    if sure_config.get("execution_mode") == "slurm":
        return False, "slurm"
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

    def __init__(
        self,
        config_dir: Path | None = None,
        config_path: Path | None = None,
        xlab_provider: XlabIdeaProvider | None = None,
    ):
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
        self.run_id: str | None = None
        self.initial_code: str | None = None
        self.best_score: float | None = None
        self.best_solution: str | None = None
        self.real_time_best_solution: str | None = None
        self.research_plan_and_result: list[str] = []
        self.xlab_provider = xlab_provider
        self._xlab_summary_artifacts: list[str] = []
        self._xlab_history: list[dict[str, Any]] = []
        self._xlab_history_journal: XlabHistoryJournal | None = None
        self._xlab_client_owned = False
        self._xlab_last_batch_digest: str | None = None
        self._xlab_last_batch_artifacts: list[str] = []
        self._xlab_idea_metadata: dict[Any, dict[str, Any]] = {}
        self.prefetch_descriptor: str | None = None

        from .full_training import promote_full_training_to_search
        self.sure_config = promote_full_training_to_search(self.config_manager.get("sure", {}) or {})
        self._sync_sure_config()
        self._source_snapshot_path: Path | None = None
        self.task_card = self._load_task_card()
        self.base_model_profile = self._resolve_base_model_profile()
        self.task_adapter = None
        if self.task_card.canonical_task in {"asr", "tts", "sd"} or self.sure_config.get("adapter"):
            self.task_adapter = get_adapter(self.task_card.canonical_task, self.sure_config.get("adapter"))
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
        self._ensure_run_id()
        self._setup_agents()
        self._setup_workspace()
        self._setup_xlab_history()
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

    def _ensure_run_id(self) -> str:
        if self.run_id:
            return self.run_id
        run_dir = Path(getattr(self, "run_dir", "") or "").resolve()
        task_id = str(getattr(self, "task_id", None) or self.task_card.task_id)
        identity = f"{run_dir}:{task_id}"
        self.run_id = "sure-" + digest(identity)[len("sha256:") : 24]
        return self.run_id

    def _create_prefetch_exp(self, exp_index: int) -> PrefetchExp:
        return PrefetchExp(
            self.agents.prefetch_agent,
            self.config,
            f"exp_{exp_index}_prefetch",
            self.task_card,
            self.base_model_profile,
        )

    def _setup_xlab_history(self) -> None:
        if not self._xlab_enabled():
            return
        xlab_config = self.config_manager.get("xlab", {}) or {}
        configured_path = str(xlab_config.get("history_path") or "artifacts/xlab_history.json")
        history_path = Path(configured_path)
        if history_path.is_absolute() or ".." in history_path.parts:
            raise ValueError("xlab.history_path must be workspace-relative")
        self._xlab_history_journal = XlabHistoryJournal(
            Path(self.session.config.workspace_path) / history_path
        )
        self._xlab_history = self._xlab_history_journal.round_history()
        self._xlab_summary_artifacts = self._xlab_history_journal.summary_references()

    def _record_xlab_event(self, event: str, payload: dict[str, Any]) -> str | None:
        if self._xlab_history_journal is None:
            return None
        return self._xlab_history_journal.append(event, payload)

    def _draft_candidate_type_hint(self) -> str:
        remote = draft_runs_remotely(self.config)
        adapter = getattr(self, "task_adapter", None)
        if adapter is not None:
            return adapter.baseline_candidate_type(self.sure_config, remote)
        return FINE_TUNE if remote else INFERENCE

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
            self._write_run_json("preflight_failure.json", payload)
            raise RuntimeError("SURE preflight failed: " + "; ".join(errors))
        self._write_run_json("preflight_ok.json", {"success": True, "reason_code": "preflight_ok"})

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
            device=(self.sure_config.get("metric_runtime") or {}).get("device", self.sure_config.get("device", "cpu")),
            python=(self.sure_config.get("metric_runtime") or {}).get("python"),
            tts_runtime=(self.sure_config.get("metric_runtime") or {}).get("tts_runtime", "node_local"),
            timeout_seconds=(self.sure_config.get("metric_runtime") or {}).get("timeout_seconds", 21600),
            cache_dir=self.sure_config.get("cache_dir"),
            validate_env=bool(self.sure_config.get("validate_env", False)),
            metric_gpu=self.sure_config.get("metric_gpu"),
        )

    def _role_paths(self) -> dict[str, str | None]:
        configured = self.sure_config.get("inputs", {}) or {}
        result = dict(self.task_card.artifact_contract)
        result.update(configured)
        search = split_specs(self.sure_config).get("search")
        if search:
            result.update(search.roles)
        return result

    def _is_valid_score(self, score: Any) -> bool:
        if score is None:
            return False
        try:
            return math.isfinite(float(score))
        except (TypeError, ValueError):
            return False

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
        if (
            str(self.sure_config.get("search_scope", "architecture_only")).lower()
            == "architecture_only"
            and ARCH in remote_types
        ):
            return remote_workers
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
        architecture_only = (
            str(self.sure_config.get("search_scope", "architecture_only")).lower()
            == "architecture_only"
        )
        for idea in ideas:
            metadata = getattr(self, "_xlab_idea_metadata", {}).get(self._idea_result_key(idea), {})
            if architecture_only:
                declared = metadata.get("candidate_type")
                if declared and declared != ARCH:
                    raise ValueError(
                        "architecture_only search received a non-arch XLab candidate: "
                        + str(declared)
                    )
                candidate_type = ARCH
            else:
                candidate_type = (
                    metadata.get("candidate_type")
                    or (candidate_type_from_idea(idea) if mixed_enabled else INFERENCE)
                )
            if mixed_enabled and not metadata and not architecture_only:
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

        if mixed_enabled and not architecture_only:
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
        env = {str(k): str(v) for k, v in (self.sure_config.get("execution_env") or {}).items()}
        env.update(runtime_environment(self.sure_config.get("runtime") or {}))
        adapter = getattr(self, "task_adapter", None)
        if adapter is None:
            card = getattr(self, "task_card", None)
            task = getattr(card, "canonical_task", self.sure_config.get("task_id", "asr_en_wer").split("_", 1)[0])
            adapter = get_adapter(task, self.sure_config.get("adapter"))
        env.update(adapter.environment(self.sure_config))
        if self.sure_config.get("require_model_artifact", False):
            env["SURE_REQUIRE_MODEL_ARTIFACT"] = "1"
        return env


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


    @staticmethod
    def _string_dict(value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        return {str(key): str(item) for key, item in value.items() if item is not None}






    def _xlab_enabled(self) -> bool:
        xlab_config = self.config_manager.get("xlab", {}) or {}
        provider_config = xlab_config.get("idea_provider", {}) or {}
        return bool(xlab_config.get("enabled", False) and provider_config.get("enabled", False))

    def _configure_xlab_provider(self) -> None:
        if not self._xlab_enabled() or self.xlab_provider is not None:
            return
        xlab_config = self.config_manager.get("xlab", {}) or {}
        provider_config = xlab_config.get("idea_provider", {}) or {}
        command = provider_config.get("command")
        if not isinstance(command, list) or not command or any(not isinstance(item, str) or not item.strip() for item in command):
            raise RuntimeError("xlab.idea_provider.command must be a configured argv list")
        timeout = int(provider_config.get("timeout_seconds", xlab_config.get("launcher_timeout_seconds", 3600)))
        receipt_path = (
            provider_config.get("receipt_path")
            or xlab_config.get("receipt_path")
            or "artifacts/xlab_operations.json"
        )
        self.xlab_provider = XlabIdeaClient(
            command,
            timeout_seconds=timeout,
            receipt_path=receipt_path,
            workspace_root=self.session.config.workspace_path,
            environment={str(k): str(v) for k, v in (provider_config.get("environment") or {}).items()},
        )
        self._xlab_client_owned = True

    def _close_xlab_provider(self) -> None:
        if self._xlab_client_owned and self.xlab_provider is not None:
            self.xlab_provider.close()
            self.xlab_provider = None
            self._xlab_client_owned = False

    def _summarize_xlab_result(self, result: RoundResult):
        directory = Path(self.session.config.workspace_path) / "artifacts/xlab_rounds"
        directory.mkdir(parents=True, exist_ok=True)
        key = digest(asdict(result)).removeprefix("sha256:")
        (directory / f"{key}.request.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
        if (directory / f"{key}.summary.json").exists():
            from .contracts import RoundSummary
            return RoundSummary(**json.loads((directory / f"{key}.summary.json").read_text()))
        summary = self.xlab_provider.summarize(result)
        (directory / f"{key}.summary.json").write_text(json.dumps(asdict(summary), indent=2) + "\n")
        return summary

    def _xlab_request(
        self,
        *,
        task_description: str,
        search_mode: str,
        round_index: int,
        requested_idea_count: int,
        axis: str | None = None,
    ) -> IdeaRequest:
        metric_name = str(getattr(self.task_card, "primary_metric", "metric"))
        metric_direction = "lower" if bool(getattr(self.task_card, "is_lower_better", True)) else "higher"
        run_id = self._ensure_run_id()
        axis_index = {"arch": 0, "train": 1, "inference": 2}.get(axis)
        request_id = f"{run_id}-{search_mode}-{axis or 'ordinary'}-r{round_index}"
        task_card = self.task_card.to_dict() if hasattr(self.task_card, "to_dict") else {}
        base_model = self.base_model_profile.to_dict() if self.base_model_profile else {}
        parent_lineage = list(self._xlab_summary_artifacts) + list(self._xlab_last_batch_artifacts)
        history_digest = digest(self._xlab_history)
        current_best = {
            "solution_digest": digest(self.best_solution or self.initial_code or ""),
            "score": self.best_score,
            "model_artifact": getattr(self, "best_model_artifact", {}),
            "implementation": self.best_solution or self.initial_code or "",
            "evaluation_scope": {"search": self._role_paths(), "manifest": self._execution_env().get("SURE_EVAL_MANIFEST")},
        }
        # Recover lineage from persisted history, including after controller restart.
        for history in reversed(self._xlab_history):
            for candidate in history.get("candidates", []):
                native = candidate.get("idea", {}).get("native_artifact")
                checkpoint = current_best["model_artifact"].get("model_artifact")
                if checkpoint and not any(rung.get("checkpoint_artifact") == checkpoint for rung in candidate.get("rungs", [])):
                    continue
                if (candidate.get("code_digest") == current_best["solution_digest"]
                        and candidate.get("status") == "success" and native):
                    # Public materialization stores risks as a list; native search
                    # consumes a text field. Preserve the original artifact separately.
                    mature = dict(native)
                    if isinstance(mature.get("risks"), list):
                        mature["risks"] = "\n".join(str(risk) for risk in mature["risks"])
                    mature.setdefault("tags", [])
                    mature.setdefault("root_domains", [])
                    current_best.update(native_idea=mature, native_idea_digest=digest(mature),
                                        native_artifact_digest=digest(native),
                                        idea_id=candidate["idea_id"])
                    break
            if "native_idea" in current_best:
                break
        artifact_path = current_best["model_artifact"].get("model_artifact")
        if artifact_path and Path(artifact_path).is_file():
            manifest_path = Path(artifact_path)
            current_best["model_manifest"] = load_bundle(manifest_path)
            for name in ("candidate_changes.json", "official_baseline.json"):
                profile_path = manifest_path.parent / "artifacts" / name
                if profile_path.is_file():
                    current_best["actual_model_configuration"] = json.loads(profile_path.read_text())
                    break
        from .ablation import policy, research_view
        ablation = policy(self.sure_config)
        baseline = getattr(self, "_ablation_baseline", {})
        if not ablation["use_feedback"]:
            current_best = {"implementation": baseline.get("code", self.initial_code or ""),
                            "solution_digest": digest(baseline.get("code", self.initial_code or "")),
                            "model_artifact": {"model_artifact": baseline.get("model_artifact")}}
            if baseline.get("model_artifact"):
                current_best["model_manifest"] = load_bundle(baseline["model_artifact"])
        view = research_view(self.sure_config, current_best, current_best,
                             self._xlab_history, self._xlab_summary_artifacts, parent_lineage)
        history_digest = digest(view["prior_rounds"])
        parent_lineage = view["parent_lineage"]
        xlab_config = self.config_manager.get("xlab", {}) or {}
        generation_policy = {"max_attempts": (xlab_config.get("idea_generation") or {}).get("max_attempts", 8)}
        if "ablation" in self.sure_config:
            generation_policy.update(ablation=ablation, allow_partial_batch=True)
        execution_contract = scoped_execution_contract(self.task_adapter.context(), self.sure_config)
        payload = {
            "request_id": request_id,
            "sure_run_id": run_id,
            "task_id": self.task_card.task_id,
            "task_description": task_description,
            "search_mode": search_mode,
            "axis": axis,
            "axis_index": axis_index,
            "phase": "research",
            "round_index": round_index,
            "requested_idea_count": requested_idea_count,
            "metric": {"name": metric_name, "direction": metric_direction},
            "current_best": current_best,
            "task_card": task_card,
            "base_model_profile": base_model,
            "execution_contract": execution_contract,
            "generation_policy": generation_policy,
            "history_artifacts": view["history_artifacts"],
            "prior_rounds": view["prior_rounds"],
            "history_digest": history_digest,
            "parent_lineage": parent_lineage,
        }
        request = IdeaRequest(
            request_id=request_id,
            sure_run_id=run_id,
            task_id=str(self.task_card.task_id),
            task_description=task_description,
            search_mode=search_mode,
            axis=axis,
            axis_index=axis_index,
            phase="research",
            round_index=round_index,
            requested_idea_count=requested_idea_count,
            metric=MetricSpec(name=metric_name, direction=metric_direction),
            input_digest=digest(payload),
            current_best=current_best,
            task_card=task_card,
            execution_contract=execution_contract,
            generation_policy=generation_policy,
            history_artifacts=view["history_artifacts"],
            prior_rounds=view["prior_rounds"],
            history_digest=history_digest,
            parent_lineage=parent_lineage,
            base_model_profile=base_model,
        )
        self._record_xlab_event(
            "request_accepted",
            {
                "request_id": request.request_id,
                "search_mode": request.search_mode,
                "axis": request.axis,
                "research_round": request.round_index,
                "input_digest": request.input_digest,
                "history_digest": request.history_digest,
                "parent_lineage": request.history_artifacts,
            },
        )
        return request

    def _xlab_ordinary_plan(self, task_description: str, round_index: int) -> dict[str, Any]:
        if self.xlab_provider is None:
            raise RuntimeError("XLab provider is required when XLab idea provider is enabled")
        request = self._xlab_request(
            task_description=task_description,
            search_mode="ordinary",
            round_index=round_index,
            requested_idea_count=int((self.sure_config.get("search_budget") or {}).get("ideas_per_round", 4)),
        )
        batch_path = Path(self.session.config.workspace_path) / "artifacts/xlab_batches" / f"{request.request_id}.json"
        if batch_path.exists():
            from .contracts import IdeaBatch, IdeaItem, IdeaSpec, validate_idea_batch
            payload = json.loads(batch_path.read_text())
            payload["ideas"] = [IdeaItem(**{**item, "spec": IdeaSpec(**item["spec"])}) for item in payload["ideas"]]
            batch = IdeaBatch(**payload)
            validate_idea_batch(batch, request)
        else:
            batch = self.xlab_provider.generate(request)
        from .contracts import validate_idea_batch
        validate_idea_batch(batch, request)
        batch_path.parent.mkdir(parents=True, exist_ok=True)
        batch_path.write_text(json.dumps(asdict(batch), ensure_ascii=False, indent=2) + "\n")
        self._xlab_last_batch_digest = batch.batch_digest
        self._xlab_last_batch_artifacts = [idea.artifact_id for idea in batch.ideas]
        self._record_xlab_event(
            "batch_published",
            {
                "request_id": request.request_id,
                "request_digest": request.input_digest,
                "batch_digest": batch.batch_digest,
                "idea_artifacts": [idea.artifact_id for idea in batch.ideas],
                "idea_ids": [idea.idea_id for idea in batch.ideas],
            },
        )
        plan: dict[str, Any] = {"xlab": {}}
        for idea in batch.ideas:
            instructions = self._xlab_instructions(idea)
            plan["xlab"][idea.idea_id] = instructions
            self._xlab_idea_metadata[(idea.idea_id, instructions)] = {
                "idea": {**json.loads(instructions), "native_artifact": idea.native_artifact,
                         "novelty": idea.novelty, "evidence_refs": idea.evidence_refs},
                "idea_id": idea.idea_id,
                "artifact_id": idea.artifact_id,
                "artifact_digest": idea.artifact_digest,
                "axis": idea.axis,
                "candidate_type": idea.candidate_type,
            }
        return plan

    @staticmethod
    def _xlab_instructions(idea) -> str:
        return json.dumps({"title": idea.title, "hypothesis": idea.hypothesis,
                           "mechanism": idea.mechanism, "candidate_type": idea.candidate_type,
                           "spec": asdict(idea.spec)}, ensure_ascii=False, indent=2)


    def _workspace_ref(self, value: Any) -> str | None:
        if not value:
            return None
        workspace = Path(self.session.config.workspace_path).resolve()
        path = Path(str(value))
        if not path.is_absolute():
            return path.as_posix()
        try:
            return path.resolve().relative_to(workspace).as_posix()
        except ValueError:
            return None


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
            "worker_failed",
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


    def _commit_best_state(self) -> None:
        """The state manifest is the atomic source of truth; .py is a convenience copy."""
        target = Path(self.session.config.workspace_path) / "best_solution"
        target.mkdir(parents=True, exist_ok=True)
        state = {"score": self.best_score, "code": self.best_solution,
                 "model_artifact": self.best_model_artifact}
        pending = target / ".best_state.pending"
        pending.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        os.replace(pending, target / "best_state.json")

    def _write_run_json(self, name: str, payload: Any) -> None:
        target = Path(self.session.config.workspace_path) / "metric" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    def _final_evaluation(self, baseline: dict, candidates: list[dict]) -> dict:
        specs = split_specs(self.sure_config)
        if "selection" not in specs:
            return {}
        eligible = [r for r in candidates if r.get("model_artifact") and self._is_valid_score(r.get("score"))]
        eligible.sort(key=lambda r: r["score"], reverse=not self.task_card.is_lower_better)
        if not baseline.get("model_artifact"):
            raise RuntimeError("Final evaluation requires a retained baseline")

        def evaluate(record: dict, phase: str) -> dict:
            load_bundle(record["model_artifact"])
            split = specs[phase]
            exp = self._create_run_exp("improve", self.exp_index)
            self.exp_index += 1
            exp.candidate_phase = phase
            exp.candidate_idea_id = record["idea_id"]
            exp.execution_env.update(self.task_adapter.phase_environment(split))
            exp.execution_env["SURE_FROZEN_MODEL_ARTIFACT"] = record["model_artifact"]
            exp.execution_env.pop("SURE_PARENT_MODEL_ARTIFACT", None)
            result = self.execute_parallel_tasks(
                [partial(exp.run_existing_code, code=self.task_adapter.frozen_code(record["model_artifact"]),
                         role_paths={**self._role_paths(), **split.roles}, candidate_type_hint=INFERENCE,
                         base_model_source_overrides=split.base_model_source_paths)],
                max_workers=1, workspace_names=[exp.exp_name],
            )[0]
            if isinstance(result, Exception):
                raise result
            success, score, _uid, _code, _details = result
            if not success or not self._is_valid_score(score):
                raise RuntimeError(f"Frozen {phase} failed for {record['idea_id']}")
            return {**record, "score": score, "phase": phase, "workspace": exp.workspace_path}

        selection = [evaluate(r, "selection") for r in [baseline, *eligible[:2]]]
        selection.sort(key=lambda r: r["score"], reverse=not self.task_card.is_lower_better)
        winner = selection[0]
        self.best_score, self.best_solution = winner["score"], winner["code"]
        self.best_model_artifact = {"model_artifact": winner["model_artifact"]}
        holdout = []
        if "holdout" in specs and not self.sure_config.get("defer_holdout", False):
            records = [baseline] if winner["idea_id"] == baseline["idea_id"] else [baseline, winner]
            holdout = [evaluate(r, "holdout") for r in records]
        result = {"selection": selection, "holdout": holdout, "winner": winner["idea_id"]}
        self._write_run_json("final_evaluation.json", result)
        return result


    def run(self, task_description: str, output_file: str | None = None, *, ideas_only: bool = False) -> dict:
        if ideas_only and not self.sure_config.get('initial_baseline_run'):
            raise ValueError('Ideas-only execution requires a verified imported baseline')
        watchdog = TimeoutWatchdog(int(self.sure_config.get("controller_timeout_seconds", RUN_TIMEOUT_SECONDS)))
        watchdog.start()
        self.logger.info("Watchdog started (%s seconds)", RUN_TIMEOUT_SECONDS)
        try:
            strategy = str(self.sure_config.get("search_strategy", "ordinary")).lower()
            if strategy not in {"ordinary", "origin"} or (self.sure_config.get("staged_axes") or {}).get("enabled"):
                raise ValueError("staged_axes is retired; migrate to ordinary + XLab")
            if not self._xlab_enabled():
                raise ValueError("ordinary search requires the XLab idea provider")
            self.sure_config["search_strategy"] = "ordinary"
            if self.task_adapter is None:
                self.task_adapter = get_adapter(self.task_card.canonical_task, self.sure_config.get("adapter"))
            preflight = self.sure_config.get("preflight", False)
            if preflight is True or (isinstance(preflight, dict) and preflight.get("enabled")):
                from ..tools.preflight import check_config
                remote_complete, _remote_diagnostics = complete_remote_coverage_from(self.config)
                check_config({"sure": self.sure_config, "xlab": self.config_manager.get("xlab", {})},
                             check_model=(
                                 self.sure_config.get("execution_mode") != "slurm"
                                 and not remote_complete
                             ))
            self.setup()
            self._configure_xlab_provider()
            self._setup_trajectory_file(output_file)
            data_preview = self._build_data_preview()
            role_paths = self._role_paths()

            state_path = Path(self.session.config.workspace_path) / "metric/controller_state.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            state_keys = ("exp_index", "run_id", "initial_code", "best_score", "best_solution", "baseline_score",
                          "best_model_artifact", "real_time_best_solution", "research_plan_and_result",
                          "_xlab_history", "_xlab_summary_artifacts", "_xlab_last_batch_artifacts")
            if state:
                if state.get("contract_digest") != digest(self.sure_config):
                    raise ValueError("Existing run has a different execution contract; use a new run directory")
                for key in state_keys:
                    setattr(self, key, state["attributes"][key])
                baseline_record = state["baseline"]
                scored_candidates = state["candidates"]
                successful_training_candidates = state["successful_training_candidates"]
            elif self.sure_config.get("initial_baseline_run"):
                from .baseline_import import import_baseline
                baseline_record, self.best_model_artifact = import_baseline(
                    Path(self.sure_config["initial_baseline_run"]), self.sure_config,
                    Path(self.session.config.workspace_path))
                self.baseline_score = self.best_score = baseline_record["score"]
                self.initial_code = self.best_solution = self.real_time_best_solution = baseline_record["code"]
                scored_candidates = []
                successful_training_candidates = 0
            else:
                data_knowledge = ""
                model_knowledge = ""
                prefetch_exp = self._create_prefetch_exp(self.exp_index)
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

                if not self._is_valid_score(validation_score):
                    raise RuntimeError("Baseline returned an invalid score")
                self.best_score = validation_score
                self.baseline_score = validation_score
                self.best_model_artifact = (_details or {}).get("produced_artifacts", {})
                baseline_record = {"idea_id": "baseline", "score": validation_score,
                                   "model_artifact": self.best_model_artifact.get("model_artifact"), "code": self.best_solution}
                scored_candidates: list[dict[str, Any]] = []
                successful_training_candidates = 0
                self.real_time_best_solution = self.best_solution
                save_code_to_file(
                    os.path.join(self.session.config.workspace_path, "best_solution"),
                    "best_solution.py",
                    self.best_solution or "",
                )

            self._ablation_baseline = dict(baseline_record)
            from .ablation import policy
            feedback_enabled = policy(self.sure_config)["use_feedback"]
            search_policy = self.sure_config.get("search_budget") or {}
            minimum_rounds = int(search_policy.get("min_rounds", self.max_research_rounds))
            maximum_rounds = int(search_policy.get("max_rounds", self.max_research_rounds))
            patience = int(search_policy.get("patience", 3))
            from .utils.slurm import atomic_json
            def checkpoint_controller(completed_rounds, stale):
                atomic_json(state_path, {"contract_digest": digest(self.sure_config),
                    "attributes": {key: getattr(self, key) for key in state_keys},
                    "baseline": baseline_record, "candidates": scored_candidates,
                    "successful_training_candidates": successful_training_candidates,
                "search_best_score": self.best_score,
                "best_score_split": "search",
                    "completed_rounds": completed_rounds, "stale_rounds": stale})
            stale_rounds = state.get("stale_rounds", 0)
            completed_rounds = state.get("completed_rounds", 0)
            checkpoint_controller(completed_rounds, stale_rounds)
            if self.sure_config.get("baseline_only"):
                return {"status": "baseline_ready", "baseline": baseline_record}
            if completed_rounds >= minimum_rounds and stale_rounds >= patience:
                maximum_rounds = completed_rounds
            for research_round in range(completed_rounds, maximum_rounds):
                round_start_score = self.best_score
                base_solution = self.best_solution or ""
                round_results: dict[str, dict[tuple, dict]] = {}

                round_model = dict(self.best_model_artifact) if feedback_enabled else {"model_artifact": baseline_record["model_artifact"]}
                research_plan = self._xlab_ordinary_plan(task_description, research_round + 1)

                if ideas_only:
                    return {'status': 'ideas_ready', 'round': research_round + 1,
                            'batch_digest': self._xlab_last_batch_digest,
                            'candidate_count': sum(len(v) for v in research_plan.values()),
                            'training_performed': False}

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
                    if self.max_ideas_per_direction > 0 and not self._xlab_enabled():
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
                        improve_exp.enforce_candidate_type = True
                        improve_exp.candidate_phase = "search"
                        improve_exp.candidate_idea_id = self._xlab_idea_metadata.get(self._idea_result_key(idea), {}).get("idea_id", "")
                        if (entry["candidate_type"] == INFERENCE or self.sure_config.get("task", {}).get("training", {}).get("recipe") == "diarizen.evolution.v1") and round_model.get("model_artifact"):
                            improve_exp.execution_env["SURE_PARENT_MODEL_ARTIFACT"] = round_model["model_artifact"]
                        else:
                            improve_exp.execution_env.pop("SURE_PARENT_MODEL_ARTIFACT", None)
                        improve_exps.append(improve_exp)
                        workspace_names.append(improve_exp.exp_name)
                        tasks.append(
                            partial(
                                improve_exp.run,
                                task_description=task_description,
                                data_preview=data_preview,
                                previous_solution=(direction_best_solution if feedback_enabled else baseline_record["code"]) or "",
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
                            _details = {"reason_code": "exception", "error": str(result)}
                        else:
                            is_success, validation_score, _uid, solution, _details = result

                        improved = self.compare_score(direction_baseline_score, validation_score)
                        if is_success and entry["candidate_type"] in TRAINING_TYPES:
                            successful_training_candidates += 1
                        metadata = self._xlab_idea_metadata.get(idea_key, {})
                        if is_success:
                            scored_candidates.append({"idea_id": metadata.get("idea_id", str(idea_key)),
                                                      "score": validation_score, "code": solution,
                                                      "model_artifact": (_details or {}).get("produced_artifacts", {}).get("model_artifact")})
                        reason = str((_details or {}).get("reason_code") or ("success" if is_success else "execution_failed"))
                        round_results[direction][idea_key] = {
                            "improved": improved,
                            "is_best_in_direction": False,
                            "score": validation_score,
                            "success": bool(is_success),
                            "candidate_type": entry["candidate_type"],
                            "idea_id": metadata.get("idea_id", str(idea_key)),
                            "artifact_id": metadata.get("artifact_id"),
                            "artifact_digest": metadata.get("artifact_digest"),
                            "reason_code": reason,
                            "failure_category": None if is_success else self._failure_category_from_reason(reason),
                            "metric_feedback": improve_exp.metric_feedback,
                            "runtime_seconds": (_details or {}).get("runtime_seconds"),
                            "produced_artifacts": (_details or {}).get("produced_artifacts", {}),
                            "idea": metadata.get("idea", {}),
                            "code": solution if isinstance(solution, str) else "",
                            "workspace": improve_exp.workspace_path,
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
                            self.best_model_artifact = (_details or {}).get("produced_artifacts", {})
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
                    self._commit_best_state()
                    self._record_xlab_event(
                        "best_committed",
                        {
                            "axis": None,
                            "research_round": research_round + 1,
                            "score": self.best_score,
                            "solution_digest": digest(self.best_solution or ""),
                        },
                    )

                if self.xlab_provider is not None and feedback_enabled:
                    candidates = []
                    for direction, records in round_results.items():
                        for idea_key, record in records.items():
                            candidates.append(
                                CandidateResult(
                                    idea_id=str(record.get("idea_id") or idea_key),
                                    idea_artifact_id=str(record.get("artifact_id") or idea_key),
                                    final_status="success" if record.get("success") else "failed",
                                    code_digest=digest(record.get("code") or ""),
                                    workspace_ref=record.get("workspace"),
                                    candidate_type=record.get("candidate_type"),
                                    failure_category=record.get("failure_category"),
                                    reason_code=record.get("reason_code"),
                                    improved=bool(record.get("improved")),
                                    idea=record.get("idea", {}),
                                    metric_feedback=record.get("metric_feedback", ""),
                                    rungs=[RungResult(
                                        name="search", success=bool(record.get("success")),
                                        score=record.get("score"),
                                        runtime_seconds=record.get("runtime_seconds"),
                                        checkpoint_artifact=record.get("produced_artifacts", {}).get("model_artifact"),
                                        reason_code=record.get("reason_code"),
                                        failure_category=record.get("failure_category"),
                                    )],
                                )
                            )
                    round_result = RoundResult(
                        sure_run_id=self._ensure_run_id(),
                        search_mode="ordinary",
                        round_index=research_round + 1,
                        baseline_digest=digest(base_solution),
                        idea_batch_digest=self._xlab_last_batch_digest or digest(research_plan),
                        result_digest=digest({direction: list(records.values()) for direction, records in round_results.items()}),
                        candidates=candidates,
                        ranking=sorted(
                            (candidate.idea_id for candidate in candidates if candidate.final_status == "success"),
                            key=lambda idea_id: next(c.rungs[0].score for c in candidates if c.idea_id == idea_id),
                            reverse=not self.task_card.is_lower_better,
                        ),
                        metric={"name": self.task_card.primary_metric, "direction": "lower" if self.task_card.is_lower_better else "higher"},
                        baseline_score=self.baseline_score,
                        rung_names=sorted({rung.name for candidate in candidates for rung in candidate.rungs}),
                        current_best={
                            "score": self.best_score,
                            "solution_digest": digest(self.best_solution or ""),
                        },
                        parent_artifacts=list(self._xlab_last_batch_artifacts) + list(self._xlab_summary_artifacts),
                    )
                    summary = self._summarize_xlab_result(round_result)
                    history_record = {
                        "axis": None,
                        "research_round": research_round + 1,
                        "batch_digest": round_result.idea_batch_digest,
                        "result_digest": round_result.result_digest,
                        "summary_digest": summary.summary_digest,
                        "summary": asdict(summary),
                        "baseline_score": round_result.baseline_score,
                        "metric": round_result.metric,
                        "execution_contract": dict(self.sure_config.get("execution_contract") or {}),
                        "evaluation_scope": {"search": self._role_paths(), "manifest": self._execution_env().get("SURE_EVAL_MANIFEST")},
                        "ranking": list(round_result.ranking),
                        "candidates": [
                            {
                                "idea_id": candidate.idea_id,
                                "status": candidate.final_status,
                                "idea": candidate.idea,
                                "code_digest": candidate.code_digest,
                                "idea_artifact_id": candidate.idea_artifact_id,
                                "candidate_type": candidate.candidate_type,
                                "reason_code": candidate.reason_code,
                                "improved": candidate.improved,
                                "failure_category": candidate.failure_category,
                                "rungs": [rung.__dict__ for rung in candidate.rungs],
                            }
                            for candidate in candidates
                        ],
                    }
                    self._xlab_history.append(history_record)
                    self._record_xlab_event("round_result_published", {
                        "axis": None,
                        "research_round": research_round + 1,
                        "result_digest": round_result.result_digest,
                        "batch_digest": round_result.idea_batch_digest,
                        "candidate_count": len(candidates),
                    })
                    self._record_xlab_event("summary_published", history_record)
                    if summary.summary_digest not in self._xlab_summary_artifacts:
                        self._xlab_summary_artifacts.append(summary.summary_digest)

                if feedback_enabled:
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

                records = [record for group in round_results.values() for record in group.values()]
                if any(r.get("failure_category") == "system_failure" for r in records) or (not records and "ablation" not in self.sure_config) or ("ablation" not in self.sure_config and not any(r.get("success") for r in records)):
                    raise RuntimeError("Round incomplete: restore infrastructure/valid experiments before counting stagnation")
                self._write_run_json(f"ablation_round_{research_round + 1}.json", {"round": research_round + 1, "policy": policy(self.sure_config), "candidates": records})
                stale_rounds = 0 if self.compare_score(round_start_score, self.best_score) else stale_rounds + 1
                self._write_run_json("search_progress.json", {"rounds": research_round + 1,
                                    "stale_rounds": stale_rounds, "best_score": self.best_score})
                checkpoint_controller(research_round + 1, stale_rounds)
                if research_round + 1 >= minimum_rounds and stale_rounds >= patience:
                    break

            search_best_score = self.best_score
            final_evaluation = self._final_evaluation(baseline_record, scored_candidates)
            self._commit_best_state()
            best_dir = Path(self.session.config.workspace_path) / "best_solution"
            best_dir.mkdir(parents=True, exist_ok=True)
            (best_dir / "model_artifact.json").write_text(json.dumps(self.best_model_artifact, indent=2) + "\n")
            save_code_to_file(str(best_dir), "best_solution.py", self.best_solution or "")
            return {
                "status": "incomplete" if self.sure_config.get("require_training_candidate") and not successful_training_candidates else "completed",
                "steps": 0,
                "task_id": self.task_card.task_id,
                "metric": self.task_card.primary_metric,
                "baseline_score": self.baseline_score,
                "successful_training_candidates": successful_training_candidates,
                "search_best_score": search_best_score,
                "best_score_split": "selection" if final_evaluation else "search",
                "best_model_artifact": self.best_model_artifact,
                "final_evaluation": final_evaluation,
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
        except TrainingBudgetPaused as e:
            self.logger.warning("Training paused: %s", e)
            return {"status": "paused_budget", "error": str(e),
                    "best_score": self.best_score}
        except XlabIdeaClientError as e:
            self.logger.error("XLab operation is incomplete: %s", e, exc_info=True)
            return {"status": "incomplete", "steps": 0, "error": str(e),
                    "best_score": self.best_score,
                    "best_model_artifact": getattr(self, "best_model_artifact", {})}
        except Exception as e:
            self.logger.error("SURE Master task execution failed: %s", e, exc_info=True)
            return {"status": "failed", "steps": 0, "error": str(e)}
        finally:
            watchdog.stop()
            self._close_xlab_provider()
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
                                check_run_pause(Path(exp_workspace))
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
                    except TrainingBudgetPaused:
                        raise
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
