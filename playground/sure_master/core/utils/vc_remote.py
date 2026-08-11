from __future__ import annotations

import json
import logging
import os
import py_compile
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .candidate_changes import clear_candidate_outputs
from .candidate_type import ARCH, FINE_TUNE, INFERENCE, TRAINING, normalize_candidate_type


PROJECT_ROOT = Path(__file__).resolve().parents[4]

REMOTE_ENV_OVERRIDES = {
    "SURE_REMOTE_IMAGE": "image",
    "SURE_REMOTE_PARTITION": "partition",
    "SURE_REMOTE_PARTITIONS": "partitions",
    "SURE_REMOTE_PARTITION_POLICY": "partition_policy",
    "SURE_REMOTE_PARTITION_FALLBACK": "partition_fallback",
    "SURE_REMOTE_QOS": "qos",
    "SURE_REMOTE_NUM_TASK": "num_task",
    "SURE_REMOTE_GPU_PER_TASK": "gpu_per_task",
    "SURE_REMOTE_CPU_PER_TASK": "cpu_per_task",
    "SURE_REMOTE_MEM_PER_TASK": "mem_per_task",
    "SURE_REMOTE_MAX_PARALLEL": "max_parallel",
    "SURE_REMOTE_SUBMIT_TIMEOUT": "submit_timeout",
    "SURE_REMOTE_CANDIDATE_TYPES": "candidate_types",
    "SURE_REMOTE_DRAFT_ENABLED": "draft_enabled",
}

DEFAULT_PARTITION_POLICY = "most_free_gpu"
DEFAULT_PARTITION_FALLBACK = "queue_first"
VALID_REMOTE_CANDIDATE_TYPES = {INFERENCE, FINE_TUNE, ARCH, TRAINING}


def _config_to_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return config
    if hasattr(config, "model_dump"):
        return config.model_dump()
    if hasattr(config, "dict"):
        return config.dict()
    return {}


def sure_config_from(config: Any) -> dict[str, Any]:
    root = _config_to_dict(config)
    value = root.get("sure") or {}
    return value if isinstance(value, dict) else {}


def remote_training_config_from(config: Any) -> dict[str, Any]:
    sure_config = sure_config_from(config)
    remote = sure_config.get("remote_training") or {}
    if not isinstance(remote, dict):
        remote = {}
    resolved = dict(remote)
    for env_name, config_key in REMOTE_ENV_OVERRIDES.items():
        value = os.environ.get(env_name)
        if value is not None and value.strip():
            resolved[config_key] = value.strip()
    return resolved


def remote_resource_config_from(
    config: Any,
    *,
    candidate_type: str | None = None,
    stage: str | None = None,
    workload_profile: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve workload resources using legacy, default, and specific layers.

    ``workload_profile`` is an explicit dispatch decision. This avoids treating
    every draft as train-from-scratch: draft inference remains ``inference``,
    while a training draft can select ``draft_training``. Environment resource
    overrides remain authoritative for backward compatibility.
    """
    remote = sure_config_from(config).get("remote_training") or {}
    if not isinstance(remote, dict):
        remote = {}
    resolved = dict(remote)
    profiles = remote.get("resource_profiles") or {}
    if not isinstance(profiles, dict):
        profiles = {}

    normalized = normalize_candidate_type(candidate_type or TRAINING)
    selected_profile = str(workload_profile or normalized).strip().lower().replace("-", "_")
    applied_profiles: list[str] = []

    def apply_profile(name: str) -> None:
        value = profiles.get(name)
        if isinstance(value, dict):
            resolved.update(value)
            applied_profiles.append(name)

    # Documented precedence: legacy flat fields < default < training inheritance
    # < the explicitly selected workload profile < environment overrides.
    apply_profile("default")
    if selected_profile in {TRAINING, FINE_TUNE, ARCH, "draft_training"}:
        apply_profile(TRAINING)
    if selected_profile != TRAINING:
        apply_profile(selected_profile)

    profile_name = applied_profiles[-1] if applied_profiles else "legacy"
    for env_name, config_key in REMOTE_ENV_OVERRIDES.items():
        value = os.environ.get(env_name)
        if value is not None and value.strip():
            resolved[config_key] = value.strip()

    diagnostics = {
        "candidate_type": normalized,
        "stage": str(stage or ""),
        "workload_profile": selected_profile,
        "profile_name": profile_name,
        "profile_configured": profile_name != "legacy",
        "requested_resources": {
            key: resolved.get(key)
            for key in ("num_task", "gpu_per_task", "cpu_per_task", "mem_per_task")
        },
    }
    return resolved, diagnostics


def mixed_execution_enabled(config: Any, task_id: str | None = None) -> bool:
    sure_config = sure_config_from(config)
    if str(sure_config.get("execution_mode", "")).strip().lower() == "mixed_local_vc":
        return True
    remote = remote_training_config_from(config)
    return bool(remote.get("enabled", False))


def remote_training_max_parallel(config: Any, default: int = 1) -> int:
    remote = remote_training_config_from(config)
    try:
        return max(1, int(remote.get("max_parallel", default)))
    except (TypeError, ValueError):
        return default


def remote_candidate_types_from(config: Any) -> set[str]:
    """Return candidate types that should execute in VC child jobs.

    The legacy default is training-only so existing ASR/icefall mixed configs keep
    their current behavior. F5-TTS can opt in with:

      sure.remote_training.candidate_types: ["inference", "training"]

    The legacy "training" label expands to both fine_tune and arch so older
    mixed configs keep sending all training-like candidates to VC jobs.
    """
    remote = remote_training_config_from(config)
    if "candidate_types" in remote:
        raw_value = remote.get("candidate_types")
    else:
        raw_value = sure_config_from(config).get("remote_candidate_types")

    if raw_value is None:
        return {FINE_TUNE, ARCH}

    values = [value.lower() for value in _csv_values(raw_value)]
    if any(value in {"all", "*"} for value in values):
        return {INFERENCE, FINE_TUNE, ARCH}
    expanded: set[str] = set()
    for value in values:
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized == TRAINING:
            expanded.update({FINE_TUNE, ARCH})
        elif normalized in {"architecture", "structure"}:
            expanded.add(ARCH)
        elif normalized in {"finetune", "fine_tune"}:
            expanded.add(FINE_TUNE)
        elif normalized in VALID_REMOTE_CANDIDATE_TYPES:
            expanded.add(normalize_candidate_type(normalized))
    return expanded


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def candidate_runs_remotely(config: Any, candidate_type: str, task_id: str | None = None) -> bool:
    """Whether a candidate should run through the VC child-job executor."""
    remote = remote_training_config_from(config)
    if not bool(remote.get("enabled", False)):
        return False
    normalized = normalize_candidate_type(candidate_type)
    return normalized in remote_candidate_types_from(config)


def draft_runs_remotely(config: Any) -> bool:
    """Whether draft-stage training should be routed to the VC child-job executor."""
    remote = remote_training_config_from(config)
    if not bool(remote.get("enabled", False)):
        return False
    return _truthy(remote.get("draft_enabled")) or _truthy(remote.get("draft_training_enabled"))


def complete_remote_coverage_from(config: Any) -> tuple[bool, dict[str, Any]]:
    """Prove that coordinator execution cannot require a local GPU."""
    sure_config = sure_config_from(config)
    remote = remote_training_config_from(config)
    configured = remote_candidate_types_from(config)
    required = {INFERENCE, FINE_TUNE, ARCH}
    enabled = bool(remote.get("enabled", False))
    draft_enabled = draft_runs_remotely(config)
    staged_axes = sure_config.get("staged_axes") or {}
    if not isinstance(staged_axes, dict):
        staged_axes = {}
    start_phase = str(staged_axes.get("start_phase", "draft")).strip().lower()
    draft_required = start_phase != "arch"
    missing = sorted(required - configured)
    complete = enabled and (draft_enabled or not draft_required) and not missing
    return complete, {
        "remote_enabled": enabled,
        "draft_enabled": draft_enabled,
        "draft_required": draft_required,
        "remote_candidate_types": sorted(configured),
        "missing_candidate_types": missing,
    }


def candidate_limits_from(config: Any) -> dict[str, int]:
    sure_config = sure_config_from(config)
    limits = sure_config.get("candidate_types") or {}
    if not isinstance(limits, dict):
        limits = {}

    def _limit(name: str, default: int) -> int:
        try:
            return max(0, int(limits.get(name, default)))
        except (TypeError, ValueError):
            return default

    training_limit = _limit("max_training_per_round", 0)
    fine_tune_limit = _limit("max_fine_tune_per_round", 0)
    arch_limit = _limit("max_arch_per_round", 0)
    if training_limit > 0 and fine_tune_limit <= 0 and arch_limit <= 0:
        fine_tune_limit = training_limit // 2
        arch_limit = training_limit - fine_tune_limit

    return {
        INFERENCE: _limit("max_inference_per_round", 0),
        FINE_TUNE: fine_tune_limit,
        ARCH: arch_limit,
    }


def _resolve_config_path(config_path: str | Path | None) -> Path:
    if config_path is None:
        raise ValueError("config_path is required for VC remote training execution")
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _safe_job_name(value: str, max_len: int = 63) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    cleaned = cleaned or "sure-f5tts-train"
    return cleaned[:max_len]


def _csv_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw_values = []
        for item in value:
            raw_values.extend(str(item).split(","))
    else:
        raw_values = str(value).split(",")
    return [item.strip() for item in raw_values if item.strip()]


def _positive_int(value: Any, default: int = 1) -> int:
    try:
        return max(1, int(str(value).strip()))
    except (TypeError, ValueError):
        return default


def parse_vc_info_partitions(output: str) -> dict[str, dict[str, int]]:
    """Parse `vc info` partition resource rows."""
    snapshot: dict[str, dict[str, int]] = {}
    for line in output.splitlines():
        if "|" not in line:
            continue
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) < 2:
            continue
        partition = cells[0]
        if not partition or partition.lower() == "partition" or set(partition) <= {"-"}:
            continue
        gpu_match = re.search(r"(\d+)\s*/\s*(\d+)", cells[1])
        if not gpu_match:
            continue
        allocated = int(gpu_match.group(1))
        total = int(gpu_match.group(2))
        snapshot[partition] = {
            "allocated_gpu": allocated,
            "total_gpu": total,
            "free_gpu": max(0, total - allocated),
        }
    return snapshot


class VcRemoteTrainingExecutor:
    """Submit one candidate as a synchronous VC child job."""

    def __init__(
        self,
        config: Any,
        *,
        config_path: str | Path | None,
        logger: logging.Logger | None = None,
        candidate_type: str | None = None,
        stage: str | None = None,
        workload_profile: str | None = None,
    ) -> None:
        self.config = config
        self.config_path = _resolve_config_path(config_path)
        self.remote_config, self.resource_profile = remote_resource_config_from(
            config,
            candidate_type=candidate_type,
            stage=stage,
            workload_profile=workload_profile,
        )
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.last_partition_selection: dict[str, Any] = {}
        self.last_job_name = ""
        self.last_command: list[str] = []

    @property
    def enabled(self) -> bool:
        return bool(self.remote_config.get("enabled", False))

    def _submit_timeout(self, execution_timeout: int) -> int | None:
        default = execution_timeout + 1800 if execution_timeout > 0 else 88200
        value = int(self.remote_config.get("submit_timeout", default))
        return None if value == 0 else value

    def run(
        self,
        *,
        workspace_path: str | Path,
        exp_name: str,
        execution_timeout: int,
    ) -> dict[str, Any]:
        workspace = Path(workspace_path).resolve()
        metric_dir = workspace / "metric"
        metric_dir.mkdir(parents=True, exist_ok=True)
        result_path = metric_dir / "remote_training_result.json"
        submit_log_path = metric_dir / "remote_training_vc_submit.log"
        command_path = metric_dir / "remote_training_vc_command.json"

        clear_candidate_outputs(workspace)
        if result_path.exists():
            result_path.unlink()

        try:
            self._preflight(workspace=workspace, result_path=result_path)
            command = self._build_vc_command(
                workspace=workspace,
                result_path=result_path,
                exp_name=exp_name,
                execution_timeout=execution_timeout,
            )
        except Exception as exc:
            command_path.write_text(
                json.dumps(
                    {
                        "command": None,
                        "partition_selection": self.last_partition_selection,
                        "resource_profile": self.resource_profile,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            return self._failure_result(
                str(exc),
                reason_code="remote_command_build_failed",
                submit_log_path=submit_log_path,
                result_path=result_path,
                elapsed_seconds=0.0,
            )
        self.last_command = command
        command_path.write_text(
            json.dumps(
                {
                    "command": command,
                    "partition_selection": self.last_partition_selection,
                    "resource_profile": self.resource_profile,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        submit_timeout = self._submit_timeout(execution_timeout)
        started_at = time.time()
        self.logger.info("Submitting VC remote candidate %s", exp_name)
        try:
            completed = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=submit_timeout,
                check=False,
            )
            submit_output = completed.stdout or ""
            submit_log_path.write_text(submit_output, encoding="utf-8")
            vc_exit_code = completed.returncode
        except subprocess.TimeoutExpired as exc:
            submit_output = exc.stdout or ""
            if isinstance(submit_output, bytes):
                submit_output = submit_output.decode("utf-8", errors="replace")
            message = f"VC submit timed out after {submit_timeout}s"
            submit_log_path.write_text(submit_output + "\n" + message + "\n", encoding="utf-8")
            recovery_started_at = time.time()
            recovered_payload, recovery_error = self._wait_for_remote_result(
                result_path=result_path,
                require_enabled=True,
            )
            if recovered_payload is not None:
                self.logger.info(
                    "Recovered VC remote result for %s after submit timeout", exp_name
                )
                return self._augment_remote_payload(
                    recovered_payload,
                    submit_log_path=submit_log_path,
                    command_path=command_path,
                    result_path=result_path,
                    elapsed_seconds=time.time() - started_at,
                    vc_exit_code=None,
                    extra_execution_info={
                        "submit_timed_out": True,
                        "recovered_after_submit_timeout": True,
                        "submit_timeout_seconds": submit_timeout,
                        "result_recovery_elapsed_seconds": round(
                            time.time() - recovery_started_at, 3
                        ),
                    },
                )
            if recovery_error:
                self.logger.warning(
                    "VC result recovery after submit timeout for %s failed: %s",
                    exp_name,
                    recovery_error,
                )
            return self._failure_result(
                message,
                reason_code="remote_submit_timeout",
                submit_log_path=submit_log_path,
                result_path=result_path,
                elapsed_seconds=time.time() - started_at,
                terminal_output=submit_output,
                vc_exit_code=None,
            )
        except FileNotFoundError:
            return self._failure_result(
                "vc command not found on coordinator host",
                reason_code="vc_command_not_found",
                submit_log_path=submit_log_path,
                result_path=result_path,
                elapsed_seconds=time.time() - started_at,
            )

        if result_path.is_file():
            payload, error = self._read_remote_result(result_path)
            if payload is None:
                return self._failure_result(
                    f"Remote result JSON is invalid: {error}",
                    reason_code="remote_result_invalid_json",
                    submit_log_path=submit_log_path,
                    result_path=result_path,
                    elapsed_seconds=time.time() - started_at,
                    terminal_output=submit_output,
                    vc_exit_code=vc_exit_code,
                )
            return self._augment_remote_payload(
                payload,
                submit_log_path=submit_log_path,
                command_path=command_path,
                result_path=result_path,
                elapsed_seconds=time.time() - started_at,
                vc_exit_code=vc_exit_code,
            )

        recovered_payload, recovery_error = self._wait_for_remote_result(
            result_path=result_path,
            require_enabled=True,
            require_missing_enabled=True,
        )
        if recovered_payload is not None:
            self.logger.info("Recovered delayed VC remote result for %s", exp_name)
            return self._augment_remote_payload(
                recovered_payload,
                submit_log_path=submit_log_path,
                command_path=command_path,
                result_path=result_path,
                elapsed_seconds=time.time() - started_at,
                vc_exit_code=vc_exit_code,
                extra_execution_info={
                    "recovered_after_missing_result": True,
                    "result_recovery_error": recovery_error,
                },
            )

        if vc_exit_code != 0:
            return self._failure_result(
                f"VC submit exited non-zero ({vc_exit_code}) and no remote result was written",
                reason_code="remote_result_missing",
                submit_log_path=submit_log_path,
                result_path=result_path,
                elapsed_seconds=time.time() - started_at,
                terminal_output=submit_output,
                vc_exit_code=vc_exit_code,
            )
        return self._failure_result(
            "VC child job finished but did not write remote result",
            reason_code="remote_result_missing",
            submit_log_path=submit_log_path,
            result_path=result_path,
            elapsed_seconds=time.time() - started_at,
            terminal_output=submit_output,
            vc_exit_code=vc_exit_code,
        )

    def _preflight(self, *, workspace: Path, result_path: Path) -> None:
        if not self._preflight_enabled():
            return
        errors: list[str] = []
        workdir = self._source_workdir()
        runner = str(
            self.remote_config.get("runner")
            or "playground/sure_master/tools/run_vc_sure_candidate.py"
        )
        runner_path = Path(runner)
        if not runner_path.is_absolute():
            runner_path = workdir / runner_path
        required_files = [
            workdir / "playground/sure_master/core/exp/run_exp.py",
            workdir / "playground/sure_master/core/playground.py",
            workdir / "playground/sure_master/core/utils/vc_remote.py",
            runner_path,
            self.config_path if self.config_path.is_absolute() else PROJECT_ROOT / self.config_path,
        ]
        for path in required_files:
            if not path.is_file():
                errors.append(f"required source file is missing: {path}")
            elif path.stat().st_size <= 0:
                errors.append(f"required source file is empty: {path}")
        if runner_path.is_file() and runner_path.stat().st_size > 0:
            try:
                py_compile.compile(str(runner_path), doraise=True)
            except py_compile.PyCompileError as exc:
                errors.append(f"runner does not compile: {runner_path}: {exc.msg}")
        if shutil.which("vc") is None:
            errors.append("vc command is not available on coordinator host")
        if not workspace.exists():
            errors.append(f"workspace does not exist before VC submit: {workspace}")
        if not result_path.parent.exists():
            errors.append(f"remote result parent does not exist before VC submit: {result_path.parent}")
        missing_env = self._missing_required_env_vars()
        if missing_env:
            errors.append("required runtime env var(s) are missing: " + ", ".join(missing_env))
        if errors:
            raise RuntimeError("SURE remote preflight failed: " + "; ".join(errors))

    def _preflight_enabled(self) -> bool:
        sure_config = sure_config_from(self.config)
        preflight = sure_config.get("preflight") or {}
        if isinstance(preflight, dict) and "enabled" in preflight:
            return _truthy(preflight.get("enabled"))
        if "preflight" in self.remote_config:
            return _truthy(self.remote_config.get("preflight"))
        return False

    def _missing_required_env_vars(self) -> list[str]:
        sure_config = sure_config_from(self.config)
        preflight = sure_config.get("preflight") or {}
        raw_values = preflight.get("required_env_vars") if isinstance(preflight, dict) else None
        names = _csv_values(raw_values)
        if not names:
            return []
        env_values = dict(os.environ)
        env_file = Path(str(self.remote_config.get("env_file") or PROJECT_ROOT / ".env")).expanduser()
        if env_file.is_file():
            env_values.update(_read_env_file(env_file))
        return [name for name in names if not str(env_values.get(name, "")).strip()]

    def _build_vc_command(
        self,
        *,
        workspace: Path,
        result_path: Path,
        exp_name: str,
        execution_timeout: int,
    ) -> list[str]:
        image = str(self.remote_config.get("image") or "").strip()
        if not image:
            raise ValueError("sure.remote_training.image is required")

        workdir = str(self._source_workdir())
        remote_workdir = Path(workdir)
        artifact_workdir = self._artifact_workdir()
        remote_config_path = self._source_path_for_child(self.config_path, remote_workdir)
        remote_workspace = self._source_path_for_child(workspace, artifact_workdir)
        remote_result_path = self._source_path_for_child(result_path, artifact_workdir)
        python_bin = str(self.remote_config.get("python") or "/opt/conda/envs/evomaster/bin/python")
        runner = str(
            self.remote_config.get("runner")
            or "playground/sure_master/tools/run_vc_sure_candidate.py"
        )
        job_prefix = str(self.remote_config.get("job_prefix") or "sure-f5tts-train")
        workload = str(self.resource_profile.get("candidate_type") or "training").replace("_", "-")
        job_name = _safe_job_name(f"{job_prefix}-{workload}-{exp_name}-{int(time.time())}")
        self.last_job_name = job_name
        env_file = Path(str(self.remote_config.get("env_file") or PROJECT_ROOT / ".env"))
        profile_name = str(self.resource_profile.get("profile_name") or "legacy")
        requested_resources = json.dumps(
            self.resource_profile.get("requested_resources") or {},
            ensure_ascii=True,
            separators=(",", ":"),
        )
        child_script = "\n".join(
            [
                "set -eo pipefail",
                f"cd {shlex.quote(workdir)}",
                f"if [ -f {shlex.quote(str(env_file))} ]; then set -a; source {shlex.quote(str(env_file))}; set +a; "
                "elif [ -f .env ]; then set -a; source .env; set +a; fi",
                f"export SURE_VC_RESOURCE_PROFILE={shlex.quote(profile_name)}",
                f"export SURE_VC_REQUESTED_RESOURCES={shlex.quote(requested_resources)}",
                (
                    f"{shlex.quote(python_bin)} -u {shlex.quote(runner)} "
                    f"--config {shlex.quote(str(remote_config_path))} "
                    f"--workspace {shlex.quote(str(remote_workspace))} "
                    f"--result {shlex.quote(str(remote_result_path))} "
                    f"--timeout {int(execution_timeout)}"
                ),
            ]
        )

        selected_partition = self._select_partition()

        command = ["vc", "submit", "--image", image]
        for option, config_key in (
            ("--partition", "partition"),
            ("--qos", "qos"),
            ("--num-task", "num_task"),
            ("--gpu-per-task", "gpu_per_task"),
            ("--cpu-per-task", "cpu_per_task"),
            ("--mem-per-task", "mem_per_task"),
        ):
            value = selected_partition if config_key == "partition" else self.remote_config.get(config_key)
            if value is not None and str(value).strip():
                command.extend([option, str(value)])

        if bool(self.remote_config.get("nopassenv", True)):
            command.append("--nopassenv")
        for volume in self.remote_config.get("volumes", []) or []:
            command.extend(["--volume", str(volume)])
        command.extend(["--dir", workdir])
        if bool(self.remote_config.get("sync", True)):
            command.append("--sync")
        if bool(self.remote_config.get("debug", True)):
            command.append("--debug")
        command.extend(["--job", job_name])
        command.extend(["--cmd", "bash -lc " + shlex.quote(child_script)])
        return command

    def _source_workdir(self) -> Path:
        snapshot = sure_config_from(self.config).get("source_snapshot") or {}
        if isinstance(snapshot, dict) and _truthy(snapshot.get("enabled")) and snapshot.get("path"):
            return Path(str(snapshot["path"])).expanduser()
        return Path(str(self.remote_config.get("workdir") or PROJECT_ROOT)).expanduser()

    def _artifact_workdir(self) -> Path:
        return Path(
            str(
                self.remote_config.get("artifact_workdir")
                or self.remote_config.get("workdir")
                or PROJECT_ROOT
            )
        ).expanduser()

    def _select_partition(self) -> str:
        explicit_partition = os.environ.get("SURE_REMOTE_PARTITION")
        if explicit_partition is not None and explicit_partition.strip():
            selected = explicit_partition.strip()
            self.last_partition_selection = {
                "selected_partition": selected,
                "partition_source": "SURE_REMOTE_PARTITION",
            }
            return selected

        configured_partition = str(self.remote_config.get("partition") or "").strip()
        candidates = _csv_values(self.remote_config.get("partitions"))
        if not candidates:
            self.last_partition_selection = {
                "selected_partition": configured_partition,
                "partition_source": "remote_training.partition",
            }
            return configured_partition

        policy = str(
            self.remote_config.get("partition_policy") or DEFAULT_PARTITION_POLICY
        ).strip().lower()
        fallback = str(
            self.remote_config.get("partition_fallback") or DEFAULT_PARTITION_FALLBACK
        ).strip().lower()
        gpu_per_task = _positive_int(self.remote_config.get("gpu_per_task"), default=1)

        selection: dict[str, Any] = {
            "selected_partition": None,
            "partition_source": "SURE_REMOTE_PARTITIONS",
            "partition_policy": policy,
            "partition_fallback": fallback,
            "partition_candidates": candidates,
            "required_gpu_per_task": gpu_per_task,
            "partition_snapshot": {},
            "partition_probe_error": None,
        }

        selected = ""
        if policy == "most_free_gpu":
            snapshot, error = self._probe_partition_snapshot()
            selection["partition_snapshot"] = {
                name: snapshot[name] for name in candidates if name in snapshot
            }
            selection["partition_probe_error"] = error
            eligible = [
                (index, name, snapshot[name]["free_gpu"])
                for index, name in enumerate(candidates)
                if name in snapshot and snapshot[name]["free_gpu"] >= gpu_per_task
            ]
            if eligible:
                _index, selected, _free_gpu = max(
                    eligible,
                    key=lambda item: (item[2], -item[0]),
                )
        else:
            selection["partition_probe_error"] = f"Unsupported partition policy: {policy}"

        if not selected:
            if fallback == "fail":
                selection["selected_partition"] = None
                self.last_partition_selection = selection
                raise RuntimeError(
                    "No remote partition has enough free GPUs "
                    f"for gpu_per_task={gpu_per_task}; candidates={candidates}"
                )
            selected = candidates[0]
            selection["fallback_used"] = True
        else:
            selection["fallback_used"] = False

        selection["selected_partition"] = selected
        self.last_partition_selection = selection
        return selected

    def _probe_partition_snapshot(self) -> tuple[dict[str, dict[str, int]], str | None]:
        timeout = _positive_int(
            self.remote_config.get("partition_probe_timeout")
            or os.environ.get("SURE_REMOTE_PARTITION_PROBE_TIMEOUT"),
            default=20,
        )
        try:
            completed = subprocess.run(
                ["vc", "info"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError:
            return {}, "vc command not found"
        except subprocess.TimeoutExpired:
            return {}, f"vc info timed out after {timeout}s"
        except OSError as exc:
            return {}, f"vc info failed: {exc}"

        output = completed.stdout or ""
        if completed.returncode != 0:
            tail = output[-1000:] if output else ""
            return {}, f"vc info exited non-zero ({completed.returncode}): {tail}"

        snapshot = parse_vc_info_partitions(output)
        if not snapshot:
            return {}, "vc info output did not contain parseable partition rows"
        return snapshot, None

    @staticmethod
    def _source_path_for_child(path: Path, remote_workdir: Path) -> Path:
        """Map repo-local coordinator paths to the child job's mounted repo path."""
        try:
            rel = path.resolve().relative_to(PROJECT_ROOT.resolve())
        except ValueError:
            return path
        return remote_workdir / rel

    def _read_remote_result(self, result_path: Path) -> tuple[dict[str, Any] | None, str | None]:
        if not result_path.is_file():
            return None, "remote result file is missing"
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return None, str(exc)
        if not isinstance(payload, dict):
            return None, f"remote result JSON must be an object, got {type(payload).__name__}"
        return payload, None

    def _augment_remote_payload(
        self,
        payload: dict[str, Any],
        *,
        submit_log_path: Path,
        command_path: Path,
        result_path: Path,
        elapsed_seconds: float,
        vc_exit_code: int | None,
        extra_execution_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = dict(payload)
        execution_info = dict(payload.get("execution_info") or {})
        execution_info.update(
            {
                "vc_exit_code": vc_exit_code,
                "vc_submit_log": str(submit_log_path),
                "vc_command_json": str(command_path),
                "job_name": self.last_job_name,
                "selected_partition": self.last_partition_selection.get("selected_partition"),
                "partition_selection": self.last_partition_selection,
                "resource_profile": self.resource_profile,
                "requested_resources": self.resource_profile.get("requested_resources", {}),
                "remote_result": str(result_path),
                "elapsed_seconds": round(elapsed_seconds, 3),
            }
        )
        if extra_execution_info:
            execution_info.update(extra_execution_info)
        payload["execution_info"] = execution_info
        details = payload.get("details")
        if isinstance(details, dict):
            details.setdefault("execution_info", execution_info)
        return payload

    def _result_recovery_config(self) -> dict[str, Any]:
        recovery = self.remote_config.get("result_recovery") or {}
        return recovery if isinstance(recovery, dict) else {}

    def _wait_for_remote_result(
        self,
        *,
        result_path: Path,
        require_enabled: bool = False,
        require_missing_enabled: bool = False,
    ) -> tuple[dict[str, Any] | None, str | None]:
        recovery = self._result_recovery_config()
        if require_enabled and not _truthy(recovery.get("enabled")):
            return None, None
        if require_missing_enabled and not _truthy(recovery.get("recover_missing_result", True)):
            return None, None
        grace_seconds = _nonnegative_float(recovery.get("grace_seconds"), default=0.0)
        if grace_seconds <= 0:
            return None, None
        poll_seconds = _positive_float(recovery.get("poll_seconds"), default=10.0)
        deadline = time.time() + grace_seconds
        last_error: str | None = None
        while time.time() <= deadline:
            payload, error = self._read_remote_result(result_path)
            if payload is not None:
                return payload, None
            last_error = error
            time.sleep(min(poll_seconds, max(0.0, deadline - time.time())))
        payload, error = self._read_remote_result(result_path)
        if payload is not None:
            return payload, None
        return None, error or last_error

    def _failure_result(
        self,
        error: str,
        *,
        reason_code: str,
        submit_log_path: Path,
        result_path: Path,
        elapsed_seconds: float,
        terminal_output: str = "",
        vc_exit_code: int | None = None,
    ) -> dict[str, Any]:
        submit_log_tail = terminal_output[-12000:] if terminal_output else _read_tail(submit_log_path)
        execution_info = {
            "vc_exit_code": vc_exit_code,
            "vc_submit_log": str(submit_log_path),
            "vc_submit_log_tail": submit_log_tail,
            "vc_command": self.last_command,
            "job_name": self.last_job_name,
            "selected_partition": self.last_partition_selection.get("selected_partition"),
            "partition_selection": self.last_partition_selection,
            "remote_result": str(result_path),
            "elapsed_seconds": round(elapsed_seconds, 3),
        }
        return {
            "success": False,
            "reason_code": reason_code,
            "score": None,
            "metric": "",
            "metric_feedback": f"SURE metric not available because remote candidate failed: {error}",
            "terminal_output": terminal_output or error,
            "error": error,
            "details": {"execution_info": execution_info, "error": error},
            "execution_info": execution_info,
        }


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _nonnegative_float(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _positive_float(value: Any, default: float = 1.0) -> float:
    try:
        return max(0.001, float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _read_tail(path: Path, limit: int = 12000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text if len(text) <= limit else text[-limit:]


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
        if not key:
            continue
        values[key] = value.strip().strip("'\"")
    return values
