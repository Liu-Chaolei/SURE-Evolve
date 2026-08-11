#!/usr/bin/env python3
"""Run one already-generated SURE candidate inside a VC child job.

The coordinator writes `run_sure.py` into an experiment workspace, then submits
this script to a VC child job for training-level ASR/TTS candidates. This child
process executes the candidate, runs the normal SURE metric, and writes result
and status JSON back to the shared workspace.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REMOTE_CONTEXT_NAME = "remote_candidate_context.json"
REMOTE_ENV_OVERRIDE_DENYLIST = {
    "SURE_ICEFALL_PYTHON",
    "SURE_LOCAL_ICEFALL_PYTHON",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="SURE Master config path")
    parser.add_argument("--workspace", required=True, help="Experiment workspace containing run_sure.py")
    parser.add_argument("--result", required=True, help="Result JSON path to write")
    parser.add_argument("--timeout", type=int, default=21600, help="Candidate execution timeout")
    return parser.parse_args()


def ensure_project_imports() -> None:
    """Import project modules lazily so setup failures can still write a result."""
    global ConfigManager
    global SureRunExp
    global candidate_type_from_code
    global clear_candidate_outputs
    global validate_run_sure_script
    global validate_sure_candidate_boundary
    global SureMetricRunner
    global format_metric_feedback
    global merge_base_model_profile
    global resolve_task_card
    global validate_base_model_profile

    from evomaster.config import ConfigManager
    from playground.sure_master.core.exp.run_exp import SureRunExp
    from playground.sure_master.core.utils.candidate_changes import clear_candidate_outputs
    from playground.sure_master.core.utils.candidate_type import candidate_type_from_code
    from playground.sure_master.core.utils.code import (
        validate_run_sure_script,
        validate_sure_candidate_boundary,
    )
    from playground.sure_master.core.utils.metric import SureMetricRunner, format_metric_feedback
    from playground.sure_master.core.utils.task_cards import (
        merge_base_model_profile,
        resolve_task_card,
        validate_base_model_profile,
    )


def load_config(config_path: Path):
    ensure_project_imports()
    return ConfigManager(config_dir=config_path.parent, config_file=config_path.name).load()


def load_sure_objects(config) -> tuple[Any, Any, SureMetricRunner, dict[str, str], dict[str, str | None]]:
    ensure_project_imports()
    config_dict = config.model_dump() if hasattr(config, "model_dump") else dict(config)
    sure_config = config_dict.get("sure") or {}
    task_cards_path = Path(
        sure_config.get(
            "task_cards_path",
            PROJECT_ROOT / "playground/sure_master/task_cards/sure_tasks.yaml",
        )
    )
    if not task_cards_path.is_absolute():
        task_cards_path = PROJECT_ROOT / task_cards_path
    task_card = resolve_task_card(task_cards_path, sure_config.get("task_id", "asr_en_wer"))

    overrides = sure_config.get("base_models", {}) or {}
    override = overrides.get(task_card.task_id) or overrides.get(task_card.canonical_task)
    base_model_profile = merge_base_model_profile(task_card.base_model, override)
    if base_model_profile is None and bool(sure_config.get("require_base_model", True)):
        raise ValueError(f"SURE task {task_card.task_id!r} requires a base_model profile")
    validate_base_model_profile(base_model_profile, task_card.task_id)

    sure_root = sure_config.get("root", "/hpc_stor03/sjtu_home/chaolei.liu/sure")
    pythonpath = sure_config.get("pythonpath") or str(Path(sure_root) / "src")
    metric_runner = SureMetricRunner(
        sure_root=sure_root,
        pythonpath=pythonpath,
        device=sure_config.get("device", "cuda"),
        cache_dir=sure_config.get("cache_dir"),
        validate_env=bool(sure_config.get("validate_env", False)),
        metric_gpu=sure_config.get("metric_gpu"),
    )

    execution_env = {
        str(key): str(value)
        for key, value in (sure_config.get("execution_env", {}) or {}).items()
    }
    for key in list(execution_env):
        if (
            key in os.environ
            and key not in REMOTE_ENV_OVERRIDE_DENYLIST
            and is_runtime_env_override_key(key)
        ):
            execution_env[key] = os.environ[key]
    remote_icefall_python = os.environ.get("SURE_REMOTE_ICEFALL_PYTHON")
    if remote_icefall_python:
        execution_env["SURE_ICEFALL_PYTHON"] = remote_icefall_python
    config_remote_icefall_python = execution_env.pop("SURE_REMOTE_ICEFALL_PYTHON", "")
    if config_remote_icefall_python:
        execution_env["SURE_ICEFALL_PYTHON"] = config_remote_icefall_python
    execution_env.pop("SURE_LOCAL_ICEFALL_PYTHON", None)
    for key in ("ASR_WORLD_SIZE", "SURE_BASELINE_WORLD_SIZE"):
        if key in os.environ and is_runtime_env_override_key(key):
            execution_env.setdefault(key, os.environ[key])
    role_paths = dict(task_card.artifact_contract)
    role_paths.update(sure_config.get("inputs", {}) or {})
    return task_card, base_model_profile, metric_runner, execution_env, role_paths


def is_runtime_env_override_key(key: str) -> bool:
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


def sanitize_remote_execution_env(env: dict[str, Any]) -> dict[str, str]:
    sanitized = {
        str(key): str(value)
        for key, value in env.items()
        if value is not None and str(key) not in REMOTE_ENV_OVERRIDE_DENYLIST
    }
    remote_icefall_python = sanitized.pop("SURE_REMOTE_ICEFALL_PYTHON", "")
    if remote_icefall_python:
        sanitized["SURE_ICEFALL_PYTHON"] = remote_icefall_python
    return sanitized


def scheduler_gpu_allocation(env: dict[str, str] | None = None) -> dict[str, Any]:
    """Return scheduler-provided exact visibility or an isolated task GPU count."""
    values = os.environ if env is None else env
    disabled = {"", "-1", "none", "nodevfiles", "void"}
    source = "CUDA_VISIBLE_DEVICES"
    visible = str(values.get(source, "")).strip()
    devices = (
        []
        if visible.lower() in disabled
        else [item.strip() for item in visible.split(",") if item.strip()]
    )
    gpu_count = len(devices)
    visibility_kind = "exact" if devices else "missing"

    if not devices:
        nvidia_visible = str(values.get("NVIDIA_VISIBLE_DEVICES", "")).strip()
        if nvidia_visible.lower() not in disabled and nvidia_visible.lower() != "all":
            source = "NVIDIA_VISIBLE_DEVICES"
            visible = nvidia_visible
            devices = [
                item.strip() for item in visible.split(",") if item.strip()
            ]
            gpu_count = len(devices)
            visibility_kind = "exact"

    if not devices:
        for count_key in ("GPU_PER_TASK", "GPU_NUM"):
            try:
                candidate_count = int(str(values.get(count_key, "0")).strip() or "0")
            except ValueError:
                candidate_count = 0
            if candidate_count > 0:
                source = count_key
                visible = ""
                gpu_count = candidate_count
                visibility_kind = "task_count"
                break

    requested: dict[str, Any] = {}
    try:
        raw_requested = values.get("SURE_VC_REQUESTED_RESOURCES", "")
        parsed = json.loads(raw_requested) if raw_requested else {}
        if isinstance(parsed, dict):
            requested = parsed
    except (TypeError, json.JSONDecodeError):
        pass
    return {
        "allocation_source": source,
        "visibility_kind": visibility_kind,
        "cuda_visible_devices": str(values.get("CUDA_VISIBLE_DEVICES", "")).strip(),
        "nvidia_visible_devices": str(values.get("NVIDIA_VISIBLE_DEVICES", "")).strip(),
        "effective_visible_devices": visible,
        "gpu_devices": devices,
        "gpu_count": gpu_count,
        "resource_profile": str(values.get("SURE_VC_RESOURCE_PROFILE", "legacy")),
        "requested_resources": requested,
    }


def apply_scheduler_allocation(
    execution_env: dict[str, str],
    *,
    asr_workload: bool = True,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Apply scheduler authority and ASR-only world-size variables."""
    allocation = scheduler_gpu_allocation(env)
    effective = str(allocation.get("effective_visible_devices") or "")
    if allocation.get("visibility_kind") == "task_count":
        execution_env.pop("CUDA_VISIBLE_DEVICES", None)
    elif effective:
        execution_env["CUDA_VISIBLE_DEVICES"] = effective

    gpu_count = int(allocation["gpu_count"])
    if asr_workload and gpu_count > 0:
        world_size = str(gpu_count)
        execution_env["ASR_WORLD_SIZE"] = world_size
        execution_env["SURE_BASELINE_WORLD_SIZE"] = world_size
    else:
        execution_env.pop("ASR_WORLD_SIZE", None)
        execution_env.pop("SURE_BASELINE_WORLD_SIZE", None)
    return allocation


def normalize_process_gpu_environment(
    allocation: dict[str, Any],
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Preserve exact masks, or remove a stale blank mask for count-only VC jobs."""
    normalized = dict(os.environ if env is None else env)
    effective = str(allocation.get("effective_visible_devices") or "")
    if allocation.get("visibility_kind") == "task_count":
        normalized.pop("CUDA_VISIBLE_DEVICES", None)
    elif effective:
        normalized["CUDA_VISIBLE_DEVICES"] = effective
    return normalized


def make_exp(
    *,
    config,
    workspace: Path,
    task_card,
    base_model_profile,
    metric_runner,
    execution_env: dict[str, str],
) -> SureRunExp:
    ensure_project_imports()
    workspace_parent = str(workspace.parent)
    fake_session = SimpleNamespace(
        config=SimpleNamespace(workspace_path=workspace_parent, config_dir=str(PROJECT_ROOT))
    )
    fake_agent = SimpleNamespace(session=fake_session)
    exp = SureRunExp(
        stage="remote_training",
        main_agent=fake_agent,
        debug_agent=fake_agent,
        config=config,
        exp_name=workspace.name,
        task_card=task_card,
        base_model_profile=base_model_profile,
        metric_runner=metric_runner,
        execution_env=execution_env,
        config_path=None,
    )
    exp.workspace_path = str(workspace)
    return exp


def ensure_base_model_paths(workspace: Path, base_model_profile) -> list[str]:
    """Ensure configured base_model paths exist in a remote candidate workspace."""
    if base_model_profile is None:
        return []
    required_paths = getattr(base_model_profile, "required_paths", {}) or {}
    source_paths = getattr(base_model_profile, "source_paths", {}) or {}
    missing: list[str] = []
    for name, rel_path in required_paths.items():
        target = workspace / str(rel_path)
        if target.exists():
            continue
        source_value = source_paths.get(name)
        if not source_value:
            missing.append(f"{name}:{rel_path} has no configured source path")
            continue
        source = Path(str(source_value)).expanduser()
        if not source.is_absolute():
            source = PROJECT_ROOT / source
        source = source.resolve(strict=False)
        if not source.exists():
            missing.append(f"{name}:{rel_path} source does not exist: {source}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.symlink_to(source, target_is_directory=source.is_dir())
        except OSError as exc:
            missing.append(f"{name}:{rel_path} could not be linked to {source}: {exc}")
            continue

    for name, rel_path in required_paths.items():
        target = workspace / str(rel_path)
        if not target.exists():
            missing.append(f"{name}:{rel_path} missing after preparation")
    _ensure_icefall_data_links(workspace)
    return missing


def _ensure_icefall_data_links(workspace: Path) -> None:
    recipe_data = workspace / "base_model" / "recipe" / "data"
    base_data = workspace / "base_model" / "data"
    workspace_data = workspace / "data"
    if base_data.exists():
        _ensure_relative_symlink(recipe_data, Path("../data"))
        _ensure_relative_symlink(workspace_data, Path("base_model/data"))


def _ensure_relative_symlink(path: Path, target: Path) -> None:
    try:
        if path.is_symlink() or path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target, target_is_directory=True)
    except OSError:
        return


def terminate_process_group(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        proc.terminate()
    try:
        proc.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        proc.kill()
    proc.wait()


def run_logged(
    command: str,
    *,
    cwd: Path,
    timeout: int | None,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    wait_timeout = None if timeout is not None and timeout <= 0 else timeout
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("+ " + command + "\n")
        log_file.flush()
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=env,
        )
        try:
            exit_code = proc.wait(timeout=wait_timeout)
        except subprocess.TimeoutExpired:
            terminate_process_group(proc)
            return {"exit_code": -1, "timed_out": True, "log_path": str(log_path)}
        if exit_code != 0:
            terminate_process_group(proc)
        return {"exit_code": exit_code, "timed_out": False, "log_path": str(log_path)}


def read_tail(path: Path, limit: int = 12000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"Could not read log {path}: {exc}"
    if len(text) <= limit:
        return text
    return text[-limit:]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def failure_payload(
    error: str,
    *,
    reason_code: str = "remote_candidate_failed",
    terminal_output: str = "",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "success": False,
        "reason_code": reason_code,
        "score": None,
        "metric": "",
        "metric_feedback": f"SURE metric not available because remote candidate failed: {error}",
        "terminal_output": terminal_output or error,
        "error": error,
        "details": details or {},
    }


def is_bpe_validation_failure(terminal_output: str) -> bool:
    lowered = str(terminal_output or "").lower()
    return any(
        marker in lowered
        for marker in (
            "zipformer train/decode bpe mismatch",
            "train_decode candidates must decode with the same bpe model",
            "missing zipformer bpe model for candidate",
            "zipformer training requires --bpe-model",
            "train_decode candidates must not override decode --bpe-model",
        )
    )


def _duration_contract_marker(terminal_output: str) -> tuple[str, dict[str, Any]] | None:
    marker_names = (
        "duration_probe_help_discovery_failed",
        "duration_probe_helper_cli_incompatible",
        "duration_probe_candidate_cli_incompatible",
    )
    for line in reversed(str(terminal_output or "").splitlines()):
        for marker in marker_names:
            marker_prefix = f"{marker}:"
            marker_index = line.find(marker_prefix)
            if marker_index < 0:
                continue
            raw_details = line[marker_index + len(marker_prefix) :].strip()
            try:
                parsed = json.loads(raw_details)
            except json.JSONDecodeError:
                parsed = {}
            return marker, parsed if isinstance(parsed, dict) else {}
    return None


def execution_failure_reason_code(
    terminal_output: str,
    execution_info: dict[str, Any] | None = None,
) -> str:
    if bool((execution_info or {}).get("timed_out")):
        return "candidate_execution_timeout"
    lowered = str(terminal_output or "").lower()
    if is_bpe_validation_failure(lowered):
        return "candidate_bpe_validation_failed"
    contract_marker = _duration_contract_marker(terminal_output)
    if contract_marker is not None:
        return contract_marker[0]
    is_duration_failure = "duration_autotune" in lowered or "resolve-max-duration" in lowered
    if is_duration_failure and (
        "cuda out of memory" in lowered or "torch.cuda.outofmemoryerror" in lowered
    ):
        return "duration_probe_oom"
    if is_duration_failure and (
        "modulenotfounderror" in lowered
        or "importerror" in lowered
        or "no module named" in lowered
        or "duration probe failed before proving a resource limit" in lowered
        or "duration probe exited non-zero without a recognized resource failure" in lowered
    ):
        return "duration_probe_startup_failed"
    if is_duration_failure:
        return "duration_autotune_failed"
    if "cudnn_status_execution_failed" in lowered or "cudnn error" in lowered:
        return "training_runtime_failed"
    if "cuda out of memory" in lowered or "torch.cuda.outofmemoryerror" in lowered:
        return "training_oom"
    return "execution_failed"


def execution_failure_details(
    execution_info: dict[str, Any],
    terminal_output: str,
    *,
    workspace: Path | None = None,
) -> dict[str, Any]:
    details: dict[str, Any] = {"execution_info": execution_info}
    lowered = str(terminal_output or "").lower()
    if bool(execution_info.get("timed_out")):
        details["fatal_error"] = "candidate_execution_timeout"
    elif is_bpe_validation_failure(lowered):
        details["fatal_error"] = "candidate_bpe_validation_failed"
    elif (contract_marker := _duration_contract_marker(terminal_output)) is not None:
        details.update(contract_marker[1])
        details["fatal_error"] = contract_marker[0]
    elif "duration_autotune" in lowered or "resolve-max-duration" in lowered:
        if "cuda out of memory" in lowered or "torch.cuda.outofmemoryerror" in lowered:
            details["fatal_error"] = "duration_probe_oom"
        elif (
            "modulenotfounderror" in lowered
            or "importerror" in lowered
            or "no module named" in lowered
            or "duration probe failed before proving a resource limit" in lowered
            or "duration probe exited non-zero without a recognized resource failure" in lowered
        ):
            details["fatal_error"] = "duration_probe_startup_failed"
        else:
            details["fatal_error"] = "duration_autotune_failed"
    elif "cudnn_status_execution_failed" in lowered or "cudnn error" in lowered:
        details["fatal_error"] = "cudnn_status_execution_failed"
    elif "cuda out of memory" in lowered or "torch.cuda.outofmemoryerror" in lowered:
        details["fatal_error"] = "cuda_out_of_memory"
    if workspace is not None:
        duration_logs = collect_duration_log_tails(workspace)
        if duration_logs:
            details["duration_log_tails"] = duration_logs
    return details


def collect_duration_log_tails(
    workspace: Path,
    *,
    limit: int = 4000,
    max_logs: int = 12,
) -> list[dict[str, str]]:
    log_paths: list[Path] = []
    for root in (
        workspace / "working",
        workspace / ".sure_runtime" / "duration_autotune",
    ):
        if not root.exists() or root.is_symlink():
            continue
        try:
            for path in sorted(root.rglob("*.log")):
                if path.is_file() and not path.is_symlink():
                    rel_text = str(path.relative_to(workspace)).lower()
                    if "duration" in rel_text or path.name == "probe.log":
                        log_paths.append(path)
        except OSError:
            continue

    entries: list[dict[str, str]] = []
    for path in log_paths[:max_logs]:
        try:
            source = str(path.relative_to(workspace))
        except ValueError:
            source = str(path)
        entries.append({"source": source, "tail": read_tail(path, limit)})
    return entries


def write_status_json(
    workspace: Path,
    *,
    success: bool,
    reason_code: str,
    stage: str = "remote_training",
    phase: str = "",
    rung: str = "",
    idea_id: str = "",
    candidate_type: str = "",
    score: float | None = None,
    remote_success: bool | None = None,
    details: dict[str, Any] | None = None,
    metric_feedback: str = "",
    terminal_output: str = "",
) -> None:
    payload = {
        "success": bool(success),
        "reason_code": reason_code,
        "stage": stage,
        "phase": phase,
        "rung": rung,
        "idea_id": idea_id,
        "candidate_type": candidate_type,
        "score": score,
        "score_valid": score is not None,
        "metric_accepted": bool(success),
        "remote_success": remote_success,
        "missing_artifacts": (details or {}).get("missing_artifacts", []),
        "artifact_guard_errors": (details or {}).get("artifact_guard_errors", []),
        "execution_info": (details or {}).get("execution_info", {}),
        "details": details or {},
        "metric_feedback": metric_feedback,
        "terminal_output_tail": terminal_output[-12000:] if terminal_output else "",
        "written_at": int(time.time()),
    }
    write_json(workspace / "artifacts" / "candidate_status.json", payload)


def load_remote_candidate_context(workspace: Path) -> dict[str, Any]:
    context_path = workspace / "metric" / REMOTE_CONTEXT_NAME
    if not context_path.is_file():
        return {}
    try:
        payload = json.loads(context_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    result_path = Path(args.result).expanduser().resolve()
    metric_dir = workspace / "metric"
    run_log = metric_dir / "remote_training_run_sure.log"

    remote_context = load_remote_candidate_context(workspace)

    def context_value(name: str, default: str = "") -> str:
        value = remote_context.get(name)
        return default if value is None else str(value)

    def write_failure_and_exit(
        *,
        reason_code: str,
        error: str,
        terminal_output: str = "",
        details: dict[str, Any] | None = None,
        exit_code: int = 1,
        candidate_type: str = "",
    ) -> None:
        payload = failure_payload(
            error,
            reason_code=reason_code,
            terminal_output=terminal_output,
            details=details,
        )
        write_json(result_path, payload)
        write_status_json(
            workspace,
            success=False,
            reason_code=reason_code,
            stage=context_value("stage", "remote_training"),
            phase=context_value("phase"),
            rung=context_value("rung"),
            idea_id=context_value("idea_id"),
            candidate_type=candidate_type or context_value("candidate_type_hint"),
            remote_success=False,
            details=details,
            metric_feedback=payload["metric_feedback"],
            terminal_output=terminal_output,
        )
        sys.exit(exit_code)

    try:
        ensure_project_imports()
        config = load_config(config_path)
        task_card, base_model_profile, metric_runner, execution_env, role_paths = load_sure_objects(config)
        context_env = remote_context.get("execution_env")
        if isinstance(context_env, dict):
            execution_env.update(sanitize_remote_execution_env(context_env))
        scheduler_allocation = apply_scheduler_allocation(
            execution_env,
            asr_workload=str(getattr(task_card, "canonical_task", "")).lower() == "asr",
        )
        if scheduler_allocation["gpu_count"] <= 0:
            write_failure_and_exit(
                reason_code="scheduler_gpu_allocation_missing",
                error="VC child job has no scheduler-visible GPU allocation",
                details={"scheduler_allocation": scheduler_allocation},
                exit_code=2,
                candidate_type=context_value("candidate_type_hint"),
            )
        process_env = normalize_process_gpu_environment(scheduler_allocation)
        os.environ.clear()
        os.environ.update(process_env)
        context_roles = remote_context.get("role_paths")
        if isinstance(context_roles, dict):
            role_paths.update(
                {
                    str(key): (None if value is None else str(value))
                    for key, value in context_roles.items()
                }
            )
        runtime_helper = PROJECT_ROOT / "playground/sure_master/core/utils/runtime_env.py"
        if runtime_helper.is_file():
            execution_env.setdefault("SURE_RUNTIME_ENV_HELPER", str(runtime_helper))
        context_base_overrides = remote_context.get("base_model_source_overrides")
        if not isinstance(context_base_overrides, dict):
            context_base_overrides = {}
        exp = make_exp(
            config=config,
            workspace=workspace,
            task_card=task_card,
            base_model_profile=base_model_profile,
            metric_runner=metric_runner,
            execution_env=execution_env,
        )
        exp.base_model_source_overrides = {
            str(key): str(value)
            for key, value in context_base_overrides.items()
            if value is not None
        }
        exp.candidate_stage_name = str(remote_context.get("stage") or "remote_training")
        exp.candidate_phase = str(remote_context.get("phase") or "")
        exp.candidate_rung_name = str(remote_context.get("rung") or "")
        exp.candidate_idea_id = str(remote_context.get("idea_id") or "")
        exp._ensure_workspace_dirs()
        exp._prepare_base_model_source_overrides()
        exp._prepare_workspace_inputs(role_paths)
        missing_base_model = ensure_base_model_paths(workspace, base_model_profile)
        if missing_base_model:
            write_failure_and_exit(
                reason_code="remote_base_model_missing",
                error="remote candidate workspace is missing required base_model path(s)",
                details={"missing_base_model_paths": missing_base_model},
                exit_code=2,
                candidate_type=exp.candidate_type_hint,
            )
        clear_candidate_outputs(workspace, role_paths)

        run_sure = workspace / "run_sure.py"
        if not run_sure.is_file():
            write_failure_and_exit(
                reason_code="missing_run_sure",
                error=f"Missing run_sure.py: {run_sure}",
                exit_code=2,
                candidate_type=exp.candidate_type_hint,
            )
        exp.code = run_sure.read_text(encoding="utf-8", errors="replace")
        boundary_errors = validate_sure_candidate_boundary(
            exp.code,
            metric_runner.sure_root,
            base_model_profile,
            canonical_task=task_card.canonical_task,
            require_asr_wrapper=exp._execution_env_truthy(
                "SURE_ASR_REQUIRE_ZIPFORMER_WRAPPER",
                False,
            ),
        )
        if boundary_errors:
            write_failure_and_exit(
                reason_code="boundary_error",
                error="run_sure.py violated SURE candidate constraints",
                details={"boundary_errors": boundary_errors},
                exit_code=2,
                candidate_type=exp.candidate_type_hint,
            )
        script_errors = validate_run_sure_script(exp.code, role_paths)
        if script_errors:
            write_failure_and_exit(
                reason_code="script_error",
                error="run_sure.py failed execution-readiness checks",
                details={"script_errors": script_errors},
                exit_code=2,
                candidate_type=exp.candidate_type_hint,
            )
        exp.candidate_type_hint = candidate_type_from_code(
            exp.code,
            default=str(remote_context.get("candidate_type_hint") or "inference"),
        )

        command = exp._execution_command()
        exp._candidate_started_at = time.time()
        execution_info = run_logged(
            command,
            cwd=workspace,
            timeout=args.timeout,
            log_path=run_log,
            env=process_env,
        )
        execution_info["scheduler_allocation"] = scheduler_allocation
        terminal_output = read_tail(run_log)
        if execution_info["exit_code"] != 0:
            reason_code = execution_failure_reason_code(
                terminal_output,
                execution_info,
            )
            write_failure_and_exit(
                reason_code=reason_code,
                error="run_sure.py exited non-zero in VC child job",
                terminal_output=terminal_output,
                details=execution_failure_details(
                    execution_info,
                    terminal_output,
                    workspace=workspace,
                ),
                exit_code=1,
                candidate_type=exp.candidate_type_hint,
            )

        missing = exp._missing_artifacts(role_paths)
        if missing:
            write_failure_and_exit(
                reason_code="missing_artifacts",
                error=f"required artifacts are missing: {missing}",
                terminal_output=terminal_output,
                details={"missing_artifacts": missing, "execution_info": execution_info},
                exit_code=1,
                candidate_type=exp.candidate_type_hint,
            )

        artifact_errors = exp._artifact_guard_errors(role_paths)
        if artifact_errors:
            write_failure_and_exit(
                reason_code="artifact_guard_failed",
                error="generated artifacts failed integrity checks",
                terminal_output=terminal_output,
                details={"artifact_guard_errors": artifact_errors, "execution_info": execution_info},
                exit_code=1,
                candidate_type=exp.candidate_type_hint,
            )

        metric_result = metric_runner.run(
            task_card=task_card,
            workspace_path=workspace,
            output_dir=metric_dir,
            role_paths=role_paths,
        )
        metric_feedback = format_metric_feedback(metric_result)
        payload = {
            "success": bool(metric_result.success),
            "reason_code": "success" if metric_result.success else "metric_failed",
            "score": metric_result.score,
            "metric": metric_result.metric,
            "metric_feedback": metric_feedback,
            "terminal_output": terminal_output,
            "execution_info": execution_info,
            "details": SureRunExp._metric_details(metric_result),
        }
        if not metric_result.success:
            payload["error"] = metric_result.error
        write_json(result_path, payload)
        write_status_json(
            workspace,
            success=bool(metric_result.success),
            reason_code="success" if metric_result.success else "metric_failed",
            stage=exp.candidate_stage_name or exp.stage,
            phase=exp.candidate_phase,
            rung=exp.candidate_rung_name,
            idea_id=exp.candidate_idea_id,
            candidate_type=exp.candidate_type_hint,
            score=metric_result.score,
            remote_success=bool(metric_result.success),
            details={"execution_info": execution_info, **payload["details"]},
            metric_feedback=metric_feedback,
            terminal_output=terminal_output,
        )
        sys.exit(0 if metric_result.success else 1)
    except Exception as exc:
        details = {
            "exception_type": type(exc).__name__,
            "traceback_tail": traceback.format_exc()[-12000:],
            "workspace": str(workspace),
            "config_path": str(config_path),
            "cwd": os.getcwd(),
            "python_executable": sys.executable,
        }
        write_json(
            result_path,
            failure_payload(
                str(exc),
                reason_code="remote_exception",
                terminal_output=read_tail(run_log),
                details=details,
            ),
        )
        write_status_json(
            workspace,
            success=False,
            reason_code="remote_exception",
            stage=context_value("stage", "remote_training"),
            phase=context_value("phase"),
            rung=context_value("rung"),
            idea_id=context_value("idea_id"),
            candidate_type=context_value("candidate_type_hint"),
            remote_success=False,
            details=details,
            metric_feedback=f"SURE metric not available because remote candidate failed: {exc}",
            terminal_output=read_tail(run_log),
        )
        raise


if __name__ == "__main__":
    main()
