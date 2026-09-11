from __future__ import annotations

import contextlib
import math
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .task_cards import SureTaskCard

try:
    import fcntl
except ImportError:  # pragma: no cover - SURE Master runs on Linux, this keeps import portable.
    fcntl = None  # type: ignore[assignment]


@dataclass
class SureMetricResult:
    success: bool
    score: float | None
    metric: str
    pipeline_id: str | None = None
    report_path: str | None = None
    summary_path: str | None = None
    details: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class MetricGpuAllocation:
    mode: str
    devices: list[str]
    cuda_visible_devices: str | None
    device_override: str | None = None
    attempt: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "devices": self.devices,
            "cuda_visible_devices": self.cuda_visible_devices,
            "device_override": self.device_override,
            "attempt": self.attempt,
        }


@dataclass
class MetricGpuSnapshot:
    index: str
    total_mib: int
    used_mib: int
    free_mib: int
    utilization: int


class MetricGpuAllocator:
    """Choose and lock GPUs for SURE metric scoring.

    Candidate execution has its own CUDA allocation. Metric scoring can need a
    different card because ASR/TTS scorers may load large models such as Whisper.
    """

    _NO_GPU_VALUES = {"", "none", "null", "false", "off", "disabled", "-1"}
    _AUTO_VALUES = {"auto", "all"}
    _IDLE_VALUES = {"idle", "free", "available"}

    def __init__(self, config: dict[str, Any] | None = None):
        config = dict(config or {})
        has_config = bool(config)
        self.enabled = self._bool(config.get("enabled"), default=has_config)
        self.devices = str(config.get("devices", "idle")).strip()
        self.gpus_per_metric = self._positive_int(config.get("gpus_per_metric"), 1)
        self.min_free_mib = self._non_negative_int(config.get("min_free_mib"), 0)
        self.max_utilization = self._non_negative_int(config.get("max_utilization"), 100)
        self.wait_timeout_sec = self._non_negative_float(config.get("wait_timeout_sec"), 0.0)
        self.poll_interval_sec = self._non_negative_float(config.get("poll_interval_sec"), 10.0)
        self.oom_retry = self._bool(config.get("oom_retry"), default=False)
        self.max_retries = self._non_negative_int(config.get("max_retries"), 0)
        self.cleanup_before_score = self._bool(config.get("cleanup_before_score"), default=False)
        self.lock_dir = Path(str(config.get("lock_dir") or "/tmp/sure_master_metric_gpu_locks"))
        self.respect_cuda_visible_devices = self._bool(
            config.get("respect_cuda_visible_devices"),
            default=True,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "MetricGpuAllocator | None":
        if not config:
            return None
        allocator = cls(config)
        return allocator if allocator.enabled else None

    @property
    def total_attempts(self) -> int:
        return 1 + self.max_retries if self.oom_retry else 1

    @contextlib.contextmanager
    def allocate(
        self,
        *,
        attempt: int,
        excluded_devices: set[str] | None = None,
    ) -> Iterator[MetricGpuAllocation]:
        excluded_devices = set(excluded_devices or set())
        mode = self.devices.strip().lower()
        if mode in self._NO_GPU_VALUES or mode == "cpu":
            with self._temporary_cuda_visible_devices("-1"):
                yield MetricGpuAllocation(
                    mode="cpu",
                    devices=[],
                    cuda_visible_devices="-1",
                    device_override="cpu",
                    attempt=attempt,
                )
            return

        if mode == "same":
            yield MetricGpuAllocation(
                mode="same",
                devices=self._visible_cuda_devices(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                attempt=attempt,
            )
            return

        handles: list[Any] = []
        try:
            devices = self._wait_for_devices(excluded_devices)
            handles = self._lock_devices(devices)
            cuda_visible = ",".join(devices)
            with self._temporary_cuda_visible_devices(cuda_visible):
                yield MetricGpuAllocation(
                    mode=mode or "idle",
                    devices=devices,
                    cuda_visible_devices=cuda_visible,
                    attempt=attempt,
                )
        finally:
            self._unlock(handles)

    def _wait_for_devices(self, excluded_devices: set[str]) -> list[str]:
        deadline = time.time() + self.wait_timeout_sec
        last_error = ""
        while True:
            snapshots = self._query_gpu_snapshots()
            groups = self._candidate_groups(snapshots, excluded_devices)
            for group in groups:
                handles = self._try_lock_devices(group)
                if not handles:
                    continue
                self._unlock(handles)
                return group

            last_error = self._format_unavailable_message(snapshots, excluded_devices)
            if self.wait_timeout_sec <= 0 or time.time() >= deadline:
                raise RuntimeError(last_error)
            time.sleep(max(0.1, self.poll_interval_sec))

    def _candidate_groups(
        self,
        snapshots: dict[str, MetricGpuSnapshot],
        excluded_devices: set[str],
    ) -> list[list[str]]:
        configured = self._configured_devices(snapshots)
        devices = [device for device in configured if device not in excluded_devices]
        mode = self.devices.strip().lower()
        if mode in self._IDLE_VALUES:
            devices = [
                device
                for device in devices
                if self._snapshot_is_idle(snapshots.get(device))
            ]
            devices.sort(
                key=lambda device: (
                    -(snapshots[device].free_mib if device in snapshots else 0),
                    snapshots[device].utilization if device in snapshots else 100,
                    int(device) if device.isdigit() else 10**9,
                )
            )
        else:
            devices.sort(key=lambda device: int(device) if device.isdigit() else 10**9)

        groups: list[list[str]] = []
        for start in range(0, len(devices), self.gpus_per_metric):
            group = devices[start : start + self.gpus_per_metric]
            if len(group) == self.gpus_per_metric:
                groups.append(group)
        return groups

    def _configured_devices(self, snapshots: dict[str, MetricGpuSnapshot]) -> list[str]:
        mode = self.devices.strip().lower()
        if mode in self._AUTO_VALUES or mode in self._IDLE_VALUES:
            devices = list(snapshots) or self._visible_cuda_devices()
        else:
            devices = self._parse_devices(self.devices)

        visible = set(self._visible_cuda_devices())
        if self.respect_cuda_visible_devices and visible:
            devices = [device for device in devices if device in visible]
        return list(dict.fromkeys(devices))

    def _snapshot_is_idle(self, snapshot: MetricGpuSnapshot | None) -> bool:
        if snapshot is None:
            return False
        return snapshot.free_mib >= self.min_free_mib and snapshot.utilization <= self.max_utilization

    def _query_gpu_snapshots(self) -> dict[str, MetricGpuSnapshot]:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=10,
            )
        except Exception:
            return {}
        if result.returncode != 0:
            return {}

        snapshots: dict[str, MetricGpuSnapshot] = {}
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                snapshots[parts[0]] = MetricGpuSnapshot(
                    index=parts[0],
                    total_mib=int(float(parts[1])),
                    used_mib=int(float(parts[2])),
                    free_mib=int(float(parts[3])),
                    utilization=int(float(parts[4])),
                )
            except ValueError:
                continue
        return snapshots

    def _try_lock_devices(self, devices: list[str]) -> list[Any] | None:
        try:
            self.lock_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return []

        handles: list[Any] = []
        for device in devices:
            path = self.lock_dir / f"gpu_{self._safe_lock_name(device)}.lock"
            handle = path.open("a+")
            if fcntl is not None:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    self._unlock(handles)
                    return None
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid={os.getpid()} time={time.time()} device={device}\n")
            handle.flush()
            handles.append(handle)
        return handles

    def _lock_devices(self, devices: list[str]) -> list[Any]:
        handles = self._try_lock_devices(devices)
        if handles is None:
            raise RuntimeError(f"Metric GPU lock race for devices {devices}; retry later.")
        return handles

    def _unlock(self, handles: list[Any]) -> None:
        for handle in reversed(handles):
            try:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    @contextlib.contextmanager
    def _temporary_cuda_visible_devices(self, value: str) -> Iterator[None]:
        old_value = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = value
        try:
            yield
        finally:
            if old_value is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_value

    def _format_unavailable_message(
        self,
        snapshots: dict[str, MetricGpuSnapshot],
        excluded_devices: set[str],
    ) -> str:
        mode = self.devices.strip().lower() or "idle"
        rows = [
            (
                f"{snapshot.index}:free={snapshot.free_mib}MiB,"
                f"util={snapshot.utilization}%,used={snapshot.used_mib}MiB"
            )
            for snapshot in snapshots.values()
        ]
        return (
            f"No metric GPU available for mode={mode!r}, gpus_per_metric={self.gpus_per_metric}, "
            f"min_free_mib={self.min_free_mib}, max_utilization={self.max_utilization}, "
            f"excluded={sorted(excluded_devices)}. Snapshot: {rows or '[unavailable]'}"
        )

    @staticmethod
    def _visible_cuda_devices() -> list[str]:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not visible or visible.lower() in MetricGpuAllocator._NO_GPU_VALUES:
            return []
        return MetricGpuAllocator._parse_devices(visible)

    @staticmethod
    def _parse_devices(raw: Any) -> list[str]:
        if isinstance(raw, (list, tuple)):
            values: list[str] = []
            for item in raw:
                values.extend(MetricGpuAllocator._parse_devices(item))
            return values
        return [
            item.strip()
            for item in str(raw).split(",")
            if item.strip() and item.strip().lower() not in MetricGpuAllocator._NO_GPU_VALUES
        ]

    @staticmethod
    def _safe_lock_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)

    @staticmethod
    def _bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        text = str(value).strip().lower()
        if text == "":
            return default
        return text in {"1", "true", "yes", "on"}

    @staticmethod
    def _positive_int(value: Any, default: int) -> int:
        try:
            return max(1, int(float(str(value).strip())))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _non_negative_int(value: Any, default: int) -> int:
        try:
            return max(0, int(float(str(value).strip())))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _non_negative_float(value: Any, default: float) -> float:
        try:
            parsed = float(str(value).strip())
        except (TypeError, ValueError):
            return default
        return max(0.0, parsed)


def _is_cuda_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "cuda out of memory",
            "torch.cuda.outofmemoryerror",
            "cuda error: out of memory",
            "cublas_status_alloc_failed",
        )
    )


def _empty_cuda_cache() -> None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _proc_text(path: Path, *, binary: bool = False) -> str:
    try:
        if binary:
            return path.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="ignore")
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _workspace_process_matches(entry: Path, workspace: Path) -> bool:
    raw_cmd = _proc_text(entry / "cmdline", binary=True)
    if not raw_cmd:
        return False
    cmd_lower = raw_cmd.lower()
    if "sure_eval" in cmd_lower or "whisper_large_v3" in cmd_lower:
        return False
    markers = (
        "run_sure.py",
        "run_f5tts_batch_infer.py",
        "run_f5tts_finetune.py",
        "run_f5tts_arch_finetune.py",
        "f5_tts",
        "f5tts",
        "torchrun",
        "accelerate",
    )
    if not any(marker in cmd_lower for marker in markers):
        return False
    workspace_s = str(workspace.resolve())
    if workspace_s in raw_cmd:
        return True
    try:
        cwd = (entry / "cwd").resolve()
        if cwd == workspace or workspace in cwd.parents:
            return True
    except Exception:
        pass
    return workspace_s in _proc_text(entry / "environ", binary=True)


def cleanup_workspace_candidate_processes(workspace: str | Path, wait_seconds: float = 10.0) -> list[int]:
    """Best-effort cleanup of stale candidate-side worker process groups."""
    workspace_path = Path(workspace).resolve()
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return []

    protected = {os.getpid(), os.getppid()}
    protected_pgids = {os.getpgrp()}
    try:
        protected_pgids.add(os.getpgid(os.getppid()))
    except Exception:
        pass
    try:
        uid = os.getuid()
    except Exception:
        uid = None

    pgids: set[int] = set()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in protected:
            continue
        try:
            if uid is not None and entry.stat().st_uid != uid:
                continue
            if _workspace_process_matches(entry, workspace_path):
                pgid = os.getpgid(pid)
                if pgid > 1 and pgid not in protected and pgid not in protected_pgids:
                    pgids.add(pgid)
        except Exception:
            continue

    killed: list[int] = []
    for pgid in sorted(pgids):
        try:
            os.killpg(pgid, signal.SIGTERM)
            killed.append(pgid)
        except (ProcessLookupError, PermissionError):
            continue
    if killed:
        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            remaining = []
            for pgid in killed:
                try:
                    os.killpg(pgid, 0)
                    remaining.append(pgid)
                except (ProcessLookupError, PermissionError):
                    pass
            if not remaining:
                break
            time.sleep(0.25)
        for pgid in killed:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    return killed


class SureMetricRunner:
    """Read-only adapter around SURE evaluation pipelines."""

    def __init__(
        self,
        sure_root: str | Path,
        pythonpath: str | Path | None = None,
        device: str = "cuda",
        cache_dir: str | None = None,
        validate_env: bool = False,
        metric_gpu: dict[str, Any] | None = None,
        python: str | None = None,
    ):
        self.worker_python = python
        self.sure_root = Path(sure_root)
        self.pythonpath = Path(pythonpath) if pythonpath else self.sure_root / "src"
        self.device = device
        self.cache_dir = cache_dir
        self.validate_env = validate_env
        self.metric_gpu_allocator = MetricGpuAllocator.from_config(metric_gpu)

    def run(
        self,
        task_card: SureTaskCard,
        workspace_path: str | Path,
        output_dir: str | Path,
        role_paths: dict[str, str | None],
    ) -> SureMetricResult:
        """Run SURE for a generated artifact set.

        This method only imports/calls SURE. It never writes into the SURE source
        tree; all generated specs/reports are written under output_dir.
        """
        if self.worker_python:
            return self._run_isolated(task_card, workspace_path, output_dir, role_paths)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        workspace = Path(workspace_path)
        old_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        attempts: list[dict[str, Any]] = []
        excluded_devices: set[str] = set()
        try:
            self._prepare_import_path()
            from sure_eval.evaluation.cli_adapters import (  # type: ignore
                build_pipeline_spec,
                run_pipeline_spec,
            )
            if self.validate_env:
                from sure_eval.evaluation.env_check import (  # type: ignore
                    check_pipeline_environment,
                    raise_if_environment_failed,
                )
            else:
                check_pipeline_environment = None
                raise_if_environment_failed = None

            pipeline = self._build_pipeline_spec(build_pipeline_spec, task_card)
            pipeline_path = output_path / "pipeline_spec.json"
            pipeline_path.write_text(
                json.dumps(pipeline, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            if self.validate_env and check_pipeline_environment and raise_if_environment_failed:
                env_results = check_pipeline_environment(pipeline)
                raise_if_environment_failed(env_results)

            args = self._build_run_args(task_card, workspace, role_paths)
            total_attempts = (
                self.metric_gpu_allocator.total_attempts
                if self.metric_gpu_allocator is not None
                else 1
            )
            if self.metric_gpu_allocator and self.metric_gpu_allocator.cleanup_before_score:
                killed_pgids = cleanup_workspace_candidate_processes(workspace)
                if killed_pgids:
                    attempts.append({"event": "cleanup_before_score", "killed_pgids": killed_pgids})

            last_error: Exception | None = None
            for attempt_no in range(1, total_attempts + 1):
                try:
                    with self._metric_gpu_context(attempt_no, excluded_devices) as allocation:
                        attempts.append({"event": "attempt_start", "metric_gpu": allocation.to_dict()})
                        summary = run_pipeline_spec(
                            pipeline,
                            output_dir=str(output_path),
                            device=allocation.device_override or self.device,
                            cache_dir=self.cache_dir,
                            **args,
                        )
                    score = summary.get("score")
                    if score is None:
                        raise ValueError(f"SURE returned no score: {summary}")
                    score = float(score)
                    if not math.isfinite(score):
                        raise ValueError("SURE returned a nonfinite score")
                    summary_path = output_path / "score_summary.json"
                    payload = {
                        **summary,
                        "task_card": task_card.to_dict(),
                        "role_paths": args,
                        "metric_gpu_attempts": attempts,
                    }
                    summary_path.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    self._write_metric_gpu_attempts(output_path, attempts)
                    return SureMetricResult(
                        success=True,
                        score=score,
                        metric=str(summary.get("metric") or task_card.primary_metric),
                        pipeline_id=summary.get("pipeline_id"),
                        report_path=summary.get("report_path"),
                        summary_path=str(summary_path),
                        details=payload,
                    )
                except Exception as e:
                    last_error = e
                    oom = _is_cuda_oom_error(e)
                    attempts.append(
                        {
                            "event": "attempt_error",
                            "attempt": attempt_no,
                            "is_cuda_oom": oom,
                            "error": str(e),
                        }
                    )
                    self._write_attempt_error(output_path, attempt_no, e, task_card, role_paths, attempts)
                    _empty_cuda_cache()
                    if self.metric_gpu_allocator and oom:
                        for event in reversed(attempts):
                            gpu = event.get("metric_gpu") if isinstance(event, dict) else None
                            if isinstance(gpu, dict):
                                excluded_devices.update(str(device) for device in gpu.get("devices") or [])
                                break
                    if (
                        self.metric_gpu_allocator is not None
                        and self.metric_gpu_allocator.oom_retry
                        and oom
                        and attempt_no < total_attempts
                    ):
                        continue
                    break
            if last_error is not None:
                raise last_error
            raise RuntimeError("SURE metric failed before running any attempt.")
        except Exception as e:
            error_path = output_path / "metric_error.json"
            error_payload = {
                "status": "error",
                "error": str(e),
                "task_card": task_card.to_dict(),
                "role_paths": role_paths,
                "metric_gpu_attempts": attempts,
            }
            error_path.write_text(
                json.dumps(error_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self._write_metric_gpu_attempts(output_path, attempts)
            return SureMetricResult(
                success=False,
                score=None,
                metric=task_card.primary_metric,
                summary_path=str(error_path),
                error=str(e),
                details=error_payload,
            )
        finally:
            sys.dont_write_bytecode = old_dont_write_bytecode

    def _run_isolated(self, task_card, workspace_path, output_dir, role_paths):
        import json
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        request = {"sure_root": str(self.sure_root.resolve()), "pythonpath": str(self.pythonpath.resolve()),
                   "device": self.device, "cache_dir": self.cache_dir, "validate_env": self.validate_env,
                   "task_card": task_card.to_dict(), "workspace": str(Path(workspace_path).resolve()),
                   "output": str(output), "roles": role_paths}
        request_path, response_path = output / "score_request.json", output / "score_result.json"
        request_path.write_text(json.dumps(request))
        response_path.unlink(missing_ok=True)
        # Use the package root, not the model runtime.
        tool = Path(__file__).resolve().parents[2] / "tools" / "score_task.py"
        try:
            with (output / "score_worker.log").open("w") as log:
                from ...runtime.process import run_bounded
                run_bounded([self.worker_python, str(tool), str(request_path), str(response_path)],
                            output=log, timeout=21600)
            return SureMetricResult(**json.loads(response_path.read_text()))
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return SureMetricResult(False, None, task_card.primary_metric, error=f"Scoring worker failed: {exc}; see {output / 'score_worker.log'}")

    @contextlib.contextmanager
    def _metric_gpu_context(
        self,
        attempt_no: int,
        excluded_devices: set[str],
    ) -> Iterator[MetricGpuAllocation]:
        if self.metric_gpu_allocator is None:
            yield MetricGpuAllocation(
                mode="disabled",
                devices=[],
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                attempt=attempt_no,
            )
            return
        with self.metric_gpu_allocator.allocate(
            attempt=attempt_no,
            excluded_devices=excluded_devices,
        ) as allocation:
            yield allocation

    @staticmethod
    def _write_metric_gpu_attempts(output_path: Path, attempts: list[dict[str, Any]]) -> None:
        if not attempts:
            return
        (output_path / "metric_gpu_attempts.json").write_text(
            json.dumps(attempts, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_attempt_error(
        output_path: Path,
        attempt_no: int,
        error: Exception,
        task_card: SureTaskCard,
        role_paths: dict[str, str | None],
        attempts: list[dict[str, Any]],
    ) -> None:
        payload = {
            "status": "error",
            "attempt": attempt_no,
            "error": str(error),
            "task_card": task_card.to_dict(),
            "role_paths": role_paths,
            "metric_gpu_attempts": attempts,
        }
        (output_path / f"metric_error_attempt_{attempt_no}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _prepare_import_path(self) -> None:
        if not self.sure_root.exists():
            raise FileNotFoundError(f"SURE root does not exist: {self.sure_root}")
        if not self.pythonpath.exists():
            raise FileNotFoundError(f"SURE pythonpath does not exist: {self.pythonpath}")
        pythonpath = str(self.pythonpath)
        if pythonpath not in sys.path:
            sys.path.insert(0, pythonpath)
        existing = os.environ.get("PYTHONPATH", "")
        parts = [p for p in existing.split(os.pathsep) if p]
        if pythonpath not in parts:
            os.environ["PYTHONPATH"] = pythonpath + (os.pathsep + existing if existing else "")

    def _build_pipeline_spec(self, build_pipeline_spec, task_card: SureTaskCard) -> dict[str, Any]:
        old_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            try:
                return build_pipeline_spec(
                    task_card.task_alias,
                    language=task_card.language,
                    metric=task_card.primary_metric,
                )
            except TypeError as e:
                if task_card.task_alias != "classification":
                    raise
                if "multiple values for argument 'task'" not in str(e):
                    raise
                return self._build_generic_classification_pipeline_spec(task_card)
        finally:
            sys.dont_write_bytecode = old_dont_write_bytecode

    @staticmethod
    def _build_generic_classification_pipeline_spec(task_card: SureTaskCard) -> dict[str, Any]:
        """Compatibility shim for SURE versions whose CLI adapter cannot describe generic classification."""
        from sure_eval.evaluation.cli_adapters import (  # type: ignore
            ROLE_TO_CLI_ARG,
            _match_selected_route,
            _node_slots,
            _requested_metrics,
            _route_choices,
        )
        from sure_eval.evaluation.scripts.classification import (  # type: ignore
            describe_pipeline as describe_classification_pipeline,
        )
        from sure_eval.evaluation.scripts.contracts import load_task_routes  # type: ignore

        description = describe_classification_pipeline(
            task="classification",
            metric=task_card.primary_metric,
        )
        routes, _ = load_task_routes("classification")
        route_choices = _route_choices(routes)
        selected_route = _match_selected_route(route_choices, description.pipeline_id)
        run_args = {
            ROLE_TO_CLI_ARG.get(role, role): None
            for role in description.required_roles
        }
        run_args["output_dir"] = None
        return {
            "schema": "sure.metric.pipeline.v1",
            "task": "classification",
            "task_alias": "classification",
            "language": description.language,
            "metric": description.metric,
            "metrics": list(
                _requested_metrics(
                    "classification",
                    metric=task_card.primary_metric,
                    description_metric=description.metric,
                )
            ),
            "pipeline_id": description.pipeline_id,
            "route_id": selected_route.get("route_id", description.pipeline_id),
            "pipeline": _node_slots(
                description.node_ids,
                selected_route=selected_route,
                route_choices=route_choices,
            ),
            "required_roles": list(description.required_roles),
            "optional_roles": list(description.optional_roles),
            "run_args": run_args,
            "route_choices": route_choices,
            "task_config_path": description.task_config_path,
            "nodes": list(description.nodes),
            "conversion_steps": list(description.conversion_steps),
        }

    def _build_run_args(
        self,
        task_card: SureTaskCard,
        workspace: Path,
        configured_role_paths: dict[str, str | None],
    ) -> dict[str, str]:
        values: dict[str, str] = {}
        for role in set(task_card.required_roles + task_card.optional_roles):
            candidate = configured_role_paths.get(role)
            if not candidate:
                candidate = task_card.artifact_contract.get(role)
            if not candidate:
                continue
            values[self._role_to_cli_arg(role)] = self._resolve_path(candidate, workspace)

        missing = [
            role
            for role in task_card.required_roles
            if self._role_to_cli_arg(role) not in values
        ]
        if missing:
            raise ValueError(f"Missing required SURE role paths: {missing}")
        return values

    @staticmethod
    def _role_to_cli_arg(role: str) -> str:
        mapping = {
            "ref": "ref_file",
            "hyp": "hyp_file",
            "src": "src_file",
            "prompt_jsonl": "prompt_jsonl",
            "label_spec": "label_spec",
            "reference_jsonl": "reference_jsonl",
            "sample_output": "sample_output",
            "wekws_label_file": "wekws_label_file",
            "wekws_score_file": "wekws_score_file",
            "wekws_frame_score_file": "wekws_frame_score_file",
            "keyword": "keyword",
            "samples_jsonl": "samples_jsonl",
        }
        return mapping.get(role, role)

    @staticmethod
    def _resolve_path(value: str, workspace: Path) -> str:
        if value.startswith("literal:"):
            return value.removeprefix("literal:")
        path = Path(value)
        if path.is_absolute():
            return str(path)
        return str((workspace / path).resolve())


def format_metric_feedback(metric_result: SureMetricResult) -> str:
    if metric_result.success:
        return (
            f"SURE metric succeeded. metric={metric_result.metric}, "
            f"score={metric_result.score}, pipeline_id={metric_result.pipeline_id}, "
            f"report_path={metric_result.report_path}"
        )
    return (
        f"SURE metric failed. metric={metric_result.metric}, "
        f"error={metric_result.error}, details_path={metric_result.summary_path}"
    )
