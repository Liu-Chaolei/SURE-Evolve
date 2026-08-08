"""Local environment implementation.

Provides the low-level operations interface for local environments.
"""

from __future__ import annotations

import os
import signal
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from .base import BaseEnv, EnvConfig
from evomaster.agent.session.base import SessionConfig


class LocalEnvConfig(EnvConfig):
    """Local environment configuration."""
    session_config: SessionConfig = Field(
        ...,
        description="Session configuration"
    )


class ResourceAllocator:
    """Resource allocator.

    Allocates GPU and CPU resources based on the parallel index.
    """

    def __init__(
        self,
        gpu_devices: str | list[str] | None,
        cpu_devices: str | list[int] | None,
        max_parallel: int,
        gpus_per_exp: int = 1,
        logger: Any = None,
        refresh_idle_gpus: bool = False,
        idle_gpu_min_free_mib: int = 0,
        idle_gpu_max_utilization: int = 100,
        idle_gpu_allow_busy_fallback: bool = True,
        gpu_lock_enabled: bool = False,
        gpu_lock_dir: str | None = None,
        gpu_lock_wait_seconds: int = 0,
        gpu_lock_poll_seconds: float = 1.0,
    ):
        """Initialize the resource allocator.

        Args:
            gpu_devices: GPU device configuration.
            cpu_devices: CPU device configuration.
            max_parallel: Maximum number of parallel executions.
            gpus_per_exp: Number of adjacent GPUs allocated to each parallel execution.
            logger: Logger instance.
        """
        self.gpu_devices = gpu_devices
        self.cpu_devices = cpu_devices
        self.max_parallel = max_parallel
        self.gpus_per_exp = max(1, int(gpus_per_exp or 1))
        self.logger = logger
        self.refresh_idle_gpus = bool(refresh_idle_gpus)
        self.idle_gpu_min_free_mib = max(0, int(idle_gpu_min_free_mib or 0))
        self.idle_gpu_max_utilization = max(0, int(idle_gpu_max_utilization or 100))
        self.idle_gpu_allow_busy_fallback = bool(idle_gpu_allow_busy_fallback)
        self.gpu_lock_enabled = bool(gpu_lock_enabled)
        self.gpu_lock_dir = Path(gpu_lock_dir or "/tmp/evomaster_gpu_locks")
        self.gpu_lock_wait_seconds = max(0, int(gpu_lock_wait_seconds or 0))
        self.gpu_lock_poll_seconds = max(0.1, float(gpu_lock_poll_seconds or 1.0))
        self._lock = threading.Lock()
        self._active_executions: dict[int, threading.Thread] = {}
        
        # Parse GPU device list
        self._gpu_list: list[str] = []
        if gpu_devices is not None:
            if isinstance(gpu_devices, str):
                raw_gpu_devices = gpu_devices.strip()
                if raw_gpu_devices == "all":
                    # If configured as "all", available GPUs should be obtained from the environment
                    # Simplified here: assume the user has explicitly specified the GPU list
                    self._gpu_list = []
                else:
                    self._gpu_list = [
                        gpu.strip()
                        for gpu in raw_gpu_devices.split(",")
                        if gpu.strip()
                    ]
            elif isinstance(gpu_devices, list):
                self._gpu_list = [str(gpu) for gpu in gpu_devices]
        
        # Parse CPU device list
        self._cpu_list: list[int] = []
        if cpu_devices is not None:
            if isinstance(cpu_devices, str):
                # Parse a range string, e.g., "0-35"
                if "-" in cpu_devices:
                    start, end = map(int, cpu_devices.split("-"))
                    self._cpu_list = list(range(start, end + 1))
                else:
                    self._cpu_list = [int(cpu_devices)]
            elif isinstance(cpu_devices, list):
                self._cpu_list = cpu_devices
    
    def allocate_resources(self, parallel_index: int) -> tuple[str | None, str | None]:
        """Allocate resources for the specified parallel index.

        Args:
            parallel_index: Parallel index (starting from 0).

        Returns:
            (gpu_allocation, cpu_allocation) tuple.
            - gpu_allocation: GPU device string, e.g., "0" or "0,1"; None means no GPU restriction.
            - cpu_allocation: CPU device string, e.g., "0-11" or "0,1,2"; None means no CPU restriction.
        """
        # Allocate GPU
        gpu_allocation = None
        if self._gpu_list:
            # If there are not enough GPUs, some parallel processes will share GPUs.
            # gpu_devices entries may themselves be comma groups (e.g. "0,1");
            # gpus_per_exp groups adjacent entries for single experiments that
            # need DDP (e.g. ["1", "2"] + gpus_per_exp=2 -> "1,2").
            if self.gpus_per_exp == 1:
                gpu_index = parallel_index % len(self._gpu_list)
                gpu_allocation = self._gpu_list[gpu_index]
            else:
                effective_index = parallel_index % self.max_parallel
                total_gpus = len(self._gpu_list)
                group_size = min(self.gpus_per_exp, total_gpus)
                start_index = (effective_index * self.gpus_per_exp) % total_gpus
                allocated_gpus = [
                    self._gpu_list[(start_index + offset) % total_gpus]
                    for offset in range(group_size)
                ]
                gpu_allocation = ",".join(allocated_gpus)
        
        # Allocate CPU (evenly distributed)
        # When the number of tasks exceeds max_parallel, use modulo to reuse resource slots
        # (e.g., with 3 tasks and max_parallel=2, task 2 reuses slot 0)
        cpu_allocation = None
        if self._cpu_list:
            total_cpus = len(self._cpu_list)
            cpus_per_parallel = total_cpus // self.max_parallel
            if cpus_per_parallel > 0:
                effective_index = parallel_index % self.max_parallel
                start_index = effective_index * cpus_per_parallel
                end_index = start_index + cpus_per_parallel - 1
                # Handle the last parallel process: allocate all remaining CPUs
                if effective_index == self.max_parallel - 1:
                    end_index = total_cpus - 1
                
                allocated_cpus = self._cpu_list[start_index:end_index + 1]
                if allocated_cpus:
                    if len(allocated_cpus) == 1:
                        cpu_allocation = str(allocated_cpus[0])
                    else:
                        # Check if the CPUs are contiguous
                        if allocated_cpus == list(range(allocated_cpus[0], allocated_cpus[-1] + 1)):
                            cpu_allocation = f"{allocated_cpus[0]}-{allocated_cpus[-1]}"
                        else:
                            cpu_allocation = ",".join(str(cpu) for cpu in allocated_cpus)
        
        return gpu_allocation, cpu_allocation

    def prepare_gpu_allocation(
        self,
        gpu_allocation: str | None,
    ) -> tuple[str | None, list[Any]]:
        """Validate/reselect a GPU allocation and hold cross-process locks.

        ``allocate_resources`` is deterministic inside one process. This method
        adds the expensive checks that should happen immediately before process
        launch: refresh idle GPU state and acquire file locks so concurrent
        EvoMaster runs do not race onto the same GPU.
        """
        if gpu_allocation is None:
            return None, []

        requested = _split_gpu_allocation(gpu_allocation)
        if not requested:
            return gpu_allocation, []

        candidates = self._gpu_candidate_groups(requested)
        deadline = time.time() + self.gpu_lock_wait_seconds
        last_error = ""
        while True:
            for group in candidates:
                handles, error = self._try_lock_gpu_group(group)
                if handles is not None:
                    prepared = ",".join(group)
                    if prepared != gpu_allocation and self.logger:
                        self.logger.info(
                            "重新选择空闲 GPU: requested=%s, selected=%s",
                            gpu_allocation,
                            prepared,
                        )
                    return prepared, handles
                last_error = error or last_error

            if time.time() >= deadline:
                break
            time.sleep(self.gpu_lock_poll_seconds)
            candidates = self._gpu_candidate_groups(requested)

        suffix = f"; last lock error: {last_error}" if last_error else ""
        raise RuntimeError(
            "No eligible GPU allocation is available "
            f"for requested={gpu_allocation}, min_free_mib={self.idle_gpu_min_free_mib}, "
            f"max_utilization={self.idle_gpu_max_utilization}, lock_enabled={self.gpu_lock_enabled}"
            f"{suffix}"
        )

    def release_gpu_locks(self, handles: list[Any]) -> None:
        """Release locks returned by prepare_gpu_allocation."""
        for handle in handles:
            try:
                import fcntl  # type: ignore

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                handle.close()
            except Exception:
                pass

    def _gpu_candidate_groups(self, requested: list[str]) -> list[list[str]]:
        required = max(1, len(requested))
        candidates: list[list[str]] = []
        idle_devices: list[str] = []
        if self.refresh_idle_gpus:
            allowed = set(self._gpu_list) if self._gpu_list else None
            idle_devices = _discover_idle_gpu_devices_for_allocator(
                allowed=allowed,
                min_free_mib=self.idle_gpu_min_free_mib,
                max_utilization=self.idle_gpu_max_utilization,
                logger=self.logger,
            )
            idle_set = set(idle_devices)
            if all(gpu in idle_set for gpu in requested):
                candidates.append(requested)
            for start in range(0, max(0, len(idle_devices) - required + 1)):
                group = idle_devices[start:start + required]
                if len(group) == required and group not in candidates:
                    candidates.append(group)
            if not candidates and not self.idle_gpu_allow_busy_fallback:
                return []

        if requested not in candidates and (
            not self.refresh_idle_gpus or self.idle_gpu_allow_busy_fallback
        ):
            candidates.append(requested)
        return candidates

    def _try_lock_gpu_group(self, group: list[str]) -> tuple[list[Any] | None, str]:
        if not self.gpu_lock_enabled:
            return [], ""
        self.gpu_lock_dir.mkdir(parents=True, exist_ok=True)
        handles: list[Any] = []
        try:
            import fcntl  # type: ignore

            for gpu in group:
                lock_path = self.gpu_lock_dir / f"gpu_{_safe_lock_name(gpu)}.lock"
                handle = lock_path.open("a+")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as e:
                    handle.close()
                    self.release_gpu_locks(handles)
                    return None, f"{lock_path}: {e}"
                handle.seek(0)
                handle.truncate()
                handle.write(f"pid={os.getpid()} thread={threading.get_ident()} gpu={gpu}\n")
                handle.flush()
                handles.append(handle)
        except Exception as e:
            self.release_gpu_locks(handles)
            return None, str(e)
        return handles, ""
    
    def register_execution(self, parallel_index: int) -> None:
        """Register an execution task.

        Args:
            parallel_index: Parallel index.

        Raises:
            RuntimeError: If the maximum parallel count is reached or the index is already executing.
        """
        with self._lock:
            # Check if the maximum parallel count has been reached
            if len(self._active_executions) >= self.max_parallel:
                raise RuntimeError(
                    f"已达到最大并行数量限制 ({self.max_parallel})。"
                    f"当前活跃执行数: {len(self._active_executions)}"
                )
            
            # Check if the index is already executing
            if parallel_index in self._active_executions:
                raise RuntimeError(
                    f"并行索引 {parallel_index} 已在执行中，不能重复执行"
                )
            
            current_thread = threading.current_thread()
            self._active_executions[parallel_index] = current_thread
            
            if self.logger:
                self.logger.info(
                    f"注册并行执行: index={parallel_index}, "
                    f"当前活跃数={len(self._active_executions)}/{self.max_parallel}"
                )
    
    def unregister_execution(self, parallel_index: int) -> None:
        """Unregister an execution task.

        Args:
            parallel_index: Parallel index.
        """
        with self._lock:
            if parallel_index in self._active_executions:
                del self._active_executions[parallel_index]
                
            if self.logger:
                self.logger.info(
                    f"注销并行执行: index={parallel_index}, "
                    f"当前活跃数={len(self._active_executions)}/{self.max_parallel}"
                )


def _split_gpu_allocation(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _safe_lock_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "y", "on"}


def _positive_int(value: Any, default: int) -> int:
    try:
        return max(0, int(float(str(value).strip())))
    except (TypeError, ValueError):
        return default


def _parse_nvidia_smi_number(value: str) -> float:
    text = str(value).strip()
    number = "".join(ch for ch in text if ch.isdigit() or ch in ".-")
    if number in {"", ".", "-", "-."}:
        raise ValueError(f"not a number: {value!r}")
    return float(number)


def _discover_idle_gpu_devices_for_allocator(
    *,
    allowed: set[str] | None,
    min_free_mib: int,
    max_utilization: int,
    logger: Any = None,
) -> list[str]:
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
        if logger:
            logger.debug("nvidia-smi idle GPU refresh unavailable: %s", e)
        return []

    if result.returncode != 0:
        if logger and result.stderr.strip():
            logger.debug("nvidia-smi idle GPU refresh failed: %s", result.stderr.strip())
        return []

    idle: list[tuple[str, float, float, float, float]] = []
    skipped: list[tuple[str, float, float, str]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
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
    devices = [item[0] for item in idle]
    if logger:
        logger.info(
            "Refreshed idle GPU candidates: %s (min_free_mib=%s, max_utilization=%s, allowed=%s)",
            devices,
            min_free_mib,
            max_utilization,
            sorted(allowed) if allowed is not None else "all",
        )
        if skipped:
            logger.debug("Skipped busy GPUs during refresh: %s", skipped)
    return devices


class LocalEnv(BaseEnv):
    """Local environment implementation.

    Provides the low-level operations interface for local environments:
    - Command execution
    - File operations
    - Workspace management
    """

    def __init__(self, config: LocalEnvConfig | None = None):
        """Initialize the local environment.

        Args:
            config: Local environment configuration.
        """
        if config is None:
            raise ValueError("LocalEnv requires LocalEnvConfig with session_config")
        super().__init__(config)
        self.config: LocalEnvConfig = config
        self._resource_allocator: ResourceAllocator | None = None
        self._init_resource_allocator()
    
    def _init_resource_allocator(self) -> None:
        """Initialize the resource allocator."""
        session_config = self.config.session_config
        parallel_config = getattr(session_config, 'parallel', None)
        
        if parallel_config and parallel_config.get('enabled', False):
            max_parallel = parallel_config.get('max_parallel', 1)
            gpus_per_exp = parallel_config.get('gpus_per_exp', 1)
            gpu_devices = getattr(session_config, 'gpu_devices', None)
            cpu_devices = getattr(session_config, 'cpu_devices', None)
            
            self._resource_allocator = ResourceAllocator(
                gpu_devices=gpu_devices,
                cpu_devices=cpu_devices,
                max_parallel=max_parallel,
                gpus_per_exp=gpus_per_exp,
                logger=self.logger,
                refresh_idle_gpus=_truthy(parallel_config.get('refresh_idle_gpu_before_exec')),
                idle_gpu_min_free_mib=_positive_int(
                    parallel_config.get('idle_gpu_min_free_mib'),
                    0,
                ),
                idle_gpu_max_utilization=_positive_int(
                    parallel_config.get('idle_gpu_max_utilization'),
                    100,
                ),
                idle_gpu_allow_busy_fallback=_truthy(
                    parallel_config.get('idle_gpu_allow_busy_fallback'),
                    default=True,
                ),
                gpu_lock_enabled=_truthy(parallel_config.get('gpu_lock_enabled')),
                gpu_lock_dir=parallel_config.get('gpu_lock_dir'),
                gpu_lock_wait_seconds=_positive_int(
                    parallel_config.get('gpu_lock_wait_seconds'),
                    0,
                ),
                gpu_lock_poll_seconds=float(parallel_config.get('gpu_lock_poll_seconds', 1.0)),
            )
            self.logger.info(
                f"初始化资源分配器: max_parallel={max_parallel}, "
                f"gpus_per_exp={gpus_per_exp}, "
                f"gpu_devices={gpu_devices}, cpu_devices={cpu_devices}, "
                f"refresh_idle_gpu_before_exec={parallel_config.get('refresh_idle_gpu_before_exec')}, "
                f"gpu_lock_enabled={parallel_config.get('gpu_lock_enabled')}"
            )

    def _is_split_workspace_enabled(self) -> bool:
        """Check if split_workspace_for_exp is enabled.

        Returns:
            Whether independent experiment workspaces are enabled.
        """
        session_config = self.config.session_config
        parallel_config = getattr(session_config, 'parallel', None)
        if parallel_config and isinstance(parallel_config, dict):
            return parallel_config.get('split_workspace_for_exp', False)
        return False

    def setup(self) -> None:
        """Initialize the local environment."""
        if self._is_ready:
            self.logger.warning("Environment already setup")
            return

        self.logger.info("Setting up local environment")
        
        # Ensure the workspace directory exists
        workspace = Path(self.config.session_config.workspace_path)
        workspace.mkdir(parents=True, exist_ok=True)
        
        # Create symlinks (if configured)
        # When split_workspace_for_exp is enabled, skip symlink creation for the main workspace
        # Symlinks will be created in each experiment's independent workspace (via setup_exp_workspace)
        if not self._is_split_workspace_enabled():
            session_config = self.config.session_config
            if hasattr(session_config, 'symlinks') and session_config.symlinks:
                self._create_symlinks(workspace, session_config.symlinks)
        else:
            self.logger.info(
                "split_workspace_for_exp 已启用，跳过主工作空间的软链接创建，"
                "将在各 exp 工作空间中单独创建"
            )
        
        self._is_ready = True
        self.logger.info("Local environment setup complete")

    def setup_exp_workspace(self, exp_workspace_path: str) -> None:
        """Create an experiment-specific workspace directory.

        When split_workspace_for_exp is enabled, creates an independent workspace
        subdirectory for each experiment, and creates symlinks within it (if configured).

        Args:
            exp_workspace_path: Absolute path of the experiment workspace.
        """
        workspace = Path(exp_workspace_path)
        workspace.mkdir(parents=True, exist_ok=True)
        
        # Create symlinks in the experiment workspace
        session_config = self.config.session_config
        if hasattr(session_config, 'symlinks') and session_config.symlinks:
            self._create_symlinks(workspace, session_config.symlinks)
        
        self.logger.info(f"创建实验独立工作空间: {exp_workspace_path}")

    def teardown(self) -> None:
        """Clean up local environment resources."""
        if not self._is_ready:
            return

        self.logger.info("Tearing down local environment")
        self._is_ready = False
        self.logger.info("Local environment teardown complete")

    def get_session(self) -> Any:
        """Get a Session (LocalEnv does not provide Sessions directly; managed by the caller)."""
        raise NotImplementedError("LocalEnv does not provide session directly")

    def submit_job(
        self,
        command: str,
        job_type: str = "debug",
        **kwargs: Any,
    ) -> str:
        """Submit a job (LocalEnv does not support job scheduling directly)."""
        raise NotImplementedError("LocalEnv does not support job submission")

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Query job status (LocalEnv does not support job scheduling directly)."""
        raise NotImplementedError("LocalEnv does not support job status")

    def cancel_job(self, job_id: str) -> None:
        """Cancel a job (LocalEnv does not support job scheduling directly)."""
        raise NotImplementedError("LocalEnv does not support job cancellation")

    def _create_symlinks(self, workspace: Path, symlinks: dict[str, str]) -> None:
        """Create symlinks.

        Args:
            workspace: Workspace path.
            symlinks: Symlink configuration, format: {source_directory_path: target_path_within_workspace}.
        """
        # Get the project root directory, used for resolving relative paths
        # Prefer searching upward from the config file directory for the project root (directory containing evomaster)
        project_root = None
        if hasattr(self.config.session_config, 'config_dir') and self.config.session_config.config_dir:
            config_dir = Path(self.config.session_config.config_dir)
            # Search upward from the config file directory for the project root
            current = config_dir.resolve()
            while current != current.parent:
                if (current / "evomaster").exists() and (current / "evomaster").is_dir():
                    project_root = current
                    break
                current = current.parent
        
        # If project root was not found, try using the current working directory
        if project_root is None:
            current = Path.cwd()
            while current != current.parent:
                if (current / "evomaster").exists() and (current / "evomaster").is_dir():
                    project_root = current
                    break
                current = current.parent
        
        for source_dir, target_rel_path in symlinks.items():
            # Resolve source path: if relative, resolve against the project root; if absolute, use directly
            source_path = Path(source_dir)
            if not source_path.is_absolute():
                # Relative path: resolve against project root if found, otherwise against current working directory
                if project_root is not None:
                    source_path = (project_root / source_dir).resolve()
                    self.logger.debug(f"Relative path '{source_dir}' resolved to: {source_path} (relative to project root {project_root})")
                else:
                    source_path = Path(source_dir).resolve()
                    self.logger.debug(f"Relative path '{source_dir}' resolved to: {source_path} (relative to current working directory)")
            else:
                # Absolute path: use directly
                source_path = source_path.resolve()
                self.logger.debug(f"Absolute path '{source_dir}' resolved to: {source_path}")
            
            if not source_path.exists():
                self.logger.warning(f"Source directory does not exist, skipping symlink: {source_dir} (resolved: {source_path})")
                continue

            if not source_path.is_dir():
                self.logger.warning(f"Source path is not a directory, skipping symlink: {source_dir} (resolved: {source_path})")
                continue

            # Target path is relative to the workspace
            target_path = workspace / target_rel_path

            # If the target path already exists, remove it first (may be a previous symlink or file)
            if target_path.exists() or target_path.is_symlink():
                if target_path.is_symlink():
                    target_path.unlink()
                    self.logger.debug(f"Removed existing symlink: {target_path}")
                else:
                    # If it is a directory, remove it recursively
                    shutil.rmtree(target_path)
                    self.logger.debug(f"Removed existing directory: {target_path}")

            # Ensure the parent directory of the target path exists
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Create the target directory (if it does not exist)
            target_path.mkdir(parents=True, exist_ok=True)

            # Ensure the _generated/ subdirectory exists in the source directory (playground/configs),
            # so that files generated by agent_builder are written directly to the project root via symlink
            if source_path.name in ("playground", "configs"):
                generated_dir = source_path / "_generated"
                if not generated_dir.exists():
                    generated_dir.mkdir(exist_ok=True)

            # Link all contents from the source directory to the target directory
            self._link_directory_contents(source_path, target_path)
            self.logger.info(f"Created symlinks: contents of {source_dir} -> {target_path}")
    
    def _link_directory_contents(self, source_dir: Path, target_dir: Path) -> None:
        """Link all contents from the source directory to the target directory.

        Args:
            source_dir: Source directory path.
            target_dir: Target directory path.
        """
        for item in source_dir.iterdir():
            source_item = source_dir / item.name
            target_item = target_dir / item.name
            
            # If the target already exists, skip
            if target_item.exists() or target_item.is_symlink():
                self.logger.debug(f"Target already exists, skipping: {target_item}")
                continue
            
            try:
                os.symlink(source_item, target_item)
                self.logger.debug(f"Created symlink: {source_item} -> {target_item}")
            except OSError as e:
                self.logger.warning(f"Failed to create symlink: {source_item} -> {target_item}, error: {e}")

    def _terminate_process_group(
        self,
        proc: subprocess.Popen[str],
        grace_seconds: int = 10,
        force: bool = False,
    ) -> None:
        """Terminate a command and all descendants started in its process group."""
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
        except OSError as e:
            self.logger.debug("Failed to terminate process group %s: %s", proc.pid, e)
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
        except OSError as e:
            self.logger.debug("Failed to kill process group %s: %s", proc.pid, e)
            proc.kill()

    def local_exec(
        self,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
        parallel_index: int | None = None,
    ) -> dict[str, Any]:
        """Execute a command locally.

        Args:
            command: Command to execute.
            timeout: Timeout in seconds.
            workdir: Working directory.
            parallel_index: Parallel index (optional, used for resource allocation).

        Returns:
            Result dictionary containing:
            - stdout: Standard output
            - stderr: Standard error
            - exit_code: Exit code
            - output: Combined stdout + stderr
        """
        if not self._is_ready:
            raise RuntimeError("Environment not ready")

        timeout = timeout or self.config.session_config.timeout
        workdir = workdir or self.config.session_config.workspace_path

        # Check if the working directory exists
        workspace = Path(workdir)
        cwd = workdir if workspace.exists() else None

        # If parallel resource allocation is enabled, use the resource allocator
        gpu_allocation = None
        gpu_lock_handles: list[Any] = []
        cpu_allocation = None
        
        session_config = self.config.session_config
        cpu_devices = getattr(session_config, 'cpu_devices', None)
        gpu_devices = getattr(session_config, 'gpu_devices', None)

        if self._resource_allocator is not None and parallel_index is not None:
            # Register the execution task (check parallel limits)
            self._resource_allocator.register_execution(parallel_index)
            try:
                # Allocate resources
                gpu_allocation, cpu_allocation = self._resource_allocator.allocate_resources(parallel_index)
                gpu_allocation, gpu_lock_handles = self._resource_allocator.prepare_gpu_allocation(gpu_allocation)
                self.logger.info(
                    f"并行索引 {parallel_index}: GPU={gpu_allocation}, CPU={cpu_allocation}"
                )
                # When the resource allocator returns None (e.g., cpus_per_parallel=0), fall back to the original cpu_devices
                if cpu_allocation is None and cpu_devices is not None:
                    if isinstance(cpu_devices, str):
                        cpu_allocation = cpu_devices
                    elif isinstance(cpu_devices, list):
                        cpu_allocation = ",".join(str(cpu) for cpu in cpu_devices)
            finally:
                # Note: cannot unregister immediately because the command is still executing
                # We will unregister after the command completes
                pass
        else:
            # Parallel resource allocation not enabled, or parallel_index not set; use original config
            if gpu_devices is not None:
                if isinstance(gpu_devices, str):
                    gpu_allocation = gpu_devices
                elif isinstance(gpu_devices, list):
                    gpu_allocation = ",".join(str(gpu) for gpu in gpu_devices)
            
            if cpu_devices is not None:
                if isinstance(cpu_devices, str):
                    cpu_allocation = cpu_devices
                elif isinstance(cpu_devices, list):
                    cpu_allocation = ",".join(str(cpu) for cpu in cpu_devices)

        # Build environment variables
        env = os.environ.copy()
        
        # Set GPU devices
        if gpu_allocation is not None:
            env['CUDA_VISIBLE_DEVICES'] = gpu_allocation
            parallel_config = getattr(session_config, 'parallel', None) or {}
            world_size = len([gpu for gpu in gpu_allocation.split(",") if gpu.strip()])
            if parallel_config.get('set_asr_world_size', False) and world_size > 0:
                env['ASR_WORLD_SIZE'] = str(world_size)
            self.logger.debug(f"Setting CUDA_VISIBLE_DEVICES={gpu_allocation}")

        # Build CPU affinity command prefix
        # Note: taskset cannot directly execute shell built-in commands (e.g., cd); they need to be wrapped in sh -c
        if cpu_allocation is not None and sys.platform != "win32":
            # Use shlex.quote to safely escape the command, then wrap in sh -c
            final_command = f"taskset -c {cpu_allocation} sh -c {shlex.quote(command)}"
            self.logger.info(f"Applying CPU affinity restriction: taskset -c {cpu_allocation}")
        else:
            final_command = command

        try:
            proc = subprocess.Popen(
                final_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=cwd,
                env=env,
                start_new_session=(sys.platform != "win32"),
            )
            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []

            def read_stream(stream: Any, chunks: list[str]) -> None:
                if stream is None:
                    return
                try:
                    for line in stream:
                        chunks.append(line)
                except ValueError:
                    return

            stdout_thread = threading.Thread(
                target=read_stream,
                args=(proc.stdout, stdout_chunks),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=read_stream,
                args=(proc.stderr, stderr_chunks),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()

            timed_out = False
            try:
                exit_code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                exit_code = -1
                self._terminate_process_group(proc, force=True)
            else:
                if exit_code != 0:
                    self._terminate_process_group(proc, force=True)

            for reader in (stdout_thread, stderr_thread):
                reader.join(timeout=1)
            if stdout_thread.is_alive() or stderr_thread.is_alive():
                self._terminate_process_group(proc, force=True)
                for reader in (stdout_thread, stderr_thread):
                    reader.join(timeout=5)
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()

            stdout = "".join(stdout_chunks)
            stderr = "".join(stderr_chunks)
            if timed_out:
                timeout_message = f"Command timed out after {timeout}s"
                if stderr:
                    stderr += "\n"
                stderr += timeout_message
                return {
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": -1,
                    "output": stdout + stderr,
                }
            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
                "output": stdout + stderr,
            }
        except subprocess.TimeoutExpired:
            self._terminate_process_group(proc, force=True)
            stdout = ""
            stderr = ""
            timeout_message = f"Command timed out after {timeout}s"
            stderr += timeout_message
            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": -1,
                "output": stdout + stderr,
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1,
                "output": str(e),
            }
        finally:
            # Unregister the execution task
            if self._resource_allocator is not None and parallel_index is not None:
                self._resource_allocator.release_gpu_locks(gpu_lock_handles)
                self._resource_allocator.unregister_execution(parallel_index)

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a file to the local environment.

        Args:
            local_path: Local file path.
            remote_path: Remote file path (path in the local environment).
        """
        if not self._is_ready:
            raise RuntimeError("Environment not ready")

        local_file = Path(local_path)
        remote_file = Path(remote_path)

        if not local_file.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        # Create the remote directory
        remote_file.parent.mkdir(parents=True, exist_ok=True)

        if local_file.is_file():
            shutil.copy2(local_file, remote_file)
            self.logger.debug(f"Uploaded file {local_path} to {remote_path}")
        elif local_file.is_dir():
            shutil.copytree(local_file, remote_file, dirs_exist_ok=True)
            self.logger.debug(f"Uploaded directory {local_path} to {remote_path}")

    def download_file(self, remote_path: str, timeout: int | None = None) -> bytes:
        """Download a file from the local environment.

        Args:
            remote_path: Remote file path (path in the local environment).
            timeout: Timeout (not used for local).

        Returns:
            File content (bytes).
        """
        if not self._is_ready:
            raise RuntimeError("Environment not ready")

        remote_file = Path(remote_path)

        if not remote_file.exists():
            raise FileNotFoundError(f"Remote file not found: {remote_path}")

        if not remote_file.is_file():
            raise IsADirectoryError(f"Remote path is not a file: {remote_path}")

        with open(remote_file, "rb") as f:
            return f.read()

    def read_file_content(self, remote_path: str, encoding: str = "utf-8") -> str:
        """Read remote file content (text).

        Args:
            remote_path: Remote file path (path in the local environment).
            encoding: File encoding.

        Returns:
            File content (string).
        """
        if not self._is_ready:
            raise RuntimeError("Environment not ready")

        remote_file = Path(remote_path)

        if not remote_file.exists():
            raise FileNotFoundError(f"Remote file not found: {remote_path}")

        if not remote_file.is_file():
            raise IsADirectoryError(f"Remote path is not a file: {remote_path}")

        with open(remote_file, "r", encoding=encoding) as f:
            return f.read()

    def write_file_content(self, remote_path: str, content: str, encoding: str = "utf-8") -> None:
        """Write content to a remote file.

        Args:
            remote_path: Remote file path (path in the local environment).
            content: File content.
            encoding: File encoding.
        """
        if not self._is_ready:
            raise RuntimeError("Environment not ready")

        remote_file = Path(remote_path)

        # Ensure the directory exists
        remote_file.parent.mkdir(parents=True, exist_ok=True)

        # Write the file
        with open(remote_file, "w", encoding=encoding) as f:
            f.write(content)

    def path_exists(self, remote_path: str) -> bool:
        """Check if a remote path exists.

        Args:
            remote_path: Remote path (path in the local environment).

        Returns:
            Whether the path exists.
        """
        if not self._is_ready:
            raise RuntimeError("Environment not ready")

        return os.path.exists(remote_path)

    def is_file(self, remote_path: str) -> bool:
        """Check if a remote path is a file.

        Args:
            remote_path: Remote path (path in the local environment).

        Returns:
            Whether the path is a file.
        """
        if not self._is_ready:
            raise RuntimeError("Environment not ready")

        return os.path.isfile(remote_path)

    def is_directory(self, remote_path: str) -> bool:
        """Check if a remote path is a directory.

        Args:
            remote_path: Remote path (path in the local environment).

        Returns:
            Whether the path is a directory.
        """
        if not self._is_ready:
            raise RuntimeError("Environment not ready")

        return os.path.isdir(remote_path)
