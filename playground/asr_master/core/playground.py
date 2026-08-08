import math
import os
import logging
import sys
import json
from pathlib import Path
import shutil
import threading
import subprocess
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evomaster.core import BasePlayground, register_playground
from evomaster.agent.session import (
    LocalSessionConfig,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evomaster.agent import Agent

from ..agent.session.local import MLMaster2LocalSession
from .exp.draft_exp import DraftExp
from .exp.research_exp import ResearchExp
from .exp.improve_exp import ImproveExp
from .exp.prefetch_exp import PrefetchExp
from .exp.knowledge_promotion_exp import KnowledgePromotionExp
from .exp.wisdom_promotion_exp import WisdomPromotionExp
from .utils.asr_data_preview import generate_asr_preview
from .utils.code import save_code_to_file
from .utils.metric import normalize_wer_score
from .utils.watch_dog import (
    TimeoutWatchdog,
    GlobalTimeoutInterrupt,
    RUN_TIMEOUT_SECONDS,
    _async_raise,
)
from typing import List, Any, Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial


_AUTO_SENTINELS = {"auto", "detect"}
_NO_GPU_SENTINELS = {"", "none", "null", "false", "off", "no", "-1", "void", "nodevfiles"}


def _is_auto_value(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in _AUTO_SENTINELS


def _parse_gpu_devices(value: str | list[str] | None) -> list[str]:
    """Parse config/env GPU device specs into a flat list of IDs."""
    if value is None:
        return []

    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.extend(str(item).split(","))
    else:
        raw = str(value).strip()
        if raw.lower() in _NO_GPU_SENTINELS or raw.lower() in _AUTO_SENTINELS:
            return []
        parts = raw.split(",")

    devices = []
    for part in parts:
        device = part.strip()
        if device and device.lower() not in _NO_GPU_SENTINELS:
            devices.append(device)
    return devices


def _discover_gpu_devices(logger: logging.Logger) -> list[str]:
    """Return visible GPU IDs, preferring CUDA_VISIBLE_DEVICES when set."""
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
            devices = [
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip()
            ]
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


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _apply_auto_gpu_config(
    local_config: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, Any]:
    """Resolve ASR auto GPU settings before LocalSessionConfig validation.

    Supported knobs:
    - ``session.local.auto_gpu_config: true`` resolves both GPU list and
      ``parallel.max_parallel`` when they are unset/auto.
    - ``session.local.gpu_devices: auto`` discovers visible GPUs.
    - ``session.local.parallel.max_parallel: auto`` derives concurrency from
      ``len(gpu_devices) // gpus_per_exp``.
    """
    parallel_config = local_config.setdefault("parallel", {})
    if not isinstance(parallel_config, dict):
        parallel_config = {}
        local_config["parallel"] = parallel_config

    auto_gpu_config = bool(
        local_config.get("auto_gpu_config", False)
        or parallel_config.get("auto_gpu_config", False)
    )
    raw_gpu_devices = local_config.get("gpu_devices")
    raw_max_parallel = parallel_config.get("max_parallel")
    should_auto_gpus = auto_gpu_config or _is_auto_value(raw_gpu_devices)
    should_auto_parallel = auto_gpu_config or _is_auto_value(raw_max_parallel)

    if not should_auto_gpus and not should_auto_parallel:
        return local_config

    gpus_per_exp = _positive_int(parallel_config.get("gpus_per_exp", 1), 1)

    configured_gpus = _parse_gpu_devices(raw_gpu_devices)
    if should_auto_gpus and (
        not configured_gpus
        or _is_auto_value(raw_gpu_devices)
        or str(raw_gpu_devices).strip().lower() == "all"
    ):
        gpu_devices = _discover_gpu_devices(logger)
        local_config["gpu_devices"] = gpu_devices or None
    else:
        gpu_devices = configured_gpus
        if configured_gpus:
            local_config["gpu_devices"] = configured_gpus

    if should_auto_parallel:
        max_parallel = max(1, len(gpu_devices) // gpus_per_exp) if gpu_devices else 1
        parallel_config["enabled"] = True
        parallel_config["max_parallel"] = max_parallel

    logger.info(
        "Resolved ASR GPU config: gpu_devices=%s, gpus_per_exp=%s, max_parallel=%s",
        local_config.get("gpu_devices"),
        gpus_per_exp,
        parallel_config.get("max_parallel"),
    )
    return local_config

@register_playground("asr_master")
class ASRMasterPlayground(BasePlayground):
    """Main orchestrator for the ASR Master automated speech recognition research system.

    Implements an iterative improvement workflow for ASR model optimization:
    prefetch -> draft -> (research -> sequential improve)* -> knowledge/wisdom promotion.

    Adapted from ML Master 2 for ASR tasks (e.g., icefall Zipformer optimization).
    Key differences from ml_master_2:
    - WER is the evaluation metric (lower is better)
    - No submission CSV; agent produces wrapper scripts that run train+decode
    - Sequential improvement execution (GPU-heavy training doesn't benefit from parallelism)
    - No grading server (WER is extracted from terminal output)
    - Configurable number of research rounds for long-running training tasks
    """

    def __init__(self, config_dir: Path = None, config_path: Path = None):
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "asr_master"
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents.declare("draft_agent", "debug_agent", "improve_agent", "reseach_agent", "knowledge_promotion_agent", "metric_agent", "prefetch_agent", "wisdom_promotion_agent")

        self.initial_code = None
        self.best_score = None
        self.best_solution = None
        self.real_time_best_solution = None
        self.research_plan_and_result = []
        self.prefetch_descriptor = None

        # ASR-specific: WER is always lower-is-better
        self.is_lower_better = True
        self.asr_task_id = self.config_manager.get("asr_task_id", "librispeech_zipformer")

        # Configurable research rounds (default 3 for long-running ASR training)
        self.max_research_rounds = self.config_manager.get("max_research_rounds", 3)

        # Icefall paths
        self.icefall_root = self.config_manager.get("icefall_root", "")
        self.recipe_dir = self.config_manager.get("recipe_dir", "egs/librispeech/ASR/zipformer")
        self.data_dir = self.config_manager.get("data_dir", "")

        self.mcp_manager = None
        self.exp_index = 0

    def setup(self) -> None:
        """Initialize the playground: session, agents, and workspace directories."""
        self.logger.info("Setting up ASR Master playground...")

        self._setup_session()
        self._setup_agents()
        self._setup_workspace()

        self.logger.info("ASR Master playground setup complete")

    def _setup_session(self) -> None:
        """Create and open a session using MLMaster2LocalSession."""
        if self.session is None:
            session_type = self.config.session.get("type", "local")
            if session_type == "docker":
                raise ValueError("Docker session is not supported for ASR Master")
            else:
                session_config_dict = self.config.session.get("local", {}).copy()
                if "working_dir" in session_config_dict and "workspace_path" not in session_config_dict:
                    session_config_dict["workspace_path"] = session_config_dict["working_dir"]
                elif "workspace_path" in session_config_dict and "working_dir" not in session_config_dict:
                    session_config_dict["working_dir"] = session_config_dict["workspace_path"]
                if "config_dir" not in session_config_dict:
                    session_config_dict["config_dir"] = str(self.config_dir)
                session_config_dict = _apply_auto_gpu_config(
                    session_config_dict,
                    self.logger,
                )
                self.config.session["local"] = session_config_dict.copy()
                session_config_model_dict = session_config_dict.copy()
                session_config_model_dict.pop("auto_gpu_config", None)
                session_config = LocalSessionConfig(**session_config_model_dict)
                self.session = MLMaster2LocalSession(session_config)
                self.logger.info("Using ASR Master Local session")

        if not self.session.is_open:
            self.session.open()
        else:
            self.logger.debug("Session already open, reusing existing session")

    def _setup_workspace(self) -> None:
        """Create required workspace subdirectories (best_solution, working)."""
        os.makedirs(os.path.join(self.session.config.workspace_path, "best_solution"), exist_ok=True)
        os.makedirs(os.path.join(self.session.config.workspace_path, "working"), exist_ok=True)
        self.logger.info(f"working_dir: {self.session.config.workspace_path}")

    def _is_valid_score(self, score) -> bool:
        """Check whether a score is valid. NaN and None are treated as invalid."""
        if score is None:
            return False
        if isinstance(score, float) and math.isnan(score):
            return False
        return True

    def compare_score(self, old_score, new_score) -> bool:
        """Determine whether new_score is an improvement over old_score.

        For ASR, lower WER is always better.

        Args:
            old_score: The previous best WER (may be None or NaN).
            new_score: The candidate WER to compare (may be None or NaN).

        Returns:
            True if new_score is a valid improvement (lower WER) over old_score.
        """
        # Invalid new_score (None/NaN) is never an improvement
        if not self._is_valid_score(new_score):
            return False
        # Invalid old_score (None/NaN) means any valid new_score is an improvement
        if not self._is_valid_score(old_score):
            return True
        # Both valid: lower WER is better
        old_score = normalize_wer_score(old_score)
        new_score = normalize_wer_score(new_score)
        if new_score is None:
            return False
        if old_score is None:
            return True
        return new_score < old_score

    def _create_improve_exp(self, exp_index: int) -> ImproveExp:
        """Create an independent ImproveExp instance for sequential execution.

        Args:
            exp_index: Experiment index used to generate unique exp_name and agent names.

        Returns:
            A new ImproveExp instance with independent agent copies.
        """
        improve_agent_copy = self.copy_agent(
            self.agents.improve_agent, new_agent_name=f"improve_exp_{exp_index}"
        ) if self.agents.improve_agent else None
        debug_agent_copy = self.copy_agent(
            self.agents.debug_agent, new_agent_name=f"debug_exp_{exp_index}"
        ) if self.agents.debug_agent else None
        metric_agent_copy = self.copy_agent(
            self.agents.metric_agent, new_agent_name=f"metric_exp_{exp_index}"
        ) if self.agents.metric_agent else None
        exp_name = f"exp_{exp_index}_improve"
        return ImproveExp(
            improve_agent_copy, debug_agent_copy, metric_agent_copy,
            self.config, exp_name
        )

    def run(self, task_description: str, output_file: str | None = None) -> dict:
        """Execute the full ASR Master pipeline.

        Runs prefetch -> draft -> iterative (research -> sequential improve) cycles
        with a global timeout watchdog. On timeout, triggers wisdom promotion.

        Args:
            task_description: Natural language description of the ASR optimization task.
            output_file: Optional path to save the trajectory output.

        Returns:
            A dict with 'status' ('completed' or 'failed') and optional 'error'.
        """
        # Start the watchdog daemon thread
        watchdog = TimeoutWatchdog(RUN_TIMEOUT_SECONDS)
        watchdog.start()
        self.logger.info(f"Watchdog started ({RUN_TIMEOUT_SECONDS} seconds)")
        try:
            self.setup()

            self._setup_trajectory_file(output_file)

            # Prefetch knowledge from wisdom database (optional for ASR)
            data_knowledge = ""
            model_knowledge = ""
            prefetch_exp = PrefetchExp(self.agents.prefetch_agent, self.config, f"exp_{self.exp_index}_prefetch")
            self.exp_index += 1
            try:
                embedding_config = getattr(self.config, "embedding", {})
                embedding_model = embedding_config.get("openai", {}).get("model", "text-embedding-3-large")
                wisdom_dir = os.path.join(os.getcwd(), "playground/asr_master/example_wisdom")
                if os.path.exists(wisdom_dir):
                    data_knowledge, model_knowledge, self.prefetch_descriptor = prefetch_exp.run(
                        task_description,
                        vec_dir=wisdom_dir,
                        nodes_data=os.path.join(wisdom_dir, "db.json"),
                        model=embedding_model
                    )
                else:
                    self.logger.warning("No ASR wisdom database found, skipping prefetch")
                    self.prefetch_descriptor = None
            except Exception as e:
                self.logger.warning(f"Prefetch failed (non-fatal): {e}")
                self.prefetch_descriptor = None

            # Generate ASR-specific data preview
            data_preview = generate_asr_preview(
                icefall_root=self.icefall_root,
                recipe_dir=self.recipe_dir,
                data_dir=self.data_dir,
            )
            self.logger.info(f"Data preview: {data_preview}")
            self.logger.info("Running experiment...")

            # Draft phase
            draft_exp = DraftExp(self.agents.draft_agent, self.agents.debug_agent, self.agents.metric_agent, self.config, f"exp_{self.exp_index}_draft")
            draft_workspace_name = f"exp_{self.exp_index}_draft"
            self.exp_index += 1
            draft_result = self.execute_parallel_tasks(
                [partial(draft_exp.run, task_description=task_description, data_preview=data_preview, data_knowledge=data_knowledge, model_knowledge=model_knowledge)],
                max_workers=1,
                workspace_names=[draft_workspace_name]
            )
            if isinstance(draft_result[0], Exception):
                raise draft_result[0]
            is_sucess, validation_score, uid, self.best_solution = draft_result[0]
            self.initial_code = self.best_solution
            if not is_sucess:
                self.logger.error("Draft phase failed - cannot proceed to research without a working baseline")
                return {
                    "status": "failed",
                    "steps": 0,
                    "best_wer": None,
                    "error": "Draft phase failed to produce a working solution"
                }
            if is_sucess:
                self.best_score = validation_score
                save_code_to_file(os.path.join(self.session.config.workspace_path, "best_solution"), "best_solution.py", self.best_solution)
                self.real_time_best_solution = self.best_solution

            # Iterative research + improve cycles
            for reseach_round in range(self.max_research_rounds):
                research_round_idea_results: dict[str, dict[tuple, dict]] = {}
                base_solution = self.best_solution

                research_exp = ResearchExp(self.agents.reseach_agent, self.config, self.initial_code, f"exp_{self.exp_index}_research")
                research_workspace_name = f"exp_{self.exp_index}_research"
                self.exp_index += 1
                research_results = self.execute_parallel_tasks(
                    [partial(research_exp.run, task_description=task_description, data_preview=data_preview, best_solution=self.best_solution, research_plan_and_result=self.research_plan_and_result)],
                    max_workers=1,
                    workspace_names=[research_workspace_name]
                )
                research_result = research_results[0]
                if isinstance(research_result, Exception):
                    self.logger.error(f"Research failed: {research_result}")
                    raise research_result
                research_plan = research_result

                session_config = self.config.session.get("local", {})
                parallel_config = session_config.get("parallel", {})
                idea_max_workers = _positive_int(parallel_config.get("max_parallel", 1), 1)

                for direction in research_plan:
                    direction_best_solution = self.best_solution
                    direction_best_score = self.best_score
                    direction_baseline_score = self.best_score
                    direction_best_idea = None
                    research_round_idea_results[direction] = {}

                    ideas = list(research_plan[direction].items())
                    if not ideas:
                        continue

                    if idea_max_workers <= 1:
                        for i, idea in enumerate(ideas):
                            exp_index = self.exp_index + i
                            improve_exp = self._create_improve_exp(exp_index)
                            task = partial(
                                improve_exp.run,
                                task_description=task_description,
                                data_preview=data_preview,
                                best_solution=direction_best_solution,
                                idea=idea,
                            )

                            improve_results = self.execute_parallel_tasks(
                                [task], max_workers=1, workspace_names=[improve_exp.exp_name]
                            )
                            result = improve_results[0]

                            if isinstance(result, Exception):
                                self.logger.error(f"Idea {idea} failed: {result}")
                                validation_score = None
                                is_sucess = False
                                uid = None
                                solution = None
                            else:
                                is_sucess, validation_score, uid, solution = result

                            improved = self.compare_score(direction_baseline_score, validation_score)
                            research_round_idea_results[direction][idea] = {
                                "improved": improved,
                                "is_best_in_direction": False,
                                "score": validation_score,
                            }
                            if improved and is_sucess and solution is not None:
                                direction_best_score = validation_score
                                direction_best_solution = solution
                                direction_best_idea = idea
                                save_code_to_file(
                                    os.path.join(self.session.config.workspace_path, "best_solution"),
                                    "best_solution.py",
                                    direction_best_solution,
                                )
                                self.real_time_best_solution = direction_best_solution

                        self.exp_index += len(ideas)
                        if direction_best_idea is not None:
                            research_round_idea_results[direction][direction_best_idea]["is_best_in_direction"] = True

                        self.best_solution = direction_best_solution
                        self.best_score = direction_best_score
                        continue

                    tasks = []
                    workspace_names = []
                    improve_exp_list = []
                    for i, idea in enumerate(ideas):
                        exp_index = self.exp_index + i
                        improve_exp = self._create_improve_exp(exp_index)
                        improve_exp_list.append(improve_exp)
                        task = partial(
                            improve_exp.run,
                            task_description=task_description,
                            data_preview=data_preview,
                            best_solution=direction_best_solution,
                            idea=idea,
                        )
                        tasks.append(task)
                        workspace_names.append(improve_exp.exp_name)

                    improve_results = self.execute_parallel_tasks(
                        tasks, max_workers=idea_max_workers, workspace_names=workspace_names
                    )

                    for idea, improve_exp, result in zip(ideas, improve_exp_list, improve_results):
                        if isinstance(result, Exception):
                            self.logger.error(f"Idea {idea} failed: {result}")
                            validation_score = None
                            is_sucess = False
                            uid = None
                            solution = None
                        else:
                            is_sucess, validation_score, uid, solution = result

                        improved = self.compare_score(direction_baseline_score, validation_score)
                        research_round_idea_results[direction][idea] = {
                            "improved": improved,
                            "is_best_in_direction": False,
                            "score": validation_score,
                        }
                        if improved and is_sucess and solution is not None:
                            direction_best_score = validation_score
                            direction_best_solution = solution
                            direction_best_idea = idea
                            save_code_to_file(
                                os.path.join(self.session.config.workspace_path, "best_solution"),
                                "best_solution.py",
                                direction_best_solution,
                            )
                            self.real_time_best_solution = direction_best_solution

                    self.exp_index += len(ideas)

                    # Mark the best idea in this direction
                    if direction_best_idea is not None:
                        research_round_idea_results[direction][direction_best_idea]["is_best_in_direction"] = True

                    self.best_solution = direction_best_solution
                    self.best_score = direction_best_score

                # Store research plan and results
                plan_text = json.dumps(research_plan, ensure_ascii=False, indent=2)
                self.research_plan_and_result.extend([plan_text])

                self.logger.info(f"Round {reseach_round} results: {research_round_idea_results}")

                # Knowledge promotion after each round
                knowledge_promotion_exp = KnowledgePromotionExp(self.agents.knowledge_promotion_agent, self.config, f"exp_{self.exp_index}_knowledge_promotion")
                knowledge_promotion_workspace_name = f"exp_{self.exp_index}_knowledge_promotion"
                self.exp_index += 1
                knowledge_promotion_results = self.execute_parallel_tasks(
                    [partial(knowledge_promotion_exp.run, task_description=task_description, data_preview=data_preview, base_solution=base_solution, best_solution=self.best_solution, research_plan=research_plan, research_round_idea_results=research_round_idea_results)],
                    max_workers=1,
                    workspace_names=[knowledge_promotion_workspace_name]
                )
                knowledge_promotion_result = knowledge_promotion_results[0]
                if isinstance(knowledge_promotion_result, Exception):
                    self.logger.error(f"Knowledge promotion failed: {knowledge_promotion_result}")
                    raise knowledge_promotion_result
                self.research_plan_and_result.extend([knowledge_promotion_result])

            result = {
                "status": "completed",
                "steps": 0,
                "best_wer": self.best_score,
                "wisdom_promotion_completed": False,
            }
            return result
        except GlobalTimeoutInterrupt:
            self.logger.warning(f"Watchdog triggered: experiment has run for {RUN_TIMEOUT_SECONDS} seconds, forced interruption, starting wisdom promotion")
            wisdom_promotion_completed = False
            wisdom_promotion_result = None
            wisdom_promotion_exp = WisdomPromotionExp(self.agents.wisdom_promotion_agent, self.config, f"exp_{self.exp_index}_wisdom_promotion")
            wisdom_promotion_workspace_name = f"exp_{self.exp_index}_wisdom_promotion"
            self.exp_index += 1
            wisdom_promotion_results = self.execute_parallel_tasks(
                [partial(wisdom_promotion_exp.run, task_description=task_description, best_solution=self.real_time_best_solution)],
                max_workers=1,
                workspace_names=[wisdom_promotion_workspace_name]
            )
            wisdom_promotion_completed = not any(isinstance(item, Exception) for item in wisdom_promotion_results)
            wisdom_promotion_result = wisdom_promotion_results
            self.logger.info(f"Wisdom promotion finished")
            self.logger.info(f"Task descriptor: {self.prefetch_descriptor}")
            self.logger.info(f"Wisdom promotion result: {wisdom_promotion_results}")
            result = {
                "status": "timeout",
                "steps": 0,
                "best_wer": self.best_score,
                "timeout_seconds": RUN_TIMEOUT_SECONDS,
                "wisdom_promotion_completed": wisdom_promotion_completed,
                "wisdom_promotion_result": wisdom_promotion_result,
            }
            return result
        except Exception as e:
            self.logger.error(f"ASR Master task execution failed: {e}", exc_info=True)
            result = {
                "status": "failed",
                "steps": 0,
                "error": str(e),
            }
            return result

        finally:
            if 'watchdog' in locals():
                watchdog.stop()
            self.cleanup()

    def execute_parallel_tasks(self, tasks: List[Callable], max_workers: int = 1, workspace_names: List[str] | None = None) -> List[Any]:
        """Generic task executor with resource allocation and workspace isolation.

        For ASR tasks, max_workers is typically 1 (sequential execution) since
        GPU training doesn't benefit from parallelism.

        Args:
            tasks: List of callable tasks to execute.
            max_workers: Maximum number of concurrent workers (default 1 for ASR).
            workspace_names: Optional list of workspace names for each task.

        Returns:
            List of results in the same order as tasks.
        """
        self.logger.info(f"Starting execution of {len(tasks)} tasks with {max_workers} workers.")

        results = [None] * len(tasks)

        session_config = self.config.session.get("local", {})
        parallel_config = session_config.get("parallel", {})
        parallel_enabled = parallel_config.get("enabled", False)
        split_workspace = parallel_config.get("split_workspace_for_exp", False)

        active_worker_tids = set()
        tids_lock = threading.Lock()

        def wrap_task(task_func, parallel_index):
            def wrapped():
                current_tid = threading.get_ident()
                with tids_lock:
                    active_worker_tids.add(current_tid)

                try:
                    if parallel_enabled and self.session is not None:
                        from evomaster.agent.session.local import LocalSession
                        if isinstance(self.session, LocalSession):
                            self.session.set_parallel_index(parallel_index)
                            self.logger.debug(f"Set parallel index: {parallel_index}")

                            if split_workspace:
                                import os
                                main_workspace = self.session.config.workspace_path
                                exp_name = workspace_names[parallel_index] if workspace_names and parallel_index < len(workspace_names) else f"exp_{parallel_index}"
                                exp_workspace = os.path.join(main_workspace, exp_name)
                                self.session._env.setup_exp_workspace(exp_workspace)
                                os.makedirs(os.path.join(exp_workspace, "working"), exist_ok=True)
                                self.session.set_workspace_path(exp_workspace)
                                self.logger.info(f"Exp {parallel_index} using independent workspace: {exp_workspace}")

                    return task_func()

                except GlobalTimeoutInterrupt:
                    self.logger.warning(f"Task {parallel_index} (TID: {current_tid}) received interrupt signal, exiting...")
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
        wrapped_tasks = [wrap_task(task, i) for i, task in enumerate(tasks)]
        future_to_index = {executor.submit(wrapped_task): i for i, wrapped_task in enumerate(wrapped_tasks)}

        try:
            from concurrent.futures import wait, FIRST_COMPLETED
            not_done = set(future_to_index.keys())

            while not_done:
                done, not_done = wait(
                    not_done,
                    timeout=0.5,
                    return_when=FIRST_COMPLETED
                )

                for future in done:
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        results[index] = result
                    except Exception as exc:
                        self.logger.error(f"Task {index} generated an exception: {exc}")
                        results[index] = exc

            self.logger.info("Execution completed.")
            return results

        finally:
            for future in future_to_index:
                future.cancel()

            with tids_lock:
                for tid in active_worker_tids:
                    try:
                        _async_raise(tid, GlobalTimeoutInterrupt)
                    except Exception as e:
                        self.logger.error(f"Unable to send interrupt signal to child thread {tid}: {e}")

            if sys.version_info >= (3, 9):
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=False)
