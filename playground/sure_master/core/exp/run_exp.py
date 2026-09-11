from __future__ import annotations

import filecmp
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from evomaster.agent import BaseAgent
from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function

from ..utils.candidate_changes import (
    clear_candidate_outputs,
    validate_arch_candidate_changes,
    validate_candidate_changes,
    write_candidate_status,
)
from ..utils.candidate_type import (
    ARCH,
    FINE_TUNE,
    INFERENCE,
    candidate_type_from_code,
    normalize_candidate_type,
)
from ..utils.code import (
    read_code,
    save_code_to_file,
    validate_run_sure_script,
    validate_sure_candidate_boundary,
)
from ..utils.metric import SureMetricResult, SureMetricRunner, format_metric_feedback
from ...tasks import get_adapter
from ..search_scope import restricted_search, GUIDANCE, validate_search_entrypoint, execution_contract as scoped_execution_contract
from ..utils.task_cards import BaseModelProfile, SureTaskCard
from ..utils.vc_remote import (
    VcRemoteTrainingExecutor,
    candidate_runs_remotely,
    sure_config_from,
)
from ..utils.workspace_cleanup import (
    cleanup_candidate_workspace,
    workspace_cleanup_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
ASR_ARCH_CODE_MARKERS = (
    "--num-encoder-layers",
    "--encoder-dim",
    "--feedforward-dim",
    "--encoder-unmasked-dim",
)
ASR_TRAINING_CANDIDATE_TYPES = {ARCH, FINE_TUNE}
ASR_FATAL_TRAIN_PATTERNS = (
    "npu out of memory",
    "acl error: 207001",
    "cuda out of memory",
    "torch.cuda.outofmemoryerror",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
)


class SureRunExp(BaseExp):
    """Generate, execute, and score one SURE candidate script."""

    def __init__(
        self,
        stage: str,
        main_agent: BaseAgent,
        debug_agent: BaseAgent,
        config: Any,
        exp_name: str,
        task_card: SureTaskCard,
        base_model_profile: BaseModelProfile | None = None,
        metric_runner: SureMetricRunner | None = None,
        execution_env: dict[str, str] | None = None,
        config_path: str | Path | None = None,
    ):
        super().__init__(main_agent, config)
        self.stage = stage
        self.main_agent = main_agent
        self.debug_agent = debug_agent
        self._exp_name = exp_name
        self.task_card = task_card
        self.base_model_profile = base_model_profile
        self.metric_runner = metric_runner
        self.execution_env = dict(execution_env or {})
        self.config_path = config_path
        self.candidate_type_hint = INFERENCE
        self.uid = uuid.uuid4()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.workspace_path = os.path.join(
            main_agent.session.config.workspace_path,
            exp_name,
        )
        self.terminal_output = ""
        self.metric_feedback = ""
        self.code = ""
        self.base_model_source_overrides: dict[str, str] = {}
        self._current_role_paths: dict[str, str | None] = {}
        self.enforce_candidate_type = False
        self.candidate_stage_name = ""
        self.candidate_phase = ""
        self.candidate_rung_name = ""
        self.candidate_idea_id = ""
        self._candidate_started_at = 0.0
        self._current_response_is_initial_solution = False

    @property
    def exp_name(self) -> str:
        return self._exp_name

    def run(
        self,
        task_description: str,
        data_preview: str,
        data_knowledge: str = "",
        model_knowledge: str = "",
        previous_solution: str = "",
        improve_idea: Any = "",
        role_paths: dict[str, str | None] | None = None,
        task_id: str = "exp_001",
        candidate_type_hint: str = INFERENCE,
        base_model_source_overrides: dict[str, str] | None = None,
    ) -> tuple[bool, float | None, uuid.UUID, str, dict[str, Any]]:
        """Generate/repair run_sure.py, execute it, and score generated artifacts."""
        self.logger.info("Starting SURE %s experiment: %s", self.stage, self.exp_name)
        self.candidate_type_hint = normalize_candidate_type(
            candidate_type_hint,
            default=INFERENCE,
        )
        self.base_model_source_overrides = dict(base_model_source_overrides or {})
        role_paths = dict(role_paths or self.task_card.artifact_contract)
        self.workspace_path = self._resolve_workspace_path()
        self._ensure_workspace_dirs()
        self._prepare_base_model_source_overrides()
        self._prepare_workspace_inputs(role_paths)

        response = self._initial_solution_response()
        used_initial_solution = response is not None
        saved_response = Path(self.workspace_path) / "metric/generated_response.json"
        if response is None and sure_config_from(self.config).get("execution_mode") == "slurm" and saved_response.exists():
            response = json.loads(saved_response.read_text())["response"]
        if response is None:
            response = self._run_main_agent(
                task_description,
                data_preview,
                data_knowledge,
                model_knowledge,
                previous_solution,
                improve_idea,
                role_paths,
                task_id,
            )

        if sure_config_from(self.config).get("execution_mode") == "slurm":
            from ..utils.slurm import atomic_json
            saved_response = Path(self.workspace_path) / "metric/generated_response.json"
            if saved_response.exists():
                response = json.loads(saved_response.read_text())["response"]
            else:
                atomic_json(saved_response, {"response": response})
        self._current_response_is_initial_solution = used_initial_solution
        try:
            result = self._execute_and_score(response, role_paths)
        finally:
            self._current_response_is_initial_solution = False
        if result[0]:
            return result[0], result[1], self.uid, self.code, result[2]
        if result[2].get("reason_code") == "worker_failed":
            return result[0], result[1], self.uid, self.code, result[2]
        if used_initial_solution and not self._debug_initial_solution_enabled():
            return result[0], result[1], self.uid, self.code, result[2]

        for _ in range(3):
            response = self._run_debug_agent(
                task_description,
                data_preview,
                role_paths,
                task_id,
            )
            if sure_config_from(self.config).get("execution_mode") == "slurm":
                from ..utils.slurm import atomic_json
                atomic_json(saved_response, {"response": response})
            result = self._execute_and_score(response, role_paths)
            if result[0]:
                break
        return result[0], result[1], self.uid, self.code, result[2]

    def run_existing_code(
        self,
        *,
        code: str,
        role_paths: dict[str, str | None] | None = None,
        candidate_type_hint: str = INFERENCE,
        base_model_source_overrides: dict[str, str] | None = None,
    ) -> tuple[bool, float | None, uuid.UUID, str, dict[str, Any]]:
        """Execute a previously generated candidate under this experiment context.

        Frozen selection/holdout checks use this entry point. It intentionally does not call debug agents:
        a rerun should measure the same candidate under a different budget or
        dataset, not silently mutate it.
        """
        self.logger.info("Rerunning existing SURE candidate: %s", self.exp_name)
        self.candidate_type_hint = normalize_candidate_type(
            candidate_type_hint,
            default=INFERENCE,
        )
        self.base_model_source_overrides = dict(base_model_source_overrides or {})
        role_paths = dict(role_paths or self.task_card.artifact_contract)
        self.workspace_path = self._resolve_workspace_path()
        self._ensure_workspace_dirs()
        self._prepare_base_model_source_overrides()
        self._prepare_workspace_inputs(role_paths)
        response = "```python\n" + code.strip() + "\n```"
        ok, score, details = self._execute_and_score(response, role_paths)
        return ok, score, self.uid, self.code, details

    def _session_workspace_base(self) -> str:
        session = getattr(self.main_agent, "session", None)
        if session is None:
            return ""
        getter = getattr(session, "get_workspace_path", None)
        if callable(getter):
            value = getter()
            if value:
                return str(value)
        config = getattr(session, "config", None)
        return str(getattr(config, "workspace_path", "") or "")

    def _resolve_workspace_path(self) -> str:
        workspace_base = self._session_workspace_base()
        if not workspace_base:
            return self.workspace_path
        base = Path(workspace_base)
        if base.name == self.exp_name:
            return str(base)
        return str(base / self.exp_name)

    def _ensure_workspace_dirs(self) -> None:
        workspace = Path(self.workspace_path)
        for name in ("artifacts", "models", "metric", "working"):
            (workspace / name).mkdir(parents=True, exist_ok=True)

    def _initial_solution_response(self) -> str | None:
        path = self._configured_initial_solution_path()
        if path is None:
            return None
        if not path.is_file():
            raise FileNotFoundError(f"Configured initial solution does not exist: {path}")
        code = path.read_text(encoding="utf-8").strip()
        self.logger.info("Using configured initial SURE solution for draft: %s", path)
        return "```python\n" + code + "\n```"

    def _configured_initial_solution_path(self) -> Path | None:
        if self.stage != "draft":
            return None
        if isinstance(self.config, dict):
            config_dict = self.config
        elif hasattr(self.config, "model_dump"):
            config_dict = self.config.model_dump()
        else:
            config_dict = {}
        sure_config = config_dict.get("sure") or {}
        raw_path = sure_config.get("initial_solution_path") if isinstance(sure_config, dict) else None
        if not raw_path:
            return None
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    def _run_main_agent(
        self,
        task_description: str,
        data_preview: str,
        data_knowledge: str,
        model_knowledge: str,
        previous_solution: str,
        improve_idea: Any,
        role_paths: dict[str, str | None],
        task_id: str,
    ) -> str:
        BaseAgent.set_exp_info(exp_name=self.exp_name, exp_index=1)
        original_kwargs = getattr(self.main_agent, "_prompt_format_kwargs", {}).copy()
        try:
            self.main_agent._prompt_format_kwargs.update(
                {
                    "task_description": task_description,
                    "data_preview": data_preview,
                    "data_knowledge": data_knowledge,
                    "model_knowledge": model_knowledge,
                    "task_card": self.task_card.to_prompt_json(),
                    "base_model_profile": self._base_model_prompt_text(),
                    "role_paths": self._role_paths_prompt_text(role_paths),
                    "execution_env": self._execution_env_prompt_text(),
                    "previous_solution": previous_solution or "No previous solution yet.",
                    "improve_idea": improve_idea or "Create a strong initial solution.",
                    "candidate_type_hint": self.candidate_type_hint,
                }
            )
            task = TaskInstance(
                task_id=f"{task_id}_{self.stage}",
                task_type="sure_run",
                description=task_description,
                input_data={},
            )
            trajectory = self.main_agent.run(task)
            return self._extract_agent_response(trajectory)
        finally:
            self.main_agent._prompt_format_kwargs = original_kwargs

    def _run_debug_agent(
        self,
        task_description: str,
        data_preview: str,
        role_paths: dict[str, str | None],
        task_id: str,
    ) -> str:
        BaseAgent.set_exp_info(exp_name=self.exp_name + "_debug", exp_index=1)
        original_kwargs = getattr(self.debug_agent, "_prompt_format_kwargs", {}).copy()
        try:
            self.debug_agent._prompt_format_kwargs.update(
                {
                    "task_description": task_description,
                    "data_preview": data_preview,
                    "task_card": self.task_card.to_prompt_json(),
                    "base_model_profile": self._base_model_prompt_text(),
                    "role_paths": self._role_paths_prompt_text(role_paths),
                    "execution_env": self._execution_env_prompt_text(),
                    "buggy_code": self.code,
                    "terminal_output": self.terminal_output,
                    "metric_feedback": self.metric_feedback,
                    "candidate_type_hint": self.candidate_type_hint,
                }
            )
            task = TaskInstance(
                task_id=f"{task_id}_{self.stage}_debug",
                task_type="sure_debug",
                description=task_description,
                input_data={},
            )
            trajectory = self.debug_agent.run(task)
            return self._extract_agent_response(trajectory)
        finally:
            self.debug_agent._prompt_format_kwargs = original_kwargs

    def _execute_and_score(
        self,
        response: str,
        role_paths: dict[str, str | None],
    ) -> tuple[bool, float | None, dict[str, Any]]:
        self._current_role_paths = dict(role_paths)
        if sure_config_from(self.config).get("execution_mode") != "slurm":
            clear_candidate_outputs(self.workspace_path, role_paths)
        code_to_run, self.code = read_code(response)
        if not code_to_run:
            self.terminal_output = "No usable Python code was found in the agent response."
            self.metric_feedback = "SURE metric not run because run_sure.py was not generated."
            details = {"error": self.terminal_output}
            self._write_status(False, "no_code", details=details)
            return False, None, details

        if self.metric_runner is None:
            self.terminal_output = "SURE metric runner is not configured."
            self.metric_feedback = self.terminal_output
            details = {"error": self.terminal_output}
            self._write_status(False, "metric_runner_missing", details=details)
            return False, None, details

        boundary_errors = validate_sure_candidate_boundary(
            code_to_run,
            self.metric_runner.sure_root,
            self.base_model_profile,
            canonical_task=self.task_card.canonical_task,
            require_asr_wrapper=self._execution_env_truthy(
                "SURE_ASR_REQUIRE_ZIPFORMER_WRAPPER",
                False,
            ),
        )
        if boundary_errors:
            self.terminal_output = (
                "Candidate code violates SURE candidate constraints:\n"
                + "\n".join(f"- {error}" for error in boundary_errors)
            )
            self.metric_feedback = (
                "SURE metric not run because candidate code violated safety or boundary constraints."
            )
            details = {"boundary_errors": boundary_errors}
            self._write_status(False, "boundary_error", details=details)
            return False, None, details

        expected_candidate_type = normalize_candidate_type(
            self.candidate_type_hint,
            default=INFERENCE,
        )
        candidate_type = self._candidate_type_from_code(code_to_run)
        self.candidate_type_hint = candidate_type
        if self.enforce_candidate_type and candidate_type != expected_candidate_type:
            self.terminal_output = (
                "Candidate code does not match the reviewed candidate type: "
                f"required={expected_candidate_type}, detected={candidate_type}."
            )
            self.metric_feedback = (
                "SURE metric not run because the reviewed candidate type contract was violated."
            )
            details = {
                "candidate_type_error": {
                    "required": expected_candidate_type,
                    "detected": candidate_type,
                }
            }
            self._write_status(False, "candidate_type_mismatch", details=details)
            return False, None, details

        scope_env = {**self.execution_env, "SURE_CANDIDATE_PHASE": self.candidate_phase or self.stage}
        try:
            validate_search_entrypoint(code_to_run, scope_env)
        except ValueError as exc:
            self.metric_feedback = str(exc)
            details = {"error": str(exc)}
            self._write_status(False, "boundary_error", details=details)
            return False, None, details
        if restricted_search(scope_env) and candidate_type != ARCH:
            self.metric_feedback = GUIDANCE
            details = {"error": "Architecture-only search requires arch"}
            self._write_status(False, "candidate_type_mismatch", details=details)
            return False, None, details

        script_errors = validate_run_sure_script(code_to_run, role_paths)
        if script_errors:
            self.terminal_output = (
                "Candidate run_sure.py failed execution-readiness checks:\n"
                + "\n".join(f"- {error}" for error in script_errors)
            )
            self.metric_feedback = (
                "SURE metric not run because run_sure.py was empty, inert, or did not reference "
                "the required output artifact."
            )
            details = {"script_errors": script_errors}
            self._write_status(False, "script_error", details=details)
            return False, None, details

        save_code_to_file(self.workspace_path, "run_sure.py", code_to_run)

        self.execution_env["SURE_CANDIDATE_TYPE_HINT"] = candidate_type
        self.execution_env["SURE_CANDIDATE_PHASE"] = self.candidate_phase or self.stage
        self._candidate_started_at = float(self.execution_env.get("SURE_CANDIDATE_ORIGIN_TIME") or time.time())
        if self._should_run_remote_candidate(candidate_type):
            self._write_remote_candidate_context()
            return self._execute_and_score_remote_candidate()

        command = self._execution_command()
        tool_call_obj = ChatCompletionMessageToolCall(
            id="call_123",
            type="function",
            function=Function(
                name="execute_bash",
                arguments=json.dumps(
                    {
                        "command": command,
                        "timeout": self._execution_timeout(),
                    }
                ),
            ),
        )
        observation, info = self.main_agent._execute_tool(tool_call_obj)
        self.terminal_output = observation
        if info.get("exit_code") != 0:
            self.metric_feedback = "SURE metric not run because run_sure.py exited non-zero."
            details = {"execution_info": info, "terminal_output": observation}
            self._write_status(
                False,
                "execution_failed",
                details=details,
                execution_info=info,
                terminal_output=observation,
            )
            return False, None, details

        missing = self._missing_artifacts(role_paths)
        if missing:
            self.metric_feedback = (
                f"SURE metric not run because required artifacts are missing: {missing}"
            )
            details = {"missing_artifacts": missing, "terminal_output": observation}
            self._write_status(
                False,
                "missing_artifacts",
                details=details,
                missing_artifacts=missing,
                terminal_output=observation,
            )
            return False, None, details

        artifact_errors = self._artifact_guard_errors(role_paths)
        if artifact_errors:
            self.metric_feedback = (
                "SURE metric not run because generated artifacts failed integrity checks:\n"
                + "\n".join(f"- {error}" for error in artifact_errors)
            )
            details = {"artifact_guard_errors": artifact_errors, "terminal_output": observation}
            self._write_status(
                False,
                "artifact_guard_failed",
                details=details,
                artifact_guard_errors=artifact_errors,
                terminal_output=observation,
            )
            return False, None, details

        role_paths = get_adapter(self.task_card.canonical_task).scoring_roles(
            self.workspace_path, self.execution_env, role_paths
        )
        metric_result = self.metric_runner.run(
            task_card=self.task_card,
            workspace_path=self.workspace_path,
            output_dir=Path(self.workspace_path) / "metric",
            role_paths=role_paths,
        )
        self.metric_feedback = format_metric_feedback(metric_result)
        details = self._metric_details(metric_result)
        if not metric_result.success:
            self._write_status(
                False,
                "metric_failed",
                details=details,
                score=metric_result.score,
                score_valid=False,
                metric_feedback=self.metric_feedback,
                terminal_output=observation,
            )
            return False, None, details
        self._write_status(
            True,
            "success",
            details=details,
            score=metric_result.score,
            score_valid=metric_result.score is not None,
            metric_accepted=True,
            metric_feedback=self.metric_feedback,
            terminal_output=observation,
        )
        return True, metric_result.score, details

    def _write_remote_candidate_context(self) -> None:
        path = Path(self.workspace_path) / "metric" / "remote_candidate_context.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "execution_env": self._remote_candidate_execution_env(),
            "role_paths": self._current_role_paths,
            "base_model_source_overrides": self.base_model_source_overrides,
            "candidate_type_hint": self.candidate_type_hint,
            "stage": self.candidate_stage_name or self.stage,
            "phase": self.candidate_phase,
            "rung": self.candidate_rung_name,
            "idea_id": self.candidate_idea_id,
            "candidate_started_at": self._candidate_started_at,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _should_run_remote_candidate(self, candidate_type: str) -> bool:
        if sure_config_from(self.config).get("execution_mode") == "slurm":
            return True
        return candidate_runs_remotely(
            self.config,
            candidate_type,
            task_id=self.task_card.task_id,
        )

    def _remote_workload_profile(self) -> str:
        """Return the VC profile for the work the candidate actually performs."""
        candidate_type = normalize_candidate_type(self.candidate_type_hint)
        stage = str(self.candidate_stage_name or self.stage).strip().lower()
        if "draft" in stage:
            candidate_type = candidate_type_from_code(self.code, default=INFERENCE)
            if candidate_type in {FINE_TUNE, ARCH}:
                return "draft_training"
        return candidate_type

    def _execute_and_score_remote_candidate(self) -> tuple[bool, float | None, dict[str, Any]]:
        if sure_config_from(self.config).get("execution_mode") == "slurm":
            from ..utils.slurm import run_candidate
            payload = run_candidate(self)
            self.terminal_output = payload.get("terminal_output", "")
            self.metric_feedback = payload.get("metric_feedback") or payload.get("error", "")
            return bool(payload["success"]), payload.get("score"), payload
        timeout = int(self._execution_timeout())
        executor = VcRemoteTrainingExecutor(
            self.config,
            config_path=self.config_path,
            logger=self.logger,
            candidate_type=self.candidate_type_hint,
            stage=self.candidate_stage_name or self.stage,
            workload_profile=self._remote_workload_profile(),
        )
        if not executor.enabled:
            self.metric_feedback = (
                "Remote candidate execution was selected but sure.remote_training.enabled is false."
            )
            details = {"error": self.metric_feedback}
            self._write_status(False, "remote_disabled", details=details, remote_success=False)
            return False, None, details

        payload = executor.run(
            workspace_path=self.workspace_path,
            exp_name=self.exp_name,
            execution_timeout=timeout,
        )
        self.terminal_output = str(payload.get("terminal_output") or "")
        self.metric_feedback = str(
            payload.get("metric_feedback")
            or payload.get("error")
            or "Remote candidate finished without metric feedback."
        )

        score = payload.get("score")
        success = bool(payload.get("success")) and score is not None
        try:
            score_value = float(score) if score is not None else None
        except (TypeError, ValueError):
            score_value = None
            success = False
        reason_code = str(
            payload.get("reason_code")
            or ("success" if success else "remote_candidate_failed")
        )
        details = payload.get("details") if isinstance(payload.get("details"), dict) else payload
        self._write_status(
            success,
            reason_code,
            details=details,
            score=score_value,
            score_valid=score_value is not None,
            metric_accepted=success,
            remote_success=bool(payload.get("success")),
            missing_artifacts=details.get("missing_artifacts") if isinstance(details, dict) else None,
            artifact_guard_errors=details.get("artifact_guard_errors") if isinstance(details, dict) else None,
            execution_info=payload.get("execution_info") if isinstance(payload.get("execution_info"), dict) else {},
            metric_feedback=self.metric_feedback,
            terminal_output=self.terminal_output,
        )
        return success, score_value, payload

    def _missing_artifacts(self, role_paths: dict[str, str | None]) -> list[str]:
        missing: list[str] = []
        for role in self.task_card.required_roles:
            path = self._resolve_role_path(role, role_paths)
            if path is not None and not path.exists():
                missing.append(f"{role}:{path}")
        return missing

    def _artifact_guard_errors(self, role_paths: dict[str, str | None]) -> list[str]:
        errors = get_adapter(self.task_card.canonical_task).validate_outputs(self, role_paths)
        record = Path(self.workspace_path) / "artifacts/candidate_changes.json"
        if self.enforce_candidate_type and record.is_file():
            try:
                actual = normalize_candidate_type(json.loads(record.read_text()).get("candidate_type"))
                if actual != self.candidate_type_hint:
                    errors.append(f"Executed candidate type {actual} differs from reviewed {self.candidate_type_hint}")
            except (ValueError, AttributeError):
                errors.append("Invalid candidate execution record")
        return errors


    def _candidate_changes_guard_errors(self) -> list[str]:
        if self._execution_env_truthy("SURE_REQUIRE_CANDIDATE_CHANGES", False):
            return validate_candidate_changes(self.workspace_path)
        return []

    def _architecture_guard_errors(self) -> list[str]:
        # Kept as a compatibility hook; guards are task-specific now.
        from ...tasks.guards import AsrGuards
        if self.task_card.canonical_task == "asr":
            return AsrGuards(self)._architecture_guard_errors()
        return validate_arch_candidate_changes(self.workspace_path) if self.candidate_type_hint == ARCH else []






    @staticmethod
    def _parse_positive_int(value: Any) -> int:
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return 0

    def _execution_env_truthy(self, name: str, default: bool = False) -> bool:
        raw = self.execution_env.get(name, os.environ.get(name))
        if raw is None:
            return default
        text = str(raw).strip().lower()
        if text == "":
            return default
        return text in {"1", "true", "yes", "on"}

    def _execution_env_path(self, name: str) -> Path | None:
        raw = self.execution_env.get(name, os.environ.get(name))
        text = str(raw or "").strip()
        if not text:
            return None
        return Path(text).expanduser()

    @staticmethod
    def _path_is_relative_to(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    @staticmethod
    def _path_mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    @staticmethod
    def _checkpoint_matches_baseline_copy(path: Path, baseline_root: Path) -> bool:
        if path.name.startswith("bad-model-") or not path.is_file():
            return False
        try:
            for baseline in baseline_root.rglob(path.name):
                try:
                    if path.samefile(baseline):
                        return True
                except OSError:
                    pass
                try:
                    if path.stat().st_size == baseline.stat().st_size and filecmp.cmp(path, baseline, shallow=False):
                        return True
                except OSError:
                    continue
        except OSError:
            return False
        return False





    @staticmethod
    def _workspace_relative(path: Path, workspace: Path) -> str:
        try:
            return str(path.relative_to(workspace))
        except ValueError:
            return str(path)



    @staticmethod
    def _resolve_samples_jsonl_path(samples_path: Path, value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path.resolve(strict=False)
        return (samples_path.parent / path).resolve(strict=False)

    @staticmethod
    def _same_file_content(left: Path, right: Path) -> bool:
        try:
            if left.samefile(right):
                return True
        except OSError:
            pass
        try:
            return filecmp.cmp(left, right, shallow=False)
        except OSError:
            return False

    def _resolve_role_path(
        self,
        role: str,
        role_paths: dict[str, str | None],
    ) -> Path | None:
        value = role_paths.get(role, self.task_card.artifact_contract.get(role))
        if value is None:
            return None
        if str(value).startswith("literal:"):
            return None
        path = Path(str(value))
        if path.is_absolute():
            return path
        return Path(self.workspace_path) / path

    @staticmethod
    def _read_key_text_file(path: Path) -> dict[str, str]:
        rows: dict[str, str] = {}
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.rstrip("\n")
                if "\t" not in line:
                    continue
                key, text = line.split("\t", 1)
                key = key.strip()
                if key:
                    rows[key] = text.strip()
        return rows

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9']+", str(text).lower()))

    def _prepare_workspace_inputs(self, role_paths: dict[str, str | None]) -> None:
        """Copy configured read-only input roles to their task-card workspace paths."""
        workspace = Path(self.workspace_path)
        for role in self.task_card.required_roles:
            default_value = self.task_card.artifact_contract.get(role)
            configured_value = role_paths.get(role)
            if not default_value or configured_value is None:
                continue
            if str(default_value).startswith("literal:") or str(configured_value).startswith("literal:"):
                continue
            default_path = Path(str(default_value))
            if default_path.is_absolute() or not default_path.parts or default_path.parts[0] != "input":
                continue
            configured_path = Path(str(configured_value))
            if not configured_path.is_absolute() and configured_path == default_path:
                continue
            source = self._resolve_configured_input(str(configured_value))
            if not source.is_file():
                continue
            target = workspace / default_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.exists() and target.is_dir():
                self.logger.warning("Cannot copy configured input over directory: %s", target)
                continue
            shutil.copy2(source, target)
            self.logger.info("Copied configured SURE input %s: %s -> %s", role, source, target)

    def _prepare_base_model_source_overrides(self) -> None:
        """Apply per-experiment base_model source-path overrides.

        This is used by staged selection/holdout runs where, for example,
        `base_model/eval_data` should point to a selection prompt set while the
        main search workspace still points to regular_search.
        """
        if not self.base_model_source_overrides or self.base_model_profile is None:
            return
        workspace = Path(self.workspace_path)
        for name, source in self.base_model_source_overrides.items():
            target_rel = self.base_model_profile.required_paths.get(name)
            if not target_rel:
                self.logger.warning("Ignoring base_model override without required path: %s", name)
                continue
            source_path = Path(str(source)).expanduser()
            if not source_path.is_absolute():
                source_path = Path.cwd() / source_path
            source_path = source_path.resolve(strict=False)
            if not source_path.exists():
                self.logger.warning("Ignoring missing base_model override %s: %s", name, source_path)
                continue
            target = workspace / str(target_rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink():
                target.unlink()
            elif target.is_dir() and name == "recipe" and self.task_card.canonical_task == "asr":
                target.resolve().relative_to(workspace.resolve())
                shutil.rmtree(target)
            elif target.is_dir():
                entries = list(target.iterdir())
                if any(not entry.is_symlink() for entry in entries):
                    self.logger.warning(
                        "Cannot apply base_model override over directory with non-symlink contents: %s",
                        target,
                    )
                    continue
                for entry in entries:
                    entry.unlink()
                target.rmdir()
            elif target.exists():
                self.logger.warning("Cannot apply base_model override over non-symlink path: %s", target)
                continue
            target.symlink_to(source_path, target_is_directory=source_path.is_dir())
            self.logger.info("Applied base_model override %s: %s -> %s", name, target, source_path)

    def _resolve_configured_input(self, value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        workspace_path = Path(self.workspace_path) / path
        if workspace_path.exists():
            return workspace_path
        return Path.cwd() / path

    def _execution_command(self) -> str:
        env = self._candidate_command_env()
        env.setdefault(
            "SURE_DURATION_CANDIDATE_KEY",
            hashlib.sha256((self.code or "").encode("utf-8")).hexdigest()[:16],
        )
        workspace = Path(self.workspace_path)
        cache_dir = workspace / ".sure_runtime" / "duration_autotune"
        recipe_dir = workspace / "base_model" / "recipe"
        recipe_key_source = str(recipe_dir)
        if self.base_model_profile is not None:
            source_recipe = self.base_model_source_overrides.get(
                "recipe",
                self.base_model_profile.source_paths.get("recipe", ""),
            )
            if source_recipe:
                recipe_key_source = source_recipe
        recipe_key = hashlib.sha256(recipe_key_source.encode("utf-8")).hexdigest()[:16]

        lines: list[str] = []
        for key, value in sorted(env.items()):
            key_text = str(key)
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_text):
                self.logger.warning("Ignoring invalid execution environment key: %s", key)
                continue
            lines.append(f"export {key_text}={shlex.quote(str(value))}")
        lines.extend(
            [
                "set -e",
                'if [ -z "${SURE_DURATION_CACHE_DIR:-}" ]; then',
                f"  export SURE_DURATION_CACHE_DIR={shlex.quote(str(cache_dir))}",
                "fi",
                'if [ -z "${SURE_DURATION_RECIPE_DIR:-}" ]; then',
                f"  export SURE_DURATION_RECIPE_DIR={shlex.quote(str(recipe_dir))}",
                "fi",
                'if [ -z "${SURE_DURATION_RECIPE_KEY:-}" ]; then',
                f"  export SURE_DURATION_RECIPE_KEY={shlex.quote(recipe_key)}",
                "fi",
                'if [ -z "${SURE_RUNTIME_ENV_HELPER:-}" ]; then',
                f"  export SURE_RUNTIME_ENV_HELPER={shlex.quote(str(self._runtime_env_helper_path()))}",
                "fi",
                f"exec {shlex.quote(env.get('SURE_CANDIDATE_PYTHON', sys.executable))} run_sure.py",
            ]
        )
        return "bash -lc " + shlex.quote("\n".join(lines))

    def _candidate_command_env(self) -> dict[str, str]:
        env = dict(self.execution_env)
        if self.task_card.canonical_task != "asr":
            self._ensure_candidate_pythonpath(env)
            return env
        local_icefall_python = env.get("SURE_LOCAL_ICEFALL_PYTHON") or os.environ.get(
            "SURE_LOCAL_ICEFALL_PYTHON"
        )
        if local_icefall_python and not self._should_run_remote_candidate(self.candidate_type_hint):
            env["SURE_ICEFALL_PYTHON"] = str(local_icefall_python)
        env.pop("SURE_LOCAL_ICEFALL_PYTHON", None)
        env.pop("SURE_REMOTE_ICEFALL_PYTHON", None)
        self._ensure_candidate_pythonpath(env)
        return env

    def _remote_candidate_execution_env(self) -> dict[str, str]:
        env = dict(self.execution_env)
        remote_icefall_python = env.pop("SURE_REMOTE_ICEFALL_PYTHON", "")
        env.pop("SURE_LOCAL_ICEFALL_PYTHON", None)
        if remote_icefall_python:
            env["SURE_ICEFALL_PYTHON"] = str(remote_icefall_python)
            env["SURE_REMOTE_ICEFALL_PYTHON"] = str(remote_icefall_python)
        self._ensure_candidate_pythonpath(env)
        return env

    def _ensure_candidate_pythonpath(self, env: dict[str, str]) -> None:
        entries: list[str] = [str(PROJECT_ROOT)]
        current = str(env.get("PYTHONPATH") or "")
        if current:
            entries.extend(part for part in current.split(os.pathsep) if part)
        if self.base_model_profile is not None:
            for key in ("root", "recipe", "data"):
                source = self.base_model_source_overrides.get(
                    key,
                    self.base_model_profile.source_paths.get(key, ""),
                )
                if source:
                    entries.append(str(source))
        workspace_root = Path(self.workspace_path) / "base_model" / "root"
        entries.append(str(workspace_root))

        deduped: list[str] = []
        seen: set[str] = set()
        for entry in entries:
            if not entry:
                continue
            normalized = str(Path(entry).expanduser())
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        env["PYTHONPATH"] = os.pathsep.join(deduped)

    def _candidate_type_from_code(self, code: str) -> str:
        if self._current_response_is_initial_solution and not self._debug_initial_solution_enabled():
            return normalize_candidate_type(self.candidate_type_hint, default=INFERENCE)
        return candidate_type_from_code(code, default=self.candidate_type_hint)

    def _debug_initial_solution_enabled(self) -> bool:
        if isinstance(self.config, dict):
            config_dict = self.config
        elif hasattr(self.config, "model_dump"):
            config_dict = self.config.model_dump()
        else:
            config_dict = {}
        sure_config = config_dict.get("sure") or {}
        raw = sure_config.get("debug_initial_solution", False) if isinstance(sure_config, dict) else False
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _runtime_env_helper_path(self) -> Path:
        candidates: list[Path] = []
        for root_value in (
            self.workspace_path,
            getattr(getattr(getattr(self, "main_agent", None), "session", None), "config", None)
            and getattr(getattr(self.main_agent.session, "config", None), "workspace_path", None),
            getattr(getattr(getattr(self, "main_agent", None), "session", None), "config", None)
            and getattr(getattr(self.main_agent.session, "config", None), "config_dir", None),
        ):
            if not root_value:
                continue
            root = Path(str(root_value))
            candidates.append(root)
            candidates.extend(root.parents)
        for root in candidates:
            path = root / "playground/sure_master/core/utils/runtime_env.py"
            if path.is_file():
                return path
        return Path(__file__).parent.parent / "utils" / "runtime_env.py"

    def _execution_timeout(self) -> str:
        raw = self.execution_env.get("SURE_RUN_TIMEOUT", "86400")
        try:
            parsed = int(float(str(raw).strip()))
            value = 0 if parsed == 0 else max(1, parsed)
        except (TypeError, ValueError):
            self.logger.warning("Invalid SURE_RUN_TIMEOUT=%r; using 86400 seconds", raw)
            value = 86400
        return str(value)

    @staticmethod
    def _metric_details(metric_result: SureMetricResult) -> dict[str, Any]:
        return {
            "success": metric_result.success,
            "score": metric_result.score,
            "metric": metric_result.metric,
            "pipeline_id": metric_result.pipeline_id,
            "report_path": metric_result.report_path,
            "summary_path": metric_result.summary_path,
            "error": metric_result.error,
            "details": metric_result.details,
        }

    def _base_model_prompt_text(self) -> str:
        if self.base_model_profile is None:
            return "No base model profile is configured for this task."
        return self.base_model_profile.to_prompt_json()

    @staticmethod
    def _role_paths_prompt_text(role_paths: dict[str, str | None]) -> str:
        return json.dumps(role_paths, ensure_ascii=False, indent=2)

    def _execution_env_prompt_text(self) -> str:
        if not self.execution_env:
            return "No SURE execution environment variables are configured."
        return json.dumps(self.execution_env, ensure_ascii=False, indent=2)

    def _candidate_type_guidance_text(self) -> str:
        sure = sure_config_from(self.config)
        contract = scoped_execution_contract(get_adapter(self.task_card.canonical_task).context(), sure)
        return (contract.get("research_guidance") or
                "Implement the reviewed XLab idea with its declared execution type and fixed experiment budget.") + "\n" + json.dumps(contract)

    def _write_status(
        self,
        success: bool,
        reason_code: str,
        *,
        details: dict[str, Any] | None = None,
        score: float | None = None,
        score_valid: bool = False,
        metric_accepted: bool = False,
        remote_success: bool | None = None,
        missing_artifacts: list[str] | None = None,
        artifact_guard_errors: list[str] | None = None,
        execution_info: dict[str, Any] | None = None,
        metric_feedback: str | None = None,
        terminal_output: str | None = None,
    ) -> None:
        if details is not None:
            details["reason_code"] = reason_code
            started = getattr(self, "_candidate_started_at", 0.0)
            if started:
                details["runtime_seconds"] = max(0.0, time.time() - started)
        if success:
            retained = get_adapter(self.task_card.canonical_task).collect_model(
                self.workspace_path, self.execution_env
            )
            if retained:
                if details is None:
                    details = {}
                details["produced_artifacts"] = retained
        write_candidate_status(
            self.workspace_path,
            success=success,
            reason_code=reason_code,
            stage=self.candidate_stage_name or self.stage,
            phase=self.candidate_phase,
            rung=self.candidate_rung_name,
            idea_id=self.candidate_idea_id,
            candidate_type=self.candidate_type_hint,
            score=score,
            score_valid=score_valid,
            metric_accepted=metric_accepted,
            remote_success=remote_success,
            missing_artifacts=missing_artifacts,
            artifact_guard_errors=artifact_guard_errors,
            execution_info=execution_info,
            details=details,
            metric_feedback=metric_feedback if metric_feedback is not None else self.metric_feedback,
            terminal_output=terminal_output if terminal_output is not None else self.terminal_output,
        )
        self._cleanup_candidate_workspace(
            success=success,
            reason_code=reason_code,
            score=score,
            details=details,
        )

    def _cleanup_candidate_workspace(
        self,
        *,
        success: bool,
        reason_code: str,
        score: float | None,
        details: dict[str, Any] | None,
    ) -> None:
        try:
            cleanup_config = workspace_cleanup_config(self.config, self.execution_env)
            manifest = cleanup_candidate_workspace(
                self.workspace_path,
                cleanup_config,
                success=success,
                reason_code=reason_code,
                score=score,
                metadata={
                    "stage": self.candidate_stage_name or self.stage,
                    "phase": self.candidate_phase,
                    "rung": self.candidate_rung_name,
                    "idea_id": self.candidate_idea_id,
                    "candidate_type": self.candidate_type_hint,
                    "details_keys": sorted((details or {}).keys()),
                },
            )
            if manifest.get("cleaned"):
                self.logger.info(
                    "Cleaned SURE candidate workspace %s: removed %.2f MiB",
                    self.workspace_path,
                    int(manifest.get("removed_bytes") or 0) / (1024 * 1024),
                )
        except Exception as exc:
            self.logger.warning(
                "SURE candidate workspace cleanup failed for %s: %s",
                self.workspace_path,
                exc,
            )
