import logging
import json
from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function
from ..utils.code import read_code, save_code_to_file
from ..utils.metric import extract_wer_from_metric
import uuid
import os
from evomaster.agent import BaseAgent


class DraftExp(BaseExp):
    """Experiment for generating and validating an initial ASR solution draft.

    Orchestrates the draft -> execute -> metric -> debug cycle to produce
    a working initial wrapper script for the given ASR optimization task.

    Unlike ml_master_2, this does NOT require a submission CSV. Success is
    determined by the wrapper script completing (exit_code == 0) and the
    metric agent successfully extracting a WER from the terminal output.
    """

    def __init__(self, draft_agent, debug_agent, metric_agent, config, exp_name):
        super().__init__(draft_agent, config)
        self.draft_agent = draft_agent
        self.debug_agent = debug_agent
        self.metric_agent = metric_agent
        self.uid = uuid.uuid4()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.terminal_output = ""
        self.code = ""
        self.debug_times = 0
        self._exp_name = exp_name
        self.workspace_path = os.path.join(self.draft_agent.session.config.workspace_path, exp_name)

    @property
    def exp_name(self) -> str:
        """Return the experiment stage name."""
        return self._exp_name

    def _extract_wer_from_metric(self, metric_result: str) -> tuple[bool, float | None]:
        """Extract the WER value from the metric agent's response.

        Args:
            metric_result: The raw response text from the metric agent.

        Returns:
            A tuple of (success, wer_value). wer_value is None on failure.
        """
        return extract_wer_from_metric(metric_result)

    def run(self, task_description: str, data_preview: str, data_knowledge: str, model_knowledge: str, task_id: str = "exp_001") -> dict:
        """Execute the draft experiment pipeline.

        Generates initial wrapper code via draft agent, executes it, extracts
        the WER metric, and retries with debug agent on failure (up to 3 times).

        Args:
            task_description: Natural language description of the ASR task.
            data_preview: Textual preview of the dataset and recipe.
            data_knowledge: Retrieved data knowledge from wisdom database.
            model_knowledge: Retrieved model knowledge from wisdom database.
            task_id: Unique task identifier.

        Returns:
            Tuple of (is_success, validation_score (WER), uid, code).
        """
        self.logger.info("Starting draft task execution")
        self.logger.info(f"Task: {task_description}")

        try:
            while True:
                if self.draft_agent:
                    self.logger.info("=" * 60)
                    self.logger.info("Step 1: Draft Agent analyzing task...")
                    self.logger.info("=" * 60)
                    BaseAgent.set_exp_info(exp_name=self.exp_name, exp_index=1)

                    draft_original_format_kwargs = self.draft_agent._prompt_format_kwargs.copy()
                    self.draft_agent._prompt_format_kwargs.update({
                        'task_description': task_description,
                        'data_preview': data_preview,
                        'data_knowledge': data_knowledge,
                        'model_knowledge': model_knowledge,
                    })

                    draft_task = TaskInstance(
                        task_id=f"{task_id}_draft",
                        task_type="draft",
                        description=task_description,
                        input_data={},
                    )

                    draft_trajectory = self.draft_agent.run(draft_task)
                    draft_result = self._extract_agent_response(draft_trajectory)
                    draft_code, self.code = read_code(draft_result, self.uid)
                    save_code_to_file(self.workspace_path, "run_asr.py", draft_code)
                    # Execute with relative path - cwd is workspace_path where symlinks exist
                    tool_call_obj = ChatCompletionMessageToolCall(
                        id="call_123",
                        type="function",
                        function=Function(
                            name="execute_bash",
                            arguments=json.dumps({"command": "python run_asr.py", "timeout": "86400"})
                        )
                    )
                    observation, info = self.draft_agent._execute_tool(tool_call_obj)
                    self.terminal_output = observation

                    # ASR success: script must complete (exit_code == 0)
                    is_success = info.get("exit_code") == 0
                    self.logger.info(f"Draft Agent execute_bash result: {observation}")
                    self.logger.info(f"Draft Agent execute_bash info: {info}")

                    self.logger.info("Draft completed")
                    self.logger.info(f"Draft result: {draft_result[:2000]}...")
                    self.draft_agent._prompt_format_kwargs = draft_original_format_kwargs

                if self.metric_agent and is_success:
                    self.logger.info("=" * 60)
                    self.logger.info("Step 2: Metric Agent extracting WER...")
                    self.logger.info("=" * 60)
                    metric_original_format_kwargs = self.metric_agent._prompt_format_kwargs.copy()
                    self.metric_agent._prompt_format_kwargs.update({
                        'terminal_output': observation
                    })
                    metric_task = TaskInstance(
                        task_id=f"{task_id}_metric",
                        task_type="metric",
                        input_data={},
                    )

                    metric_trajectory = self.metric_agent.run(metric_task)
                    metric_result = self._extract_agent_response(metric_trajectory)
                    metric_ok, validation_score = self._extract_wer_from_metric(metric_result)
                    is_success = metric_ok
                    self.logger.info(f"validation score (WER): {validation_score}")
                    self.logger.info("Metric completed")
                    self.logger.info(f"Metric result: {metric_result[:2000]}...")
                    self.metric_agent._prompt_format_kwargs = metric_original_format_kwargs

                debug_times = 0
                while is_success is False and debug_times < 3:
                    self.logger.info("=" * 60)
                    self.logger.info("Step 3: Debug Agent executing task...")
                    self.logger.info("=" * 60)
                    debug_original_format_kwargs = self.debug_agent._prompt_format_kwargs.copy()
                    self.debug_agent._prompt_format_kwargs.update({
                        'task_description': task_description,
                        'terminal_output': self.terminal_output,
                        'buggy_code': self.code,
                        'data_preview': data_preview,
                    })
                    debug_task = TaskInstance(
                        task_id=f"{task_id}_debug",
                        task_type="debug",
                        task_description=task_description,
                        input_data={},
                    )
                    debug_trajectory = self.debug_agent.run(debug_task)
                    debug_result = self._extract_agent_response(debug_trajectory)
                    debug_code, self.code = read_code(debug_result, self.uid)
                    save_code_to_file(self.workspace_path, "run_asr.py", debug_code)
                    # Execute with relative path - cwd is workspace_path where symlinks exist
                    tool_call_obj = ChatCompletionMessageToolCall(
                        id="call_123",
                        type="function",
                        function=Function(
                            name="execute_bash",
                            arguments=json.dumps({"command": "python run_asr.py", "timeout": "86400"})
                        )
                    )
                    observation, info = self.debug_agent._execute_tool(tool_call_obj)
                    self.terminal_output = observation
                    debug_success = info.get("exit_code") == 0
                    self.logger.info(f"Debug Agent execute_bash result: {observation}")
                    self.logger.info(f"Debug Agent execute_bash info: {info}")
                    self.logger.info("Debug completed")
                    self.logger.info(f"Debug result: {debug_result[:2000]}...")
                    self.debug_agent._prompt_format_kwargs = debug_original_format_kwargs

                    if self.metric_agent and debug_success:
                        self.logger.info("=" * 60)
                        self.logger.info("Step 4: Metric Agent extracting WER...")
                        self.logger.info("=" * 60)
                        metric_original_format_kwargs = self.metric_agent._prompt_format_kwargs.copy()
                        self.metric_agent._prompt_format_kwargs.update({
                            'terminal_output': observation
                        })
                        metric_task = TaskInstance(
                            task_id=f"{task_id}_metric",
                            task_type="metric",
                            input_data={},
                        )

                        metric_trajectory = self.metric_agent.run(metric_task)
                        metric_result = self._extract_agent_response(metric_trajectory)
                        metric_ok, validation_score = self._extract_wer_from_metric(metric_result)
                        debug_success = metric_ok
                        self.logger.info(f"validation score (WER): {validation_score}")
                        self.logger.info("Metric completed")
                        self.logger.info(f"Metric result: {metric_result[:2000]}...")
                        self.metric_agent._prompt_format_kwargs = metric_original_format_kwargs

                    if debug_success:
                        is_success = True
                        return is_success, validation_score, self.uid, self.code
                    else:
                        is_success = False
                        validation_score = None
                        debug_times += 1

                return is_success, validation_score, self.uid, self.code

        except Exception as e:
            self.logger.error(f"Draft task execution failed: {e}", exc_info=True)
            raise ValueError(f"Draft task execution failed: {e}")
