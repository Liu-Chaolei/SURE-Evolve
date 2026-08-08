from __future__ import annotations

import logging
import uuid

from evomaster.agent import BaseAgent
from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance

from ..utils.task_cards import BaseModelProfile, SureTaskCard


class PrefetchExp(BaseExp):
    """Generate a retrieval descriptor for a SURE task.

    v1 keeps SURE as a read-only metric backend and does not require a wisdom
    database. The descriptor is still useful in logs and leaves a stable hook
    for future RAG retrieval.
    """

    def __init__(
        self,
        prefetch_agent,
        config,
        exp_name: str,
        task_card: SureTaskCard,
        base_model_profile: BaseModelProfile | None = None,
    ):
        super().__init__(prefetch_agent, config)
        self.prefetch_agent = prefetch_agent
        self._exp_name = exp_name
        self.task_card = task_card
        self.base_model_profile = base_model_profile
        self.uid = uuid.uuid4()
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def exp_name(self) -> str:
        return self._exp_name

    def run(self, task_description: str, task_id: str = "exp_001") -> tuple[str, str, str]:
        data_knowledge = ""
        model_knowledge = ""
        descriptor = task_description
        if not self.prefetch_agent:
            return data_knowledge, model_knowledge, descriptor

        BaseAgent.set_exp_info(exp_name=self.exp_name, exp_index=1)
        original_kwargs = self.prefetch_agent._prompt_format_kwargs.copy()
        try:
            self.prefetch_agent._prompt_format_kwargs.update(
                {
                    "task_description": task_description,
                    "task_card": self.task_card.to_prompt_json(),
                    "base_model_profile": self._base_model_prompt_text(),
                }
            )
            task = TaskInstance(
                task_id=f"{task_id}_prefetch",
                task_type="prefetch",
                description=task_description,
                input_data={},
            )
            trajectory = self.prefetch_agent.run(task)
            descriptor = self._extract_agent_response(trajectory) or task_description
            self.logger.info("SURE prefetch descriptor: %s", descriptor)
            return data_knowledge, model_knowledge, descriptor
        finally:
            self.prefetch_agent._prompt_format_kwargs = original_kwargs

    def _base_model_prompt_text(self) -> str:
        if self.base_model_profile is None:
            return "No base model profile is configured for this task."
        return self.base_model_profile.to_prompt_json()
