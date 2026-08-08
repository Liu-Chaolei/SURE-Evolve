from __future__ import annotations

import json
import logging
import re
import uuid

from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance

from ..utils.task_cards import BaseModelProfile, SureTaskCard


def _parse_json_from_response(text: str) -> dict:
    text = text.strip()
    candidates = [text]
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    )

    decoder = json.JSONDecoder()
    decoded: list[dict] = []
    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                decoded.append(obj)
                continue
        except json.JSONDecodeError as e:
            last_error = e

        for match in re.finditer(r"[{]", candidate):
            try:
                obj, _end = decoder.raw_decode(candidate[match.start() :])
            except json.JSONDecodeError as e:
                last_error = e
                continue
            if isinstance(obj, dict):
                decoded.append(obj)

    for obj in reversed(decoded):
        if "cmd" not in obj:
            return obj
    if decoded:
        return decoded[-1]
    if last_error is not None:
        raise last_error
    raise ValueError("No JSON object found in wisdom response.")


class WisdomPromotionExp(BaseExp):
    """Extract reusable SURE task wisdom from the best solution."""

    def __init__(
        self,
        wisdom_promotion_agent,
        config,
        exp_name: str,
        task_card: SureTaskCard,
        base_model_profile: BaseModelProfile | None = None,
    ):
        super().__init__(wisdom_promotion_agent, config)
        self.wisdom_promotion_agent = wisdom_promotion_agent
        self._exp_name = exp_name
        self.task_card = task_card
        self.base_model_profile = base_model_profile
        self.uid = uuid.uuid4()
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def exp_name(self) -> str:
        return self._exp_name

    def run(self, task_description: str, best_solution: str, task_id: str = "exp_001") -> dict:
        original_kwargs = self.wisdom_promotion_agent._prompt_format_kwargs.copy()
        try:
            self.wisdom_promotion_agent._prompt_format_kwargs.update(
                {
                    "task_description": task_description,
                    "task_card": self.task_card.to_prompt_json(),
                    "base_model_profile": self._base_model_prompt_text(),
                    "best_solution": best_solution,
                }
            )
            task = TaskInstance(
                task_id=f"{task_id}_wisdom_promotion",
                task_type="wisdom_promotion",
                description=task_description,
                input_data={},
            )
            trajectory = self.wisdom_promotion_agent.run(task)
            result = _parse_json_from_response(self._extract_agent_response(trajectory))
            self.logger.info("Wisdom promotion result: %s", result)
            return result
        finally:
            self.wisdom_promotion_agent._prompt_format_kwargs = original_kwargs

    def _base_model_prompt_text(self) -> str:
        if self.base_model_profile is None:
            return "No base model profile is configured for this task."
        return self.base_model_profile.to_prompt_json()
