from __future__ import annotations

import logging
import uuid
from typing import Any

from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance

from ..utils.task_cards import BaseModelProfile, SureTaskCard


def generate_improvement_summary_text(
    base_solution: str,
    best_solution: str,
    research_plan: dict,
    research_round_idea_results: dict,
    task_card: SureTaskCard,
) -> str:
    lines = [
        "## SURE Improvement Process Summary",
        "",
        f"Task: {task_card.task_id}",
        f"Metric: {task_card.primary_metric} ({task_card.metric_direction} is better)",
        "",
        "## Results by Direction",
    ]
    for direction in research_plan:
        lines.extend(["", f"### {direction}"])
        direction_results = research_round_idea_results.get(direction, {})
        for idea_key, idea_desc in research_plan[direction].items():
            result = direction_results.get((idea_key, idea_desc), {})
            score = result.get("score")
            improved = result.get("improved", False)
            adopted = result.get("is_best_in_direction", False)
            candidate_type = result.get("candidate_type")
            type_text = f" | type={candidate_type}" if candidate_type else ""
            lines.append(
                f"- Idea {idea_key}: {idea_desc} | score={score} | "
                f"improved={improved} | adopted={adopted}{type_text}"
            )
    lines.extend(
        [
            "",
            "## Base Code",
            "```python",
            base_solution or "",
            "```",
            "",
            "## Final Best Code",
            "```python",
            best_solution or "",
            "```",
        ]
    )
    return "\n".join(lines)


class KnowledgePromotionExp(BaseExp):
    """Summarize one SURE research round into reusable guidance."""

    def __init__(
        self,
        knowledge_promotion_agent,
        config,
        exp_name: str,
        task_card: SureTaskCard,
        base_model_profile: BaseModelProfile | None = None,
    ):
        super().__init__(knowledge_promotion_agent, config)
        self.knowledge_promotion_agent = knowledge_promotion_agent
        self._exp_name = exp_name
        self.task_card = task_card
        self.base_model_profile = base_model_profile
        self.uid = uuid.uuid4()
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def exp_name(self) -> str:
        return self._exp_name

    def run(
        self,
        task_description: str,
        data_preview: str,
        base_solution: str,
        best_solution: str,
        research_plan: dict,
        research_round_idea_results: dict,
        task_id: str = "exp_001",
    ) -> str:
        results_text = generate_improvement_summary_text(
            base_solution=base_solution,
            best_solution=best_solution,
            research_plan=research_plan,
            research_round_idea_results=research_round_idea_results,
            task_card=self.task_card,
        )
        original_kwargs = self.knowledge_promotion_agent._prompt_format_kwargs.copy()
        try:
            self.knowledge_promotion_agent._prompt_format_kwargs.update(
                {
                    "task_description": task_description,
                    "data_preview": data_preview,
                    "task_card": self.task_card.to_prompt_json(),
                    "base_model_profile": self._base_model_prompt_text(),
                    "current_base_code": base_solution,
                    "research_plan": research_plan,
                    "results": results_text,
                }
            )
            task = TaskInstance(
                task_id=f"{task_id}_knowledge_promotion",
                task_type="knowledge_promotion",
                description=task_description,
                input_data={},
            )
            trajectory = self.knowledge_promotion_agent.run(task)
            result = self._extract_agent_response(trajectory)
            self.logger.info("Knowledge promotion result: %s", result)
            return result
        finally:
            self.knowledge_promotion_agent._prompt_format_kwargs = original_kwargs

    def _base_model_prompt_text(self) -> str:
        if self.base_model_profile is None:
            return "No base model profile is configured for this task."
        return self.base_model_profile.to_prompt_json()
