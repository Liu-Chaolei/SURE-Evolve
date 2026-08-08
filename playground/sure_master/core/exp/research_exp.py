from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from evomaster.agent import BaseAgent
from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance

from ..utils.task_cards import BaseModelProfile, SureTaskCard
from ..utils.vc_remote import mixed_execution_enabled, sure_config_from


def _parse_json_from_response(text: str) -> dict:
    text = text.strip()
    candidates = [text]
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    )

    decoder = json.JSONDecoder()
    decoded: list[dict[str, Any]] = []
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
        if any(str(key).lower().startswith("major direction") for key in obj):
            return obj
    for obj in reversed(decoded):
        if "cmd" not in obj:
            return obj
    if decoded:
        return decoded[-1]
    if last_error is not None:
        raise last_error
    raise ValueError("No JSON object found in research response.")


class ResearchExp(BaseExp):
    """Generate structured SURE improvement plans."""

    def __init__(
        self,
        research_agent,
        config,
        initial_code: str,
        exp_name: str,
        task_card: SureTaskCard,
        base_model_profile: BaseModelProfile | None = None,
        research_request: str | None = None,
    ):
        super().__init__(research_agent, config)
        self.research_agent = research_agent
        self.initial_code = initial_code
        self.task_card = task_card
        self.base_model_profile = base_model_profile
        self.research_request = research_request
        self.uid = uuid.uuid4()
        self._exp_name = exp_name
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def exp_name(self) -> str:
        return self._exp_name

    def run(
        self,
        task_description: str,
        data_preview: str,
        best_solution: str,
        research_plan_and_result: list,
        task_id: str = "exp_001",
    ) -> dict:
        if not research_plan_and_result:
            history_text = "No improvement attempts have been made yet."
            best_solution = "Best solution is the same as the first draft code."
        else:
            blocks = []
            for i in range(0, len(research_plan_and_result), 2):
                plan = research_plan_and_result[i] if i < len(research_plan_and_result) else ""
                result = research_plan_and_result[i + 1] if i + 1 < len(research_plan_and_result) else ""
                blocks.append(
                    "Previous research plan:\n"
                    f"{plan}\n"
                    "Conclusion:\n"
                    f"{result}"
                )
            history_text = "\n\n".join(blocks)

        BaseAgent.set_exp_info(exp_name=self.exp_name, exp_index=1)
        original_kwargs = self.research_agent._prompt_format_kwargs.copy()
        try:
            self.research_agent._prompt_format_kwargs.update(
                {
                    "task_description": task_description,
                    "data_preview": data_preview,
                    "task_card": self.task_card.to_prompt_json(),
                    "base_model_profile": self._base_model_prompt_text(),
                    "initial_code": self.initial_code,
                    "best_code": best_solution,
                    "research_plan_and_result_text": history_text,
                    "candidate_type_guidance": self._candidate_type_guidance_text(),
                    "research_request": self._research_request_text(),
                }
            )
            task = TaskInstance(
                task_id=f"{task_id}_research",
                task_type="research",
                description=task_description,
                input_data={},
            )
            trajectory = self.research_agent.run(task)
            response = self._extract_agent_response(trajectory)
            plan = _parse_json_from_response(response)
            self.logger.info("Research plan: %s", plan)
            return plan
        finally:
            self.research_agent._prompt_format_kwargs = original_kwargs

    def _base_model_prompt_text(self) -> str:
        if self.base_model_profile is None:
            return "No base model profile is configured for this task."
        return self.base_model_profile.to_prompt_json()

    def _candidate_type_guidance_text(self) -> str:
        if not mixed_execution_enabled(self.config, task_id=self.task_card.task_id):
            return "No candidate type labels are required for this task."
        sure_config = sure_config_from(self.config)
        staged = sure_config.get("staged_axes") or {}
        strategy = str(sure_config.get("search_strategy", "")).strip().lower()
        staged_enabled = (
            isinstance(staged, dict)
            and bool(staged.get("enabled", False))
        ) or strategy in {"staged_axes", "axis_staged", "three_axis"}
        canonical_task = str(self.task_card.canonical_task or self.task_card.task_id).lower()
        if staged_enabled:
            if canonical_task == "tts":
                return (
                    "Staged axes search is enabled for this F5-TTS task. Obey the "
                    "current Research request exactly: it asks for one axis at a "
                    "time. `[inference]` ideas may only change batch inference, "
                    "text cleanup, chunking, sampling, or post-processing. "
                    "`[fine_tune]` ideas must use `SURE_TTS_FINETUNE_WRAPPER` "
                    "without changing model structure. `[arch]` ideas must use "
                    "`SURE_TTS_ARCH_WRAPPER` with `action=arch_finetune_short` "
                    "and change at least one whitelisted structure field such as "
                    "depth, ff_mult, conv_layers, or qk_norm. When the current "
                    "stage sets `SURE_TTS_ARCH_INIT_MODE=scratch`, architecture "
                    "screening must train from scratch; final combinations use "
                    "`partial_load` warm-start."
                )
            if canonical_task == "asr":
                return (
                    "Staged axes search is enabled for this ASR task. Obey the "
                    "current Research request exactly: it asks for one axis at a "
                    "time. `[inference]` ideas reuse an existing checkpoint and "
                    "change decoding, normalization, hypothesis formatting, or "
                    "post-processing. `[fine_tune]` ideas may change training "
                    "strategy without changing model structure. `[arch]` ideas "
                    "must change Zipformer structure/parameter count by passing "
                    "at least one structure argument such as --num-encoder-layers, "
                    "--encoder-dim, --feedforward-dim, or --encoder-unmasked-dim."
                )
        if canonical_task == "asr":
            return (
                "Mixed execution is enabled for this ASR task. Produce exactly 4 "
                "`[inference]` ideas, 2 `[fine_tune]` ideas, and 2 `[arch]` ideas. "
                "`[inference]` ideas reuse an existing checkpoint and change decoding, "
                "normalization, hypothesis formatting, or post-processing. `[fine_tune]` "
                "ideas may change Zipformer training, loss, optimizer, data sampling, "
                "augmentation, or checkpoint creation without changing model structure. "
                "`[arch]` ideas must change Zipformer structure/parameter count by "
                "passing at least one of --num-encoder-layers, --encoder-dim, "
                "--feedforward-dim, or --encoder-unmasked-dim to the icefall train.py "
                "command. Fine-tune and arch ideas are submitted as VC child jobs "
                "according to `sure.remote_training`."
            )
        if canonical_task == "tts":
            return (
                "Mixed execution is enabled for this F5-TTS task. Produce exactly 4 "
                "`[inference]` ideas, 2 `[fine_tune]` ideas, and 2 `[arch]` ideas. "
                "`[inference]` ideas only change inference, text normalization, chunking, "
                "sampling, or post-processing. `[fine_tune]` ideas must use "
                "`SURE_TTS_FINETUNE_WRAPPER` and must not change model structure. "
                "`[arch]` ideas must use `SURE_TTS_ARCH_WRAPPER` with "
                "`action=arch_finetune_short` and must change at least one whitelisted "
                "structure field such as depth, ff_mult, conv_layers, or qk_norm. "
                "F5-TTS architecture ideas must stay within the wrapper whitelist and "
                "must not patch raw model source files."
            )
        return (
            "Mixed execution is enabled for this task. Produce exactly 4 `[inference]` "
            "ideas, 2 `[fine_tune]` ideas, and 2 `[arch]` ideas. `[inference]` only "
            "changes decoding, formatting, or post-processing; `[fine_tune]` trains "
            "without changing structure; `[arch]` must change model structure."
        )

    def _research_request_text(self) -> str:
        if self.research_request:
            return self.research_request
        return (
            "Propose the next candidate batch for improving the SURE primary metric. "
            "The batch must contain exactly:\n"
            "- 4 `[inference]` ideas.\n"
            "- 2 `[fine_tune]` ideas.\n"
            "- 2 `[arch]` ideas.\n\n"
            "`[inference]` ideas may change inference, decoding, thresholds, normalization, "
            "post-processing, artifact generation, or task-specific data handling without "
            "creating a new checkpoint.\n"
            "`[fine_tune]` ideas may change training, loss, optimizer, data sampling, "
            "augmentation, checkpoint creation, or fine-tuning settings, but must not "
            "change model structure or parameter count.\n"
            "`[arch]` ideas must change model structure or parameter count. They must not "
            "be mere learning-rate, epoch, batch-size, SpecAugment, beam, decoding, "
            "threshold, normalization, or post-processing ideas.\n\n"
            "Return JSON only in this shape:\n"
            "{\n"
            '  "inference": {\n'
            '    "1": "[inference] specific idea",\n'
            '    "2": "[inference] specific idea",\n'
            '    "3": "[inference] specific idea",\n'
            '    "4": "[inference] specific idea"\n'
            "  },\n"
            '  "fine_tune": {\n'
            '    "1": "[fine_tune] specific idea",\n'
            '    "2": "[fine_tune] specific idea"\n'
            "  },\n"
            '  "arch": {\n'
            '    "1": "[arch] specific idea",\n'
            '    "2": "[arch] specific idea"\n'
            "  }\n"
            "}"
        )
