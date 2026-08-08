"""Stage 4: Model Design & Direction Adjudication

Agent-based stage that calls the model selection agent (via A2S2D's
build_model_selection_plan) to design each candidate relation, then runs
the reverse-direction plans. Uses A2S2D's exact implementation to ensure
correct framework validation and direction adjudication.

IMPORTANT CONSTRAINTS:
- Agent must NOT see pre-regression P-values or significance
- Direction decision uses FIXED RULES (not agent preference)
"""

from __future__ import annotations

import logging
from typing import Any

from . import A2S2DBaseExp
from ...lib.llm import build_provider
from ...lib.model_selection import (
    build_model_selection_plan,
    build_reverse_model_selection_plans,
)
from ...lib.utils import records_frame


class ModelDesignExp(A2S2DBaseExp):
    """Stage 4: Model Design & Direction Adjudication.

    This stage uses A2S2D's build_model_selection_plan to:
    1. Select model family (OLS, logit, etc.)
    2. Choose control variables (with framework validation)
    3. Suggest direction (X→Y vs Y→X) using fixed rules

    IMPORTANT: The agent must NOT see pre-regression results or P-values.
    """

    def run(
        self,
        task_description: str,
        state: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict:
        """Execute Stage 4.

        Args:
            task_description: Task description.
            state: Global state dict.
            **kwargs: Additional arguments.

        Returns:
            Result dict with 'status' and 'state_updates'.
        """
        if state is not None:
            self.state = state

        logger = logging.getLogger(self.__class__.__name__)
        logger.info("Starting Stage 4: Model Design & Direction Adjudication")

        try:
            if self.a2s2d_config is None:
                raise ValueError("A2S2D config is not initialized")
            if self.agent is None:
                raise ValueError("Agent is not initialized")

            # Get required state from previous stages
            data = self.state.get("cleaned")
            if data is None:
                data = self.state.get("data")
            candidates = self.state.get("candidates")
            roles = self.state.get("roles")
            metadata = self.state.get("cleaned_metadata")
            if metadata is None:
                metadata = self.state.get("metadata")

            if data is None:
                raise ValueError("Missing 'data' in state (Stage 1 not completed)")
            if candidates is None:
                raise ValueError("Missing 'candidates' in state (Stage 3 not completed)")
            if roles is None:
                raise ValueError("Missing 'roles' in state (Stage 2 not completed)")
            if metadata is None:
                raise ValueError("Missing 'metadata' in state (Stage 1 not completed)")

            # Build LLM provider using A2S2D's ChatCompletions/Responses provider
            provider = build_provider(
                self.a2s2d_config.llm,
                cache_dir=self.a2s2d_config.cache_root if self.a2s2d_config.cache.enabled else None,
            )

            # Run A2S2D's build_model_selection_plan (does framework validation,
            # control decisions, direction adjudication with fixed rules)
            logger.info(f"Calling model selection agent for {len(candidates)} relations...")
            model_selection = build_model_selection_plan(
                data=data,
                candidates=candidates,
                metadata=metadata,
                roles=roles,
                config=self.a2s2d_config,
                provider=provider,
            )

            # Run reverse-direction plans for relations where reverse is recommended
            logger.info("Building reverse-direction model selection plans...")
            reverse_model_selection = build_reverse_model_selection_plans(
                data=data,
                candidates=candidates,
                metadata=metadata,
                roles=roles,
                config=self.a2s2d_config,
                provider=provider,
            )

            logger.info(
                f"Received designs for {len(model_selection)} relations "
                f"({len(reverse_model_selection)} reverse plans)"
            )

            # Save model selection plan
            self._save_output(
                "proposed_model_selection_plan.csv",
                model_selection,
            )
            if not reverse_model_selection.empty:
                self._save_output(
                    "proposed_reverse_model_selection_plan.csv",
                    reverse_model_selection,
                )

            # Save updated candidate relation table (with designs applied)
            candidates_df = records_frame(candidates)
            self._save_output(
                "candidate_relation_table.csv",
                candidates_df,
            )

            # Update state
            state_updates = {
                "candidates": candidates,  # Updated with designs
                "model_selection": model_selection,
                "reverse_model_selection": reverse_model_selection,
                "_completed_stage": 4,
            }

            logger.info("Stage 4 completed")

            return {
                "status": "completed",
                "state_updates": state_updates,
                "relations_designed": len(model_selection),
                "reverse_plans": len(reverse_model_selection),
            }

        except Exception as e:
            logger.error(f"Stage 4 failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
            }