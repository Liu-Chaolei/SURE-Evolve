"""Stage 7: Robustness & Follow-up

Pure program stage that:
1. Runs robustness checks (alternative models, control sensitivity)
2. Runs subgroup analysis
3. Runs interaction analysis
4. Runs mediation analysis
5. Saves robustness_table.csv and followup_table.csv
"""

from __future__ import annotations

import logging
import pandas as pd
from typing import Any

from . import A2S2DBaseExp
from ...lib.robustness import run_robustness
from ...lib.followup import execute_followup_actions
from ...lib.utils import records_frame


class RobustnessExp(A2S2DBaseExp):
    """Stage 7: Robustness & Follow-up.

    This is a pure-program stage that does not use an LLM agent.
    It runs robustness checks and follow-up analyses.
    """

    def run(
        self,
        task_description: str,
        state: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict:
        """Execute Stage 7.

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
        logger.info("Starting Stage 7: Robustness & Follow-up")

        try:
            if self.a2s2d_config is None:
                raise ValueError("A2S2D config is not initialized")

            # Get required state from previous stages
            data = self.state.get("cleaned")
            if data is None:
                data = self.state.get("data")
            models = self.state.get("models")
            candidates = self.state.get("candidates")
            audit_results = self.state.get("audit_results")
            metadata = self.state.get("cleaned_metadata")
            if metadata is None:
                metadata = self.state.get("metadata")

            if data is None:
                raise ValueError("Missing 'data' in state (Stage 1 not completed)")
            if models is None:
                raise ValueError("Missing 'models' in state (Stage 5 not completed)")
            if candidates is None:
                raise ValueError("Missing 'candidates' in state (Stage 3 not completed)")
            if audit_results is None:
                raise ValueError("Missing 'audit_results' in state (Stage 6 not completed)")
            if metadata is None:
                raise ValueError("Missing 'metadata' in state (Stage 1 not completed)")

            # Convert to DataFrame
            models_df = records_frame(models)
            audit_df = pd.DataFrame(audit_results)

            # Run robustness checks (using A2S2D's exact signature)
            logger.info("Running robustness checks...")
            robustness_df = run_robustness(
                data=data,
                candidates=candidates,
                audits=audit_df,
                metadata=metadata,
                config=self.a2s2d_config,
            )
            logger.info(f"Completed {len(robustness_df)} robustness checks")

            # Save robustness results
            robustness_path = self._save_output(
                "robustness_table.csv",
                robustness_df,
            )

            # Run follow-up analyses (using A2S2D's exact signature)
            logger.info("Running follow-up analyses...")
            followup_df = execute_followup_actions(
                data=data,
                candidates=candidates,
                model_results=models_df,
                metadata=metadata,
                config=self.a2s2d_config,
            )
            logger.info(f"Completed {len(followup_df)} follow-up analyses")

            # Save follow-up results
            followup_path = self._save_output(
                "followup_table.csv",
                followup_df,
            )

            # Update state
            state_updates = {
                "robustness_results": robustness_df.to_dict(orient="records"),
                "followup_results": followup_df.to_dict(orient="records"),
                "_completed_stage": 7,
            }

            logger.info(f"Stage 7 completed: {robustness_path}, {followup_path}")

            return {
                "status": "completed",
                "state_updates": state_updates,
                "robustness_file": str(robustness_path),
                "followup_file": str(followup_path),
                "robustness_checks": len(robustness_df),
                "followup_analyses": len(followup_df),
            }

        except Exception as e:
            logger.error(f"Stage 7 failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
            }