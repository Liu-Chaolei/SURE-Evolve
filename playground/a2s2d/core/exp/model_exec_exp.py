"""Stage 5: Model Execution

Pure program stage that:
1. Executes the fixed model registry for each candidate relation
2. Runs diagnostics and validation
3. Saves model_result_table.csv
"""

from __future__ import annotations

import logging
from typing import Any

from . import A2S2DBaseExp
from ...lib.modeling import run_registry
from ...lib.utils import records_frame


class ModelExecExp(A2S2DBaseExp):
    """Stage 5: Model Execution.

    This is a pure-program stage that does not use an LLM agent.
    It executes the fixed model registry for all candidate relations.
    """

    def run(
        self,
        task_description: str,
        state: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict:
        """Execute Stage 5.

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
        logger.info("Starting Stage 5: Model Execution")

        try:
            if self.a2s2d_config is None:
                raise ValueError("A2S2D config is not initialized")

            # Get required state from previous stages
            data = self.state.get("cleaned")
            if data is None:
                data = self.state.get("data")
            candidates = self.state.get("candidates")
            metadata = self.state.get("cleaned_metadata")
            if metadata is None:
                metadata = self.state.get("metadata")

            if data is None:
                raise ValueError("Missing 'data' in state (Stage 1 not completed)")
            if candidates is None:
                raise ValueError("Missing 'candidates' in state (Stage 3 not completed)")
            if metadata is None:
                raise ValueError("Missing 'metadata' in state (Stage 1 not completed)")

            # Run model registry (using A2S2D's exact signature)
            logger.info("Executing model registry...")
            models = run_registry(
                data=data,
                candidates=candidates,
                metadata=metadata,
                config=self.a2s2d_config,
            )
            logger.info(f"Executed {len(models)} models")

            # Save results
            models_df = records_frame(models)
            output_path = self._save_output(
                "model_result_table.csv",
                models_df,
            )

            # Update state
            state_updates = {
                "models": models,
                "_completed_stage": 5,
            }

            logger.info(f"Stage 5 completed: {output_path}")

            return {
                "status": "completed",
                "state_updates": state_updates,
                "output_file": str(output_path),
                "models_executed": len(models),
            }

        except Exception as e:
            logger.error(f"Stage 5 failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
            }