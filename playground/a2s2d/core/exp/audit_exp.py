"""Stage 6: Objective Evidence Audit

Pure program stage that:
1. Applies FDR correction (global and family-level)
2. Runs permutation tests for null hypothesis
3. Runs bootstrap for direction and significance rates
4. Computes objective confidence scores
5. Saves discovery_audit_table.csv
"""

from __future__ import annotations

import logging
import pandas as pd
from typing import Any

from . import A2S2DBaseExp
from ...lib.audit import run_discovery_audit
from ...lib.utils import records_frame


class AuditExp(A2S2DBaseExp):
    """Stage 6: Objective Evidence Audit.

    This is a pure-program stage that does not use an LLM agent.
    It applies FDR correction, permutation tests, and bootstrap analysis.
    """

    def run(
        self,
        task_description: str,
        state: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict:
        """Execute Stage 6.

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
        logger.info("Starting Stage 6: Objective Evidence Audit")

        try:
            if self.a2s2d_config is None:
                raise ValueError("A2S2D config is not initialized")

            # Get required state from previous stages
            data = self.state.get("cleaned")
            if data is None:
                data = self.state.get("data")
            models = self.state.get("models")
            candidates = self.state.get("candidates")
            metadata = self.state.get("cleaned_metadata")
            if metadata is None:
                metadata = self.state.get("metadata")
            roles = self.state.get("roles")

            if data is None:
                raise ValueError("Missing 'data' in state (Stage 1 not completed)")
            if models is None:
                raise ValueError("Missing 'models' in state (Stage 5 not completed)")
            if candidates is None:
                raise ValueError("Missing 'candidates' in state (Stage 3 not completed)")
            if metadata is None:
                raise ValueError("Missing 'metadata' in state (Stage 1 not completed)")
            if roles is None:
                raise ValueError("Missing 'roles' in state (Stage 2 not completed)")

            # Convert models to DataFrame (A2S2D expects DataFrame)
            models_df = records_frame(models)

            # Run discovery audit (using A2S2D's exact signature)
            logger.info("Running discovery audit...")
            audit_df, permutation_df, bootstrap_df, group_df = run_discovery_audit(
                data=data,
                candidates=candidates,
                model_results=models_df,
                metadata=metadata,
                roles=roles,
                config=self.a2s2d_config,
            )
            logger.info(f"Audited {len(audit_df)} relations")

            # Save results
            audit_path = self._save_output("discovery_audit_table.csv", audit_df)
            permutation_path = self._save_output("permutation_table.csv", permutation_df)
            bootstrap_path = self._save_output("bootstrap_table.csv", bootstrap_df)
            group_path = self._save_output("group_stability_table.csv", group_df)

            # Update state
            state_updates = {
                "audit_results": audit_df.to_dict(orient="records"),
                "_completed_stage": 6,
            }

            logger.info(f"Stage 6 completed: {audit_path}")

            return {
                "status": "completed",
                "state_updates": state_updates,
                "output_file": str(audit_path),
                "relations_audited": len(audit_df),
            }

        except Exception as e:
            logger.error(f"Stage 6 failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
            }