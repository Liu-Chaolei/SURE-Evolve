"""Stage 3: Candidate Relation Generation

Pure program stage that:
1. Generates candidate X-Y relations from metadata
2. Excludes mechanical relations
3. Applies stratified rotation by Y
4. Saves candidate_relation_table.csv
"""

from __future__ import annotations

import logging
from typing import Any

from . import A2S2DBaseExp
from ...lib.candidates import generate_candidates
from ...lib.utils import records_frame


class CandidateExp(A2S2DBaseExp):
    """Stage 3: Candidate Relation Generation.

    This is a pure-program stage that does not use an LLM agent.
    It generates candidate relations from the variable metadata.
    """

    def run(
        self,
        task_description: str,
        state: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict:
        """Execute Stage 3.

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
        logger.info("Starting Stage 3: Candidate Relation Generation")

        try:
            if self.a2s2d_config is None:
                raise ValueError("A2S2D config is not initialized")

            # Get required state from previous stages
            data = self.state.get("cleaned")
            if data is None:
                data = self.state.get("data")
            metadata = self.state.get("cleaned_metadata")
            if metadata is None:
                metadata = self.state.get("metadata")
            roles = self.state.get("roles")

            if data is None:
                raise ValueError("Missing 'data' in state (Stage 1 not completed)")
            if metadata is None:
                raise ValueError("Missing 'metadata' in state (Stage 1 not completed)")
            if roles is None:
                raise ValueError("Missing 'roles' in state (Stage 2 not completed)")

            # Generate candidates (using A2S2D's exact signature)
            logger.info("Generating candidate relations...")
            candidates = generate_candidates(
                data=data,
                metadata=metadata,
                roles=roles,
                max_candidates=self.a2s2d_config.modeling.max_candidates,
                seed=self.a2s2d_config.project.seed,
                min_unique_values=self.a2s2d_config.identification.min_unique_values,
                max_missing_rate=self.a2s2d_config.identification.max_missing_rate,
            )
            logger.info(f"Generated {len(candidates)} candidate relations")

            # Save candidates
            candidates_df = records_frame(candidates)
            output_path = self._save_output(
                "candidate_relation_table.csv",
                candidates_df,
            )

            # Update state
            state_updates = {
                "candidates": candidates,
                "_completed_stage": 3,
            }

            logger.info(f"Stage 3 completed: {output_path}")

            return {
                "status": "completed",
                "state_updates": state_updates,
                "output_file": str(output_path),
                "candidates": len(candidates),
            }

        except Exception as e:
            logger.error(f"Stage 3 failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
            }