"""Stage 8: Literature Search & Assessment

Agent-based stage that:
1. Queries OpenAlex for each research lead
2. LLM assesses relevance (directly_studied/closely_related/etc.)
3. Saves literature_assessment_table.csv

This stage uses OpenAlex API + LLM judgment to find prior work.
"""

from __future__ import annotations

import logging
import pandas as pd
from typing import Any

from . import A2S2DBaseExp
from ...lib.literature import run_literature_audit
from ...lib.utils import records_frame


class LiteratureExp(A2S2DBaseExp):
    """Stage 8: Literature Search & Assessment.

    This stage uses an LLM agent + OpenAlex API to:
    1. Search for prior work on each research lead
    2. Assess relevance and novelty
    """

    def run(
        self,
        task_description: str,
        state: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict:
        """Execute Stage 8.

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
        logger.info("Starting Stage 8: Literature Search & Assessment")

        try:
            if self.a2s2d_config is None:
                raise ValueError("A2S2D config is not initialized")
            if not self.a2s2d_config.literature.enabled:
                logger.info("Literature search disabled, skipping Stage 8")
                return {
                    "status": "completed",
                    "state_updates": {"_completed_stage": 8},
                    "literature_enabled": False,
                }

            # Get required state from previous stages
            audit_results = self.state.get("audit_results")
            models = self.state.get("models")
            candidates = self.state.get("candidates")
            roles = self.state.get("roles")
            metadata = self.state.get("cleaned_metadata")
            if metadata is None:
                metadata = self.state.get("metadata")

            if audit_results is None:
                raise ValueError("Missing 'audit_results' in state (Stage 6 not completed)")
            if models is None:
                raise ValueError("Missing 'models' in state (Stage 5 not completed)")
            if candidates is None:
                raise ValueError("Missing 'candidates' in state (Stage 3 not completed)")
            if roles is None:
                raise ValueError("Missing 'roles' in state (Stage 2 not completed)")
            if metadata is None:
                raise ValueError("Missing 'metadata' in state (Stage 1 not completed)")

            # Build LLM provider
            from ...lib.llm import build_provider
            provider = build_provider(
                self.a2s2d_config.llm,
                cache_dir=self.a2s2d_config.cache_root if self.a2s2d_config.cache.enabled else None,
            )

            # Convert to DataFrame
            models_df = records_frame(models)
            audit_df = pd.DataFrame(audit_results)

            # Run literature audit (using A2S2D's exact signature)
            logger.info("Running literature audit...")
            literature_df, matched_df, report = run_literature_audit(
                audits=audit_df,
                model_results=models_df,
                candidates=candidates,
                metadata=metadata,
                roles=roles,
                config=self.a2s2d_config,
                provider=provider,
            )
            logger.info(f"Completed literature assessment for {len(literature_df)} leads")

            # Save results
            literature_path = self._save_output(
                "literature_assessment_table.csv",
                literature_df,
            )
            matched_path = self._save_output(
                "literature_matched_works.csv",
                matched_df,
            )
            report_path = self._save_output(
                "literature_report.md",
                report,
            )

            # Identify final leads (pass audit + literature check)
            final_leads = literature_df[
                literature_df["lead_class"].isin(["lead", "exploratory_lead"])
            ].to_dict(orient="records")

            # Update state
            state_updates = {
                "literature_results": literature_df.to_dict(orient="records"),
                "final_leads": final_leads,
                "_completed_stage": 8,
            }

            logger.info(f"Stage 8 completed: {literature_path}")
            logger.info(f"Final research leads: {len(final_leads)}")

            return {
                "status": "completed",
                "state_updates": state_updates,
                "literature_file": str(literature_path),
                "matched_file": str(matched_path),
                "report_file": str(report_path),
                "leads_assessed": len(literature_df),
                "final_leads": len(final_leads),
            }

        except Exception as e:
            logger.error(f"Stage 8 failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
            }