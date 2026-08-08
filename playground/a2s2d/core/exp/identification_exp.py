"""Stage 2: Variable Identification Agent

Agent-based stage that:
1. Calls LLM to identify variable roles (outcome/explanatory/control_candidate)
2. Identifies non-response values (拒答/不知道/不适用)
3. Suggests measurement transformations for X and Y
4. Saves variable_role_table.csv and variable_identification_value_plan.csv

IMPORTANT: The agent must NOT see pre-regression P-values or significance.
           It must make decisions purely based on variable semantics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from evomaster.utils.types import TaskInstance

from . import A2S2DBaseExp
from ...lib.identification import apply_identification_cleaning
from ...lib.llm import build_provider
from ...lib.metadata import build_metadata
from ...lib.utils import records_frame


class IdentificationExp(A2S2DBaseExp):
    """Stage 2: Variable Identification Agent.

    This stage uses an LLM agent to identify variable roles and
    non-response values. The agent must NOT see pre-regression results.
    """

    def run(
        self,
        task_description: str,
        state: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict:
        """Execute Stage 2.

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
        logger.info("Starting Stage 2: Variable Identification Agent")

        try:
            if self.a2s2d_config is None:
                raise ValueError("A2S2D config is not initialized")
            if self.agent is None:
                raise ValueError("Agent is not initialized")

            # Get required state from Stage 1
            data = self.state.get("data")
            metadata = self.state.get("metadata")
            stata_meta = self.state.get("stata_meta")
            codebook = self.state.get("codebook")

            if data is None:
                raise ValueError("Missing 'data' in state (Stage 1 not completed)")
            if metadata is None:
                raise ValueError("Missing 'metadata' in state (Stage 1 not completed)")

            # Build LLM provider (using A2S2D's structured output API)
            # Note: We use the A2S2D llm config, not the EvoMaster agent
            provider = build_provider(
                self.a2s2d_config.llm,
                cache_dir=self.a2s2d_config.cache_root if self.a2s2d_config.cache.enabled else None,
            )

            # Prepare variable batch for agent
            # Convert metadata to dict format
            variables = [
                {
                    "name": vm.name,
                    "label": vm.label,
                    "question": vm.question,
                    "module": vm.module,
                    "level": vm.level,
                    "variable_type": vm.variable_type,
                    "dtype": vm.dtype,
                    "value_labels": vm.value_labels,
                    "missing_values": vm.missing_values,
                    "concept_family": vm.concept_family,
                    "derived_from": vm.derived_from,
                    "skip_condition": vm.skip_condition,
                    "scale_group": vm.scale_group,
                    "direction": vm.direction,
                    "years": vm.years,
                    "n": vm.n,
                    "n_missing": vm.n_missing,
                    "missing_rate": vm.missing_rate,
                    "n_unique": vm.n_unique,
                    "minimum": vm.minimum,
                    "maximum": vm.maximum,
                    "mean": vm.mean,
                    "std": vm.std,
                    "skew": vm.skew,
                    "notes": vm.notes,
                }
                for vm in metadata
            ]

            logger.info(f"Calling identification agent for {len(variables)} variables...")

            # Call agent through A2S2D's LLM provider (structured output)
            # This bypasses the EvoMaster agent loop and directly uses structured output
            # (which matches the original A2S2D design)
            #
            # Note: To use the EvoMaster agent instead, we would need to:
            # 1. Create a TaskInstance with the variable metadata
            # 2. Call self.agent.run(task)
            # 3. Parse the agent's output (through str_replace_editor tool)
            #
            # For now, we use the A2S2D's direct API approach for simplicity
            enriched = provider.enrich_variables(variables)

            # Convert enriched results to VariableRole objects
            roles = []
            for vm in metadata:
                role_data = enriched.get(vm.name)
                if role_data is None:
                    # Fallback: create default role
                    role_data = {
                        "meaning": vm.label or vm.name,
                        "variable_type": vm.variable_type,
                        "measurement_object": "unknown",
                        "is_original": True,
                        "is_derived": bool(vm.derived_from),
                        "has_skip_logic": bool(vm.skip_condition),
                        "mechanical_overlap_risk": False,
                        "concepts": [],
                        "roles": ["control_candidate"],
                        "unusable_risk": "low",
                        "semantic_confidence": 0.5,
                        "review_reason": "LLM did not return this variable",
                        "nonresponse_values": [],
                        "nonresponse_reason": "",
                        "missing_action": "keep",
                        "suggested_x_transform": "none",
                        "suggested_y_transform": "none",
                        "transform_reason": "",
                    }

                roles.append(
                    VariableRole(
                        name=vm.name,
                        meaning=role_data.get("meaning", ""),
                        variable_type=role_data.get("variable_type", vm.variable_type),
                        measurement_object=role_data.get("measurement_object", ""),
                        is_original=role_data.get("is_original", True),
                        is_derived=role_data.get("is_derived", False),
                        has_skip_logic=role_data.get("has_skip_logic", False),
                        mechanical_overlap_risk=role_data.get("mechanical_overlap_risk", False),
                        concepts=role_data.get("concepts", []),
                        roles=role_data.get("roles", []),
                        unusable_risk=role_data.get("unusable_risk", "low"),
                        semantic_confidence=role_data.get("semantic_confidence", 0.5),
                        review_reason=role_data.get("review_reason", ""),
                        concept_family=vm.concept_family,
                        role_source="llm_api",
                        mechanical_with=[],
                        mechanical_link_details=role_data.get("mechanical_links", []),
                        nonresponse_values=role_data.get("nonresponse_values", []),
                        nonresponse_reason=role_data.get("nonresponse_reason", ""),
                        missing_action=role_data.get("missing_action", "keep"),
                        suggested_x_transform=role_data.get("suggested_x_transform", "none"),
                        suggested_y_transform=role_data.get("suggested_y_transform", "none"),
                        transform_reason=role_data.get("transform_reason", ""),
                    )
                )

            logger.info(f"Identified roles for {len(roles)} variables")

            # Apply non-response value cleaning
            logger.info("Applying non-response value cleaning...")
            cleaned, value_plan, value_log = apply_identification_cleaning(
                data=data,
                metadata=metadata,
                roles=roles,
                config=self.a2s2d_config,
            )

            # Rebuild metadata with cleaned data
            logger.info("Rebuilding metadata with cleaned data...")
            cleaned_metadata = build_metadata(
                data=cleaned,
                stata_meta=stata_meta,
                codebook=codebook,
                config=self.a2s2d_config,
            )

            # Save results
            roles_df = records_frame(roles)
            roles_path = self._save_output("variable_role_table.csv", roles_df)

            value_plan_df = value_plan
            value_plan_path = self._save_output(
                "variable_identification_value_plan.csv",
                value_plan_df,
            )

            value_log_df = value_log
            value_log_path = self._save_output(
                "variable_identification_value_log.csv",
                value_log_df,
            )

            # Update state
            state_updates = {
                "roles": roles,
                "cleaned": cleaned,
                "cleaned_metadata": cleaned_metadata,
                "value_plan": value_plan,
                "value_log": value_log,
                "_completed_stage": 2,
            }

            logger.info(f"Stage 2 completed: {roles_path}")

            return {
                "status": "completed",
                "state_updates": state_updates,
                "roles_file": str(roles_path),
                "value_plan_file": str(value_plan_path),
                "value_log_file": str(value_log_path),
                "variables_identified": len(roles),
            }

        except Exception as e:
            logger.error(f"Stage 2 failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
            }


# Import VariableRole from A2S2D library
from ...lib import VariableRole