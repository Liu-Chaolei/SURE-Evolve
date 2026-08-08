"""Stage 1: Input & Initial Metadata

Pure program stage that:
1. Loads data file (Stata/CSV/Excel)
2. Loads codebook (Excel/CSV/JSON)
3. Builds initial variable metadata
4. Saves variable_metadata_before_identification.csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from . import A2S2DBaseExp
from ...lib.io import load_data, load_codebook
from ...lib.metadata import build_metadata, metadata_frame
from ...lib.utils import file_sha256


class InputExp(A2S2DBaseExp):
    """Stage 1: Input & Initial Metadata.

    This is a pure-program stage that does not use an LLM agent.
    It loads the data and codebook, then builds initial variable metadata.
    """

    def run(
        self,
        task_description: str,
        state: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict:
        """Execute Stage 1.

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
        logger.info("Starting Stage 1: Input & Initial Metadata")

        try:
            if self.a2s2d_config is None:
                raise ValueError("A2S2D config is not initialized")

            # 1. Load data
            logger.info(f"Loading data: {self.a2s2d_config.inputs.data}")
            data, stata_meta = load_data(self.a2s2d_config.data_path)
            logger.info(f"Loaded {len(data)} rows, {len(data.columns)} columns")

            # 2. Load codebook
            logger.info(f"Loading codebook: {self.a2s2d_config.inputs.codebook}")
            codebook = load_codebook(self.a2s2d_config.codebook_path)
            logger.info(f"Loaded codebook with {len(codebook)} variables")

            # 3. Build initial metadata
            logger.info("Building initial variable metadata...")
            metadata = build_metadata(
                data=data,
                stata_meta=stata_meta,
                codebook=codebook,
                config=self.a2s2d_config,
            )
            logger.info(f"Built metadata for {len(metadata)} variables")

            # 4. Save metadata
            metadata_df = metadata_frame(metadata)
            output_path = self._save_output(
                "variable_metadata_before_identification.csv",
                metadata_df,
            )

            # 5. Compute data hash
            data_sha256 = file_sha256(self.a2s2d_config.data_path)

            # 6. Update state
            state_updates = {
                "data": data,
                "stata_meta": stata_meta,
                "codebook": codebook,
                "metadata": metadata,
                "data_sha256": data_sha256,
                "_completed_stage": 1,
            }

            logger.info(f"Stage 1 completed: {output_path}")

            return {
                "status": "completed",
                "state_updates": state_updates,
                "output_file": str(output_path),
                "data_rows": len(data),
                "variables": len(metadata),
            }

        except Exception as e:
            logger.error(f"Stage 1 failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
            }