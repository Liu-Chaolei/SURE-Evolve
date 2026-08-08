"""A2S2D Stage Experiments

This module defines the 8 stage experiments for the A2S2D pipeline.
Each stage is a BaseExp subclass that executes a specific part of the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from evomaster.core.exp import BaseExp

# Import A2S2D library (from playground/a2s2d/lib)
from ...lib import ProjectConfig


class A2S2DBaseExp(BaseExp):
    """Base class for A2S2D stage experiments.

    Extends BaseExp with:
    - Support for ProjectConfig (A2S2D-specific config)
    - State dict for passing data between stages
    - Helper methods for loading/saving stage outputs
    """

    def __init__(
        self,
        agent: Any | None,
        config: ProjectConfig | None,
        run_dir: Path | None = None,
        state: dict[str, Any] | None = None,
    ):
        """Initialize the experiment.

        Args:
            agent: Agent instance (None for pure-program stages).
            config: A2S2D ProjectConfig instance.
            run_dir: Run directory path.
            state: Global state dict (shared across stages).
        """
        # Note: BaseExp expects an EvoMaster config, but we pass ProjectConfig
        # We'll handle this by storing it separately
        self.agent = agent
        self.a2s2d_config = config
        self.run_dir = run_dir
        self.state = state or {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # Results storage
        self.results = []

    @property
    def exp_name(self) -> str:
        """Get the Exp name (derived from class name)."""
        class_name = self.__class__.__name__
        if class_name.endswith("Exp"):
            return class_name[:-3]
        return class_name

    def set_run_dir(self, run_dir: str | Path) -> None:
        """Set the run directory."""
        self.run_dir = Path(run_dir)

    def run(
        self,
        task_description: str,
        state: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict:
        """Run the experiment.

        Args:
            task_description: Task description.
            state: Global state dict (optional, updates self.state).
            **kwargs: Additional arguments.

        Returns:
            Result dict with 'status', 'state_updates', etc.

        Note: This method should be overridden by subclasses.
        """
        if state is not None:
            self.state = state

        # Default implementation (should be overridden)
        return {
            "status": "completed",
            "state_updates": {},
        }

    def _output_dir(self) -> Path:
        """Get the output directory for this stage."""
        if self.run_dir is None:
            return Path("outputs")
        return self.run_dir / "logs"

    def _save_output(self, filename: str, data: Any) -> Path:
        """Save output data to file.

        Args:
            filename: Output filename (e.g., "variable_metadata.csv").
            data: Data to save (DataFrame, dict, list).

        Returns:
            Path to saved file.
        """
        output_dir = self._output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / filename

        # Handle different data types
        if hasattr(data, "to_csv"):
            # pandas DataFrame
            data.to_csv(output_path, index=False, encoding="utf-8-sig")
        elif isinstance(data, (dict, list)):
            # JSON-serializable
            from ...lib.utils import write_json
            write_json(output_path, data)
        else:
            # Plain text
            output_path.write_text(str(data), encoding="utf-8")

        self.logger.info(f"Saved output: {output_path}")
        return output_path


# Import all stage experiments
from .input_exp import InputExp
from .identification_exp import IdentificationExp
from .candidate_exp import CandidateExp
from .model_design_exp import ModelDesignExp
from .model_exec_exp import ModelExecExp
from .audit_exp import AuditExp
from .robustness_exp import RobustnessExp
from .literature_exp import LiteratureExp


__all__ = [
    "A2S2DBaseExp",
    "InputExp",
    "IdentificationExp",
    "CandidateExp",
    "ModelDesignExp",
    "ModelExecExp",
    "AuditExp",
    "RobustnessExp",
    "LiteratureExp",
]