"""A2S2D Playground - Social Science Research Discovery Pipeline

This playground implements the A2S2D (Auditable Agent-Assisted Social Science Discovery)
system as an EvoMaster playground, with 8 stages:

1. Input & Initial Metadata (纯程序)
2. Variable Identification Agent (LLM)
3. Candidate Relation Generation (纯程序)
4. Model Design & Direction Adjudication (LLM)
5. Model Execution (纯程序)
6. Objective Evidence Audit (纯程序)
7. Robustness & Follow-up (纯程序)
8. Literature Search & Assessment (LLM)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

from evomaster.core import BasePlayground, register_playground
from evomaster.utils.types import TaskInstance

# Import A2S2D library (vendored)
from ..lib import (
    ProjectConfig,
    VariableMetadata,
    VariableRole,
    CandidateRelation,
    ModelResult,
    AuditResult,
)
from ..lib.utils import file_sha256, write_json

# Import stage experiments
from .exp import (
    InputExp,
    IdentificationExp,
    CandidateExp,
    ModelDesignExp,
    ModelExecExp,
    AuditExp,
    RobustnessExp,
    LiteratureExp,
)


PIPELINE_SCHEMA_VERSION = "2026-06-26.1"


@register_playground("a2s2d")
class A2S2DPlayground(BasePlayground):
    """Main orchestrator for A2S2D social science research discovery pipeline.

    This playground wraps the A2S2D library and provides:
    - 8-stage pipeline execution
    - JSON checkpoint save/load with schema validation
    - Resume from interruption
    - Multi-agent support (variable_identification, model_selection, literature)
    """

    def __init__(self, config_dir: Path | None = None, config_path: Path | None = None):
        """Initialize the A2S2D playground.

        Args:
            config_dir: Configuration directory (defaults to configs/a2s2d/).
            config_path: Full path to the configuration file (overrides config_dir).
        """
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Declare agents
        self.agents.declare(
            "variable_identification_agent",
            "model_selection_agent",
            "literature_agent",
        )

        # Pipeline state
        self.state: dict[str, Any] = {}
        self.current_stage: int = 0
        self.pipeline_schema_version = PIPELINE_SCHEMA_VERSION

        # A2S2D-specific config (extracted from YAML)
        self.a2s2d_config: ProjectConfig | None = None

    def setup(self) -> None:
        """Initialize session, agents, and A2S2D config."""
        self.logger.info("Setting up A2S2D playground...")

        # Build A2S2D ProjectConfig from YAML config
        self._build_a2s2d_config()

        # Setup session and agents (from BasePlayground)
        self._setup_session()
        self._setup_agents()

        # Setup workspace symlinks for data files
        self._setup_data_symlinks()

        self.logger.info("A2S2D playground setup complete")

    def _build_a2s2d_config(self) -> None:
        """Build A2S2D ProjectConfig from EvoMaster YAML config.

        The YAML config contains both EvoMaster-style fields (llm, agents, session)
        and A2S2D-specific fields (inputs, columns, modeling, etc.). This method
        extracts the A2S2D fields and constructs a ProjectConfig.
        """
        # Load the raw YAML dict (before EvoMasterConfig validation)
        import yaml

        config_path = Path(self.config_path) if self.config_path else Path(self.config_dir) / "config.yaml"

        try:
            with open(config_path) as f:
                raw_yaml = yaml.safe_load(f)

            # Build ProjectConfig from raw YAML dict
            self.a2s2d_config = ProjectConfig.from_dict(
                raw_yaml,
                config_path=config_path,
                require_api=False,  # Don't require API for setup
            )
            self.logger.info(f"A2S2D config built: {self.a2s2d_config.project.name}")
        except Exception as e:
            import traceback
            self.logger.error(f"Failed to build A2S2D config: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            self.a2s2d_config = None

    def _setup_data_symlinks(self) -> None:
        """Setup symlinks to data files in workspace.

        This allows A2S2D code to access data files from the workspace directory.
        """
        if self.a2s2d_config is None:
            return

        # Check a2s2d_options for symlink settings
        a2s2d_options = getattr(self.config, "a2s2d_options", {})
        if not a2s2d_options.get("data_symlinks", False):
            return

        workspace_path = Path(self.session.config.workspace_path)
        workspace_path.mkdir(parents=True, exist_ok=True)

        # Symlink data file
        data_path = self.a2s2d_config.data_path
        if data_path.exists():
            link_path = workspace_path / "data" / data_path.name
            link_path.parent.mkdir(parents=True, exist_ok=True)
            if not link_path.exists():
                link_path.symlink_to(data_path)
                self.logger.info(f"Symlinked data: {link_path} -> {data_path}")

        # Symlink codebook file
        codebook_path = self.a2s2d_config.codebook_path
        if codebook_path.exists():
            link_path = workspace_path / "data" / codebook_path.name
            link_path.parent.mkdir(parents=True, exist_ok=True)
            if not link_path.exists():
                link_path.symlink_to(codebook_path)
                self.logger.info(f"Symlinked codebook: {link_path} -> {codebook_path}")

    def run(
        self,
        task_description: str,
        output_file: str | None = None,
        resume_dir: str | Path | None = None,
        **kwargs,
    ) -> dict:
        """Execute the full A2S2D pipeline.

        Args:
            task_description: Task description (e.g., "Discover research leads from CGSS2023").
            output_file: Optional path to save trajectory output.
            resume_dir: Optional directory to resume from checkpoint.

        Returns:
            Result dict with 'status', 'stages_completed', etc.
        """
        try:
            # Register current thread (for log filtering)
            self.register_thread()

            # Setup if not already done
            if self.session is None:
                self.setup()

            # Handle resume
            if resume_dir is not None:
                self.state = self._load_checkpoint(Path(resume_dir))
                self.current_stage = self._resolve_next_stage()
                self.logger.info(f"Resuming from stage {self.current_stage}")
            else:
                # New run: initialize state
                self.state = {}
                self.current_stage = 1

            # Setup trajectory file
            self._setup_trajectory_file(output_file)

            # Execute stages
            while self.current_stage <= 8:
                self.logger.info(f"=" * 80)
                self.logger.info(f"Stage {self.current_stage}: {self._stage_name(self.current_stage)}")
                self.logger.info(f"=" * 80)

                exp = self._create_exp_by_stage(self.current_stage)
                result = exp.run(task_description, state=self.state, **kwargs)

                # Update state
                if result.get("status") == "completed":
                    self.state.update(result.get("state_updates", {}))
                    self._save_checkpoint(self.current_stage)
                    self.current_stage += 1
                else:
                    # Stage failed
                    self.logger.error(f"Stage {self.current_stage} failed: {result}")
                    self._save_checkpoint(self.current_stage, status="failed")
                    return {
                        "status": "failed",
                        "failed_stage": self.current_stage,
                        "error": result.get("error"),
                    }

            # All stages completed
            self.logger.info("=" * 80)
            self.logger.info("✅ A2S2D pipeline completed successfully")
            self.logger.info("=" * 80)

            # Save final manifest
            self._save_final_manifest()

            return {
                "status": "completed",
                "stages_completed": 8,
                "run_dir": str(self.run_dir),
            }

        except Exception as e:
            self.logger.error(f"A2S2D pipeline failed: {e}", exc_info=True)
            self._save_checkpoint(self.current_stage, status="failed", error=str(e))
            return {
                "status": "failed",
                "failed_stage": self.current_stage,
                "error": str(e),
            }

        finally:
            self.cleanup()

    def _stage_name(self, stage: int) -> str:
        """Get human-readable name for a stage."""
        names = {
            1: "Input & Initial Metadata",
            2: "Variable Identification Agent",
            3: "Candidate Relation Generation",
            4: "Model Design & Direction Adjudication",
            5: "Model Execution",
            6: "Objective Evidence Audit",
            7: "Robustness & Follow-up",
            8: "Literature Search & Assessment",
        }
        return names.get(stage, f"Unknown Stage {stage}")

    def _create_exp_by_stage(self, stage: int) -> Any:
        """Create the appropriate Exp for a stage.

        Args:
            stage: Stage number (1-8).

        Returns:
            BaseExp subclass instance.
        """
        if stage == 1:
            return InputExp(
                agent=None,  # No agent needed
                config=self.a2s2d_config,
                run_dir=self.run_dir,
                state=self.state,
            )
        elif stage == 2:
            return IdentificationExp(
                agent=self.agents.variable_identification_agent,
                config=self.a2s2d_config,
                run_dir=self.run_dir,
                state=self.state,
            )
        elif stage == 3:
            return CandidateExp(
                agent=None,
                config=self.a2s2d_config,
                run_dir=self.run_dir,
                state=self.state,
            )
        elif stage == 4:
            return ModelDesignExp(
                agent=self.agents.model_selection_agent,
                config=self.a2s2d_config,
                run_dir=self.run_dir,
                state=self.state,
            )
        elif stage == 5:
            return ModelExecExp(
                agent=None,
                config=self.a2s2d_config,
                run_dir=self.run_dir,
                state=self.state,
            )
        elif stage == 6:
            return AuditExp(
                agent=None,
                config=self.a2s2d_config,
                run_dir=self.run_dir,
                state=self.state,
            )
        elif stage == 7:
            return RobustnessExp(
                agent=None,
                config=self.a2s2d_config,
                run_dir=self.run_dir,
                state=self.state,
            )
        elif stage == 8:
            return LiteratureExp(
                agent=self.agents.literature_agent,
                config=self.a2s2d_config,
                run_dir=self.run_dir,
                state=self.state,
            )
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def _save_checkpoint(
        self,
        stage: int,
        status: str = "completed",
        error: str | None = None,
    ) -> None:
        """Save checkpoint to run_manifest.json.

        Args:
            stage: Completed stage number.
            status: Stage status ("completed" or "failed").
            error: Optional error message.
        """
        if self.run_dir is None:
            return

        manifest_path = self.run_dir / "run_manifest.json"

        # Build serializable state (exclude non-serializable objects)
        serializable_state = self._serialize_state(self.state)

        manifest = {
            "pipeline_schema_version": self.pipeline_schema_version,
            "current_stage": stage,
            "status": status,
            "state": serializable_state,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "config_fingerprint": (
                self.a2s2d_config.fingerprint()
                if self.a2s2d_config
                else None
            ),
        }

        if error:
            manifest["error"] = error

        # Add data file hashes
        if self.a2s2d_config:
            try:
                manifest["data_sha256"] = file_sha256(self.a2s2d_config.data_path)
                manifest["codebook_sha256"] = file_sha256(
                    self.a2s2d_config.codebook_path
                )
            except Exception:
                pass

        write_json(manifest_path, manifest)
        self.logger.info(f"Checkpoint saved: stage {stage} ({status})")

    def _load_checkpoint(self, resume_dir: Path) -> dict:
        """Load checkpoint from run_manifest.json.

        Args:
            resume_dir: Directory containing the checkpoint.

        Returns:
            Loaded state dict.

        Raises:
            ValueError: If checkpoint is invalid or incompatible.
        """
        manifest_path = resume_dir / "run_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No checkpoint found: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # Validate schema version
        checkpoint_version = manifest.get("pipeline_schema_version", "")
        if checkpoint_version != self.pipeline_schema_version:
            raise ValueError(
                f"Checkpoint schema version mismatch: "
                f"checkpoint={checkpoint_version}, current={self.pipeline_schema_version}"
            )

        # Validate data file hashes (if present)
        if self.a2s2d_config:
            expected_data_sha = manifest.get("data_sha256")
            expected_codebook_sha = manifest.get("codebook_sha256")

            if expected_data_sha:
                current_data_sha = file_sha256(self.a2s2d_config.data_path)
                if current_data_sha != expected_data_sha:
                    raise ValueError(
                        "Data file has changed since checkpoint was created"
                    )

            if expected_codebook_sha:
                current_codebook_sha = file_sha256(self.a2s2d_config.codebook_path)
                if current_codebook_sha != expected_codebook_sha:
                    raise ValueError(
                        "Codebook file has changed since checkpoint was created"
                    )

        # Validate config fingerprint
        expected_fingerprint = manifest.get("config_fingerprint")
        if expected_fingerprint and self.a2s2d_config:
            current_fingerprint = self.a2s2d_config.fingerprint()
            if current_fingerprint != expected_fingerprint:
                self.logger.warning(
                    "Config fingerprint mismatch - proceeding anyway"
                )

        self.logger.info(f"Checkpoint loaded: stage {manifest.get('current_stage')}")
        return manifest.get("state", {})

    def _resolve_next_stage(self) -> int:
        """Determine the next stage to run from loaded checkpoint."""
        # If we loaded a checkpoint, start from the next stage
        # (the checkpoint stores the *completed* stage)
        completed_stage = self.state.get("_completed_stage", 0)
        return completed_stage + 1

    def _serialize_state(self, state: dict) -> dict:
        """Serialize state for checkpointing.

        Removes non-serializable objects (like pandas DataFrames) and
        converts them to file paths.
        """
        serializable = {}
        for key, value in state.items():
            if key.startswith("_"):
                continue  # Skip internal keys

            # Check for pandas DataFrame
            if hasattr(value, "to_csv"):
                # Save DataFrame to file
                if self.run_dir:
                    df_path = self.run_dir / "logs" / f"{key}.csv"
                    df_path.parent.mkdir(parents=True, exist_ok=True)
                    value.to_csv(df_path, index=False)
                    serializable[key] = {"_type": "dataframe", "path": str(df_path)}
                else:
                    serializable[key] = {"_type": "dataframe", "data": None}
            elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                serializable[key] = value
            else:
                # Try to convert to string
                try:
                    serializable[key] = str(value)
                except Exception:
                    serializable[key] = None

        return serializable

    def _save_final_manifest(self) -> None:
        """Save final run manifest with summary statistics."""
        if self.run_dir is None:
            return

        manifest_path = self.run_dir / "run_manifest.json"

        # Count results
        final_leads = self.state.get("final_leads", [])
        n_leads = len(final_leads) if hasattr(final_leads, "__len__") else 0

        manifest = {
            "pipeline_schema_version": self.pipeline_schema_version,
            "status": "completed",
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "stages_completed": 8,
            "summary": {
                "total_variables": len(self.state.get("metadata", [])),
                "candidate_relations": len(self.state.get("candidates", [])),
                "models_executed": len(self.state.get("models", [])),
                "research_leads": n_leads,
            },
            "config_fingerprint": (
                self.a2s2d_config.fingerprint()
                if self.a2s2d_config
                else None
            ),
        }

        write_json(manifest_path, manifest)
        self.logger.info(f"Final manifest saved: {manifest_path}")
