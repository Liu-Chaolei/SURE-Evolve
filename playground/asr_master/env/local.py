"""ASR Master local environment implementation.

Inherits from evomaster's LocalEnv. Differences:
- when split_workspace_for_exp is enabled, symlink creation in the main workspace
  is not skipped;
- each workspace also gets icefall_recipe/data -> ../icefall_data so Lhotse
  feature paths embedded as data/fbank/... resolve after scripts chdir into
  icefall_recipe.
"""

from __future__ import annotations

from pathlib import Path

from evomaster.env.local import LocalEnv, LocalEnvConfig


class MLMaster2LocalEnv(LocalEnv):
    """ASR Master dedicated local environment.

    Difference from base LocalEnv: in setup(), even if split_workspace_for_exp is
    enabled, symlink creation in the main workspace is not skipped and will always
    be created. ASR workspaces also receive a recipe-local data symlink required
    by icefall/Lhotse manifests.
    """

    def _ensure_recipe_data_link(self, workspace: Path) -> None:
        """Ensure icefall_recipe/data resolves to the workspace's icefall_data."""
        recipe_dir = workspace / "icefall_recipe"
        data_dir = workspace / "icefall_data"

        if not recipe_dir.exists() or not data_dir.exists():
            self.logger.debug(
                "Skipping ASR recipe data link; missing recipe/data dir in %s",
                workspace,
            )
            return

        link_path = recipe_dir / "data"
        target = Path("../icefall_data")

        try:
            if link_path.is_symlink():
                if link_path.readlink() == target:
                    return
                link_path.unlink()
            elif link_path.exists():
                self.logger.warning(
                    "ASR recipe data path already exists and is not a symlink: %s",
                    link_path,
                )
                return

            link_path.symlink_to(target, target_is_directory=True)
            self.logger.info("Created ASR recipe data symlink: %s -> %s", link_path, target)
        except OSError as e:
            self.logger.warning(
                "Failed to create ASR recipe data symlink %s -> %s: %s",
                link_path,
                target,
                e,
            )

    def setup(self) -> None:
        """Initialize the local environment.

        Unlike the base class: symlinks are always created in the main workspace
        regardless of whether split_workspace_for_exp is enabled.
        """
        if self._is_ready:
            self.logger.warning("Environment already setup")
            return

        self.logger.info("Setting up ML Master 2 local environment")

        # Ensure working directory exists
        workspace = Path(self.config.session_config.workspace_path)
        workspace.mkdir(parents=True, exist_ok=True)

        # Always create symlinks in main workspace (not skipped due to split_workspace_for_exp)
        session_config = self.config.session_config
        if hasattr(session_config, "symlinks") and session_config.symlinks:
            self._create_symlinks(workspace, session_config.symlinks)
            self.logger.info("Main workspace symlinks created")
        else:
            self.logger.debug("No symlinks configuration, skipping symlink creation")

        self._ensure_recipe_data_link(workspace)

        self._is_ready = True
        self.logger.info("ASR Master local environment setup complete")

    def setup_exp_workspace(self, exp_workspace_path: str) -> None:
        """Create an experiment workspace and ASR-specific recipe data symlink."""
        super().setup_exp_workspace(exp_workspace_path)
        self._ensure_recipe_data_link(Path(exp_workspace_path))
