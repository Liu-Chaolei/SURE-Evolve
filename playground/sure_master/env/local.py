"""SURE Master local environment.

The environment mirrors the ASR/ML split-workspace behavior but has no SURE
specific mutation logic. SURE is mounted/read through normal symlinks or
absolute paths configured by the user.
"""

from __future__ import annotations

from pathlib import Path
import shutil

from evomaster.env.local import LocalEnv, LocalEnvConfig


class SureMasterLocalEnv(LocalEnv):
    """LocalEnv variant that always creates main-workspace symlinks."""

    def _ensure_link(self, link_path: Path, target: Path, description: str) -> None:
        try:
            if link_path.is_symlink():
                if link_path.readlink() == target:
                    return
                link_path.unlink()
            elif link_path.exists():
                self.logger.debug(
                    "%s already exists and is not a symlink: %s",
                    description,
                    link_path,
                )
                return

            link_path.symlink_to(target, target_is_directory=True)
            self.logger.info("Created %s symlink: %s -> %s", description, link_path, target)
        except OSError as e:
            self.logger.warning(
                "Failed to create %s symlink %s -> %s: %s",
                description,
                link_path,
                target,
                e,
            )

    def _ensure_base_data_links(self, workspace: Path) -> None:
        recipe_dir = workspace / "base_model" / "recipe"
        data_dir = workspace / "base_model" / "data"
        if not recipe_dir.exists() or not data_dir.exists():
            return

        if recipe_dir.is_symlink():
            source = recipe_dir.resolve()
            recipe_dir.unlink()
            shutil.copytree(source, recipe_dir, symlinks=False,
                            ignore=shutil.ignore_patterns("data", "exp", "__pycache__"))
        self._ensure_link(recipe_dir / "data", Path("../data"), "workspace recipe data")
        self._ensure_link(
            workspace / "data",
            Path("base_model/data"),
            "workspace icefall data",
        )

    def setup(self) -> None:
        if self._is_ready:
            self.logger.warning("Environment already setup")
            return

        workspace = Path(self.config.session_config.workspace_path)
        workspace.mkdir(parents=True, exist_ok=True)

        session_config = self.config.session_config
        if hasattr(session_config, "symlinks") and session_config.symlinks:
            self._create_symlinks(workspace, session_config.symlinks)

        self._ensure_base_data_links(workspace)

        self._is_ready = True
        self.logger.info("SURE Master local environment setup complete")

    def setup_exp_workspace(self, exp_workspace_path: str) -> None:
        super().setup_exp_workspace(exp_workspace_path)
        self._ensure_base_data_links(Path(exp_workspace_path))
