"""Chat Agent Bohrium Tool

Interacts with the Bohrium platform via the ``bohr-agent`` CLI. Currently supports
uploading local files to Bohrium OSS so that other platform tools can access them
through the returned URL. Additional actions (download, list, etc.) can be added
by registering new handlers in ``_ACTION_HANDLERS``.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession

logger = logging.getLogger(__name__)

_REQUIRED_ENV_VARS = ("BOHRIUM_PROJECT_ID", "BOHRIUM_ACCESS_KEY")
_UPLOAD_TIMEOUT_SECONDS = 600


class BohriumToolParams(BaseToolParams):
    """Interact with the Bohrium platform.

    Currently supports a single action:

    - ``upload``: upload a local file to Bohrium OSS and return the public URL,
      so that other Bohrium/MCP tools can access the file directly by URL.

    Requires ``BOHRIUM_PROJECT_ID`` and ``BOHRIUM_ACCESS_KEY`` environment
    variables to be set.
    """

    name: ClassVar[str] = "bohrium"

    action: Literal["upload"] = Field(
        default="upload",
        description="Action to perform. Currently only 'upload' is supported. Upload a local file to Bohrium OSS and return the public URL, so that other Bohrium/MCP tools can access the file directly by URL",
    )
    file_path: str = Field(
        description="Absolute path to the local file to upload (used when action='upload').",
    )


class BohriumTool(BaseTool):
    """Bohrium platform tool (wraps the ``bohr-agent`` CLI)."""

    name: ClassVar[str] = "bohrium"
    params_class: ClassVar[type[BaseToolParams]] = BohriumToolParams

    def __init__(self):
        super().__init__()
        self._action_handlers: dict[str, Callable[[BohriumToolParams], tuple[str, dict[str, Any]]]] = {
            "upload": self._upload,
        }

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        """Dispatch a Bohrium action based on params."""
        missing = [v for v in _REQUIRED_ENV_VARS if not os.environ.get(v)]
        if missing:
            return (
                "bohrium: missing required environment variables: "
                f"{', '.join(missing)}. Please set them before using this tool.",
                {"error": "missing_env_vars", "missing": missing},
            )

        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"Parameter validation error: {e}", {"error": str(e)}

        assert isinstance(params, BohriumToolParams)

        handler = self._action_handlers.get(params.action)
        if handler is None:
            return (
                f"bohrium: unsupported action '{params.action}'.",
                {"error": "unsupported_action", "action": params.action},
            )

        return handler(params)

    def _upload(self, params: BohriumToolParams) -> tuple[str, dict[str, Any]]:
        """Upload a local file to Bohrium OSS via ``bohr-agent artifact upload``."""
        file_path = params.file_path.strip()
        p = Path(file_path)
        if not p.exists():
            return f"File not found: {file_path}", {"error": "file_not_found"}
        if not p.is_file():
            return f"Not a file: {file_path}", {"error": "not_a_file"}

        cmd = ["bohr-agent", "artifact", "upload", "-s", "https", str(p)]
        self.logger.info("Running bohrium upload: %s", shlex.join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_UPLOAD_TIMEOUT_SECONDS,
                check=False,
            )
        except FileNotFoundError:
            return (
                "bohrium: 'bohr-agent' CLI not found. Install it with "
                "`pip install bohr-agent-sdk -i https://pypi.org/simple --upgrade`.",
                {"error": "cli_not_found"},
            )
        except subprocess.TimeoutExpired:
            return (
                f"bohrium: upload timed out after {_UPLOAD_TIMEOUT_SECONDS}s.",
                {"error": "timeout"},
            )

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        if result.returncode != 0:
            message = stderr or stdout or f"exit code {result.returncode}"
            return (
                f"bohrium: upload failed: {message}",
                {"error": "upload_failed", "returncode": result.returncode, "stderr": stderr},
            )

        output = stdout or stderr
        return output, {"action": "upload", "file_path": str(p)}
