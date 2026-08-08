"""SURE Master local session implementation."""

from __future__ import annotations

from evomaster.agent.session.base import BaseSession
from evomaster.agent.session.local import LocalSession, LocalSessionConfig
from evomaster.env.local import LocalEnvConfig

from ...env.local import SureMasterLocalEnv


class SureMasterLocalSession(LocalSession):
    """Local session using SureMasterLocalEnv."""

    def __init__(self, config: LocalSessionConfig | None = None):
        BaseSession.__init__(self, config)
        self.config: LocalSessionConfig = config or LocalSessionConfig()
        env_config = LocalEnvConfig(session_config=self.config)
        self._env = SureMasterLocalEnv(env_config)

