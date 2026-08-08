"""Multi-Agent Playground Implementation

Demonstrates how to use multiple Agents collaborating on tasks.
Contains the workflow for Planning Agent and Coding Agent.
"""

import logging
import sys
from pathlib import Path

# Ensure evomaster module can be imported
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evomaster.core import BasePlayground, register_playground
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evomaster.agent import Agent

from .exp import MultiAgentExp
from functools import partial


@register_playground("minimal_multi_agent_parallel")
class MultiAgentParallelPlayground(BasePlayground):
    """Multi-Agent Parallel Playground.

    Implements the collaborative workflow of Planning Agent and Coding
    Agent run in parallel across ``max_workers`` exps:

    1. Planning Agent analyzes the task and formulates a plan.
    2. Coding Agent executes code based on the plan.

    When the docker session config sets ``fresh_container_per_exp: true``,
    each parallel exp also gets its own brand-new Docker container that is
    auto-removed when the exp finishes.
    """

    def __init__(self, config_dir: Path = None, config_path: Path = None):
        """Initialize the Multi-Agent Parallel Playground.

        Args:
            config_dir: Configuration directory path; defaults to
                ``configs/minimal_multi_agent_parallel/``.
            config_path: Full path to a config file (overrides
                ``config_dir`` if provided).
        """
        if config_path is None and config_dir is None:
            # Default configuration directory.
            config_dir = (
                Path(__file__).parent.parent.parent.parent
                / "configs" / "minimal_multi_agent_parallel"
            )

        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents.declare("planning_agent", "coding_agent")

        # Resolve max_workers from session config: prefer the local-session
        # ``parallel.max_parallel`` knob (kept for back-compat), then fall
        # back to a docker-session knob, then a default of 3.
        local_parallel = (
            (self.config.session.get("local", {}) or {}).get("parallel", {}) or {}
        )
        docker_parallel = (
            (self.config.session.get("docker", {}) or {}).get("parallel", {}) or {}
        )
        if local_parallel.get("enabled", False):
            self.max_workers = int(local_parallel.get("max_parallel", 3))
        elif docker_parallel.get("max_parallel"):
            self.max_workers = int(docker_parallel["max_parallel"])
        else:
            self.max_workers = 3

        # Initialize mcp_manager (required by BasePlayground.cleanup).
        self.mcp_manager = None

        # Per-exp state (populated only when fresh_container_per_exp=true).
        # Maps exp_index -> (DockerSession, MultiAgentExp).
        self._per_exp_state: dict[int, tuple] = {}

    def setup(self) -> None:
        """Initialize all components."""
        self.logger.info("Setting up minimal multi-agent parallel playground...")

        self._setup_session()
        self._setup_agents()

        self.logger.info("Minimal multi-agent parallel playground setup complete")

    def _create_exp(self, exp_index):
        """Create a multi-agent experiment instance.

        Overrides the base-class single-arg signature: the parallel
        playground builds one exp per parallel slot and indexes by
        ``exp_index``. Each exp gets its own Agent copies (independent
        context) but, by default, shares the playground's session.
        """
        planning_agent_copy = (
            self.copy_agent(
                self.agents.planning_agent,
                new_agent_name=f"planning_exp_{exp_index}",
            )
            if self.agents.planning_agent
            else None
        )
        coding_agent_copy = (
            self.copy_agent(
                self.agents.coding_agent,
                new_agent_name=f"coding_exp_{exp_index}",
            )
            if self.agents.coding_agent
            else None
        )

        exp = MultiAgentExp(
            planning_agent=planning_agent_copy,
            coding_agent=coding_agent_copy,
            config=self.config,
            exp_index=exp_index,
        )
        if self.run_dir:
            exp.set_run_dir(self.run_dir)
        return exp

    # ------------------------------------------------------------------ #
    # Per-exp Docker container support
    # ------------------------------------------------------------------ #

    def _open_per_exp_session(self, exp_index: int) -> None:
        """Spin up a fresh DockerSession for one parallel exp and bind it.

        Replaces ``agent.session`` on every agent of ``exp_index``'s exp so
        the agent's tool calls land in the new container.
        """
        sess = self.make_per_exp_docker_session(exp_index)
        sess.open()
        self.logger.info(
            f"[exp_{exp_index}] opened fresh Docker container "
            f"{sess.container_name} ({(sess.container_id or '')[:12]})"
        )
        # Inject the session into the exp's agents so tool calls go to it.
        _, exp = self._per_exp_state.get(exp_index, (None, None))
        if exp is None:
            self.logger.warning(
                f"[exp_{exp_index}] no exp registered when opening per-exp session"
            )
            return
        for agent in (exp.planning_agent, exp.coding_agent):
            if agent is not None:
                agent.session = sess
        # Remember the session so the post hook can close it.
        self._per_exp_state[exp_index] = (sess, exp)

    def _close_per_exp_session(self, exp_index: int) -> None:
        """Stop+remove the fresh DockerSession for one parallel exp."""
        sess, _ = self._per_exp_state.get(exp_index, (None, None))
        if sess is None:
            return
        try:
            sess.close()  # auto_remove=True -> stop+remove the container
            self.logger.info(
                f"[exp_{exp_index}] closed and removed per-exp Docker container"
            )
        except Exception as e:
            self.logger.warning(
                f"[exp_{exp_index}] error closing per-exp Docker session: {e}"
            )
        finally:
            self._per_exp_state.pop(exp_index, None)

    def run(self, task_description: str, output_file: str | None = None,
            images: list[str] | None = None, on_step=None) -> dict:
        """Run the workflow (overrides base class method).

        Each of ``max_workers`` exps receives the same task description and
        runs in parallel. When ``fresh_container_per_exp: true`` is set on
        the docker session, each exp also gets its own throwaway container.
        """
        try:
            self.setup()
            self._setup_trajectory_file(output_file)

            task_descriptions = [task_description] * self.max_workers

            # Build all exps up front so the per-exp hooks can find them.
            exps = [self._create_exp(exp_index=i) for i in range(self.max_workers)]
            tasks = [
                partial(exps[i].run, task_description=task_descriptions[i])
                for i in range(self.max_workers)
            ]

            fresh = self._is_docker_fresh_per_exp()

            if fresh:
                # Pre-register exps so the open hook can find them.
                for i, exp in enumerate(exps):
                    self._per_exp_state[i] = (None, exp)
                pre = self._open_per_exp_session
                post = self._close_per_exp_session
                self.logger.info(
                    "Docker fresh_container_per_exp=true: each parallel exp "
                    "will get its own container (auto-removed on completion)."
                )
            else:
                pre = post = None

            results = self.execute_parallel_tasks(
                tasks,
                max_workers=self.max_workers,
                pre_task_hook=pre,
                post_task_hook=post,
            )

            return {
                "status": "completed",
                "steps": 0,
                "results": results,
            }
        finally:
            self.cleanup()


