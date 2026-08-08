"""Run manager for EvoMaster's optional self-evolution loop."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .analyzer import EvolutionAnalyzer
from .applier import EvolutionApplier
from .collector import collect_trace_digest
from .models import RunMetrics

logger = logging.getLogger(__name__)


@dataclass
class EvolutionRunConfig:
    """Configuration for one evolution loop."""

    agent_name: str
    config_path: Path
    task_description: str
    run_dir: Path
    project_root: Path
    iterations: int = 1
    images: list[str] | None = None
    use_llm_analyzer: bool = True
    extra_args: list[str] = field(default_factory=list)


class EvolutionManager:
    """Run baseline -> analyze -> materialize overlay -> rerun."""

    def __init__(self, config: EvolutionRunConfig):
        self.config = config
        self.project_root = Path(config.project_root).resolve()
        self.run_dir = Path(config.run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "evolution_state.json"
        self.logs_dir = self.run_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.logs_dir / "evolution.log"
        self._log_file_handler: logging.FileHandler | None = None
        self._setup_logging()

    def run(self) -> int:
        """Execute the evolution loop. Returns a process-style exit code."""
        try:
            logger.info("=" * 80)
            logger.info("EvoMaster evolution started")
            logger.info("Agent: %s", self.config.agent_name)
            logger.info("Initial config: %s", Path(self.config.config_path).resolve())
            logger.info("Run dir: %s", self.run_dir)
            logger.info("Evolution log: %s", self.log_path)
            logger.info("Iterations: %s", self.config.iterations)
            logger.info("LLM analyzer enabled: %s", self.config.use_llm_analyzer)
            logger.info("=" * 80)

            state: dict = {
                "agent": self.config.agent_name,
                "initial_config": str(Path(self.config.config_path).resolve()),
                "iterations_requested": self.config.iterations,
                "evolution_log": str(self.log_path),
                "runs": [],
            }

            baseline_dir = self.run_dir / "iterations" / "iter_000_baseline"
            baseline_code = self._run_agent(Path(self.config.config_path), baseline_dir)
            logger.info("Collecting baseline digest from: %s", baseline_dir)
            baseline_digest = collect_trace_digest(baseline_dir, self.config.config_path)
            logger.info("Baseline metrics: %s", baseline_digest.metrics.model_dump())
            state["runs"].append(
                {
                    "kind": "baseline",
                    "run_dir": str(baseline_dir),
                    "return_code": baseline_code,
                    "metrics": baseline_digest.metrics.model_dump(),
                }
            )
            self._write_state(state)

            current_config = Path(self.config.config_path).resolve()
            previous_digest = baseline_digest
            final_code = baseline_code

            for iteration in range(1, max(self.config.iterations, 0) + 1):
                logger.info("=" * 80)
                logger.info("Starting evolution iteration %s", iteration)
                logger.info("Analyzed run dir: %s", previous_digest.run_dir)
                logger.info("Analyzed metrics: %s", previous_digest.metrics.model_dump())
                logger.info("Current config: %s", current_config)

                analyzer = EvolutionAnalyzer(current_config, use_llm=self.config.use_llm_analyzer)
                candidates = analyzer.analyze(previous_digest)

                artifacts_dir = self.run_dir / "evolution_artifacts"
                applier = EvolutionApplier(self.project_root, self.config.config_path, artifacts_dir)
                overlay = applier.apply(candidates, current_config, iteration)
                logger.info("Overlay config: %s", overlay.config_path)
                logger.info("Overlay summary: %s", overlay.summary_path)
                logger.info("Applied skills: %s", overlay.applied_skills)
                logger.info("Applied prompt patches: %s", overlay.applied_prompt_patches)
                logger.info("Tool proposals: %s", overlay.tool_proposals)

                evolved_dir = self.run_dir / "iterations" / f"iter_{iteration:03d}_evolved"
                final_code = self._run_agent(overlay.config_path, evolved_dir)
                logger.info("Collecting evolved digest from: %s", evolved_dir)
                evolved_digest = collect_trace_digest(evolved_dir, overlay.config_path)
                logger.info("Evolved metrics: %s", evolved_digest.metrics.model_dump())

                comparison = self._compare_metrics(previous_digest.metrics, evolved_digest.metrics)
                logger.info("Comparison to previous: %s", comparison)
                state["runs"].append(
                    {
                        "kind": "evolved",
                        "iteration": iteration,
                        "run_dir": str(evolved_dir),
                        "config": str(overlay.config_path),
                        "return_code": final_code,
                        "metrics": evolved_digest.metrics.model_dump(),
                        "comparison_to_previous": comparison,
                        "overlay_summary": str(overlay.summary_path),
                    }
                )
                self._write_state(state)

                current_config = overlay.config_path
                previous_digest = evolved_digest

            logger.info("EvoMaster evolution finished with return code: %s", final_code)
            return final_code
        finally:
            self._teardown_logging()

    def _setup_logging(self) -> None:
        """Attach a dedicated file handler for the parent evolution process."""
        root_logger = logging.getLogger()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler = logging.FileHandler(self.log_path, mode="w", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        if root_logger.level > logging.INFO:
            root_logger.setLevel(logging.INFO)
        self._log_file_handler = handler
        logger.info("Evolution logging initialized: %s", self.log_path)

    def _teardown_logging(self) -> None:
        if self._log_file_handler is None:
            return
        root_logger = logging.getLogger()
        root_logger.removeHandler(self._log_file_handler)
        self._log_file_handler.close()
        self._log_file_handler = None

    def _run_agent(self, config_path: Path, run_dir: Path) -> int:
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(self.project_root / "run.py"),
            "--agent",
            self.config.agent_name,
            "--config",
            str(config_path),
            "--task",
            self.config.task_description,
            "--run-dir",
            str(run_dir),
        ]
        if self.config.images:
            cmd.extend(["--images", *self.config.images])
        cmd.extend(self.config.extra_args)

        logger.info("Executing EvoMaster run: %s", " ".join(cmd))
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")

        with subprocess.Popen(
            cmd,
            cwd=str(self.project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        ) as proc:
            if proc.stdout is not None:
                for line in proc.stdout:
                    logger.info("[agent:%s] %s", run_dir.name, line.rstrip())
            return_code = proc.wait()

        logger.info("EvoMaster run finished: run_dir=%s return_code=%s", run_dir, return_code)
        return return_code

    def _write_state(self, state: dict) -> None:
        self.state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Evolution state written: %s", self.state_path)

    @staticmethod
    def _compare_metrics(before: RunMetrics, after: RunMetrics) -> dict:
        score_delta = None
        if before.score is not None and after.score is not None:
            score_delta = after.score - before.score
        return {
            "status_changed": [before.status, after.status],
            "score_delta": score_delta,
            "step_delta": after.step_count - before.step_count,
            "tool_error_delta": after.tool_error_count - before.tool_error_count,
            "issue_delta": after.issue_count - before.issue_count,
        }
