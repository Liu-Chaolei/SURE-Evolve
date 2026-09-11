from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from playground.sure_master.core.playground import SureMasterPlayground
from playground.sure_master.core.providers import FakeXlabIdeaProvider


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TASK_CARDS = PROJECT_ROOT / "playground" / "sure_master" / "task_cards" / "sure_tasks.yaml"


class _FakePrefetch:
    def __init__(self, workspace: Path, exp_name: str) -> None:
        self.workspace_path = str(workspace / exp_name)
        self.exp_name = exp_name

    def run(self, task_description: str, task_id: str = "exp_001") -> tuple[str, str, str]:
        Path(self.workspace_path).mkdir(parents=True, exist_ok=True)
        return "fake data knowledge", "fake model knowledge", f"fake descriptor: {task_description}"


class _FakeCandidateExp:
    def __init__(self, workspace: Path, exp_name: str) -> None:
        self.workspace_path = str(workspace / exp_name)
        self.exp_name = exp_name
        self.execution_env: dict[str, str] = {}
        self.candidate_stage_name = ""
        self.candidate_phase = ""
        self.candidate_rung_name = ""
        self.candidate_idea_id = ""
        self.metric_feedback = ""
        self.terminal_output = ""
        self.enforce_candidate_type = False

    def _evaluate(self, code: str, candidate_type_hint: str) -> tuple[bool, float | None, str, dict[str, Any]]:
        workspace = Path(self.workspace_path)
        checkpoint = workspace / "artifacts" / "checkpoints" / f"epoch-{self._epoch()}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(f"fake checkpoint for {self.candidate_idea_id}\n", encoding="utf-8")

        # Keep one deterministic candidate failure in the published history.
        failed = self.candidate_idea_id.endswith("i2") and self.candidate_rung_name == "short"
        score = None if failed else self._score()
        self.metric_feedback = (
            "fake candidate failure at short rung"
            if failed
            else f"fake WER={score:.3f}; lower is better"
        )
        reason_code = "candidate_execution_failed" if failed else "success"
        details = {
            "reason_code": reason_code,
            "metric_feedback": self.metric_feedback,
            "runtime_seconds": 0.001,
            "produced_artifacts": {
                "candidate_checkpoint": str(checkpoint),
                "checkpoint_dir": str(checkpoint.parent),
            },
        }
        return not failed, score, reason_code, details

    def _epoch(self) -> int:
        return {"short": 1, "medium": 2, "final": 3}.get(self.candidate_rung_name, 1)

    def _score(self) -> float:
        idea = self.candidate_idea_id
        stable = sum(ord(char) for char in idea) % 31
        rung_bonus = {"short": 0.3, "medium": 0.2, "final": 0.1}.get(self.candidate_rung_name, 0.0)
        phase_bonus = {"combination": -0.05, "selection": -0.1, "holdout": 0.05}.get(
            self.candidate_phase,
            0.0,
        )
        return 1.0 + stable / 100.0 + rung_bonus + phase_bonus

    def run(self, **kwargs: Any) -> tuple[bool, float | None, Any, str, dict[str, Any]]:
        code = f"# fake candidate {self.candidate_idea_id}\n"
        success, score, reason_code, details = self._evaluate(
            code,
            str(kwargs.get("candidate_type_hint") or "inference"),
        )
        return success, score, uuid4(), code, details

    def run_existing_code(self, *, code: str, candidate_type_hint: str = "inference", **kwargs: Any):
        success, score, reason_code, details = self._evaluate(code, candidate_type_hint)
        return success, score, uuid4(), code, details


class _CpuFakeSureMaster(SureMasterPlayground):
    def _setup_agents(self) -> None:
        # The controller only needs slots; no LLM or MCP client is created.
        for name in (
            "draft_agent",
            "debug_agent",
            "improve_agent",
            "reseach_agent",
            "knowledge_promotion_agent",
            "prefetch_agent",
            "wisdom_promotion_agent",
        ):
            setattr(self.agents, name, None)
        self.agent = None

    def _create_prefetch_exp(self, exp_index: int) -> _FakePrefetch:
        return _FakePrefetch(
            Path(self.session.config.workspace_path),
            f"exp_{exp_index}_prefetch",
        )

    def _create_run_exp(self, stage: str, exp_index: int) -> _FakeCandidateExp:
        return _FakeCandidateExp(
            Path(self.session.config.workspace_path),
            f"exp_{exp_index}_{stage}",
        )


class ControllerSmokeTests(unittest.TestCase):
    def _write_config(self, root: Path, baseline: Path, ref: Path) -> Path:
        config = {
            "sure": {
                "root": str(root / "unused-sure-root"),
                "pythonpath": str(root / "unused-sure-src"),
                "task_cards_path": str(TASK_CARDS),
                "task_id": "asr_en_wer",
                "device": "cpu",
                "validate_env": False,
                "require_base_model": False,
                "base_models": {
                    "asr_en_wer": {
                        "source_paths": {
                            "recipe": str(root / "base-recipe"),
                            "data": str(root / "base-data"),
                            "root": str(root / "base-root"),
                        }
                    }
                },
                "execution_mode": "local",
                "inputs": {"ref": str(ref)},
                "initial_solution_path": str(baseline),
                "coordinator": {"local_gpu_policy": "auto"},
                "remote_training": {"enabled": False},
                "metric_gpu": {"enabled": False},
                "search_strategy": "staged_axes",
                "staged_axes": {
                    "enabled": True,
                    "start_phase": "arch",
                    "rounds_per_axis": 1,
                    "ideas_per_round": 4,
                    "top_arch": 1,
                    "top_train": 1,
                    "top_inference": 1,
                    "runner_up_count": 1,
                    "axes": {
                        axis: {
                            "rungs": [
                                {"name": "short", "keep": 4},
                                {"name": "medium", "keep": 3},
                                {"name": "final", "keep": 2},
                            ]
                        }
                        for axis in ("arch", "train", "inference")
                    },
                    "holdout": {"enabled": True},
                },
                "execution_contract": {"candidate_boundary": "fake_cpu_only"},
            },
            "xlab": {
                "enabled": True,
                "idea_provider": {"enabled": True},
                "history_path": "artifacts/xlab_history.json",
            },
            "session": {
                "type": "local",
                "local": {
                    "working_dir": str(root / "workspace"),
                    "timeout": 30,
                    "gpu_devices": "none",
                    "parallel": {"enabled": False, "max_parallel": 1, "split_workspace_for_exp": False},
                },
            },
            "max_research_rounds": 1,
            "agents": {},
        }
        path = root / "sure-smoke.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path

    def test_staged_profile_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, ref = root / "baseline.py", root / "ref.txt"
            baseline.write_text("# baseline")
            ref.write_text("utt\thello\n")
            path = self._write_config(root, baseline, ref)
            controller = _CpuFakeSureMaster(config_path=path, xlab_provider=FakeXlabIdeaProvider())
            result = controller.run("Retired strategy")
            self.assertEqual(result["status"], "failed")
            self.assertIn("staged_axes is retired", result["error"])


if __name__ == "__main__":
    unittest.main()
