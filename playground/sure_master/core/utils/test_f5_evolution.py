"""Free TTS execution preserves full-training evidence and resource boundaries."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import yaml

from playground.sure_master.core.training import (
    F5_EVOLUTION_TRAINING,
    canonical_digest,
    validate_training_config,
    validate_completion,
)
from playground.sure_master.core.search_scope import execution_contract
from playground.sure_master.core.utils.slurm import resource_profile
from playground.sure_master.core.utils.test_official_training import completion
from playground.sure_master.runtime.f5_evolution import (
    PROTECTED,
    validate_source,
    executable_identity,
)
from playground.sure_master.tasks.adapters import TtsAdapter
from playground.sure_master.tasks.tts_outputs import validate_samples
from playground.sure_master.tools.prepare_task_data import balanced_prompts
from playground.sure_master.tools.run_task_candidate import run


class FreeF5Tests(unittest.TestCase):
    def test_every_adapter_inference_parameter_has_a_cli_option(self):
        from playground.sure_master.tasks.f5tts import INFERENCE_KEYS
        from playground.sure_master.tools.run_f5tts_batch_infer import build_arg_parser

        parser = build_arg_parser()
        options = {flag for action in parser._actions for flag in action.option_strings}
        self.assertTrue(
            {"--" + key.replace("_", "-") for key in INFERENCE_KEYS} <= options
        )
        parsed = parser.parse_args(
            ["--target-rms", "0.2", "--cross-fade-duration", "0.05"]
        )
        self.assertEqual((parsed.target_rms, parsed.cross_fade_duration), (0.2, 0.05))

    def test_zero_inference_timeout_is_not_a_sixty_second_cap(self):
        from playground.sure_master.tools import run_f5tts_batch_infer as batch

        with patch.dict(os.environ, {"SURE_RUN_TIMEOUT": "0"}):
            self.assertEqual(batch.build_arg_parser().parse_args([]).timeout, 0)
        with tempfile.TemporaryDirectory() as temporary:
            proc = Mock()
            proc.wait.return_value = 0
            proc.poll.return_value = 0
            with (
                patch.object(batch, "LOG_DIR", Path(temporary)),
                patch.object(batch.subprocess, "Popen", return_value=proc),
            ):
                batch.launch_workers(
                    config_path=Path("config.json"),
                    shard_paths=[Path("shard.json")],
                    timeout=0,
                    device="npu:0",
                )
            proc.wait.assert_called_once_with(timeout=None)

    def test_variable_recipe_keeps_full_completion(self):
        training = {
            **F5_EVOLUTION_TRAINING,
            "learning_rate": 3e-5,
            "grad_accumulation_steps": 2,
        }
        validate_training_config("tts.f5tts", training, {"world_size": 8})
        report = completion("tts.f5tts")
        report["contract"]["training"] = training
        report["contract_digest"] = canonical_digest(report["contract"])
        report["optimizer_updates"] = 500
        validate_completion(report)
        for changes in (
            {"epochs": 10},
            {"world_size": 1},
            {"seed": 43},
            {"max_steps": 1000},
            {"learning_rate": float("nan")},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_training_config("tts.f5tts", {**training, **changes})
        with self.assertRaises(ValueError):
            validate_completion(
                {**report, "epochs_completed": 10, "optimizer_updates": 50}
            )

    def test_formal_configuration_and_free_contract(self):
        repo = Path(__file__).resolve().parents[4]
        config = yaml.safe_load(
            (
                repo / "configs/sure_master/xlab-f5tts-premium-npu-evolution.yaml"
            ).read_text()
        )
        sure = config["sure"]
        validate_training_config("tts.f5tts", sure["task"]["training"], sure["runtime"])
        contract = execution_contract(TtsAdapter().context(), sure)
        self.assertEqual(contract["search_scope"], "all")
        self.assertNotIn("allowed_change_domains", contract)
        self.assertIn("training", contract["candidate_parameters"])
        self.assertEqual(sure["search_budget"]["ideas_per_round"], 3)
        self.assertFalse(sure["require_training_candidate"])
        self.assertNotIn("full_training", sure)
        profile = resource_profile(sure["slurm"], "fine_tune", adapter="tts.f5tts")
        self.assertEqual(profile["npu"] * sure["slurm"]["max_parallel"], 24)
        self.assertEqual(
            resource_profile(sure["slurm"], "inference", adapter="tts.f5tts")["npu"], 1
        )

    def test_source_changes_and_protected_loop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pristine, candidate = root / "pristine", root / "candidate"
            for directory in (pristine, candidate):
                for name in (*PROTECTED, "src/f5_tts/model/cfm.py"):
                    path = directory / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("value = 1\n")
            with patch("pathlib.Path.cwd", return_value=root):
                (candidate / "src/f5_tts/model/cfm.py").write_text("value = 2\n")
                self.assertTrue(
                    validate_source(candidate, pristine, requires_training=True)
                )
                self.assertNotEqual(
                    executable_identity(pristine), executable_identity(candidate)
                )
                with self.assertRaisesRegex(ValueError, "require"):
                    validate_source(candidate, pristine, requires_training=False)
                (candidate / PROTECTED[0]).write_text("epochs = 1\n")
                with self.assertRaisesRegex(ValueError, "fixed"):
                    validate_source(candidate, pristine, requires_training=True)

    def test_unified_candidate_allows_coordinated_changes(self):
        env = {
            "SURE_TASK_ADAPTER": "tts.f5tts",
            "SURE_SEARCH_SCOPE": "all",
            "SURE_CANDIDATE_TYPE_HINT": "arch",
            "SURE_TASK_SETTINGS": "{}",
        }
        adapter = SimpleNamespace(execute_candidate=lambda *args: None)
        with (
            patch.dict(os.environ, env, clear=True),
            patch(
                "playground.sure_master.tools.run_task_candidate.get_adapter",
                return_value=adapter,
            ),
            patch.object(adapter, "execute_candidate") as execute,
        ):
            parameters = {
                "requires_training": True,
                "training": {"learning_rate": 2e-5},
                "architecture": {"depth": 20},
                "inference": {"nfe_step": 24},
            }
            run("candidate", parameters)
            self.assertEqual(execute.call_args.args[0], "arch")
            self.assertEqual(
                set(execute.call_args.args[1]),
                {"training", "architecture", "inference"},
            )
            with self.assertRaisesRegex(ValueError, "differs"):
                run("candidate", {"requires_training": False})
            with self.assertRaisesRegex(ValueError, "Frozen"):
                run("infer", {}, "model.json", "working/source")

    def test_fixed_evaluation_text_and_complete_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, samples = root / "input.jsonl", root / "samples.jsonl"
            manifest.write_text(
                json.dumps({"sample_id": "a", "target_text": "你好"}) + "\n"
            )
            (root / "a.wav").write_bytes(b"audio")
            record = {
                "sample_id": "a",
                "reference_text": "你好",
                "prediction_audio": "a.wav",
            }
            samples.write_text(json.dumps(record) + "\n")
            validate_samples(manifest, samples)
            for records in (
                [],
                [record, record],
                [{**record, "reference_text": "改写"}],
                [{**record, "sample_id": "b"}],
            ):
                samples.write_text("".join(json.dumps(r) + "\n" for r in records))
                with self.assertRaises(ValueError):
                    validate_samples(manifest, samples)

    def test_group_balancing_and_reproducibility(self):
        rows = [
            {"sample_id": f"{g}-{i}", "group_id": g} for g in "abcd" for i in range(5)
        ]
        selected = balanced_prompts(rows, 8)
        self.assertEqual(selected, balanced_prompts(list(reversed(rows)), 8))
        self.assertEqual(
            {g: sum(r["group_id"] == g for r in selected) for g in "abcd"},
            dict.fromkeys("abcd", 2),
        )
        with self.assertRaises(ValueError):
            balanced_prompts(rows, 21)


if __name__ == "__main__":
    unittest.main()
