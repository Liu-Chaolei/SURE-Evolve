"""Full-budget training during ordinary search; final evaluation never retrains."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from playground.sure_master.core.full_training import (
    promote_full_training_to_search,
    validate_full_training_data,
)
from playground.sure_master.core.providers import FakeXlabIdeaProvider
from playground.sure_master.core.utils import test_multitask as fixtures
from playground.sure_master.tools.benchmark_tedlium import (
    training_cost_estimates,
    training_signature,
)


class FullSearchTests(unittest.TestCase):
    def test_legacy_final_budget_is_promoted_before_all_rounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = fixtures.MultiTaskControllerTests().config(root, "asr")
            config = yaml.safe_load(path.read_text())
            config["sure"].update(
                search_scope="architecture_only",
                full_training={
                    "enabled": True,
                    "data": str(root / "full"),
                    "epochs": 30,
                },
            )
            config["sure"]["execution_env"] = {
                "SURE_MAX_TRAIN_EPOCHS": "10",
                "SURE_BASELINE_EPOCH": "10",
            }
            config["sure"]["execution_contract"]["training_hours"] = 100
            path.write_text(yaml.safe_dump(config))
            provider = FakeXlabIdeaProvider()
            controller = fixtures.Controller(config_path=path, xlab_provider=provider)
            controller.set_run_dir(root / "run", task_id="full-search")
            with (
                patch(
                    "playground.sure_master.core.playground.KnowledgePromotionExp.run",
                    return_value="findings",
                ),
                patch.object(
                    controller.task_adapter,
                    "final_candidates",
                    side_effect=AssertionError("late training must not be called"),
                ),
            ):
                result = controller.run("Train every architecture to the final budget")
            self.assertEqual(result["status"], "completed", result)
            self.assertEqual(
                len(controller.executed), 9
            )  # one baseline, four candidates per round
            for _, env, _ in controller.executed:
                self.assertEqual(env["SURE_MAX_TRAIN_EPOCHS"], "30")
                self.assertEqual(env["SURE_BASELINE_EPOCH"], "30")
            self.assertNotIn("full_training", controller.sure_config)
            self.assertEqual(
                controller.base_model_profile.source_paths["data"], str(root / "full")
            )
            for request in provider.generated_requests:
                self.assertEqual(request.execution_contract["training_epochs"], 30)
                self.assertEqual(
                    request.execution_contract["training_split"], "full_train"
                )
                self.assertNotIn("training_hours", request.execution_contract)
            self.assertTrue(controller.replays)
            self.assertTrue(
                all(
                    phase in {"selection", "holdout"}
                    for phase, _, _ in controller.replays
                )
            )

    def test_migration_is_idempotent_and_does_not_mutate_source(self):
        original = {
            "task_id": "asr_en_wer",
            "full_training": {"enabled": True, "data": "/full", "epochs": 30},
            "execution_contract": {"training_hours": 100},
        }
        migrated = promote_full_training_to_search(original)
        self.assertEqual(original["execution_contract"]["training_hours"], 100)
        self.assertEqual(migrated, promote_full_training_to_search(migrated))
        self.assertEqual(
            migrated["base_models"]["asr_en_wer"]["source_paths"]["data"], "/full"
        )

    def test_full_mode_rejects_search_and_smoke_subsets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for marker in (
                {"features_ready": True, "parent_fingerprint": "full"},
                {"features_ready": True, "training_selection": "subset"},
            ):
                (root / "preparation.json").write_text(json.dumps(marker))
                with self.assertRaisesRegex(ValueError, "subset"):
                    validate_full_training_data(root)
            (root / "preparation.json").write_text(
                json.dumps({"features_ready": True, "training_selection": "full"})
            )
            validate_full_training_data(root)

    def test_training_estimates_count_all_four_candidates(self):
        result = training_cost_estimates(400, 30, 100)
        self.assertEqual(result["search_candidate_hours"], 120)
        self.assertEqual(result["six_round_training_hours"], 25 * 120)
        self.assertEqual(result["ten_round_training_hours"], 41 * 120)
        self.assertEqual(result["post_search_training_hours"], 0)

    def test_calibration_identity_changes_with_training_data_or_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "preparation.json").write_text('{"features_ready": true}')
            sure = {
                "task_id": "asr_en_wer",
                "base_models": {"asr_en_wer": {"source_paths": {"data": str(root)}}},
                "execution_env": {"SURE_MAX_TRAIN_EPOCHS": "10"},
            }
            before = training_signature(sure)
            sure["execution_env"]["SURE_MAX_TRAIN_EPOCHS"] = "30"
            self.assertNotEqual(before, training_signature(sure))

    def test_production_profiles_train_full_data_from_the_start(self):
        repo = Path(__file__).resolve().parents[4]
        for name in (
            "xlab-tedlium3-npu-evolution.yaml",
            "glm-tedlium3-npu-evolution.yaml",
            "ordinary-asr-cuda.yaml",
            "ordinary-asr-npu.yaml",
        ):
            with self.subTest(config=name):
                sure = yaml.safe_load(
                    (repo / "configs/sure_master" / name).read_text()
                )["sure"]
                self.assertEqual(sure["execution_env"]["SURE_MAX_TRAIN_EPOCHS"], "30")
                self.assertEqual(sure["execution_env"]["SURE_BASELINE_EPOCH"], "30")
                self.assertEqual(sure["execution_contract"]["training_epochs"], 30)
                self.assertEqual(sure["training_mode"], "full_during_search")
                self.assertNotIn(
                    "search_100h",
                    sure["base_models"]["asr_en_wer"]["source_paths"]["data"],
                )
                self.assertNotIn("initial_baseline_run", sure)
                self.assertNotIn("full_training", sure)


if __name__ == "__main__":
    unittest.main()
