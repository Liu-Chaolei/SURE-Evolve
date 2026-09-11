from __future__ import annotations

import json
import gzip
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from playground.sure_master.tools.prepare_tedlium import (
    read_stm_split,
    reuse_precomputed_features,
    training_subset,
)
from playground.sure_master.runtime.accelerator import runtime_environment, prepare_npu_recipe
from playground.sure_master.runtime.data_view import evaluation_data_view
from playground.sure_master.tools.with_xlab_environment import xlab_environment
from playground.sure_master.tools.preflight import validate_xlab_survey
from .model_artifact import retain_model_artifact, restore_model_artifact
from .workspace_cleanup import cleanup_candidate_workspace, WorkspaceCleanupConfig
from . import test_xlab_controller_smoke as fixtures
from playground.sure_master.core.providers import FakeXlabIdeaProvider
from playground.sure_master.core.exp.run_exp import SureRunExp
from playground.sure_master.tools.run_icefall_zipformer_candidate import validate_candidate_args


class AsrRuntimeTests(unittest.TestCase):
    def test_extra_arguments_cannot_bypass_output_or_epoch_contract(self):
        for option in ("--exp-dir", "--num-epochs", "--world-size"):
            with self.assertRaises(ValueError):
                validate_candidate_args(candidate_type="fine_tune", action="train_decode",
                                        train_extra_args=[option, "99"], decode_extra_args=[])

    def test_metric_failure_reason_and_elapsed_time_reach_round_details(self):
        with tempfile.TemporaryDirectory() as temporary:
            exp = SimpleNamespace(workspace_path=temporary, candidate_stage_name="", stage="improve",
                candidate_phase="search", candidate_rung_name="", candidate_idea_id="idea-1",
                candidate_type_hint="inference", metric_feedback="metric failed", terminal_output="",
                _candidate_started_at=time.time() - 1,
                _cleanup_candidate_workspace=lambda **kwargs: None)
            details = {"error": "scorer unavailable"}
            SureRunExp._write_status(exp, False, "metric_failed", details=details)
            self.assertEqual(details["reason_code"], "metric_failed")
            self.assertGreaterEqual(details["runtime_seconds"], 1)

    def test_incomplete_xlab_survey_is_rejected_before_model_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "artifacts").mkdir()
            (root / "manifest.json").write_text(json.dumps({"status": "incomplete", "validation": {"passed": False}}))
            (root / "artifacts/survey_report.json").write_text(json.dumps({"passed": False}))
            with self.assertRaises(ValueError):
                validate_xlab_survey(root)
            (root / "manifest.json").write_text(json.dumps({"status": "success", "validation": {"passed": True}}))
            (root / "artifacts/survey_report.json").write_text(json.dumps({"passed": True}))
            validate_xlab_survey(root)

    def test_evaluation_view_filters_ids_without_changing_shared_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            (source / "fbank").mkdir(parents=True)
            manifest = source / "fbank/tedlium_cuts_dev.jsonl.gz"
            with gzip.open(manifest, "wt") as handle:
                for key in ("u1", "u2"):
                    handle.write(json.dumps({"id": key, "supervisions": [{"id": key}]}) + "\n")
            original = manifest.read_bytes()
            (source / "fbank/tedlium_cuts_train.jsonl.gz").write_bytes(b"training data")
            reference = root / "ref.txt"
            reference.write_text("u2\tREFERENCE\n")
            view = evaluation_data_view(source, root / "view", reference, ["dev"])
            with gzip.open(view / "fbank/tedlium_cuts_dev.jsonl.gz", "rt") as handle:
                self.assertEqual([json.loads(line)["id"] for line in handle], ["u2"])
            self.assertEqual(manifest.read_bytes(), original)
            self.assertTrue((view / "fbank/tedlium_cuts_train.jsonl.gz").is_symlink())

    def test_stm_intervals_and_ids_preserve_reference_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "legacy/dev"
            folder.mkdir(parents=True)
            (folder / "talk.sph").write_bytes(b"audio fixture")
            (folder / "talk.stm").write_text(
                ";; header\ntalk 1 speaker 1.2 2.5 <o> Hello world\n"
                "talk 1 speaker 3.0 4.0 <o> IGNORE_TIME_SEGMENT_IN_SCORING\n")
            rows = read_stm_split(root, "dev")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].key, "talk-1")
            self.assertAlmostEqual(rows[0].duration, 1.3)
            self.assertEqual(rows[0].channel, 0)
            self.assertEqual(rows[0].text, "Hello world")
            self.assertEqual(training_subset(rows, 1, 42), rows)

    def test_train_stm_directory_symlink_falls_back_to_flattened_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dev = root / "legacy/dev"
            dev.mkdir(parents=True)
            (dev / "dev.sph").write_bytes(b"audio fixture")
            (dev / "dev.stm").write_text("dev 1 speaker 0.0 1.0 <o> Dev text\n")
            data = root / "data"
            (data / "stm").mkdir(parents=True)
            (data / "sph").mkdir()
            (data / "train.sph").write_bytes(b"audio fixture")
            (data / "sph/train.sph").write_bytes(b"audio fixture")
            (data / "stm/train.stm").write_text(
                "train 1 speaker 0.0 1.0 <o> Train text\n"
            )
            train = root / "legacy/train"
            train.mkdir(parents=True)
            (train / "stm").symlink_to(Path("../../data/stm"), target_is_directory=True)
            (train / "sph").symlink_to(Path("../../data/sph"), target_is_directory=True)

            rows = read_stm_split(root, "train")

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].recording_id, "train")
            self.assertEqual(rows[0].text, "Train text")

    def test_reuses_lowercase_icefall_features_without_copying_archives(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source/data"
            fbank = source / "fbank"
            bpe = root / "bpe"
            output = root / "prepared"
            fbank.mkdir(parents=True)
            bpe.mkdir()
            (bpe / "bpe.model").write_bytes(b"tokenizer")
            for split in ("train", "dev", "test"):
                feature_dir = fbank / f"tedlium_feats_{split}"
                feature_dir.mkdir()
                (feature_dir / "feats-0.lca").write_bytes(b"features")
                with gzip.open(
                    fbank / f"tedlium_cuts_{split}_lowercase.jsonl.gz", "wt"
                ) as stream:
                    stream.write(
                        json.dumps(
                            {
                                "id": f"{split}-1",
                                "features": {
                                    "num_features": 80,
                                    "storage_path": f"data/fbank/{feature_dir.name}/feats-0.lca",
                                },
                            }
                        )
                        + "\n"
                    )

            metadata = reuse_precomputed_features(source, bpe, output)

            self.assertTrue(metadata["feature_reuse"])
            self.assertEqual(metadata["feature_manifest_rows"]["train"], 1)
            self.assertTrue((output / "lang_bpe_500").is_symlink())
            self.assertTrue((output / "fbank/tedlium_feats_train").is_symlink())
            self.assertTrue((output / "fbank/tedlium_cuts_train.jsonl.gz").is_symlink())

    def test_retained_checkpoint_survives_cleanup_and_restores_independently(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "models/zipformer/epoch-1.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"trained weights")
            (root / "models/helper.py").write_text("MODEL_SETTING = 1\n")
            (root / "working").mkdir()
            (root / "working/helper.py").write_text("WORKING_SETTING = 2\n")
            bpe = root / "data/lang_bpe_500/bpe.model"
            bpe.parent.mkdir(parents=True)
            bpe.write_bytes(b"tokenizer")
            retained = retain_model_artifact(root)
            cleanup_candidate_workspace(root, WorkspaceCleanupConfig(enabled=True), success=True, reason_code="success")
            self.assertFalse(checkpoint.exists())
            self.assertEqual(Path(retained["candidate_checkpoint"]).read_bytes(), b"trained weights")
            epoch, saved_bpe, metadata = restore_model_artifact(retained["model_artifact"], root / "replay")
            self.assertEqual(epoch, 1)
            self.assertEqual(saved_bpe.read_bytes(), b"tokenizer")
            (root / "replay/epoch-1.pt").write_bytes(b"different")
            self.assertEqual(Path(retained["candidate_checkpoint"]).read_bytes(), b"trained weights")
            restored = root / "full_replay"
            restore_model_artifact(retained["model_artifact"], restored / "models/zipformer", workspace=restored)
            self.assertEqual((restored / "models/helper.py").read_text(), "MODEL_SETTING = 1\n")
            self.assertEqual((restored / "working/helper.py").read_text(), "WORKING_SETTING = 2\n")

    def test_runtime_does_not_treat_npu_as_cuda(self):
        env = runtime_environment({"accelerator": "npu", "devices": [3], "python": "worker-python"})
        self.assertEqual(env["SURE_ACCELERATOR"], "npu")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "-1")
        self.assertEqual(env["SURE_WORKER_PYTHON"], "worker-python")
        self.assertNotIn("SURE_BASELINE_WORLD_SIZE", env)

    def test_npu_recipe_adaptation_does_not_write_through_source_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / "source", root / "target"
            source.mkdir()
            text = "import k2\nimport torch\nk2.rnnt_loss_smoothed()\nk2.rnnt_loss_pruned()\nk2.get_rnnt_prune_ranges()\n"
            (root / "original_model.py").write_text(text)
            (source / "model.py").symlink_to(root / "original_model.py")
            for name in ("train.py", "decode.py", "scaling.py"):
                (source / name).write_text("import k2\nimport torch\n")
            with patch.dict(sys.modules, {"torch_npu": SimpleNamespace()}), patch.dict(
                os.environ, {"SURE_ASR_RECIPE_PROFILE": "tedlium3_zipformer", "SURE_USE_FP16": "0", "SURE_BASELINE_WORLD_SIZE": "1"}
            ), patch("playground.sure_master.runtime.accelerator.check_accelerator", return_value={}):
                prepare_npu_recipe(source, target)
            self.assertEqual((root / "original_model.py").read_text(), text)
            self.assertFalse((target / "model.py").is_symlink())
            self.assertIn("npu_k2.rnnt_loss_pruned", (target / "model.py").read_text())
            for path in target.glob("*.py"):
                compile(path.read_text(), str(path), "exec")

    def test_xlab_credential_is_not_sent_to_an_unrelated_endpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "settings.json").write_text(json.dumps({"defaultProvider": "test", "defaultModel": "model"}))
            (root / "models.json").write_text(json.dumps({"providers": {"test": {"baseUrl": "https://expected.example/v1"}}}))
            (root / "auth.json").write_text(json.dumps({"test": {"type": "api_key", "key": "dummy-test-credential"}}))
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(xlab_environment(root)["OPENAI_BASE_URL"], "https://expected.example/v1")
            with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://other.example/v1"}, clear=True):
                with self.assertRaises(ValueError):
                    xlab_environment(root)

    def test_two_ordinary_xlab_rounds_publish_scores_and_non_improving_successes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, ref = root / "baseline.py", root / "ref.txt"
            baseline.write_text("# baseline\n")
            ref.write_text("u\thello\n")
            config_path = fixtures.ControllerSmokeTests()._write_config(root, baseline, ref)
            config = yaml.safe_load(config_path.read_text())
            config["sure"]["search_strategy"] = "ordinary"
            config["sure"]["staged_axes"]["enabled"] = False
            config["max_research_rounds"] = 2
            config_path.write_text(yaml.safe_dump(config))
            provider = FakeXlabIdeaProvider()
            class OrdinaryController(fixtures._CpuFakeSureMaster):
                def _setup_agents(self):
                    super()._setup_agents()
                    self.agents.knowledge_promotion_agent = object()
            controller = OrdinaryController(config_path=config_path, xlab_provider=provider)
            controller.set_run_dir(root / "run", task_id="test")
            with patch("playground.sure_master.core.playground.KnowledgePromotionExp",
                       return_value=SimpleNamespace(exp_name="knowledge", run=lambda **kwargs: "measured feedback")):
                result = controller.run("A fake controller regression, not a model benchmark")
            self.assertEqual(result["status"], "completed", result)
            self.assertEqual(len(provider.summarized_results), 2)
            for round_result in provider.summarized_results:
                self.assertEqual(len(round_result.candidates), 4)
                self.assertTrue(all(c.rungs[0].score is not None for c in round_result.candidates))
                self.assertTrue(all(c.final_status == "success" for c in round_result.candidates))
                scores = {c.idea_id: c.rungs[0].score for c in round_result.candidates}
                self.assertEqual([scores[k] for k in round_result.ranking], sorted(scores.values()))
                self.assertTrue(all(c.idea_id.startswith("fake-ordinary-") for c in round_result.candidates))
            self.assertTrue(provider.generated_requests[1].prior_rounds)


if __name__ == "__main__":
    unittest.main()
