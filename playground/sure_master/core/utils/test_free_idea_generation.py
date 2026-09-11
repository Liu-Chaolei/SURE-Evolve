"""Controller regressions for reviewed free exploration (no real model calls)."""
from __future__ import annotations

import json
import gzip
import tempfile
import unittest
from types import SimpleNamespace
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import yaml

from playground.sure_master.core.contracts import IdeaSpec, validate_idea_batch
from playground.sure_master.core.history import XlabHistoryJournal
from playground.sure_master.core.providers import FakeXlabIdeaProvider
from playground.sure_master.core.utils.fingerprints import digest
from playground.sure_master.core.utils import test_xlab_controller_smoke as smoke
from playground.sure_master.tools.preflight import check_config


class TrainingProvider(FakeXlabIdeaProvider):
    def generate(self, request):
        batch = super().generate(request)
        ideas = []
        for item in batch.ideas:
            changes = [{"domain": "train", "target": item.idea_id, "description": item.mechanism}]
            ideas.append(replace(item, candidate_type="fine_tune",
                                 native_artifact={"title": item.title, "components": [{"name": item.idea_id}], "risks": ["limited evidence"]},
                                 spec=replace(item.spec, change_set=changes, change_domains=["train"], requires_training=True)))
        result = replace(batch, ideas=ideas)
        validate_idea_batch(result, request)
        return result


class TrainingExp(smoke._FakeCandidateExp):
    def run(self, **kwargs):
        Path(self.workspace_path).mkdir(parents=True, exist_ok=True)
        number = int(self.exp_name.split("_")[1])
        return True, 1 - number / 100, None, f"# test implementation {number}\n", {"runtime_seconds": 0.01}


class TrainingController(smoke._CpuFakeSureMaster):
    def _setup_agents(self):
        super()._setup_agents()
        self.agents.knowledge_promotion_agent = object()

    def _create_run_exp(self, stage, exp_index):
        return TrainingExp(Path(self.session.config.workspace_path), f"exp_{exp_index}_{stage}")


class FreeIdeaControllerTests(unittest.TestCase):
    def test_two_rounds_execute_all_four_and_restore_parent_feedback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, ref = root / "baseline.py", root / "ref.txt"
            baseline.write_text("# baseline\n")
            ref.write_text("utt-1\thello\n")
            config_path = smoke.ControllerSmokeTests()._write_config(root, baseline, ref)
            config = yaml.safe_load(config_path.read_text())
            config["sure"].pop("staged_axes")
            config["sure"].update(search_strategy="ordinary", require_training_candidate=True,
                                  candidate_types={"max_fine_tune_per_round": 1})
            config.update(max_research_rounds=2, max_ideas_per_direction=1)
            config["xlab"]["idea_generation"] = {"max_attempts": 6}
            config_path.write_text(yaml.safe_dump(config))
            provider = TrainingProvider()
            controller = TrainingController(config_path=config_path, xlab_provider=provider)
            controller.set_run_dir(root / "run", task_id="free-idea-test")
            with patch("playground.sure_master.core.playground.KnowledgePromotionExp.run", return_value="test findings"):
                result = controller.run("Explicitly synthetic feedback replay, no ASR training")
            self.assertEqual(result["status"], "completed", result)
            self.assertEqual(result["successful_training_candidates"], 8)
            self.assertEqual([len(item.candidates) for item in provider.summarized_results], [4, 4])
            first, second = provider.generated_requests
            self.assertNotIn("native_idea", first.current_best)
            self.assertEqual(second.generation_policy, {"max_attempts": 6})
            self.assertEqual(digest(second.current_best["native_idea"]), second.current_best["native_idea_digest"])
            self.assertEqual(second.current_best["native_idea"]["risks"], "limited evidence")
            self.assertEqual(second.prior_rounds[0]["candidates"][0]["idea"]["native_artifact"]["risks"], ["limited evidence"])
            self.assertEqual(second.prior_rounds[0]["baseline_score"], result["baseline_score"])
            self.assertIn("summary", second.prior_rounds[0])
            workspace = Path(controller.session.config.workspace_path)
            journal = XlabHistoryJournal(workspace / "artifacts/xlab_history.json")
            controller._xlab_history = journal.round_history()
            restored = controller._xlab_request(task_description="restarted", search_mode="ordinary",
                                                round_index=3, requested_idea_count=4)
            self.assertEqual(restored.current_best["native_idea_digest"], digest(restored.current_best["native_idea"]))
            self.assertEqual(len(list((workspace / "artifacts/xlab_batches").glob("*.json"))), 2)

    def test_metadata_candidates_bypass_remote_type_cap(self):
        controller = object.__new__(TrainingController)
        ideas = [(str(i), "instructions") for i in range(4)]
        controller._xlab_idea_metadata = {idea: {"candidate_type": "fine_tune"} for idea in ideas}
        entries = controller._filter_and_order_ideas(ideas, mixed_enabled=True,
            round_candidate_counts={}, round_candidate_limits={"fine_tune": 1})
        self.assertEqual(len(entries), 4)

    def test_spec_rejects_mismatched_execution_domains(self):
        with self.assertRaisesRegex(ValueError, "training declaration"):
            IdeaSpec("change structure", change_domains=["arch"], requires_training=False)
        with self.assertRaisesRegex(ValueError, "concrete changes"):
            IdeaSpec("train", change_domains=["train"], requires_training=True,
                     change_set=[{"domain": "inference"}])

    def test_resource_preflight_blocks_before_model_probe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "lang_bpe_500").mkdir()
            (root / "lang_bpe_500/bpe.model").write_text("fixture")
            (root / "fbank").mkdir()
            for split in ("train", "dev", "test"):
                with gzip.open(root / f"fbank/tedlium_cuts_{split}.jsonl.gz", "wt") as handle:
                    handle.write('{}\n')
            (root / "ref.txt").write_text("u1\thello\n")
            config = {"sure": {"root": str(root), "base_models": {"asr_en_wer": {"source_paths": {"data": str(root)}}},
                "inputs": {"ref": str(root / "ref.txt")}, "task_cards_path": "unused", "task_id": "asr_en_wer"},
                "xlab": {"enabled": True, "idea_provider": {"command": ["native-python", "adapter.py"],
                    "preflight_command": ["native-python", "adapter.py", "--check-survey", str(root)],
                    "environment": {"OPENAI_API_KEY": "test-placeholder", "XLAB_SURE_SURVEY_PATH": str(root)}}}}
            metric = SimpleNamespace(run=lambda *args: SimpleNamespace(success=True, score=1 / 3))
            with patch("playground.sure_master.tools.preflight.validate_xlab_survey"), \
                 patch("playground.sure_master.core.utils.metric.SureMetricRunner", return_value=metric), \
                 patch("playground.sure_master.core.utils.task_cards.resolve_task_card", return_value=SimpleNamespace(canonical_task="asr")), \
                 patch("playground.sure_master.tools.preflight.subprocess.run", return_value=SimpleNamespace(
                     returncode=2, stdout="missing resource_manifest", stderr="")) as process:
                with self.assertRaisesRegex(ValueError, "resource_manifest"):
                    check_config(config)
                self.assertEqual(process.call_count, 1)
                self.assertIn("--check-survey", process.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
