"""Offline scientific-boundary and resource regressions for SD ablations."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from playground.sure_master.core.ablation import policy, research_view
from playground.sure_master.core.contracts import IdeaBatch, IdeaItem, IdeaSpec, validate_idea_batch
from playground.sure_master.core.training import SD_TRAINING, validate_training_config
from playground.sure_master.core.utils.sd_allocations import AUTHORIZED, validate_jobs
from playground.sure_master.runtime.sd_evolution import validate_source, PROTECTED
from playground.sure_master.tools.sd_ablation_ideas import evidence_context, generate
from playground.sure_master.tools.prepare_sd_ablation import common_config
from playground.sure_master.core.utils.test_free_idea_generation import TrainingController, TrainingProvider
from playground.sure_master.core.utils import test_xlab_controller_smoke as smoke


class AblationTests(unittest.TestCase):
    def test_policy_rejects_truthy_strings_and_unknown_switches(self):
        self.assertEqual(policy({}), {"use_literature": True, "use_feedback": True})
        for invalid in ({"use_feedback": "false"}, {"disable": True}):
            with self.assertRaises(ValueError):
                policy({"ablation": invalid})

    def test_no_feedback_projects_only_baseline_without_mutation(self):
        baseline = {"code": "initial"}
        result = research_view({"ablation": {"use_feedback": False}}, baseline,
            {"code": "LEAK-best"}, [{"score": "LEAK-score"}], ["LEAK-summary"], ["LEAK-parent"])
        self.assertNotIn("LEAK", json.dumps(result))
        result["current_best"]["code"] = "changed"
        self.assertEqual(baseline["code"], "initial")

    def test_no_literature_never_reads_files_even_with_survey_environment(self):
        with patch.dict(os.environ, {"XLAB_SURE_SURVEY_PATH": "/real/survey.json"}), \
             patch.object(Path, "read_text", side_effect=AssertionError("Literature read")):
            self.assertEqual(evidence_context(False, Path("/missing"))["evidence"], [])

    def test_all_authorized_jobs_and_no_fallback_targets(self):
        self.assertEqual(len(AUTHORIZED), 8)
        self.assertNotIn("10089", AUTHORIZED)
        settings = {"existing_allocations": sorted(AUTHORIZED), "allocation_overlap": False}
        self.assertEqual(len(validate_jobs(settings)), 8)
        for invalid in (["10089"], ["9980"], ["10085", "10085"], []):
            with self.assertRaises(ValueError):
                validate_jobs({**settings, "existing_allocations": invalid})

    def test_unallocated_restricted_pool_waits_instead_of_returning_to_sbatch(self):
        from playground.sure_master.core.utils.slurm_allocations import run_in_allocations
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = {"existing_allocations": ["10094"], "sd_restricted_pool": True,
                        "allocation_overlap": False, "allocation_lock_dir": str(root / "locks"),
                        "allocation_fallback_when_busy": True}
            with patch("playground.sure_master.core.utils.slurm_allocations.owned_running", return_value=False), \
                 patch("playground.sure_master.core.utils.slurm_allocations.time.sleep", side_effect=InterruptedError("waited")), \
                 patch("playground.sure_master.core.utils.slurm_allocations.subprocess.Popen") as launch:
                with self.assertRaisesRegex(InterruptedError, "waited"):
                    run_in_allocations(settings, ["srun"], root, root / "result.json")
                launch.assert_not_called()

    def test_open_recipe_allows_methods_but_not_budget_changes(self):
        training = {**deepcopy(SD_TRAINING), "recipe": "diarizen.evolution.v1",
                    "candidate_options": {"weight_decay": 0.1}, "learning_rate_network": 0.002}
        self.assertEqual(validate_training_config("sd.diarizen", training), training)
        for key, value in (("epochs", 200), ("world_size", 8), ("seed", 42), ("learning_rate_network", -1)):
            with self.assertRaises(ValueError):
                validate_training_config("sd.diarizen", {**training, key: value})
        self.assertEqual(validate_training_config("sd.diarizen", deepcopy(SD_TRAINING)), SD_TRAINING)

    def test_model_source_can_extend_beyond_original_knobs(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            pristine, candidate = root / "pristine", root / "candidate"
            for directory in (pristine, candidate):
                for name in PROTECTED:
                    target = directory / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text("# protected\n")
                (directory / "diarizen/model.py").write_text("class Model: pass\n")
            (candidate / "diarizen/model.py").write_text("class NewAttentionModel: pass\n")
            changes = validate_source(candidate, pristine, requires_training=True)
            self.assertIn("diarizen/model.py", changes)
            with self.assertRaisesRegex(ValueError, "require training"):
                validate_source(candidate, pristine, requires_training=False)
            (candidate / PROTECTED[0]).write_text("# skipped dataset\n")
            with self.assertRaisesRegex(ValueError, "Protected"):
                validate_source(candidate, pristine, requires_training=True)

    def test_common_config_budget_and_resource_envelope(self):
        config = common_config(Path("/tmp/sd-test"), "registry/app@sha256:abc", Path("/tmp/no.env"))
        sure = config["sure"]
        self.assertEqual(sure["search_scope"], "all")
        self.assertEqual(sure["search_budget"]["max_rounds"], 6)
        self.assertEqual(sure["search_budget"]["ideas_per_round"], 4)
        validate_jobs(sure["slurm"])
        self.assertEqual(sure["slurm"]["resource_profiles"]["training"]["cpu"], 64)
        self.assertEqual(sure["slurm"]["resource_profiles"]["training"]["temporary"], "100G")

    def test_two_round_controller_keeps_feedback_out_and_archives_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, ref = root / "baseline.py", root / "ref.txt"
            baseline.write_text("# baseline\n")
            ref.write_text("utt-1\thello\n")
            config_path = smoke.ControllerSmokeTests()._write_config(root, baseline, ref)
            config = yaml.safe_load(config_path.read_text())
            config["sure"].pop("staged_axes")
            config["sure"].update(search_strategy="ordinary", search_scope="all",
                ablation={"use_literature": False, "use_feedback": False},
                search_budget={"ideas_per_round": 3})
            config["max_research_rounds"] = 2
            config_path.write_text(yaml.safe_dump(config))
            provider = TrainingProvider()
            controller = TrainingController(config_path=config_path, xlab_provider=provider)
            controller.set_run_dir(root / "run")
            with patch("playground.sure_master.core.playground.KnowledgePromotionExp.run", side_effect=AssertionError("Summary leak")):
                result = controller.run("offline SD policy regression")
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(provider.generated_requests), 2)
            self.assertEqual(len(provider.summarized_results), 0)
            for request in provider.generated_requests:
                self.assertFalse(request.prior_rounds)
                self.assertFalse(request.parent_lineage)
                self.assertNotIn("score", request.current_best)
            self.assertEqual(provider.generated_requests[0].current_best, provider.generated_requests[1].current_best)
            self.assertTrue((Path(controller.session.config.workspace_path) / "metric/ablation_round_2.json").exists())

    def test_generation_budget_returns_partial_without_fabricated_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            payload = {"request_id": "test", "input_digest": "sha256:test", "requested_idea_count": 4,
                "generation_policy": {"max_attempts": 8, "ablation": {"use_literature": False, "use_feedback": False}},
                "execution_contract": {}, "current_best": {"implementation": "baseline"}}
            calls = []
            def call(stage, path, context, prompt):
                calls.append(stage)
                self.assertEqual(context.get("evidence_mode"), "none")
                if stage == "agent":
                    return {"questions": ["hypothesis"]}
                return {"title": "invalid candidate"}
            result = generate(payload, Path(temporary), Path("/no/xlab"), call)
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["ideas"], [])
            self.assertEqual(calls.count("generation"), 8)


if __name__ == "__main__":
    unittest.main()
