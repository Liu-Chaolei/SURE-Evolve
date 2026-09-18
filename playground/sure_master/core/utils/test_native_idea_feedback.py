"""Controller regression for native parent snapshots and scoped outcome memory."""
import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import yaml

from playground.sure_master.core.contracts import RoundSummary
from playground.sure_master.core.utils.test_free_idea_generation import TrainingController, TrainingProvider
from playground.sure_master.core.utils import test_xlab_controller_smoke as smoke
from playground.sure_master.tools.prepare_sd_ablation import common_config
from playground.sure_master.core.xlab_client import XlabIdeaClient, XlabIdeaClientError


class NativeFeedbackTests(unittest.TestCase):
    def test_native_binding_rejects_changed_runtime_before_local_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            client = XlabIdeaClient(["fixture-provider"], workspace_root=temporary)
            result = {"status": "ready", "profile": "xlab.sure.native.v1", "identity": "sha256:first"}
            with patch("playground.sure_master.core.xlab_client.subprocess.run",
                       side_effect=lambda *a, **k: SimpleNamespace(returncode=0, stdout=json.dumps(result))):
                self.assertEqual(client.bind_native_policy({"engine": "native"}), "sha256:first")
                self.assertEqual(client.bind_native_policy({"engine": "native"}), "sha256:first")
                result["identity"] = "sha256:changed"
                with self.assertRaisesRegex(XlabIdeaClientError, "new deployment"):
                    client.bind_native_policy({"engine": "native"})

    def test_new_sd_config_uses_native_adapter_and_full_search_budget(self):
        config = common_config(Path("/tmp/native-config-only"), "registry/image@sha256:" + "0" * 64, Path("/tmp/test.env"))
        generation = config["xlab"]["idea_generation"]
        self.assertEqual(generation["engine"], "native")
        self.assertEqual(generation["mcts"]["max_iterations"], 64)
        self.assertIs(generation["allow_auxiliary_experiments"], False)
        provider = config["xlab"]["idea_provider"]
        self.assertTrue(any("xlab_idea_client.py" in part for part in provider["command"]))
        self.assertNotIn("XLAB_SURE_EVIDENCE_JSON", provider["environment"])
        self.assertIn("--check-survey", provider["preflight_command"])

    def test_controller_publishes_parent_scope_and_restores_native_memory(self):
        xlab = Path(__file__).resolve().parents[5] / "XLab"
        script = xlab / "xlab/skills/sure_master/scripts/xlab_idea_client.py"
        if not script.exists():
            self.skipTest("Adjacent XLab checkout required for cross-repository integration")
        sys.path.insert(0, str(script.parent))
        spec = importlib.util.spec_from_file_location("native_feedback_client", script)
        client = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(client)
        from research_idea_lib.common import assert_portable_output

        class NativeProvider(TrainingProvider):
            def generate(self, request):
                batch = super().generate(replace(request, generation_policy={"max_attempts": 6}))
                self.generated_requests[-1] = request
                modes = ["moonshot_inventor", "bridge_builder", "steady_engineer", "ambitious_realist", "evidence_first"]
                items = [replace(item, native_artifact={**item.native_artifact,
                    "research_policy": {"profile": "xlab.sure.native.v1", "evidence_mode": "local_literature"},
                    "source_modes": modes, "idea_source": "fused",
                    "mcts_evolution": {"iterations": [{"fixture": True}], "counters": {mode: {"iterations": 1} for mode in modes}}})
                    for item in batch.ideas]
                return replace(batch, ideas=items)

            def summarize(self, result):
                self.summarized_results.append(result)
                return RoundSummary(**client.summarize(asdict(result), result.result_digest))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, ref = root / "baseline.py", root / "ref.txt"
            baseline.write_text("# synthetic baseline\n")
            ref.write_text("utt-1\thello\n")
            config_path = smoke.ControllerSmokeTests()._write_config(root, baseline, ref)
            config = yaml.safe_load(config_path.read_text())
            config["sure"].pop("staged_axes")
            config["sure"].update(search_strategy="ordinary", search_scope="all", search_budget={"ideas_per_round": 3})
            config["sure"]["research_topic"] = "Improve synthetic speech recognition."
            config["max_research_rounds"] = 2
            config["xlab"]["idea_generation"] = {"engine": "native", "max_attempts": 6}
            config_path.write_text(yaml.safe_dump(config))
            provider = NativeProvider()
            controller = TrainingController(config_path=config_path, xlab_provider=provider)
            controller.set_run_dir(root / "run", task_id="native-feedback-fixture")
            with patch("playground.sure_master.core.playground.KnowledgePromotionExp.run", return_value="synthetic observations"):
                result = controller.run("Implementation context at /shared/private/source.py; no real training")
            self.assertEqual(result["status"], "completed", result)
            self.assertEqual(result["successful_training_candidates"], 6)
            self.assertEqual(len(provider.summarized_results), 2)
            for request, measured in zip(provider.generated_requests, provider.summarized_results):
                self.assertEqual(request.task_description, "Improve synthetic speech recognition.")
                self.assertIn("/shared/private/source.py", request.base_model_profile["implementation_task"])
                survey = xlab / ".xlab/runs/sd-literature/artifacts/survey.json"
                native = client._native_request(asdict(request), survey)
                serialized = native.to_json(xlab)
                self.assertEqual(serialized["survey_path"], ".xlab/runs/sd-literature/artifacts/survey.json")
                assert_portable_output(serialized)
                native.survey_path = xlab.parent / "literature_bundle/survey.json"
                with self.assertRaisesRegex(ValueError, "survey_path"):
                    assert_portable_output(native.to_json(xlab))
                self.assertEqual(measured.parent_snapshot["score"], request.current_best["score"])
                self.assertEqual(measured.parent_snapshot["solution_digest"], request.current_best["solution_digest"])
                self.assertEqual(measured.evaluation_context["memory_scope"], request.generation_policy["memory_scope"])
            memory = provider.generated_requests[1].prior_rounds[0]["summary"]["symbolic_memory"]
            self.assertEqual(len(memory), 3)
            self.assertTrue(all(item["kind"] == "candidate_outcome" for item in memory))
            self.assertEqual(len({item["record_id"] for item in memory}), 3)


if __name__ == "__main__":
    unittest.main()
