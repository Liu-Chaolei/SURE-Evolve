"""Exercise the JSONL adapter with production native workflow and fake models."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_sure_native import SureProvider, round_result, idea
from research_idea_lib.algorithm.workflow import OP_REPLAN

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sure_master/scripts"))
import xlab_idea_client as client


class AdapterProvider(SureProvider):
    def _payload(self, request):
        if request.operation == "xlab.sure.candidate_review.v1":
            return {"accepted": True, "reason": "Scoped and executable local modification.",
                    "checks": {"evidence_supported": None, "feasible": True, "distinct": True, "faithful": True},
                    "implementation_instructions": "Execute the existing candidate wrapper once with the local encoder change.",
                    "change_set": [{"domain": "arch", "target": "encoder", "description": "Refine encoder mechanism"}],
                    "requires_training": True, "ablation": [], "distinction": "Local mechanism change."}
        return super()._payload(request)


class NativeAdapterTests(unittest.TestCase):
    def test_two_rounds_publish_native_memory_without_experiment_side_effects(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"OPENAI_API_KEY": "unit-fixture-only", "XLAB_SURE_FLOW_FIRST": "0"}):
            parent = idea("Current best search", "root-core")
            scope = round_result()["evaluation_context"]["memory_scope"]
            payload = {"request_id": "r1", "sure_run_id": "run-A", "task_id": "asr", "search_mode": "ordinary",
                       "round_index": 1, "run_root": temporary, "requested_idea_count": 1,
                       "task_description": "Refine the current best speech encoder.", "input_digest": "sha256:request-1",
                       "current_best": {"native_idea": parent, "native_idea_digest": client.digest(parent),
                                        "implementation": "def encoder(x): return x", "score": 10,
                                        "solution_digest": "sha256:parent", "memory_scope": scope},
                       "execution_contract": {"allowed_change_domains": ["arch"]},
                       "generation_policy": {"engine": "native", "max_attempts": 1, "memory_scope": scope,
                           "mcts": {"max_iterations": 2, "max_depth": 1, "branching_factor": 1},
                           "ablation": {"use_literature": False, "use_feedback": True}}}
            provider = AdapterProvider()
            payload["generation_policy"]["native_identity"] = client.native_identity(payload)["identity"]
            with patch("research_idea_lib.pipeline.OpenAICompatibleProvider", return_value=provider), \
                 patch("sure_idea_review.OpenAICompatibleProvider", return_value=provider), \
                 patch.object(client, "validate_survey_resources", side_effect=AssertionError("B read survey")), \
                 patch("subprocess.Popen", side_effect=AssertionError("research launched a job")):
                batch = client.generate(payload, "r1")
                self.assertEqual(batch["status"], "success")
                self.assertEqual(batch["ideas"][0]["spec"]["ablation"], [])
                self.assertEqual(batch["ideas"][0]["evidence_refs"], [])
                self.assertEqual(len(batch["ideas"][0]["native_artifact"]["source_modes"]), 5)
                call_count = len(provider.requests)
                self.assertEqual(client.generate(payload, "r1"), batch)
                self.assertEqual(len(provider.requests), call_count)
                with patch.dict(os.environ, {"XLAB_RESEARCH_IDEA_AGENT_MODEL": "changed-model"}):
                    with self.assertRaisesRegex(client.AdapterError, "identity changed"):
                        client.generate(payload, "r1")
                self.assertEqual(len(provider.requests), call_count)
                measured = round_result()
                measured["candidates"][0]["idea"] = batch["ideas"][0]
                summary = client.summarize(measured, "summary-r1")
                self.assertEqual(summary["symbolic_memory"][0]["outcome"], "improved")
                payload.update(request_id="r2", round_index=2, input_digest="sha256:request-2",
                               prior_rounds=[{"summary": summary}])
                second = client.generate(payload, "r2")
                self.assertEqual(second["status"], "success")
                replans = [r for r in provider.requests if r.operation == OP_REPLAN]
                self.assertTrue(replans)
                feedback = json.loads(replans[-1].structured_input["experiment_feedback"])
                self.assertEqual(feedback["records"][0]["candidate_id"], "candidate-1")

    def test_flow_first_conflict_is_rejected_before_provider_work(self):
        with patch.dict(os.environ, {"XLAB_SURE_FLOW_FIRST": "1"}):
            with self.assertRaisesRegex(client.AdapterError, "conflicts"):
                client.generate({"generation_policy": {"engine": "native"}}, "conflict")


if __name__ == "__main__":
    unittest.main()
