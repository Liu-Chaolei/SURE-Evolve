from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
import json
import shlex
import os
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research_idea/scripts"))
from research_idea_lib.inputs import IdeaRequest as NativeRequest

CLIENT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "xlab_idea_client.py"
spec = importlib.util.spec_from_file_location("xlab_idea_client", CLIENT_PATH)
assert spec is not None and spec.loader is not None
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


class GenerateTests(unittest.TestCase):
    def setUp(self):
        preflight = patch.object(client, "validate_survey_resources", return_value={"status": "ready"})
        preflight.start()
        self.addCleanup(preflight.stop)

    @staticmethod
    def _candidate(index: int) -> dict[str, object]:
        return {
            "title": f"Independent candidate {index}",
            "abstract": f"Candidate {index} tests a distinct speech-model mechanism.",
            "core_contribution": f"Contribution {index} changes mechanism family {index}.",
            "research_question": f"Does mechanism family {index} improve WER?",
            "hypothesis": f"Mechanism family {index} improves WER through pathway {index}.",
            "method": f"Implement pathway {index} in an isolated candidate workspace.",
            "introduction": f"Motivation for independent candidate {index}.",
            "experiment_plan": [f"Ablate pathway {index}."],
            "data_requirements": ["Speech validation set."],
            "baselines": ["Current best SURE candidate."],
            "metrics": ["WER"],
            "risks": [f"Pathway {index} may overfit."],
            "components": [{"name": f"component-{index}"}],
            "algorithm": [{"name": f"algorithm-{index}"}],
            "reference_papers": [f"Evidence paper {index}"],
            "evidence_ids": [f"evidence-{index}"],
            "source_modes": ["moonshot_inventor", "bridge_builder"],
        }

    def _payload(self, run_root: str) -> dict[str, object]:
        return {
            "requested_idea_count": 4,
            "run_root": run_root,
            "task_description": "Test task",
            "request_id": "operation-1",
            "sure_run_id": "sure-run",
            "task_id": "task-1",
            "search_mode": "staged_axes",
            "axis": "arch",
            "round_index": 1,
            "metric": {"name": "wer", "direction": "lower"},
            "input_digest": "sha256:request",
        }

    @staticmethod
    def _review(index=1, domain="train"):
        return {"accepted": True, "reason": "supported and executable",
                "checks": {key: True for key in ("evidence_supported", "feasible", "distinct", "faithful")},
                "implementation_instructions": f"Apply intervention {index}",
                "change_set": [{"domain": domain, "target": f"module-{index}", "description": f"Intervention {index}"}],
                "requires_training": domain != "inference", "ablation": ["Remove the intervention"],
                "distinction": f"Distinct causal mechanism {index}"}

    def _run(self, root, candidates, reviews=None, payload=None):
        payload = payload or {**self._payload(str(root)), "search_mode": "ordinary", "axis": None}
        native_candidates = iter(candidates)
        contexts = []
        arguments = []
        native_calls = []
        def pipeline(**kwargs):
            native_calls.append(kwargs)
            arguments.append(kwargs["request"].discussion)
            branch = kwargs["run_dir"]
            (branch / "artifacts").mkdir(exist_ok=True)
            (branch / "artifacts/idea_result.json").write_text(json.dumps(next(native_candidates)))
            return {"status": "success"}
        review_iterator = iter(reviews) if reviews is not None else None
        def review(runtime, context):
            contexts.append(deepcopy(context))
            result = next(review_iterator) if review_iterator else self._review(len(contexts))
            return {"review": result, "usage": {"total_tokens": 1}, "trace": {"status": "success"}}
        with ExitStack() as stack:
            stack.enter_context(patch.object(client, "_survey_path", return_value=root / "survey.json"))
            stack.enter_context(patch.object(client, "load_runtime_config", return_value=object()))
            stack.enter_context(patch.object(client, "run_pipeline", side_effect=pipeline))
            stack.enter_context(patch.object(client, "run_paths", side_effect=lambda path: {"idea_result_json": path / "artifacts/idea_result.json"}))
            stack.enter_context(patch.object(client, "read_json", side_effect=lambda path: json.loads(path.read_text())))
            stack.enter_context(patch.object(client, "review_candidate", side_effect=review))
            batch = client.generate(payload, "operation-1")
            replay = client.generate(payload, "operation-1")
            self.assertEqual(batch, replay)
        return batch, contexts, arguments, native_calls

    def test_four_training_candidates_and_context_and_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch, contexts, arguments, calls = self._run(root, [self._candidate(i) for i in range(4)])
            self.assertEqual(len(calls), 4)
            self.assertEqual([len(c["accepted_candidates"]) for c in contexts], [0, 1, 2, 3])
            self.assertTrue(all(i["candidate_type"] == "fine_tune" for i in batch["ideas"]))
            self.assertTrue(all("Prioritize a structurally" not in arg for arg in arguments))
            for idea in batch["ideas"]:
                artifact = json.loads((root / idea["artifact_id"]).read_text())
                self.assertEqual(client.digest(artifact), idea["artifact_digest"])
                self.assertNotIn("candidate_type", idea["native_artifact"])
            sure_root = Path(__file__).resolve().parents[5] / "SURE-Evolve"
            if sure_root.is_dir():
                sys.path.insert(0, str(sure_root))
                from playground.sure_master.core.contracts import IdeaBatch, IdeaItem, IdeaSpec, IdeaRequest, MetricSpec, validate_idea_batch
                items = [IdeaItem(**{**item, "spec": IdeaSpec(**item["spec"])}) for item in batch["ideas"]]
                payload = {**self._payload(temporary), "search_mode": "ordinary", "axis": None}
                payload.pop("run_root")
                payload["metric"] = MetricSpec(**payload["metric"])
                validate_idea_batch(IdeaBatch(**{**batch, "ideas": items}), IdeaRequest(**payload))

    def test_duplicate_title_change_is_replaced_and_actual_attempt_is_referenced(self):
        with tempfile.TemporaryDirectory() as temporary:
            original = self._candidate(1)
            duplicate = {**deepcopy(original), "title": "renamed", "candidate_type": "arch"}
            batch, contexts, _, calls = self._run(Path(temporary), [original, duplicate, *[self._candidate(i) for i in range(2, 5)]])
            self.assertEqual(len(calls), 5)
            self.assertIn("attempt-3", batch["ideas"][1]["artifact_id"])
            self.assertIn("duplicate", contexts[1]["rejected_attempts"][0]["reason"])

    def test_semantic_duplicate_and_infeasible_reviews_are_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            rejected = [{"accepted": False, "reason": reason} for reason in ("same intervention, different wording", "only numeric learning rate change", "exceeds training budget")]
            batch, contexts, _, calls = self._run(Path(temporary), [self._candidate(i) for i in range(7)],
                [*rejected, *[self._review(i) for i in range(4)]])
            self.assertEqual(len(calls), 7)
            self.assertEqual(len(contexts[3]["rejected_attempts"]), 3)
            self.assertEqual(len(batch["ideas"]), 4)

    def test_budget_exhaustion_retains_incomplete_attempts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(client.AdapterError, "accepted 1 of 4 after 8"):
                self._run(root, [self._candidate(1)] * 8)
            state = json.loads((root / "operation-1/generation.json").read_text())
            self.assertEqual(state["status"], "incomplete")
            self.assertEqual(len(state["attempts"]), 8)
            self.assertFalse((root / "operation-1/batch.json").exists())

    def test_incomplete_native_run_is_not_regenerated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(client, "run_pipeline", return_value={"status": "incomplete"}) as pipeline, \
                 patch.object(client, "_survey_path", return_value=root), \
                 patch.object(client, "load_runtime_config", return_value=object()):
                with self.assertRaisesRegex(client.AdapterError, "attempt 1 was incomplete"):
                    client.generate(self._payload(temporary), "operation-1")
                with patch.object(client, "read_json", side_effect=lambda path: json.loads(path.read_text())):
                    with self.assertRaisesRegex(client.AdapterError, "inspect existing"):
                        client.generate(self._payload(temporary), "operation-1")
                self.assertEqual(pipeline.call_count, 1)

    def test_compound_classification_and_inconsistent_training(self):
        review = self._review()
        review["change_set"].append({"domain": "arch", "target": "encoder-dim", "description": "change width"})
        self.assertEqual(client.validate_review(review)["candidate_type"], "arch")
        with self.assertRaisesRegex(ValueError, "axis"):
            client.validate_review(review, "train")
        review["requires_training"] = False
        with self.assertRaisesRegex(ValueError, "training declaration"):
            client.validate_review(review)
        self.assertEqual(client.validate_review(self._review(domain="inference"))["candidate_type"], "inference")

    def test_staged_arch_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            batch, _, _, _ = self._run(Path(temporary), [self._candidate(i) for i in range(4)],
                [self._review(i, "arch") for i in range(4)], self._payload(temporary))
            self.assertTrue(all(idea["axis"] == "arch" and idea["candidate_type"] == "arch" for idea in batch["ideas"]))

    def test_mature_feedback_and_baseline_history(self):
        parent = self._candidate(1)
        history = [{"candidates": [{"score": 1.0, "failure_category": "system_failure"}]}]
        payload = {"task_description": "A task with an apostrophe: it's ASR", "prior_rounds": history, "current_best": {}}
        feedback = client.canonical({"rounds": history})
        args = shlex.split(client._native_arguments(payload, Path("/tmp/survey"), feedback))
        self.assertNotIn("--experiment-feedback", args)
        self.assertIn("system_failure", args[args.index("--discussion") + 1])
        payload["current_best"] = {"native_idea": parent, "native_idea_digest": client.digest(parent)}
        args = shlex.split(client._native_arguments(payload, Path("/tmp/survey"), feedback))
        self.assertEqual(json.loads(args[args.index("--mature-idea") + 1]), parent)
        self.assertEqual(json.loads(args[args.index("--experiment-feedback") + 1]), {"rounds": history})
        payload["current_best"]["native_idea_digest"] = "wrong"
        with self.assertRaisesRegex(client.AdapterError, "digest"):
            client._native_arguments(payload, Path("/tmp/survey"), feedback)

    def test_structured_request_preserves_equals_and_multiline_text(self):
        task = 'ASR seed=42\n--literal "quote"'
        payload = {"task_description": task, "execution_contract": {"candidate_type": "arch", "note": "a=b"}}
        request = client._native_request(payload, Path("/tmp/survey"))
        self.assertEqual(request.topic, task)
        self.assertIn('a=b', request.discussion)
        self.assertIsInstance(request, NativeRequest)

    def test_flow_first_publishes_four_without_deep_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def practical(payload, survey, branch, runtime, accepted, rejected):
                artifact = branch / 'advisory_candidate.json'
                candidate = self._candidate(len(accepted) + 1)
                candidate['research_mode'] = 'survey_analysis_direct'
                artifact.write_text(json.dumps(candidate))
                return artifact
            with patch.dict(os.environ, {'XLAB_SURE_FLOW_FIRST': '1'}), \
                 patch.object(client, 'generate_pragmatic_candidate', side_effect=practical):
                batch, _, _, calls = self._run(root, [], reviews=[self._review(i, 'arch') for i in range(4)])
            self.assertEqual(calls, [])
            self.assertEqual(len(batch['ideas']), 4)
            self.assertTrue(all(item['candidate_type'] == 'arch' for item in batch['ideas']))

    def test_summary_distinguishes_measurement_and_scientific_claim(self):
        candidates = [
            {"idea_id": "better", "final_status": "success", "improved": True, "rungs": [{"success": True, "score": 0.9}]},
            {"idea_id": "equal", "final_status": "success", "improved": False, "rungs": [{"success": True, "score": 1.0}]},
            {"idea_id": "worse", "final_status": "success", "improved": False, "rungs": [{"success": True, "score": 1.1}]},
            {"idea_id": "unknown", "final_status": "success", "improved": True},
            *[{"idea_id": category, "final_status": "failed", "failure_category": category} for category in ("system_failure", "metric_failure", "candidate_failure")],
        ]
        summary = client.summarize({"candidates": candidates, "result_digest": "measured"}, "round-1")
        self.assertEqual(summary["supported_mechanisms"], ["better"])
        self.assertEqual(summary["rejected_mechanisms"], [])
        self.assertEqual([r["outcome"] for r in summary["observations"]],
            ["improved", "no_improvement", "no_improvement", "inconclusive", "execution_failed", "execution_failed", "execution_failed"])

    def test_rejects_invalid_count_before_native_pipeline(self):
        with patch.object(client, "run_pipeline") as pipeline:
            with self.assertRaisesRegex(client.AdapterError, "positive"):
                client.generate({"requested_idea_count": 0}, "operation-1")
            pipeline.assert_not_called()


if __name__ == "__main__":
    unittest.main()
