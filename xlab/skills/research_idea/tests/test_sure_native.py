"""Hermetic SURE feedback integration through real native search and fusion."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sure_master/scripts"))

from test_native_pipeline_integration import (
    DeterministicNativeProvider, idea, repository_fixture, novelty_runtime_fixture,
)
from research_idea_lib.algorithm.memory import MemoryState, CandidateOutcomeRecord
from research_idea_lib.algorithm.runtime_adapters import run_native_workflow
from research_idea_lib.algorithm.provider_adapter import IDEA_GENERATE_OPERATION, IDEA_DIAGNOSTIC_OPERATION
from research_idea_lib.algorithm.workflow import OP_ANALYSIS, OP_QUERY, OP_REPLAN, OP_MATERIALIZATION
from research_idea_lib.common import read_json, run_paths
from research_idea_lib.config import load_runtime_config
from research_idea_lib.inputs import IdeaRequest, request_from_json
from research_idea_lib.pipeline import run_pipeline
from research_idea_lib.research_policy import ResearchPolicy
from native_sure import outcome_records, feedback_records


class SureProvider(DeterministicNativeProvider):
    def _payload(self, request):
        result = super()._payload(request)
        if request.operation == OP_ANALYSIS and request.structured_input.get("mature_idea"):
            result["root_idea"] = deepcopy(request.structured_input["mature_idea"])
        if request.operation == OP_MATERIALIZATION:
            result["idea_result"]["experiment_plan"] = ["Execute one SURE candidate under the fixed training and search evaluation budget."]
            if request.structured_input.get("research_policy", {}).get("evidence_mode") == "task_only":
                result["idea_result"]["reference_papers"] = []
        return result


def round_result(score=8.0, **changes):
    scope = {"sure_run_id": "run-A", "task_id": "asr", "evaluation_digest": "scope-v1"}
    value = {"sure_run_id": "run-A", "result_digest": "sha256:result-1",
             "metric": {"name": "WER", "direction": "lower"},
             "parent_snapshot": {"solution_digest": "sha256:parent", "score": 10.0, "memory_scope": scope},
             "evaluation_context": {"memory_scope": scope},
             "candidates": [{"idea_id": "candidate-1", "idea_artifact_id": "candidate.json",
                             "code_digest": "sha256:actual", "final_status": "success",
                             "rungs": [{"name": "search", "success": True, "score": score}],
                             "idea": {"spec": {"change_set": [
                                 {"domain": "arch", "target": "encoder", "description": "Local encoder modification"},
                                 {"domain": "train", "target": "loss", "description": "Joint loss modification"}]}}}]}
    value.update(changes)
    return value


class CandidateMemoryTests(unittest.TestCase):
    def test_directions_parent_scope_failure_and_deduplication(self):
        for score, expected in ((8, "improved"), (10, "tied"), (12, "worse")):
            with self.subTest(score=score):
                record = outcome_records(round_result(score))[0]
                self.assertEqual(record["outcome"], expected)
                self.assertEqual(len(record["change_set"]), 2)
                self.assertNotIn("component", record)
                self.assertNotIn("op", record)
                memory = MemoryState.from_experiment_feedback({"feedback_kind": "candidate_outcome", "records": [record, record]})
                self.assertEqual(len(memory.candidate_records), 1)
                self.assertEqual(memory.symbolic_records, [])
                self.assertIn("no component causality", memory.hints()[0])
        high = round_result(12, metric={"name": "accuracy", "direction": "higher"})
        self.assertEqual(outcome_records(high)[0]["outcome"], "improved")
        mismatch = round_result(parent_snapshot={"solution_digest": "sha256:parent", "score": 10, "memory_scope": {"other": True}})
        self.assertEqual(outcome_records(mismatch)[0]["outcome"], "incomparable")
        failure = round_result()
        failure["candidates"][0]["failure_category"] = "system_failure"
        self.assertEqual(outcome_records(failure)[0]["outcome"], "execution_failed")
        self.assertEqual(outcome_records(round_result(float("nan")))[0]["outcome"], "incomparable")

    def test_scoped_history_cannot_leak_across_groups(self):
        record = outcome_records(round_result())[0]
        payload = {"generation_policy": {"memory_scope": record["scope"]},
                   "prior_rounds": [{"summary": {"symbolic_memory": [record]}}]}
        self.assertEqual(len(feedback_records(payload)["records"]), 1)
        payload["generation_policy"]["memory_scope"] = {**record["scope"], "sure_run_id": "run-B"}
        self.assertEqual(feedback_records(payload)["records"], [])
        with self.assertRaises(ValueError):
            CandidateOutcomeRecord(**{**record, "outcome": "worse"})


class NativeSureWorkflowTests(unittest.TestCase):
    def runtime(self):
        return replace(load_runtime_config(), max_iterations=2, max_depth=1, branching_factor=1)

    def request(self, *, task_only=True, feedback=""):
        memory = {"feedback_kind": "candidate_outcome", "records": outcome_records(round_result())} if feedback else {}
        return IdeaRequest(
            survey_path=None if task_only else Path("survey"), topic="Scientific agents",
            mature_idea=json.dumps(idea("Current best search", "root-core")),
            research_policy=asdict(ResearchPolicy(evidence_mode="task_only" if task_only else "local_literature")),
            task_context={"current_best": {"solution_digest": "sha256:parent", "implementation": "def encoder(x): return x"},
                          "execution_contract": {"allowed_change_domains": ["arch"]},
                          "experiment_memory": memory},
            experiment_feedback=json.dumps(memory) if feedback else "",
        )

    def test_task_only_pipeline_publishes_five_modes_without_any_literature(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"OPENAI_API_KEY": "unit-fixture-only"}):
            root = Path(temporary)
            for index in (1, 2):
                provider = SureProvider()
                request = self.request(feedback="yes" if index == 2 else "")
                restored = request_from_json(request.to_json(root), root)
                self.assertIsNone(restored.survey_path)
                self.assertEqual(restored.task_context, request.task_context)
                with patch("research_idea_lib.pipeline.OpenAICompatibleProvider", return_value=provider), \
                     patch("research_idea_lib.pipeline.SurveyArtifactRepository.from_request", side_effect=AssertionError("literature read")), \
                     patch("research_idea_lib.algorithm.runtime_adapters.build_component_novelty_runtime", side_effect=AssertionError("literature index")):
                    result = run_pipeline(cwd=root, run_dir=root / str(index), run_id=f"round-{index}", request=request, runtime=self.runtime())
                self.assertEqual(result["status"], "success", result.get("report"))
                final = read_json(run_paths(root / str(index))["idea_result_json"])
                self.assertEqual(final["reference_papers"], [])
                self.assertEqual(len(final["source_modes"]), 5)
                self.assertGreater(final["mcts_evolution"]["total_iterations"], 0)
                ops = [item.operation for item in provider.requests]
                self.assertIn(IDEA_GENERATE_OPERATION, ops)
                self.assertFalse(any("component_novelty" in op or "keynote." in op for op in ops))
                self.assertTrue(all("No auxiliary experiments" in item.system_prompt for item in provider.requests))
                if index == 2:
                    self.assertLess(ops.index(OP_QUERY), ops.index(OP_REPLAN))
                    for item in provider.requests:
                        if item.operation in {IDEA_GENERATE_OPERATION, IDEA_DIAGNOSTIC_OPERATION}:
                            grounding = item.structured_input["grounding"]
                            self.assertIn("candidate-1", "".join(grounding["memory_hints"]))
                            self.assertEqual(grounding["task_context"]["execution_contract"], request.task_context["execution_contract"])

    def test_literature_feedback_runs_fresh_query_and_invalidates_memory_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = repository_fixture(root)
            request = self.request(task_only=False, feedback="yes")
            for index in (0, 1):
                if index:
                    memory = json.loads(request.experiment_feedback)
                    memory["records"][0]["source_artifacts"].append("new-evidence.json")
                    request.experiment_feedback = json.dumps(memory)
                provider = SureProvider()
                result = run_native_workflow(run_id="same-run", request=request, runtime=self.runtime(),
                    repository=repository, source_context=repository.source_context(request), provider=provider,
                    run_dir=root / "run", novelty_runtime=novelty_runtime_fixture())
                self.assertTrue(result.succeeded, result.error)
                ops = [item.operation for item in provider.requests]
                self.assertIn(OP_QUERY, ops)
                self.assertIn(IDEA_DIAGNOSTIC_OPERATION, ops)
                self.assertLess(ops.index(OP_QUERY), ops.index(OP_REPLAN))

    def test_freeze_copies_full_declared_bundle_and_detects_mutation(self):
        from test_research_idea import write_survey_fixture
        from test_resources import write_bundle, xlab_file_artifact_id
        from freeze_survey import freeze_survey
        from research_idea_lib.survey_repository import SurveyArtifactRepository
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_survey_fixture(root / "source")
            survey = source / "artifacts/survey.json"
            write_bundle(source / "resources", survey_parent_id=xlab_file_artifact_id(survey), two_models=True)
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["resource_manifest"] = "resources/resource-manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            frozen = freeze_survey(survey, root / "frozen", root)
            repository = SurveyArtifactRepository.from_request(IdeaRequest(Path(frozen["survey_path"])), root)
            self.assertTrue(repository.validation()["passed"])
            snapshot = json.loads((root / "frozen/bundle_snapshot.json").read_text())
            self.assertGreater(len(snapshot["files"]), 8)
            graph = repository.source.graph_db_path
            self.assertTrue(graph.is_relative_to(root / "frozen"))
            graph.write_bytes(b"changed")
            broken = SurveyArtifactRepository.from_request(IdeaRequest(Path(frozen["survey_path"])), root)
            self.assertFalse(broken.source.resources.passed)


if __name__ == "__main__":
    unittest.main()
