from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PACKAGE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from research_idea_lib.algorithm.contracts import (  # noqa: E402
    OperatorAttemptRecord,
    RefinementBoundary,
    RolloutBlockerRecord,
)
from research_idea_lib.algorithm.workflow import (  # noqa: E402
    ALGORITHM_PROFILE_ID,
    CANONICAL_MODES,
    EVIDENCE_PROFILE_ID,
    OP_ANALYSIS,
    OP_BACKGROUND,
    OP_MATERIALIZATION,
    OP_QUERY,
    OP_RANKING,
    OP_REPLAN,
    RUNTIME_PROFILE_ID,
    SUCCESS_PROFILE_ID,
    BudgetUsage,
    Evidence,
    FusionWorkflowOutput,
    ModeSearchOutput,
    ProviderOutput,
    RetrievalOutput,
    WorkflowRequest,
    expected_evidence_id,
    run_workflow,
)
from research_idea_lib.providers import (  # noqa: E402
    ProviderError,
    ProviderTrace,
    structured_input_digest,
)
from research_idea_lib.research_idea_artifacts import (  # noqa: E402
    final_idea_result,
    latest_analysis_entry,
    retrieval_namespace,
    run_namespace,
)


class FakeProvider:
    def __init__(self, *, malformed_operation: str | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.malformed_operation = malformed_operation

    def execute(self, operation):
        payload = deepcopy(dict(operation.structured_input))
        self.calls.append((operation.name, payload))
        if operation.name == self.malformed_operation:
            value = {}
        elif operation.name == OP_BACKGROUND:
            value = {"background": "Grounded background"}
        elif operation.name == OP_QUERY:
            value = {"query": "scientific agent evidence"}
        elif operation.name == OP_RANKING:
            value = {"evidence_ids": [item["evidence_id"] for item in reversed(payload["evidence"])]}
        elif operation.name == OP_ANALYSIS:
            value = {
                "analysis": {"gap": "Existing agents do not retain auditable evidence."},
                "root_idea": base_idea("Provider root"),
            }
        elif operation.name == OP_REPLAN:
            revised = deepcopy(payload["mature_idea"])
            revised["title"] = "Replanned mature idea"
            revised["method"] = "Evidence-grounded search with a locally refined protected core."
            value = {
                "replan": {
                    "goal": "Refine only the mature idea's validator",
                    "root_idea": revised,
                }
            }
        elif operation.name == OP_MATERIALIZATION:
            value = {"idea_result": materialized_idea(payload["fusion"])}
        else:  # pragma: no cover - catches accidental new workflow operations
            raise AssertionError(f"unexpected operation {operation.name}")
        return ProviderOutput(
            value,
            usage=BudgetUsage(calls=1, input_tokens=2, output_tokens=3),
            metadata={
                "model": "workflow-model",
                "provider_trace": ProviderTrace(
                    provider="fake-workflow",
                    operation=operation.name,
                    input_digest=structured_input_digest(operation.structured_input),
                    output_kind="json",
                    model="workflow-model",
                    attempts=1,
                    status="success",
                ).to_dict(),
            },
        )


class FakeRetrieval:
    def __init__(self, evidence: tuple[Evidence, ...] | None = None) -> None:
        self.evidence = evidence if evidence is not None else evidence_fixture()
        self.requests = []

    def retrieve(self, request):
        self.requests.append(deepcopy(request))
        return RetrievalOutput(
            evidence=self.evidence,
            resource_ids=("survey:v1", "graph:v1"),
            usage=BudgetUsage(calls=1, input_tokens=4),
            metadata={"adapter": "deterministic"},
        )


class FakeSearch:
    def __init__(
        self,
        mode: str,
        seed: int,
        *,
        wrong_signature: bool = False,
        operator_attempts: tuple[OperatorAttemptRecord, ...] = (),
        rollout_blockers: tuple[RolloutBlockerRecord, ...] = (),
    ) -> None:
        self.mode = mode
        self.seed = seed
        self.wrong_signature = wrong_signature
        self.operator_attempts = operator_attempts
        self.rollout_blockers = rollout_blockers
        self.requests = []

    def run(self, request):
        self.requests.append(deepcopy(request))
        return ModeSearchOutput(
            mode=self.mode,
            candidate=base_idea(f"{self.mode} candidate", component=f"{self.mode}-component"),
            evidence_ids=request.evidence_ids,
            root_signature="wrong" if self.wrong_signature else request.root_signature,
            evidence_signature=request.evidence_signature,
            usage=BudgetUsage(iterations=2, evaluator_calls=3, generation_calls=1, input_tokens=5, output_tokens=7),
            trace=({"iteration": 1}, {"iteration": 2}),
            operator_attempts=self.operator_attempts,
            rollout_blockers=self.rollout_blockers,
            metadata={"seed": self.seed},
        )


class FakeSearchFactory:
    def __init__(self, *, wrong_mode: str | None = None, shared: bool = False) -> None:
        self.created: list[FakeSearch] = []
        self.wrong_mode = wrong_mode
        self.shared = shared

    def create(self, *, mode: str, seed: int):
        if self.shared and self.created:
            return self.created[0]
        search = FakeSearch(mode, seed, wrong_signature=mode == self.wrong_mode)
        self.created.append(search)
        return search


class FakeFusion:
    def __init__(self, *, incomplete: bool = False, missing_hypothesis: bool = False) -> None:
        self.incomplete = incomplete
        self.missing_hypothesis = missing_hypothesis
        self.requests = []

    def run(self, request):
        self.requests.append(deepcopy(request))
        modes = tuple(item["mode"] for item in request.mode_inputs)
        if self.incomplete:
            modes = modes[:-1]
        idea = {
            "title": "Fused idea",
            "hypothesis": "Five-mode fusion improves evidence-grounded ideation.",
            "components": [item["idea"]["components"][0] for item in request.mode_inputs],
        }
        if self.missing_hypothesis:
            idea.pop("hypothesis")
        return FusionWorkflowOutput(
            idea=idea,
            source_modes=modes,
            evidence_ids=request.evidence_ids,
            root_signature=request.root_signature,
            evidence_signature=request.evidence_signature,
            usage=BudgetUsage(calls=1, input_tokens=11, output_tokens=13),
            trace=({"phase": "fusion"},),
            metadata={"strategy": "deterministic-test"},
        )


class WorkflowTest(unittest.TestCase):
    def test_normal_workflow_executes_every_stage_mode_and_fusion(self) -> None:
        provider = FakeProvider()
        retrieval = FakeRetrieval()
        factory = FakeSearchFactory()
        fusion = FakeFusion()

        result = run_workflow(
            WorkflowRequest("run-normal", "Scientific agents", {"survey_id": "survey-1"}, seed="seed-1"),
            search_factory=factory,
            fusion_runner=fusion,
            provider=provider,
            retrieval=retrieval,
        )

        self.assertTrue(result.succeeded, result.error)
        self.assertEqual(
            [name for name, _ in provider.calls],
            [OP_BACKGROUND, OP_QUERY, OP_RANKING, OP_ANALYSIS, OP_MATERIALIZATION],
        )
        self.assertEqual(len(retrieval.requests), 1)
        self.assertEqual([search.mode for search in factory.created], list(CANONICAL_MODES))
        self.assertEqual(len({id(search) for search in factory.created}), len(CANONICAL_MODES))
        self.assertEqual(len({search.seed for search in factory.created}), len(CANONICAL_MODES))
        self.assertEqual([item["mode"] for item in fusion.requests[0].mode_inputs], list(CANONICAL_MODES))

        stages = [item["stage"] for item in run_namespace(result.artifact)["workflow_trace"]]
        self.assertEqual(
            stages,
            ["knowledge_acquisition", "advanced_analysis", *(["mcts_search"] * 5), "idea_fusion", "idea_materialization"],
        )
        roots = {search.requests[0].root_signature for search in factory.created}
        evidence = {search.requests[0].evidence_signature for search in factory.created}
        self.assertEqual(len(roots), 1)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(final_idea_result(result.artifact)["title"], "Fused idea")
        materialization_input = provider.calls[-1][1]
        self.assertIn("refinement_boundary", materialization_input)
        self.assertIsNone(materialization_input["refinement_boundary"])
        self.assertEqual(retrieval_namespace(result.artifact)["evidence_ids"], [item.evidence_id for item in reversed(evidence_fixture())])
        self.assertEqual(latest_analysis_entry(result.artifact)["analysis"]["gap"], "Existing agents do not retain auditable evidence.")

        run = run_namespace(result.artifact)
        self.assertEqual(run["algorithm_profile_id"], ALGORITHM_PROFILE_ID)
        self.assertEqual(run["runtime_profile_id"], RUNTIME_PROFILE_ID)
        self.assertEqual(run["evidence_profile_id"], EVIDENCE_PROFILE_ID)
        self.assertEqual(run["success_profile_id"], SUCCESS_PROFILE_ID)
        self.assertEqual(run["budgets"]["provider"]["calls"], 5)
        self.assertEqual(run["budgets"]["retrieval"]["calls"], 1)
        self.assertEqual(run["budgets"]["search"]["aggregate"]["iterations"], 10)
        self.assertEqual(run["budgets"]["fusion"]["calls"], 1)
        self.assertEqual(run["budgets"]["total"]["calls"], 7)
        llm_events = [event for event in run["operation_trace"] if event["event"] == "llm_call"]
        self.assertEqual([event["operation"] for event in llm_events], [name for name, _ in provider.calls])
        self.assertTrue(all(event["provider"] == "fake-workflow" for event in llm_events))
        self.assertTrue(all(event["attempts"] == 1 for event in llm_events))

    def test_provider_trace_must_correlate_with_workflow_operation(self) -> None:
        class MismatchedProvider(FakeProvider):
            def execute(self, operation):
                output = super().execute(operation)
                metadata = deepcopy(dict(output.metadata))
                trace = deepcopy(dict(metadata["provider_trace"]))
                trace["input_digest"] = "wrong-digest"
                metadata["provider_trace"] = trace
                return ProviderOutput(output.value, output.usage, metadata)

        result = run_workflow(
            WorkflowRequest("run-mismatch", "Scientific agents", {"survey_id": "survey-1"}),
            search_factory=FakeSearchFactory(),
            fusion_runner=FakeFusion(),
            provider=MismatchedProvider(),
            retrieval=FakeRetrieval(),
        )

        self.assertFalse(result.succeeded)
        self.assertIn("input digest does not match", result.error or "")

    def test_provider_error_trace_is_preserved_and_correlated(self) -> None:
        class FailingProvider:
            def execute(self, operation):
                raise ProviderError(
                    "provider failed",
                    trace=ProviderTrace(
                        provider="failing-workflow",
                        operation=operation.name,
                        input_digest=structured_input_digest(operation.structured_input),
                        output_kind="json",
                        model="failure-model",
                        attempts=2,
                        status="error",
                        error_code="upstream_failure",
                    ),
                )

        result = run_workflow(
            WorkflowRequest("run-error-trace", "Scientific agents", {"survey_id": "survey-1"}),
            search_factory=FakeSearchFactory(),
            fusion_runner=FakeFusion(),
            provider=FailingProvider(),
            retrieval=FakeRetrieval(),
        )

        self.assertFalse(result.succeeded)
        self.assertIn("provider failed", result.error or "")
        events = run_namespace(result.artifact)["operation_trace"]
        self.assertEqual(
            events,
            [
                {
                    "event": "llm_call",
                    "op_name": OP_BACKGROUND,
                    "provider": "failing-workflow",
                    "operation": OP_BACKGROUND,
                    "input_digest": events[0]["input_digest"],
                    "output_kind": "json",
                    "model": "failure-model",
                    "attempts": 2,
                    "status": "error",
                    "error_code": "upstream_failure",
                }
            ],
        )

    def test_provider_error_trace_must_correlate_with_workflow_operation(self) -> None:
        class MismatchedFailingProvider:
            def execute(self, operation):
                raise ProviderError(
                    "provider failed",
                    trace=ProviderTrace(
                        provider="failing-workflow",
                        operation="xlab.research_idea.invalid.mismatch.v1",
                        input_digest=structured_input_digest(operation.structured_input),
                        output_kind="json",
                        model="failure-model",
                        attempts=1,
                        status="error",
                        error_code="upstream_failure",
                    ),
                )

        result = run_workflow(
            WorkflowRequest("run-error-mismatch", "Scientific agents", {"survey_id": "survey-1"}),
            search_factory=FakeSearchFactory(),
            fusion_runner=FakeFusion(),
            provider=MismatchedFailingProvider(),
            retrieval=FakeRetrieval(),
        )

        self.assertFalse(result.succeeded)
        self.assertIn("operation does not match", result.error or "")
        self.assertEqual(run_namespace(result.artifact)["operation_trace"], [])

    def test_feedback_workflow_uses_analysis_replan_then_all_modes_and_fusion(self) -> None:
        provider = FakeProvider()
        factory = FakeSearchFactory()
        fusion = FakeFusion()
        evidence = evidence_fixture()
        retrieval = FakeRetrieval()
        mature = base_idea("Mature idea", component="protected-core")

        result = run_workflow(
            WorkflowRequest(
                "run-feedback",
                "Scientific agents",
                {"survey_id": "survey-1"},
                seed="seed-feedback",
                mature_idea=mature,
                refinement_scope=("Focus the refinement on validator reliability.",),
                refinement_boundary=RefinementBoundary(
                    allowed_component_ids=("protected-core",),
                    allowed_fields=("method", "components"),
                    allowed_edit_kinds=("REPLACE_COMPONENT",),
                ),
                experiment_feedback={"finding": "validator is underpowered"},
                evidence=evidence,
                resource_ids=("survey:v1", "graph:v1"),
            ),
            search_factory=factory,
            fusion_runner=fusion,
            provider=provider,
            retrieval=retrieval,
        )

        self.assertTrue(result.succeeded, result.error)
        self.assertEqual([name for name, _ in provider.calls], [OP_ANALYSIS, OP_REPLAN, OP_MATERIALIZATION])
        self.assertEqual(retrieval.requests, [])
        stages = [item["stage"] for item in run_namespace(result.artifact)["workflow_trace"]]
        self.assertEqual(stages[:2], ["advanced_analysis", "re_analysis_replan"])
        mode_count = len(CANONICAL_MODES)
        self.assertEqual(stages[2 : 2 + mode_count], ["mcts_search"] * mode_count)
        self.assertEqual(stages[-2:], ["idea_fusion", "idea_materialization"])
        for search in factory.created:
            self.assertEqual(search.requests[0].root["title"], "Replanned mature idea")
            self.assertNotEqual(search.requests[0].root, mature)
            self.assertEqual(search.requests[0].root["components"], mature["components"])
            self.assertEqual(
                search.requests[0].context["refinement_scope"],
                ["Focus the refinement on validator reliability."],
            )
            self.assertEqual(
                search.requests[0].context["refinement_boundary"],
                {
                    "allowed_component_ids": ["protected-core"],
                    "allowed_fields": ["method", "components"],
                    "allowed_edit_kinds": ["REPLACE_COMPONENT"],
                },
            )
        materialization = provider.calls[-1][1]
        self.assertEqual(
            materialization["refinement_boundary"]["allowed_component_ids"],
            ["protected-core"],
        )
        self.assertEqual(latest_analysis_entry(result.artifact)["replan"]["goal"], "Refine only the mature idea's validator")

    def test_fails_closed_on_malformed_provider_and_does_not_run_search(self) -> None:
        factory = FakeSearchFactory()
        result = run_workflow(
            WorkflowRequest("run-bad-provider", "Scientific agents", {"survey_id": "survey-1"}),
            search_factory=factory,
            fusion_runner=FakeFusion(),
            provider=FakeProvider(malformed_operation=OP_ANALYSIS),
            retrieval=FakeRetrieval(),
        )

        self.assertEqual(result.status, "incomplete")
        self.assertFalse(result.succeeded)
        self.assertNotIn("idea_result", result.artifact["persistence"])
        self.assertEqual(factory.created, [])
        self.assertIn("advanced_analysis.analysis", result.error or "")

    def test_fails_closed_on_unstable_evidence_id(self) -> None:
        invalid = Evidence("evidence:not-stable", "paper", "Evidence A", {"source": "survey"}, ("p1",))
        result = run_workflow(
            WorkflowRequest("run-bad-evidence", "Scientific agents", {"survey_id": "survey-1"}),
            search_factory=FakeSearchFactory(),
            fusion_runner=FakeFusion(),
            provider=FakeProvider(),
            retrieval=FakeRetrieval((invalid,)),
        )
        self.assertEqual(result.status, "incomplete")
        self.assertIn("unstable evidence ID", result.error or "")

    def test_fails_closed_on_shared_or_malformed_mode_search(self) -> None:
        shared_result = run_workflow(
            WorkflowRequest("run-shared", "Scientific agents", {"survey_id": "survey-1"}),
            search_factory=FakeSearchFactory(shared=True),
            fusion_runner=FakeFusion(),
            provider=FakeProvider(),
            retrieval=FakeRetrieval(),
        )
        self.assertEqual(shared_result.status, "incomplete")
        self.assertIn("independent search instance", shared_result.error or "")

        malformed = run_workflow(
            WorkflowRequest("run-bad-search", "Scientific agents", {"survey_id": "survey-1"}),
            search_factory=FakeSearchFactory(wrong_mode=CANONICAL_MODES[2]),
            fusion_runner=FakeFusion(),
            provider=FakeProvider(),
            retrieval=FakeRetrieval(),
        )
        self.assertEqual(malformed.status, "incomplete")
        self.assertIn("changed the shared root/evidence signatures", malformed.error or "")
        self.assertNotIn("idea_result", malformed.artifact["persistence"])

    def test_fails_closed_on_invalid_rollout_record_relationships(self) -> None:
        error_attempt = OperatorAttemptRecord(
            0,
            "mechanism-commit-innovation",
            "plan-a",
            "error",
        )
        applied_attempt = OperatorAttemptRecord(
            0,
            "mechanism-commit-innovation",
            "plan-a",
            "applied",
        )
        blocker = RolloutBlockerRecord(
            0,
            "mechanism-commit-innovation",
            "plan-a",
            "generation",
            "ProviderError",
            "generation failed (ProviderError)",
        )
        cases = (
            (("invalid",), (), "OperatorAttemptRecord values"),
            (
                (
                    OperatorAttemptRecord(0, "unknown-operator", "plan-a", "applied"),
                ),
                (),
                "unknown operator",
            ),
            ((error_attempt,), (), "requires one corresponding rollout blocker"),
            ((), (blocker,), "orphan rollout blocker"),
            ((applied_attempt,), (blocker,), "only correspond to error attempts"),
            ((error_attempt, error_attempt), (blocker,), "duplicate terminal attempts"),
            ((error_attempt,), (blocker, blocker), "duplicate blockers"),
        )

        for attempts, blockers, expected_error in cases:
            with self.subTest(expected_error=expected_error):
                class RecordFactory:
                    def create(self, *, mode: str, seed: int):
                        return FakeSearch(
                            mode,
                            seed,
                            operator_attempts=attempts,  # type: ignore[arg-type]
                            rollout_blockers=blockers,
                        )

                result = run_workflow(
                    WorkflowRequest(
                        f"run-invalid-record-{expected_error}",
                        "Scientific agents",
                        {"survey_id": "survey-1"},
                    ),
                    search_factory=RecordFactory(),
                    fusion_runner=FakeFusion(),
                    provider=FakeProvider(),
                    retrieval=FakeRetrieval(),
                )
                self.assertEqual(result.status, "incomplete")
                self.assertIn(expected_error, result.error or "")
                self.assertNotIn("idea_result", result.artifact["persistence"])

    def test_accepts_correlated_error_attempt_and_blocker(self) -> None:
        attempt = OperatorAttemptRecord(
            0,
            "mechanism-commit-innovation",
            "plan-a",
            "error",
        )
        blocker = RolloutBlockerRecord(
            0,
            "mechanism-commit-innovation",
            "plan-a",
            "generation",
            "ProviderError",
            "generation failed (ProviderError)",
        )

        class RecordFactory:
            def create(self, *, mode: str, seed: int):
                return FakeSearch(
                    mode,
                    seed,
                    operator_attempts=(attempt,),
                    rollout_blockers=(blocker,),
                )

        result = run_workflow(
            WorkflowRequest(
                "run-valid-records",
                "Scientific agents",
                {"survey_id": "survey-1"},
            ),
            search_factory=RecordFactory(),
            fusion_runner=FakeFusion(),
            provider=FakeProvider(),
            retrieval=FakeRetrieval(),
        )
        self.assertTrue(result.succeeded, result.error)

    def test_fails_closed_on_incomplete_fusion(self) -> None:
        provider = FakeProvider()
        result = run_workflow(
            WorkflowRequest("run-bad-fusion", "Scientific agents", {"survey_id": "survey-1"}),
            search_factory=FakeSearchFactory(),
            fusion_runner=FakeFusion(incomplete=True),
            provider=provider,
            retrieval=FakeRetrieval(),
        )
        self.assertEqual(result.status, "incomplete")
        self.assertIn("every canonical mode", result.error or "")
        self.assertNotIn(OP_MATERIALIZATION, [name for name, _ in provider.calls])
        self.assertNotIn("idea_result", result.artifact["persistence"])

        missing_hypothesis = run_workflow(
            WorkflowRequest("run-bad-hypothesis", "Scientific agents", {"survey_id": "survey-1"}),
            search_factory=FakeSearchFactory(),
            fusion_runner=FakeFusion(missing_hypothesis=True),
            provider=provider,
            retrieval=FakeRetrieval(),
        )
        self.assertEqual(missing_hypothesis.status, "incomplete")
        self.assertIn("fusion.idea.hypothesis", missing_hypothesis.error or "")
        self.assertNotIn("idea_result", missing_hypothesis.artifact["persistence"])

    def test_mature_idea_without_scope_remains_valid_and_anchored(self) -> None:
        mature = base_idea("Mature")
        factory = FakeSearchFactory()
        result = run_workflow(
            WorkflowRequest(
                "run-boundary",
                "Scientific agents",
                {"survey_id": "survey-1"},
                mature_idea=mature,
            ),
            search_factory=factory,
            fusion_runner=FakeFusion(),
            provider=FakeProvider(),
            retrieval=FakeRetrieval(),
        )
        self.assertTrue(result.succeeded, result.error)
        self.assertEqual(factory.created[0].requests[0].context["refinement_scope"], [])
        self.assertIsNone(factory.created[0].requests[0].context["refinement_boundary"])
        self.assertEqual(factory.created[0].requests[0].root["components"], mature["components"])

    def test_feedback_without_scope_builds_replanned_search_context(self) -> None:
        mature = base_idea("Mature", component="protected-core")
        factory = FakeSearchFactory()
        result = run_workflow(
            WorkflowRequest(
                "run-feedback-no-scope",
                "Scientific agents",
                {"survey_id": "survey-1"},
                mature_idea=mature,
                experiment_feedback={
                    "ablation_results": {
                        "components": {
                            "protected-core": {
                                "operation": "remove",
                                "result": "negative",
                            }
                        }
                    }
                },
                evidence=evidence_fixture(),
                resource_ids=("survey:v1", "graph:v1"),
            ),
            search_factory=factory,
            fusion_runner=FakeFusion(),
            provider=FakeProvider(),
            retrieval=FakeRetrieval(),
        )
        self.assertTrue(result.succeeded, result.error)
        for search in factory.created:
            self.assertEqual(search.requests[0].root["title"], "Replanned mature idea")
            self.assertEqual(search.requests[0].context["refinement_scope"], [])
            self.assertEqual(
                search.requests[0].context["experiment_feedback"]["ablation_results"]
                ["components"]["protected-core"]["result"],
                "negative",
            )


def base_idea(title: str, *, component: str = "core") -> dict[str, Any]:
    return {"title": title, "components": [component], "method": "Evidence-grounded search"}


def evidence_fixture() -> tuple[Evidence, ...]:
    values = [
        Evidence("", "paper", "Evidence A", {"source": "survey", "rank": 1}, ("p1",), "survey-1"),
        Evidence("", "graph", "Evidence B", {"source": "graph", "rank": 2}, ("p2",), "graph-1"),
    ]
    return tuple(replace(value, evidence_id=expected_evidence_id(value)) for value in values)


def materialized_idea(fusion: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "success",
        "title": fusion["title"],
        "components": deepcopy(fusion["components"]),
        "source_modes": list(CANONICAL_MODES),
        "abstract": "A complete provider-materialized research idea.",
        "core_contribution": "Auditable scientific-agent ideation.",
        "method": "Five independent searches followed by fusion.",
        "risks": ["Evidence drift"],
    }


if __name__ == "__main__":
    unittest.main()
