from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PACKAGE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from research_idea_lib.algorithm.component_novelty import (  # noqa: E402
    COMPONENT_NOVELTY_OPERATION,
    ComponentHit,
    ComponentRetrievalOutput,
    CoreNode,
    DeclaredNoveltyResource,
)
from research_idea_lib.algorithm.evaluation import METRICS  # noqa: E402
from research_idea_lib.algorithm.fusion import REFEREE_METRICS  # noqa: E402
from research_idea_lib.algorithm.keynote_pipeline import (  # noqa: E402
    KEYNOTE_COMPRESSION_OPERATION,
    KEYNOTE_ROLLUP_OPERATION,
    KEYNOTE_SCORE_OPERATION,
)
from research_idea_lib.algorithm.operator_grounding import (  # noqa: E402
    OperatorComponentHit,
    OperatorComponentRetrieval,
)
from research_idea_lib.algorithm.provider_adapter import (  # noqa: E402
    FUSION_GENERATE_OPERATION,
    FUSION_REFEREE_OPERATION,
    FUSION_REPAIR_OPERATION,
    IDEA_DIAGNOSTIC_OPERATION,
    IDEA_EVALUATE_OPERATION,
    IDEA_GENERATE_OPERATION,
    MECHANISM_COMMIT_QUERY_OPERATION,
    THEORY_TRANSFER_QUERY_OPERATION,
)
from research_idea_lib.algorithm.runtime_adapters import _search_config, run_native_workflow  # noqa: E402
from research_idea_lib.algorithm.workflow import (  # noqa: E402
    CANONICAL_MODES,
    OP_ANALYSIS,
    OP_BACKGROUND,
    OP_MATERIALIZATION,
    OP_QUERY,
    OP_RANKING,
    OP_REPLAN,
    RUNTIME_PROFILE_ID,
    stable_signature,
)
from research_idea_lib.common import atomic_write_json, read_json, run_paths  # noqa: E402
from research_idea_lib.resources.manifest import sha256_file  # noqa: E402
from research_idea_lib.config import (  # noqa: E402
    DEFAULT_BRANCHING_FACTOR,
    DEFAULT_EXPLORATION_CONSTANT,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_ITERATIONS,
    IDEA_TASTE_MODES,
    RUNTIME_CONFIG_SEMANTIC_SIGNATURE,
    RUNTIME_CONFIG_SEMANTIC_VERSION,
    RuntimeConfig,
    runtime_config_from_json,
)
from research_idea_lib.inputs import IdeaRequest  # noqa: E402
from research_idea_lib.pipeline import run_research_idea  # noqa: E402
from research_idea_lib.providers import (  # noqa: E402
    ProviderError,
    ProviderResult,
    ProviderTrace,
    ProviderUsage,
)
from research_idea_lib.resources.component_novelty import ComponentNoveltyRuntime  # noqa: E402
from research_idea_lib.resources.evidence import ResourceEvidenceResult  # noqa: E402
from research_idea_lib.resources.manifest import (  # noqa: E402
    FileDigest,
    LoadedResourceBundle,
    ResourceDescriptor,
    ResourceManifest,
)
from research_idea_lib.resources.resolver import ResourceResolution  # noqa: E402
from research_idea_lib.survey_repository import (  # noqa: E402
    SurveyArtifactRepository,
    SurveySource,
)


class DeterministicNativeProvider:
    """A hermetic generic Provider that exercises the production native facade."""

    def __init__(self) -> None:
        self.requests = []

    def complete(self, request):
        self.requests.append(deepcopy(request))
        payload = self._payload(request)
        return ProviderResult(
            text=json.dumps(payload),
            json_value=payload,
            usage=ProviderUsage(input_tokens=2, output_tokens=3, total_tokens=5),
            trace=ProviderTrace(
                provider="deterministic-native",
                operation=request.operation,
                input_digest=request.input_digest,
                output_kind="json",
                model=request.model,
                attempts=1,
                status="success",
            ),
        )

    def _payload(self, request):
        data = request.structured_input
        if request.operation == KEYNOTE_SCORE_OPERATION:
            return {"score": 87}
        if request.operation == KEYNOTE_COMPRESSION_OPERATION:
            return {
                "summary": "Stable evidence identities support auditable ideation.",
                "insight": "Carry citation provenance through every search operation.",
            }
        if request.operation == KEYNOTE_ROLLUP_OPERATION:
            return {"summary": "Remaining cited work supports evidence-grounded search."}
        if request.operation == COMPONENT_NOVELTY_OPERATION:
            return {
                "retrieval_similarity": 3,
                "perceived_novelty": 4,
                "rubric_score": 4,
                "rationale": "Candidate differs from the retrieved component.",
                "provenance": {"fixture": "component-novelty"},
            }
        if request.operation == OP_BACKGROUND:
            return {"background": "Survey-grounded scientific-agent ideation.", "key_questions": ["How can evidence remain auditable?"], "canonical_methods": ["Tree search"]}
        if request.operation == OP_QUERY:
            return {"query": "auditable evidence scientific agent search"}
        if request.operation == OP_RANKING:
            return {"evidence_ids": [item["evidence_id"] for item in data["evidence"]]}
        if request.operation == OP_ANALYSIS:
            return {
                "analysis": {"gap": "Existing agents lose source-grounded evidence during idea search."},
                "root_idea": idea("Evidence-rooted search", "root-core"),
            }
        if request.operation == OP_REPLAN:
            return {"replan": {"root_idea": deepcopy(data["mature_idea"])}}
        if request.operation in {IDEA_DIAGNOSTIC_OPERATION, IDEA_EVALUATE_OPERATION}:
            return {
                "metrics": {name: 4 for name in METRICS},
                "confidence": 0.9,
                "detected_defects": ["unclear_mechanism"],
                "feedback": "Explore one bounded evidence-grounded edit.",
            }
        if request.operation in {
            THEORY_TRANSFER_QUERY_OPERATION,
            MECHANISM_COMMIT_QUERY_OPERATION,
        }:
            if request.operation == THEORY_TRANSFER_QUERY_OPERATION:
                return {
                    "query": "cross-domain auditable evidence mechanism",
                    "needed_content": "A transferable evidence mechanism.",
                    "expected_role": "Support evidence-preserving search.",
                }
            return {
                "query": "commit auditable evidence mechanism",
                "mechanism_gap": "The evidence mechanism is underspecified.",
                "expected_role": "Make evidence preservation explicit.",
            }
        if request.operation == IDEA_GENERATE_OPERATION:
            parent = deepcopy(data["parent"])
            mode = data["idea_taste_mode"]
            parent["title"] = f"{mode} evidence candidate"
            parent["abstract"] = "A concrete evidence-grounded scientific-agent candidate."
            parent["core_contribution"] = "Stable evidence identities throughout tree search."
            parent["method"] = "Carry survey evidence through typed Monte Carlo tree search."
            parent["risks"] = "Evidence coverage can remain incomplete."
            current = parent["components"][0]["name"]
            replacement = f"{mode}-core"
            parent["components"] = [
                {"name": replacement, "description": "Evidence-grounded mode contribution."}
            ]
            return {
                "state": parent,
                "component_mapping": {
                    "weak_internal_component": current,
                    "refined_internal_component": replacement,
                },
                "component_role_explanations": {
                    replacement: "Evidence-grounded mode contribution."
                },
            }
        if request.operation == FUSION_GENERATE_OPERATION:
            mode_inputs = data["mode_inputs"]
            components = [deepcopy(item["idea"]["components"][0]) for item in mode_inputs]
            selected = [
                {"component": components[0]["name"], "source_mode": mode_inputs[0]["mode"], "evidence": mode_inputs[0]["evidence"]}
            ]
            components = components[:1]
            return {
                "idea": {
                    "title": "Fused evidence-grounded idea",
                    "abstract": "Five native search modes fused with explicit evidence provenance.",
                    "core_contribution": "Auditable five-mode scientific ideation.",
                    "method": "Fuse typed MCTS candidates while retaining stable evidence IDs.",
                    "research_question": "Can stable survey evidence improve scientific-agent idea traceability?",
                    "hypothesis": "Five evidence-grounded MCTS modes improve traceability without losing novelty.",
                    "risks": "Provider scoring may be noisy.",
                    "components": components,
                    "tags": ["scientific agents"],
                    "root_domains": ["machine learning"],
                },
                "selected_components": selected,
                "rejected_components": [],
                "conflict_resolutions": [],
            }
        if request.operation == FUSION_REFEREE_OPERATION:
            return {"score": 4.2, "metrics": {name: 4 for name in REFEREE_METRICS}}
        if request.operation == FUSION_REPAIR_OPERATION:
            return {"stop": True}
        if request.operation == OP_MATERIALIZATION:
            fusion = data["fusion"]
            return {
                "idea_result": {
                    **deepcopy(fusion),
                    "research_question": "Can stable survey evidence improve scientific-agent idea traceability?",
                    "hypothesis": "Five evidence-grounded MCTS modes improve traceability without losing novelty.",
                    "experiment_plan": ["Compare five-mode native search with a single-mode ablation."],
                    "data_requirements": ["A completed literature survey with citation traces."],
                    "baselines": ["Single-mode evidence-grounded ideation."],
                    "metrics": ["Evidence traceability and idea novelty."],
                    "risks": [fusion["risks"]],
                    "introduction": "Scientific agents need auditable idea-generation evidence.",
                    "algorithm": [{"name": "Native five-mode MCTS fusion", "steps": ["retrieve", "search", "fuse"]}],
                    "reference_papers": ["Evidence Paper"],
                    "source_modes": list(CANONICAL_MODES),
                }
            }
        raise AssertionError(f"unexpected operation: {request.operation}")


class FailingKeynoteProvider(DeterministicNativeProvider):
    def __init__(self, *, malformed: bool) -> None:
        super().__init__()
        self.malformed = malformed

    def complete(self, request):
        if request.operation != KEYNOTE_SCORE_OPERATION:
            return super().complete(request)
        self.requests.append(deepcopy(request))
        trace = ProviderTrace(
            provider="failing-native",
            operation=request.operation,
            input_digest=request.input_digest,
            output_kind="json",
            model=request.model,
            attempts=1,
            status="success" if self.malformed else "error",
            error_code=None if self.malformed else "provider_failure",
        )
        if not self.malformed:
            raise ProviderError("keynote provider failed", trace=trace)
        payload = {"score": True}
        return ProviderResult(
            text=json.dumps(payload),
            json_value=payload,
            usage=ProviderUsage(input_tokens=7, output_tokens=2, total_tokens=9),
            trace=trace,
        )


class DeterministicComponentRetriever:
    def __init__(self, model, component_index) -> None:
        self.embedding_model = model
        self.component_index = component_index

    def retrieve_operator_components(self, query, *, limit):
        del limit
        return OperatorComponentRetrieval(
            hits=(
                OperatorComponentHit(
                    core_node_id="core-1",
                    component_id="component:fixture",
                    component_name="Evidence gate",
                    component_description="An auditable evidence-preserving gate.",
                    score=0.7,
                    domain="Information Science",
                    paper_ids=("p1",),
                    resource_identity=self.component_index.resource_id,
                    trace_identity=f"trace:{query}",
                ),
            ),
            provenance_json=json.dumps({"fixture": "operator-component-retrieval"}),
        )

    def retrieve(self, request):
        node = CoreNode(
            evidence_id="evidence:component-core",
            node_id="core-1",
            label="Core evidence",
            paper_title="Evidence Paper",
            summary="A related evidence-grounded search component.",
            insight="Stable identities improve traceability.",
            provenance_json=json.dumps({"fixture": "component-core"}),
        )
        return ComponentRetrievalOutput(
            request_identity=request.request_identity,
            candidate_id=request.candidate_id,
            candidate_textual_identity=request.candidate_textual_identity,
            idea_taste_mode=request.idea_taste_mode,
            query_id=request.query.query_id,
            hits=(
                ComponentHit(
                    record_id=f"record:{request.query.query_id}",
                    node=node,
                    matched_component=request.query.component_name,
                    similarity=0.5,
                ),
            ),
            embedding_model=self.embedding_model,
            component_index=self.component_index,
            native_embedding_inference=True,
            native_faiss_search=True,
            provenance_json=json.dumps({"fixture": "component-retrieval"}),
        )


def novelty_runtime_fixture() -> ComponentNoveltyRuntime:
    model = DeclaredNoveltyResource(
        "model:component:v1",
        "a" * 64,
        "sentence-transformers.v1",
        "xlab-resource://bundle/model%3Acomponent%3Av1",
    )
    index = DeclaredNoveltyResource(
        "components:v1",
        "b" * 64,
        "faiss-flat-ip.v1",
        "xlab-resource://bundle/components%3Av1",
    )
    return ComponentNoveltyRuntime(
        DeterministicComponentRetriever(model, index),
        model,
        index,
    )


class NativePipelineIntegrationTest(unittest.TestCase):
    def test_runtime_config_serializes_effective_production_search_profile(self) -> None:
        runtime = runtime_config_from_json({})

        self.assertEqual(
            (
                runtime.max_iterations,
                runtime.max_depth,
                runtime.branching_factor,
                runtime.exploration_constant,
            ),
            (
                DEFAULT_MAX_ITERATIONS,
                DEFAULT_MAX_DEPTH,
                DEFAULT_BRANCHING_FACTOR,
                DEFAULT_EXPLORATION_CONSTANT,
            ),
        )
        self.assertEqual(
            (
                runtime.max_children,
                runtime.max_evaluator_calls,
                runtime.max_tokens,
                runtime.max_cost,
            ),
            (None, None, None, None),
        )
        serialized = runtime.to_json()
        self.assertEqual(serialized["runtime_config_semantic_version"], RUNTIME_CONFIG_SEMANTIC_VERSION)
        self.assertEqual(serialized["runtime_config_semantic_signature"], RUNTIME_CONFIG_SEMANTIC_SIGNATURE)
        self.assertFalse(serialized.get("root_diagnostic_counts_as_iteration", False))

    def test_runtime_config_requires_current_semantic_identity(self) -> None:
        for field in ("runtime_config_semantic_version", "runtime_config_semantic_signature"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                runtime_config_from_json({field: "stale"})

    def test_runtime_budgets_are_wired_into_native_search(self) -> None:
        runtime = runtime_config_from_json(
            {
                "max_iterations": 2,
                "max_depth": 1,
                "branching_factor": 1,
                "exploration_constant": 1.7,
                "max_children": 3,
                "max_evaluator_calls": 4,
                "max_tokens": 5,
                "max_cost": 0.25,
            }
        )

        search = _search_config(runtime, seed=7, evidence_digest="evidence")

        self.assertEqual(
            (
                search.max_iterations,
                search.max_depth,
                search.branching_factor,
                search.exploration_constant,
                search.max_children,
                search.max_evaluator_calls,
                search.max_tokens,
                search.max_cost,
            ),
            (2, 1, 1, 1.7, 3, 4, 5, 0.25),
        )
        self.assertEqual(
            search.runtime_profile,
            stable_signature(
                {
                    "runtime_profile": RUNTIME_PROFILE_ID,
                    "runtime_config_semantic_version": RUNTIME_CONFIG_SEMANTIC_VERSION,
                    "runtime_config_semantic_signature": RUNTIME_CONFIG_SEMANTIC_SIGNATURE,
                }
            ),
        )
        self.assertEqual(search.evaluator_profile, runtime.evaluation_model)
        self.assertEqual(search.evidence_digest, "evidence")

    def test_runtime_config_requires_exact_canonical_taste_modes(self) -> None:
        for modes in (
            IDEA_TASTE_MODES[:-1],
            list(reversed(IDEA_TASTE_MODES)),
            [*IDEA_TASTE_MODES, "extra"],
            [],
        ):
            with self.subTest(modes=modes), self.assertRaisesRegex(ValueError, "canonical modes"):
                runtime_config_from_json({"idea_taste_modes": modes})

    def test_native_facade_resumes_completed_taste_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / ".xlab" / "runs" / "native-resume"
            request = IdeaRequest(root / "survey", topic="Scientific agents")
            runtime = RuntimeConfig(
                agent_model="agent-model",
                generation_model="generation-model",
                evaluation_model="evaluation-model",
                fusion_model="fusion-model",
                openai_base_url="https://provider.invalid/v1",
                chat_completions_url="https://provider.invalid/v1/chat/completions",
                max_iterations=1,
                max_depth=1,
                branching_factor=1,
                request_timeout_seconds=10,
                max_retries=0,
            )
            repository = repository_fixture(root)
            source_context = repository.source_context(request)
            first_provider = DeterministicNativeProvider()
            first = run_native_workflow(
                run_id="native-resume",
                request=request,
                runtime=runtime,
                repository=repository,
                source_context=source_context,
                provider=first_provider,
                run_dir=run_dir,
                novelty_runtime=novelty_runtime_fixture(),
            )
            self.assertTrue(first.succeeded, first.error)
            self.assertEqual(
                sum(request.operation == IDEA_DIAGNOSTIC_OPERATION for request in first_provider.requests),
                len(CANONICAL_MODES),
            )

            second_provider = DeterministicNativeProvider()
            second = run_native_workflow(
                run_id="native-resume",
                request=request,
                runtime=runtime,
                repository=repository,
                source_context=source_context,
                provider=second_provider,
                run_dir=run_dir,
                novelty_runtime=novelty_runtime_fixture(),
            )
            self.assertTrue(second.succeeded, second.error)
            second_operations = [request.operation for request in second_provider.requests]
            self.assertNotIn(IDEA_DIAGNOSTIC_OPERATION, second_operations)
            self.assertNotIn(IDEA_EVALUATE_OPERATION, second_operations)
            self.assertNotIn(IDEA_GENERATE_OPERATION, second_operations)
            candidates = second.artifact["ideation"]["mode_candidates"]
            self.assertEqual(len(candidates), len(CANONICAL_MODES))
            self.assertTrue(all(candidate["metadata"].get("resumed") for candidate in candidates))
            first_trace = first.artifact["run"]["operation_trace"]
            second_trace = second.artifact["run"]["operation_trace"]
            first_search_provider = [
                event for event in first_trace if event.get("provider_operation", "").startswith("xlab.research_idea.idea.")
            ]
            resumed_search_provider = [
                event for event in second_trace if event.get("provider_operation", "").startswith("xlab.research_idea.idea.")
            ]
            self.assertEqual(resumed_search_provider, first_search_provider)
            resume_events = [event for event in second_trace if event.get("event") == "search_resume"]
            self.assertEqual(len(resume_events), len(CANONICAL_MODES))
            self.assertTrue(all(event["status"] == "resumed" for event in resume_events))
            for mode in CANONICAL_MODES:
                self.assertTrue(run_paths(run_dir)[f"workflow_search_{mode}"].exists())

            keynote_path = run_dir / "state" / "research_idea" / "keynote.json"
            self.assertTrue(keynote_path.exists())
            first_keynote_operations = [
                item.operation
                for item in first_provider.requests
                if item.operation.startswith("xlab.research_idea.keynote.")
            ]
            self.assertEqual(
                first_keynote_operations,
                [KEYNOTE_SCORE_OPERATION, KEYNOTE_COMPRESSION_OPERATION],
            )
            self.assertFalse(
                any(
                    item.operation.startswith("xlab.research_idea.keynote.")
                    for item in second_provider.requests
                )
            )
            keynote_resume = [
                event
                for event in second_trace
                if event.get("event") == "keynote_resume"
            ]
            self.assertEqual(len(keynote_resume), 1)
            self.assertEqual(
                second.artifact["retrieval"]["metadata"]["keynote_pipeline"][
                    "provider_traces"
                ],
                first.artifact["retrieval"]["metadata"]["keynote_pipeline"][
                    "provider_traces"
                ],
            )
            checkpoint = read_json(run_paths(run_dir)["checkpoint"])
            self.assertIn("keynote_grounding", checkpoint["completed_stages"])
            for mode in CANONICAL_MODES:
                self.assertIn(f"search.{mode}", checkpoint["completed_stages"])

    def test_keynote_provider_failures_are_incomplete_and_traced(self) -> None:
        for malformed in (False, True):
            with self.subTest(malformed=malformed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                provider = FailingKeynoteProvider(malformed=malformed)
                request = IdeaRequest(root / "survey", topic="Scientific agents")
                repository = repository_fixture(root)
                result = run_native_workflow(
                    run_id=f"keynote-failure-{malformed}",
                    request=request,
                    runtime=RuntimeConfig(
                        agent_model="agent-model",
                        generation_model="generation-model",
                        evaluation_model="evaluation-model",
                        fusion_model="fusion-model",
                        openai_base_url="https://provider.invalid/v1",
                        chat_completions_url="https://provider.invalid/v1/chat/completions",
                        max_iterations=1,
                        max_depth=1,
                        branching_factor=1,
                        request_timeout_seconds=10,
                        max_retries=0,
                    ),
                    repository=repository,
                    source_context=repository.source_context(request),
                    provider=provider,
                    run_dir=root / ".xlab" / "runs" / "keynote-failure",
                    novelty_runtime=novelty_runtime_fixture(),
                )

                self.assertFalse(result.succeeded)
                self.assertEqual(result.artifact["run"]["status"], "incomplete")
                self.assertFalse(
                    any(item.operation == OP_ANALYSIS for item in provider.requests)
                )
                traces = result.artifact["run"]["operation_trace"]
                keynote_traces = [
                    item
                    for item in traces
                    if item.get("provider_operation") == KEYNOTE_SCORE_OPERATION
                ]
                self.assertEqual(len(keynote_traces), 1)
                self.assertEqual(
                    keynote_traces[0]["status"], "success" if malformed else "error"
                )
                self.assertIn(
                    {"event": "keynote", "op_name": "xlab.research_idea.keynote.ground.v1", "status": "error"},
                    traces,
                )
                expected = {"calls": 1, "input_tokens": 7, "output_tokens": 2}
                budget = result.artifact["run"]["budgets"]["keynote"]
                if malformed:
                    for field, value in expected.items():
                        self.assertEqual(budget[field], value)
                else:
                    self.assertEqual(budget["calls"], 1)
                    self.assertEqual(budget["input_tokens"], 0)
                    self.assertEqual(budget["output_tokens"], 0)

    def test_feedback_path_curates_keynotes_before_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = IdeaRequest(
                root / "survey",
                topic="Scientific agents",
                mature_idea=json.dumps(idea("Mature evidence search", "root-core")),
                experiment_feedback=json.dumps({"finding": "The evidence gate is weak."}),
            )
            runtime = RuntimeConfig(
                agent_model="agent-model",
                generation_model="generation-model",
                evaluation_model="evaluation-model",
                fusion_model="fusion-model",
                openai_base_url="https://provider.invalid/v1",
                chat_completions_url="https://provider.invalid/v1/chat/completions",
                max_iterations=1,
                max_depth=1,
                branching_factor=1,
                request_timeout_seconds=10,
                max_retries=0,
            )
            repository = repository_fixture(root)
            provider = DeterministicNativeProvider()
            result = run_native_workflow(
                run_id="feedback-keynotes",
                request=request,
                runtime=runtime,
                repository=repository,
                source_context=repository.source_context(request),
                provider=provider,
                novelty_runtime=novelty_runtime_fixture(),
            )

            self.assertTrue(result.succeeded, result.error)
            keynote = result.artifact["retrieval"]["metadata"]["keynote_pipeline"]
            self.assertEqual(keynote["selected_paper_ids"], ["p1"])
            analysis = next(item for item in provider.requests if item.operation == OP_ANALYSIS)
            self.assertEqual(
                [item["paper_ids"] for item in analysis.structured_input["capsules"]],
                [["p1"]],
            )
            self.assertEqual(
                analysis.structured_input["curated_references"],
                analysis.structured_input["capsules"],
            )
            operations = [item.operation for item in provider.requests]
            self.assertLess(operations.index(KEYNOTE_SCORE_OPERATION), operations.index(OP_ANALYSIS))
            trace_operations = [
                event.get("provider_operation")
                for event in result.artifact["run"]["operation_trace"]
            ]
            self.assertIn(KEYNOTE_SCORE_OPERATION, trace_operations)

    def test_run_research_idea_enters_native_facade_and_writes_five_namespaces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / ".xlab" / "runs" / "native-cli"
            paths = run_paths(run_dir)
            for directory in (paths["artifacts"], paths["state"], paths["research_idea"], paths["logs"], paths["memory"]):
                directory.mkdir(parents=True, exist_ok=True)
            provider = DeterministicNativeProvider()
            request = IdeaRequest(root / "survey", topic="Scientific agents")
            runtime = RuntimeConfig(
                agent_model="agent-model",
                generation_model="generation-model",
                evaluation_model="evaluation-model",
                fusion_model="fusion-model",
                openai_base_url="https://provider.invalid/v1",
                chat_completions_url="https://provider.invalid/v1/chat/completions",
                max_iterations=1,
                max_depth=1,
                branching_factor=1,
                request_timeout_seconds=10,
                max_retries=0,
            )
            repository = repository_fixture(root)
            source_context = repository.source_context(request)
            request_json = request.to_json(root)
            runtime_json = runtime.to_json()
            atomic_write_json(paths["request"], request_json)
            atomic_write_json(paths["runtime_config"], runtime_json)

            with patch(
                "research_idea_lib.pipeline.OpenAICompatibleProvider",
                return_value=provider,
            ) as provider_constructor, patch(
                "research_idea_lib.algorithm.runtime_adapters.build_component_novelty_runtime",
                return_value=novelty_runtime_fixture(),
            ), patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "live-secret-key"},
                clear=False,
            ):
                run_research_idea(
                    root,
                    run_dir,
                    "native-cli",
                    request,
                    runtime,
                    request_json,
                    runtime_json,
                    repository,
                    source_context,
                )

            provider_constructor.assert_called_once()
            self.assertEqual(provider_constructor.call_args.kwargs["api_key"], "live-secret-key")
            self.assertEqual(provider_constructor.call_args.kwargs["endpoint"], runtime.chat_completions_url)
            for key in (
                "workflow_run",
                "workflow_retrieval",
                "workflow_analysis",
                "workflow_ideation",
                "workflow_persistence",
                "workflow_artifact",
                "idea_result_json",
                "research_idea_json",
                "idea_trace_json",
            ):
                self.assertTrue(paths[key].exists(), key)

            artifact = read_json(paths["workflow_artifact"])
            self.assertEqual(set(artifact), {"run", "retrieval", "analysis", "ideation", "persistence"})
            self.assertEqual(
                [item["stage"] for item in artifact["run"]["workflow_trace"]],
                ["knowledge_acquisition", "advanced_analysis", "idea_generation"],
            )
            self.assertEqual(len(artifact["ideation"]["mode_candidates"]), 5)
            keynote = artifact["retrieval"]["metadata"]["keynote_pipeline"]
            self.assertEqual(keynote["selected_paper_ids"], ["p1"])
            self.assertEqual(keynote["keynote_paper_ids"], ["p1"])
            self.assertEqual(len(keynote["curated_references"]), 1)
            self.assertEqual(
                [item["paper_id"] for item in artifact["retrieval"]["ranked_keynotes"]],
                ["p1"],
            )
            ranking = next(request for request in provider.requests if request.operation == OP_RANKING)
            analysis = next(request for request in provider.requests if request.operation == OP_ANALYSIS)
            self.assertNotIn("capsules", ranking.structured_input)
            self.assertEqual(
                [item["paper_ids"] for item in analysis.structured_input["capsules"]],
                [["p1"]],
            )
            trace_operations = [event.get("provider_operation") for event in artifact["run"]["operation_trace"]]
            self.assertIn(KEYNOTE_SCORE_OPERATION, trace_operations)
            self.assertIn(KEYNOTE_COMPRESSION_OPERATION, trace_operations)
            idea_result = read_json(paths["idea_result_json"])
            self.assertEqual(
                idea_result["hypothesis"],
                "Five evidence-grounded MCTS modes improve traceability without losing novelty.",
            )
            self.assertEqual(idea_result["source_modes"], list(CANONICAL_MODES))
            self.assertEqual(idea_result["algorithm_provenance"]["runtime"], "package-native")
            self.assertTrue(idea_result["algorithm_provenance"]["native_reconstruction"])
            self.assertEqual(set(idea_result["mcts_evolution"]["stop_reasons"]), set(CANONICAL_MODES))
            public_idea = read_json(paths["research_idea_json"])
            self.assertEqual(public_idea["source_context"]["keynote_paper_ids"], ["p1"])
            self.assertEqual(len(public_idea["source_context"]["curated_references"]), 1)
            serialized = json.dumps({path.name: path.read_text() for path in run_dir.rglob("*") if path.is_file()})
            self.assertNotIn("live-secret-key", serialized)
            self.assertNotIn(str(repository.source.outcome_model_path), serialized)

            operations = [request.operation for request in provider.requests]
            self.assertIn(OP_ANALYSIS, operations)
            self.assertIn(FUSION_GENERATE_OPERATION, operations)
            self.assertIn(OP_MATERIALIZATION, operations)
            self.assertEqual(operations.count(IDEA_DIAGNOSTIC_OPERATION), len(CANONICAL_MODES))


def idea(title: str, component: str) -> dict:
    return {
        "title": title,
        "abstract": "A complete root idea grounded in survey evidence.",
        "core_contribution": "Evidence remains stable throughout idea search.",
        "method": "Run typed MCTS over evidence-grounded ideas.",
        "risks": "Provider judgments may vary.",
        "components": [{"name": component, "description": "Carries explicit survey evidence."}],
        "tags": ["scientific agents"],
        "root_domains": ["machine learning"],
    }


def repository_fixture(root: Path) -> SurveyArtifactRepository:
    keynote_root = root / "survey" / "resources" / "keynotes"
    keynote_root.mkdir(parents=True, exist_ok=True)
    keynote_file = keynote_root / "keynotes.json"
    keynote_file.write_text(json.dumps({"p1": "Evidence-grounded ideation keynote."}), encoding="utf-8")
    keynote_descriptor = ResourceDescriptor(
        resource_id="keynotes:v1",
        kind="keynotes",
        schema_version="keynotes.v1",
        algorithm_version="keynotes.v1",
        path="resources/keynotes",
        digest="0" * 64,
        files=(
            FileDigest(
                path="keynotes.json",
                sha256=sha256_file(keynote_file),
                size=keynote_file.stat().st_size,
            ),
        ),
    )
    placeholder = ResourceDescriptor(
        resource_id="survey:v1",
        kind="survey",
        schema_version="survey.v1",
        algorithm_version="survey.v1",
        path="survey",
        digest="1" * 64,
        files=(),
    )
    manifest = ResourceManifest(
        schema_version="xlab.research_idea.resources.v1",
        bundle_id="bundle:v1",
        survey=placeholder,
        graph=ResourceDescriptor(**{**placeholder.__dict__, "resource_id": "graph:v1", "kind": "graph"}),
        component_index=ResourceDescriptor(**{**placeholder.__dict__, "resource_id": "components:v1", "kind": "component_index"}),
        keynotes=keynote_descriptor,
        models=(),
    )
    resources = ResourceResolution(bundle=LoadedResourceBundle(root=root / "survey", manifest=manifest))
    resources.portable_resources = lambda: {  # type: ignore[method-assign]
        "survey": {"resource_id": "survey:v1", "logical_uri": "xlab-resource://bundle/survey%3Av1"},
        "graph": {"resource_id": "graph:v1", "logical_uri": "xlab-resource://bundle/graph%3Av1"},
        "component_index": {"resource_id": "components:v1", "logical_uri": "xlab-resource://bundle/components%3Av1"},
        "keynotes": {"resource_id": "keynotes:v1", "logical_uri": "xlab-resource://bundle/keynotes%3Av1"},
        "outcome_model": {"resource_id": "model:outcome:v1", "logical_uri": "xlab-cache://models/outcome"},
        "component_model": {"resource_id": "model:component:v1", "logical_uri": "xlab-cache://models/component"},
    }
    source = SurveySource(
        survey_path=root / "survey",
        survey_json_path=root / "survey" / "artifacts" / "survey.json",
        survey_md_path=root / "survey" / "artifacts" / "survey.md",
        citations_path=root / "survey" / "artifacts" / "citations.json",
        report_path=root / "survey" / "artifacts" / "survey_report.json",
        manifest_path=root / "survey" / "manifest.json",
        survey={"topic": "Scientific agents"},
        citations={
            "references": [{"paper_id": "p1", "title": "Evidence Paper"}],
            "traces": [{"paper_id": "p1"}],
        },
        resources=resources,
        keynote_cache_path=keynote_root,
        outcome_model_path=root / "private-model-cache" / "outcome-model",
        component_model_path=root / "private-model-cache" / "component-model",
    )
    repository = SurveyArtifactRepository(
        source=source,
        evidence_items=[
            {
                "id": "paragraph-1",
                "kind": "survey_markdown_paragraph",
                "title": "Evidence loss",
                "text": "Scientific agents lose auditable survey evidence during idea search.",
                "paper_ids": ["p1"],
                "source": "survey.md",
                "resource_score": 0.9,
                "resource_provenance": {
                    "role": "outcome_model",
                    "paragraph_id": "paragraph-1",
                    "section_path": ["Evidence loss"],
                },
            }
        ],
        references=[{"paper_id": "p1", "title": "Evidence Paper"}],
    )
    repository.retrieve_resource_evidence = lambda query, limit: ResourceEvidenceResult(  # type: ignore[method-assign]
        evidence_items=tuple(deepcopy(repository.evidence_items[:limit])),
        usage={},
    )
    return repository


if __name__ == "__main__":
    unittest.main()
