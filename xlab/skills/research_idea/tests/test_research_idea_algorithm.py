from __future__ import annotations

import hashlib
import random
import sys
import unittest
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PACKAGE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from research_idea_lib.algorithm.contracts import (  # noqa: E402
    ALGORITHM_ID,
    EvaluationResponse,
    GenerationRequest,
    GenerationResponse,
    IdeaComponent,
    IdeaProviderContext,
    IdeaState,
    OperatorGrounding,
    OperatorAttemptRecord,
    ProviderUsage,
    RefinementBoundary,
)
from research_idea_lib.algorithm.evaluation import (  # noqa: E402
    Evaluation,
    EvaluationPostprocessing,
    METRICS,
)
from research_idea_lib.algorithm.memory import MemoryState, SymbolicMemoryRecord  # noqa: E402
from research_idea_lib.algorithm.operator_grounding import (  # noqa: E402
    OperatorGroundingResult,
)
from research_idea_lib.algorithm.operators import (  # noqa: E402
    AtomicEdit,
    EditKind,
    EditPlan,
    MAX_EDITS,
    approved_plans,
)
from research_idea_lib.algorithm.search import (  # noqa: E402
    MCTSEngine,
    PathStep,
    SearchConfig,
    TreeEdge,
    TreeNode,
)
from research_idea_lib.algorithm.spec import IMPLEMENTATION_METADATA  # noqa: E402
from research_idea_lib.algorithm.tastes import IDEA_TASTE_MODES, get_taste  # noqa: E402


def idea(title: str = "Seed") -> IdeaState:
    return IdeaState(
        title=title,
        abstract="A focused abstract.",
        core_contribution="A testable mechanism",
        method="Measure a controlled intervention",
        risks="Distribution shift",
        components=(IdeaComponent("core", "The core mechanism"),),
        tags=("agents",),
        root_domains=("computer science",),
    )


def metrics(**overrides: float) -> dict[str, float]:
    values = {name: 3.0 for name in METRICS}
    values.update(overrides)
    return values


class StubProvider:
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []
        self.diagnostic_calls = 0
        self.evaluator_calls = 0

    def evaluate(
        self,
        state: IdeaState,
        *,
        idea_taste_mode: str,
        prompt_mode: str,
        diagnostic: bool,
        context=None,
    ) -> EvaluationResponse:
        self.last_evaluation_modes = (idea_taste_mode, prompt_mode)
        if diagnostic:
            self.diagnostic_calls += 1
        else:
            self.evaluator_calls += 1
        digest = hashlib.sha256(f"{idea_taste_mode}:{state.title}".encode()).digest()
        values = metrics(
            novelty=digest[0] % 6,
            feasibility=digest[1] % 6,
            conciseness=digest[2] % 6,
            risk=digest[3] % 6,
            complexity_penalty=digest[4] % 6,
        )
        defects = ("stagnant_novelty", "unclear_mechanism") if diagnostic else ("monolithic_design",)
        return EvaluationResponse(
            values,
            confidence=0.8,
            detected_defects=defects,
            usage=ProviderUsage(evaluator_calls=1, input_tokens=7, output_tokens=3, cost=0.01),
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        parent = request.parent
        child = IdeaState(
            title=f"{parent.title}/{request.plan.operator}/{request.seed:016x}",
            abstract=parent.abstract,
            core_contribution=f"{parent.core_contribution}; {request.plan.operator}",
            method=f"{parent.method}; bounded edit {request.plan.operator}",
            risks=parent.risks,
            components=parent.components,
            tags=parent.tags,
            root_domains=parent.root_domains,
        )
        return GenerationResponse(
            child,
            ProviderUsage(generation_calls=1, input_tokens=5, output_tokens=4, cost=0.02),
        )


class GroundingStub:
    def __init__(self, outcomes) -> None:
        self.outcomes = outcomes
        self.calls: list[str] = []

    def evaluate(self, operator, state, *, prompt_mode):
        del state, prompt_mode
        self.calls.append(operator)
        outcome = self.outcomes[operator]
        if isinstance(outcome, Exception):
            raise outcome
        kind = "theory_transfer" if operator == "theory-transfer-injection" else "mechanism_commit"
        return OperatorGroundingResult(
            grounding=OperatorGrounding(
                kind=kind,
                query=f"query for {operator}",
                needed_content="needed",
                expected_role="role",
            ),
            skip_operator=outcome == "skipped_empty",
            explicit_empty=outcome == "explicit_empty",
            query_provenance_json='{"query":"recorded"}',
            references_provenance_json='{"component_references":{}}',
            retrieval_provenance_json='{"retrieval":"recorded"}',
        )


class ResearchIdeaAlgorithmTest(unittest.TestCase):
    def test_exact_modes_weights_and_canonical_metadata(self) -> None:
        self.assertEqual(
            IDEA_TASTE_MODES,
            ("moonshot_inventor", "bridge_builder", "steady_engineer", "ambitious_realist", "evidence_first"),
        )
        for mode in IDEA_TASTE_MODES:
            self.assertAlmostEqual(sum(get_taste(mode).weights.values()), 1.0)
        self.assertEqual(ALGORITHM_ID, "xlab.research_idea.algorithm.v2")
        self.assertEqual(IMPLEMENTATION_METADATA["algorithm"], ALGORITHM_ID)
        self.assertTrue(IMPLEMENTATION_METADATA["native_reconstruction"])

    def test_ten_metrics_clip_round_and_invert_penalties(self) -> None:
        taste = get_taste("steady_engineer")
        favorable = Evaluation.from_payload(metrics(risk=-9, complexity_penalty=-4))
        adverse = Evaluation.from_payload(metrics(risk=99, complexity_penalty=99))
        self.assertEqual(tuple(favorable.metrics), METRICS)
        self.assertGreater(favorable.scalar(taste), adverse.scalar(taste))
        for raw in range(-20, 21):
            evaluation = Evaluation.from_payload(metrics(novelty=raw / 3))
            self.assertGreaterEqual(evaluation.metrics["novelty"], 0)
            self.assertLessEqual(evaluation.metrics["novelty"], 5)
            self.assertGreaterEqual(evaluation.scalar(taste), 0.0)
            self.assertLessEqual(evaluation.scalar(taste), 5.0)

    def test_postprocessing_overrides_novelty_and_falls_back_from_zero_protocol(self) -> None:
        evaluation = Evaluation.from_payload(metrics(novelty=1, protocol_score=0))
        processed = evaluation.postprocess(
            EvaluationPostprocessing(component_novelty=4.6, derived_protocol_score=2.6)
        )
        self.assertEqual(processed.metrics["novelty"], 5)
        self.assertEqual(processed.metrics["protocol_score"], 3)
        self.assertEqual(evaluation.metrics["novelty"], 1)
        self.assertEqual(evaluation.metrics["protocol_score"], 0)
        self.assertEqual(tuple(processed.metrics), METRICS)
        self.assertTrue(all(isinstance(value, int) for value in processed.metrics.values()))

    def test_postprocessing_preserves_positive_protocol_and_optional_novelty(self) -> None:
        evaluation = Evaluation.from_payload(metrics(novelty=2, protocol_score=4))
        processed = evaluation.postprocess(
            EvaluationPostprocessing(derived_protocol_score=1.0)
        )
        self.assertEqual(processed.metrics["novelty"], 2)
        self.assertEqual(processed.metrics["protocol_score"], 4)

    def test_postprocessing_rejects_invalid_external_scores(self) -> None:
        for field, value in (
            ("component_novelty", float("nan")),
            ("component_novelty", 6),
            ("derived_protocol_score", True),
            ("derived_protocol_score", -1),
        ):
            with self.subTest(field=field, value=value), self.assertRaisesRegex(
                ValueError, field
            ):
                EvaluationPostprocessing(**{field: value})

    def test_idea_state_rejects_empty_mandatory_scientific_text(self) -> None:
        for field in ("title", "abstract", "core_contribution", "method", "risks"):
            values = {
                "title": "Seed",
                "abstract": "Abstract",
                "core_contribution": "Contribution",
                "method": "Method",
                "risks": "Risks",
                "components": (IdeaComponent("core", "Core mechanism"),),
            }
            values[field] = "   "
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                IdeaState(**values)

    def test_textual_and_scientific_identity_are_separate(self) -> None:
        first = idea("First wording")

        def changed(**overrides) -> IdeaState:
            values = {
                "title": first.title,
                "abstract": first.abstract,
                "core_contribution": first.core_contribution,
                "method": first.method,
                "risks": first.risks,
                "components": first.components,
                "tags": first.tags,
                "root_domains": first.root_domains,
            }
            values.update(overrides)
            return IdeaState(**values)

        case_only = changed(title="FIRST WORDING")
        changed_whitespace = changed(title="  FIRST   WORDING ")
        changed_non_science = changed(
            abstract="Different abstract",
            risks="Different risks",
        )

        self.assertNotEqual(first.textual_identity, case_only.textual_identity)
        self.assertEqual(first.scientific_identity, case_only.scientific_identity)
        self.assertNotEqual(first.scientific_identity, changed_whitespace.scientific_identity)
        self.assertEqual(first.scientific_identity, changed_non_science.scientific_identity)

        mutations = (
            {"title": "Other title"},
            {"core_contribution": "Other contribution"},
            {"method": "Other method"},
            {"tags": ("other",)},
            {"root_domains": ("physics",)},
            {"components": (IdeaComponent("other", "The core mechanism"),)},
            {"components": (IdeaComponent("core", "Other description"),)},
            {
                "components": (
                    IdeaComponent("first", "First"),
                    IdeaComponent("second", "Second"),
                )
            },
            {
                "components": (
                    IdeaComponent("second", "Second"),
                    IdeaComponent("first", "First"),
                )
            },
        )
        identities = [changed(**mutation).scientific_identity for mutation in mutations]
        self.assertTrue(all(identity != first.scientific_identity for identity in identities))
        self.assertNotEqual(identities[-2], identities[-1])

    def test_scientific_identity_matches_upstream_canonical_signature(self) -> None:
        state = IdeaState(
            "Mixed CASE",
            "Abstract",
            "Core Contribution",
            "Method",
            "Risks",
            (
                IdeaComponent("zeta", "FIRST Explanation"),
                IdeaComponent("Alpha", "Second Explanation"),
            ),
            tags=("z-tag", "A-tag"),
            root_domains=("Computer Science", "Biology"),
        )
        canonical = "|".join(
            (
                "mixed case",
                "core contribution",
                "method",
                "A-tag,z-tag",
                "Computer Science,Biology",
                "Alpha,zeta",
                "zeta:first explanation|Alpha:second explanation",
            )
        )
        self.assertEqual(
            state.scientific_identity,
            hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )

        reversed_domains = IdeaState(
            state.title,
            state.abstract,
            state.core_contribution,
            state.method,
            state.risks,
            state.components,
            tags=state.tags,
            root_domains=tuple(reversed(state.root_domains)),
        )
        renamed_component = IdeaState(
            state.title,
            state.abstract,
            state.core_contribution,
            state.method,
            state.risks,
            (IdeaComponent("ZETA", "FIRST Explanation"), state.components[1]),
            tags=state.tags,
            root_domains=state.root_domains,
        )
        self.assertNotEqual(state.scientific_identity, reversed_domains.scientific_identity)
        self.assertNotEqual(state.scientific_identity, renamed_component.scientific_identity)

    def test_edit_plans_are_typed_approved_targeted_and_bounded(self) -> None:
        plans = approved_plans(("stagnant_novelty",), limit=8)
        self.assertTrue(plans)
        for plan in plans:
            self.assertTrue(plan.approved)
            self.assertIn("stagnant_novelty", plan.target_defects)
            self.assertLessEqual(len(plan.edits), MAX_EDITS)
            self.assertTrue(all(isinstance(edit.kind, EditKind) for edit in plan.edits))
        with self.assertRaises(ValueError):
            EditPlan(
                "bad",
                ("stagnant_novelty",),
                (AtomicEdit(EditKind.ADD_PROTOCOL, "ablation"),),
                "not approved",
                False,
            )

    def test_uct_prioritizes_unvisited_canonical_nodes_and_shares_transposition_stats(self) -> None:
        plan = approved_plans(("stagnant_novelty",), limit=1)[0]
        left = TreeNode(1, idea("left"), visits=2, value_sum=4.0)
        shared = TreeNode(2, idea("shared"))
        left_shared = TreeEdge(1, 2, plan)
        right_shared = TreeEdge(9, 2, plan)

        self.assertNotEqual(left.uct(10, 1.15), float("inf"))
        self.assertEqual(shared.uct(10, 1.15), float("inf"))
        MCTSEngine._backup([PathStep(shared, left_shared)], 3.0)
        self.assertEqual((shared.visits, shared.value_sum), (1, 3.0))
        self.assertEqual(left_shared.child_id, right_shared.child_id)
        self.assertNotEqual(shared.uct(10, 1.15), float("inf"))

    def test_attach_edge_supports_transpositions_and_rejects_invalid_edges(self) -> None:
        first_plan, second_plan = approved_plans(("stagnant_novelty",), limit=2)
        nodes = [
            TreeNode(0, idea("root")),
            TreeNode(1, idea("left")),
            TreeNode(2, idea("right")),
            TreeNode(3, idea("shared")),
        ]
        edge_keys: set[tuple[int, int, EditPlan]] = set()

        left_shared = MCTSEngine._attach_edge(
            nodes, nodes[1], nodes[3], first_plan, None, edge_keys
        )
        right_shared = MCTSEngine._attach_edge(
            nodes, nodes[2], nodes[3], first_plan, None, edge_keys
        )
        alternate = MCTSEngine._attach_edge(
            nodes, nodes[1], nodes[3], second_plan, None, edge_keys
        )

        self.assertIsNotNone(left_shared)
        self.assertIsNotNone(right_shared)
        self.assertIsNotNone(alternate)
        self.assertIsNone(
            MCTSEngine._attach_edge(
                nodes, nodes[1], nodes[3], first_plan, None, edge_keys
            )
        )
        self.assertIsNone(
            MCTSEngine._attach_edge(
                nodes, nodes[3], nodes[3], first_plan, None, edge_keys
            )
        )
        self.assertIsNone(
            MCTSEngine._attach_edge(
                nodes, nodes[3], nodes[1], first_plan, None, edge_keys
            )
        )

    def test_select_path_depth_is_path_local_for_shared_node(self) -> None:
        plan = approved_plans(("stagnant_novelty",), limit=1)[0]
        nodes = [
            TreeNode(0, idea("root"), visits=2, expanded=True),
            TreeNode(1, idea("deep-parent"), visits=1, value_sum=1.0, expanded=True),
            TreeNode(2, idea("shared"), visits=1, value_sum=5.0),
        ]
        direct = TreeEdge(0, 2, plan)
        deep = TreeEdge(0, 1, plan)
        to_shared = TreeEdge(1, 2, plan)
        nodes[0].children.extend((direct, deep))
        nodes[1].children.append(to_shared)
        engine = MCTSEngine(StubProvider(), SearchConfig(max_depth=2))

        selected = engine._select_path(nodes, nodes[0], random.Random(0))

        self.assertEqual([step.node.node_id for step in selected], [0, 2])
        self.assertEqual(len(selected) - 1, 1)

    def test_equal_uct_values_choose_first_inserted_child(self) -> None:
        plan = approved_plans(("stagnant_novelty",), limit=1)[0]
        nodes = [
            TreeNode(0, idea("root"), visits=2, expanded=True),
            TreeNode(1, idea("first")),
            TreeNode(2, idea("second")),
        ]
        nodes[0].children.extend((TreeEdge(0, 1, plan), TreeEdge(0, 2, plan)))
        engine = MCTSEngine(StubProvider(), SearchConfig(max_depth=1))

        for seed in range(5):
            selected = engine._select_path(nodes, nodes[0], random.Random(seed))
            self.assertEqual([step.node.node_id for step in selected], [0, 1])

    def test_search_config_validates_adaptive_success_threshold(self) -> None:
        self.assertEqual(SearchConfig().adaptive_success_threshold, 0.7)
        for value in (-0.01, 1.01, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "adaptive_success_threshold"
            ):
                SearchConfig(adaptive_success_threshold=value)

    def test_root_diagnostic_is_counted_not_backed_up_and_is_candidate(self) -> None:
        result = MCTSEngine(StubProvider(), SearchConfig(max_iterations=0)).search(
            idea(), "moonshot_inventor"
        )
        self.assertEqual(result.counters.iterations, 0)
        self.assertEqual(result.counters.root_diagnostics, 1)
        self.assertEqual(result.counters.evaluator_calls, 0)
        self.assertEqual(result.nodes[0].visits, 0)
        self.assertEqual(result.best.source, "root_diagnostic")
        self.assertEqual(len(result.candidates), 1)

    def test_backup_updates_only_canonical_nodes_on_selected_path(self) -> None:
        plan = approved_plans(("stagnant_novelty",), limit=1)[0]
        root = type("Node", (), {"visits": 0, "value_sum": 0.0})()
        left_edge = TreeEdge(0, 1, plan)
        right_edge = TreeEdge(0, 2, plan)
        left = type("Node", (), {"visits": 0, "value_sum": 0.0})()
        right = type("Node", (), {"visits": 0, "value_sum": 0.0})()
        MCTSEngine._backup([PathStep(root), PathStep(left, left_edge)], 3.25)
        self.assertEqual((root.visits, root.value_sum), (1, 3.25))
        self.assertEqual((left.visits, left.value_sum), (1, 3.25))
        self.assertEqual((right.visits, right.value_sum), (0, 0.0))
        self.assertEqual((left_edge.parent_id, left_edge.child_id), (0, 1))
        self.assertEqual((right_edge.parent_id, right_edge.child_id), (0, 2))

    def test_deterministic_per_mode_rng_and_explicit_tree(self) -> None:
        config = SearchConfig(max_iterations=6, max_depth=3, branching_factor=2, seed="fixture")
        first = MCTSEngine(StubProvider(), config).search(idea(), "bridge_builder")
        second = MCTSEngine(StubProvider(), config).search(idea(), "bridge_builder")
        other = MCTSEngine(StubProvider(), config).search(idea(), "steady_engineer")
        def shape(result):
            return (
                [(node.state.textual_identity, node.visits) for node in result.nodes],
                [
                    (edge.parent_id, edge.child_id, edge.plan.operator)
                    for node in result.nodes
                    for edge in node.children
                ],
            )

        first_shape = shape(first)
        second_shape = shape(second)
        self.assertEqual(first.rng_seed, second.rng_seed)
        self.assertEqual(first_shape, second_shape)
        self.assertNotEqual(first.rng_seed, other.rng_seed)
        for node in first.nodes:
            for edge in node.children:
                self.assertEqual(edge.parent_id, node.node_id)
                self.assertLess(edge.child_id, len(first.nodes))
        self.assertGreater(first.nodes[0].visits, 0)
        self.assertEqual(
            first.nodes[0].visits,
            sum(1 for candidate in first.candidates if candidate.source != "root_diagnostic"),
        )

    def test_typed_refinement_boundary_filters_search_plans(self) -> None:
        provider = StubProvider()
        result = MCTSEngine(
            provider,
            SearchConfig(max_iterations=1, branching_factor=3),
            context=IdeaProviderContext(
                refinement_scope=("Improve the current mechanism without widening the system.",),
                refinement_boundary=RefinementBoundary(
                    allowed_fields=("components",),
                    allowed_edit_kinds=("REPLACE_COMPONENT", "ADD_PROTOCOL"),
                ),
            ),
        ).search(idea(), "moonshot_inventor")

        self.assertEqual(result.counters.children, 1)
        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(
            {edit.kind.value for edit in provider.requests[0].plan.edits},
            {"REPLACE_COMPONENT", "ADD_PROTOCOL"},
        )
        self.assertEqual(
            provider.requests[0].context.refinement_scope,
            ("Improve the current mechanism without widening the system.",),
        )

    def test_one_outer_iteration_expands_and_evaluates_all_selected_children(self) -> None:
        provider = StubProvider()
        result = MCTSEngine(
            provider,
            SearchConfig(max_iterations=1, branching_factor=2, seed="batch-expansion"),
        ).search(idea(), "moonshot_inventor")

        self.assertEqual(result.counters.iterations, 1)
        self.assertEqual(result.counters.children, 2)
        self.assertEqual(result.counters.generation_calls, 2)
        self.assertEqual(result.counters.evaluator_calls, 2)
        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(len(result.nodes), 3)
        self.assertEqual(result.nodes[0].visits, 2)
        self.assertTrue(result.nodes[0].expanded)

    def test_parent_depth_one_evaluates_every_generated_child(self) -> None:
        class FixedPlanner:
            def plans(self, defects, *, limit, **kwargs):
                del defects, kwargs
                return approved_plans(("stagnant_novelty",), limit=limit)

            def update_prior(self, *args, **kwargs):
                del args, kwargs

        engine = MCTSEngine(
            StubProvider(),
            SearchConfig(max_iterations=3, max_depth=4, branching_factor=2),
        )
        engine.operator_planner = FixedPlanner()
        result = engine.search(idea(), "moonshot_inventor")

        expanded_depth_one = result.nodes[1]
        self.assertEqual(len(expanded_depth_one.children), 2)
        self.assertTrue(
            all(result.nodes[edge.child_id].evaluation for edge in expanded_depth_one.children)
        )

    def test_deep_expansion_keeps_unevaluated_provisional_sibling(self) -> None:
        class FixedPlanner:
            def plans(self, defects, *, limit, **kwargs):
                del defects, kwargs
                return approved_plans(("stagnant_novelty",), limit=limit)

            def update_prior(self, *args, **kwargs):
                del args, kwargs

        engine = MCTSEngine(
            StubProvider(),
            SearchConfig(max_iterations=4, max_depth=4, branching_factor=2),
        )
        engine.operator_planner = FixedPlanner()
        result = engine.search(idea(), "moonshot_inventor")

        deep_parents = [
            node
            for node in result.nodes
            if node.children
            and node.node_id != 0
            and any(result.nodes[edge.child_id].evaluation is None for edge in node.children)
        ]
        self.assertTrue(deep_parents)
        self.assertTrue(all(len(node.children) == 2 for node in deep_parents))
        self.assertTrue(
            all(
                sum(result.nodes[edge.child_id].evaluation is not None for edge in node.children)
                == 1
                for node in deep_parents
            )
        )
        self.assertEqual(result.counters.evaluator_calls, len(result.candidates) - 1)

    def test_counters_stopping_and_truthful_metric_champions(self) -> None:
        provider = StubProvider()
        result = MCTSEngine(
            provider,
            SearchConfig(max_iterations=20, max_children=2, branching_factor=2, seed="limits"),
        ).search(idea(), "evidence_first")
        self.assertEqual(result.stop_reason, "child_budget")
        self.assertEqual(result.counters.children, 2)
        self.assertEqual(result.counters.generation_calls, 2)
        self.assertEqual(result.counters.root_diagnostics, 1)
        self.assertEqual(result.counters.evaluator_calls, 2)
        self.assertEqual(result.counters.input_tokens, 31)
        self.assertEqual(result.counters.output_tokens, 17)
        self.assertAlmostEqual(result.counters.cost, 0.07)
        self.assertEqual(result.metadata["strategy"], "uct_mcts")
        self.assertEqual(result.metadata["champion_strategy"], "metric_champions_v1")
        self.assertFalse(result.metadata["champions_form_non_dominated_frontier"])
        for champion in result.champions:
            expected = max(candidate.evaluation.metrics[champion.metric] for candidate in result.candidates)
            self.assertEqual(champion.candidate.evaluation.metrics[champion.metric], expected)
            self.assertEqual(champion.objective, "max")
            self.assertEqual(champion.strategy, "metric_champions_v1")

    def test_each_budget_has_distinct_stop_reason(self) -> None:
        cases = (
            (SearchConfig(max_iterations=2, max_children=0), "child_budget"),
            (SearchConfig(max_iterations=2, max_evaluator_calls=0), "evaluator_budget"),
            (SearchConfig(max_iterations=2, max_tokens=10), "token_budget"),
            (SearchConfig(max_iterations=2, max_cost=0.01), "cost_budget"),
        )
        for config, reason in cases:
            with self.subTest(reason=reason):
                result = MCTSEngine(StubProvider(), config).search(idea(), "ambitious_realist")
                self.assertEqual(result.stop_reason, reason)

    def test_child_and_evaluator_limits_do_not_overshoot_or_launch_orphans(self) -> None:
        child_provider = StubProvider()
        child_limited = MCTSEngine(
            child_provider,
            SearchConfig(max_iterations=3, max_children=0),
        ).search(idea(), "bridge_builder")
        self.assertEqual(child_limited.counters.children, 0)
        self.assertEqual(child_limited.counters.generation_calls, 0)
        self.assertEqual(child_limited.counters.evaluator_calls, 0)
        self.assertEqual(len(child_provider.requests), 0)

        evaluator_provider = StubProvider()
        evaluator_limited = MCTSEngine(
            evaluator_provider,
            SearchConfig(max_iterations=3, max_children=3, max_evaluator_calls=0),
        ).search(idea(), "bridge_builder")
        self.assertEqual(evaluator_limited.counters.children, 0)
        self.assertEqual(evaluator_limited.counters.generation_calls, 0)
        self.assertEqual(evaluator_limited.counters.evaluator_calls, 0)
        self.assertEqual(len(evaluator_provider.requests), 0)

        exact_provider = StubProvider()
        exact = MCTSEngine(
            exact_provider,
            SearchConfig(max_iterations=5, max_children=1, max_evaluator_calls=1),
        ).search(idea(), "bridge_builder")
        self.assertEqual(exact.counters.children, 1)
        self.assertEqual(exact.counters.generation_calls, 1)
        self.assertEqual(exact.counters.evaluator_calls, 1)
        self.assertEqual(len(exact_provider.requests), 1)

    def test_cache_identity_covers_all_adapter_cache_dimensions(self) -> None:
        config = SearchConfig(
            max_iterations=0,
            runtime_profile="package-native",
            evaluator_profile="judge-model-v3",
            evidence_digest="evidence-sha256",
        )
        result = MCTSEngine(StubProvider(), config).search(idea(), "evidence_first")
        identity = result.best.cache_identity
        self.assertEqual(identity.state_textual_identity, result.best.state.textual_identity)
        self.assertEqual(identity.state_semantic_identity, result.best.state.semantic_identity)
        self.assertEqual(identity.idea_taste_mode, "evidence_first")
        self.assertEqual(identity.runtime_profile, "package-native")
        self.assertEqual(identity.evaluator_profile, "judge-model-v3")
        self.assertEqual(identity.algorithm_version, ALGORITHM_ID)
        self.assertEqual(identity.evidence_digest, "evidence-sha256")
        self.assertEqual(result.metadata["root_cache_identity"], identity.to_metadata())
        changed = MCTSEngine(
            StubProvider(),
            SearchConfig(max_iterations=0, runtime_profile="package-native", evaluator_profile="judge-model-v4", evidence_digest="evidence-sha256"),
        ).search(idea(), "evidence_first")
        self.assertNotEqual(identity.digest, changed.best.cache_identity.digest)
        other_mode = MCTSEngine(StubProvider(), config).search(idea(), "steady_engineer")
        self.assertNotEqual(identity.digest, other_mode.best.cache_identity.digest)

    def test_full_identity_cache_hits_do_not_consume_evaluator_budget(self) -> None:
        provider = StubProvider()
        engine = MCTSEngine(
            provider,
            SearchConfig(max_iterations=1, branching_factor=1, seed="cache"),
        )
        first = engine.search(idea(), "moonshot_inventor")
        second = engine.search(idea(), "moonshot_inventor")
        self.assertEqual(first.counters.evaluator_calls, 1)
        self.assertEqual(second.counters.evaluator_calls, 0)
        self.assertEqual(second.counters.evaluation_cache_hits, 1)
        self.assertEqual(provider.evaluator_calls, 1)

    def test_prompt_modes_are_explicit_and_priors_update_after_rollouts(self) -> None:
        class RecordingPlanner:
            def __init__(self) -> None:
                self.thresholds = []

            def plans(self, defects, *, limit, **kwargs):
                del defects, kwargs
                return approved_plans(("stagnant_novelty",), limit=limit)

            def update_prior(self, *args, success_threshold, **kwargs):
                del args, kwargs
                self.thresholds.append(success_threshold)

        provider = StubProvider()
        engine = MCTSEngine(
            provider,
            SearchConfig(
                max_iterations=1,
                branching_factor=1,
                adaptive_success_threshold=0.83,
            ),
        )
        planner = RecordingPlanner()
        engine.operator_planner = planner
        engine.search(idea(), "moonshot_inventor")
        self.assertEqual(provider.last_evaluation_modes, ("moonshot_inventor", "conceptual_surprise"))
        self.assertEqual(provider.requests[0].idea_taste_mode, "moonshot_inventor")
        self.assertEqual(provider.requests[0].prompt_mode, "conceptual_surprise")
        self.assertEqual(planner.thresholds, [0.83])

    def test_attempt_records_preserve_empty_grounding_semantics_without_retry(self) -> None:
        provider = StubProvider()
        grounding = GroundingStub({
            "theory-transfer-injection": "skipped_empty",
            "mechanism-commit-innovation": "explicit_empty",
        })
        result = MCTSEngine(
            provider,
            SearchConfig(max_iterations=3, branching_factor=2),
            operator_grounding_evaluator=grounding,
        ).search(idea(), "moonshot_inventor")

        root_records = [
            record for record in result.operator_attempts if record.parent_node_id == 0
        ]
        self.assertEqual(
            [(record.operator, record.outcome) for record in root_records],
            [
                ("mechanism-commit-innovation", "applied"),
                ("theory-transfer-injection", "skipped_empty"),
            ],
        )
        self.assertTrue(root_records[0].grounding_explicit_empty)
        self.assertFalse(root_records[1].grounding_explicit_empty)
        self.assertEqual(len({record.plan_digest for record in root_records}), 2)
        self.assertEqual(
            sum(request.parent.scientific_identity == idea().scientific_identity for request in provider.requests),
            1,
        )
        self.assertEqual(root_records[0].to_payload()["query_provenance"], {"query": "recorded"})
        self.assertEqual(result.rollout_blockers, ())

    def test_attempt_payload_requires_a_literal_boolean(self) -> None:
        payload = {
            "parent_node_id": 0,
            "operator": "mechanism-commit-innovation",
            "plan_digest": "plan",
            "outcome": "applied",
            "grounding_explicit_empty": True,
            "query_provenance": None,
            "references_provenance": None,
            "retrieval_provenance": None,
        }
        record = OperatorAttemptRecord.from_payload(payload)
        self.assertTrue(record.grounding_explicit_empty)
        self.assertEqual(OperatorAttemptRecord.from_payload(record.to_payload()), record)

        payload["grounding_explicit_empty"] = "false"
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            OperatorAttemptRecord.from_payload(payload)
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            OperatorAttemptRecord(0, "operator", "plan", "applied", 1)  # type: ignore[arg-type]

    def test_plan_local_generation_failure_is_blocked_without_search_mutation(self) -> None:
        class FailingProvider(StubProvider):
            def generate(self, request):
                self.requests.append(request)
                raise RuntimeError("generation failed")

        provider = FailingProvider()
        engine = MCTSEngine(
            provider,
            SearchConfig(max_iterations=2, branching_factor=1),
        )
        result = engine.search(idea(), "moonshot_inventor")

        self.assertEqual(len(provider.requests), 1)
        self.assertEqual([record.outcome for record in result.operator_attempts], ["error"])
        self.assertEqual([record.stage for record in result.rollout_blockers], ["generation"])
        self.assertEqual(len(result.nodes), 1)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.counters.children, 0)
        self.assertEqual(result.counters.evaluator_calls, 0)
        self.assertEqual(result.nodes[0].visits, 0)
        self.assertNotIn(provider.requests[0].plan.operator, engine.operator_planner.priors)
        self.assertEqual(result.counters.generation_calls, 1)

    def test_plan_local_evaluation_failure_does_not_create_or_cache_node(self) -> None:
        class FailingProvider(StubProvider):
            def evaluate(self, state, *, diagnostic, **kwargs):
                if not diagnostic:
                    raise ValueError("evaluation failed")
                return super().evaluate(state, diagnostic=diagnostic, **kwargs)

        provider = FailingProvider()
        engine = MCTSEngine(provider, SearchConfig(max_iterations=2, branching_factor=1))
        result = engine.search(idea(), "moonshot_inventor")

        self.assertEqual(len(provider.requests), 1)
        self.assertEqual([record.stage for record in result.rollout_blockers], ["evaluation"])
        self.assertEqual(len(result.nodes), 1)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.counters.children, 0)
        self.assertEqual(result.nodes[0].visits, 0)
        self.assertEqual(len(engine._evaluation_cache), 1)
        self.assertNotIn(provider.requests[0].plan.operator, engine.operator_planner.priors)
        self.assertEqual(result.counters.evaluator_calls, 1)

    def test_root_diagnostic_failure_remains_mode_fatal(self) -> None:
        class FailingProvider(StubProvider):
            def evaluate(self, *args, **kwargs):
                raise RuntimeError("root diagnostic failed")

        with self.assertRaisesRegex(RuntimeError, "root diagnostic failed"):
            MCTSEngine(FailingProvider(), SearchConfig(max_iterations=1)).search(
                idea(), "moonshot_inventor"
            )

    def test_symbolic_memory_is_method_scoped(self) -> None:
        memory = MemoryState([
            SymbolicMemoryRecord("gate", "control.gate", "positive", method_context="method alpha"),
            SymbolicMemoryRecord("router", "control.router", "negative", method_context="method beta"),
        ])
        self.assertEqual(len(memory.hints(method_context="Method Alpha with controls")), 1)

    def test_symbolic_signs_and_pro_vector_memory_state(self) -> None:
        records = [
            SymbolicMemoryRecord("gate", "control.gate", "positive"),
            SymbolicMemoryRecord("router", "control.router", "negative"),
        ]
        memory = MemoryState(records, vector_memory_requested=True, symbolic_memory_only=True)
        self.assertEqual(records[0].removal_effect, "removal_helped")
        self.assertEqual(records[1].removal_effect, "removal_hurt")
        self.assertFalse(memory.vector_memory_effective)
        self.assertTrue(MemoryState(records, True, False).vector_memory_effective)

    def test_structured_feedback_builds_symbolic_memory_and_cache_identity(self) -> None:
        feedback = {
            "ablation_results": {
                "components": {
                    "gate": {
                        "component_family": "control.gate",
                        "operation": "remove",
                        "result": "positive",
                        "confidence": 0.9,
                        "metric": "accuracy",
                    },
                    "router": {
                        "component_family": "control.router",
                        "operation": "remove",
                        "result": "negative",
                        "confidence": 0.8,
                    },
                }
            }
        }
        memory = MemoryState.from_experiment_feedback(feedback)
        self.assertEqual(
            [record.removal_effect for record in memory.symbolic_records],
            ["removal_helped", "removal_hurt"],
        )
        self.assertIn("control.gate/gate: removal_helped", memory.hints()[0])
        result = MCTSEngine(
            StubProvider(), SearchConfig(max_iterations=0), memory=memory
        ).search(idea(), "evidence_first")
        self.assertEqual(result.best.cache_identity.memory_digest, memory.digest)


if __name__ == "__main__":
    unittest.main()
