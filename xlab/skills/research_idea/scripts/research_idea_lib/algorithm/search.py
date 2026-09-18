"""Deterministic explicit-tree Monte Carlo tree search."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field, asdict, is_dataclass
from enum import Enum
from pathlib import Path
from . import contracts, component_novelty, evaluation, operator_grounding, operators
from .search_checkpoint import SearchCheckpoint
from typing import Literal

from ..providers.failures import raise_infrastructure_failure

from .component_novelty import (
    ComponentNoveltyEvaluator,
    ComponentNoveltyResult,
    apply_component_novelty,
)
from .contracts import (
    CacheIdentity,
    GenerationRequest,
    IdeaProviderContext,
    IdeaState,
    OperatorAttemptRecord,
    PromptMode,
    ResearchIdeaProvider,
    ProviderUsage,
    RolloutBlockerRecord,
)
from .evaluation import Evaluation
from .memory import MemoryState
from .operator_grounding import (
    MECHANISM_COMMIT_OPERATOR,
    THEORY_TRANSFER_OPERATOR,
    OperatorGroundingEvaluator,
    OperatorGroundingResult,
)
from .operators import EditPlan, OperatorPlanner, StructuralProfile
from .spec import IMPLEMENTATION_METADATA
from .tastes import IDEA_TASTE_MODES, get_taste

StopReason = Literal[
    "iteration_budget", "child_budget", "evaluator_budget", "token_budget", "cost_budget", "exhausted"
]


@dataclass(frozen=True)
class SearchConfig:
    max_iterations: int = 128
    max_depth: int = 4
    branching_factor: int = 3
    exploration_constant: float = 1.15
    max_children: int | None = None
    max_evaluator_calls: int | None = None
    max_tokens: int | None = None
    max_cost: float | None = None
    seed: str = "0"
    runtime_profile: str = "package-native"
    evaluator_profile: str = "provider-default"
    evidence_digest: str = "no-evidence"
    component_novelty_required: bool = False
    prompt_mode: PromptMode = "conceptual_surprise"
    max_duplicate_attempts: int = 3
    adaptive_success_threshold: float = 0.7

    def __post_init__(self) -> None:
        if self.max_iterations < 0 or self.max_depth < 0 or self.branching_factor < 1:
            raise ValueError("invalid search bounds")
        if self.max_duplicate_attempts < 1:
            raise ValueError("max_duplicate_attempts must be positive")
        if (
            not math.isfinite(self.adaptive_success_threshold)
            or not 0 <= self.adaptive_success_threshold <= 1
        ):
            raise ValueError("adaptive_success_threshold must be finite and in [0, 1]")
        if any(value is not None and value < 0 for value in (
            self.max_children,
            self.max_evaluator_calls,
            self.max_tokens,
            self.max_cost,
        )):
            raise ValueError("budget limits cannot be negative")
        if not self.runtime_profile or not self.evaluator_profile or not self.evidence_digest:
            raise ValueError("cache identity profiles and evidence digest cannot be empty")


@dataclass
class SearchCounters:
    iterations: int = 0
    children: int = 0
    root_diagnostics: int = 0
    evaluator_calls: int = 0
    evaluation_cache_hits: int = 0
    duplicate_generations: int = 0
    generation_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add_usage(self, usage: ProviderUsage) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cost += usage.cost

    def add_root_diagnostic(self, usage: ProviderUsage | None = None) -> None:
        self.root_diagnostics += 1
        if usage is not None:
            self.add_usage(usage)

    def add_evaluator_call(self, usage: ProviderUsage | None = None) -> None:
        self.evaluator_calls += 1
        if usage is not None:
            self.add_usage(usage)

    def add_generation_call(self, usage: ProviderUsage | None = None) -> None:
        self.generation_calls += 1
        if usage is not None:
            self.add_usage(usage)


@dataclass(frozen=True)
class TreeEdge:
    parent_id: int
    child_id: int
    plan: EditPlan
    grounding_result: OperatorGroundingResult | None = None


@dataclass
class TreeNode:
    node_id: int
    state: IdeaState
    visits: int = 0
    value_sum: float = 0.0
    evaluation: Evaluation | None = None
    children: list[TreeEdge] = field(default_factory=list)
    expanded: bool = False

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def uct(self, parent_visits: int, exploration_constant: float) -> float:
        if self.visits == 0:
            return math.inf
        return self.mean_value + exploration_constant * math.sqrt(
            math.log(max(1, parent_visits)) / self.visits
        )


@dataclass(frozen=True)
class PathStep:
    node: TreeNode
    incoming_edge: TreeEdge | None = None


@dataclass(frozen=True)
class Candidate:
    node_id: int
    state: IdeaState
    evaluation: Evaluation
    score: float
    source: Literal["root_diagnostic", "rollout"]
    cache_identity: CacheIdentity
    component_novelty: ComponentNoveltyResult | None = None


@dataclass(frozen=True)
class Champion:
    label: str
    metric: str
    objective: Literal["max"]
    candidate: Candidate
    strategy: str = "metric_champions_v1"


@dataclass(frozen=True)
class SearchResult:
    mode: str
    best: Candidate
    champions: tuple[Champion, ...]
    candidates: tuple[Candidate, ...]
    nodes: tuple[TreeNode, ...]
    counters: SearchCounters
    stop_reason: StopReason
    rng_seed: int
    operator_attempts: tuple[OperatorAttemptRecord, ...]
    rollout_blockers: tuple[RolloutBlockerRecord, ...]
    metadata: dict[str, object]


class _RolloutStageError(RuntimeError):
    def __init__(self, stage: Literal["evaluation", "component_novelty"], error: Exception) -> None:
        super().__init__(str(error))
        self.stage = stage
        self.error = error


class MCTSEngine:
    def __init__(
        self,
        provider: ResearchIdeaProvider,
        config: SearchConfig | None = None,
        memory: MemoryState | None = None,
        context: IdeaProviderContext | None = None,
        novelty_evaluator: ComponentNoveltyEvaluator | None = None,
        operator_grounding_evaluator: OperatorGroundingEvaluator | None = None,
        component_novelty_resource_digest: str = "no-component-novelty-resource",
    ):
        self.provider = provider
        self.config = config or SearchConfig()
        self.memory = memory or MemoryState()
        self.novelty_evaluator = novelty_evaluator
        self.operator_grounding_evaluator = operator_grounding_evaluator
        self.component_novelty_resource_digest = component_novelty_resource_digest
        if not self.component_novelty_resource_digest:
            raise ValueError("component novelty resource digest cannot be empty")
        if self.novelty_evaluator is None and self.config.component_novelty_required:
            raise ValueError("search requires a component novelty evaluator")
        supplied = context or IdeaProviderContext()
        scoped_hints = self.memory.hints(method_context=(supplied.mature_idea.method if supplied.mature_idea else ""))
        self.context = IdeaProviderContext(
            evidence=supplied.evidence,
            mature_idea=supplied.mature_idea,
            refinement_scope=supplied.refinement_scope,
            refinement_boundary=supplied.refinement_boundary,
            memory_hints=tuple(dict.fromkeys((*supplied.memory_hints, *scoped_hints))),
            task_context_json=supplied.task_context_json,
            research_policy_json=supplied.research_policy_json,
        )
        self.operator_planner = OperatorPlanner()
        self._evaluation_cache: dict[str, tuple[Evaluation, float, ComponentNoveltyResult | None]] = {}

    @staticmethod
    def mode_seed(seed: str, mode: str) -> int:
        digest = hashlib.sha256(f"{seed}\0{mode}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big")

    def search(self, root_state: IdeaState, mode: str, *, checkpoint_path: Path | None = None,
               export_provider_state=None, restore_provider_state=None) -> SearchResult:
        if mode not in IDEA_TASTE_MODES:
            raise ValueError(f"unknown idea taste: {mode}")
        taste = get_taste(mode)
        rng_seed = self.mode_seed(self.config.seed, mode)
        rng = random.Random(rng_seed)
        counters = SearchCounters()
        root = TreeNode(0, root_state)
        nodes = [root]
        candidates: list[Candidate] = []
        candidates_by_node: dict[int, Candidate] = {}
        operator_attempts: list[OperatorAttemptRecord] = []
        rollout_blockers: list[RolloutBlockerRecord] = []
        attempted_plans: dict[int, set[EditPlan]] = {}

        registry = {}
        for namespace in [globals(), *(vars(module) for module in
                         (contracts, component_novelty, evaluation, operator_grounding, operators))]:
            for name, value in namespace.items():
                if isinstance(value, type) and (is_dataclass(value) or issubclass(value, Enum)):
                    registry[name] = value
        identity = hashlib.sha256(json.dumps({'root': root_state.to_payload(), 'mode': mode,
            'config': asdict(self.config), 'context': self.context.to_payload(),
            'memory': self.memory.digest}, sort_keys=True).encode()).hexdigest()
        checkpoint = SearchCheckpoint(checkpoint_path, identity, registry) if checkpoint_path else None
        resumed = checkpoint.load() if checkpoint else None
        if resumed:
            nodes, candidates = resumed['nodes'], resumed['candidates']
            root = nodes[0]
            counters = resumed['counters']
            operator_attempts, rollout_blockers = resumed['operator_attempts'], resumed['rollout_blockers']
            attempted_plans, edge_keys = resumed['attempted_plans'], resumed['edge_keys']
            rng.setstate(resumed['rng_state'])
            self.operator_planner = resumed['operator_planner']
            self._evaluation_cache = resumed['evaluation_cache']
            candidates_by_node = {item.node_id:item for item in candidates}
            states_by_scientific_identity = {node.state.scientific_identity:node for node in nodes}
            if restore_provider_state:
                restore_provider_state(resumed['provider_state'])
        else:
            counters.add_root_diagnostic()
            diagnostic = self.provider.evaluate(
                root_state,
                idea_taste_mode=mode,
                prompt_mode=self.config.prompt_mode,
                diagnostic=True,
                context=self.context,
            )
            counters.add_usage(diagnostic.usage)
            root.evaluation = Evaluation.from_payload(
                diagnostic.metrics,
                confidence=diagnostic.confidence,
                detected_defects=diagnostic.detected_defects,
                feedback=diagnostic.feedback,
            )
            root.evaluation, root_score, root_novelty = self._apply_novelty(
                root_state,
                root.evaluation,
                taste,
                candidate_id="root",
                idea_taste_mode=mode,
            )
            candidates.append(Candidate(
                0,
                root_state,
                root.evaluation,
                root_score,
                "root_diagnostic",
                self._cache_identity(root_state, mode),
                root_novelty,
            ))
            candidates_by_node[0] = candidates[0]
            self._evaluation_cache[candidates[0].cache_identity.digest] = (
                root.evaluation,
                root_score,
                root_novelty,
            )
            states_by_scientific_identity = {root_state.scientific_identity: root}
            edge_keys: set[tuple[int, int, EditPlan]] = set()
    
        def save_iteration(status='running'):
            if checkpoint:
                checkpoint.save({'nodes':nodes, 'candidates':candidates, 'counters':counters,
                    'operator_attempts':operator_attempts, 'rollout_blockers':rollout_blockers,
                    'attempted_plans':attempted_plans, 'edge_keys':edge_keys,
                    'rng_state':rng.getstate(), 'operator_planner':self.operator_planner,
                    'evaluation_cache':self._evaluation_cache,
                    'provider_state':export_provider_state() if export_provider_state else {}}, status=status)

        stop_reason: StopReason = "iteration_budget"
        while counters.iterations < self.config.max_iterations:
            save_iteration()
            limit = self._budget_reason(counters)
            if limit:
                stop_reason = limit
                break
            path = self._select_path(nodes, root, rng)
            leaf = path[-1].node
            grounding_digest = self._path_grounding_digest(path)
            candidate = candidates_by_node.get(leaf.node_id)
            expected_identity = self._cache_identity(
                leaf.state,
                mode,
                grounding_digest=grounding_digest,
            )
            if (
                candidate is None
                or candidate.cache_identity.digest != expected_identity.digest
            ):
                try:
                    candidate = self._evaluate(
                        leaf,
                        mode,
                        taste,
                        counters,
                        grounding_digest=grounding_digest,
                    )
                except _RolloutStageError as staged:
                    raise_infrastructure_failure(staged.error)
                    incoming_edge = path[-1].incoming_edge
                    if incoming_edge is None:
                        raise
                    plan = incoming_edge.plan
                    plan_digest = self._plan_digest(plan)
                    operator_attempts.append(self._attempt_record(
                        incoming_edge.parent_id,
                        plan,
                        plan_digest,
                        "error",
                        incoming_edge.grounding_result,
                    ))
                    rollout_blockers.append(self._blocker_record(
                        incoming_edge.parent_id,
                        plan,
                        plan_digest,
                        staged.stage,
                        staged.error,
                    ))
                    counters.iterations += 1
                    continue
                candidates.append(candidate)
                candidates_by_node[leaf.node_id] = candidate
                incoming_edge = path[-1].incoming_edge
                if incoming_edge is not None:
                    self.operator_planner.update_prior(
                        incoming_edge.plan.operator,
                        candidate.score / 5.0,
                        failure_modes=candidate.evaluation.detected_defects,
                        success_threshold=self.config.adaptive_success_threshold,
                    )
                self._backup(path, candidate.score)
                counters.iterations += 1
                continue
            if len(path) - 1 >= self.config.max_depth:
                self._backup(path, candidate.score)
                counters.iterations += 1
                continue

            defects = leaf.evaluation.detected_defects if leaf.evaluation else ("unexplored_gap",)
            plans = self.operator_planner.plans(
                defects or ("unexplored_gap",),
                limit=self.config.branching_factor,
                profile=self._structural_profile(leaf.state),
                taste=taste,
                rng=rng,
            )
            plans = tuple(plan for plan in plans if self._plan_allowed(plan))
            if not plans:
                stop_reason = "exhausted"
                break
            attempted_for_leaf = attempted_plans.setdefault(leaf.node_id, set())
            unexpanded = tuple(plan for plan in plans if plan not in attempted_for_leaf)
            if not unexpanded:
                leaf.expanded = True
                if not leaf.children:
                    stop_reason = "exhausted"
                    break
                continue

            parent_depth = len(path) - 1
            remaining_children = (
                len(unexpanded)
                if self.config.max_children is None
                else self.config.max_children - counters.children
            )
            remaining_evaluations = (
                len(unexpanded) if parent_depth <= 1 else 1
            )
            if self.config.max_evaluator_calls is not None:
                remaining_evaluations = min(
                    remaining_evaluations,
                    self.config.max_evaluator_calls - counters.evaluator_calls,
                )
            batch_limit = (
                min(remaining_children, remaining_evaluations)
                if parent_depth <= 1
                else remaining_children
            )
            batch = unexpanded[:batch_limit]
            if not batch or remaining_evaluations <= 0:
                stop_reason = (
                    "child_budget" if remaining_children <= 0 else "evaluator_budget"
                )
                break

            evaluated_deep_target = False
            for plan in batch:
                attempted_for_leaf.add(plan)
                plan_digest = self._plan_digest(plan)
                grounding_result: OperatorGroundingResult | None = None
                try:
                    grounding_result = self._ground_plan(plan, leaf.state)
                except Exception as error:
                    raise_infrastructure_failure(error)
                    operator_attempts.append(self._attempt_record(
                        leaf.node_id, plan, plan_digest, "error"
                    ))
                    rollout_blockers.append(self._blocker_record(
                        leaf.node_id, plan, plan_digest, "grounding", error
                    ))
                    continue
                if grounding_result is not None and grounding_result.skip_operator:
                    operator_attempts.append(self._attempt_record(
                        leaf.node_id,
                        plan,
                        plan_digest,
                        "skipped_empty",
                        grounding_result,
                    ))
                    continue
                grounding = grounding_result.grounding if grounding_result is not None else None
                grounding_digest = (
                    grounding.digest if grounding is not None else "no-operator-grounding"
                )
                child: TreeNode | None = None
                existing_candidate: Candidate | None = None
                generation_failed = False
                for _ in range(self.config.max_duplicate_attempts):
                    counters.add_generation_call()
                    try:
                        response = self.provider.generate(
                            GenerationRequest(
                                idea_taste_mode=mode,
                                parent=leaf.state,
                                plan=plan,
                                seed=rng.getrandbits(64),
                                memory_hints=self.context.memory_hints,
                                context=self.context,
                                prompt_mode=self.config.prompt_mode,
                                grounding=grounding,
                            )
                        )
                    except Exception as error:
                        raise_infrastructure_failure(error)
                        operator_attempts.append(self._attempt_record(
                            leaf.node_id,
                            plan,
                            plan_digest,
                            "error",
                            grounding_result,
                        ))
                        rollout_blockers.append(self._blocker_record(
                            leaf.node_id, plan, plan_digest, "generation", error
                        ))
                        generation_failed = True
                        child = None
                        break
                    counters.add_usage(response.usage)
                    identity = response.state.scientific_identity
                    child = states_by_scientific_identity.get(identity)
                    if child is None:
                        child = TreeNode(len(nodes), response.state)
                    elif not self._edge_allowed(nodes, leaf, child, plan, edge_keys):
                        counters.duplicate_generations += 1
                        child = None
                        continue
                    existing_candidate = candidates_by_node.get(child.node_id)
                    if child.node_id < len(nodes) and existing_candidate is not None:
                        current_identity = self._cache_identity(
                            child.state,
                            mode,
                            grounding_digest=grounding_digest,
                        )
                        if existing_candidate.cache_identity.digest != current_identity.digest:
                            existing_candidate = None
                    break
                else:
                    child = None
                if child is None:
                    if not generation_failed:
                        operator_attempts.append(self._attempt_record(
                            leaf.node_id,
                            plan,
                            plan_digest,
                            "duplicate_exhausted",
                            grounding_result,
                        ))
                    continue

                should_evaluate = parent_depth <= 1 or not evaluated_deep_target
                candidate = existing_candidate
                candidate_was_evaluated = False
                if should_evaluate and candidate is None:
                    try:
                        candidate = self._evaluate(
                            child,
                            mode,
                            taste,
                            counters,
                            grounding_digest=grounding_digest,
                        )
                        candidate_was_evaluated = True
                    except _RolloutStageError as staged:
                        raise_infrastructure_failure(staged.error)
                        operator_attempts.append(self._attempt_record(
                            leaf.node_id,
                            plan,
                            plan_digest,
                            "error",
                            grounding_result,
                        ))
                        rollout_blockers.append(self._blocker_record(
                            leaf.node_id,
                            plan,
                            plan_digest,
                            staged.stage,
                            staged.error,
                        ))
                        continue

                edge = self._attach_edge(
                    nodes,
                    leaf,
                    child,
                    plan,
                    grounding_result,
                    edge_keys,
                )
                if edge is None:
                    counters.duplicate_generations += 1
                    operator_attempts.append(self._attempt_record(
                        leaf.node_id,
                        plan,
                        plan_digest,
                        "attachment_rejected",
                        grounding_result,
                    ))
                    continue
                if child.node_id == len(nodes):
                    nodes.append(child)
                    states_by_scientific_identity[child.state.scientific_identity] = child
                if candidate_was_evaluated:
                    candidates.append(candidate)
                    candidates_by_node[child.node_id] = candidate
                counters.children += 1
                operator_attempts.append(self._attempt_record(
                    leaf.node_id,
                    plan,
                    plan_digest,
                    "applied",
                    grounding_result,
                ))
                if should_evaluate and candidate is not None:
                    evaluated_deep_target = True
                    rollout_path = [*path, PathStep(child, edge)]
                    self.operator_planner.update_prior(
                        plan.operator,
                        candidate.score / 5.0,
                        failure_modes=candidate.evaluation.detected_defects,
                        success_threshold=self.config.adaptive_success_threshold,
                    )
                    self._backup(rollout_path, candidate.score)
                if self._budget_reason(counters) in {"token_budget", "cost_budget"}:
                    break
            leaf.expanded = all(plan in attempted_for_leaf for plan in plans)
            counters.iterations += 1
        else:
            stop_reason = "iteration_budget"

        save_iteration('completed')
        best = max(candidates, key=lambda item: (item.score, -item.node_id))
        champions = tuple(
            Champion(label, metric, "max", max(candidates, key=lambda item: (item.evaluation.metrics[metric], item.score, -item.node_id)))
            for label, metric in (("novel", "novelty"), ("feasible", "feasibility"), ("concise", "conciseness"))
        )
        metadata = dict(IMPLEMENTATION_METADATA)
        metadata.update({
            "mode": mode,
            "strategy": "uct_mcts",
            "champion_strategy": "metric_champions_v1",
            "champions_form_non_dominated_frontier": False,
            "root_cache_identity": candidates[0].cache_identity.to_metadata(),
        })
        return SearchResult(
            mode,
            best,
            champions,
            tuple(candidates),
            tuple(nodes),
            counters,
            stop_reason,
            rng_seed,
            tuple(operator_attempts),
            tuple(rollout_blockers),
            metadata,
        )

    def _select_path(
        self,
        nodes: list[TreeNode],
        root: TreeNode,
        rng: random.Random,
    ) -> list[PathStep]:
        del rng  # UCT ties intentionally retain first-insertion order.
        path = [PathStep(root)]
        node = root
        while (
            len(path) - 1 < self.config.max_depth
            and node.expanded
            and node.children
        ):
            values = [
                nodes[edge.child_id].uct(
                    node.visits,
                    self.config.exploration_constant,
                )
                for edge in node.children
            ]
            edge = node.children[values.index(max(values))]
            node = nodes[edge.child_id]
            path.append(PathStep(node, edge))
        return path

    @staticmethod
    def _edge_allowed(
        nodes: list[TreeNode],
        parent: TreeNode,
        child: TreeNode,
        plan: EditPlan,
        edge_keys: set[tuple[int, int, EditPlan]],
    ) -> bool:
        key = (parent.node_id, child.node_id, plan)
        if parent.node_id == child.node_id or key in edge_keys:
            return False
        if child.node_id < len(nodes):
            pending = [child.node_id]
            seen: set[int] = set()
            while pending:
                node_id = pending.pop()
                if node_id == parent.node_id:
                    return False
                if node_id in seen:
                    continue
                seen.add(node_id)
                pending.extend(edge.child_id for edge in nodes[node_id].children)
        return True

    @staticmethod
    def _attach_edge(
        nodes: list[TreeNode],
        parent: TreeNode,
        child: TreeNode,
        plan: EditPlan,
        grounding_result: OperatorGroundingResult | None,
        edge_keys: set[tuple[int, int, EditPlan]],
    ) -> TreeEdge | None:
        if not MCTSEngine._edge_allowed(nodes, parent, child, plan, edge_keys):
            return None
        key = (parent.node_id, child.node_id, plan)
        edge = TreeEdge(
            parent.node_id,
            child.node_id,
            plan,
            grounding_result=grounding_result,
        )
        parent.children.append(edge)
        edge_keys.add(key)
        return edge

    def _evaluate(
        self,
        node: TreeNode,
        mode: str,
        taste,
        counters: SearchCounters,
        *,
        grounding_digest: str = "no-operator-grounding",
    ) -> Candidate:
        identity = self._cache_identity(
            node.state,
            mode,
            grounding_digest=grounding_digest,
        )
        cached = self._evaluation_cache.get(identity.digest)
        if cached is not None:
            counters.evaluation_cache_hits += 1
            node.evaluation, score, novelty = cached
            return Candidate(node.node_id, node.state, node.evaluation, score, "rollout", identity, novelty)
        try:
            counters.add_evaluator_call()
            response = self.provider.evaluate(
                node.state,
                idea_taste_mode=mode,
                prompt_mode=self.config.prompt_mode,
                diagnostic=False,
                context=self.context,
            )
            counters.add_usage(response.usage)
            evaluation = Evaluation.from_payload(
                response.metrics,
                confidence=response.confidence,
                detected_defects=response.detected_defects,
                feedback=response.feedback,
            )
        except Exception as error:
            raise _RolloutStageError("evaluation", error) from error
        try:
            evaluation, score, novelty = self._apply_novelty(
                node.state,
                evaluation,
                taste,
                candidate_id=f"node:{node.node_id}",
                idea_taste_mode=mode,
            )
        except Exception as error:
            raise _RolloutStageError("component_novelty", error) from error
        node.evaluation = evaluation
        self._evaluation_cache[identity.digest] = (node.evaluation, score, novelty)
        return Candidate(
            node.node_id,
            node.state,
            node.evaluation,
            score,
            "rollout",
            identity,
            novelty,
        )

    def _apply_novelty(
        self,
        state: IdeaState,
        evaluation: Evaluation,
        taste,
        *,
        candidate_id: str,
        idea_taste_mode: str,
    ) -> tuple[Evaluation, float, ComponentNoveltyResult | None]:
        if self.novelty_evaluator is None:
            return evaluation, evaluation.scalar(taste), None
        novelty = self.novelty_evaluator.evaluate(
            state,
            candidate_id=candidate_id,
            idea_taste_mode=idea_taste_mode,
            topic=state.title,
        )
        adjusted = apply_component_novelty(evaluation, novelty, taste)
        return adjusted.evaluation, adjusted.composite, novelty

    @staticmethod
    def _path_grounding_digest(path: list[PathStep]) -> str:
        edge = path[-1].incoming_edge
        if edge is None or edge.grounding_result is None:
            return "no-operator-grounding"
        grounding = edge.grounding_result.grounding
        return grounding.digest if grounding is not None else "no-operator-grounding"

    def _cache_identity(
        self,
        state: IdeaState,
        mode: str,
        *,
        grounding_digest: str = "no-operator-grounding",
    ) -> CacheIdentity:
        return CacheIdentity.for_state(
            state,
            idea_taste_mode=mode,
            runtime_profile=self.config.runtime_profile,
            evaluator_profile=self.config.evaluator_profile,
            evidence_digest=self.config.evidence_digest,
            memory_digest=self.memory.digest,
            prompt_mode=self.config.prompt_mode,
            component_novelty_resource_digest=self.component_novelty_resource_digest,
            operator_grounding_digest=grounding_digest,
        )

    def _ground_plan(
        self,
        plan: EditPlan,
        state: IdeaState,
    ) -> OperatorGroundingResult | None:
        if plan.operator not in {THEORY_TRANSFER_OPERATOR, MECHANISM_COMMIT_OPERATOR}:
            return None
        if self.operator_grounding_evaluator is None:
            return None
        return self.operator_grounding_evaluator.evaluate(
            plan.operator,
            state,
            prompt_mode=self.config.prompt_mode,
        )

    @staticmethod
    def _plan_digest(plan: EditPlan) -> str:
        payload = {
            "operator": plan.operator,
            "target_defects": list(plan.target_defects),
            "edits": [
                {
                    "kind": edit.kind.value,
                    "target": edit.target,
                    "replacement": edit.replacement,
                }
                for edit in plan.edits
            ],
            "rationale": plan.rationale,
            "memory_refs": list(plan.memory_refs),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _attempt_record(
        parent_node_id: int,
        plan: EditPlan,
        plan_digest: str,
        outcome,
        grounding_result: OperatorGroundingResult | None = None,
    ) -> OperatorAttemptRecord:
        return OperatorAttemptRecord(
            parent_node_id=parent_node_id,
            operator=plan.operator,
            plan_digest=plan_digest,
            outcome=outcome,
            grounding_explicit_empty=(
                grounding_result.explicit_empty if grounding_result is not None else False
            ),
            query_provenance_json=(
                grounding_result.query_provenance_json if grounding_result is not None else None
            ),
            references_provenance_json=(
                grounding_result.references_provenance_json if grounding_result is not None else None
            ),
            retrieval_provenance_json=(
                grounding_result.retrieval_provenance_json if grounding_result is not None else None
            ),
        )

    @staticmethod
    def _blocker_record(
        parent_node_id: int,
        plan: EditPlan,
        plan_digest: str,
        stage,
        error: Exception,
    ) -> RolloutBlockerRecord:
        error_type = type(error).__name__
        return RolloutBlockerRecord(
            parent_node_id=parent_node_id,
            operator=plan.operator,
            plan_digest=plan_digest,
            stage=stage,
            error_type=error_type,
            message=f"{stage} failed ({error_type})"[:160],
        )

    def _structural_profile(self, state: IdeaState) -> StructuralProfile:
        text = " ".join((state.core_contribution, state.method, *(item.description for item in state.components))).casefold()
        boundary = self.context.refinement_boundary
        scope_kind = "existing_subsystem"
        if boundary is not None:
            if boundary.allowed_component_ids:
                scope_kind = "existing_component"
            elif boundary.allowed_fields and set(boundary.allowed_fields) <= {"components"}:
                scope_kind = "existing_subsystem"
            elif boundary.allowed_edit_kinds and "REWIRE" in boundary.allowed_edit_kinds:
                scope_kind = "execution_path"
        return StructuralProfile(
            scope_kind=scope_kind,
            control_centered=any(token in text for token in ("control", "gate", "router", "feedback")),
            has_multi_path_shape=any(token in text for token in ("branch", "fallback", "route", "path")),
            training_free_like="training-free" in text or "training free" in text,
        )

    def _plan_allowed(self, plan: EditPlan) -> bool:
        boundary = self.context.refinement_boundary
        if boundary is None:
            return True
        if boundary.allowed_edit_kinds and not {
            edit.kind.value for edit in plan.edits
        }.issubset(boundary.allowed_edit_kinds):
            return False
        return not (
            boundary.allowed_fields
            and any(edit.kind.value in {"ADD_COMPONENT", "REMOVE_COMPONENT", "REPLACE_COMPONENT"} for edit in plan.edits)
            and "components" not in boundary.allowed_fields
        )

    @staticmethod
    def _backup(path: list[PathStep], value: float) -> None:
        for step in reversed(path):
            step.node.visits += 1
            step.node.value_sum += value

    def _budget_reason(self, counters: SearchCounters) -> StopReason | None:
        config = self.config
        if config.max_children is not None and counters.children >= config.max_children:
            return "child_budget"
        if config.max_evaluator_calls is not None and counters.evaluator_calls >= config.max_evaluator_calls:
            return "evaluator_budget"
        if config.max_tokens is not None and counters.tokens >= config.max_tokens:
            return "token_budget"
        if config.max_cost is not None and counters.cost >= config.max_cost:
            return "cost_budget"
        return None
