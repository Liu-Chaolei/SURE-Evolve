"""Defect-targeted, typed, bounded edit operators."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from .defects import validate_defects
from .skills import OperatorSkill, load_operator_skill_catalog
from .tastes import Taste

MIN_COMPONENTS = 1
MAX_EDITS = 6
PRODUCTION_ADAPTIVE_SUCCESS_THRESHOLD = 0.7
ScopeKind = Literal[
    "existing_component",
    "existing_subsystem",
    "execution_path",
    "broad_architecture",
]


class EditKind(str, Enum):
    ADD_COMPONENT = "ADD_COMPONENT"
    REMOVE_COMPONENT = "REMOVE_COMPONENT"
    REPLACE_COMPONENT = "REPLACE_COMPONENT"
    REWIRE = "REWIRE"
    ADD_PROTOCOL = "ADD_PROTOCOL"


@dataclass(frozen=True)
class AtomicEdit:
    kind: EditKind
    target: str
    replacement: str = ""

    def __post_init__(self) -> None:
        if not self.target.strip():
            raise ValueError("edit target cannot be empty")
        if self.kind is EditKind.REPLACE_COMPONENT and not self.replacement.strip():
            raise ValueError("replacement edit requires replacement text")


@dataclass(frozen=True)
class EditPlan:
    operator: str
    target_defects: tuple[str, ...]
    edits: tuple[AtomicEdit, ...]
    rationale: str
    approved: bool
    memory_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_defects", validate_defects(self.target_defects))
        if not self.approved:
            raise ValueError("only approved edit plans may enter search")
        if not self.target_defects:
            raise ValueError("an edit plan must target at least one defect")
        if not 1 <= len(self.edits) <= MAX_EDITS:
            raise ValueError(f"edit plans require 1..{MAX_EDITS} atomic edits")


@dataclass(frozen=True)
class Operator:
    name: str
    defects: frozenset[str]
    blueprint: tuple[AtomicEdit, ...]

    @property
    def skill(self) -> OperatorSkill:
        return OPERATOR_SKILLS[self.name]

    def plan(self, defects: tuple[str, ...], *, exploratory: bool = False) -> EditPlan:
        requested = validate_defects(defects)
        targets = tuple(defect for defect in requested if defect in self.defects)
        if not targets:
            if not exploratory:
                raise ValueError(f"operator {self.name} does not address requested defects")
            targets = requested
        return EditPlan(
            operator=self.name,
            target_defects=targets,
            edits=self.blueprint,
            rationale=f"Apply {self.name} to {', '.join(targets)}",
            approved=True,
        )


OPERATORS = (
    Operator("mechanism-commit-innovation", frozenset({"stagnant_novelty", "unclear_mechanism", "validation_gap"}), (AtomicEdit(EditKind.REPLACE_COMPONENT, "weak_internal_component", "refined_internal_component"), AtomicEdit(EditKind.ADD_PROTOCOL, "ablation"))),
    Operator("alternative-path-contrast", frozenset({"brittle_single_path", "rare_regime_failure", "weak_fallback_behavior"}), (AtomicEdit(EditKind.ADD_COMPONENT, "alternative_path_module"), AtomicEdit(EditKind.REWIRE, "alternative_path_module", "failure_regime_interface"), AtomicEdit(EditKind.ADD_PROTOCOL, "ablation,stress"))),
    Operator("surgical-modularity", frozenset({"feature_dumping", "monolithic_design", "harder_to_ablate"}), (AtomicEdit(EditKind.REPLACE_COMPONENT, "weak_block", "modular_block"), AtomicEdit(EditKind.REWIRE, "modular_block", "downstream_interface"), AtomicEdit(EditKind.ADD_PROTOCOL, "ablation"))),
    Operator("multi-scale-coordinator", frozenset({"scale_mismatch", "coordination_failure", "latency_bottleneck"}), (AtomicEdit(EditKind.ADD_COMPONENT, "scale_consistency_module"), AtomicEdit(EditKind.REWIRE, "scale_consistency_module", "scale_interface"), AtomicEdit(EditKind.ADD_PROTOCOL, "ablation"))),
    Operator("hierarchical-decomposition", frozenset({"responsibility_entanglement", "monolithic_design", "scale_mismatch"}), (AtomicEdit(EditKind.REPLACE_COMPONENT, "flat_pipeline", "hierarchical_pipeline"), AtomicEdit(EditKind.REWIRE, "hierarchical_pipeline", "execution_path"), AtomicEdit(EditKind.ADD_PROTOCOL, "ablation"))),
    Operator("feedback-closed-loop", frozenset({"silent_failure", "drift", "open_loop_fragility"}), (AtomicEdit(EditKind.ADD_COMPONENT, "feedback_monitor"), AtomicEdit(EditKind.REWIRE, "feedback_monitor", "adaptation_rule"), AtomicEdit(EditKind.ADD_PROTOCOL, "ablation,stress"))),
    Operator("theory-transfer-injection", frozenset({"stagnant_novelty", "theory_gap", "weak_generalization"}), (AtomicEdit(EditKind.ADD_COMPONENT, "theory_transfer_module"), AtomicEdit(EditKind.REWIRE, "theory_transfer_module", "core_objective"), AtomicEdit(EditKind.ADD_PROTOCOL, "ablation,stress"))),
    Operator("speculative-execution-with-repair", frozenset({"over_conservative_execution", "latency_bottleneck", "rollback_blindspot"}), (AtomicEdit(EditKind.ADD_COMPONENT, "speculative_executor"), AtomicEdit(EditKind.ADD_COMPONENT, "repair_handler"), AtomicEdit(EditKind.REWIRE, "speculative_executor", "repair_handler"), AtomicEdit(EditKind.ADD_PROTOCOL, "ablation,stress"))),
)

OPERATOR_SKILLS = load_operator_skill_catalog(tuple(operator.name for operator in OPERATORS))
_OPERATOR_BY_NAME = {operator.name: operator for operator in OPERATORS}


@dataclass(frozen=True)
class StructuralProfile:
    scope_kind: ScopeKind
    control_centered: bool = False
    has_multi_path_shape: bool = False
    training_free_like: bool = False


@dataclass(frozen=True)
class OperatorPrior:
    attempts: int = 0
    successes: int = 0
    reward_ema: float = 0.5
    prior: float = 0.5
    rule_constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.attempts < 0 or not 0 <= self.successes <= self.attempts:
            raise ValueError("operator prior counts are inconsistent")
        for field_name in ("reward_ema", "prior"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"operator prior {field_name} must be in [0, 1]")

    def update(
        self,
        reward: float,
        *,
        failure_modes: Sequence[str] = (),
        success_threshold: float = PRODUCTION_ADAPTIVE_SUCCESS_THRESHOLD,
    ) -> OperatorPrior:
        clipped_reward = max(0.0, min(1.0, float(reward)))
        attempts = self.attempts + 1
        successes = self.successes + int(clipped_reward >= max(0.0, min(1.0, success_threshold)))
        reward_ema = 0.8 * self.reward_ema + 0.2 * clipped_reward
        beta_mean = (successes + 1.0) / (attempts + 2.0)
        constraints = list(self.rule_constraints)
        for failure_mode in failure_modes:
            text = str(failure_mode).strip()
            rule = f"Avoid failure mode: {text}" if text else ""
            if rule and rule not in constraints:
                constraints.append(rule)
        return OperatorPrior(
            attempts=attempts,
            successes=successes,
            reward_ema=reward_ema,
            prior=0.55 * beta_mean + 0.45 * reward_ema,
            rule_constraints=tuple(constraints[:8]),
        )


@dataclass(frozen=True)
class OperatorCandidate:
    operator: Operator
    defect_score: float
    prior_score: float
    taste_bias: float
    structural_fit: float
    structural_reason: str
    effective_score: float
    attempts: int = 0
    exploratory: bool = False

    @property
    def skill(self) -> OperatorSkill:
        return self.operator.skill


@dataclass
class OperatorPlanner:
    priors: dict[str, OperatorPrior] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.priors) - set(_OPERATOR_BY_NAME)
        if unknown:
            raise ValueError(f"operator priors contain unknown operators: {sorted(unknown)}")

    def rank(
        self,
        defects: Sequence[str],
        *,
        profile: StructuralProfile,
        taste: Taste,
    ) -> tuple[OperatorCandidate, ...]:
        requested = set(validate_defects(tuple(defects))) or {"unexplored_gap"}
        scored: list[OperatorCandidate] = []
        for operator in OPERATORS:
            prior = self.priors.get(operator.name, OperatorPrior())
            defect_score = len(requested & operator.defects) / len(requested)
            structural_fit, reason = _structural_fit(operator.skill, profile)
            if structural_fit == 0:
                continue
            taste_bias = max(0.0, min(1.0, float(taste.skill_bias[operator.name])))
            effective_score = (
                0.60 * defect_score + 0.20 * prior.prior + 0.20 * taste_bias
            ) * structural_fit
            scored.append(
                OperatorCandidate(
                    operator=operator,
                    defect_score=defect_score,
                    prior_score=prior.prior,
                    taste_bias=taste_bias,
                    structural_fit=structural_fit,
                    structural_reason=reason,
                    effective_score=effective_score,
                    attempts=prior.attempts,
                )
            )
        return tuple(
            sorted(
                scored,
                key=lambda candidate: (
                    -candidate.effective_score,
                    -candidate.defect_score,
                    candidate.operator.name,
                ),
            )
        )

    def select(
        self,
        defects: Sequence[str],
        *,
        limit: int,
        profile: StructuralProfile,
        taste: Taste,
        rng: random.Random,
    ) -> tuple[OperatorCandidate, ...]:
        if limit < 1:
            return ()
        ranked = self.rank(defects, profile=profile, taste=taste)
        if limit == 1 or len(ranked) <= 1:
            return ranked[:limit]

        exploit_count = min(len(ranked), limit - 1)
        selected = list(ranked[:exploit_count])
        remaining = ranked[exploit_count:]
        if remaining and len(selected) < limit:
            eligible = [candidate for candidate in remaining if candidate.defect_score > 0]
            pool = eligible or list(remaining)
            if eligible:
                weights = [
                    candidate.defect_score * (1.0 + 1.0 / math.sqrt(candidate.attempts + 1.0))
                    for candidate in pool
                ]
                explored = rng.choices(pool, weights=weights, k=1)[0]
            else:
                explored = rng.choice(pool)
            selected.append(
                OperatorCandidate(
                    operator=explored.operator,
                    defect_score=explored.defect_score,
                    prior_score=explored.prior_score,
                    taste_bias=explored.taste_bias,
                    structural_fit=explored.structural_fit,
                    structural_reason=explored.structural_reason,
                    effective_score=explored.effective_score,
                    attempts=explored.attempts,
                    exploratory=True,
                )
            )
        return tuple(selected)

    def plans(
        self,
        defects: Sequence[str],
        *,
        limit: int,
        profile: StructuralProfile,
        taste: Taste,
        rng: random.Random,
    ) -> tuple[EditPlan, ...]:
        requested = tuple(defects) or ("unexplored_gap",)
        candidates = self.select(requested, limit=limit, profile=profile, taste=taste, rng=rng)
        return tuple(
            candidate.operator.plan(
                requested,
                exploratory=candidate.exploratory or candidate.defect_score == 0,
            )
            for candidate in candidates
        )

    def update_prior(
        self,
        operator_name: str,
        reward: float,
        *,
        failure_modes: Sequence[str] = (),
        success_threshold: float = PRODUCTION_ADAPTIVE_SUCCESS_THRESHOLD,
    ) -> OperatorPrior:
        if operator_name not in _OPERATOR_BY_NAME:
            raise ValueError(f"unknown operator: {operator_name}")
        updated = self.priors.get(operator_name, OperatorPrior()).update(
            reward,
            failure_modes=failure_modes,
            success_threshold=success_threshold,
        )
        self.priors[operator_name] = updated
        return updated


def _scope_fit(scope_preference: str, scope_kind: ScopeKind) -> float:
    if scope_preference == scope_kind:
        return 1.0
    if {scope_preference, scope_kind} <= {"existing_component", "existing_subsystem"}:
        return 0.85
    if {scope_preference, scope_kind} <= {"execution_path", "broad_architecture"}:
        return 0.75
    if scope_preference == "core_objective" and scope_kind in {
        "existing_subsystem",
        "execution_path",
        "broad_architecture",
    }:
        return 0.7
    return 0.0


def _structural_fit(skill: OperatorSkill, profile: StructuralProfile) -> tuple[float, str]:
    if skill.requires_control_centered_parent and not profile.control_centered:
        return 0.0, "requires_control_centered_parent"
    fit = _scope_fit(skill.scope_preference, profile.scope_kind)
    if fit == 0:
        return 0.0, "scope_mismatch"
    if skill.structural_mode == "path_branching" and profile.scope_kind == "existing_component":
        return 0.0, "path_branching_out_of_scope"
    if skill.structural_mode == "feedback_loop" and profile.training_free_like:
        return 0.0, "feedback_loop_out_of_scope"
    if skill.structural_mode == "path_branching" and not profile.has_multi_path_shape:
        return fit * 0.75, "scope_only"
    return fit, "aligned"


def approved_plans(defects: tuple[str, ...], *, limit: int) -> tuple[EditPlan, ...]:
    candidates: list[tuple[int, str, Operator]] = []
    requested = set(validate_defects(defects))
    for operator in OPERATORS:
        overlap = len(requested & operator.defects)
        if overlap:
            candidates.append((-overlap, operator.name, operator))
    return tuple(operator.plan(defects) for _, _, operator in sorted(candidates)[:limit])
