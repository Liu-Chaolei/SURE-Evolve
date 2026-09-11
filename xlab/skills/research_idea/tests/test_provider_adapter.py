from __future__ import annotations

import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PACKAGE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from research_idea_lib.algorithm.contracts import (  # noqa: E402
    ALGORITHM_ID,
    GenerationRequest,
    IdeaComponent,
    IdeaProviderContext,
    IdeaState,
    RankedEvidence,
)
from research_idea_lib.algorithm.evaluation import METRICS  # noqa: E402
from research_idea_lib.algorithm.fusion import (  # noqa: E402
    ALGORITHM_VERSION,
    CANONICAL_MODES,
    REFEREE_METRICS,
    FusionRequest,
    fuse_five_modes,
)
from research_idea_lib.algorithm.operators import approved_plans  # noqa: E402
from research_idea_lib.algorithm.provider_adapter import (  # noqa: E402
    AdapterOutputError,
    FUSION_GENERATE_OPERATION,
    FUSION_REFEREE_OPERATION,
    FUSION_REPAIR_OPERATION,
    IDEA_DIAGNOSTIC_OPERATION,
    IDEA_EVALUATE_OPERATION,
    IDEA_GENERATE_OPERATION,
    ProviderAdapter,
)
from research_idea_lib.algorithm.search import MCTSEngine, SearchConfig  # noqa: E402
from research_idea_lib.providers import (  # noqa: E402
    DeterministicFakeProvider,
    FakeFixture,
    ProviderUsage,
    structured_input_digest,
)


def seed_idea() -> IdeaState:
    return IdeaState(
        title="Seed idea",
        abstract="A concrete seed abstract.",
        core_contribution="A controlled mechanism.",
        method="Apply the mechanism under intervention.",
        risks="Distribution shift.",
        components=(IdeaComponent("core", "The core mechanism."),),
        tags=("agents",),
        root_domains=("computer science",),
    )


def metric_payload(*, score: float | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {name: 3 for name in METRICS}
    payload.update(
        {
            "confidence": 0.8,
            "detected_defects": ["stagnant_novelty"],
            "feedback": "Commit to a clearer mechanism.",
        }
    )
    if score is not None:
        payload["score"] = score
    return payload


def generated_idea() -> dict[str, Any]:
    return {
        "title": "Generated idea",
        "abstract": "A concrete generated abstract.",
        "core_contribution": "A refined controlled mechanism.",
        "method": "Apply the bounded edit and test it.",
        "risks": "The intervention can overfit.",
        "components": [
            {"name": "refined core", "description": "A specific refined mechanism."}
        ],
        "component_mapping": {
            "weak_internal_component": "core",
            "refined_internal_component": "refined core",
        },
        "component_role_explanations": {
            "refined core": "A specific refined mechanism."
        },
        "tags": ["agents"],
        "root_domains": ["computer science"],
    }


def mode_inputs() -> list[dict[str, Any]]:
    components = {
        "moonshot_inventor": ["core", "core-v2"],
        "bridge_builder": ["bridge"],
        "steady_engineer": ["steady component"],
        "ambitious_realist": ["ambitious component"],
        "evidence_first": ["validator"],
    }
    return [
        {
            "mode": mode,
            "idea": {
                "title": mode,
                "components": components[mode],
                "component_explanations": {
                    name: f"Description of {name}." for name in components[mode]
                },
            },
            "evidence": [f"evidence:{mode}"],
        }
        for mode in CANONICAL_MODES
    ]


def fused_output() -> dict[str, Any]:
    return {
        "idea": {
            "title": "Fused idea",
            "abstract": "A fused abstract.",
            "core_contribution": "A fused mechanism.",
            "hypothesis": "The fused mechanism improves evidence-grounded ideation.",
            "method": "Test the fused mechanism.",
            "risks": "The mechanism may fail.",
            "components": ["core", "validator"],
            "tags": ["agents"],
            "root_domains": ["computer science"],
        },
        "selected_components": [
            {
                "component": "core",
                "source_mode": "moonshot_inventor",
                "evidence": ["evidence:moonshot_inventor"],
            },
            {
                "component": "validator",
                "source_mode": "evidence_first",
                "evidence": ["evidence:evidence_first"],
            },
        ],
        "rejected_components": [
            {
                "component": "bridge",
                "source_mode": "bridge_builder",
                "evidence": ["evidence:bridge_builder"],
            }
        ],
        "conflict_resolutions": [
            {
                "resolution": "Keep the directly testable core.",
                "source_modes": ["moonshot_inventor", "bridge_builder"],
                "evidence": [
                    "evidence:moonshot_inventor",
                    "evidence:bridge_builder",
                ],
            }
        ],
    }


def provider_tighter_role_variant() -> dict[str, Any]:
    return {
        "source_mode": "moonshot_inventor",
        "source_component_name": "core",
        "source_component_description": "Description of core.",
        "narrowed_name": "tighter core",
        "narrowed_description": "A narrower implementation of the original core role.",
        "role": "core_mechanism",
        "evidence_ids": ["evidence:moonshot_inventor"],
        "narrowing_rationale": "Restricts the original core to one testable mechanism.",
    }


def fixture(payload: dict[str, Any]) -> FakeFixture:
    return FakeFixture(
        text=json.dumps(payload),
        json_value=payload,
        usage=ProviderUsage(input_tokens=7, output_tokens=3, total_tokens=10),
    )


class ProviderAdapterTests(unittest.TestCase):
    def test_search_adapter_evaluates_and_generates_with_stable_structured_operations(self) -> None:
        root = seed_idea()
        plan = approved_plans(("stagnant_novelty",), limit=1)[0]
        generation_request = GenerationRequest(
            "moonshot_inventor",
            root,
            plan,
            19,
            ("ablation evidence",),
            IdeaProviderContext(memory_hints=("ablation evidence",)),
        )
        evaluation_input = {
            "algorithm": ALGORITHM_ID,
            "idea_taste_mode": "moonshot_inventor",
            "prompt_mode": "conceptual_surprise",
            "diagnostic": True,
            "idea": root.to_payload(),
            "grounding": {
                "evidence": [],
                "mature_idea": None,
                "refinement_scope": [],
                "memory_hints": [],
            },
        }
        plan_input = {
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
            "approved": True,
            "memory_refs": list(plan.memory_refs),
        }
        generation_input = {
            "algorithm": ALGORITHM_ID,
            "idea_taste_mode": generation_request.idea_taste_mode,
            "prompt_mode": generation_request.prompt_mode,
            "parent": root.to_payload(),
            "plan": plan_input,
            "seed": generation_request.seed,
            "memory_hints": list(generation_request.memory_hints),
            "grounding": {
                "evidence": [],
                "mature_idea": None,
                "refinement_scope": [],
                "memory_hints": list(generation_request.memory_hints),
            },
            "operator_grounding": None,
        }
        provider = DeterministicFakeProvider(
            {
                (
                    IDEA_DIAGNOSTIC_OPERATION,
                    structured_input_digest(evaluation_input),
                ): fixture(metric_payload()),
                (
                    IDEA_GENERATE_OPERATION,
                    structured_input_digest(generation_input),
                ): fixture(generated_idea()),
            }
        )
        adapter = ProviderAdapter(provider, model="fixture-model")

        evaluated = adapter.evaluate(root, idea_taste_mode="moonshot_inventor", diagnostic=True)
        generated = adapter.generate(generation_request)

        self.assertEqual(tuple(evaluated.metrics), METRICS)
        self.assertEqual(evaluated.detected_defects, ("stagnant_novelty",))
        self.assertEqual(generated.state.title, "Generated idea")
        self.assertEqual(generated.state.components[0].name, "refined core")
        self.assertEqual(
            [trace.operation for trace in adapter.traces],
            [IDEA_DIAGNOSTIC_OPERATION, IDEA_GENERATE_OPERATION],
        )
        self.assertEqual(adapter.usage.provider_calls, 2)
        self.assertEqual(adapter.usage.attempts, 2)
        self.assertEqual(adapter.usage.total_tokens, 20)
        self.assertEqual(evaluated.usage.evaluator_calls, 1)
        self.assertEqual(generated.usage.generation_calls, 1)

    def test_mcts_metrics_require_integers_but_referee_metrics_allow_numbers(self) -> None:
        root = seed_idea()
        evaluation_input = {
            "algorithm": ALGORITHM_ID,
            "idea_taste_mode": "steady_engineer",
            "prompt_mode": "conceptual_surprise",
            "diagnostic": True,
            "idea": root.to_payload(),
            "grounding": {
                "evidence": [],
                "mature_idea": None,
                "refinement_scope": [],
                "memory_hints": [],
            },
        }
        mcts_payload = metric_payload()
        mcts_payload["novelty"] = 3.5
        referee_candidate = fused_output()["idea"]
        referee_input = {"algorithm": ALGORITHM_ID, "candidate": referee_candidate}
        referee_payload = metric_payload(score=3.5)
        referee_payload["novelty"] = 3.5
        adapter = ProviderAdapter(
            DeterministicFakeProvider(
                {
                    (IDEA_DIAGNOSTIC_OPERATION, structured_input_digest(evaluation_input)): fixture(mcts_payload),
                    (FUSION_REFEREE_OPERATION, structured_input_digest(referee_input)): fixture(referee_payload),
                }
            ),
            model="fixture-model",
        )

        with self.assertRaisesRegex(AdapterOutputError, "MCTS metrics must be integers"):
            adapter.evaluate(root, idea_taste_mode="steady_engineer", diagnostic=True)
        referee = adapter.evaluate(referee_candidate)
        self.assertEqual(referee["metrics"]["novelty"], 3.5)

    def test_adapter_runs_native_mcts_evaluate_and_generate(self) -> None:
        root = seed_idea()
        plan = approved_plans(("stagnant_novelty",), limit=1)[0]
        diagnostic_input = {
            "algorithm": ALGORITHM_ID,
            "idea_taste_mode": "moonshot_inventor",
            "prompt_mode": "conceptual_surprise",
            "diagnostic": True,
            "idea": root.to_payload(),
            "grounding": {
                "evidence": [],
                "mature_idea": None,
                "refinement_scope": [],
                "memory_hints": [],
            },
        }
        generation_input = {
            "algorithm": ALGORITHM_ID,
            "idea_taste_mode": "moonshot_inventor",
            "prompt_mode": "conceptual_surprise",
            "parent": root.to_payload(),
            "plan": {
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
                "approved": True,
                "memory_refs": [],
            },
            "seed": MCTSEngine.mode_seed("adapter", "moonshot_inventor"),
            "memory_hints": [],
            "grounding": {
                "evidence": [],
                "mature_idea": None,
                "refinement_scope": [],
                "memory_hints": [],
            },
            "operator_grounding": None,
        }
        # Plan ranking consumes one seeded exploration tie-break before child generation.
        import random

        rng = random.Random(MCTSEngine.mode_seed("adapter", "moonshot_inventor"))
        generation_input["seed"] = rng.getrandbits(64)
        child = generated_idea()
        child_state = IdeaState(
            title=child["title"],
            abstract=child["abstract"],
            core_contribution=child["core_contribution"],
            method=child["method"],
            risks=child["risks"],
            components=(IdeaComponent("refined core", "A specific refined mechanism."),),
            tags=("agents",),
            root_domains=("computer science",),
        )
        rollout_input = {
            "algorithm": ALGORITHM_ID,
            "idea_taste_mode": "moonshot_inventor",
            "prompt_mode": "conceptual_surprise",
            "diagnostic": False,
            "idea": child_state.to_payload(),
            "grounding": {
                "evidence": [],
                "mature_idea": None,
                "refinement_scope": [],
                "memory_hints": [],
            },
        }
        provider = DeterministicFakeProvider(
            {
                (IDEA_DIAGNOSTIC_OPERATION, structured_input_digest(diagnostic_input)): fixture(metric_payload()),
                (IDEA_GENERATE_OPERATION, structured_input_digest(generation_input)): fixture(child),
                (IDEA_EVALUATE_OPERATION, structured_input_digest(rollout_input)): fixture(metric_payload()),
            }
        )
        result = MCTSEngine(
            ProviderAdapter(provider, model="fixture-model"),
            SearchConfig(max_iterations=1, branching_factor=1, seed="adapter"),
        ).search(root, "moonshot_inventor")

        self.assertEqual(result.counters.generation_calls, 1)
        self.assertEqual(result.counters.root_diagnostics, 1)
        self.assertEqual(result.counters.evaluator_calls, 1)
        self.assertEqual(len(result.nodes), 2)

    def test_every_idea_operation_receives_ranked_evidence_mature_scope_and_memory(self) -> None:
        root = seed_idea()
        plan = approved_plans(("stagnant_novelty",), limit=1)[0]
        evidence = RankedEvidence.from_payload(
            {
                "evidence_id": "evidence:ranked",
                "kind": "survey_gap",
                "text": "A ranked survey finding.",
                "provenance": {"source": "survey", "rank": 1},
                "paper_ids": ["p1"],
            },
            rank=1,
        )
        context = IdeaProviderContext(
            evidence=(evidence,),
            mature_idea=root,
            refinement_scope=("core",),
            memory_hints=("controller/gate: removal_hurt; confidence=0.900",),
        )
        grounding = context.to_payload()
        diagnostic_input = {
            "algorithm": ALGORITHM_ID,
            "idea_taste_mode": "moonshot_inventor",
            "prompt_mode": "conceptual_surprise",
            "diagnostic": True,
            "idea": root.to_payload(),
            "grounding": grounding,
        }
        generation_request = GenerationRequest(
            "moonshot_inventor", root, plan, 7, context.memory_hints, context
        )
        generation_input = {
            "algorithm": ALGORITHM_ID,
            "idea_taste_mode": "moonshot_inventor",
            "prompt_mode": "conceptual_surprise",
            "parent": root.to_payload(),
            "plan": {
                "operator": plan.operator,
                "target_defects": list(plan.target_defects),
                "edits": [
                    {"kind": edit.kind.value, "target": edit.target, "replacement": edit.replacement}
                    for edit in plan.edits
                ],
                "rationale": plan.rationale,
                "approved": True,
                "memory_refs": [],
            },
            "seed": 7,
            "memory_hints": list(context.memory_hints),
            "grounding": grounding,
            "operator_grounding": None,
        }
        generated = generated_idea()
        child = IdeaState(
            title=generated["title"],
            abstract=generated["abstract"],
            core_contribution=generated["core_contribution"],
            method=generated["method"],
            risks=generated["risks"],
            components=(IdeaComponent("refined core", "A specific refined mechanism."),),
            tags=root.tags,
            root_domains=root.root_domains,
        )
        rollout_input = {
            "algorithm": ALGORITHM_ID,
            "idea_taste_mode": "moonshot_inventor",
            "prompt_mode": "conceptual_surprise",
            "diagnostic": False,
            "idea": child.to_payload(),
            "grounding": grounding,
        }
        provider = DeterministicFakeProvider(
            {
                (IDEA_DIAGNOSTIC_OPERATION, structured_input_digest(diagnostic_input)): fixture(metric_payload()),
                (IDEA_GENERATE_OPERATION, structured_input_digest(generation_input)): fixture(generated),
                (IDEA_EVALUATE_OPERATION, structured_input_digest(rollout_input)): fixture(metric_payload()),
            }
        )
        adapter = ProviderAdapter(provider, model="fixture-model")
        adapter.evaluate(root, idea_taste_mode="moonshot_inventor", diagnostic=True, context=context)
        adapter.generate(generation_request)
        adapter.evaluate(child, idea_taste_mode="moonshot_inventor", diagnostic=False, context=context)
        self.assertEqual(
            [trace.input_digest for trace in adapter.traces],
            [
                structured_input_digest(diagnostic_input),
                structured_input_digest(generation_input),
                structured_input_digest(rollout_input),
            ],
        )

    def test_direct_state_generation_rejects_unapproved_component_diff(self) -> None:
        root = seed_idea()
        plan = approved_plans(("stagnant_novelty",), limit=1)[0]
        request = GenerationRequest("steady_engineer", root, plan, 5)
        structured = {
            "algorithm": ALGORITHM_ID,
            "idea_taste_mode": "steady_engineer",
            "prompt_mode": "conceptual_surprise",
            "parent": root.to_payload(),
            "plan": {
                "operator": plan.operator,
                "target_defects": list(plan.target_defects),
                "edits": [
                    {"kind": edit.kind.value, "target": edit.target, "replacement": edit.replacement}
                    for edit in plan.edits
                ],
                "rationale": plan.rationale,
                "approved": True,
                "memory_refs": [],
            },
            "seed": 5,
            "memory_hints": [],
            "grounding": IdeaProviderContext().to_payload(),
            "operator_grounding": None,
        }
        adversarial = generated_idea()
        adversarial["components"].append(
            {"name": "unapproved extra", "description": "An unapproved structural addition."}
        )
        adapter = ProviderAdapter(
            DeterministicFakeProvider(
                {(IDEA_GENERATE_OPERATION, structured_input_digest(structured)): fixture(adversarial)}
            ),
            model="fixture-model",
        )
        with self.assertRaisesRegex(AdapterOutputError, "exact approved EditPlan"):
            adapter.generate(request)

    def test_fusion_adapter_generates_referees_and_bounded_repairs(self) -> None:
        inputs = mode_inputs()
        generation_input = {
            "algorithm": ALGORITHM_VERSION,
            "topic": "agent reliability",
            "context": {},
            "source_modes": list(CANONICAL_MODES),
            "mode_inputs": inputs,
            "draft_attempt": 1,
            "max_draft_attempts": 5,
        }
        candidate = fused_output()["idea"]
        referee_input = {"algorithm": ALGORITHM_ID, "candidate": candidate}
        initial_scalar = 2.9200000000000004
        repaired_scalar = 3.3400000000000003
        repair_input = {
            "algorithm": ALGORITHM_VERSION,
            "step": 1,
            "best_idea": candidate,
            "best_evaluation": {
                "score": initial_scalar,
                "metrics": {name: 3.0 for name in REFEREE_METRICS},
                "details": {
                    "confidence": 0.8,
                    "detected_defects": ["stagnant_novelty"],
                    "feedback": "Commit to a clearer mechanism.",
                    "fusion_evaluation_profile": "xlab.research_idea.fusion-referee.v2",
                    "provider_score": 3.0,
                },
            },
            "selected_components": fused_output()["selected_components"],
            "rejected_components": fused_output()["rejected_components"],
            "conflict_resolutions": fused_output()["conflict_resolutions"],
            "source_modes": list(CANONICAL_MODES),
            "mode_inputs": inputs,
            "minimum_components": 1,
            "refinement_scope": [],
            "refinement_boundary": None,
            "protected_components": [],
            "allowed_operations": ["remove", "replace", "rewire"],
            "score_epsilon": 0.02,
            "evolution": [
                {
                    "step": 0,
                    "phase": "fusion",
                    "status": "accepted",
                    "score_after": initial_scalar,
                    "evaluation": {
                        "score": initial_scalar,
                        "metrics": {name: 3.0 for name in REFEREE_METRICS},
                        "details": {
                            "confidence": 0.8,
                            "detected_defects": ["stagnant_novelty"],
                            "feedback": "Commit to a clearer mechanism.",
                            "fusion_evaluation_profile": "xlab.research_idea.fusion-referee.v2",
                            "provider_score": 3.0,
                        },
                    },
                    "component_count": 2,
                }
            ],
        }
        repaired_candidate = {**candidate, "components": ["core-v2", "validator"]}
        repaired_referee_input = {"algorithm": ALGORITHM_ID, "candidate": repaired_candidate}
        repair_payload = {
            "stop": False,
            "component_edits": [
                {
                    "op": "REPLACE_COMPONENT",
                    "component": "core-v2",
                    "target": "core",
                    "source_mode": "moonshot_inventor",
                    "evidence": ["evidence:moonshot_inventor"],
                }
            ],
        }
        repaired_score = metric_payload(score=3.5)
        repaired_score.update({name: 3.5 for name in REFEREE_METRICS})
        initial_score = metric_payload(score=3.0)
        provider = DeterministicFakeProvider(
            {
                (FUSION_GENERATE_OPERATION, structured_input_digest(generation_input)): fixture(fused_output()),
                (FUSION_REFEREE_OPERATION, structured_input_digest(referee_input)): fixture(initial_score),
                (FUSION_REPAIR_OPERATION, structured_input_digest(repair_input)): fixture(repair_payload),
                (FUSION_REFEREE_OPERATION, structured_input_digest(repaired_referee_input)): fixture(repaired_score),
            }
        )
        adapter = ProviderAdapter(provider, model="fixture-model")

        result = fuse_five_modes(
            FusionRequest(inputs, topic="agent reliability", max_repair_steps=1),
            generator=adapter,
            evaluator=adapter,
            repair_generator=adapter,
        )

        self.assertEqual(result.idea["components"], ["core-v2", "validator"])
        self.assertEqual(result.idea["hypothesis"], candidate["hypothesis"])
        self.assertAlmostEqual(result.score, repaired_scalar)
        self.assertEqual(
            [trace.operation for trace in adapter.traces],
            [
                FUSION_GENERATE_OPERATION,
                FUSION_REFEREE_OPERATION,
                FUSION_REPAIR_OPERATION,
                FUSION_REFEREE_OPERATION,
            ],
        )
        self.assertLessEqual(len(result.evolution), 2)

    def test_fusion_adapter_rejects_typed_tighter_role_variants(self) -> None:
        inputs = mode_inputs()
        fusion_request = {
            "algorithm": ALGORITHM_VERSION,
            "topic": "",
            "context": {},
            "source_modes": list(CANONICAL_MODES),
            "mode_inputs": inputs,
        }
        variant = provider_tighter_role_variant()
        invalid = fused_output()
        invalid["idea"]["components"][0] = {
            "name": variant["narrowed_name"],
            "description": variant["narrowed_description"],
        }
        invalid["selected_components"][0].update(
            {
                "component": variant["narrowed_name"],
                "tighter_role_variant": variant,
            }
        )
        adapter = ProviderAdapter(
            DeterministicFakeProvider(
                {(FUSION_GENERATE_OPERATION, structured_input_digest(fusion_request)): fixture(invalid)}
            ),
            model="fixture-model",
        )

        with self.assertRaisesRegex(AdapterOutputError, "does not permit tighter_role_variant"):
            adapter.generate(fusion_request)

    def test_fusion_adapter_validates_typed_replacement_variants(self) -> None:
        variant = provider_tighter_role_variant()
        repair_request = {
            "allowed_operations": ["remove", "replace", "rewire"],
            "mode_inputs": mode_inputs(),
        }
        payload = {
            "stop": False,
            "component_edits": [
                {
                    "op": "REPLACE_COMPONENT",
                    "component": variant["narrowed_name"],
                    "target": "core",
                    "source_mode": variant["source_mode"],
                    "evidence": list(variant["evidence_ids"]),
                    "tighter_role_variant": variant,
                }
            ],
        }
        adapter = ProviderAdapter(
            DeterministicFakeProvider(
                {(FUSION_REPAIR_OPERATION, structured_input_digest(repair_request)): fixture(payload)}
            ),
            model="fixture-model",
        )
        result = adapter.propose_repair(repair_request)
        self.assertIsNotNone(result)
        operation = result["operations"][0]  # type: ignore[index]
        self.assertEqual(operation["component"], "core")
        self.assertEqual(operation["target"], variant["narrowed_name"])
        self.assertEqual(operation["tighter_role_variant"], variant)

        forged = deepcopy(payload)
        forged["component_edits"][0]["tighter_role_variant"][
            "source_component_description"
        ] = "forged description"
        forged_adapter = ProviderAdapter(
            DeterministicFakeProvider(
                {(FUSION_REPAIR_OPERATION, structured_input_digest(repair_request)): fixture(forged)}
            ),
            model="fixture-model",
        )
        with self.assertRaisesRegex(AdapterOutputError, "does not match"):
            forged_adapter.propose_repair(repair_request)

    def test_fusion_adapter_rejects_incomplete_and_forged_sources(self) -> None:
        fusion_request = {
            "algorithm": ALGORITHM_VERSION,
            "topic": "",
            "context": {},
            "source_modes": list(CANONICAL_MODES),
            "mode_inputs": mode_inputs(),
        }
        incomplete = fused_output()
        incomplete["idea"].pop("method")
        adapter = ProviderAdapter(
            DeterministicFakeProvider(
                {(FUSION_GENERATE_OPERATION, structured_input_digest(fusion_request)): fixture(incomplete)}
            ),
            model="fixture-model",
        )
        with self.assertRaisesRegex(AdapterOutputError, "fused idea method"):
            adapter.generate(fusion_request)

        missing_hypothesis = fused_output()
        missing_hypothesis["idea"].pop("hypothesis")
        adapter = ProviderAdapter(
            DeterministicFakeProvider(
                {
                    (FUSION_GENERATE_OPERATION, structured_input_digest(fusion_request)): fixture(
                        missing_hypothesis
                    )
                }
            ),
            model="fixture-model",
        )
        with self.assertRaisesRegex(AdapterOutputError, "fused idea hypothesis"):
            adapter.generate(fusion_request)

        forged = fused_output()
        forged["selected_components"][0]["component"] = "invented core"
        forged["idea"]["components"][0] = "invented core"
        adapter = ProviderAdapter(
            DeterministicFakeProvider(
                {(FUSION_GENERATE_OPERATION, structured_input_digest(fusion_request)): fixture(forged)}
            ),
            model="fixture-model",
        )
        with self.assertRaisesRegex(AdapterOutputError, "does not belong to source mode"):
            adapter.generate(fusion_request)

    def test_malformed_and_placeholder_outputs_fail_closed(self) -> None:
        root = seed_idea()
        malformed_metrics = metric_payload()
        malformed_metrics.pop("novelty")
        evaluation_input = {
            "algorithm": ALGORITHM_ID,
            "idea_taste_mode": "steady_engineer",
            "prompt_mode": "conceptual_surprise",
            "diagnostic": True,
            "idea": root.to_payload(),
            "grounding": {
                "evidence": [],
                "mature_idea": None,
                "refinement_scope": [],
                "memory_hints": [],
            },
        }
        adapter = ProviderAdapter(
            DeterministicFakeProvider(
                {
                    (IDEA_DIAGNOSTIC_OPERATION, structured_input_digest(evaluation_input)): fixture(malformed_metrics)
                }
            ),
            model="fixture-model",
        )
        with self.assertRaisesRegex(AdapterOutputError, "novelty"):
            adapter.evaluate(root, idea_taste_mode="steady_engineer", diagnostic=True)

        plan = approved_plans(("stagnant_novelty",), limit=1)[0]
        generation_request = GenerationRequest("steady_engineer", root, plan, 3)
        generation_input = {
            "algorithm": ALGORITHM_ID,
            "idea_taste_mode": "steady_engineer",
            "prompt_mode": "conceptual_surprise",
            "parent": root.to_payload(),
            "plan": {
                "operator": plan.operator,
                "target_defects": list(plan.target_defects),
                "edits": [
                    {"kind": edit.kind.value, "target": edit.target, "replacement": edit.replacement}
                    for edit in plan.edits
                ],
                "rationale": plan.rationale,
                "approved": True,
                "memory_refs": [],
            },
            "seed": 3,
            "memory_hints": [],
            "grounding": {
                "evidence": [],
                "mature_idea": None,
                "refinement_scope": [],
                "memory_hints": [],
            },
            "operator_grounding": None,
        }
        bad_generation = generated_idea()
        bad_generation["title"] = "TODO"
        adapter = ProviderAdapter(
            DeterministicFakeProvider(
                {(IDEA_GENERATE_OPERATION, structured_input_digest(generation_input)): fixture(bad_generation)}
            ),
            model="fixture-model",
        )
        with self.assertRaisesRegex(AdapterOutputError, "placeholder"):
            adapter.generate(generation_request)

        fusion_request = {
            "algorithm": ALGORITHM_VERSION,
            "topic": "",
            "context": {},
            "source_modes": list(CANONICAL_MODES),
            "mode_inputs": mode_inputs(),
        }
        bad_fusion = fused_output()
        bad_fusion["selected_components"] = []
        adapter = ProviderAdapter(
            DeterministicFakeProvider(
                {(FUSION_GENERATE_OPERATION, structured_input_digest(fusion_request)): fixture(bad_fusion)}
            ),
            model="fixture-model",
        )
        with self.assertRaisesRegex(AdapterOutputError, "exactly cover"):
            adapter.generate(fusion_request)

        repair_request = {"allowed_operations": ["remove", "replace", "rewire"]}
        bad_repair = {
            "stop": False,
            "operations": [{"op": "add", "component": "extra"}],
        }
        adapter = ProviderAdapter(
            DeterministicFakeProvider(
                {(FUSION_REPAIR_OPERATION, structured_input_digest(repair_request)): fixture(bad_repair)}
            ),
            model="fixture-model",
        )
        with self.assertRaisesRegex(AdapterOutputError, "prohibited"):
            adapter.propose_repair(repair_request)


if __name__ == "__main__":
    unittest.main()
