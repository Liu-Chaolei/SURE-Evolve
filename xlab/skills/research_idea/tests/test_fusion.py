from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PACKAGE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from research_idea_lib.algorithm.contracts import RefinementBoundary  # noqa: E402
from research_idea_lib.algorithm.fusion import (  # noqa: E402
    ALGORITHM_VERSION,
    CANONICAL_MODES,
    FUSION_EVALUATION_PROFILE,
    FUSION_METRIC_WEIGHTS,
    REFEREE_METRICS,
    FusionRequest,
    fuse_five_modes,
)
from research_idea_lib.research_idea_spec import IDEA_TASTE_MODES  # noqa: E402


class StaticGenerator:
    def __init__(self, output: Mapping[str, Any]) -> None:
        self.output = deepcopy(output)
        self.requests: list[dict[str, Any]] = []

    def generate(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        self.requests.append(deepcopy(dict(request)))
        return deepcopy(self.output)


class QueueGenerator:
    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = deepcopy(outputs)
        self.requests: list[dict[str, Any]] = []

    def generate(self, request: Mapping[str, Any]) -> Any:
        self.requests.append(deepcopy(dict(request)))
        return deepcopy(self.outputs.pop(0))


class QueueRepairer:
    def __init__(self, proposals: list[Mapping[str, Any] | None]) -> None:
        self.proposals = deepcopy(proposals)
        self.requests: list[dict[str, Any]] = []

    def propose_repair(self, request: Mapping[str, Any]) -> Mapping[str, Any] | None:
        self.requests.append(deepcopy(dict(request)))
        return deepcopy(self.proposals.pop(0)) if self.proposals else None


class QueueEvaluator:
    def __init__(
        self,
        scores: list[float],
        metrics: list[Mapping[str, float]] | None = None,
    ) -> None:
        self.scores = list(scores)
        self.metrics = [dict(value) for value in metrics] if metrics is not None else []
        self.candidates: list[dict[str, Any]] = []

    def evaluate(self, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        self.candidates.append(deepcopy(dict(candidate)))
        score = self.scores.pop(0)
        metrics = (
            self.metrics.pop(0)
            if self.metrics
            else {
                metric: 5.0 - score if metric in {"risk", "complexity_penalty"} else score
                for metric in REFEREE_METRICS
            }
        )
        return {
            "score": score,
            "metrics": metrics,
            "feedback": f"score={score}",
        }


class MutatingRejectEvaluator(QueueEvaluator):
    def evaluate(self, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        candidate["components"].append("evaluator mutation")  # type: ignore[attr-defined]
        return super().evaluate(candidate)


class FusionAlgorithmTest(unittest.TestCase):
    def test_rejects_missing_and_duplicate_modes(self) -> None:
        inputs = mode_inputs()[:-1]
        with self.assertRaisesRegex(ValueError, "missing modes: evidence_first"):
            run(inputs)

        duplicate = mode_inputs()
        duplicate[-1]["mode"] = CANONICAL_MODES[0]
        with self.assertRaisesRegex(ValueError, "duplicate modes: moonshot_inventor"):
            run(duplicate)

    def test_uses_frozen_canonical_order_and_keeps_coherent_provenance(self) -> None:
        inputs = list(reversed(mode_inputs()))
        generator = StaticGenerator(fusion_output())
        result = fuse_five_modes(
            FusionRequest(inputs, max_repair_steps=0),
            generator=generator,
            evaluator=QueueEvaluator([1.0]),
        )

        self.assertEqual(CANONICAL_MODES, tuple(IDEA_TASTE_MODES))
        self.assertEqual([item["mode"] for item in generator.requests[0]["mode_inputs"]], list(CANONICAL_MODES))
        self.assertEqual(result.source_modes, CANONICAL_MODES)
        self.assertEqual(result.metadata["algorithm"], ALGORITHM_VERSION)
        self.assertEqual(ALGORITHM_VERSION, "xlab.research_idea.algorithm.v2")
        self.assertEqual(FUSION_EVALUATION_PROFILE, "xlab.research_idea.fusion-referee.v2")
        self.assertEqual(result.selected_components[0]["source_mode"], "moonshot_inventor")
        self.assertEqual(result.selected_components[0]["evidence"], ["ev:moonshot_inventor"])
        self.assertEqual(set(result.evaluation.metrics), set(REFEREE_METRICS))

        incoherent = fusion_output()
        incoherent["selected_components"][0]["evidence"] = ["ev:evidence_first"]
        with self.assertRaisesRegex(ValueError, "does not belong"):
            run(mode_inputs(), generated=incoherent)

    def test_rejects_invented_component_provenance(self) -> None:
        generated = fusion_output()
        generated["idea"]["components"][0] = "invented core"
        generated["selected_components"][0]["component"] = "invented core"
        with self.assertRaisesRegex(ValueError, "does not belong to source mode"):
            run(mode_inputs(), generated=generated)

    def test_object_components_survive_remove_replace_and_rewire(self) -> None:
        inputs = mode_inputs()
        inputs[0]["idea"]["components"] = [
            {"name": "core", "description": "Original core."},
            {"name": "core-v2", "description": "Replacement core."},
        ]
        inputs[-1]["idea"]["components"] = [
            {"name": "validator", "description": "Validator."}
        ]
        generated = fusion_output()
        generated["idea"]["components"] = [
            {"name": "core", "description": "Original core."},
            {"name": "validator", "description": "Validator."},
            {"name": "bridge", "description": "Removable bridge."},
        ]
        generated["selected_components"].append(
            {
                "component": "bridge",
                "source_mode": "bridge_builder",
                "evidence": ["ev:bridge_builder"],
            }
        )
        result = run(
            inputs,
            generated=generated,
            evaluator=QueueEvaluator([1.0, 2.0]),
            repairer=QueueRepairer(
                [
                    {
                        "operations": [
                            {
                                "op": "replace",
                                "component": "core",
                                "target": "core-v2",
                                "source_mode": "moonshot_inventor",
                                "evidence": ["ev:moonshot_inventor"],
                            },
                            {"op": "rewire", "component": "core-v2", "target": "validator"},
                            {"op": "remove", "component": "bridge"},
                        ]
                    }
                ]
            ),
        )
        self.assertEqual(
            result.idea["components"],
            [
                {"name": "core-v2", "description": "Replacement core."},
                {"name": "validator", "description": "Validator."},
            ],
        )
        self.assertEqual(result.idea["rewires"][0]["source"], "core-v2")

    def test_replacement_requires_exact_source_provenance(self) -> None:
        for proposal in (
            {"operations": [{"op": "replace", "component": "core", "target": "core-v2"}]},
            replace("core", "novel tighter core"),
        ):
            with self.subTest(proposal=proposal):
                result = run(
                    mode_inputs(),
                    evaluator=QueueEvaluator([1.0]),
                    repairer=QueueRepairer([proposal]),
                )
                self.assertEqual(result.evolution[1]["status"], "rejected")
                self.assertEqual(result.idea["components"], ["core", "validator"])

    def test_metric_profile_overrides_provider_score_and_inverts_penalties(self) -> None:
        strong = {metric: 5.0 for metric in REFEREE_METRICS}
        strong["risk"] = 0.0
        strong["complexity_penalty"] = 0.0
        weak = {metric: 0.0 for metric in REFEREE_METRICS}
        weak["risk"] = 5.0
        weak["complexity_penalty"] = 5.0
        result = run(
            mode_inputs(),
            evaluator=QueueEvaluator([0.0, 5.0], [strong, weak]),
            repairer=QueueRepairer([replace("core", "core-v2")]),
            score_epsilon=0.0,
        )

        self.assertEqual(sum(FUSION_METRIC_WEIGHTS.values()), 1.0)
        self.assertEqual(result.metadata["fusion_evaluation_profile"], FUSION_EVALUATION_PROFILE)
        self.assertEqual(result.score, 5.0)
        self.assertEqual(result.evaluation.details["provider_score"], 0.0)
        self.assertEqual(result.idea["components"], ["core", "validator"])
        self.assertEqual(result.evolution[1]["status"], "rejected")

    def test_uses_audited_default_referee_weights(self) -> None:
        self.assertEqual(
            FUSION_METRIC_WEIGHTS,
            {
                "alignment_score": 0.14,
                "complexity_penalty": 0.06,
                "novelty": 0.27,
                "surprise": 0.22,
                "impact": 0.18,
                "feasibility": 0.05,
                "clarity": 0.03,
                "conciseness": 0.02,
                "risk": 0.02,
                "protocol_score": 0.01,
            },
        )

    def test_semantically_invalid_drafts_are_retried_and_fail_closed(self) -> None:
        invalid = fusion_output()
        invalid["idea"].pop("method")
        generator = QueueGenerator([invalid, fusion_output()])
        result = fuse_five_modes(
            FusionRequest(mode_inputs(), max_repair_steps=0),
            generator=generator,
            evaluator=QueueEvaluator([1.0]),
        )
        self.assertEqual(result.metadata["fusion_draft_attempts"], 2)
        feedback = generator.requests[1]["validation_feedback"]
        self.assertEqual(feedback["code"], "invalid_fusion_draft")
        self.assertIn("requires method", feedback["message"])

        exhausted = QueueGenerator([invalid, invalid])
        with self.assertRaisesRegex(ValueError, "exhausted semantic draft attempts"):
            fuse_five_modes(
                FusionRequest(mode_inputs(), max_fusion_draft_attempts=2),
                generator=exhausted,
                evaluator=QueueEvaluator([1.0]),
            )

    def test_default_fifth_semantic_draft_can_succeed(self) -> None:
        invalid = fusion_output()
        invalid["idea"].pop("method")
        generator = QueueGenerator([invalid] * 4 + [fusion_output()])
        request = FusionRequest(mode_inputs(), max_repair_steps=0)
        result = fuse_five_modes(
            request,
            generator=generator,
            evaluator=QueueEvaluator([1.0]),
        )

        self.assertEqual(request.max_fusion_draft_attempts, 5)
        self.assertEqual(result.metadata["fusion_draft_attempts"], 5)
        self.assertEqual(len(generator.requests), 5)
        self.assertIn("requires method", generator.requests[4]["validation_feedback"]["message"])

    def test_initial_fusion_rejects_tighter_role_variants(self) -> None:
        generated = fusion_output()
        variant = tighter_role_variant()
        generated["idea"]["components"][0] = {
            "name": variant["narrowed_name"],
            "description": variant["narrowed_description"],
        }
        generated["selected_components"][0].update(
            {
                "component": variant["narrowed_name"],
                "tighter_role_variant": variant,
            }
        )

        with self.assertRaisesRegex(ValueError, "does not permit tighter_role_variant"):
            run(mode_inputs(), generated=generated, max_fusion_draft_attempts=1)

    def test_tighter_role_variant_repair_materializes_narrowed_description(self) -> None:
        variant = tighter_role_variant()
        result = run(
            mode_inputs(),
            evaluator=QueueEvaluator([1.0, 2.0]),
            repairer=QueueRepairer(
                [
                    {
                        "operations": [
                            {
                                "op": "replace",
                                "component": "core",
                                "target": variant["narrowed_name"],
                                "source_mode": variant["source_mode"],
                                "evidence": list(variant["evidence_ids"]),
                                "tighter_role_variant": variant,
                            }
                        ]
                    }
                ]
            ),
        )

        self.assertEqual(
            result.idea["components"][0],
            {
                "name": variant["narrowed_name"],
                "description": variant["narrowed_description"],
            },
        )
        self.assertEqual(
            result.idea["component_explanations"][variant["narrowed_name"]],
            variant["narrowed_description"],
        )
        self.assertEqual(
            result.selected_components[0]["tighter_role_variant"],
            variant,
        )

    def test_missing_fused_hypothesis_is_accepted(self) -> None:
        generated = fusion_output()
        generated["idea"].pop("hypothesis")
        result = run(mode_inputs(), generated=generated)
        self.assertNotIn("hypothesis", result.idea)

    def test_strict_epsilon_rejects_equal_threshold(self) -> None:
        repairer = QueueRepairer([replace("core", "core-v2"), None])
        result = run(
            mode_inputs(),
            evaluator=QueueEvaluator([1.0, 1.1]),
            repairer=repairer,
            score_epsilon=0.1,
        )

        self.assertEqual(result.idea["components"], ["core", "validator"])
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.evolution[1]["status"], "rejected")
        self.assertEqual(result.evolution[1]["reason"], "insufficient_score_improvement")

    def test_rejected_candidate_or_evaluator_mutation_cannot_mutate_best(self) -> None:
        repairer = QueueRepairer([replace("core", "core-v2"), None])
        result = run(
            mode_inputs(),
            evaluator=MutatingRejectEvaluator([1.0, 0.5]),
            repairer=repairer,
        )

        self.assertEqual(result.idea["components"], ["core", "validator"])
        self.assertNotIn("evaluator mutation", result.idea["components"])
        self.assertEqual(repairer.requests[1]["best_idea"]["components"], ["core", "validator"])

    def test_prohibited_add_is_rejected_without_evaluation(self) -> None:
        repairer = QueueRepairer(
            [{"operations": [{"op": "add", "component": "new-component"}]}, None]
        )
        evaluator = QueueEvaluator([1.0])
        result = run(mode_inputs(), evaluator=evaluator, repairer=repairer)

        self.assertEqual(len(evaluator.candidates), 1)
        self.assertEqual(result.idea["components"], ["core", "validator"])
        self.assertIn("Prohibited repair operation", result.evolution[1]["reason"])

    def test_repair_steps_and_patience_are_bounded(self) -> None:
        proposals = [replace("core", "core-v2") for _ in range(8)]
        repairer = QueueRepairer(proposals)
        result = run(
            mode_inputs(),
            evaluator=QueueEvaluator([1.0, 0.9, 0.8, 0.7]),
            repairer=repairer,
            max_repair_steps=7,
            repair_patience=2,
        )
        self.assertEqual(len(repairer.requests), 2)
        self.assertEqual(result.metadata["repair_steps_attempted"], 2)

        repairer = QueueRepairer(proposals)
        result = run(
            mode_inputs(),
            evaluator=QueueEvaluator([1.0, 0.9, 0.8]),
            repairer=repairer,
            max_repair_steps=1,
            repair_patience=5,
        )
        self.assertEqual(len(repairer.requests), 1)
        self.assertEqual(result.metadata["repair_steps_attempted"], 1)

    def test_protected_components_and_typed_refinement_boundary_are_enforced(self) -> None:
        protected = QueueRepairer([replace("core", "core-v2"), None])
        result = run(
            mode_inputs(),
            evaluator=QueueEvaluator([1.0]),
            repairer=protected,
            protected_components=["core"],
        )
        self.assertIn("protected component", result.evolution[1]["reason"])
        self.assertEqual(result.idea["components"], ["core", "validator"])

        out_of_boundary = QueueRepairer([replace("core", "core-v2"), None])
        result = run(
            mode_inputs(),
            evaluator=QueueEvaluator([1.0]),
            repairer=out_of_boundary,
            refinement_boundary=RefinementBoundary(allowed_component_ids=("validator",)),
        )
        self.assertIn("outside refinement_boundary", result.evolution[1]["reason"])

        free_text_scope = QueueRepairer([replace("core", "core-v2")])
        result = run(
            mode_inputs(),
            evaluator=QueueEvaluator([1.0, 2.0]),
            repairer=free_text_scope,
            refinement_scope=["Improve only the reliability mechanism."],
        )
        self.assertEqual(result.idea["components"], ["core-v2", "validator"])
        self.assertEqual(
            free_text_scope.requests[0]["refinement_scope"],
            ["Improve only the reliability mechanism."],
        )
        self.assertIsNone(free_text_scope.requests[0]["refinement_boundary"])

        below_minimum = QueueRepairer([{"operations": [{"op": "remove", "component": "core"}]}, None])
        result = run(
            mode_inputs(),
            evaluator=QueueEvaluator([1.0]),
            repairer=below_minimum,
            minimum_components=2,
        )
        self.assertIn("minimum_components", result.evolution[1]["reason"])


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
            "evidence": [f"ev:{mode}"],
        }
        for mode in CANONICAL_MODES
    ]


def fusion_output() -> dict[str, Any]:
    return {
        "idea": {
            "title": "Fused idea",
            "abstract": "A fused abstract.",
            "core_contribution": "A fused mechanism.",
            "hypothesis": "The fused mechanism improves evidence-grounded ideation.",
            "method": "Test the fused mechanism.",
            "risks": "The mechanism may fail.",
            "components": ["core", "validator"],
            "component_explanations": {"core": "mechanism", "validator": "test"},
        },
        "selected_components": [
            {
                "component": "core",
                "source_mode": "moonshot_inventor",
                "evidence": ["ev:moonshot_inventor"],
            },
            {
                "component": "validator",
                "source_mode": "evidence_first",
                "evidence": ["ev:evidence_first"],
            },
        ],
        "rejected_components": [
            {
                "component": "bridge",
                "source_mode": "bridge_builder",
                "evidence": ["ev:bridge_builder"],
                "reason": "conflicts with core",
            }
        ],
        "conflict_resolutions": [
            {
                "conflict": "core versus bridge",
                "resolution": "keep the directly testable core",
                "source_modes": ["moonshot_inventor", "bridge_builder"],
                "evidence": ["ev:moonshot_inventor", "ev:bridge_builder"],
            }
        ],
    }


def tighter_role_variant() -> dict[str, Any]:
    return {
        "source_mode": "moonshot_inventor",
        "source_component_name": "core",
        "source_component_description": "Description of core.",
        "narrowed_name": "tighter core",
        "narrowed_description": "A narrower implementation of the original core role.",
        "role": "core_mechanism",
        "evidence_ids": ["ev:moonshot_inventor"],
        "narrowing_rationale": "Restricts the original core to one testable mechanism.",
    }


def replace(component: str, target: str) -> dict[str, Any]:
    return {
        "operations": [
            {
                "op": "replace",
                "component": component,
                "target": target,
                "source_mode": "moonshot_inventor",
                "evidence": ["ev:moonshot_inventor"],
            }
        ]
    }


def run(
    inputs: list[dict[str, Any]],
    *,
    generated: Mapping[str, Any] | None = None,
    evaluator: QueueEvaluator | None = None,
    repairer: QueueRepairer | None = None,
    **options: Any,
):
    return fuse_five_modes(
        FusionRequest(inputs, **options),
        generator=StaticGenerator(generated or fusion_output()),
        evaluator=evaluator or QueueEvaluator([1.0]),
        repair_generator=repairer,
    )


if __name__ == "__main__":
    unittest.main()
