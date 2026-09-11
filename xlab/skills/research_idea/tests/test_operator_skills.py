from __future__ import annotations

import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PACKAGE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from research_idea_lib.algorithm.operators import (  # noqa: E402
    OPERATORS,
    OPERATOR_SKILLS,
    OperatorPlanner,
    StructuralProfile,
)
from research_idea_lib.algorithm.skills import (  # noqa: E402
    OperatorSkillValidationError,
    load_operator_skill_catalog,
)
from research_idea_lib.algorithm.tastes import IDEA_TASTE_MODES, get_taste  # noqa: E402

EXPECTED_OPERATORS = {
    "alternative-path-contrast",
    "feedback-closed-loop",
    "hierarchical-decomposition",
    "mechanism-commit-innovation",
    "multi-scale-coordinator",
    "speculative-execution-with-repair",
    "surgical-modularity",
    "theory-transfer-injection",
}
RESOURCE_ROOT = SCRIPTS_DIR / "research_idea_lib" / "algorithm" / "resources"
EXPECTED_CATALOG_DIGEST = "31a52b03c28dca7916856d90dfbd4e6519b8feb1875d6b2b42a45edd78f25ad0"


class OperatorSkillCatalogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="xlab-operator-skills-test-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def copy_resources(self) -> Path:
        target = self.tmp / "resources"
        shutil.copytree(RESOURCE_ROOT, target)
        return target

    def load(self, root: Path):
        return load_operator_skill_catalog(
            tuple(operator.name for operator in OPERATORS),
            resource_root=root,
        )

    def test_loads_all_eight_package_native_skills_immutably(self) -> None:
        self.assertEqual({operator.name for operator in OPERATORS}, EXPECTED_OPERATORS)
        self.assertEqual(set(OPERATOR_SKILLS), EXPECTED_OPERATORS)
        self.assertEqual(tuple(OPERATOR_SKILLS), tuple(sorted(EXPECTED_OPERATORS)))
        self.assertEqual(OPERATOR_SKILLS.content_digest, EXPECTED_CATALOG_DIGEST)

        for operator in OPERATORS:
            skill = operator.skill
            self.assertIs(skill, OPERATOR_SKILLS[operator.name])
            self.assertEqual(skill.name, operator.name)
            self.assertTrue(skill.description)
            self.assertTrue(skill.instructions)
            self.assertTrue(skill.references)
            self.assertTrue(skill.structural_mode)
            self.assertTrue(skill.scope_preference)
            self.assertIsInstance(skill.requires_control_centered_parent, bool)
            self.assertTrue(all(reference.content for reference in skill.references))
            self.assertRegex(skill.content_digest, r"^[0-9a-f]{64}$")
            self.assertNotIn("xcientist", skill.instructions.lower())

        with self.assertRaises(TypeError):
            OPERATOR_SKILLS["new"] = OPERATOR_SKILLS[next(iter(OPERATOR_SKILLS))]  # type: ignore[index]
        with self.assertRaises(TypeError):
            OPERATOR_SKILLS["alternative-path-contrast"].template["description"] = "changed"  # type: ignore[index]

    def test_digest_is_stable_when_resources_are_relocated(self) -> None:
        copied = self.copy_resources()
        relocated = self.load(copied)
        self.assertEqual(relocated.content_digest, OPERATOR_SKILLS.content_digest)
        self.assertEqual(
            {name: skill.content_digest for name, skill in relocated.items()},
            {name: skill.content_digest for name, skill in OPERATOR_SKILLS.items()},
        )

    def test_missing_operator_resource_is_rejected(self) -> None:
        copied = self.copy_resources()
        shutil.rmtree(copied / "edit_operator_skills" / "feedback-closed-loop")
        with self.assertRaisesRegex(OperatorSkillValidationError, "missing feedback-closed-loop"):
            self.load(copied)

    def test_mismatched_skill_identity_is_rejected(self) -> None:
        copied = self.copy_resources()
        skill_path = copied / "edit_operator_skills" / "feedback-closed-loop" / "SKILL.md"
        skill_path.write_text(
            skill_path.read_text(encoding="utf-8").replace(
                "name: feedback-closed-loop", "name: wrong-operator", 1
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(OperatorSkillValidationError, "mismatched skill name"):
            self.load(copied)

    def test_empty_reference_is_rejected_and_tampering_changes_digest(self) -> None:
        copied = self.copy_resources()
        reference = next(
            (copied / "edit_operator_skills" / "alternative-path-contrast" / "references").iterdir()
        )
        reference.write_text("tampered but nonempty\n", encoding="utf-8")
        tampered = self.load(copied)
        self.assertNotEqual(tampered.content_digest, OPERATOR_SKILLS.content_digest)
        self.assertNotEqual(
            tampered["alternative-path-contrast"].content_digest,
            OPERATOR_SKILLS["alternative-path-contrast"].content_digest,
        )

        reference.write_text("\n", encoding="utf-8")
        with self.assertRaisesRegex(OperatorSkillValidationError, "empty reference"):
            self.load(copied)

    def test_symlink_escape_is_rejected_structurally(self) -> None:
        copied = self.copy_resources()
        reference = next(
            (copied / "edit_operator_skills" / "alternative-path-contrast" / "references").iterdir()
        )
        outside = self.tmp / "outside.md"
        outside.write_text("outside", encoding="utf-8")
        reference.unlink()
        try:
            reference.symlink_to(outside)
        except OSError as exc:
            self.skipTest(f"symlinks unavailable: {exc}")
        with self.assertRaisesRegex(OperatorSkillValidationError, "escapes package resource root"):
            self.load(copied)


class OperatorPlannerTest(unittest.TestCase):
    def test_preserves_exact_five_taste_biases_and_guidance(self) -> None:
        self.assertEqual(len(IDEA_TASTE_MODES), 5)
        for mode in IDEA_TASTE_MODES:
            taste = get_taste(mode)
            self.assertEqual(set(taste.skill_bias), EXPECTED_OPERATORS)
            self.assertTrue(taste.instantiation_guidance)
            with self.assertRaises(TypeError):
                taste.skill_bias["surgical-modularity"] = 0.0  # type: ignore[index]

    def test_taste_changes_selection_for_same_defect_and_structure(self) -> None:
        planner = OperatorPlanner()
        profile = StructuralProfile(scope_kind="execution_path", has_multi_path_shape=True)
        defects = ("brittle_single_path", "over_conservative_execution")
        moonshot = planner.rank(defects, profile=profile, taste=get_taste("moonshot_inventor"))
        bridge = planner.rank(defects, profile=profile, taste=get_taste("bridge_builder"))
        self.assertEqual(moonshot[0].operator.name, "speculative-execution-with-repair")
        self.assertEqual(bridge[0].operator.name, "alternative-path-contrast")

    def test_adaptive_priors_affect_rank(self) -> None:
        taste = get_taste("moonshot_inventor")
        baseline = OperatorPlanner().rank(
            ("stagnant_novelty",),
            profile=StructuralProfile(scope_kind="existing_subsystem"),
            taste=taste,
        )
        self.assertEqual(baseline[0].operator.name, "mechanism-commit-innovation")

        profile = StructuralProfile(scope_kind="execution_path", has_multi_path_shape=True)
        defects = ("brittle_single_path", "over_conservative_execution")
        planner = OperatorPlanner()
        default = planner.rank(defects, profile=profile, taste=taste)
        for _ in range(5):
            planner.update_prior("speculative-execution-with-repair", 0.0)
            planner.update_prior("alternative-path-contrast", 1.0)
        adapted = planner.rank(defects, profile=profile, taste=taste)
        self.assertEqual(default[0].operator.name, "speculative-execution-with-repair")
        self.assertEqual(adapted[0].operator.name, "alternative-path-contrast")

    def test_production_adaptive_success_threshold_is_point_seven(self) -> None:
        planner = OperatorPlanner()
        below = planner.update_prior("surgical-modularity", 0.69)
        at_threshold = planner.update_prior("surgical-modularity", 0.7)

        self.assertEqual(below.successes, 0)
        self.assertEqual(at_threshold.successes, 1)

    def test_structural_and_scope_constraints_apply(self) -> None:
        planner = OperatorPlanner()
        taste = get_taste("evidence_first")
        component_names = {
            candidate.operator.name
            for candidate in planner.rank(
                ("silent_failure", "brittle_single_path"),
                profile=StructuralProfile(scope_kind="existing_component", control_centered=True),
                taste=taste,
            )
        }
        self.assertNotIn("feedback-closed-loop", component_names)
        self.assertNotIn("alternative-path-contrast", component_names)

        training_free_names = {
            candidate.operator.name
            for candidate in planner.rank(
                ("silent_failure",),
                profile=StructuralProfile(
                    scope_kind="execution_path",
                    control_centered=True,
                    training_free_like=True,
                ),
                taste=taste,
            )
        }
        self.assertNotIn("feedback-closed-loop", training_free_names)

    def test_exploration_candidate_is_deterministic_with_explicit_rng(self) -> None:
        planner = OperatorPlanner()
        profile = StructuralProfile(scope_kind="execution_path", has_multi_path_shape=True)
        defects = (
            "brittle_single_path",
            "rare_regime_failure",
            "weak_fallback_behavior",
            "over_conservative_execution",
            "latency_bottleneck",
            "rollback_blindspot",
        )

        def select(seed: str):
            return planner.select(
                defects,
                limit=2,
                profile=profile,
                taste=get_taste("ambitious_realist"),
                rng=random.Random(seed),
            )

        first = select("fixture")
        second = select("fixture")
        self.assertEqual(
            tuple(candidate.operator.name for candidate in first),
            tuple(candidate.operator.name for candidate in second),
        )
        self.assertFalse(first[0].exploratory)
        self.assertTrue(first[1].exploratory)
        ranked = planner.rank(defects, profile=profile, taste=get_taste("ambitious_realist"))
        self.assertNotEqual(first[1].operator.name, ranked[0].operator.name)
        self.assertIn(first[1].operator.name, {candidate.operator.name for candidate in ranked[1:]})


if __name__ == "__main__":
    unittest.main()
