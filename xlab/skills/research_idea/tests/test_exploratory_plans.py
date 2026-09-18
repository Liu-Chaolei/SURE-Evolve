from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from research_idea_lib.algorithm.operators import OperatorPlanner, StructuralProfile, OPERATORS
from research_idea_lib.algorithm.tastes import IDEA_TASTE_MODES, get_taste


class ExploratoryPlanTests(unittest.TestCase):
    def test_unmatched_ranked_operations_are_exploratory(self):
        for mode in IDEA_TASTE_MODES:
            for defects in [(),('unexplored_gap',),('latency_bottleneck',),('validation_gap',)]:
                plans=OperatorPlanner().plans(defects,limit=3,profile=StructuralProfile('broad_architecture'),taste=get_taste(mode),rng=random.Random(42))
                self.assertTrue(plans)
                self.assertTrue(all(p.target_defects for p in plans))

    def test_direct_nonexploratory_plan_still_checks_target(self):
        with self.assertRaisesRegex(ValueError,'does not address'):
            OPERATORS[2].plan(('latency_bottleneck',))


if __name__=='__main__':
    unittest.main()
