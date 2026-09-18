import sys
from pathlib import Path
import unittest

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root.parent/'research_idea/scripts'))
sys.path.insert(0,str(root/'scripts'))
from sure_idea_review import normalize_architecture_review


class ReviewInvariantsTests(unittest.TestCase):
    def test_explicit_no_change_is_an_invariant(self):
        value={'change_set':[{'domain':'arch','target':'encoder-dim','description':'Change stage widths'},
                             {'domain':'train','target':'training','description':'No train-domain intervention is proposed. Keep the fixed recipe.'}]}
        fixed=normalize_architecture_review(value)
        self.assertEqual(fixed['change_domains'],['arch'])
        self.assertEqual(len(fixed['change_set']),1)
        self.assertEqual(len(fixed['execution_invariants']),1)
        self.assertEqual(len(value['change_set']),2)

    def test_actual_training_change_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'non-architecture'):
            normalize_architecture_review({'change_set':[{'domain':'train','description':'Change learning rate to 0.01'}]})


if __name__=='__main__':
    unittest.main()
