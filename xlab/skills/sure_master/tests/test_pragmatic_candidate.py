from pathlib import Path
import sys
import unittest

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root.parent/'research_idea/scripts'))
sys.path.insert(0,str(root/'scripts'))
from pragmatic_candidate import BASE_ARCH, validate_architecture


class PragmaticCandidateTests(unittest.TestCase):
    def test_valid_distinct_stage_allocation(self):
        values={**BASE_ARCH,'num-encoder-layers':'2,2,4,4,2,2'}
        self.assertEqual(validate_architecture(values,[]),values)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            validate_architecture(values,[{'architecture_arguments':values}])

    def test_baseline_and_invalid_shapes_rejected(self):
        for values in [BASE_ARCH,{**BASE_ARCH,'num-encoder-layers':'1,2'},
                       {**BASE_ARCH,'encoder-unmasked-dim':'512,512,512,512,512,512'}]:
            with self.assertRaises(ValueError):validate_architecture(values,[])


if __name__=='__main__':
    unittest.main()
