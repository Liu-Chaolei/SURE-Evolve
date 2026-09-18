from pathlib import Path
import sys
import unittest
from copy import deepcopy

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root.parent/'research_idea/scripts'))
sys.path.insert(0, str(root/'scripts'))
from pragmatic_tts_candidate import validate_tts_candidate


class TtsCandidateTests(unittest.TestCase):
    def setUp(self):
        self.contract = {'candidate_parameters': {'training':['learning_rate'], 'inference':['cfg_strength']}}
        self.candidate = {'requires_training':False, 'evidence_ids':['paper1'],
            'change_set':[{'domain':'inference','target':'cfg_strength','description':'calibrate guidance'}],
            'parameters':{'inference':{'cfg_strength':1.8}}}

    def test_inference_and_cross_domain(self):
        self.assertEqual(validate_tts_candidate(deepcopy(self.candidate),self.contract,{'paper1'})['candidate_type'],'inference')
        c=deepcopy(self.candidate);c['requires_training']=True
        c['change_set'].append({'domain':'train','target':'learning_rate','description':'adjust rate'})
        c['parameters']['training']={'learning_rate':5e-6}
        self.assertEqual(validate_tts_candidate(c,self.contract,{'paper1'})['candidate_type'],'fine_tune')

    def test_fixed_budget_and_false_declarations_rejected(self):
        c=deepcopy(self.candidate);c['requires_training']=True
        with self.assertRaises(ValueError):validate_tts_candidate(c,self.contract,{'paper1'})
        c['change_set'].append({'domain':'train','target':'epochs','description':'shorten'})
        c['parameters']['training']={'epochs':1}
        with self.assertRaises(ValueError):validate_tts_candidate(c,self.contract,{'paper1'})

    def test_unknown_evidence_and_missing_change_rejected(self):
        with self.assertRaises(ValueError):validate_tts_candidate(self.candidate,self.contract,{'other'})
        c=deepcopy(self.candidate);c['change_set']=[]
        with self.assertRaises(ValueError):validate_tts_candidate(c,self.contract,{'paper1'})


if __name__ == '__main__':
    unittest.main()
