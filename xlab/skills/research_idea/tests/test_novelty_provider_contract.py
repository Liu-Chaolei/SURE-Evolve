from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from research_idea_lib.algorithm.runtime_adapters import _ComponentNoveltyProvider
from research_idea_lib.providers import ProviderResult, ProviderTrace, ProviderUsage


class NoveltyContractTests(unittest.TestCase):
    def evaluate(self, raw):
        requests=[]
        def complete(request):
            requests.append(request)
            return ProviderResult(text=json.dumps(raw),json_value=deepcopy(raw),usage=ProviderUsage(),
                trace=ProviderTrace(provider='fixture',operation=request.operation,input_digest=request.input_digest,
                    output_kind='json',model=request.model,attempts=1,status='success'))
        result=_ComponentNoveltyProvider(SimpleNamespace(complete=complete),model='glm-5.3-flash').execute(
            SimpleNamespace(to_payload=lambda:{'candidate_id':'root','retrieved_nodes':[]}))
        return result,requests

    def test_wrapped_provenance_retains_all_entries_and_scores(self):
        raw={'answer':{'retrieval_similarity':3,'perceived_novelty':2,'rubric_score':3,
                      'rationale':'Related precedent exists.','provenance':[{'evidence_id':'core:fixture'}]}}
        original=deepcopy(raw)
        result,requests=self.evaluate(raw)
        self.assertEqual(result.retrieval_similarity,3)
        self.assertEqual(json.loads(result.provenance_json),{'evidence':[{'evidence_id':'core:fixture'}]})
        self.assertEqual(raw,original)
        self.assertIn('0, 1, 2, 3, 4, 5',requests[0].system_prompt)
        self.assertIn('NOT percentages',requests[0].system_prompt)

    def test_actual_wrong_scale_and_labels_are_not_guessed(self):
        base={'retrieval_similarity':3,'perceived_novelty':2,'rubric_score':3,
              'rationale':'Related precedent exists.','provenance':{'evidence_ids':['core:fixture']}}
        for field,value in [('retrieval_similarity',58),('perceived_novelty','moderate'),('rubric_score',True),('provenance',[])]:
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.evaluate({'answer':{**base,field:value}})


if __name__=='__main__':unittest.main()
