import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from research_idea_lib.providers.openai_compatible import OpenAICompatibleProvider
from research_idea_lib.providers.contracts import ProviderRequest, ProviderResult, ProviderUsage, ProviderTrace
from research_idea_lib.algorithm.provider_adapter import ProviderAdapter
from research_idea_lib.algorithm.contracts import IdeaState, IdeaComponent
from research_idea_lib.algorithm.evaluation import METRICS
from research_idea_lib.algorithm.tastes import IDEA_TASTE_MODES


class ProviderRecoveryTests(unittest.TestCase):
    def test_empty_detected_defects_is_valid(self):
        class Provider:
            def complete(self, request):
                value = {'metrics': {k:3 for k in METRICS}, 'confidence':.8,
                         'detected_defects':[], 'feedback':'No registry defect is supported by the evidence.'}
                return ProviderResult(json.dumps(value),value,ProviderUsage(),
                    ProviderTrace('fake',request.operation,request.input_digest,'json','test',1,'success'))
        idea = IdeaState(title='ASR',abstract='A concrete model',core_contribution='Structural change',
                         method='Change the encoder',risks='May overfit',components=(IdeaComponent('encoder','Acoustic encoder'),))
        result = ProviderAdapter(Provider(),model='test').evaluate(idea,idea_taste_mode=IDEA_TASTE_MODES[0],diagnostic=True)
        self.assertEqual(result.detected_defects,())

    def test_replay_preserves_occurrences_and_invalidates_changed_prompt(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{'XLAB_SURE_PROVIDER_CACHE_ROOT':tmp}):
            calls=[]
            def transport(*args):
                calls.append(1)
                return 200,{},json.dumps({'choices':[{'message':{'content':json.dumps({'n':len(calls)})}}]}).encode()
            request=ProviderRequest('operation','model',{},'system','user')
            first=OpenAICompatibleProvider(api_key='secret-test-key',endpoint='https://example.test/v1/chat/completions',transport=transport)
            self.assertEqual(first.complete(request).json_value,{'n':1})
            self.assertEqual(first.complete(request).json_value,{'n':2})
            second=OpenAICompatibleProvider(api_key='secret-test-key',endpoint='https://example.test/v1/chat/completions',transport=transport)
            self.assertEqual(second.complete(request).json_value,{'n':1})
            self.assertEqual(second.complete(request).json_value,{'n':2})
            self.assertEqual(len(calls),2)
            second.complete(ProviderRequest('operation','model',{},'changed','user'))
            self.assertEqual(len(calls),3)
            self.assertTrue(all('secret-test-key' not in p.read_text() for p in Path(tmp).glob('*.json')))


if __name__=='__main__':
    unittest.main()
