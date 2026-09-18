from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

root=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root/'skills/research_idea/scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

import native_sure
import sure_idea_review
from research_idea_lib.providers import ProviderResult, ProviderTrace, ProviderUsage
from research_idea_lib.providers import OpenAICompatibleProvider as RealProvider, OpenAICompatibleConfig
from research_idea_lib.providers.routing import API_KEY_ORDER, build_routing_policy


class FixedProvider:
    def __init__(self,payload):self.payload=deepcopy(payload)
    def complete(self,request):
        return ProviderResult(text=json.dumps(self.payload),json_value=deepcopy(self.payload),usage=ProviderUsage(),
            trace=ProviderTrace(provider='fixture',operation=request.operation,input_digest=request.input_digest,
                                output_kind='json',model=request.model,attempts=1,status='success'))


class SureProjectionTests(unittest.TestCase):
    def test_real_router_keeps_a_negative_scientific_review(self):
        values={'XI_BASE_URL':'https://xi.example/v1','ZAI_BASE_URL':'https://zai.example/v4',
                'OPENAI_BASE_URL':'https://openai.example/v1',**{k:'private-'+k for k in API_KEY_ORDER}}
        values['XLAB_API_ROUTING_POLICY']=json.dumps(build_routing_policy(values))
        calls=[]
        def transport(url,headers,body,timeout):
            calls.append(url)
            data={'object':'response','status':'completed','output':[{'type':'message','role':'assistant',
                'content':[{'type':'output_text','text':json.dumps({'accepted':False,'reason':'Unsupported candidate.'})}]}]}
            return 200,{},json.dumps(data).encode()
        def factory(**kwargs):
            return RealProvider(api_key='routing',endpoint='https://xi.example/v1/chat/completions',
                config=OpenAICompatibleConfig(timeout_seconds=1,max_attempts=1),transport=transport)
        runtime=SimpleNamespace(evaluation_model='gpt-6-astra',chat_completions_url='https://xi.example/v1/chat/completions',
                                request_timeout_seconds=1,max_retries=0)
        with patch.dict(os.environ,values,clear=True),patch.object(sure_idea_review,'OpenAICompatibleProvider',factory):
            result=sure_idea_review.review_candidate(runtime,{})
        self.assertFalse(result['review']['accepted'])
        self.assertEqual(len(calls),1)
        self.assertEqual(result['trace']['routing']['selected_slot'],'XI_API_KEY')

    def test_parent_envelope_retains_identity_and_raw_projection_receipt(self):
        idea={'title':'Existing model','abstract':'Existing speech model.','core_contribution':'Flow matching.',
              'method':'Predict flow velocity.','risks':'Distribution shift.',
              'components':[{'name':'DiT','description':'The supplied backbone.'}],
              'tags':['speech'],'root_domains':['TTS']}
        wire={'answer':idea}
        runtime=SimpleNamespace(agent_model='glm-5.3-flash',chat_completions_url='https://fixture/v1/chat/completions',
                                request_timeout_seconds=1,max_retries=0)
        with tempfile.TemporaryDirectory() as directory,patch.object(native_sure,'OpenAICompatibleProvider',lambda **_:FixedProvider(wire)):
            output=native_sure.prepare_parent({'current_best':{'solution_digest':'known'}},Path(directory)/'operation',runtime)
            self.assertEqual(output['native_idea'],idea)
            self.assertEqual(output['native_idea_digest'],native_sure.digest(idea))
            cached=json.loads(next((Path(directory)/'parent_projections').glob('*.json')).read_text())
            self.assertEqual(cached['response_projection']['raw_response_digest'],native_sure.digest(wire))
            self.assertEqual(wire,{'answer':idea})

    def test_review_envelope_does_not_turn_a_rejection_into_acceptance(self):
        runtime=SimpleNamespace(evaluation_model='glm-5.3-flash',chat_completions_url='https://fixture/v1/chat/completions',
                                request_timeout_seconds=1,max_retries=0)
        for wire in [{'answer':{'accepted':False,'reason':'Unsupported change.'}},
                     {'accepted':False,'reason':'Unsupported change.','answer':{'accepted':True}}]:
            with self.subTest(wire=wire),patch.object(sure_idea_review,'OpenAICompatibleProvider',lambda **_:FixedProvider(wire)):
                result=sure_idea_review.review_candidate(runtime,{})
                self.assertEqual(result['raw_review'],wire)
                with self.assertRaisesRegex(ValueError,'Unsupported change'):
                    sure_idea_review.validate_review(result['review'])


if __name__=='__main__':unittest.main()
