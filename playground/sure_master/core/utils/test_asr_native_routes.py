"""Validate the deployed three-slot policy with fake HTTP, never live credentials."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace
ROOT = Path(__file__).resolve().parents[4] / 'runs/asr_native_mcts_v2_20260917/external/XLab/xlab/skills/research_idea'
sys.path.insert(0, str(ROOT / 'scripts'))
from research_idea_lib.providers.routing import API_KEY_ORDER, build_routing_policy, routing_policy_from_environment
from research_idea_lib.providers import OpenAICompatibleProvider, OpenAICompatibleConfig, ProviderRequest

class ASRRoutes(unittest.TestCase):
    def env(self):
        values = {'ZAI_BASE_URL':'https://z.example/v4','OPENAI_BASE_URL':'https://o.example/v1',
                  'XI_API_KEY':'must-never-use', 'XI_BASE_URL':'https://xi.example/v1',
                  **{k:'fake-'+k for k in API_KEY_ORDER}}
        values['XLAB_API_ROUTING_POLICY'] = json.dumps(build_routing_policy(values))
        return values

    def test_order_protocol_provenance_and_cooldown(self):
        self.assertEqual(API_KEY_ORDER, ('ZAI_API_KEY','ZAI_API_KEY2','OPENAI_API_KEY'))
        calls=[]
        def transport(url, headers, raw, timeout):
            slot=headers['Authorization'].removeprefix('Bearer fake-')
            body=json.loads(raw); calls.append((slot,url,body))
            if slot!='OPENAI_API_KEY': return 429, {'Retry-After':'120'}, b'{}'
            return 200,{},json.dumps({'object':'response','status':'completed','output':[{'type':'message','role':'assistant','content':[{'type':'output_text','text':'{"score":0}'}]}]}).encode()
        with tempfile.TemporaryDirectory() as tmp:
            env=self.env();env['XLAB_API_ROUTING_STATE_DIR']=tmp
            with patch.dict(os.environ,env,clear=True):
                provider=OpenAICompatibleProvider(api_key='fake',endpoint='https://z.example/v4/chat/completions',
                    config=OpenAICompatibleConfig(max_attempts=3),transport=transport,sleep=lambda _:None)
                req=ProviderRequest(operation='test',model='glm-5.3-flash',structured_input={'id':1},system_prompt='JSON',user_prompt='JSON',output_kind='json')
                first=provider.complete(req)
                second=provider.complete(replace(req,structured_input={'id':2}))
        self.assertEqual([c[0] for c in calls], [*API_KEY_ORDER,'OPENAI_API_KEY'])
        self.assertTrue(calls[0][1].endswith('/chat/completions'))
        self.assertEqual(calls[0][1],calls[1][1])
        self.assertTrue(calls[2][1].endswith('/responses'))
        self.assertEqual(first.trace.model,'gpt-6-astra')
        self.assertEqual(first.trace.routing['selected_slot'],'OPENAI_API_KEY')
        self.assertEqual(second.json_value,{'score':0})

    def test_xi_policy_is_rejected(self):
        env=self.env();policy=json.loads(env['XLAB_API_ROUTING_POLICY'])
        policy['routes'].insert(0,dict(slot='XI_API_KEY',model='gpt-6-astra',responses=True,base_url='https://xi.example/v1'))
        env['XLAB_API_ROUTING_POLICY']=json.dumps(policy)
        with patch.dict(os.environ,env,clear=True),self.assertRaises(ValueError):
            routing_policy_from_environment()

if __name__ == '__main__': unittest.main()
