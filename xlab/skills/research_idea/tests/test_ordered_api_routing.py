from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from research_idea_lib.providers import OpenAICompatibleConfig, OpenAICompatibleProvider, ProviderExhaustedError, ProviderRequest
from research_idea_lib.providers.contracts import ProviderContractError
from research_idea_lib.providers.routing import API_KEY_ORDER, build_routing_policy, routing_policy_from_environment, trace_matches_requested_model
from research_idea_lib.algorithm.keynote_pipeline import run_keynote_pipeline
from research_idea_lib.algorithm.keynote_pipeline import _complete, _score_output, KEYNOTE_SCORE_OPERATION
from research_idea_lib.algorithm.provider_adapter import ProviderAdapter
from research_idea_lib.algorithm.runtime_adapters import _keynote_output, _keynote_output_dict, _keynote_output_from_dict, _trace_record
from research_idea_lib.algorithm.runtime_adapters import GenericWorkflowProvider
from research_idea_lib.algorithm.workflow import ProviderOperation, OP_MATERIALIZATION
from research_idea_lib.audit import scan_artifact_safety
from test_keynote_pipeline import registry, hit, request as keynote_request


def environment(cache=None):
    values={'XI_BASE_URL':'https://xi.example/v1','ZAI_BASE_URL':'https://zai.example/v4',
            'OPENAI_BASE_URL':'https://openai.example/v1',
            **{key:'private-'+key for key in API_KEY_ORDER}}
    values['XLAB_API_ROUTING_POLICY']=json.dumps(build_routing_policy(values))
    values['XLAB_SURE_HTTP_ATTEMPTS']='3'
    if cache:
        values.update(XLAB_SURE_PROVIDER_CACHE_ROOT=str(cache),XLAB_NATIVE_RECOVERY='1')
    return values


def provider(transport,sleep=lambda _:None):
    return OpenAICompatibleProvider(api_key='routing-enabled',endpoint='https://xi.example/v1/chat/completions',
        config=OpenAICompatibleConfig(timeout_seconds=30,max_attempts=3),transport=transport,sleep=sleep)


def query():
    return ProviderRequest(operation='fixture',model='gpt-6-astra',structured_input={'task':'test'},
                           system_prompt='Return JSON.',user_prompt='Test',output_kind='json')


def success(body,payload):
    if body['model']=='gpt-6-astra':
        data={'object':'response','status':'completed','output':[{'type':'message','role':'assistant',
            'content':[{'type':'output_text','text':json.dumps(payload)}]}]}
    else:
        data={'choices':[{'message':{'content':json.dumps(payload)}}]}
    return 200,{},json.dumps(data).encode()


class OrderedRoutingTests(unittest.TestCase):
    def test_materialization_missing_required_fields_uses_next_route(self):
        fusion={'title':'A bounded speech refinement','abstract':'Improve speech adaptation.',
                'core_contribution':'A fixed schedule change.','hypothesis':'The schedule improves CER.',
                'method':'Apply the supplied schedule.','risks':'Possible overfitting.',
                'components':['schedule'],'tags':['speech'],'root_domains':['TTS']}
        modes=['moonshot_inventor','bridge_builder','steady_engineer','ambitious_realist','evidence_first']
        idea={**fusion,'introduction':'The baseline adaptation can plateau.', 'research_question':'Can decay help?',
              'experiment_plan':['Run the fixed candidate budget.'],'data_requirements':['Use the declared split.'],
              'baselines':['Use the supplied frozen baseline.'],'metrics':['Chinese CER'],
              'risks':['Possible overfitting.'],'algorithm':['Apply the schedule.'],
              'source_modes':modes,'evidence_ids':['evidence:allowed'],
              'reference_ids':['paper:allowed'],'reference_papers':['Supplied reference.']}
        data={'topic':'TTS','fusion':fusion,'source_modes':modes,'evidence_ids':['evidence:allowed'],
              'evidence':[{'paper_ids':['paper:allowed']}], 'research_policy':{'evidence_mode':'local_literature'}}
        calls=[]
        def transport(url,headers,raw,timeout):
            slot=headers['Authorization'].removeprefix('Bearer private-');calls.append(slot)
            payload=deepcopy(idea)
            if slot=='XI_API_KEY':payload.pop('introduction')
            return success(json.loads(raw),{'idea_result':payload})
        with patch.dict(os.environ,environment(),clear=True):
            result=GenericWorkflowProvider(provider(transport),SimpleNamespace(fusion_model='gpt-6-astra')).execute(
                ProviderOperation(OP_MATERIALIZATION,data))
        self.assertEqual(calls,['XI_API_KEY','ZAI_API_KEY'])
        self.assertEqual(result.value['idea_result'],idea)
        self.assertEqual(result.metadata['model'],'glm-5.3-flash')

    def test_valid_repair_stop_does_not_shop_for_more_edits(self):
        calls=[]
        def transport(url,headers,raw,timeout):
            calls.append(url);return success(json.loads(raw),{'stop':True})
        with patch.dict(os.environ,environment(),clear=True):
            result=ProviderAdapter(provider(transport),model='gpt-6-astra').propose_repair(
                {'allowed_operations':['remove','replace','rewire']})
        self.assertIsNone(result)
        self.assertEqual(len(calls),1)

    def test_http_200_with_invalid_score_falls_through_without_guessing(self):
        calls=[]
        def transport(url,headers,raw,timeout):
            slot=headers['Authorization'].removeprefix('Bearer private-');calls.append(slot);body=json.loads(raw)
            if slot=='XI_API_KEY':return 401,{},b'{}'
            if slot=='ZAI_API_KEY':return 429,{},b'{}'
            value={'answer':'score=5'} if slot=='ZAI_API_KEY2' else {'score':5}
            status,headers,response=success(body,value)
            data=json.loads(response)
            data['usage']=({'input_tokens':2,'output_tokens':1,'total_tokens':3} if body['model']=='gpt-6-astra'
                           else {'prompt_tokens':2,'completion_tokens':1,'total_tokens':3})
            return status,headers,json.dumps(data).encode()
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,environment(directory),clear=True):
            result=_complete(provider(transport),operation=KEYNOTE_SCORE_OPERATION,model='gpt-6-astra',
                structured_input={'paper_id':'fixture'},prompt='Return a score.',temperature=0.0)
            self.assertEqual(calls,list(API_KEY_ORDER))
            self.assertEqual(_score_output(result),5)
            self.assertEqual(result.trace.routing['selected_slot'],'OPENAI_API_KEY')
            self.assertEqual(result.trace.routing['attempts'][2]['error_code'],'invalid_response_contract')
            self.assertEqual(result.usage.total_tokens,6)
            rejected=list(Path(directory).glob('routes/ZAI_API_KEY2/rejected-*.json'))
            self.assertEqual(len(rejected),1)
            self.assertEqual(json.loads(rejected[0].read_text())['result']['json_value'],{'answer':'score=5'})

    def test_valid_zero_score_is_not_retried_for_a_better_grade(self):
        calls=[]
        def transport(url,headers,raw,timeout):
            calls.append(url);return success(json.loads(raw),{'score':0})
        with patch.dict(os.environ,environment(),clear=True):
            result=_complete(provider(transport),operation=KEYNOTE_SCORE_OPERATION,model='gpt-6-astra',
                structured_input={'paper_id':'fixture'},prompt='Return a score.',temperature=0.0)
        self.assertEqual(len(calls),1)
        self.assertEqual(_score_output(result),0)

    def test_all_invalid_contracts_preserve_last_draft_for_semantic_repair(self):
        def validate(result):raise ValueError('Required field is missing')
        request=replace(query(),response_validator=validate,validation_profile='fixture.response.v1')
        calls=[]
        def transport(url,headers,raw,timeout):
            calls.append(url);return success(json.loads(raw),{'wrong':True})
        with patch.dict(os.environ,environment(),clear=True):
            with self.assertRaises(ProviderContractError) as caught:provider(transport).complete(request)
            self.assertIsInstance(caught.exception,ValueError)
            self.assertEqual(caught.exception.previous_draft,{'wrong':True})
            self.assertEqual(len(calls),4)
            self.assertNotIn('response_validator',request.cache_payload())
            json.dumps(request.cache_payload())

    def test_exact_order_distinct_zai_credentials_and_correct_protocols(self):
        calls=[]
        def transport(url,headers,raw,timeout):
            body=json.loads(raw);key=headers['Authorization'].removeprefix('Bearer private-')
            calls.append((key,url,body))
            if key=='XI_API_KEY':return 401,{},b'{}'
            if key=='ZAI_API_KEY':return 429,{},b'{}'
            return success(body,{'score':42})
        sleeps=[]
        with patch.dict(os.environ,environment(),clear=True):
            result=provider(transport,sleeps.append).complete(query())
            self.assertEqual([x[0] for x in calls],list(API_KEY_ORDER[:3]))
            self.assertTrue(calls[0][1].endswith('/responses'))
            self.assertIn('input',calls[0][2])
            self.assertTrue(calls[1][1].endswith('/chat/completions'))
            self.assertEqual(calls[1][1],calls[2][1])
            self.assertIn('messages',calls[2][2])
            self.assertEqual(result.trace.model,'glm-5.3-flash')
            self.assertEqual(result.trace.routing['selected_slot'],'ZAI_API_KEY2')
            self.assertEqual(result.trace.attempts,3)
            self.assertTrue(trace_matches_requested_model(result.trace,'gpt-6-astra'))
            self.assertEqual(_trace_record(result.trace,'gpt-6-astra')['model'],'glm-5.3-flash')
        self.assertEqual(sleeps,[])

    def test_openai_is_last_and_success_stops_fallback(self):
        for winner in API_KEY_ORDER:
            with self.subTest(winner=winner):
                calls=[]
                def transport(url,headers,raw,timeout):
                    key=headers['Authorization'].removeprefix('Bearer private-');calls.append(key)
                    return success(json.loads(raw),{'ok':True}) if key==winner else (503,{},b'{}')
                with patch.dict(os.environ,environment(),clear=True):
                    result=provider(transport).complete(query())
                self.assertEqual(calls,list(API_KEY_ORDER[:API_KEY_ORDER.index(winner)+1]))
                self.assertEqual(result.trace.routing['selected_slot'],winner)

    def test_each_new_request_starts_with_the_preferred_route(self):
        calls=[]
        def transport(url,headers,raw,timeout):
            key=headers['Authorization'].removeprefix('Bearer private-');calls.append(key)
            if len(calls)==1:return 429,{},b'{}'
            return success(json.loads(raw),{'ok':True})
        with patch.dict(os.environ,environment(),clear=True):
            client=provider(transport);client.complete(query());client.complete(query())
        self.assertEqual(calls,['XI_API_KEY','ZAI_API_KEY','XI_API_KEY'])

    def test_missing_keys_are_skipped_without_network_calls(self):
        env=environment();env.pop('XI_API_KEY');env.pop('ZAI_API_KEY');env.pop('ZAI_API_KEY2')
        calls=[]
        def transport(url,headers,raw,timeout):
            calls.append(headers['Authorization']);return success(json.loads(raw),{'ok':True})
        with patch.dict(os.environ,env,clear=True):result=provider(transport).complete(query())
        self.assertEqual(calls,['Bearer private-OPENAI_API_KEY'])
        self.assertEqual([a['status'] for a in result.trace.routing['attempts'][:3]],['unconfigured']*3)

    def test_cache_replay_keeps_actual_route_and_never_logs_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            calls=[]
            def transport(url,headers,raw,timeout):
                key=headers['Authorization'].removeprefix('Bearer private-');calls.append(key)
                return success(json.loads(raw),{'ok':True}) if key=='ZAI_API_KEY2' else (429,{},b'{}')
            with patch.dict(os.environ,environment(directory),clear=True):
                first=provider(transport).complete(query())
                second=provider(transport).complete(query())
                self.assertEqual(first,second)
                self.assertEqual(len(calls),3)
                text=''.join(p.read_text() for p in Path(directory).rglob('*') if p.is_file())
                for slot in API_KEY_ORDER:self.assertNotIn('private-'+slot,text)
                self.assertIn('ZAI_API_KEY2',text)
                self.assertEqual(scan_artifact_safety(Path(directory))['secrets'],[])
                cached=[p for p in Path(directory).glob('*.json') if not p.name.endswith('.operation.json')]
                self.assertEqual(len(cached),1)
                cached[0].unlink()
                with self.assertRaisesRegex(ValueError,'Unconfirmed native operation'):
                    provider(transport).complete(query())
                self.assertEqual(len(calls),3)

    def test_artifact_audit_covers_all_configured_route_credentials(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,environment(),clear=True):
            path=Path(directory)/'artifact.json'
            for slot in API_KEY_ORDER:
                path.write_text(json.dumps({'value':os.environ[slot]}))
                self.assertIn(str(path),scan_artifact_safety(Path(directory))['secrets'])

    def test_all_routes_fail_with_a_bounded_shared_recovery_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            calls=[];delays=[]
            def transport(url,headers,raw,timeout):
                calls.append(headers['Authorization']);return 429,{},b'{}'
            with patch.dict(os.environ,environment(directory),clear=True):
                with self.assertRaisesRegex(ProviderExhaustedError,'All configured API routes failed'):
                    provider(transport,delays.append).complete(query())
            self.assertEqual(len(calls),12)
            self.assertEqual(delays,[60,180])
            receipt=json.loads(next(Path(directory).glob('*.operation.json')).read_text())
            self.assertEqual(receipt['cycles'],3)
            self.assertEqual(receipt['status'],'failed')

    def test_deadline_prevents_sending_to_later_routes(self):
        now=[0.0];calls=[]
        def transport(url,headers,raw,timeout):
            calls.append(url);now[0]=2.0;return 401,{},b'{}'
        with patch.dict(os.environ,environment(),clear=True),patch('research_idea_lib.providers.openai_compatible.time.time',lambda:now[0]):
            with self.assertRaises(ProviderExhaustedError) as caught:
                provider(transport)._complete_uncached(query(),deadline=1.0)
        self.assertEqual(len(calls),1)
        self.assertEqual(caught.exception.trace.error_code,'timeout')

    def test_keynote_fallback_accepts_real_model_and_preserves_route_on_resume(self):
        def transport(url,headers,raw,timeout):
            key=headers['Authorization'].removeprefix('Bearer private-');body=json.loads(raw)
            if key=='XI_API_KEY':return 401,{},b'{}'
            text=body['messages'][0]['content']
            value={'answer':'42'} if 'Scoring rule:' in text else {'summary':'Supported summary.','insight':'Bounded insight.'}
            return success(body,value)
        with patch.dict(os.environ,environment(),clear=True):
            request=replace(keynote_request(registry('p1'),hit('paragraph',('p1',),score=0.9)),model='gpt-6-astra')
            result=run_keynote_pipeline(request,provider=provider(transport))
            self.assertEqual(result.ranked_keynotes[0].relevance,42)
            trace=result.provider_traces[0]
            self.assertEqual(trace.model,'glm-5.3-flash')
            restored=_keynote_output_from_dict(_keynote_output_dict(_keynote_output(result,resumed=False)),resumed=True)
            self.assertEqual(restored.provider_traces[0].routing,trace.routing)
            self.assertFalse(trace_matches_requested_model(replace(trace,model='forged-model'),'gpt-6-astra'))
            self.assertFalse(trace_matches_requested_model(trace,'unexpected-request-model'))

    def test_policy_change_and_secret_fields_fail_closed(self):
        env=environment();policy=json.loads(env['XLAB_API_ROUTING_POLICY'])
        for bad in [dict(policy,routes=list(reversed(policy['routes']))),
                    dict(policy,api_key='must-not-be-public')]:
            with patch.dict(os.environ,{**env,'XLAB_API_ROUTING_POLICY':json.dumps(bad)},clear=True):
                with self.assertRaises(ValueError):routing_policy_from_environment()
        env['ZAI_BASE_URL']='https://changed.example/v4'
        with patch.dict(os.environ,env,clear=True):
            with self.assertRaisesRegex(ValueError,'endpoints changed'):
                provider(lambda *args:self.fail('must not send after endpoint drift')).complete(query())


if __name__=='__main__':
    unittest.main()
