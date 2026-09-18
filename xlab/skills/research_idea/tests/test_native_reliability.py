from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from research_idea_lib.algorithm.search import MCTSEngine, SearchConfig
from research_idea_lib.algorithm.search_checkpoint import SearchCheckpoint
from research_idea_lib.algorithm.runtime_adapters import _ComponentNoveltyProvider
from research_idea_lib.providers.contracts import ProviderExhaustedError, ProviderTrace, ProviderRecoveryBlockedError
from research_idea_lib.providers.native_recovery import bounded_request
from research_idea_lib.providers.failures import raise_infrastructure_failure
from research_idea_lib.providers.route_health import RouteHealth, cooldown_seconds
from test_research_idea_algorithm import StubProvider, idea
from test_ordered_api_routing import environment, provider, success, query


class ReliabilityTests(unittest.TestCase):
    def test_unknown_request_and_wrapped_outage_are_not_branch_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'receipt.json'
            path.write_text(json.dumps({'request_digest':'identity','status':'running','cycles':1}))
            before=path.read_bytes()
            with self.assertRaises(ProviderRecoveryBlockedError):
                bounded_request(path,'identity',lambda _: self.fail('must not resend'))
            self.assertEqual(path.read_bytes(),before)
        original=ProviderExhaustedError('outage',trace=ProviderTrace('fake','op','digest','json','m',3,'error','http_503'))
        try:
            raise RuntimeError('novelty wrapper') from original
        except RuntimeError as wrapped:
            with self.assertRaises(ProviderExhaustedError):
                raise_infrastructure_failure(wrapped)

    def test_routing_skips_cooling_key_on_subsequent_requests(self):
        calls=[]
        def transport(url,headers,raw,timeout):
            slot=headers['Authorization'];calls.append(slot)
            if slot.endswith('XI_API_KEY'):return 429,{'Retry-After':'120'},b'{}'
            return success(json.loads(raw),{'score':3})
        with tempfile.TemporaryDirectory() as tmp:
            env=environment();env['XLAB_API_ROUTING_STATE_DIR']=tmp
            with patch.dict(os.environ,env,clear=True):
                provider(transport).complete(query())
                provider(transport).complete(replace(query(),structured_input={'task':'second'}))
            self.assertEqual(calls,['Bearer private-XI_API_KEY','Bearer private-ZAI_API_KEY','Bearer private-ZAI_API_KEY'])

    def test_interrupted_iteration_resumes_identical_tree_without_rediagnosing(self):
        config=SearchConfig(max_iterations=12, max_depth=3, seed='resume-test')
        expected=MCTSEngine(StubProvider(), config).search(idea(), 'moonshot_inventor')
        first=StubProvider()
        original=SearchCheckpoint.save
        def interrupt(checkpoint,state,**kwargs):
            original(checkpoint,state,**kwargs)
            if state['counters'].iterations==3:
                raise KeyboardInterrupt('simulated shutdown')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'iteration.json'
            with patch.object(SearchCheckpoint,'save',interrupt),self.assertRaises(KeyboardInterrupt):
                MCTSEngine(first,config).search(idea(),'moonshot_inventor',checkpoint_path=path)
            second=StubProvider()
            actual=MCTSEngine(second,config).search(idea(),'moonshot_inventor',checkpoint_path=path)
            self.assertEqual(second.diagnostic_calls,0)
            self.assertEqual(asdict(actual),asdict(expected))
            self.assertEqual(len(first.requests)+len(second.requests),expected.counters.generation_calls)
            data=json.loads(path.read_text());data['state_digest']='wrong';path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'digest mismatch'):
                MCTSEngine(StubProvider(),config).search(idea(),'moonshot_inventor',checkpoint_path=path)

    def test_unavailable_generation_never_returns_exhausted_mode(self):
        class Failed(StubProvider):
            def generate(self,request):
                raise ProviderExhaustedError('rate limit',trace=ProviderTrace('fake','generate','digest','json','model',3,'error','http_429'))
        with self.assertRaises(ProviderExhaustedError):
            MCTSEngine(Failed(),SearchConfig(max_iterations=4)).search(idea(),'steady_engineer')

    def test_cooldown_survives_reload_and_only_one_probe_acquires(self):
        now=[100.0]
        with tempfile.TemporaryDirectory() as tmp:
            h=RouteHealth(Path(tmp),clock=lambda:now[0])
            h.finish('ZAI','https://test',delay=60)
            other=RouteHealth(Path(tmp),clock=lambda:now[0])
            self.assertEqual(other.acquire('ZAI','https://test',lease_seconds=30),160)
            now[0]=161
            self.assertEqual(other.acquire('ZAI','https://test',lease_seconds=30),0)
            self.assertEqual(h.acquire('ZAI','https://test',lease_seconds=30),191)
            other.finish('ZAI','https://test')
            self.assertEqual(h.acquire('ZAI','https://test',lease_seconds=30),0)
        self.assertEqual(cooldown_seconds(429,{'Retry-After':'120'},b'{}',100),120)
        self.assertEqual(cooldown_seconds(429,{},b'{"error":{"code":"1113"}}',100),300)

    def test_novelty_invalid_score_falls_through_and_valid_zero_is_kept(self):
        calls=[]
        def transport(url,headers,raw,timeout):
            calls.append(headers['Authorization'])
            payload={'retrieval_similarity':0,'perceived_novelty':0,'rubric_score':58 if len(calls)==1 else 0,
                     'rationale':'Supported by supplied evidence.','provenance':{'evidence_ids':['core:fixture']}}
            return success(json.loads(raw),payload)
        with patch.dict(os.environ,environment(),clear=True):
            result=_ComponentNoveltyProvider(provider(transport),model='gpt-6-astra').execute(
                SimpleNamespace(to_payload=lambda:{'candidate_id':'root','retrieved_nodes':[]}))
        self.assertEqual(result.rubric_score,0)
        self.assertEqual(len(calls),2)

    def test_same_invalid_cache_is_preserved_and_repaired_once(self):
        calls=[]
        def transport(url,headers,raw,timeout):
            calls.append(1);return success(json.loads(raw),{'score':5})
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,environment(tmp),clear=True):
            request=replace(query(),validation_profile='score-v1',response_validator=lambda r: None)
            provider(transport).complete(request)
            path=next(p for p in Path(tmp).glob('*.json') if 'result' in json.loads(p.read_text()))
            doc=json.loads(path.read_text());doc['result']['json_value']={'score':'bad'};path.write_text(json.dumps(doc))
            def validate(result):
                if not isinstance(result.json_value['score'],int): raise ValueError('integer required')
            request=replace(request,response_validator=validate)
            self.assertEqual(provider(transport).complete(request).json_value['score'],5)
            self.assertEqual(len(list(Path(tmp).glob('*.rejected-cache.json'))),1)
            self.assertEqual(provider(transport).complete(request).json_value['score'],5)
            self.assertEqual(len(calls),2)
