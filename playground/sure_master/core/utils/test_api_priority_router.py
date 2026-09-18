import json
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from playground.sure_master.tools.api_priority_router import ORDER, SUPPORTED, PriorityRouter, cooldown_seconds, upstreams


class ApiPriorityRouterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.audit = Path(self.temp.name) / 'audit.jsonl'
        self.values = {'XI_API_KEY':'secret-xi','XI_BASE_URL':'https://xi.example',
                       'ZAI_API_KEY':'secret-zai','ZAI_API_KEY2':'secret-zai2',
                       'ZAI_BASE_URL':'https://zai.example/api/v4',
                       'OPENAI_API_KEY':'secret-last','OPENAI_BASE_URL':'https://last.example/v1',
                       'ZHOU_API_KEY':'secret-zhou','ZHOU_API_BASE_URL':'https://zhou.example/v1',
                       'XCODE_API_KEY':'secret-xcode','XCODE_API_BASE_URL':'https://xcode.example/v1'}
        self.payload = {'model':'sure-priority-router','messages':[{'role':'user','content':'hello'}], 'stream':True}

    def router(self, transport, clock=lambda:1000):
        return PriorityRouter(lambda:self.values, self.audit, transport=transport, clock=clock,
                              load_order=lambda:[name for name in SUPPORTED if name not in ('ZHOU_API_KEY','XCODE_API_KEY')])

    def test_glm_role_uses_zhou_then_openai_without_old_keys(self):
        calls = []
        def transport(endpoint, headers, body, timeout):
            calls.append((headers['Authorization'], json.loads(body)['model']))
            return (200 if 'last.example' in endpoint else 429), {}, b'{}'
        router = PriorityRouter(lambda:self.values, self.audit, transport=transport,
                                load_order=lambda:SUPPORTED)
        status, _, _ = router.complete({**self.payload, 'model':'sure-glm-router'})
        self.assertEqual(status, 200)
        self.assertEqual(calls, [('Bearer secret-zai', 'glm-5.3-flash'),
                                ('Bearer secret-zhou', 'glm-5.3-flash'),
                                ('Bearer secret-last', 'gpt-6-astra')])

    def test_openai_role_never_uses_glm(self):
        calls = []
        def transport(endpoint, headers, body, timeout):
            calls.append(headers['Authorization'])
            return 200, {}, b'{}'
        router = PriorityRouter(lambda:self.values, self.audit, transport=transport,
                                load_order=lambda:SUPPORTED)
        router.complete({**self.payload, 'model':'sure-openai-router'})
        self.assertEqual(calls, ['Bearer secret-last'])

    def test_order_endpoint_and_model_mapping(self):
        profiles=upstreams(self.values, SUPPORTED)
        self.assertEqual([p.name for p in profiles],list(SUPPORTED))
        self.assertEqual(profiles[0].endpoint,'https://xi.example/v1/chat/completions')
        self.assertEqual(profiles[1].endpoint,profiles[2].endpoint)
        self.assertEqual([p.model for p in profiles],['gpt-6-astra','glm-5.3-flash','glm-5.3-flash','glm-5.3-flash','gpt-6-astra','gpt-6-astra'])
        self.assertNotIn('secret-xi',repr(profiles[0]))

    def test_both_roles_use_xcode_only_after_prior_routes_fail(self):
        for alias, expected in (
            ('sure-glm-router',['secret-zai','secret-zhou','secret-last','secret-xcode']),
            ('sure-openai-router',['secret-last','secret-xcode']),
        ):
            with self.subTest(alias=alias):
                calls=[]
                def transport(endpoint, headers, body, timeout):
                    calls.append(headers['Authorization'].removeprefix('Bearer '))
                    return (200 if 'xcode.example' in endpoint else 503), {}, b'{}'
                router=PriorityRouter(lambda:self.values,self.audit,transport=transport,
                                      load_order=lambda:SUPPORTED)
                status, headers, _=router.complete({**self.payload,'model':alias})
                self.assertEqual(status,200)
                self.assertEqual(headers['X-SURE-Upstream'],'XCODE_API_KEY')
                self.assertEqual(calls,expected)

    def test_missing_xcode_is_skipped_and_no_glm_for_premium(self):
        self.values.pop('XCODE_API_KEY')
        calls=[]
        def transport(endpoint, headers, body, timeout):
            calls.append(endpoint)
            return 503, {}, b'{}'
        router=PriorityRouter(lambda:self.values,self.audit,transport=transport,
                              load_order=lambda:SUPPORTED)
        self.assertEqual(router.complete({**self.payload,'model':'sure-openai-router'})[0],503)
        self.assertEqual(calls,['https://last.example/v1/chat/completions'])

    def test_failover_all_four_and_no_credential_leak(self):
        calls=[]
        statuses=iter([401,429,503,200])
        def transport(endpoint,headers,body,timeout):
            calls.append((headers['Authorization'],json.loads(body)['model']))
            status=next(statuses)
            return status,{'Content-Type':'text/event-stream'},b'data: [DONE]\n\n'
        status,headers,body=self.router(transport).complete(self.payload)
        self.assertEqual(status,200)
        self.assertEqual([c[0] for c in calls],['Bearer secret-xi','Bearer secret-zai','Bearer secret-zai2','Bearer secret-last'])
        self.assertEqual(headers['X-SURE-Upstream'],'OPENAI_API_KEY')
        self.assertEqual(body,b'data: [DONE]\n\n')
        self.assertNotIn('secret-',self.audit.read_text())

    def test_cooldown_then_return_to_highest_priority(self):
        now=[1000];calls=[];xi_status=[401]
        def transport(endpoint,headers,body,timeout):
            calls.append(endpoint)
            return (xi_status[0] if 'xi.example' in endpoint else 200),{},b'{}'
        router=self.router(transport,lambda:now[0])
        router.complete(self.payload);router.complete(self.payload)
        self.assertEqual(len(calls),3)
        self.assertIn('zai.example',calls[-1])
        now[0]=1301;xi_status[0]=200
        router.complete(self.payload)
        self.assertIn('xi.example',calls[-1])

    def test_changed_key_is_not_stuck_in_old_cooldown(self):
        calls=[]
        def transport(endpoint,headers,body,timeout):
            calls.append(headers['Authorization'])
            return (401 if headers['Authorization']=='Bearer secret-xi' else 200),{},b'{}'
        router=self.router(transport);router.complete(self.payload)
        self.values['XI_API_KEY']='secret-replaced';router.complete(self.payload)
        self.assertEqual(calls[-1],'Bearer secret-replaced')

    def test_bad_request_does_not_switch_and_redacts_key(self):
        calls=[]
        def transport(*args):
            calls.append(args)
            return 400,{},b'{"error":{"message":"context length exceeded secret-xi"}}'
        status,headers,body=self.router(transport).complete(self.payload)
        self.assertEqual(status,400);self.assertEqual(len(calls),1)
        self.assertIn(b'context length',body);self.assertNotIn(b'secret-xi',body)

    def test_interrupted_stream_can_switch_before_any_output_is_returned(self):
        calls=[]
        def transport(endpoint,*args):
            calls.append(endpoint)
            if len(calls)==1:raise ConnectionError('partial stream')
            return 200,{'Content-Type':'text/event-stream'},b'data: complete\n\n'
        status,headers,body=self.router(transport).complete(self.payload)
        self.assertEqual(status,200);self.assertEqual(body,b'data: complete\n\n')
        self.assertEqual(headers['X-SURE-Upstream'],'ZAI_API_KEY')

    def test_all_unavailable_fails_explicitly_without_raw_errors(self):
        router=self.router(lambda *args:(401,{},b'secret-xi'))
        status,headers,body=router.complete(self.payload)
        self.assertEqual(status,503);self.assertNotIn(b'secret-',body)
        self.assertNotIn('secret-',self.audit.read_text())

    def test_retry_after_is_respected(self):
        self.assertEqual(cooldown_seconds(429,{'Retry-After':'3600'},b'',1000),3600)

    def test_default_uses_only_two_zai_keys(self):
        self.assertEqual([p.name for p in upstreams(self.values)], ['ZAI_API_KEY','ZAI_API_KEY2'])
        calls=[]
        def transport(endpoint,headers,body,timeout):
            calls.append(headers['Authorization'])
            return (429 if len(calls)==1 else 200),{},b'{}'
        router=PriorityRouter(lambda:self.values,self.audit,transport=transport)
        self.assertEqual(router.complete(self.payload)[0],200)
        self.assertEqual(calls,['Bearer secret-zai','Bearer secret-zai2'])

    def test_live_policy_removes_endpoint_from_pending_fallback(self):
        order=list(SUPPORTED);calls=[]
        def transport(endpoint,headers,body,timeout):
            calls.append(headers['Authorization'])
            order[:]=list(ORDER)
            return 429,{},b'{}'
        router=PriorityRouter(lambda:self.values,self.audit,transport=transport,load_order=lambda:order)
        self.assertEqual(router.complete(self.payload)[0],503)
        self.assertEqual(calls,['Bearer secret-xi','Bearer secret-zai','Bearer secret-zai2'])
        self.assertEqual(router.health()['order'],list(ORDER))

    def test_restart_preserves_quota_cooldown(self):
        cooldown=Path(self.temp.name)/'cooldown.json'
        calls=[]
        def transport(endpoint,headers,body,timeout):
            calls.append(headers['Authorization'])
            return (429 if headers['Authorization']=='Bearer secret-zai' else 200),{},b'{}'
        for _ in range(2):
            router=PriorityRouter(lambda:self.values,self.audit,transport=transport,clock=lambda:1000,
                                  cooldown_file=cooldown)
            self.assertEqual(router.complete(self.payload)[0],200)
        self.assertEqual(calls,['Bearer secret-zai','Bearer secret-zai2','Bearer secret-zai2'])

    def test_one_request_per_key_while_other_requests_wait(self):
        values={k:v for k,v in self.values.items() if k!='ZAI_API_KEY2'}
        live=[0];peak=[0];lock=threading.Lock()
        def transport(*args):
            with lock:
                live[0]+=1;peak[0]=max(peak[0],live[0])
            time.sleep(.05)
            with lock:live[0]-=1
            return 200,{},b'{}'
        router=PriorityRouter(lambda:values,self.audit,transport=transport,wait_timeout=2)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(lambda _:router.complete(self.payload),range(4)))
        self.assertTrue(all(result[0]==200 for result in results))
        self.assertEqual(peak[0],1)

    def test_short_cooldown_waits_instead_of_failing_caller(self):
        values={k:v for k,v in self.values.items() if k!='ZAI_API_KEY2'}
        calls=[]
        router=PriorityRouter(lambda:values,self.audit,transport=lambda *args:(calls.append(1) or (200,{},b'{}')),wait_timeout=2)
        identity=upstreams(values)[0].identity
        router.blocked_until[identity]=time.time()+.1
        self.assertEqual(router.complete(self.payload)[0],200)
        self.assertEqual(calls,[1])

    def test_four_requests_per_key_and_fifth_queues(self):
        values={k:v for k,v in self.values.items() if k!='ZAI_API_KEY2'}
        live=[0];peak=[0];lock=threading.Lock();four=threading.Event();release=threading.Event()
        def transport(*args):
            with lock:
                live[0]+=1;peak[0]=max(peak[0],live[0])
                if live[0]==4:four.set()
            release.wait(timeout=2)
            with lock:live[0]-=1
            return 200,{},b'{}'
        router=PriorityRouter(lambda:values,self.audit,transport=transport,wait_timeout=3,max_in_flight_per_key=4)
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures=[pool.submit(router.complete,self.payload) for _ in range(5)]
            try:
                self.assertTrue(four.wait(timeout=2))
                self.assertEqual(peak[0],4)
                self.assertFalse(any(f.done() for f in futures))
            finally:release.set()
            self.assertTrue(all(f.result()[0]==200 for f in futures))
        self.assertEqual(peak[0],4)

    def test_balance_error_and_unicode_reset_are_classified(self):
        self.assertEqual(cooldown_seconds(429,{},b'{"error":{"code":"1113"}}',1000),300)
        body=json.dumps({'error':{'code':'other_quota','message':'将在 2099-01-01 12:00:00 重置'}}).encode()
        self.assertGreater(cooldown_seconds(429,{},body,1000),300)


if __name__=='__main__':
    unittest.main()
