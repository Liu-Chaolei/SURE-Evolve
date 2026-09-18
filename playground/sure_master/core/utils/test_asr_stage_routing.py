"""Offline route-contract tests; no real credentials or network requests."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from playground.sure_master.tools import asr_stage_routing as stage

ROOT = Path(__file__).resolve().parents[4]
NATIVE = ROOT / "runs/asr_native_mcts_v2_20260917/external/XLab/xlab/skills/research_idea/scripts"
sys.path.insert(0, str(NATIVE))
from research_idea_lib.providers import OpenAICompatibleProvider, OpenAICompatibleConfig, ProviderRequest
from research_idea_lib.providers.contracts import ProviderExhaustedError


class StageRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stage.install_native()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.values = {"ZAI_API_KEY": "fake-zai", "ZHOU_API_KEY": "fake-zhou", "OPENAI_API_KEY": "fake-openai",
                       "ZAI_BASE_URL": "https://zai.invalid/v4", "ZHOU_API_BASE_URL": "https://zhou.invalid/v1",
                       "OPENAI_BASE_URL": "https://openai.invalid/v1",
                       "XCODE_API_KEY": "fake-xcode", "XCODE_API_BASE_URL": "https://xcode.invalid/v1"}
        self.env_file = Path(self.directory.name) / "credentials.env"
        self.env_file.write_text("\n".join(k + "=" + v for k, v in self.values.items()))
        self.policy = stage.build_policy(self.values)
        environment = {"XLAB_API_ROUTING_POLICY": json.dumps(self.policy),
                       "XLAB_API_ROUTING_ENV_FILE": str(self.env_file)}
        context = patch.dict(os.environ, environment, clear=True)
        context.start()
        self.addCleanup(context.stop)

    def request(self, operation):
        return ProviderRequest(operation=operation, model="glm-5.3-flash", structured_input={"id": 1},
                               system_prompt="Return JSON", user_prompt="Return JSON", output_kind="json")

    def provider(self, transport):
        return OpenAICompatibleProvider(api_key="unused", endpoint="https://unused.invalid/v1/chat/completions",
            config=OpenAICompatibleConfig(max_attempts=1, timeout_seconds=20), transport=transport, sleep=lambda _: None)

    @staticmethod
    def success(raw):
        value = json.loads(raw)
        payload = ({"object": "response", "status": "completed", "output": [{"type": "message",
                   "role": "assistant", "content": [{"type": "output_text", "text": '{"score":0}'}]}]}
                   if value["model"] == "gpt-6-astra" else {"choices": [{"message": {"content": '{"score":0}'}}]})
        return 200, {}, json.dumps(payload).encode()

    def test_all_operations_are_classified(self):
        self.assertFalse(stage.OPENAI_OPERATIONS & stage.GLM_OPERATIONS)
        self.assertEqual(len(self.policy["operations"]), 21)
        for operation in stage.OPENAI_OPERATIONS:
            self.assertEqual([r['slot'] for r in stage.select_routes(self.policy, operation)], ['OPENAI_API_KEY', 'XCODE_API_KEY'])
        for operation in stage.GLM_OPERATIONS:
            self.assertEqual([r['slot'] for r in stage.select_routes(self.policy, operation)], list(stage.SLOTS))
        with self.assertRaises(ValueError):
            stage.select_routes(self.policy, 'unknown.operation')

    def test_glm_uses_zhou_second_and_keeps_valid_negative_judgment(self):
        calls = []
        def transport(url, headers, raw, timeout):
            calls.append((url, headers['Authorization'], json.loads(raw)['model']))
            return (429, {}, b'{}') if len(calls) == 1 else self.success(raw)
        result = self.provider(transport).complete(self.request('xlab.research_idea.idea.evaluate.v1'))
        self.assertEqual([c[1] for c in calls], ['Bearer fake-zai', 'Bearer fake-zhou'])
        self.assertEqual(result.json_value, {'score': 0})
        self.assertEqual(result.trace.routing['selected_slot'], 'ZHOU_API_KEY')
        self.assertTrue(stage.trace_matches_requested_model(result.trace, 'glm-5.3-flash'))

    def test_glm_fallback_reaches_openai_responses(self):
        calls = []
        def transport(url, headers, raw, timeout):
            calls.append(url)
            return (503, {}, b'{}') if len(calls) < 3 else self.success(raw)
        result = self.provider(transport).complete(self.request('xlab.research_idea.keynote.compress.v1'))
        self.assertEqual(len(calls), 3)
        self.assertTrue(calls[-1].endswith('/responses'))
        self.assertEqual(result.trace.model, 'gpt-6-astra')

    def test_openai_generation_never_falls_back_to_glm(self):
        calls = []
        def transport(url, headers, raw, timeout):
            calls.append(headers['Authorization'])
            return 503, {}, b'{}'
        with self.assertRaises(ProviderExhaustedError):
            self.provider(transport).complete(self.request('xlab.research_idea.idea.generate.v1'))
        self.assertEqual(calls, ['Bearer fake-openai', 'Bearer fake-xcode'])

    def test_policy_drift_and_unwanted_slot_fail(self):
        self.assertNotIn('XI_API_KEY', json.dumps(self.policy))
        self.assertNotIn('ZAI_API_KEY2', json.dumps(self.policy))
        policy = dict(self.policy, agents={})
        with patch.dict(os.environ, XLAB_API_ROUTING_POLICY=json.dumps(policy)), self.assertRaises(ValueError):
            stage.routing_policy_from_environment()

    def test_controller_routes_and_knowledge_agent_selection(self):
        from openai.resources.chat.completions import Completions
        from openai import OpenAI, APIConnectionError
        import httpx
        calls = []
        def original(client, *args, **kwargs):
            calls.append((str(client._client.base_url), kwargs['model']))
            if 'zai.invalid' in calls[-1][0]:
                raise APIConnectionError(request=httpx.Request('POST', calls[-1][0]))
            return {'ok': True}
        policy_path = Path(self.directory.name) / 'policy.json'
        policy_path.write_text(json.dumps(self.policy))
        with patch.object(Completions, 'create', original):
            stage.install_controller(self.env_file, policy_path, Path(self.directory.name) / 'audit.jsonl')
            client = OpenAI(api_key='dummy', base_url='https://unused.invalid/v1')
            client.chat.completions.create(model='glm-5.3-flash', messages=[])
            client.chat.completions.create(model='gpt-6-astra', messages=[])
        self.assertEqual([model for _,model in calls], ['glm-5.3-flash', 'glm-5.3-flash', 'gpt-6-astra'])
        self.assertIn('zhou.invalid', calls[1][0])
        self.assertIn('openai.invalid', calls[2][0])
        self.assertEqual(stage.AGENT_ROUTES['debug'], 'glm')
        self.assertEqual(stage.AGENT_ROUTES['knowledge_promotion'], 'openai')

    def test_controller_openai_failure_never_uses_glm(self):
        from openai.resources.chat.completions import Completions
        from openai import OpenAI, APIConnectionError
        import httpx
        calls = []
        def original(client, *args, **kwargs):
            calls.append(str(client._client.base_url))
            raise APIConnectionError(request=httpx.Request('POST', calls[-1]))
        policy_path = Path(self.directory.name) / 'policy.json'
        policy_path.write_text(json.dumps(self.policy))
        with patch.object(Completions, 'create', original):
            stage.install_controller(self.env_file, policy_path, Path(self.directory.name) / 'audit.jsonl')
            client = OpenAI(api_key='dummy', base_url='https://unused.invalid/v1')
            with self.assertRaisesRegex(RuntimeError, 'routes unavailable'):
                client.chat.completions.create(model='gpt-6-astra', messages=[])
        self.assertEqual(len(calls), 2)
        self.assertIn('openai.invalid', calls[0])
        self.assertIn('xcode.invalid', calls[1])

    def test_native_xcode_fallback_uses_responses_and_preserves_model(self):
        calls = []
        def transport(url, headers, raw, timeout):
            calls.append((url, headers['Authorization']))
            return (503, {}, b'{}') if len(calls) == 1 else self.success(raw)
        result = self.provider(transport).complete(self.request('xlab.research_idea.fusion.generate.v1'))
        self.assertEqual(calls, [('https://openai.invalid/v1/responses', 'Bearer fake-openai'),
                                 ('https://xcode.invalid/v1/responses', 'Bearer fake-xcode')])
        self.assertEqual(result.trace.model, 'gpt-6-astra')
        self.assertEqual(result.trace.routing['selected_slot'], 'XCODE_API_KEY')
        self.assertTrue(stage.trace_matches_requested_model(result.trace, 'glm-5.3-flash'))

    def test_glm_can_fall_back_through_all_four_slots(self):
        calls = []
        def transport(url, headers, raw, timeout):
            calls.append(headers['Authorization'])
            return (503, {}, b'{}') if len(calls) < 4 else self.success(raw)
        result = self.provider(transport).complete(self.request('xlab.research_idea.idea.diagnostic.v1'))
        self.assertEqual(calls, ['Bearer fake-zai', 'Bearer fake-zhou', 'Bearer fake-openai', 'Bearer fake-xcode'])
        self.assertEqual(result.trace.routing['selected_slot'], 'XCODE_API_KEY')


if __name__ == '__main__':
    unittest.main()
