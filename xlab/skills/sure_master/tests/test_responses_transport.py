import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'research_idea/scripts'))
from research_idea_lib.providers.openai_compatible import OpenAICompatibleProvider, _decode_completion
from research_idea_lib.providers.contracts import ProviderRequest


class ResponsesTransportTests(unittest.TestCase):
    def test_native_response_and_request_mapping(self):
        calls=[]
        def transport(endpoint, headers, body, timeout):
            calls.append((endpoint,json.loads(body)))
            return 200, {}, json.dumps({'object':'response','status':'completed',
                'output':[{'type':'reasoning'}, {'type':'message','role':'assistant',
                           'content':[{'type':'output_text','text':'{"ok":true}'}]}],
                'usage':{'input_tokens':10,'output_tokens':2,'total_tokens':12}}).encode()
        with patch.dict(os.environ,{'XLAB_SURE_RESPONSES':'1','XLAB_RESEARCH_IDEA_STREAM':'1'}):
            provider=OpenAICompatibleProvider(api_key='test',endpoint='https://example.test/v1/chat/completions',transport=transport)
            result=provider.complete(ProviderRequest('test','test-model',{},'JSON system','JSON user',output_kind='json',temperature=0))
        self.assertEqual(result.json_value,{'ok':True})
        self.assertEqual(result.usage.total_tokens,12)
        self.assertEqual(calls[0][0],'https://example.test/v1/responses')
        self.assertEqual(calls[0][1]['temperature'],0)
        self.assertTrue(calls[0][1]['stream'])

    def test_native_stream_requires_completed_event(self):
        payload={'object':'response','status':'completed','output':[{'type':'message','role':'assistant',
                 'content':[{'type':'output_text','text':'done'}]}]}
        raw=('event: response.completed\ndata: '+json.dumps({'type':'response.completed','response':payload})+'\n\n').encode()
        self.assertEqual(_decode_completion(raw)[0],'done')
        with self.assertRaises(ValueError):
            _decode_completion(b'event: response.created\ndata: {"type":"response.created"}\n\n')

    def test_incomplete_and_reasoning_only_rejected(self):
        for status in ['incomplete','completed']:
            with self.assertRaises(ValueError):
                _decode_completion(json.dumps({'object':'response','status':status,'output':[{'type':'reasoning'}]}).encode())


if __name__=='__main__':
    unittest.main()
