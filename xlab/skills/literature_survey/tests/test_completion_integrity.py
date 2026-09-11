import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from literature_survey_lib.xcientist.utils.completion import complete_chat
from literature_survey_lib.xcientist.utils.utils import extract_json
from literature_survey_lib.xcientist.utils.api_call import ChatAgent


class CompletionIntegrityTest(unittest.TestCase):
    def test_long_stream_wait_does_not_extend_connection_timeout(self):
        response = Mock(status_code=200)
        response.iter_lines.return_value = [b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}', b'data: [DONE]']
        with tempfile.TemporaryDirectory() as temporary, patch("requests.Session.post", return_value=response) as post:
            result, _ = complete_chat("https://example.invalid", {}, {"model": "test", "stream": True, "max_tokens": 100}, 600, temporary, stream_timeout=1800)
            self.assertEqual(result, "ok")
            self.assertEqual(post.call_args.kwargs["timeout"], (30, 1800))

    def test_retry_explains_validation_failure_and_keeps_successful_results(self):
        agent = ChatAgent.__new__(ChatAgent)
        agent.batch_workers = 32
        agent.model_name = "test"
        agent.logger = Mock()
        agent.config = SimpleNamespace(BasicInfo=SimpleNamespace(debug=False))
        agent.batch_remote_chat = Mock(side_effect=[["bad", "good"], ["fixed"]])

        def validate(value, info):
            if value == "bad":
                raise ValueError("Missing exact paper ID title:abc")
            return True, value

        result = agent.batch_remote_chat_with_retry(["first", "second"], validate, max_retry=2, info_dict={})
        self.assertEqual(result, ["fixed", "good"])
        correction = agent.batch_remote_chat.call_args_list[1].args[0]
        self.assertEqual(len(correction), 1)
        self.assertIn("Missing exact paper ID title:abc", correction[0])

    def test_stream_completion_and_diagnostics_exclude_credentials(self):
        response = Mock(status_code=200)
        response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"reasoning_content":"private reasoning"}}]}',
            b'data: {"choices":[{"delta":{"content":"{\\"ok\\":true}"},"finish_reason":"stop"}]}',
            b'data: [DONE]',
        ]
        with tempfile.TemporaryDirectory() as temporary, patch("requests.Session.post", return_value=response):
            value, _ = complete_chat("https://example.invalid", {"Authorization": "secret-credential"},
                                     {"model": "test", "stream": True, "max_tokens": 100}, 10, temporary)
            self.assertEqual(json.loads(value), {"ok": True})
            record_text = next((Path(temporary) / "state/llm_responses").glob("*.json")).read_text()
            self.assertNotIn("secret-credential", record_text)
            self.assertNotIn("private reasoning", record_text)
            self.assertEqual(json.loads(record_text)["finish_reason"], "stop")
        response.close.assert_called_once()

    def test_partial_stream_and_length_limit_are_rejected(self):
        for finish, expected in ((None, requests.ConnectionError), ("length", ValueError)):
            response = Mock(status_code=200)
            chunk = {"choices": [{"delta": {"content": '{"unfinished":'}, "finish_reason": finish}]}
            response.iter_lines.return_value = [b"data: " + json.dumps(chunk).encode()]
            with tempfile.TemporaryDirectory() as temporary, patch("requests.Session.post", return_value=response):
                with self.assertRaises(expected):
                    complete_chat("https://example.invalid", {}, {"model": "test", "stream": True, "max_tokens": 100}, 10, temporary)
            response.close.assert_called_once()

    def test_reasoning_is_not_used_as_final_content(self):
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": "", "reasoning_content": "analysis"}, "finish_reason": "stop"}]}
        with tempfile.TemporaryDirectory() as temporary, patch("requests.Session.post", return_value=response):
            with self.assertRaises(requests.ConnectionError):
                complete_chat("https://example.invalid", {}, {"model": "test", "stream": False, "max_tokens": 100}, 10, temporary)

    def test_json_with_trailing_explanation_but_not_truncated_json(self):
        self.assertEqual(extract_json('```json\n[{"name":"Speech"}]\n```\nThis is the result.'), [{"name": "Speech"}])
        with self.assertRaises(json.JSONDecodeError):
            extract_json('[{"name":"Speech"}, {"name":"Cut off')

    def test_stream_is_drained_and_connection_session_is_reused(self):
        response = Mock(status_code=200)
        drained = []

        def lines():
            yield b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}'
            yield b'data: [DONE]'
            drained.append(True)

        response.iter_lines.side_effect = lines
        with tempfile.TemporaryDirectory() as temporary, patch("requests.Session.post", autospec=True, return_value=response) as post:
            for _ in range(2):
                complete_chat("https://example.invalid", {}, {"model": "test", "stream": True, "max_tokens": 100}, 10, temporary)
            self.assertIs(post.call_args_list[0].args[0], post.call_args_list[1].args[0])
            self.assertEqual(drained, [True, True])


if __name__ == "__main__":
    unittest.main()
