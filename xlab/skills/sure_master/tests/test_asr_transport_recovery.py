import json
import sys
import unittest
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

SKILLS = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SKILLS / "sure_master/scripts"), str(SKILLS / "research_idea/scripts")]
from research_idea_lib.providers.openai_compatible import _urllib_transport, _decode_completion
from sure_idea_review import normalize_architecture_review
from native_paths import portable_task_context
from research_idea_lib.common import assert_portable_output
from research_idea_lib.providers.contracts import ProviderUsage
from native_sure import prepare_parent


class RecoveryTest(unittest.TestCase):
    def response(self, lines):
        class Response:
            headers = {"Content-Type": "text/event-stream"}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def getcode(self): return 200
            def __iter__(self):
                yield from lines
                raise AssertionError("Read beyond completed response")
        return Response()

    def test_chat_completion_stops_at_done(self):
        lines = [b'data: {"choices":[{"delta":{"content":"OK"}}]}\n', b'data: [DONE]\n']
        with patch("urllib.request.urlopen", return_value=self.response(lines)):
            _, _, raw = _urllib_transport("https://example.test", {}, b'{"stream":true}', 1)
        self.assertEqual(_decode_completion(raw)[0], "OK")

    def test_responses_stops_at_completed_event(self):
        value = {"type": "response.completed", "response": {"object": "response", "status": "completed",
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "OK"}]}]}}
        with patch("urllib.request.urlopen", return_value=self.response([b'data: ' + json.dumps(value).encode() + b'\n'])):
            _, _, raw = _urllib_transport("https://example.test", {}, b'{"stream":true}', 1)
        self.assertEqual(_decode_completion(raw)[0], "OK")

    def test_truncated_stream_rejected(self):
        with self.assertRaises(ValueError):
            _decode_completion(b'data: {"choices":[]}\n')

    def test_domain_flags_preserve_meaning(self):
        review = normalize_architecture_review({"change_set": [{"domain": {"arch": True, "train": False}}]})
        self.assertEqual(review["change_set"][0]["domain"], "arch")
        with self.assertRaises(ValueError):
            normalize_architecture_review({"change_set": [{"domain": {"arch": True, "train": True}}]})

    def test_external_checkpoint_location_becomes_reference(self):
        source = {"model_artifact": {"checkpoint_dir": "/outside/immutable/checkpoints"},
                  "survey_path": "/project/literature/survey.json"}
        result = portable_task_context(source, Path("/project"))
        assert_portable_output(result)
        self.assertEqual(result["survey_path"], "literature/survey.json")
        self.assertEqual(result["model_artifact"]["checkpoint_dir"]["location_kind"], "external_reference")
        self.assertEqual(source["model_artifact"]["checkpoint_dir"], "/outside/immutable/checkpoints")

    def test_parent_projection_cache_binds_wire_policy(self):
        idea = {key: "Existing baseline" for key in ("title", "abstract", "core_contribution", "method", "risks")}
        idea.update(components=[{"name": "encoder", "description": "Existing encoder"}], tags=[], root_domains=[])
        response = SimpleNamespace(json_value=idea, usage=ProviderUsage(1, 1, 2),
                                   trace=SimpleNamespace(to_dict=lambda: {"status": "success"}))
        runtime = SimpleNamespace(agent_model="test-model", chat_completions_url="https://example.test/v1/chat/completions",
                                  request_timeout_seconds=10, max_retries=0)
        with tempfile.TemporaryDirectory() as directory, \
             patch("native_sure.provider_api_key", return_value="test-placeholder"), \
             patch("native_sure.OpenAICompatibleProvider.complete", return_value=response) as complete:
            root = Path(directory) / "operation"
            prepare_parent({"current_best": {"implementation": "baseline"}}, root, runtime)
            prepare_parent({"current_best": {"implementation": "baseline"}}, root, runtime)
            self.assertEqual(complete.call_count, 1)
            with patch.dict("os.environ", {"XLAB_SURE_MAX_OUTPUT_TOKENS": "4096"}):
                prepare_parent({"current_best": {"implementation": "baseline"}}, root, runtime)
            self.assertEqual(complete.call_count, 2)


if __name__ == "__main__":
    unittest.main()
