from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


SKILL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_DIR / "scripts"))

from research_idea_lib.providers import (  # noqa: E402
    DeterministicFakeProvider,
    FakeFixture,
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    ProviderError,
    ProviderExhaustedError,
    ProviderRequest,
    extract_json_object,
    structured_input_digest,
)


def completion(content: str) -> bytes:
    return json.dumps(
        {
            "choices": [{"message": {"content": content}}],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
        }
    ).encode()


class ProviderTests(unittest.TestCase):
    def test_identified_user_agent_preserves_model_and_credentials(self) -> None:
        captured = {}

        def transport(url, headers, body, timeout):
            captured.update(headers=headers, payload=json.loads(body))
            return 200, {}, completion('{"ok":true}')

        OpenAICompatibleProvider(
            api_key="test-key", endpoint="https://provider.invalid/v1/chat/completions",
            transport=transport,
        ).complete(self.request())
        self.assertEqual(captured["headers"]["User-Agent"], "XLab/1.0")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(captured["payload"]["model"], "test-model")

    def request(self, *, output_kind: str = "json") -> ProviderRequest:
        return ProviderRequest(
            operation="xlab.research_idea.idea.generate.v1",
            model="test-model",
            structured_input={"topic": "graph learning", "seed": 7},
            system_prompt="secret-system-prefix: act as a scientist",
            user_prompt="private full prompt body",
            output_kind=output_kind,
        )

    def test_fake_dispatches_by_operation_and_structured_input_digest(self) -> None:
        request = self.request()
        fixture = FakeFixture(text='{"idea":"stable"}', json_value={"idea": "stable"})
        provider = DeterministicFakeProvider(
            {(request.operation, request.input_digest): fixture}
        )

        changed_prompts = ProviderRequest(
            operation=request.operation,
            model=request.model,
            structured_input={"seed": 7, "topic": "graph learning"},
            system_prompt="different system prompt",
            user_prompt="different user prompt",
        )
        result = provider.complete(changed_prompts)

        self.assertEqual(result.json_value, {"idea": "stable"})
        self.assertEqual(result.trace.input_digest, request.input_digest)
        with self.assertRaises(ProviderError):
            provider.complete(
                ProviderRequest(
                    operation="xlab.research_idea.idea.evaluate.v1",
                    model=request.model,
                    structured_input=request.structured_input,
                    system_prompt=request.system_prompt,
                    user_prompt=request.user_prompt,
                )
            )

    def test_malformed_response_is_retried(self) -> None:
        responses = [completion("not json"), completion('{"idea":"recovered"}')]
        calls: list[float] = []

        def transport(url: str, headers: object, body: bytes, timeout: float):
            calls.append(timeout)
            return 200, {}, responses.pop(0)

        result = OpenAICompatibleProvider(
            api_key="test-key",
            endpoint="https://provider.invalid/v1/chat/completions",
            config=OpenAICompatibleConfig(timeout_seconds=4, max_attempts=2),
            transport=transport,
            sleep=lambda _: None,
        ).complete(self.request())

        self.assertEqual(result.json_value, {"idea": "recovered"})
        self.assertEqual(result.trace.attempts, 2)
        self.assertEqual(calls, [4, 4])

    def test_exhausted_retry_returns_bounded_redacted_trace(self) -> None:
        calls = 0

        def transport(url: str, headers: object, body: bytes, timeout: float):
            nonlocal calls
            calls += 1
            return 200, {}, completion("still malformed")

        provider = OpenAICompatibleProvider(
            api_key="sk-super-secret",
            endpoint="https://user:password@provider.invalid/secret-prefix",
            config=OpenAICompatibleConfig(max_attempts=2),
            transport=transport,
            sleep=lambda _: None,
        )
        with self.assertRaises(ProviderExhaustedError) as raised:
            provider.complete(self.request())

        self.assertEqual(calls, 2)
        self.assertEqual(raised.exception.trace.attempts, 2)
        serialized = json.dumps(raised.exception.trace.to_dict())
        serialized += repr(self.request()) + repr(provider) + str(raised.exception)
        self.assertNotIn("sk-super-secret", serialized)
        self.assertNotIn("user:password", serialized)
        self.assertNotIn("secret-prefix", serialized)
        self.assertNotIn("secret-system-prefix", serialized)
        self.assertNotIn("private full prompt body", serialized)

    def test_text_completion_does_not_request_or_parse_json(self) -> None:
        captured: dict[str, object] = {}

        def transport(url: str, headers: object, body: bytes, timeout: float):
            captured.update(json.loads(body))
            return 200, {}, completion("plain answer")

        result = OpenAICompatibleProvider(
            api_key="test-key",
            endpoint="https://provider.invalid/v1/chat/completions",
            transport=transport,
        ).complete(self.request(output_kind="text"))

        self.assertEqual(result.text, "plain answer")
        self.assertIsNone(result.json_value)
        self.assertNotIn("response_format", captured)
        self.assertEqual(result.usage.total_tokens, 18)

    def test_json_extraction_handles_fenced_and_prefixed_object(self) -> None:
        self.assertEqual(
            extract_json_object('Result follows:\n```json\n{"idea":{"score":1}}\n```'),
            {"idea": {"score": 1}},
        )
        with self.assertRaises(ValueError):
            extract_json_object("[1, 2, 3]")

    def test_structured_digest_is_canonical(self) -> None:
        self.assertEqual(
            structured_input_digest({"b": 2, "a": [1, True]}),
            structured_input_digest({"a": [1, True], "b": 2}),
        )


if __name__ == "__main__":
    unittest.main()
