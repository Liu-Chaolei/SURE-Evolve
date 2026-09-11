from __future__ import annotations

import unittest

from playground.sure_master.core.contracts import IdeaRequest, MetricSpec
from playground.sure_master.core.utils.fingerprints import canonical_json, digest, portable_path, redact


class XlabFingerprintTests(unittest.TestCase):
    def test_canonical_json_and_digest_are_stable(self) -> None:
        self.assertEqual(canonical_json({"b": 2, "a": 1}), '{"a":1,"b":2}')
        self.assertEqual(digest({"a": 1, "b": 2}), digest({"b": 2, "a": 1}))
        self.assertNotEqual(digest({"a": 1}), digest({"a": 2}))

    def test_redact_secrets_and_tokens(self) -> None:
        value = {"api_key": "secret", "nested": "Bearer abcdefghijklmnop", "safe": "value"}
        redacted = redact(value)
        self.assertEqual(redacted["api_key"], "[REDACTED]")
        self.assertEqual(redacted["nested"], "[REDACTED]")
        self.assertEqual(redacted["safe"], "value")

    def test_portable_path_rejects_host_paths_and_traversal(self) -> None:
        self.assertEqual(portable_path("artifacts/result.json", "/tmp/workspace"), "artifacts/result.json")
        with self.assertRaises(ValueError):
            portable_path("/tmp/workspace/artifacts/result.json", "/tmp/workspace")
        with self.assertRaises(ValueError):
            portable_path("../outside.json", "/tmp/workspace")

    def test_request_requires_axis_only_for_staged(self) -> None:
        common = dict(
            request_id="request-1",
            sure_run_id="sure-1",
            task_id="task-1",
            task_description="test task",
            round_index=1,
            requested_idea_count=4,
            metric=MetricSpec(name="wer", direction="lower"),
            input_digest="sha256:input",
        )
        request = IdeaRequest(search_mode="ordinary", **common)
        self.assertIsNone(request.axis)
        with self.assertRaises(ValueError):
            IdeaRequest(search_mode="staged_axes", **common)


if __name__ == "__main__":
    unittest.main()
