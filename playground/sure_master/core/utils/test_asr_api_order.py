import json
from pathlib import Path
import tempfile
import unittest

from playground.sure_master.tools.api_priority_router import PriorityRouter, upstreams


class ASRApiOrderTests(unittest.TestCase):
    def test_exact_asr_order_and_matching_endpoints(self):
        values = {"XI_API_KEY": "unused-xi", "XI_BASE_URL": "https://xi.example/v1",
                  "ZAI_API_KEY": "primary", "ZAI_API_KEY2": "secondary", "ZAI_BASE_URL": "https://zai.example/v4",
                  "OPENAI_API_KEY": "last", "OPENAI_BASE_URL": "https://openai.example/v1"}
        order = ["ZAI_API_KEY", "ZAI_API_KEY2", "OPENAI_API_KEY"]
        profiles = upstreams(values, order)
        self.assertEqual([p.name for p in profiles], order)
        calls = []
        statuses = iter([401, 429, 200])
        def transport(endpoint, headers, body, timeout):
            calls.append((endpoint, headers["Authorization"], json.loads(body)["model"]))
            return next(statuses), {}, b'{}'
        with tempfile.TemporaryDirectory() as directory:
            router = PriorityRouter(lambda: values, Path(directory) / "audit.jsonl", transport=transport,
                                    load_order=lambda: order)
            status, headers, _ = router.complete({"messages": [{"role": "user", "content": "test"}]})
            self.assertEqual(status, 200)
            self.assertEqual(headers["X-SURE-Upstream"], "OPENAI_API_KEY")
        self.assertEqual(calls, [("https://zai.example/v4/chat/completions", "Bearer primary", "glm-5.3-flash"),
                               ("https://zai.example/v4/chat/completions", "Bearer secondary", "glm-5.3-flash"),
                               ("https://openai.example/v1/chat/completions", "Bearer last", "gpt-6-astra")])

    def test_available_primary_does_not_call_backups(self):
        values = {"ZAI_API_KEY": "primary", "ZAI_API_KEY2": "secondary", "ZAI_BASE_URL": "https://zai.example/v4",
                  "OPENAI_API_KEY": "last"}
        calls = []
        def transport(endpoint, headers, body, timeout):
            calls.append(headers["Authorization"])
            return 200, {}, b'{}'
        with tempfile.TemporaryDirectory() as directory:
            router = PriorityRouter(lambda: values, Path(directory) / "audit.jsonl", transport=transport,
                load_order=lambda: ["ZAI_API_KEY", "ZAI_API_KEY2", "OPENAI_API_KEY"])
            self.assertEqual(router.complete({"messages": []})[0], 200)
        self.assertEqual(calls, ["Bearer primary"])


if __name__ == "__main__":
    unittest.main()
