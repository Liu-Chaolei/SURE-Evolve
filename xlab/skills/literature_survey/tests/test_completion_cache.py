import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from literature_survey_lib.xcientist.utils.completion import complete_chat


class CompletionCacheTest(unittest.TestCase):
    def test_exact_success_is_reused_but_failed_or_changed_requests_are_not(self):
        url = "http://localhost:28091/v1/chat/completions"
        payload = {"model": "test", "stream": False, "max_tokens": 100, "messages": [{"role": "user", "content": "Claim"}]}
        for error in [None, "ConnectionError"]:
            with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"XLAB_LITERATURE_SURVEY_REUSE_COMPLETIONS": "1"}):
                root = Path(temporary)
                records = root / "state/llm_responses"
                records.mkdir(parents=True)
                (records / "old.json").write_text(json.dumps({
                    "request_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                    "endpoint_sha256": hashlib.sha256(url.encode()).hexdigest(),
                    "http_status": 200, "error_type": error, "finish_reason": "stop", "content": "Yes"}))
                response = Mock(status_code=200)
                response.json.return_value = {"choices": [{"message": {"content": "No"}, "finish_reason": "stop"}]}
                with patch("requests.Session.post", return_value=response) as post:
                    result, _ = complete_chat(url, {}, payload, 10, root)
                    self.assertEqual(result, "Yes" if error is None else "No")
                    self.assertEqual(post.call_count, 0 if error is None else 1)
                    changed = {**payload, "messages": [{"role": "user", "content": "Different claim"}]}
                    self.assertEqual(complete_chat(url, {}, changed, 10, root)[0], "No")


if __name__ == "__main__":
    unittest.main()
