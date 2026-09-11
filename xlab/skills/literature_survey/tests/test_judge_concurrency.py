import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from literature_survey_lib.xcientist.modules.judge import Judge


class JudgeConcurrencyTest(unittest.TestCase):
    def test_provider_concurrency_is_bounded_and_failures_propagate(self):
        judge = Judge.__new__(Judge)
        judge.config = SimpleNamespace(APIInfo=SimpleNamespace(batch_chat_agent_worker=2),
                                       ModuleInfo=SimpleNamespace(Judge=SimpleNamespace(remove_failed_citation_in_eval=True, nli_temperature=0)))
        judge.judge_model = "fake"
        judge.logger = SimpleNamespace(warning=lambda message: None)
        judge.work_analyzer = SimpleNamespace(get_paper_keynote=lambda pid: {"source": pid})
        lock = threading.Lock()
        active = 0
        maximum = 0

        def respond(**kwargs):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.005)
            with lock:
                active -= 1
            return "Yes"

        judge.chat_agent = SimpleNamespace(remote_chat=respond)
        survey = " ".join(f"Claim number {i} [1]." for i in range(20))
        self.assertEqual(judge.citation_quality(survey, ["p1"]), (1.0, 1.0))
        self.assertEqual(maximum, 2)

        def fail(**kwargs):
            raise ConnectionError("provider unavailable")

        judge.chat_agent.remote_chat = fail
        with self.assertRaisesRegex(ConnectionError, "provider unavailable"):
            judge.citation_quality(survey, ["p1"])


if __name__ == "__main__":
    unittest.main()
