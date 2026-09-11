import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_literature_survey import FAKE_LLM_ENV, PROJECT_ROOT, SCRIPT, write_graph_fixture


class ExportRefreshTest(unittest.TestCase):
    def test_old_export_refreshes_without_redrafting(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            graph = write_graph_fixture(base)
            run = base / "survey"
            env = {**os.environ, **FAKE_LLM_ENV, "LLM_MODEL": "fake-survey-agent", "LLM_BASE_URL": "https://example.invalid/v1"}
            command = [sys.executable, str(SCRIPT)]
            first = subprocess.run(command + ["smoke", "--run-dir", str(run), "--run-id", "refresh", "--graph", str(graph), "--cwd", str(PROJECT_ROOT)],
                                   env=env, text=True, capture_output=True)
            self.assertEqual(first.returncode, 0, first.stdout[-2000:] + first.stderr[-2000:])
            draft = run / "state/xcientist/draft.raw.json"
            before = draft.stat().st_mtime_ns
            path = run / "state/survey_agent_result.json"
            value = json.loads(path.read_text())
            value.pop("export_version")
            path.write_text(json.dumps(value))
            second = subprocess.run(command + ["resume", "--run-dir", str(run), "--run-id", "refresh", "--cwd", str(PROJECT_ROOT)],
                                    env=env, text=True, capture_output=True)
            self.assertEqual(second.returncode, 0, second.stdout[-2000:] + second.stderr[-2000:])
            self.assertEqual(json.loads(path.read_text())["export_version"], 4)
            self.assertEqual(draft.stat().st_mtime_ns, before)
            evaluation = run / "state/xcientist/evaluation.json"
            evaluation.write_text(json.dumps({"evaluation_version": 2, "error": "provider disconnected", "scores": {}, "reasons": {}}))
            recovered = subprocess.run(command + ["resume", "--run-dir", str(run), "--run-id", "refresh", "--cwd", str(PROJECT_ROOT)],
                                       env=env, text=True, capture_output=True)
            self.assertEqual(recovered.returncode, 0, recovered.stdout[-2000:] + recovered.stderr[-2000:])
            self.assertIsNone(json.loads(evaluation.read_text())["error"])
            self.assertEqual(draft.stat().st_mtime_ns, before)


if __name__ == "__main__":
    unittest.main()
