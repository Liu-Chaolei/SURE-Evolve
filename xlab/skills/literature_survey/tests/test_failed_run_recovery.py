import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_literature_survey import FAKE_LLM_ENV, PROJECT_ROOT, write_graph_fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from literature_survey_lib.config import load_runtime_config
from literature_survey_lib.inputs import parse_request_args
from literature_survey_lib.pipeline import run_pipeline


class FailedRunRecoveryTest(unittest.TestCase):
    def test_successful_resume_clears_current_failure_but_keeps_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, FAKE_LLM_ENV):
            base = Path(temporary)
            graph = write_graph_fixture(base)
            run = base / "run"
            runtime = load_runtime_config()
            request = parse_request_args(f'"Scientific agents" --graph "{graph}"', PROJECT_ROOT, runtime)
            with patch("literature_survey_lib.pipeline.run_survey_agent", side_effect=ValueError("fixture provider failure")):
                failed = run_pipeline(cwd=PROJECT_ROOT, run_dir=run, run_id="recovery", request=request, runtime=runtime)
            self.assertEqual(failed["status"], "incomplete")
            recovered = run_pipeline(cwd=PROJECT_ROOT, run_dir=run, run_id="recovery", request=request, runtime=runtime)
            self.assertEqual(recovered["status"], "success", recovered["report"].get("blocking_errors"))
            self.assertIn("fixture provider failure", (run / "logs/diagnostics.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
