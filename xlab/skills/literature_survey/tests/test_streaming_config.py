from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from literature_survey_lib.xcientist.config import load_survey_agent_config


class StreamingConfigTests(unittest.TestCase):
    def test_streaming_can_be_enabled_and_explicitly_disabled(self):
        for setting, expected in (("1", True), ("0", False)):
            with tempfile.TemporaryDirectory() as temporary, patch.dict(
                os.environ, {"XLAB_LITERATURE_SURVEY_USE_STREAM": setting}
            ):
                config = load_survey_agent_config(run_dir=temporary, topic="ASR")
                self.assertEqual(config.APIInfo.use_stream_mode, expected)


if __name__ == "__main__":
    unittest.main()
