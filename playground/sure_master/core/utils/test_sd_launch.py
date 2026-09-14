"""Regression checks for SD launch boundaries and mixed-channel AMI audio."""

import tempfile
import unittest
import importlib.util
import io
from pathlib import Path
from unittest.mock import Mock

from playground.sure_master.runtime.training_budget import (
    TrainingBudgetPaused, check_run_pause, estimate_training_seconds,
)
from playground.sure_master.tasks.diarization import mono_collate, write_session_rttm
from playground.sure_master.core.utils.slurm import resource_profile


class SdLaunchTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("pyannote"), "pyannote test environment required")
    def test_padded_prediction_is_clipped_without_changing_input_or_speakers(self):
        from pyannote.core import Annotation, Segment

        annotation = Annotation(uri="original")
        annotation[Segment(0.5, 1.1)] = "A"
        annotation[Segment(0.9, 1.148)] = "B"
        annotation[Segment(1.2, 1.3)] = "padding"
        output = io.StringIO()
        write_session_rttm(annotation, {"session_id": "session", "duration": 1.0}, output)
        lines = [line.split() for line in output.getvalue().splitlines()]
        self.assertEqual(len(lines), 2)
        self.assertEqual({line[7] for line in lines}, {"A", "B"})
        self.assertTrue(all(line[1] == "session" for line in lines))
        self.assertTrue(all(float(line[3]) + float(line[4]) <= 1.0 for line in lines))
        self.assertEqual(annotation.uri, "original")
        self.assertEqual(len(annotation), 3)

    def test_mixed_channels_are_selected_before_native_stacking(self):
        labels = Mock()
        expected = {"ts": labels.float.return_value}
        native = Mock(return_value={"ts": labels})
        batch = [([[1, 2]], "a", "mono"), ([[3, 4], [5, 6]], "b", "stereo")]
        self.assertEqual(mono_collate(batch, native_collate=native), expected)
        native.assert_called_once_with(
            [([[1, 2]], "a", "mono"), ([[3, 4]], "b", "stereo")],
            max_speakers_per_chunk=4)

    def test_full_epoch_estimate_includes_validation(self):
        self.assertEqual(estimate_training_seconds(2, 540, 100, 120), 120000)
        with self.assertRaises(ValueError):
            estimate_training_seconds(float("nan"), 540, 100)

    def test_pause_blocks_subsequent_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "candidate"
            check_run_pause(work)
            (Path(tmp) / "metric").mkdir()
            (Path(tmp) / "metric/budget_pause.json").write_text("{}")
            with self.assertRaises(TrainingBudgetPaused):
                check_run_pause(work)

    def test_sd_training_and_replay_resource_counts(self):
        self.assertEqual(resource_profile({}, "arch", adapter="sd.diarizen")["npu"], 4)
        self.assertEqual(resource_profile({}, "inference", adapter="sd.diarizen")["npu"], 1)
