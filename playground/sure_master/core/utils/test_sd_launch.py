"""Regression checks for SD launch boundaries and mixed-channel AMI audio."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from playground.sure_master.runtime.training_budget import (
    TrainingBudgetPaused, check_run_pause, estimate_training_seconds,
)
from playground.sure_master.tasks.diarization import mono_collate
from playground.sure_master.core.utils.slurm import resource_profile


class SdLaunchTests(unittest.TestCase):
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
