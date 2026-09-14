"""Tiny real F5 numerical tests; run in the F5 image with SURE_F5_TEST_SOURCE."""

from __future__ import annotations

import os
import json
import wave
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(
    os.environ.get("SURE_F5_TEST_SOURCE"), "Requires the F5 worker image and source"
)
class F5RuntimeTests(unittest.TestCase):
    def test_real_trainer_counts_complete_epochs_and_rejects_short_completion(self):
        import torch
        from playground.sure_master.runtime.f5_evolution import prepare_source
        from playground.sure_master.runtime.official_trainers import (
            F5TrainingSession,
            f5_trainer_class,
        )
        from playground.sure_master.core.training import F5_TRAINING

        before = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            os.chdir(root)
            try:
                source = prepare_source(
                    Path(os.environ["SURE_F5_TEST_SOURCE"]), root / "f5"
                )
                sys.path.insert(0, str(source / "src"))
                from f5_tts.model import CFM, DiT, Trainer
                from f5_tts.model.dataset import DynamicBatchSampler

                torch.set_num_threads(1)
                torch.manual_seed(42)

                class Data(torch.utils.data.Dataset):
                    def __len__(self):
                        return 3

                    def __getitem__(self, index):
                        return {
                            "mel_spec": torch.ones(100, 8) * (index + 1) / 10,
                            "text": "ab",
                        }

                    def get_frame_len(self, index):
                        return 8

                data = Data()
                with patch.dict(os.environ, {"WORLD_SIZE": "8"}):
                    sampler = DynamicBatchSampler(
                        torch.utils.data.SequentialSampler(data), 4
                    )
                    batches = list(sampler)
                    self.assertEqual(len(batches) % 8, 0)
                    self.assertEqual({i for batch in batches for i in batch}, {0, 1, 2})
                model = CFM(
                    transformer=DiT(
                        dim=32,
                        depth=1,
                        heads=2,
                        dim_head=16,
                        text_dim=32,
                        conv_layers=0,
                        text_num_embeds=2,
                    ),
                    vocab_char_map={"a": 0, "b": 1},
                )
                trainer = f5_trainer_class(Trainer)(
                    model,
                    2,
                    1e-5,
                    num_warmup_updates=1,
                    batch_size_type="sample",
                    batch_size_per_gpu=1,
                    logger=None,
                    log_samples=False,
                    checkpoint_path=str(root / "state"),
                    save_per_updates=50000,
                    last_per_updates=5000,
                    accelerate_kwargs={"cpu": True},
                )
                session = F5TrainingSession(
                    trainer,
                    root / "training",
                    {
                        "adapter": "tts.f5tts",
                        "training": dict(F5_TRAINING),
                        "backend": "cpu",
                        "component_test": False,
                    },
                )
                trainer.sure_training = session
                from playground.sure_master.runtime.f5_validation import F5Validation

                with wave.open(str(root / "validation.wav"), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(24000)
                    wav.writeframes(b"\x01\x00" * 24000)
                validation = root / "validation.jsonl"
                validation.write_text(
                    json.dumps(
                        {
                            "sample_id": "v",
                            "audio": str(root / "validation.wav"),
                            "text": "ab",
                            "duration": 1.0,
                        }
                    )
                    + "\n"
                )
                trainer.sure_validation = F5Validation(trainer, validation, {}, "cpu")
                rng = torch.get_rng_state().clone()
                self.assertEqual(trainer.sure_validation()["samples"], 1)
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))
                trainer.train(data, num_workers=0, resumable_with_seed=666)
                self.assertEqual(session.progress["epoch"], 2)
                self.assertEqual(session.progress["updates"], 6)
                self.assertEqual(len(session.progress["validation_history"]), 2)
                with self.assertRaisesRegex(ValueError, "100 epochs"):
                    session.finish()
                self.assertFalse(
                    (root / "training/evidence/training_completion.json").exists()
                )
            finally:
                os.chdir(before)


if __name__ == "__main__":
    unittest.main()
