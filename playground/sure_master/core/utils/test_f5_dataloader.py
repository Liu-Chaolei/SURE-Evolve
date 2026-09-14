"""Exercise resampling in fresh workers after parent Torch initialization."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
import wave

from playground.sure_master.runtime.f5_dataloader import dataloader_options


@unittest.skipUnless(os.environ.get("SURE_F5_TEST_SOURCE"), "Requires the F5 image")
class F5DataLoaderTests(unittest.TestCase):
    def test_resampling_workers_after_parent_initialization(self):
        import torch
        from torch.utils.data import DataLoader
        from playground.sure_master.runtime.f5_evolution import prepare_source

        before = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            os.chdir(root)
            try:
                source = prepare_source(Path(os.environ["SURE_F5_TEST_SOURCE"]), root / "source")
                # Reapplying preparation must retain the same protected source.
                from playground.sure_master.runtime.training_sources import prepare_f5_training_source
                trainer = source / "src/f5_tts/model/trainer.py"
                first = trainer.read_bytes()
                prepare_f5_training_source(source)
                self.assertEqual(first, trainer.read_bytes())
                sys.path.insert(0, str(source / "src"))
                from f5_tts.model.dataset import CustomDataset, collate_fn

                rows = []
                for index, rate in enumerate((16000, 48000, 24000, 44100)):
                    audio = root / f"{index}.wav"
                    with wave.open(str(audio), "wb") as stream:
                        stream.setnchannels(1)
                        stream.setsampwidth(2)
                        stream.setframerate(rate)
                        stream.writeframes(b"\x01\x00" * rate)
                    rows.append({"audio_path": str(audio), "text": "测试", "duration": 1.0})
                previous_threads = torch.get_num_threads()
                torch.set_num_threads(4)
                torch.nn.functional.conv1d(torch.randn(2, 1, 48000), torch.randn(16, 1, 32))
                try:
                    dataset = CustomDataset(rows, durations=[1.0] * len(rows))
                    options = {**dataloader_options(2), "timeout": 60}
                    loader = DataLoader(dataset, batch_size=2, num_workers=2,
                                        collate_fn=collate_fn, **options)
                    batches = list(loader)
                    self.assertEqual(sum(len(batch["text"]) for batch in batches), 4)
                    self.assertTrue(all(torch.isfinite(batch["mel"]).all() for batch in batches))
                finally:
                    torch.set_num_threads(previous_threads)
            finally:
                os.chdir(before)


if __name__ == "__main__":
    unittest.main()
