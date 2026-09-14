"""Real preparation subprocess regression: workspace imports and original vocab."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave


@unittest.skipUnless(os.environ.get("SURE_F5_TEST_SOURCE"), "Requires the F5 image")
class F5PreparationTests(unittest.TestCase):
    def test_training_preparation_imports_workspace_and_preserves_vocab(self):
        from playground.sure_master.runtime.f5_evolution import prepare_source
        from playground.sure_master.tools.run_official_training import f5_train

        class Prepared(Exception):
            pass

        before = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            os.chdir(root)
            try:
                source = prepare_source(
                    Path(os.environ["SURE_F5_TEST_SOURCE"]), root / "source"
                )
                sys.path.insert(0, str(source / "src"))
                vocab = root / "vocab.txt"
                vocab.write_text("a\nb\n \n")
                audio = root / "audio.wav"
                with wave.open(str(audio), "wb") as stream:
                    stream.setnchannels(1)
                    stream.setsampwidth(2)
                    stream.setframerate(24000)
                    stream.writeframes(b"\x01\x00" * 24000)
                metadata = root / "metadata.csv"
                with metadata.open("w", newline="") as stream:
                    writer = csv.writer(stream, delimiter="|")
                    writer.writerow(["audio_file", "text"])
                    writer.writerow([str(audio), "ab"])
                output = root / "training"
                output.mkdir()
                job = {
                    "source": str(source),
                    "output": str(output),
                    "contract": {"training": {}},
                    "settings": {
                        "resources": {"vocab": str(vocab), "checkpoint": "unused.pt"}
                    },
                    "manifests": {"train_csv": str(metadata)},
                    "structural": False,
                    "architecture": {
                        "dim": 32,
                        "depth": 1,
                        "heads": 2,
                        "dim_head": 16,
                        "text_dim": 32,
                        "conv_layers": 0,
                    },
                }
                with (
                    patch.dict(
                        os.environ,
                        {"PYTHONPATH": str(Path(__file__).resolve().parents[4])},
                    ),
                    patch(
                        "playground.sure_master.tools.run_official_training.load_initial_f5",
                        side_effect=Prepared,
                    ),
                ):
                    with self.assertRaises(Prepared):
                        f5_train(job)
                self.assertEqual(
                    (output / "prepared_data/vocab.txt").read_bytes(),
                    vocab.read_bytes(),
                )
                self.assertTrue((output / "prepared_data/prepared.json").exists())
            finally:
                os.chdir(before)


if __name__ == "__main__":
    unittest.main()
