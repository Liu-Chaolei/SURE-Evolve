from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path

from playground.sure_master.core.artifacts import file_digest
from playground.sure_master.core.search_scope import execution_contract
from playground.sure_master.core.training import (
    F5_DDP8_RECIPE,
    F5_DDP8_TRAINING,
    validate_training_config,
)
from playground.sure_master.tasks.adapters import TtsAdapter
from playground.sure_master.tasks.training_resources import validate_prepared_data
from playground.sure_master.tools.training_data_integrity import verify_extracted_premium


class TtsFormalSearchTests(unittest.TestCase):
    def test_architecture_scope_exposes_only_registered_structure_parameters(self):
        contract = execution_contract(TtsAdapter().context(), {"search_scope": "architecture_only"})
        self.assertEqual(contract["candidate_types"], ["arch"])
        self.assertEqual(contract["allowed_change_domains"], ["arch"])
        self.assertNotIn("checkpoint_activations", contract["candidate_parameters"]["architecture"])
        self.assertIn("long_skip_connection", contract["candidate_parameters"]["architecture"])
        self.assertIn("text_embedding_average_upsampling", contract["candidate_parameters"]["architecture"])

    def test_ddp8_recipe_requires_eight_processes(self):
        config = dict(F5_DDP8_TRAINING)
        self.assertEqual(
            validate_training_config(
                "tts.f5tts", config, {"world_size": 8, "precision": "fp32"}
            )["recipe"],
            F5_DDP8_RECIPE,
        )
        with self.assertRaises(ValueError):
            validate_training_config(
                "tts.f5tts", config, {"world_size": 1, "precision": "fp32"}
            )

    def test_extracted_premium_inventory_is_explicitly_unverified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "Premium_md5check.txt").write_text(
                "0" * 32 + "  WenetSpeech4TTS_Premium_0.tar.gz\n"
            )
            directory = root / "WenetSpeech4TTS_Premium_0"
            (directory / "txts").mkdir(parents=True)
            (directory / "wavs").mkdir()
            sample = "sample_S00001"
            (directory / "txts" / f"{sample}.txt").write_text(f"{sample}\t你好\n")
            with wave.open(str(directory / "wavs" / f"{sample}.wav"), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(b"\0\0" * 16000)
            result = verify_extracted_premium(root)
            self.assertEqual(result["source_mode"], "extracted_only")
            self.assertFalse(result["archive_md5_verified"])
            self.assertEqual(result["utterance_count"], 1)

    def test_extracted_preparation_requires_explicit_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.jsonl"
            manifest.write_text('{"sample_id":"x"}\n')
            preparation = root / "preparation.json"
            preparation.write_text(
                json.dumps(
                    {
                        "schema_version": "sure.prepared_training.v1",
                        "adapter": "tts.f5tts",
                        "source_complete": True,
                        "source_mode": "extracted_only",
                        "digests": {"train": file_digest(manifest)},
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "extracted-only"):
                validate_prepared_data("tts.f5tts", preparation, {"train": manifest})
            validate_prepared_data(
                "tts.f5tts",
                preparation,
                {"train": manifest},
                allow_extracted_only=True,
            )


if __name__ == "__main__":
    unittest.main()
