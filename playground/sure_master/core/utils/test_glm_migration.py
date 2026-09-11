"""GLM routing and baseline-only migration regression tests."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from playground.sure_master.tools.with_xlab_environment import zai_environment
from playground.sure_master.tools.zai_preflight import check_zai_api
from playground.sure_master.core.baseline_import import import_baseline


class GlmEnvironmentTests(unittest.TestCase):
    def test_role_mapping_overrides_old_provider_without_exporting_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "ZAI_API_KEY=fixture-zai\nZAI_BASE_URL=https://example.com/v1\n"
            )
            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "old",
                    "SURE_AGENT_MODEL": "old",
                    "XLAB_RESEARCH_IDEA_AGENT_MODEL": "old",
                },
            ):
                env = zai_environment(path)
            self.assertEqual(env["OPENAI_API_KEY"], "fixture-zai")
            self.assertEqual(env["OPENAI_BASE_URL"], "https://example.com/v1")
            self.assertEqual(env["SURE_AGENT_MODEL"], "glm-5.3-flash")
            for role in ("AGENT", "GENERATION", "EVALUATION", "FUSION"):
                self.assertEqual(
                    env[f"XLAB_RESEARCH_IDEA_{role}_MODEL"],
                    "glm-5.3" if role in {"AGENT", "GENERATION"} else "glm-5.3-flash",
                )

    def test_missing_zai_key_never_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("ZAI_BASE_URL=https://example.com/v1\n")
            with (
                patch.dict(os.environ, {"OPENAI_API_KEY": "old"}),
                self.assertRaises(ValueError),
            ):
                zai_environment(path)

    def test_denied_models_stop_before_streaming_and_redact_key(self):
        with patch("openai.OpenAI") as client:
            client.return_value.chat.completions.create.side_effect = RuntimeError(
                "rejected fixture-secret"
            )
            result = check_zai_api(
                {
                    "ZAI_API_KEY": "fixture-secret",
                    "ZAI_BASE_URL": "https://example.com/v1",
                }
            )
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(len(result["checks"]), 2)
        self.assertNotIn("fixture-secret", json.dumps(result))
        self.assertEqual(
            [r["model"] for r in result["checks"]], ["glm-5.3", "glm-5.3-flash"]
        )


class BaselineImportTests(unittest.TestCase):
    def test_reuse_and_reject_changed_training_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old"
            workspace = old / "search/workspace"
            (workspace / "metric").mkdir(parents=True)
            ref = workspace / "exp_1_draft/input/ref.txt"
            ref.parent.mkdir(parents=True)
            ref.write_text("id\tTHE CAT SAT\n")
            data = root / "data"
            (data / "lang_bpe_500").mkdir(parents=True)
            bpe = data / "lang_bpe_500/bpe.model"
            bpe.write_bytes(b"fixed-tokenizer")
            artifact = root / "model/manifest.json"
            artifact.parent.mkdir()
            (artifact.parent / "artifacts").mkdir()
            (artifact.parent / "artifacts/candidate_changes.json").write_text(
                json.dumps({"training_config": {"actual_train_epoch": 10}})
            )
            code = root / "baseline.py"
            code.write_text('print("baseline")\n')
            sure = {
                "task_id": "asr_en_wer",
                "execution_env": {
                    "SURE_MAX_TRAIN_EPOCHS": "10",
                    "SURE_ASR_FIXED_SEED": "42",
                },
                "execution_contract": {"training_epochs": 10},
                "initial_solution_path": str(code),
                "inputs": {"ref": str(ref)},
                "base_models": {"asr_en_wer": {"source_paths": {"data": str(data)}}},
            }
            (old / "execution.yaml").write_text(yaml.safe_dump({"sure": sure}))
            baseline = {
                "score": 0.28,
                "code": code.read_text(),
                "model_artifact": str(artifact),
            }
            state = {
                "completed_rounds": 0,
                "candidates": [],
                "baseline": baseline,
                "attributes": {
                    "best_model_artifact": {"model_artifact": str(artifact)}
                },
            }
            (workspace / "metric/controller_state.json").write_text(json.dumps(state))
            with patch(
                "playground.sure_master.core.baseline_import.bundle_resources",
                return_value=({"adapter": "asr.zipformer"}, {"tokenizer": bpe}),
            ):
                imported, model = import_baseline(old, sure, root / "new")
                self.assertEqual(imported, baseline)
                self.assertEqual(model["model_artifact"], str(artifact))
                report = json.loads(
                    (root / "new/metric/baseline_import.json").read_text()
                )
                self.assertFalse(report["training_performed"])
                sure["execution_env"]["SURE_MAX_TRAIN_EPOCHS"] = "30"
                with self.assertRaisesRegex(ValueError, "SURE_MAX_TRAIN_EPOCHS"):
                    import_baseline(old, sure, root / "full_budget")
                sure["execution_env"]["SURE_MAX_TRAIN_EPOCHS"] = "10"
                sure["execution_env"]["SURE_ASR_FIXED_SEED"] = "7"
                with self.assertRaisesRegex(ValueError, "SURE_ASR_FIXED_SEED"):
                    import_baseline(old, sure, root / "other")


if __name__ == "__main__":
    unittest.main()
