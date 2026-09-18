import unittest
import json
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch

from playground.sure_master.tools.asr_native_controller import compile_architecture
from playground.sure_master.tools.asr_deployment_ops import handoff, resume_old, checkpoint_response


class NativeASRCompilerTest(unittest.TestCase):
    def setUp(self):
        self.arguments = {"num-encoder-layers": "3,2,3,3,3,2", "encoder-dim": "192,256,384,480,384,256",
            "feedforward-dim": "512,768,1152,1440,1152,768", "encoder-unmasked-dim": "192,192,256,288,256,192"}

    def test_reviewed_parameters_reach_wrapper(self):
        code = compile_architecture(self.arguments)
        compile(code, "candidate.py", "exec")
        self.assertIn("train_decode", code)
        self.assertIn("192,256,384,480,384,256", code)
        self.assertNotIn("--start-epoch", code)

    def test_fixed_training_settings_cannot_be_added(self):
        with self.assertRaises(ValueError):
            compile_architecture({**self.arguments, "num-epochs": "2"})

    def test_checkpoint_record_survives_native_library_log_noise(self):
        value = {"checkpoint": "epoch-6.pt", "epoch": 6, "next_epoch": 7, "sha256": "abc", "batch_idx_train": 200}
        text = "library warning\nASR_CHECKPOINT_JSON=" + json.dumps(value) + "\nshutdown warning\n"
        self.assertEqual(checkpoint_response(text), value)
        with self.assertRaises(ValueError):
            checkpoint_response("library warning only")

    def test_unmasked_width_and_resource_bounds(self):
        for key, value in [("encoder-unmasked-dim", "224,192,256,288,256,192"),
                           ("num-encoder-layers", "20,2,3,3,3,2"), ("encoder-dim", "192,256,384,481,384,256")]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                compile_architecture({**self.arguments, key: value})

    def test_handoff_cannot_pause_old_before_four_ideas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "deployment.json").write_text(json.dumps({"old_run": "/unused/old"}))
            (root / "ideas_ready.json").write_text(json.dumps({"status": "ideas_ready", "candidate_count": 3}))
            with patch("playground.sure_master.tools.asr_deployment_ops.pause") as pause:
                with self.assertRaises(ValueError):
                    handoff(root)
                pause.assert_not_called()

    def test_resume_refuses_another_active_deployment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "PAUSED.json").write_text(json.dumps({"workers": []}))
            pointer = root / "current.json"
            pointer.write_text(json.dumps({"run_dir": "/another/run", "status": "running"}))
            with patch("playground.sure_master.tools.asr_deployment_ops.POINTER", pointer):
                with self.assertRaises(ValueError):
                    resume_old(root)
            self.assertTrue((root / "PAUSED.json").exists())

    def test_resume_reconciles_only_paused_receipts_and_keeps_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "exp_10_improve"
            work = workspace / "metric/slurm/original-identity"
            work.mkdir(parents=True)
            checkpoint = workspace / "models/zipformer_candidate/epoch-6.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"verified checkpoint fixture")
            batch = checkpoint.parent / "checkpoint-4000.pt"
            batch.write_bytes(b"unselected batch snapshot")
            archive = root / "archive"
            (archive / workspace.name).mkdir(parents=True)
            (work / "request.json").write_text('{"immutable":true}')
            (work / "allocation.json").write_text('{"status":"finished"}')
            (work / "result.json").write_text('{"success":false}')
            (root / "PAUSED.json").write_text(json.dumps({"name": "ASR-Direct-v1", "archive": str(archive),
                "workers": [{"directory": str(work), "workspace": str(workspace), "pid": 123,
                    "process_identity": "old-start", "resume": {"checkpoint": str(checkpoint),
                    "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}}]}))
            pointer = root / "current.json"
            pointer.write_text(json.dumps({"run_dir": str(root), "status": "paused"}))
            with patch("playground.sure_master.tools.asr_deployment_ops.POINTER", pointer), \
                 patch("playground.sure_master.tools.asr_deployment_ops.process_identity", return_value=None), \
                 patch("playground.sure_master.tools.asr_deployment_ops.subprocess.run") as run, \
                 patch("playground.sure_master.tools.asr_deployment_ops.register"):
                resume_old(root)
            self.assertEqual((work / "request.json").read_text(), '{"immutable":true}')
            self.assertFalse((work / "result.json").exists())
            self.assertTrue(checkpoint.exists())
            self.assertTrue((archive / workspace.name / batch.name).exists())
            self.assertIn("resume_controller.py", str(run.call_args))

    def test_resume_keeps_successful_api_results_and_response_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "archive"
            archive.mkdir()
            receipts = root / "search/workspace/artifacts/xlab_operations.json"
            receipts.parent.mkdir(parents=True)
            interrupted = {"status": "incomplete", "request_digest": "same-request", "operation": "generate"}
            published = {"status": "published", "response": {"score": 0.05}}
            receipts.write_text(json.dumps({"operations": {"round-1": published, "round-2": interrupted}}))
            operation = root / "xlab_operations/round-2"
            operation.mkdir(parents=True)
            (operation / "generation.json").write_text('{"status":"incomplete"}')
            cache = operation / "completed-response.json"
            cache.write_text('{"success":true}')
            (root / "PAUSED.json").write_text(json.dumps({"name": "ASR-Direct-v1", "archive": str(archive),
                "workers": [], "api_operations": {"round-2": interrupted}}))
            pointer = root / "current.json"
            pointer.write_text(json.dumps({"run_dir": str(root), "status": "paused"}))
            with patch("playground.sure_master.tools.asr_deployment_ops.POINTER", pointer), \
                 patch("playground.sure_master.tools.asr_deployment_ops.subprocess.run"), \
                 patch("playground.sure_master.tools.asr_deployment_ops.register"):
                resume_old(root)
            self.assertEqual(json.loads(receipts.read_text())["operations"], {"round-1": published})
            self.assertTrue(cache.exists())
            self.assertFalse((operation / "generation.json").exists())


if __name__ == "__main__":
    unittest.main()
