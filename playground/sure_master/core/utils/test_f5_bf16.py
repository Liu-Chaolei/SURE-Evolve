"""BF16 is an explicit training protocol, distinct from FP32 inference."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from playground.sure_master.core.training import (
    F5_BF16_TRAINING, F5_EVOLUTION_TRAINING, build_training_contract,
    canonical_digest, validate_completion, validate_contract_precision,
    validate_training_config,
)
from playground.sure_master.core.utils.test_official_training import completion
from playground.sure_master.runtime.accelerator import runtime_environment
from playground.sure_master.core.search_scope import execution_contract
from playground.sure_master.tasks.adapters import TtsAdapter


class BF16ContractTests(unittest.TestCase):
    def test_training_precision_is_separate_from_inference(self):
        runtime = {"accelerator": "npu", "world_size": 8,
                   "precision": "fp32", "training_precision": "bf16"}
        validate_training_config("tts.f5tts", F5_BF16_TRAINING, runtime)
        env = runtime_environment(runtime)
        self.assertEqual(env["SURE_PRECISION"], "fp32")
        self.assertEqual(env["SURE_TRAIN_PRECISION"], "bf16")
        with self.assertRaises(ValueError):
            validate_training_config("tts.f5tts", F5_BF16_TRAINING,
                                     {"world_size": 8, "precision": "fp32"})
        with self.assertRaises(ValueError):
            validate_training_config("tts.f5tts", F5_EVOLUTION_TRAINING, runtime)

    def test_fp32_completion_cannot_satisfy_bf16_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "asset"
            artifact.write_text("fixed input")
            fp32_training = {**F5_EVOLUTION_TRAINING, "batch_size_per_gpu": 25600}
            def build(training):
                return build_training_contract("tts.f5tts", training, {},
                                               {"train": artifact}, artifact, {}, "npu")
            before, after = build(fp32_training), build(F5_BF16_TRAINING)
            self.assertEqual(after["precision"], "bf16")
            self.assertEqual(after["master_parameter_precision"], "fp32")
            report = completion("tts.f5tts")
            report.update(contract=before, contract_digest=canonical_digest(before))
            validate_completion(report, before)
            with self.assertRaises(ValueError):
                validate_completion(report, after)
            bad = deepcopy(after)
            bad["precision"] = "fp32"
            with self.assertRaises(ValueError):
                validate_contract_precision(bad)

    def test_free_xlab_entrypoint_remains_available(self):
        sure = {"search_scope": "all", "task": {"training": F5_BF16_TRAINING}}
        contract = execution_contract(TtsAdapter().context(), sure)
        self.assertIn("BF16 training", contract["research_guidance"])
        self.assertIn("training", contract["candidate_parameters"])
        self.assertNotIn("allowed_change_domains", contract)

    def test_bf16_completion_requires_exactly_fifty_epochs(self):
        report = completion("tts.f5tts", epochs=50)
        report["contract"].update(training=deepcopy(F5_BF16_TRAINING), precision="bf16",
                                  precision_mode="autocast", master_parameter_precision="fp32")
        report["contract_digest"] = canonical_digest(report["contract"])
        validate_completion(report)
        for epochs in (49, 100):
            invalid = deepcopy(report)
            invalid.update(epochs_completed=epochs, optimizer_updates=epochs * 10)
            with self.assertRaises(ValueError):
                validate_completion(invalid)


@unittest.skipUnless(os.environ.get("SURE_F5_TEST_SOURCE"), "Requires the F5 image")
class BF16ExecutionTests(unittest.TestCase):
    def test_audit_requires_actual_bf16_execution(self):
        import torch
        from playground.sure_master.runtime.f5_precision import install_precision_audit

        model = torch.nn.Linear(4, 4)
        accelerator = SimpleNamespace(mixed_precision="bf16", scaler=None,
                                      process_index=0, unwrap_model=lambda value: value)
        trainer = SimpleNamespace(model=model, accelerator=accelerator)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            install_precision_audit(trainer, output, {"precision": "bf16", "backend": "cpu"})
            model(torch.ones(2, 4))
            self.assertFalse(trainer.sure_precision_verified)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                model(torch.ones(2, 4))
            self.assertTrue(trainer.sure_precision_verified)
            report = json.loads((output / "precision-rank-0.json").read_text())
            self.assertEqual(report["observed_linear_output_dtype"], "torch.bfloat16")
            self.assertEqual(report["master_parameter_dtypes"], ["torch.float32"])
