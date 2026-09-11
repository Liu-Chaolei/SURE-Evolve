"""Official budgets, completion proof, source patches and resource isolation (no model calls)."""

from __future__ import annotations
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from playground.sure_master.core.training import (
    official_training,
    validate_training_config,
    validate_completion,
    canonical_digest,
    REPORT_SCHEMA,
    selected_epochs,
    validation_state,
)
from playground.sure_master.core.artifacts import (
    publish_bundle,
    bundle_resources,
    file_digest,
)
from playground.sure_master.tasks import get_adapter
from playground.sure_master.tasks.training_jobs import training_command
from playground.sure_master.core.utils.slurm import resource_profile, device_gres
from playground.sure_master.tasks.training_resources import (
    validate_prepared_data,
    validate_wavlm_provenance,
)
from playground.sure_master.tools.training_data_integrity import verify_premium_source


def contract(adapter):
    return {
        "adapter": adapter,
        "training": official_training(adapter),
        "component_test": False,
        "backend": "cpu",
    }


def completion(adapter, epochs=None, history=None):
    c = contract(adapter)
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "completed",
        "contract": c,
        "contract_digest": canonical_digest(c),
        "epochs_completed": epochs or 100,
        "optimizer_updates": (epochs or 100) * 10,
        "batches_per_epoch": 10,
        "checkpoint_selection": c["training"]["checkpoint_selection"],
        "stop_reason": "max_epochs",
        "checkpoints": {"final": {"path": "final.pt", "sha256": "unused"}},
    }
    if adapter == "sd.diarizen":
        history = history or [
            {"epoch": i, "loss": 1 / i, "batches": 3} for i in range(1, 101)
        ]
        report["validation_history"] = history
        report["validation_batches_per_epoch"] = 3
        _, patience, early = validation_state(history, 10)
        report.update(
            patience=patience,
            stop_reason="early_stop" if early else "max_epochs",
            selected_epochs=selected_epochs(history),
        )
        report["checkpoints"].update(
            {
                str(i): {"path": f"{i}.pt", "sha256": "unused"}
                for i in report["selected_epochs"]
            }
        )
    return report


class OfficialTrainingTests(unittest.TestCase):
    def test_four_profiles_use_official_epoch_budgets_and_trained_baselines(self):
        repo = Path(__file__).resolve().parents[4]
        for task, adapter, world in [("tts", "tts.f5tts", 1), ("sd", "sd.diarizen", 4)]:
            for backend in ["cuda", "npu"]:
                sure = yaml.safe_load(
                    (
                        repo / f"configs/sure_master/ordinary-{task}-{backend}.yaml"
                    ).read_text()
                )["sure"]
                validate_training_config(
                    adapter, sure["task"]["training"], sure["runtime"]
                )
                self.assertNotIn("max_steps", sure["task"]["training"])
                self.assertEqual(sure["runtime"]["world_size"], world)
                self.assertEqual(
                    get_adapter(task).baseline_candidate_type(sure), "fine_tune"
                )
                if task == "sd":
                    self.assertIn("wavlm", sure["task"]["resources"])
                    self.assertNotIn("model", sure["task"]["resources"])

    def test_short_budget_and_changed_recipe_are_rejected(self):
        for adapter in ["tts.f5tts", "sd.diarizen"]:
            values = official_training(adapter)
            for update in [{"max_steps": 1000}, {"epochs": 99}, {"world_size": 8}]:
                with self.assertRaises(ValueError):
                    validate_training_config(adapter, {**values, **update})

    def test_f5_must_finish_all_epochs_and_updates(self):
        report = completion("tts.f5tts")
        validate_completion(report)
        for update in [
            {"epochs_completed": 1, "optimizer_updates": 10},
            {"optimizer_updates": 999},
            {"stop_reason": "timeout"},
            {"checkpoint_selection": "non_ema"},
        ]:
            with self.assertRaises(ValueError):
                validate_completion({**report, **update})

    def test_sd_early_stop_requires_complete_validation_history(self):
        history = [{"epoch": 1, "loss": 1.0, "batches": 3}] + [
            {"epoch": i, "loss": 2.0, "batches": 3} for i in range(2, 12)
        ]
        report = completion("sd.diarizen", 11, history)
        validate_completion(report)
        self.assertEqual(report["stop_reason"], "early_stop")
        self.assertEqual(report["selected_epochs"], [1, 2, 3, 4, 5])
        for update in [
            {"patience": 9},
            {"validation_history": history[:-1]},
            {"selected_epochs": [7, 8, 9, 10, 11]},
            {"stop_reason": "max_steps"},
        ]:
            with self.assertRaises(ValueError):
                validate_completion({**report, **update})
        with self.assertRaises(ValueError):
            validate_completion(completion("sd.diarizen", 10, history[:-1]))

    def test_components_and_foreign_contracts_cannot_enter_scoring(self):
        report = completion("tts.f5tts")
        report["contract"]["component_test"] = True
        report["contract_digest"] = canonical_digest(report["contract"])
        with self.assertRaisesRegex(ValueError, "component"):
            validate_completion(report)
        with self.assertRaisesRegex(ValueError, "changed"):
            validate_completion(
                completion("tts.f5tts"),
                {**contract("tts.f5tts"), "data": {"train": "changed"}},
            )

    def test_training_launcher_uses_four_sd_ranks_without_step_cap(self):
        for adapter, world in [("tts.f5tts", 1), ("sd.diarizen", 4)]:
            command = training_command(
                Path("/job.json"), adapter, {"training": official_training(adapter)}
            )
            self.assertNotIn("--max-steps", command)
            self.assertEqual("--nproc-per-node=4" in command, world == 4)

    def test_completed_training_retry_never_launches_worker(self):
        from playground.sure_master.tasks import training_jobs

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            evidence = root / "evidence"
            evidence.mkdir()
            report = completion("tts.f5tts")
            (evidence / "final.pt").write_bytes(b"completed weights")
            report["checkpoints"]["final"]["sha256"] = file_digest(
                evidence / "final.pt"
            )
            (evidence / "training_completion.json").write_text(json.dumps(report))
            stack.enter_context(
                patch.dict(
                    "os.environ",
                    {
                        "SURE_TRAIN_MANIFEST": str(root / "train.jsonl"),
                        "SURE_TRAIN_VALIDATION_MANIFEST": str(
                            root / "validation.jsonl"
                        ),
                        "SURE_DATA_PREPARATION": str(root / "preparation.json"),
                        "SURE_TRAIN_WORLD_SIZE": "1",
                        "SURE_PRECISION": "fp32",
                    },
                )
            )
            stack.enter_context(
                patch(
                    "playground.sure_master.tasks.training_resources.validate_prepared_data"
                )
            )
            stack.enter_context(
                patch.object(training_jobs, "training_source_identity", return_value={})
            )
            stack.enter_context(
                patch.object(
                    training_jobs,
                    "build_training_contract",
                    return_value=report["contract"],
                )
            )
            launch = stack.enter_context(patch.object(training_jobs, "run_bounded"))
            settings = {
                "training": {
                    **official_training("tts.f5tts"),
                    "manifest": str(root / "train.csv"),
                },
                "resources": {
                    "checkpoint": str(root / "initial.pt"),
                    "source": str(root),
                },
            }
            actual, directory = training_jobs.run_training(
                "tts.f5tts",
                settings,
                {},
                root,
                SimpleNamespace(name="cpu", torch=SimpleNamespace(__version__="test")),
                root,
            )
            self.assertEqual(actual, report)
            self.assertEqual(directory, evidence)
            launch.assert_not_called()

    def test_slurm_counts_are_task_specific_on_both_backends(self):
        for backend in ["cuda", "npu"]:
            for adapter, count in [
                ("asr.zipformer", 8),
                ("tts.f5tts", 1),
                ("sd.diarizen", 4),
            ]:
                profile = resource_profile(
                    {}, "arch", adapter=adapter, accelerator=backend
                )
                self.assertEqual(profile["npu"], count)
                self.assertIn(str(count), device_gres({}, profile))
                if backend == "cuda":
                    self.assertNotIn("ascend", device_gres({}, profile))
                self.assertEqual(
                    resource_profile({}, "inference", adapter=adapter)["npu"], 1
                )

    def test_evidence_bundle_survives_relocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "candidate"
            root.mkdir()
            evidence = root / "evidence"
            evidence.mkdir()
            checkpoint = evidence / "final.pt"
            checkpoint.write_bytes(b"test weights")
            report = completion("tts.f5tts")
            report["checkpoints"]["final"]["sha256"] = file_digest(checkpoint)
            (evidence / "training_completion.json").write_text(json.dumps(report))
            result = publish_bundle(
                root,
                "tts.f5tts",
                {"training_evidence": evidence, "checkpoint": checkpoint},
            )
            _, resources = bundle_resources(result["model_artifact"])
            self.assertEqual(
                resources["checkpoint"].parent, resources["training_evidence"]
            )
            from playground.sure_master.core.training import require_completion

            require_completion(
                resources["training_evidence"] / "training_completion.json"
            )

    def test_missing_premium_part_cannot_be_full_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "Premium_md5check.txt").write_text(
                "00000000000000000000000000000000 WenetSpeech4TTS_Premium_0.tar.gz\n"
            )
            with self.assertRaisesRegex(FileNotFoundError, "missing Premium archive"):
                verify_premium_source(root, root / "receipt.json")

    def test_matching_partial_text_and_audio_sets_are_not_complete(self):
        import hashlib
        import io
        import tarfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            name = "WenetSpeech4TTS_Premium_0"
            archive = root / (name + ".tar.gz")
            with tarfile.open(archive, "w:gz") as stream:
                for folder, suffix in (("txts", ".txt"), ("wavs", ".wav")):
                    for index in (1, 2):
                        item = tarfile.TarInfo(f"{name}/{folder}/{index}{suffix}")
                        item.size = 4
                        stream.addfile(item, io.BytesIO(b"data"))
            (root / "Premium_md5check.txt").write_text(
                hashlib.md5(archive.read_bytes()).hexdigest()
                + "  "
                + archive.name
                + "\n"
            )
            for folder, suffix in (("txts", ".txt"), ("wavs", ".wav")):
                directory = root / name / folder
                directory.mkdir(parents=True)
                (directory / ("1" + suffix)).write_bytes(b"data")
            with self.assertRaisesRegex(ValueError, "extraction is incomplete"):
                verify_premium_source(root, root / "receipt.json")

    def test_old_prepared_data_and_wrong_ssl_origin_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "preparation.json").write_text('{"source_complete":true}')
            with self.assertRaises(ValueError):
                validate_prepared_data("tts.f5tts", root / "preparation.json", {})
            wavlm = root / "wavlm.bin"
            wavlm.write_bytes(b"fixture")
            wavlm.with_suffix(".bin.provenance.json").write_text(
                json.dumps(
                    {"kind": "diarization_checkpoint", "sha256": file_digest(wavlm)}
                )
            )
            with self.assertRaises(ValueError):
                validate_wavlm_provenance(wavlm)


if __name__ == "__main__":
    unittest.main()
