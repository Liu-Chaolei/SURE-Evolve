"""Fast contract tests: real controller and bundles, fake models/XLab, no training services."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from playground.sure_master.core.artifacts import (
    publish_bundle,
    load_bundle,
    bundle_resources,
)
from playground.sure_master.core.datasets import DatasetSplitSpec, validate_split_groups
from playground.sure_master.core.providers import FakeXlabIdeaProvider
from playground.sure_master.core.utils import test_xlab_controller_smoke as smoke

_CpuFakeSureMaster = smoke._CpuFakeSureMaster
from playground.sure_master.runtime.accelerator import runtime_environment
from playground.sure_master.tasks import get_adapter
from playground.sure_master.tasks.diarization import clip_rttm, validate_rttm_outputs
from playground.sure_master.tools.prepare_task_data import (
    grouped_split,
    premium_rows,
    tts_prompts,
)
from playground.sure_master.tools.run_task_candidate import run as run_candidate
from playground.sure_master.core.utils.code import (
    validate_run_sure_script,
    validate_sure_candidate_boundary,
)


class MixedProvider(FakeXlabIdeaProvider):
    def generate(self, request):
        batch = super().generate(request)
        ideas = [
            replace(idea, candidate_type=kind)
            for idea, kind in zip(
                batch.ideas, ["inference", "fine_tune", "arch", "inference"]
            )
        ]
        return replace(batch, ideas=ideas)


class FakeExp:
    def __init__(self, owner, stage, index):
        self.owner, self.stage, self.index = owner, stage, index
        self.exp_name = f"exp_{index}_{stage}"
        self.workspace_path = str(
            Path(owner.session.config.workspace_path) / self.exp_name
        )
        self.execution_env = owner._execution_env()
        self.metric_feedback = ""
        self.candidate_phase = self.candidate_idea_id = ""
        self.enforce_candidate_type = True

    def run(self, **kwargs):
        root = Path(self.workspace_path)
        root.mkdir(parents=True, exist_ok=True)
        weight = root / "weight.bin"
        weight.write_bytes(f"weights:{self.index}".encode())
        bundle = publish_bundle(
            root,
            self.owner.task_adapter.name,
            {"checkpoint": weight},
            inference_config={"value": self.index},
        )
        self.owner.executed.append(
            (kwargs.get("candidate_type_hint"), dict(self.execution_env), self.index)
        )
        score = 1.0 if self.stage == "draft" else 1 / (self.index + 2)
        if self.owner.inject_failure and len(self.owner.executed) == 3:
            return False, None, None, "# failed", {"reason_code": "execution_failed"}
        return (
            True,
            score,
            None,
            f"# candidate {self.index}",
            {"produced_artifacts": bundle, "reason_code": "success"},
        )

    def run_existing_code(self, **kwargs):
        artifact = self.execution_env["SURE_FROZEN_MODEL_ARTIFACT"]
        load_bundle(artifact)
        self.owner.replays.append((self.candidate_phase, artifact, kwargs["code"]))
        return True, 0.1 + self.index / 10000, None, kwargs["code"], {}


class Controller(_CpuFakeSureMaster):
    def __init__(self, *args, **kwargs):
        self.executed, self.replays = [], []
        self.inject_failure = True
        super().__init__(*args, **kwargs)

    def _setup_agents(self):
        super()._setup_agents()
        self.agents.knowledge_promotion_agent = object()

    def _create_run_exp(self, stage, exp_index):
        return FakeExp(self, stage, exp_index)


class MultiTaskControllerTests(unittest.TestCase):
    def config(self, root, task):
        baseline, ref = root / "baseline.py", root / "ref.txt"
        baseline.write_text("# baseline")
        ref.write_text("u\thello\n")
        path = smoke.ControllerSmokeTests()._write_config(root, baseline, ref)
        config = yaml.safe_load(path.read_text())
        config["sure"].pop("staged_axes")
        config["sure"].update(
            search_strategy="ordinary",
            task_id={"asr": "asr_en_wer", "tts": "tts_zh_cer", "sd": "sd_der"}[task],
            require_base_model=False,
            final_evaluation={"selection_ref": str(ref), "test_ref": str(ref)},
        )
        if task != "asr":
            config["sure"]["base_models"] = {}
        config["max_research_rounds"] = 2
        path.write_text(yaml.safe_dump(config))
        return path

    def test_all_tasks_use_same_round_loop_and_frozen_replay(self):
        for task in ("asr", "tts", "sd"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                provider = MixedProvider()
                controller = Controller(
                    config_path=self.config(root, task), xlab_provider=provider
                )
                controller.set_run_dir(root / "run", task_id=task)
                with patch(
                    "playground.sure_master.core.playground.KnowledgePromotionExp.run",
                    return_value="findings",
                ):
                    result = controller.run("Synthetic contract test")
                self.assertEqual(result["status"], "completed", result)
                self.assertEqual(len(controller.executed), 9)
                self.assertEqual(
                    [len(r.candidates) for r in provider.summarized_results], [4, 4]
                )
                self.assertEqual(
                    provider.summarized_results[0].candidates[1].final_status, "failed"
                )
                self.assertEqual(len(provider.generated_requests[1].prior_rounds), 1)
                for offset in (1, 5):
                    chunk = controller.executed[offset : offset + 4]
                    parents = [
                        env.get("SURE_PARENT_MODEL_ARTIFACT")
                        for kind, env, _ in chunk
                        if kind == "inference"
                    ]
                    self.assertEqual(len(set(parents)), 1)
                    self.assertTrue(parents[0])
                    self.assertTrue(
                        all(
                            "SURE_PARENT_MODEL_ARTIFACT" not in env
                            for kind, env, _ in chunk
                            if kind != "inference"
                        )
                    )
                self.assertEqual(
                    [phase for phase, _, _ in controller.replays[:3]], ["selection"] * 3
                )
                self.assertTrue(
                    all(
                        "--action', 'infer'" in code
                        for _, _, code in controller.replays
                    )
                )
                self.assertEqual(
                    len(provider.summarized_results), 2
                )  # holdout is never research feedback

    def test_staged_and_missing_xlab_fail_before_setup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.config(root, "asr")
            config = yaml.safe_load(path.read_text())
            config["sure"]["search_strategy"] = "staged_axes"
            path.write_text(yaml.safe_dump(config))
            controller = Controller(config_path=path, xlab_provider=MixedProvider())
            with patch.object(controller, "setup") as setup:
                result = controller.run("invalid")
            self.assertIn("retired", result["error"])
            setup.assert_not_called()
            config["sure"]["search_strategy"] = "ordinary"
            config["xlab"]["enabled"] = False
            path.write_text(yaml.safe_dump(config))
            controller = Controller(config_path=path)
            with patch.object(controller, "setup") as setup:
                result = controller.run("invalid")
            self.assertIn("requires the XLab", result["error"])
            setup.assert_not_called()


class BundleTests(unittest.TestCase):
    def test_multiple_resources_survive_cleanup_and_detect_corruption(self):
        import shutil

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "candidate"
            root.mkdir()
            source = root / "source"
            source.mkdir()
            (source / "model.py").write_text("model")
            weights = root / "model.bin"
            weights.write_bytes(b"weights")
            result = publish_bundle(
                root,
                "sd.diarizen",
                {"model": source, "checkpoint": weights},
                inference_config={"threshold": 0.6},
            )
            shutil.rmtree(root)
            manifest, resources = bundle_resources(result["model_artifact"])
            self.assertEqual(manifest["inference_config"]["threshold"], 0.6)
            self.assertTrue((resources["model"] / "model.py").is_file())
            resources["checkpoint"].write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                load_bundle(result["model_artifact"])

    def test_inference_settings_change_bundle_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "candidate"
            root.mkdir()
            weight = root / "weight"
            weight.write_text("same")
            a = publish_bundle(
                root, "tts.f5tts", {"checkpoint": weight}, inference_config={"speed": 1}
            )
            b = publish_bundle(
                root,
                "tts.f5tts",
                {"checkpoint": weight},
                inference_config={"speed": 0.9},
            )
            self.assertNotEqual(a["model_digest"], b["model_digest"])

    def test_untrusted_resource_paths_rejected(self):
        from playground.sure_master.core.artifacts import safe_relative

        with tempfile.TemporaryDirectory() as temporary:
            for path in ("../secret", "/tmp/secret"):
                with self.assertRaises(ValueError):
                    safe_relative(Path(temporary), path)

    def test_frozen_replay_cannot_train_or_override_parameters(self):
        with patch.dict(
            "os.environ",
            {"SURE_TASK_ADAPTER": "tts.f5tts", "SURE_FROZEN_MODEL_ARTIFACT": "ignored"},
        ):
            with self.assertRaisesRegex(ValueError, "Frozen replay"):
                run_candidate("fine_tune", {})
            with self.assertRaisesRegex(ValueError, "Frozen replay"):
                run_candidate("infer", {"inference": {"speed": 0.8}})


class DataAndRuntimeTests(unittest.TestCase):
    def test_group_splits_are_disjoint_and_deterministic(self):
        data = [
            {"group_id": str(g), "sample_id": f"{g}-{i}"}
            for g in range(100)
            for i in range(2)
        ]
        a = grouped_split(
            data,
            {"train": 0.9, "train_validation": 0.02, "search": 0.04, "selection": 0.04},
        )
        self.assertEqual(
            a,
            grouped_split(
                data,
                {
                    "train": 0.9,
                    "train_validation": 0.02,
                    "search": 0.04,
                    "selection": 0.04,
                },
            ),
        )
        self.assertEqual([len(v) for v in a.values()], [180, 4, 8, 8])
        self.assertEqual(len({r["sample_id"] for v in a.values() for r in v}), 200)

    def test_missing_premium_download_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError, "not ready"):
                premium_rows(Path(temporary), Path(temporary) / "groups.json")

    def test_split_manifest_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("train", "holdout"):
                (root / f"{name}.jsonl").write_text(
                    json.dumps({"sample_id": name, "group_id": "same"}) + "\n"
                )
            specs = {
                name: DatasetSplitSpec(name, str(root / f"{name}.jsonl"))
                for name in ("train", "holdout")
            }
            with self.assertRaisesRegex(ValueError, "overlaps"):
                validate_split_groups(specs)

    def test_sd_clips_both_boundaries_and_rejects_missing_sessions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.jsonl"
            hyp = root / "hyp.rttm"
            manifest.write_text(
                json.dumps({"session_id": "s", "duration": 10, "uem": [[2, 4], [6, 8]]})
                + "\n"
            )
            hyp.write_text("SPEAKER s 1 1 8 <NA> <NA> speaker <NA> <NA>\n")
            (root / "processed_sessions.json").write_text('["s"]')
            validate_rttm_outputs(hyp, manifest)
            clip_rttm(hyp, manifest, root / "scored.rttm")
            self.assertEqual(
                [
                    line.split()[3:5]
                    for line in (root / "scored.rttm").read_text().splitlines()
                ],
                [["2.000000", "2.000000"], ["6.000000", "2.000000"]],
            )
            (root / "processed_sessions.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "cover exactly"):
                validate_rttm_outputs(hyp, manifest)

    def test_device_environment_has_no_asr_dependencies(self):
        for backend in ("cuda", "npu"):
            env = runtime_environment(
                {"accelerator": backend, "devices": ["0"], "python": "worker"}
            )
            self.assertEqual(env["SURE_WORKER_PYTHON"], "worker")
            self.assertFalse(any("ASR" in k or "ICEFALL" in k for k in env))
        with self.assertRaisesRegex(ValueError, "accelerator"):
            runtime_environment({"accelerator": "invalid"})

    def test_generic_baseline_and_replay_pass_script_boundary(self):
        for task in ("asr", "tts", "sd"):
            code = get_adapter(task).frozen_code("/safe/model/manifest.json")
            self.assertEqual(
                validate_run_sure_script(code, {"hyp": "artifacts/hyp.txt"}), []
            )
            self.assertEqual(
                validate_sure_candidate_boundary(
                    code, "/external/sure-eval", canonical_task=task
                ),
                [],
            )


class BackendPatchTests(unittest.TestCase):
    def test_f5_namespace_patch_compiles_and_is_portable(self):
        import ast
        import os
        from playground.sure_master.runtime.model_source import prepare_f5_source

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "src/f5_tts"
            (package / "infer").mkdir(parents=True)
            (package / "model").mkdir()
            (package / "infer/utils_infer.py").write_text("import os\n")
            (package / "model/trainer.py").write_text("value = dict(fused=True)\n")
            (package / "model/modules.py").write_text(
                "class MelSpec:\n    def forward(self, waveform):\n        return waveform\n"
            )
            previous = Path.cwd()
            try:
                os.chdir(root)
                prepare_f5_source(root)
                content = {str(p): p.read_text() for p in root.rglob("*.py")}
                prepare_f5_source(root)
                for p in root.rglob("*.py"):
                    ast.parse(p.read_text())
                    self.assertEqual(content[str(p)], p.read_text())
                self.assertIn(
                    "if _sure_os.environ.get('SURE_ACCELERATOR') == 'npu'",
                    (package / "__init__.py").read_text(),
                )
                self.assertIn("fused=False", (package / "model/trainer.py").read_text())
            finally:
                os.chdir(previous)

    def test_bounded_process_timeout(self):
        import subprocess
        import sys
        from playground.sure_master.runtime.process import run_bounded

        with tempfile.TemporaryFile(mode="w+") as output:
            with self.assertRaises(subprocess.TimeoutExpired):
                run_bounded(
                    [sys.executable, "-c", "import time; time.sleep(10)"],
                    timeout=0.05,
                    output=output,
                )


class IsolatedScoringTests(unittest.TestCase):
    def test_scoring_uses_separate_interpreter_and_rejects_nonfinite_scores(self):
        import sys
        from playground.sure_master.core.utils.metric import SureMetricRunner
        from playground.sure_master.core.utils.task_cards import SureTaskCard

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "src/sure_eval/evaluation"
            package.mkdir(parents=True)
            (root / "ref.txt").write_text("u\thello\n")
            (root / "hyp.txt").write_text("u\thello\n")
            stub = (
                "from pathlib import Path\n"
                "def build_pipeline_spec(task, **kwargs): return {'task': task}\n"
                "def run_pipeline_spec(pipeline, *, device, ref_file, hyp_file, **kwargs):\n"
                "    assert device == 'cpu'\n"
                "    assert Path(ref_file).read_text() == Path(hyp_file).read_text()\n"
                "    return {'score': SCORE, 'metric': 'WER'}\n"
            )
            card = SureTaskCard(
                task_id="asr_en_wer",
                canonical_task="asr",
                task_alias="asr",
                primary_metric="WER",
                required_roles=["ref", "hyp"],
            )
            for value, expected in (("0.0", True), ("float('nan')", False)):
                (package / "cli_adapters.py").write_text(stub.replace("SCORE", value))
                runner = SureMetricRunner(root, device="cpu", python=sys.executable)
                result = runner.run(
                    card,
                    root,
                    root / "metric",
                    {"ref": str(root / "ref.txt"), "hyp": str(root / "hyp.txt")},
                )
                self.assertEqual(result.success, expected, result.error)
                if expected:
                    self.assertEqual(result.score, 0)
                    self.assertTrue((root / "metric/score_result.json").is_file())
                else:
                    self.assertIn("nonfinite", result.error)


if __name__ == "__main__":
    unittest.main()
