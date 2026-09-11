"""Regression tests for the production ASR execution and comparison contracts."""

import gzip
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from playground.sure_master.core.utils.slurm import (
    batch_script,
    resource_profile,
    job_state,
    run_candidate,
)
from playground.sure_master.runtime.accelerator import runtime_environment
from playground.sure_master.runtime.data_view import evaluation_data_view
from playground.sure_master.core.full_training import training_plan
from playground.sure_master.baselines.asr_profiles import RECIPE_PROFILES


class EvolutionTests(unittest.TestCase):
    def test_runtime_preserves_scheduler_device_mapping(self):
        with patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "3,5"}, clear=True):
            env = runtime_environment(
                {"accelerator": "npu", "world_size": 8, "devices": list(range(8))}
            )
            self.assertNotIn("ASCEND_RT_VISIBLE_DEVICES", env)
            self.assertEqual(os.environ["ASCEND_RT_VISIBLE_DEVICES"], "3,5")

    def test_slurm_profiles_and_container_boundary(self):
        settings = {"image": "registry.cluster.local:5000/users/u/train@sha256:abc"}
        train = resource_profile(settings, "arch")
        infer = resource_profile(settings, "inference")
        self.assertEqual((train["npu"], infer["npu"]), (8, 1))
        script = batch_script(
            settings,
            Path("/shared/user/run/request.json"),
            Path("/shared/user/run"),
            train,
        )
        self.assertIn("sudo -n slurm-docker-run", script)
        self.assertNotIn("ASCEND_RT_VISIBLE_DEVICES", script)
        self.assertNotIn("docker run", script)
        self.assertIn("--gres=gpu:ascend910b3:8", script)

    def test_accounting_disabled_uses_controller_state(self):
        with patch(
            "playground.sure_master.core.utils.slurm.command",
            side_effect=["", "JobId=7 JobState=TIMEOUT ExitCode=0:15"],
        ) as cmd:
            self.assertEqual(job_state("7"), "TIMEOUT")
            self.assertEqual(cmd.call_count, 2)

    def test_search_data_view_excludes_other_eval_sets_and_refs(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            data = root / "data"
            (data / "fbank").mkdir(parents=True)
            (data / "refs").mkdir()
            for split in ("train", "dev", "test"):
                with gzip.open(
                    data / "fbank" / f"tedlium_cuts_{split}.jsonl.gz", "wt"
                ) as f:
                    f.write(
                        json.dumps({"id": "u", "supervisions": [{"id": "u"}]}) + "\n"
                    )
            ref = root / "ref"
            ref.write_text("u\tTEXT\n")
            view = evaluation_data_view(data, root / "view", ref, ["dev"])
            self.assertTrue((view / "fbank/tedlium_cuts_train.jsonl.gz").exists())
            self.assertFalse((view / "fbank/tedlium_cuts_test.jsonl.gz").exists())
            self.assertFalse((view / "refs").exists())
            self.assertTrue((data / "fbank/tedlium_cuts_test.jsonl.gz").exists())

    def test_full_training_resolves_inference_parent_without_reusing_weights(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)

            def artifact(name, changes):
                base = root / name
                (base / "artifacts").mkdir(parents=True)
                (base / "recipe").mkdir()
                (base / "epoch-10.pt").write_bytes(b"opaque fixture")
                (base / "bpe.model").write_bytes(b"tokenizer")
                (base / "manifest.json").write_text(
                    json.dumps(
                        {
                            "schema_version": "sure.model_artifact.v1",
                            "checkpoint": "epoch-10.pt",
                            "bpe_model": "bpe.model",
                        }
                    )
                )
                (base / "artifacts/candidate_changes.json").write_text(
                    json.dumps(changes)
                )
                return str(base / "manifest.json")

            training = {
                "training_config": {"actual_train_epoch": 10},
                "inference_config": {
                    "decoding_method": "greedy_search",
                    "decode_avg": 1,
                    "use_averaged_model": "0",
                },
                "diff_from_defaults": {
                    "train_extra_args": ["--base-lr", "0.03"],
                    "decode_extra_args": [],
                },
            }
            parent = artifact("train", training)
            child = artifact(
                "infer",
                {
                    **training,
                    "training_config": {"actual_train_epoch": None},
                    "parent_model_artifact": parent,
                },
            )
            plan = training_plan(child)
            self.assertEqual(plan["training_artifact"], parent)
            self.assertEqual(plan["train_args"], ["--base-lr", "0.03"])
            self.assertNotIn("checkpoint", plan)

    def test_completed_slurm_result_is_not_resubmitted(self):
        with tempfile.TemporaryDirectory() as t:
            workspace = Path(t) / "exp"
            (workspace / "metric").mkdir(parents=True)
            context = {
                "execution_env": {},
                "role_paths": {},
                "stage": "improve",
                "candidate_type_hint": "arch",
            }
            (workspace / "metric/remote_candidate_context.json").write_text(
                json.dumps(context)
            )
            exp = SimpleNamespace(
                config={
                    "sure": {
                        "execution_mode": "slurm",
                        "slurm": {"image": "r@sha256:abc"},
                    }
                },
                workspace_path=str(workspace),
                candidate_type_hint="arch",
                code="print(1)",
            )

            def call(argv):
                self.assertEqual(argv[0], "sbatch")
                request = next((workspace / "metric/slurm").glob("*/request.json"))
                payload = json.loads(request.read_text())
                Path(payload["result"]).write_text(
                    json.dumps({"success": True, "score": 0.2})
                )
                return "123"

            with patch(
                "playground.sure_master.core.utils.slurm.command", side_effect=call
            ) as cmd:
                self.assertEqual(run_candidate(exp)["score"], 0.2)
                self.assertEqual(run_candidate(exp)["score"], 0.2)
                self.assertEqual(cmd.call_count, 1)

    def test_native_profile_does_not_change_large_profile(self):
        self.assertIn(
            "192,256,384,512,384,256",
            RECIPE_PROFILES["tedlium3_zipformer_native"].arch_args,
        )
        self.assertIn(
            "192,256,512,768,512,256", RECIPE_PROFILES["tedlium3_zipformer"].arch_args
        )


class SearchControllerTests(unittest.TestCase):
    def test_six_round_stagnation_and_restart_do_not_retrain_baseline(self):
        import yaml
        from playground.sure_master.core.utils.test_xlab_controller_smoke import (
            ControllerSmokeTests,
            _CpuFakeSureMaster,
        )
        from playground.sure_master.core.providers import FakeXlabIdeaProvider

        class Harness(_CpuFakeSureMaster):
            def _setup_agents(self):
                super()._setup_agents()
                self.agents.knowledge_promotion_agent = object()

        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            baseline = root / "baseline.py"
            ref = root / "ref.txt"
            baseline.write_text("# fixture")
            ref.write_text("u\tTEXT\n")
            path = ControllerSmokeTests()._write_config(root, baseline, ref)
            config = yaml.safe_load(path.read_text())
            config["sure"].pop("staged_axes")
            config["sure"].update(
                search_strategy="ordinary",
                controller_timeout_seconds=0,
                search_budget={"min_rounds": 6, "max_rounds": 10, "patience": 3},
            )
            for directory in ("base-recipe", "base-data", "base-root"):
                (root / directory).mkdir()
            path.write_text(yaml.safe_dump(config))
            provider = FakeXlabIdeaProvider()
            controller = Harness(config_path=path, xlab_provider=provider)
            # Fixtures are intentionally synthetic; no real model or scoring claims.
            with patch(
                "playground.sure_master.core.playground.KnowledgePromotionExp"
            ) as knowledge:
                knowledge.return_value.exp_name = "knowledge"
                knowledge.return_value.run.return_value = "observations"
                result = controller.run("test fixed search")
            self.assertEqual(result["status"], "completed", result)
            self.assertEqual(len(provider.generated_requests), 6)
            state = json.loads(
                (root / "workspace/metric/controller_state.json").read_text()
            )
            self.assertEqual(state["completed_rounds"], 6)
            self.assertEqual(len(state["candidates"]), 24)
            second_provider = FakeXlabIdeaProvider()
            restarted = Harness(config_path=path, xlab_provider=second_provider)
            with patch.object(
                restarted,
                "_create_run_exp",
                side_effect=AssertionError("Must reuse completed baseline/search"),
            ):
                result = restarted.run("test fixed search")
            self.assertEqual(result["status"], "completed", result)
            self.assertEqual(second_provider.generated_requests, [])


class FullTrainingControllerTests(unittest.TestCase):
    def test_baseline_and_finalists_use_fresh_full_budget_and_one_parallel_batch(self):
        from playground.sure_master.core.full_training import retrain_selected

        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            dispatched = []

            class Experiment:
                def __init__(self, index):
                    self.exp_name = f"exp_{index}"
                    self.execution_env = {"SURE_PARENT_MODEL_ARTIFACT": "subset-parent"}

                def run_existing_code(self, **kwargs):
                    dispatched.append((self.execution_env.copy(), kwargs))
                    return (
                        True,
                        0.2,
                        "id",
                        kwargs["code"],
                        {
                            "produced_artifacts": {
                                "model_artifact": str(
                                    root / self.exp_name / "manifest.json"
                                )
                            }
                        },
                    )

            class Controller:
                exp_index = 20
                sure_config = {
                    "full_training": {
                        "enabled": True,
                        "data": "/shared/full",
                        "epochs": 30,
                    }
                }
                task_card = SimpleNamespace(is_lower_better=True)
                session = SimpleNamespace(
                    config=SimpleNamespace(workspace_path=str(root))
                )

                def _is_valid_score(self, s):
                    return s is not None

                def _create_run_exp(self, stage, index):
                    return Experiment(index)

                def _role_paths(self):
                    return {"ref": "/shared/regular.txt"}

                def execute_parallel_tasks(self, jobs, **kwargs):
                    self.workers = kwargs["max_workers"]
                    return [job() for job in jobs]

            controller = Controller()
            plan = {
                "recipe": "/shared/frozen-recipe",
                "train_args": [],
                "decode_args": [],
                "inference": {
                    "decoding_method": "greedy_search",
                    "decode_avg": 1,
                    "use_averaged_model": "0",
                },
            }
            baseline = {
                "idea_id": "baseline",
                "score": 0.5,
                "model_artifact": "baseline",
            }
            candidates = [
                {"idea_id": str(i), "score": s, "model_artifact": str(i)}
                for i, s in enumerate((0.4, 0.3, 0.2))
            ]
            with patch(
                "playground.sure_master.core.full_training.training_plan",
                side_effect=lambda artifact: {
                    **plan,
                    "train_args": [
                        "--base-lr",
                        "0.04" if artifact == "baseline" else "0.0" + artifact,
                    ],
                },
            ):
                full_baseline, finalists = retrain_selected(
                    controller, baseline, candidates
                )
            self.assertEqual(controller.workers, 3)
            self.assertEqual([r["idea_id"] for r in finalists], ["2", "1"])
            self.assertEqual(full_baseline["idea_id"], "baseline")
            for env, args in dispatched:
                self.assertEqual(env["SURE_MAX_TRAIN_EPOCHS"], "30")
                self.assertNotIn("SURE_PARENT_MODEL_ARTIFACT", env)
                self.assertEqual(
                    args["base_model_source_overrides"]["data"], "/shared/full"
                )
                self.assertNotIn("--model-artifact", args["code"])
                self.assertEqual(args["role_paths"]["ref"], "/shared/regular.txt")

            # Two decoding variants of the same training recipe must share training.
            dispatched.clear()
            controller = Controller()
            with patch(
                "playground.sure_master.core.full_training.training_plan",
                return_value=plan,
            ):
                _, shared = retrain_selected(controller, baseline, candidates)
            kinds = [args["candidate_type_hint"] for _, args in dispatched]
            self.assertEqual(kinds.count("fine_tune"), 1)
            self.assertEqual(kinds.count("inference"), 2)
            self.assertTrue(
                all(record["shared_training_with"] == "baseline" for record in shared)
            )


class RestartRuntimeTests(unittest.TestCase):
    def test_removed_job_local_data_is_relinked_on_restart(self):
        from playground.sure_master.baselines import (
            zipformer_large_cr_ctc_rnnt_baseline as baseline,
        )

        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            for name in ("recipe", "source-data", "icefall"):
                (root / name).mkdir()
            (root / "data").symlink_to(root / "deleted-job-cache")
            values = {
                "RECIPE_DIR": root / "recipe",
                "SOURCE_RECIPE_DIR": root / "recipe",
                "DATA_DIR": root / "source-data",
                "ICEFALL_ROOT": root / "icefall",
                "MODELS_DIR": root / "models",
                "ARTIFACTS_DIR": root / "artifacts",
                "WORKING_DIR": root / "working",
            }
            previous = Path.cwd()
            try:
                os.chdir(root)
                with (
                    patch.multiple(baseline, **values),
                    patch.dict(os.environ, {}, clear=True),
                ):
                    baseline.ensure_workspace()
                self.assertEqual((root / "data").resolve(), root / "source-data")
            finally:
                os.chdir(previous)

    def test_zero_subprocess_timeout_means_no_deadline(self):
        import sys
        from playground.sure_master.baselines import (
            zipformer_large_cr_ctc_rnnt_baseline as baseline,
        )

        with tempfile.TemporaryDirectory() as t:
            with patch.multiple(baseline, WORKING_DIR=Path(t), WORKSPACE=Path(t)):
                log = baseline.run_command(
                    "fixture",
                    [
                        sys.executable,
                        "-c",
                        'import time; time.sleep(0.05); print("completed")',
                    ],
                    timeout=0,
                )
            self.assertIn("completed", log.read_text())


class ExtendedSearchTests(unittest.TestCase):
    def test_continuing_improvement_reaches_ten_round_cap(self):
        import yaml
        from playground.sure_master.core.utils.test_xlab_controller_smoke import (
            ControllerSmokeTests,
            _CpuFakeSureMaster,
        )
        from playground.sure_master.core.providers import FakeXlabIdeaProvider

        class Harness(_CpuFakeSureMaster):
            def _setup_agents(self):
                super()._setup_agents()
                self.agents.knowledge_promotion_agent = object()

            def _create_run_exp(self, stage, index):
                exp = super()._create_run_exp(stage, index)
                exp._score = lambda: 1 / (index + 1)
                return exp

        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            baseline = root / "baseline.py"
            ref = root / "ref.txt"
            baseline.write_text("# fixture")
            ref.write_text("u\tTEXT\n")
            path = ControllerSmokeTests()._write_config(root, baseline, ref)
            config = yaml.safe_load(path.read_text())
            config["sure"].pop("staged_axes")
            config["sure"].update(
                search_strategy="ordinary",
                controller_timeout_seconds=0,
                search_budget={"min_rounds": 6, "max_rounds": 10, "patience": 3},
            )
            for name in ("base-recipe", "base-data", "base-root"):
                (root / name).mkdir()
            path.write_text(yaml.safe_dump(config))
            provider = FakeXlabIdeaProvider()
            controller = Harness(config_path=path, xlab_provider=provider)
            with patch(
                "playground.sure_master.core.playground.KnowledgePromotionExp"
            ) as knowledge:
                knowledge.return_value.exp_name = "knowledge"
                knowledge.return_value.run.return_value = "observations"
                result = controller.run("synthetic improvement")
            self.assertEqual(result["status"], "completed", result)
            self.assertEqual(len(provider.generated_requests), 10)
            state = json.loads(
                (root / "workspace/metric/controller_state.json").read_text()
            )
            self.assertEqual(len(state["candidates"]), 40)


class SourceSnapshotTests(unittest.TestCase):
    def test_snapshot_keeps_workspace_modules_and_unexpanded_credentials(self):
        import yaml
        from playground.sure_master.tools import freeze_tedlium_run as freezing
        from playground.sure_master.core.utils import slurm

        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / "evomaster").mkdir()
            (root / "evomaster/__init__.py").write_text("")
            package = root / "playground/sure_master/core/utils"
            package.mkdir(parents=True)
            (root / "playground/__init__.py").write_text("")
            (package / "workspace_cleanup.py").write_text("VALUE = 1\n")
            config = root / "input.yaml"
            config.write_text(
                yaml.safe_dump(
                    {
                        "sure": {"execution_env": {}, "slurm": {}},
                        "llm": {"api_key": "${OPENAI_API_KEY}"},
                    }
                )
            )
            with (
                patch.object(freezing, "PROJECT", root),
                patch.object(slurm, "PROJECT", root),
                patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-not-for-export"}),
            ):
                snapshot, deployment = freezing.freeze(config, root / "run")
                self.assertTrue(
                    (
                        snapshot
                        / "playground/sure_master/core/utils/workspace_cleanup.py"
                    ).is_file()
                )
                self.assertEqual(snapshot.name, slurm.source_digest(snapshot))
                self.assertIn("${OPENAI_API_KEY}", deployment.read_text())
                self.assertNotIn("fixture-not-for-export", deployment.read_text())


if __name__ == "__main__":
    unittest.main()
