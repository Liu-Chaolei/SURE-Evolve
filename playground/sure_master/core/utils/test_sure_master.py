from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

from playground.sure_master.core import playground as sure_playground_module
from playground.sure_master.core.playground import SureMasterPlayground
from playground.sure_master.core.exp.research_exp import ResearchExp
from playground.sure_master.core.exp.run_exp import SureRunExp
from playground.sure_master.env.local import SureMasterLocalEnv
from playground.sure_master.baselines import (
    f5tts_v1_base_official_baseline as f5tts_official_baseline,
)
from playground.sure_master.baselines import (
    zipformer_large_cr_ctc_rnnt_baseline as zipformer_baseline,
)
from playground.sure_master.baselines.asr_profiles import (
    RECIPE_PROFILES,
    get_asr_dataset_profile,
    normalize_asr_cut_id,
    parse_eval_splits_for_profile,
    recipe_arch_args,
    recipe_decode_data_args,
    recipe_train_data_args,
    recipe_train_only_args,
)
from playground.sure_master.baselines.zipformer_large_cr_ctc_rnnt_baseline import (
    bounded_train_max_duration,
    duration_retry_sequence,
    is_oom_failure,
    normalize_librispeech_cut_id,
    world_size,
)
from playground.sure_master.tools import build_asr_refs
from playground.sure_master.tools.build_asr_refs import (
    RefRow,
    select_tiered_rows,
)
from evomaster.agent.session.local import LocalSessionConfig
from evomaster.env import local as evomaster_local_env_module
from evomaster.env.local import LocalEnv, LocalEnvConfig, ResourceAllocator

from .code import validate_run_sure_script, validate_sure_candidate_boundary
from .candidate_type import (
    ARCH,
    FINE_TUNE,
    INFERENCE,
    TRAINING,
    candidate_type_from_code,
    candidate_type_from_idea,
)
from .candidate_changes import validate_arch_candidate_changes, validate_candidate_changes
from .metric import MetricGpuAllocator, MetricGpuSnapshot, SureMetricRunner
from playground.sure_master.tools import (
    run_f5tts_arch_finetune,
    run_f5tts_batch_infer,
    run_icefall_zipformer_candidate,
    run_vc_sure_candidate,
)
from playground.sure_master.tools.cleanup_sure_workspaces import find_candidate_workspaces
from . import runtime_env
from .task_cards import (
    BaseModelProfile,
    SureTaskCard,
    infer_metric_direction,
    load_task_cards,
    merge_base_model_profile,
    resolve_task_card,
    validate_base_model_profile,
)
from .vc_remote import (
    VcRemoteTrainingExecutor,
    candidate_runs_remotely,
    draft_runs_remotely,
    mixed_execution_enabled,
    parse_vc_info_partitions,
    remote_candidate_types_from,
    remote_resource_config_from,
    remote_training_max_parallel,
)
from .workspace_cleanup import (
    WorkspaceCleanupConfig,
    cleanup_candidate_workspace,
    workspace_cleanup_config,
)
from playground.sure_master.tools.run_vc_sure_candidate import load_remote_candidate_context
from playground.sure_master.tools.run_vc_sure_candidate import ensure_base_model_paths


class SureTaskCardsTest(unittest.TestCase):
    def test_f5tts_candidate_type_prefers_explicit_label(self):
        self.assertEqual(candidate_type_from_idea(("1", "[training] fine-tune for 1000 steps")), FINE_TUNE)
        self.assertEqual(candidate_type_from_idea(("2", "[inference] adjust nfe steps")), INFERENCE)
        self.assertEqual(candidate_type_from_idea({"idea_type": "training", "idea": "use wrapper"}), FINE_TUNE)
        self.assertEqual(candidate_type_from_idea(("3", "[arch] reduce Zipformer encoder_dim")), ARCH)
        self.assertEqual(
            candidate_type_from_idea("[fine_tune] tune loss without changing model structure"),
            FINE_TUNE,
        )

    def test_f5tts_candidate_type_code_forces_training_wrapper(self):
        code = "cmd = [os.environ['SURE_TTS_FINETUNE_WRAPPER'], '--action', 'finetune_short']"
        self.assertEqual(candidate_type_from_code(code, default=INFERENCE), FINE_TUNE)

    def test_f5tts_candidate_type_code_forces_arch_wrapper(self):
        code = "cmd = [os.environ['SURE_TTS_ARCH_WRAPPER'], '--action', 'arch_finetune_short']"
        self.assertEqual(candidate_type_from_code(code, default=INFERENCE), ARCH)

    def test_asr_candidate_type_code_forces_training_recipe(self):
        code = (
            "cmd = ['/opt/conda/envs/icefall/bin/python', "
            "'base_model/recipe/train.py', '--world-size', '8']"
        )
        self.assertEqual(candidate_type_from_code(code, default=INFERENCE), FINE_TUNE)

    def test_asr_candidate_type_code_forces_arch_structure_args(self):
        code = (
            "cmd = ['/opt/conda/envs/icefall/bin/python', "
            "'base_model/recipe/train.py', '--encoder-dim', '256']"
        )
        self.assertEqual(candidate_type_from_code(code, default=INFERENCE), ARCH)

    def test_asr_baseline_train_duration_respects_sure_max_duration_cap(self):
        env = {
            "SURE_MAX_DURATION": "600",
            "SURE_BASELINE_TRAIN_MAX_DURATION": "1400",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(bounded_train_max_duration(), 600)

    def test_asr_baseline_train_duration_clamps_to_configured_floor(self):
        env = {
            "SURE_MAX_DURATION": "16",
            "SURE_BASELINE_TRAIN_MAX_DURATION": "16",
            "SURE_DURATION_AUTOTUNE_MIN": "100",
            "SURE_TRAIN_DURATION_MIN": "100",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(bounded_train_max_duration(), 100)

    def test_asr_baseline_world_size_uses_requested_when_cuda_visible_unset(self):
        with patch.dict(os.environ, {"ASR_WORLD_SIZE": "8"}, clear=True):
            self.assertEqual(world_size(), 8)
        env = {"ASR_WORLD_SIZE": "8", "CUDA_VISIBLE_DEVICES": "0,1,2,3"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(world_size(), 4)

    def test_asr_baseline_decode_command_omits_train_only_args(self):
        env = {
            "SURE_BASELINE_USE_PRETRAINED": "1",
            "SURE_BASELINE_DECODE_MAX_DURATION": "300",
            "SURE_ICEFALL_PYTHON": "/python",
        }
        with patch.dict(os.environ, env, clear=True):
            command = zipformer_baseline.build_decode_command(
                50,
                Path("working/decode_sure_eval.py"),
            )

        self.assertIn("--use-cr-ctc", command)
        self.assertIn("--encoder-dim", command)
        self.assertIn("--full-libri", command)
        self.assertNotIn("--ctc-loss-scale", command)
        self.assertNotIn("--cr-loss-scale", command)
        self.assertNotIn("--time-mask-ratio", command)
        self.assertNotIn("--enable-spec-aug", command)

    def test_asr_profiles_default_to_librispeech_recipe_args(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(zipformer_baseline.dataset_profile().name, "librispeech")
            self.assertEqual(
                zipformer_baseline.recipe_profile().name,
                "librispeech_zipformer_large_cr_ctc_rnnt",
            )
            self.assertEqual(zipformer_baseline.parse_eval_splits(), ["test-clean", "test-other"])
            self.assertIn("--full-libri", zipformer_baseline.large_cr_ctc_rnnt_train_args())
            self.assertIn("--full-libri", zipformer_baseline.large_cr_ctc_rnnt_decode_args())

    def test_asr_profiles_directly_own_recipe_data_args_and_composition(self):
        librispeech = RECIPE_PROFILES["librispeech_zipformer_large_cr_ctc_rnnt"]
        tedlium3 = RECIPE_PROFILES["tedlium3_zipformer"]

        self.assertEqual(librispeech.train_data_args, ("--full-libri", "1"))
        self.assertEqual(librispeech.decode_data_args, ("--full-libri", "1"))
        self.assertEqual(tedlium3.train_data_args, ())
        self.assertEqual(tedlium3.decode_data_args, ())
        self.assertNotIn("--full-libri", runtime_env._PROBE_CONTROLLED_OPTIONS)
        self.assertNotIn("--full-libri", runtime_env._PROBE_REQUIRED_OPTIONS)

        for profile in (librispeech, tedlium3):
            with self.subTest(profile=profile.name):
                expected_train = (
                    recipe_arch_args(profile)
                    + recipe_train_only_args(profile)
                    + recipe_train_data_args(profile)
                )
                expected_decode = recipe_arch_args(profile) + recipe_decode_data_args(profile)
                env = {
                    "SURE_ASR_DATASET": profile.dataset,
                    "SURE_ASR_RECIPE_PROFILE": profile.name,
                }
                with patch.dict(os.environ, env, clear=True):
                    self.assertEqual(
                        zipformer_baseline.large_cr_ctc_rnnt_train_args(),
                        expected_train,
                    )
                    self.assertEqual(
                        zipformer_baseline.large_cr_ctc_rnnt_decode_args(),
                        expected_decode,
                    )

    def test_asr_profiles_tedlium3_uses_dataset_splits_and_supported_recipe_args(self):
        env = {
            "SURE_ASR_DATASET": "tedlium3",
            "SURE_ASR_RECIPE_PROFILE": "tedlium3_zipformer",
            "SURE_ASR_EVAL_SPLITS": "dev,test",
            "SURE_ICEFALL_PYTHON": "/python",
        }
        unsupported_librispeech_args = {
            "--full-libri",
            "--use-cr-ctc",
            "--use-ctc",
            "--use-transducer",
            "--use-attention-decoder",
            "--ctc-loss-scale",
            "--cr-loss-scale",
            "--time-mask-ratio",
        }
        supported_zipformer_args = {
            "--num-encoder-layers",
            "--feedforward-dim",
            "--encoder-dim",
            "--encoder-unmasked-dim",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(zipformer_baseline.dataset_profile().name, "tedlium3")
            self.assertEqual(zipformer_baseline.recipe_profile().name, "tedlium3_zipformer")
            self.assertEqual(zipformer_baseline.parse_eval_splits(), ["dev", "test"])
            train_args = zipformer_baseline.large_cr_ctc_rnnt_train_args()
            decode_args = zipformer_baseline.large_cr_ctc_rnnt_decode_args()
            command = zipformer_baseline.build_decode_command(1, Path("decode.py"))

        for arg in unsupported_librispeech_args:
            self.assertNotIn(arg, train_args)
            self.assertNotIn(arg, decode_args)
            self.assertNotIn(arg, command)
        for arg in supported_zipformer_args:
            self.assertIn(arg, train_args)
            self.assertIn(arg, decode_args)
            self.assertIn(arg, command)
        self.assertIn("--enable-spec-aug", train_args)
        self.assertNotIn("--enable-spec-aug", decode_args)
        self.assertEqual(
            command[command.index("--bpe-model") + 1],
            "data/lang_bpe_500/bpe.model",
        )

    def test_asr_profiles_reject_mismatched_recipe_dataset(self):
        env = {
            "SURE_ASR_DATASET": "tedlium3",
            "SURE_ASR_RECIPE_PROFILE": "librispeech_zipformer_large_cr_ctc_rnnt",
        }
        with patch.dict(os.environ, env, clear=True), self.assertRaisesRegex(ValueError, "is for dataset"):
            zipformer_baseline.recipe_profile()

    def test_asr_profiles_validate_eval_splits_per_dataset(self):
        tedlium3 = get_asr_dataset_profile("tedlium3")
        self.assertEqual(parse_eval_splits_for_profile(None, tedlium3), ["dev"])
        with self.assertRaisesRegex(ValueError, "Unsupported SURE_ASR_EVAL_SPLITS"):
            parse_eval_splits_for_profile("dev-clean", tedlium3)

    def test_asr_baseline_train_command_keeps_train_only_args(self):
        env = {
            "ASR_WORLD_SIZE": "8",
            "SURE_ENABLE_MUSAN": "0",
            "SURE_ICEFALL_PYTHON": "/python",
        }
        with patch.dict(os.environ, env, clear=True):
            command = zipformer_baseline.build_train_command(
                train_epochs=1,
                train_max_duration=300,
                fp16="1",
                attempt_index=1,
            )

        self.assertIn("--ctc-loss-scale", command)
        self.assertIn("--cr-loss-scale", command)
        self.assertIn("--time-mask-ratio", command)
        self.assertIn("--enable-spec-aug", command)

    def test_asr_baseline_averaged_model_requires_endpoint_checkpoints(self):
        env = {
            "SURE_BASELINE_EPOCH": "50",
            "SURE_BASELINE_AVG": "26",
        }
        with patch.dict(os.environ, env, clear=True):
            paths = zipformer_baseline.required_decode_checkpoints(50)
        self.assertEqual([path.name for path in paths], ["epoch-24.pt", "epoch-50.pt"])

    def test_decode_duration_selector_caches_only_demonstrated_safe_with_safety_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls: list[int] = []

            def probe(duration: int, _path: Path) -> zipformer_baseline.DecodeDurationEvidence | None:
                calls.append(duration)
                if duration > 200:
                    return None
                return zipformer_baseline.DecodeDurationEvidence(duration, 700, 1000)

            env = {
                "SURE_DECODE_DURATION_CACHE_DIR": tmp,
                "SURE_DECODE_DURATION_MIN": "100",
                "SURE_DECODE_DURATION_MAX": "300",
                "SURE_DECODE_DURATION_STEP": "100",
                "SURE_DECODE_DURATION_SAFETY_STEPS": "1",
                "SURE_DECODE_DURATION_RESERVE_MB": "100",
            }
            with patch.dict(os.environ, env, clear=True):
                selected = zipformer_baseline.select_decode_max_duration(
                    signature="workload", signature_payload={"x": 1}, default=200, probe=probe
                )
                cached = zipformer_baseline.select_decode_max_duration(
                    signature="workload", signature_payload={"x": 1}, default=200,
                    probe=lambda *_: self.fail("safe cache should avoid probing"),
                )
        self.assertEqual(selected, 100)
        self.assertEqual(cached, 100)
        self.assertEqual(calls, [100, 200, 300])

    def test_decode_duration_headroom_requires_memory_evidence(self):
        evidence = zipformer_baseline.DecodeDurationEvidence(100, None, 1000)
        self.assertFalse(evidence.has_headroom(100))
        self.assertTrue(evidence.has_headroom(0))
        self.assertFalse(zipformer_baseline.DecodeDurationEvidence(100, 950, 1000).has_headroom(100))

    def test_decode_duration_reversed_bounds_are_normalized(self):
        env = {
            "SURE_DECODE_DURATION_MIN": "300",
            "SURE_DECODE_DURATION_MAX": "100",
            "SURE_DECODE_DURATION_STEP": "0",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(zipformer_baseline._decode_duration_bounds(200)[:3], (100, 300, 1))

    def test_candidate_cannot_override_controlled_decode_duration(self):
        with self.assertRaisesRegex(ValueError, "selector owns"):
            run_icefall_zipformer_candidate.validate_candidate_args(
                candidate_type=INFERENCE,
                action="decode_only",
                train_extra_args=[],
                decode_extra_args=["--max-duration=999"],
            )

    def test_decode_cleanup_is_scoped_and_stale_outputs_are_invalidated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp = root / "exp"
            artifacts = root / "artifacts"
            exp.mkdir()
            artifacts.mkdir()
            stale = exp / "recogs-old.txt"
            unrelated = exp / "epoch-1.pt"
            candidate = artifacts / "candidate_changes.json"
            hyp = artifacts / "hyp.txt"
            for path in (stale, unrelated, candidate, hyp):
                path.write_text("old", encoding="utf-8")
            with patch.object(zipformer_baseline, "ARTIFACTS_DIR", artifacts):
                zipformer_baseline.invalidate_decode_artifacts(exp, include_candidate_record=True)
                before = set(zipformer_baseline._decode_artifact_paths(exp))
                current = exp / "recogs-current.txt"
                current.write_text("new", encoding="utf-8")
                zipformer_baseline.cleanup_decode_attempt(exp, before)
            self.assertTrue(unrelated.exists())
            self.assertFalse(stale.exists())
            self.assertFalse(current.exists())
            self.assertFalse(candidate.exists())
            self.assertFalse(hyp.exists())

    def test_decode_signature_changes_for_checkpoint_bpe_and_decode_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "models"
            model.mkdir()
            checkpoint = model / "epoch-1.pt"
            checkpoint.write_bytes(b"checkpoint-a")
            bpe = root / "bpe.model"
            bpe.write_bytes(b"bpe-a")
            env = {"SURE_BASELINE_USE_PRETRAINED": "1", "SURE_ASR_EVAL_SPLITS": "test-clean"}
            with patch.dict(os.environ, env, clear=True), patch.object(
                zipformer_baseline, "MODELS_DIR", model
            ), patch.object(zipformer_baseline, "_gpu_signature", return_value={"gpus": ["GPU, 1000"]}):
                kwargs = dict(epoch=1, decode_method="greedy_search", decode_args=["--beam-size", "4"], bpe_model=bpe, exp_dir=model, avg=1, use_averaged_model="0")
                first, _ = zipformer_baseline.decode_workload_signature(**kwargs)
                bpe.write_bytes(b"bpe-b")
                second, _ = zipformer_baseline.decode_workload_signature(**kwargs)
                kwargs["decode_args"] = ["--beam-size", "8"]
                third, _ = zipformer_baseline.decode_workload_signature(**kwargs)
            self.assertEqual(len({first, second, third}), 3)

    def test_asr_baseline_reports_missing_averaged_start_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "models"
            models.mkdir()
            (models / "epoch-50.pt").write_text("checkpoint", encoding="utf-8")
            env = {
                "SURE_BASELINE_EPOCH": "50",
                "SURE_BASELINE_AVG": "26",
            }
            with patch.object(zipformer_baseline, "MODELS_DIR", models), patch.dict(
                os.environ,
                env,
                clear=True,
            ):
                missing = zipformer_baseline.missing_decode_checkpoints(50)

        self.assertEqual([path.name for path in missing], ["epoch-24.pt"])

    def test_asr_baseline_pretrained_checkpoint_maps_to_target_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "official_exp"
            source.mkdir()
            (source / "pretrained.pt").write_text("pretrained", encoding="utf-8")
            (source / "epoch-50.pt").write_text("raw epoch", encoding="utf-8")
            models = Path(tmp) / "workspace_models"
            models.mkdir()
            env = {
                "SURE_BASELINE_CHECKPOINT_DIR": str(source),
                "SURE_BASELINE_EPOCH": "50",
                "SURE_BASELINE_AVG": "26",
                "SURE_BASELINE_USE_PRETRAINED": "1",
            }
            with patch.object(zipformer_baseline, "MODELS_DIR", models), patch.dict(
                os.environ,
                env,
                clear=True,
            ):
                zipformer_baseline.copy_checkpoint_source()
                self.assertEqual(zipformer_baseline.decode_avg(50), 1)
                self.assertFalse(zipformer_baseline.decode_uses_averaged_model(50))
                missing = zipformer_baseline.missing_decode_checkpoints(50)
                self.assertTrue((models / "epoch-50.pt").is_file())
                self.assertEqual(
                    (models / "epoch-50.pt").read_text(encoding="utf-8"),
                    "pretrained",
                )

        self.assertEqual(missing, [])

    def test_asr_baseline_decode_uses_official_package_bpe(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "official_package"
            source = package / "exp"
            source.mkdir(parents=True)
            lang_dir = package / "data" / "lang_bpe_500"
            lang_dir.mkdir(parents=True)
            official_bpe = lang_dir / "bpe.model"
            official_bpe.write_text("official bpe", encoding="utf-8")
            (source / "pretrained.pt").write_text("pretrained", encoding="utf-8")
            (source / "epoch-50.pt").write_text("raw epoch", encoding="utf-8")
            models = Path(tmp) / "workspace_models"
            models.mkdir()
            env = {
                "SURE_BASELINE_CHECKPOINT_DIR": str(source),
                "SURE_BASELINE_EPOCH": "50",
                "SURE_BASELINE_USE_PRETRAINED": "1",
                "SURE_ICEFALL_PYTHON": "/python",
            }
            with patch.object(zipformer_baseline, "MODELS_DIR", models), patch.dict(
                os.environ,
                env,
                clear=True,
            ):
                command = zipformer_baseline.build_decode_command(50, Path("decode.py"))

            bpe_arg = command[command.index("--bpe-model") + 1]
            self.assertEqual(bpe_arg, str(official_bpe))

    def test_zipformer_candidate_train_decode_uses_training_bpe_with_official_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_bpe = root / zipformer_baseline.DEFAULT_BPE_MODEL
            local_bpe.parent.mkdir(parents=True)
            local_bpe.write_bytes(b"local candidate bpe")
            package = root / zipformer_baseline.OFFICIAL_MODEL_DIR_NAME
            source = package / "exp"
            source.mkdir(parents=True)
            official_bpe = package / zipformer_baseline.OFFICIAL_LANG_BPE_MODEL
            official_bpe.parent.mkdir(parents=True)
            official_bpe.write_bytes(b"official checkpoint bpe")
            env = {
                "SURE_BASELINE_CHECKPOINT_DIR": str(source),
                "SURE_ICEFALL_PYTHON": "/python",
            }
            cwd = Path.cwd()
            os.chdir(root)
            try:
                with patch.dict(os.environ, env, clear=True), patch.object(
                    run_icefall_zipformer_candidate.baseline,
                    "patched_decode_script",
                    return_value=Path("decode.py"),
                ):
                    final_train_args = run_icefall_zipformer_candidate.merge_cli_args(
                        zipformer_baseline.LARGE_CR_CTC_RNNT_ARGS,
                        [],
                    )
                    train_bpe = run_icefall_zipformer_candidate.selected_train_bpe_model(
                        final_train_args
                    )
                    decode_bpe = run_icefall_zipformer_candidate.selected_decode_bpe_model(
                        action="train_decode",
                        final_train_args=final_train_args,
                    )
                    run_icefall_zipformer_candidate.validate_train_decode_bpe_models(
                        train_bpe,
                        decode_bpe,
                    )
                    command = run_icefall_zipformer_candidate.decode_command(
                        exp_dir=Path("models/candidate"),
                        epoch=1,
                        avg=1,
                        use_averaged_model="0",
                        decode_method="modified_beam_search",
                        decode_max_duration=300,
                        bpe_model=decode_bpe,
                        final_decode_args=[],
                    )
            finally:
                os.chdir(cwd)

        bpe_arg = command[command.index("--bpe-model") + 1]
        self.assertEqual(bpe_arg, str(zipformer_baseline.DEFAULT_BPE_MODEL))
        self.assertNotEqual(bpe_arg, str(official_bpe))

    def test_zipformer_candidate_train_decode_allows_matching_decode_bpe_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bpe_model = Path("data/lang_bpe_500/bpe.model")
            resolved_bpe = root / bpe_model
            resolved_bpe.parent.mkdir(parents=True)
            resolved_bpe.write_bytes(b"candidate bpe")
            cwd = Path.cwd()
            os.chdir(root)
            try:
                with patch.object(
                    run_icefall_zipformer_candidate.baseline,
                    "patched_decode_script",
                    return_value=Path("decode.py"),
                ):
                    final_train_args = run_icefall_zipformer_candidate.merge_cli_args(
                        zipformer_baseline.LARGE_CR_CTC_RNNT_ARGS,
                        ["--bpe-model", str(bpe_model)],
                    )
                    final_decode_args = run_icefall_zipformer_candidate.merge_cli_args(
                        zipformer_baseline.LARGE_CR_CTC_RNNT_DECODE_ARGS,
                        [f"--bpe-model={bpe_model}", "--beam-size", "4"],
                    )
                    train_bpe = run_icefall_zipformer_candidate.selected_train_bpe_model(
                        final_train_args
                    )
                    run_icefall_zipformer_candidate.validate_requested_decode_bpe_model(
                        train_bpe_model=train_bpe,
                        final_decode_args=final_decode_args,
                    )
                    decode_args = run_icefall_zipformer_candidate.remove_cli_arg(
                        final_decode_args,
                        "--bpe-model",
                    )
                    command = run_icefall_zipformer_candidate.decode_command(
                        exp_dir=Path("models/candidate"),
                        epoch=1,
                        avg=1,
                        use_averaged_model="0",
                        decode_method="modified_beam_search",
                        decode_max_duration=300,
                        bpe_model=train_bpe,
                        final_decode_args=decode_args,
                    )
            finally:
                os.chdir(cwd)

        self.assertEqual(command.count("--bpe-model"), 1)
        self.assertEqual(command[command.index("--bpe-model") + 1], str(bpe_model))
        self.assertIn("--beam-size", command)

    def test_zipformer_candidate_train_decode_rejects_bpe_content_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_bpe = root / "data/lang_bpe_500/bpe.model"
            train_bpe.parent.mkdir(parents=True)
            train_bpe.write_bytes(b"training bpe")
            decode_bpe = root / "official/data/lang_bpe_500/bpe.model"
            decode_bpe.parent.mkdir(parents=True)
            decode_bpe.write_bytes(b"different decode bpe")
            cwd = Path.cwd()
            os.chdir(root)
            try:
                with self.assertRaises(RuntimeError) as ctx:
                    run_icefall_zipformer_candidate.validate_requested_decode_bpe_model(
                        train_bpe_model=Path("data/lang_bpe_500/bpe.model"),
                        final_decode_args=["--bpe-model", str(decode_bpe)],
                    )
            finally:
                os.chdir(cwd)

        self.assertIn("BPE mismatch", str(ctx.exception))

    def _zipformer_candidate_args_with_structure_sync(
        self,
        train_extra_args: list[str],
        decode_extra_args: list[str],
    ) -> tuple[list[str], list[str]]:
        final_train_args = run_icefall_zipformer_candidate.merge_cli_args(
            zipformer_baseline.large_cr_ctc_rnnt_train_args(),
            train_extra_args,
        )
        final_decode_args = run_icefall_zipformer_candidate.merge_cli_args(
            zipformer_baseline.large_cr_ctc_rnnt_decode_args(),
            decode_extra_args,
        )
        final_decode_args = run_icefall_zipformer_candidate.force_decode_structure_from_train(
            final_train_args=final_train_args,
            final_decode_args=final_decode_args,
            decode_extra_args=decode_extra_args,
        )
        return final_train_args, final_decode_args

    def test_zipformer_candidate_train_decode_propagates_train_structure_to_decode(self):
        _, final_decode_args = self._zipformer_candidate_args_with_structure_sync(
            ["--num-encoder-layers", "2,2,3,4,3,2"],
            ["--decoding-method", "modified_beam_search"],
        )

        selected_layers = run_icefall_zipformer_candidate.last_arg_value(
            final_decode_args,
            "--num-encoder-layers",
        )
        self.assertEqual(selected_layers, "2,2,3,4,3,2")
        self.assertIn("--decoding-method", final_decode_args)

    def test_zipformer_candidate_train_decode_allows_matching_decode_structure(self):
        _, final_decode_args = self._zipformer_candidate_args_with_structure_sync(
            ["--num-encoder-layers", "2,2,3,4,3,2"],
            ["--num-encoder-layers", "2,2,3,4,3,2", "--beam-size", "4"],
        )

        selected_layers = run_icefall_zipformer_candidate.last_arg_value(
            final_decode_args,
            "--num-encoder-layers",
        )
        self.assertEqual(selected_layers, "2,2,3,4,3,2")
        self.assertIn("--beam-size", final_decode_args)

    def test_zipformer_candidate_train_decode_rejects_conflicting_decode_structure(self):
        final_train_args = run_icefall_zipformer_candidate.merge_cli_args(
            zipformer_baseline.LARGE_CR_CTC_RNNT_ARGS,
            ["--num-encoder-layers", "2,2,3,4,3,2"],
        )
        final_decode_args = run_icefall_zipformer_candidate.merge_cli_args(
            zipformer_baseline.LARGE_CR_CTC_RNNT_DECODE_ARGS,
            ["--num-encoder-layers", "2,2,4,5,4,2"],
        )

        with self.assertRaises(ValueError) as ctx:
            run_icefall_zipformer_candidate.force_decode_structure_from_train(
                final_train_args=final_train_args,
                final_decode_args=final_decode_args,
                decode_extra_args=["--num-encoder-layers", "2,2,4,5,4,2"],
            )

        self.assertIn("--num-encoder-layers", str(ctx.exception))
        self.assertIn("structure mismatch", str(ctx.exception))

    def test_zipformer_candidate_train_decode_replays_exp16_structure_pattern(self):
        final_train_args, final_decode_args = self._zipformer_candidate_args_with_structure_sync(
            [
                "--num-encoder-layers",
                "2,2,3,4,3,2",
                "--encoder-dim",
                "192,256,512,768,512,256",
                "--feedforward-dim",
                "512,768,1536,2048,1536,768",
                "--encoder-unmasked-dim",
                "192,192,256,320,256,192",
            ],
            ["--decoding-method", "modified_beam_search"],
        )

        for option in run_icefall_zipformer_candidate.STRUCTURE_ARGS:
            self.assertEqual(
                run_icefall_zipformer_candidate.last_arg_value(final_decode_args, option),
                run_icefall_zipformer_candidate.last_arg_value(final_train_args, option),
            )

    def test_zipformer_candidate_decode_only_keeps_decode_structure_independent(self):
        final_decode_args = run_icefall_zipformer_candidate.merge_cli_args(
            zipformer_baseline.LARGE_CR_CTC_RNNT_DECODE_ARGS,
            ["--decoding-method", "modified_beam_search"],
        )

        self.assertEqual(
            run_icefall_zipformer_candidate.last_arg_value(
                final_decode_args,
                "--num-encoder-layers",
            ),
            "2,2,4,5,4,2",
        )

    def test_zipformer_candidate_record_includes_bpe_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bpe_model = Path("data/lang_bpe_500/bpe.model")
            resolved_bpe = root / bpe_model
            resolved_bpe.parent.mkdir(parents=True)
            resolved_bpe.write_bytes(b"candidate bpe")
            cwd = Path.cwd()
            os.chdir(root)
            try:
                expected_sha256 = run_icefall_zipformer_candidate.file_sha256(resolved_bpe)
                run_icefall_zipformer_candidate.write_candidate_record(
                    candidate_type=ARCH,
                    idea_text="change encoder dim",
                    changed_fields=[],
                    final_train_args=["--encoder-dim", "192"],
                    final_decode_args=["--beam-size", "4"],
                    train_extra_args=["--encoder-dim", "192"],
                    decode_extra_args=["--beam-size", "4"],
                    exp_dir=Path("models/candidate"),
                    trained_epoch=1,
                    selected_duration=100,
                    attempts=[],
                    train_bpe_model=bpe_model,
                    decode_bpe_model=bpe_model,
                    elapsed_seconds=1.0,
                )
                payload = json.loads(
                    (root / "artifacts/candidate_changes.json").read_text(
                        encoding="utf-8"
                    )
                )
            finally:
                os.chdir(cwd)

        self.assertEqual(payload["training_config"]["actual_bpe_model"], str(bpe_model))
        self.assertEqual(payload["training_config"]["actual_bpe_sha256"], expected_sha256)
        self.assertEqual(payload["inference_config"]["actual_bpe_model"], str(bpe_model))
        self.assertEqual(payload["inference_config"]["actual_bpe_sha256"], expected_sha256)

    def test_zipformer_staged_resume_config_sets_start_and_target_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "retained" / "epoch-1.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text("checkpoint", encoding="utf-8")
            exp_dir = root / "models" / "candidate"
            env = {
                "SURE_STAGED_RESUME_ENABLED": "1",
                "SURE_STAGED_RESUME_CHECKPOINT": str(checkpoint),
                "SURE_STAGED_RESUME_EPOCH": "1",
                "SURE_STAGED_TARGET_EPOCH": "2",
                "SURE_STAGED_RESUME_SOURCE_RUNG": "short",
            }
            with patch.dict(os.environ, env, clear=True):
                config = run_icefall_zipformer_candidate.staged_resume_config(
                    exp_dir,
                    requested_train_epochs=1,
                )

            self.assertTrue(config["enabled"])
            self.assertEqual(config["resume_epoch"], 1)
            self.assertEqual(config["start_epoch"], 2)
            self.assertEqual(config["target_epoch"], 2)
            self.assertEqual(config["source_rung"], "short")
            self.assertTrue((exp_dir / "epoch-1.pt").exists())

    def test_zipformer_staged_resume_config_rejects_non_increasing_target_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "epoch-2.pt"
            checkpoint.write_text("checkpoint", encoding="utf-8")
            env = {
                "SURE_STAGED_RESUME_ENABLED": "1",
                "SURE_STAGED_RESUME_CHECKPOINT": str(checkpoint),
                "SURE_STAGED_RESUME_EPOCH": "2",
                "SURE_STAGED_TARGET_EPOCH": "2",
            }
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(RuntimeError) as ctx:
                    run_icefall_zipformer_candidate.staged_resume_config(
                        Path(tmp) / "models",
                        requested_train_epochs=2,
                    )

        self.assertIn("must be greater", str(ctx.exception))

    def test_zipformer_train_command_uses_staged_resume_start_epoch(self):
        with patch.dict(os.environ, {"ASR_WORLD_SIZE": "8", "SURE_USE_FP16": "1"}, clear=True):
            command = run_icefall_zipformer_candidate.train_command(
                exp_dir=Path("models/candidate"),
                train_epochs=2,
                start_epoch=2,
                train_duration=300,
                final_train_args=["--encoder-dim", "192"],
                attempt_index=1,
            )

        self.assertEqual(command[command.index("--num-epochs") + 1], "2")
        self.assertEqual(command[command.index("--start-epoch") + 1], "2")

    def test_zipformer_retry_restages_staged_resume_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "retained" / "epoch-1.pt"
            source.parent.mkdir(parents=True)
            source.write_text("checkpoint", encoding="utf-8")
            exp_dir = root / "models"
            exp_dir.mkdir()
            staged_resume = {
                "enabled": True,
                "resume_checkpoint": str(source),
                "local_checkpoint": str(exp_dir / "epoch-1.pt"),
            }
            calls = {"count": 0}

            def fake_run_command(name, command, timeout):
                calls["count"] += 1
                log_path = root / f"{name}.log"
                log_path.write_text("CUDA out of memory\n", encoding="utf-8")
                if calls["count"] == 1:
                    raise zipformer_baseline.CommandFailedError(command, 1, log_path)
                return log_path

            def fake_clean():
                for item in exp_dir.iterdir():
                    item.unlink()

            with patch.object(run_icefall_zipformer_candidate, "selected_train_duration", return_value=300), patch.object(
                zipformer_baseline,
                "duration_retry_sequence",
                return_value=[300, 200],
            ), patch.object(zipformer_baseline, "clean_training_exp_dir_for_retry", side_effect=fake_clean), patch.object(
                zipformer_baseline,
                "run_command",
                side_effect=fake_run_command,
            ), patch.object(zipformer_baseline, "is_oom_failure", return_value=True), patch.object(
                zipformer_baseline,
                "parse_max_memory_mb",
                return_value=None,
            ), patch.dict(os.environ, {"SURE_BASELINE_TRAIN_TIMEOUT": "10"}, clear=True):
                trained_epoch, selected_duration, attempts = run_icefall_zipformer_candidate.run_training(
                    exp_dir=exp_dir,
                    train_epochs=2,
                    start_epoch=2,
                    final_train_args=[],
                    staged_resume=staged_resume,
                )

            self.assertEqual(trained_epoch, 2)
            self.assertEqual(selected_duration, 200)
            self.assertEqual(len(attempts), 2)
            self.assertEqual((exp_dir / "epoch-1.pt").read_text(encoding="utf-8"), "checkpoint")

    def test_asr_baseline_official_source_requires_matching_bpe(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / zipformer_baseline.OFFICIAL_MODEL_DIR_NAME
            source = package / "exp"
            source.mkdir(parents=True)
            (source / "pretrained.pt").write_text("pretrained", encoding="utf-8")
            models = Path(tmp) / "workspace_models"
            models.mkdir()
            env = {
                "SURE_BASELINE_CHECKPOINT_DIR": str(source),
                "SURE_BASELINE_USE_PRETRAINED": "1",
            }
            with patch.object(zipformer_baseline, "MODELS_DIR", models), patch.dict(
                os.environ,
                env,
                clear=True,
            ):
                with self.assertRaises(FileNotFoundError) as ctx:
                    zipformer_baseline.validate_decode_bpe_model()

        self.assertIn("matching data/lang_bpe_500/bpe.model", str(ctx.exception))

    def test_asr_baseline_pretrained_replaces_epoch_symlink_without_touching_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "official_exp"
            source.mkdir()
            (source / "pretrained.pt").write_text("pretrained", encoding="utf-8")
            raw_epoch = source / "epoch-50.pt"
            raw_epoch.write_text("raw epoch", encoding="utf-8")
            models = Path(tmp) / "workspace_models"
            models.mkdir()
            target = models / "epoch-50.pt"
            target.symlink_to(raw_epoch)
            env = {
                "SURE_BASELINE_CHECKPOINT_DIR": str(source),
                "SURE_BASELINE_EPOCH": "50",
                "SURE_BASELINE_USE_PRETRAINED": "1",
            }
            with patch.object(zipformer_baseline, "MODELS_DIR", models), patch.dict(
                os.environ,
                env,
                clear=True,
            ):
                zipformer_baseline.copy_checkpoint_source()

            self.assertFalse(target.is_symlink())
            self.assertEqual(target.read_text(encoding="utf-8"), "pretrained")
            self.assertEqual(raw_epoch.read_text(encoding="utf-8"), "raw epoch")

    def test_asr_configured_checkpoint_source_fails_fast_when_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "official_exp"
            source.mkdir()
            (source / "epoch-50.pt").write_text("checkpoint", encoding="utf-8")
            models = Path(tmp) / "workspace_models"
            models.mkdir()
            env = {
                "SURE_BASELINE_CHECKPOINT_DIR": str(source),
                "SURE_BASELINE_EPOCH": "50",
                "SURE_BASELINE_AVG": "26",
            }
            with patch.object(zipformer_baseline, "MODELS_DIR", models), patch.dict(
                os.environ,
                env,
                clear=True,
            ):
                with self.assertRaises(FileNotFoundError) as ctx:
                    zipformer_baseline.train_if_needed()

        self.assertIn("epoch-24.pt", str(ctx.exception))

    def test_asr_mixed_research_guidance_uses_split_training_labels(self):
        task_card = SureTaskCard(
            task_id="asr_en_wer",
            canonical_task="asr",
            task_alias="asr",
            primary_metric="WER",
        )
        exp = ResearchExp(
            research_agent=None,
            config={
                "sure": {
                    "execution_mode": "mixed_local_vc",
                    "remote_training": {"enabled": True},
                }
            },
            initial_code="",
            exp_name="exp_0_research",
            task_card=task_card,
        )
        guidance = exp._candidate_type_guidance_text()
        self.assertIn("ASR", guidance)
        self.assertIn("Produce exactly 4", guidance)
        self.assertIn("2 `[fine_tune]` ideas", guidance)
        self.assertIn("2 `[arch]` ideas", guidance)
        self.assertIn("Zipformer", guidance)
        self.assertIn("VC child jobs", guidance)

    def test_remote_resource_profiles_resolve_with_documented_precedence(self):
        config = {
            "sure": {
                "remote_training": {
                    "gpu_per_task": 8,
                    "cpu_per_task": 64,
                    "mem_per_task": "256G",
                    "resource_profiles": {
                        "default": {"num_task": 2, "cpu_per_task": 16},
                        "training": {"cpu_per_task": 48},
                        "fine_tune": {"gpu_per_task": 4},
                        "draft_training": {"mem_per_task": "192G"},
                    },
                }
            }
        }
        resolved, diagnostics = remote_resource_config_from(
            config,
            candidate_type=FINE_TUNE,
            stage="draft",
            workload_profile="draft_training",
        )
        self.assertEqual(resolved["num_task"], 2)
        self.assertEqual(resolved["gpu_per_task"], 8)
        self.assertEqual(resolved["cpu_per_task"], 48)
        self.assertEqual(resolved["mem_per_task"], "192G")
        self.assertEqual(diagnostics["profile_name"], "draft_training")

        with patch.dict(os.environ, {"SURE_REMOTE_GPU_PER_TASK": "2"}, clear=False):
            overridden, _ = remote_resource_config_from(
                config,
                candidate_type=FINE_TUNE,
                stage="draft",
            )
        self.assertEqual(overridden["gpu_per_task"], "2")

    def test_draft_profile_uses_explicit_workload_not_stage_name(self):
        config = {
            "sure": {"remote_training": {"resource_profiles": {
                "inference": {"gpu_per_task": 1},
                "training": {"gpu_per_task": 8},
                "draft_training": {"mem_per_task": "192G"},
            }}}
        }
        inference, inference_diag = remote_resource_config_from(
            config, candidate_type=INFERENCE, stage="draft", workload_profile="inference"
        )
        training, training_diag = remote_resource_config_from(
            config, candidate_type=ARCH, stage="draft", workload_profile="draft_training"
        )
        self.assertEqual(inference["gpu_per_task"], 1)
        self.assertEqual(inference_diag["profile_name"], "inference")
        self.assertEqual(training["gpu_per_task"], 8)
        self.assertEqual(training["mem_per_task"], "192G")
        self.assertEqual(training_diag["profile_name"], "draft_training")

    def test_remote_inference_profile_drives_command_and_partition_requirement(self):
        config = {
            "sure": {
                "remote_training": {
                    "enabled": True,
                    "image": "image:test",
                    "partitions": ["small", "large"],
                    "resource_profiles": {
                        "inference": {
                            "gpu_per_task": 1,
                            "cpu_per_task": 8,
                            "mem_per_task": "32G",
                        }
                    },
                }
            }
        }
        executor = VcRemoteTrainingExecutor(
            config,
            config_path=Path(__file__),
            candidate_type=INFERENCE,
            stage="improve",
        )
        with patch.object(
            executor,
            "_probe_partition_snapshot",
            return_value=({"small": {"free_gpu": 1}, "large": {"free_gpu": 0}}, None),
        ):
            command = executor._build_vc_command(
                workspace=Path.cwd(),
                result_path=Path.cwd() / "result.json",
                exp_name="candidate",
                execution_timeout=10,
            )
        text = " ".join(command)
        self.assertIn("--gpu-per-task 1", text)
        self.assertIn("--cpu-per-task 8", text)
        self.assertIn("--mem-per-task 32G", text)
        self.assertEqual(executor.last_partition_selection["required_gpu_per_task"], 1)
        self.assertNotIn("ASR_WORLD_SIZE", text)
        self.assertNotIn("SURE_BASELINE_WORLD_SIZE", text)

    def test_mixed_remote_training_vc_command_uses_a10_template(self):
        config = {
            "sure": {
                "execution_mode": "mixed_local_vc",
                "task_id": "tts_en_wer",
                "remote_training": {
                    "enabled": True,
                    "image": "docker.v2.aispeech.com/sjtu/sjtu_yukai-chaolei-suremaster_f5tts:v1.3",
                    "partition": "pdgpu-a10",
                    "qos": "30m",
                    "num_task": 1,
                    "gpu_per_task": 8,
                    "cpu_per_task": 64,
                    "mem_per_task": "256G",
                    "volumes": ["/host:/host"],
                    "workdir": "/hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster",
                    "sync": True,
                    "debug": True,
                },
            }
        }
        self.assertTrue(mixed_execution_enabled(config, task_id="tts_en_wer"))
        executor = VcRemoteTrainingExecutor(config, config_path=Path(__file__))
        command = executor._build_vc_command(
            workspace=Path("/hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster/playground/sure_master/workspace_f5tts_mixed/exp_1"),
            result_path=Path("/hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster/playground/sure_master/workspace_f5tts_mixed/exp_1/metric/remote_training_result.json"),
            exp_name="exp_1_improve",
            execution_timeout=21600,
        )
        command_text = " ".join(shlex.quote(part) for part in command)
        self.assertIn("--gpu-per-task 8", command_text)
        self.assertIn("--cpu-per-task 64", command_text)
        self.assertIn("--mem-per-task 256G", command_text)
        self.assertNotIn("ASR_WORLD_SIZE", command_text)
        self.assertNotIn("SURE_BASELINE_WORLD_SIZE", command_text)
        self.assertIn("run_vc_sure_candidate.py", command_text)

    def test_remote_training_zero_submit_timeout_means_unlimited(self):
        config = {
            "sure": {
                "remote_training": {
                    "enabled": True,
                    "image": "image:test",
                    "partition": "pdgpu-a10",
                    "submit_timeout": 0,
                }
            }
        }
        executor = VcRemoteTrainingExecutor(config, config_path=Path(__file__))

        self.assertIsNone(executor._submit_timeout(0))
        self.assertIsNone(executor._submit_timeout(21600))

    def test_remote_training_missing_submit_timeout_keeps_finite_default(self):
        config = {
            "sure": {
                "remote_training": {
                    "enabled": True,
                    "image": "image:test",
                    "partition": "pdgpu-a10",
                }
            }
        }
        executor = VcRemoteTrainingExecutor(config, config_path=Path(__file__))

        self.assertEqual(executor._submit_timeout(0), 88200)
        self.assertEqual(executor._submit_timeout(21600), 23400)

    def test_remote_training_vc_command_preserves_zero_execution_timeout(self):
        config = {
            "sure": {
                "remote_training": {
                    "enabled": True,
                    "image": "image:test",
                    "partition": "pdgpu-a10",
                    "workdir": "/hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster",
                }
            }
        }
        executor = VcRemoteTrainingExecutor(config, config_path=Path(__file__))
        workspace = Path(
            "/hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster/runs/demo/workspaces/task_0/exp_1"
        )
        command = executor._build_vc_command(
            workspace=workspace,
            result_path=workspace / "metric" / "remote_training_result.json",
            exp_name="exp_1_draft",
            execution_timeout=0,
        )

        self.assertIn("--timeout 0", " ".join(command))

    def test_remote_training_vc_command_allows_env_resource_overrides(self):
        config = {
            "sure": {
                "execution_mode": "mixed_local_vc",
                "task_id": "asr_en_wer",
                "remote_training": {
                    "enabled": True,
                    "image": "docker.v2.aispeech.com/sjtu/sjtu_yukai-chaolei-suremaster_icefall:v1.0",
                    "partition": "pdgpu-a10",
                    "qos": "30m",
                    "num_task": 1,
                    "gpu_per_task": 8,
                    "cpu_per_task": 64,
                    "mem_per_task": "256G",
                    "max_parallel": 1,
                    "volumes": ["/host:/host"],
                    "workdir": "/hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster",
                    "sync": True,
                    "debug": True,
                },
            }
        }
        env = {
            "SURE_REMOTE_PARTITION": "pdgpu-4090",
            "SURE_REMOTE_GPU_PER_TASK": "4",
            "SURE_REMOTE_CPU_PER_TASK": "32",
            "SURE_REMOTE_MEM_PER_TASK": "128G",
            "SURE_REMOTE_MAX_PARALLEL": "2",
        }
        with patch.dict(os.environ, env, clear=False):
            self.assertTrue(mixed_execution_enabled(config, task_id="asr_en_wer"))
            self.assertEqual(remote_training_max_parallel(config), 2)
            executor = VcRemoteTrainingExecutor(config, config_path=Path(__file__))
            command = executor._build_vc_command(
                workspace=Path("/hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster/playground/sure_master/workspace_icefall/exp_1"),
                result_path=Path("/hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster/playground/sure_master/workspace_icefall/exp_1/metric/remote_training_result.json"),
                exp_name="exp_1_improve",
                execution_timeout=21600,
            )

        command_text = " ".join(shlex.quote(part) for part in command)
        self.assertIn("--partition pdgpu-4090", command_text)
        self.assertIn("--gpu-per-task 4", command_text)
        self.assertIn("--cpu-per-task 32", command_text)
        self.assertIn("--mem-per-task 128G", command_text)

    def test_remote_candidate_types_default_to_training_like_only(self):
        config = {
            "sure": {
                "execution_mode": "mixed_local_vc",
                "remote_training": {"enabled": True},
            }
        }
        self.assertEqual(remote_candidate_types_from(config), {FINE_TUNE, ARCH})
        self.assertTrue(candidate_runs_remotely(config, FINE_TUNE))
        self.assertTrue(candidate_runs_remotely(config, ARCH))
        self.assertTrue(candidate_runs_remotely(config, TRAINING))
        self.assertFalse(candidate_runs_remotely(config, INFERENCE))

    def test_remote_draft_enabled_does_not_remote_inference_candidates(self):
        config = {
            "sure": {
                "execution_mode": "mixed_local_vc",
                "remote_training": {"enabled": True, "draft_enabled": True},
            }
        }
        self.assertEqual(remote_candidate_types_from(config), {FINE_TUNE, ARCH})
        self.assertFalse(candidate_runs_remotely(config, INFERENCE))
        self.assertTrue(draft_runs_remotely(config))

    def test_draft_runs_remotely_requires_remote_training_and_draft_flag(self):
        disabled = {"sure": {"remote_training": {"enabled": False, "draft_enabled": True}}}
        enabled_without_draft = {"sure": {"remote_training": {"enabled": True}}}
        enabled_with_draft = {"sure": {"remote_training": {"enabled": True, "draft_enabled": True}}}
        enabled_with_legacy_name = {
            "sure": {"remote_training": {"enabled": True, "draft_training_enabled": "yes"}}
        }
        self.assertFalse(draft_runs_remotely(disabled))
        self.assertFalse(draft_runs_remotely(enabled_without_draft))
        self.assertTrue(draft_runs_remotely(enabled_with_draft))
        self.assertTrue(draft_runs_remotely(enabled_with_legacy_name))

    def test_draft_runs_remotely_supports_env_override(self):
        config = {"sure": {"remote_training": {"enabled": True}}}
        with patch.dict(os.environ, {"SURE_REMOTE_DRAFT_ENABLED": "1"}, clear=False):
            self.assertTrue(draft_runs_remotely(config))

    def test_draft_candidate_type_hint_uses_fine_tune_only_when_draft_remote_enabled(self):
        playground = SureMasterPlayground.__new__(SureMasterPlayground)
        playground.config = {"sure": {"remote_training": {"enabled": True}}}
        self.assertEqual(playground._draft_candidate_type_hint(), INFERENCE)

        playground.config = {"sure": {"remote_training": {"enabled": True, "draft_enabled": True}}}
        self.assertEqual(playground._draft_candidate_type_hint(), FINE_TUNE)

    def test_remote_candidate_types_can_include_inference(self):
        config = {
            "sure": {
                "execution_mode": "mixed_local_vc",
                "remote_training": {
                    "enabled": True,
                    "candidate_types": ["inference", "training"],
                    "max_parallel": 4,
                },
            },
            "session": {
                "local": {
                    "parallel": {
                        "enabled": True,
                        "max_parallel": 4,
                    }
                }
            },
        }
        self.assertEqual(remote_candidate_types_from(config), {INFERENCE, FINE_TUNE, ARCH})
        self.assertTrue(candidate_runs_remotely(config, INFERENCE))
        self.assertTrue(candidate_runs_remotely(config, FINE_TUNE))
        self.assertTrue(candidate_runs_remotely(config, ARCH))
        self.assertTrue(candidate_runs_remotely(config, TRAINING))
        playground = SureMasterPlayground.__new__(SureMasterPlayground)
        playground.config = config
        playground.task_card = SureTaskCard(
            task_id="tts_en_wer",
            canonical_task="tts",
            task_alias="tts",
            primary_metric="tts_wer",
        )
        self.assertEqual(playground._idea_max_workers(config["session"]["local"]["parallel"]), 4)

    def test_mixed_candidate_limits_default_to_four_two_two(self):
        playground = SureMasterPlayground.__new__(SureMasterPlayground)
        playground.config = {
            "sure": {
                "execution_mode": "mixed_local_vc",
                "remote_training": {"enabled": True},
            }
        }
        playground.task_card = SureTaskCard(
            task_id="tts_en_wer",
            canonical_task="tts",
            task_alias="tts",
            primary_metric="tts_wer",
        )
        self.assertEqual(
            playground._candidate_limits(),
            {INFERENCE: 4, FINE_TUNE: 2, ARCH: 2},
        )





    def test_tedlium3_smoke_starts_at_real_arch_training_probe(self):
        config_path = (
            Path(__file__).resolve().parents[4]
            / "configs"
            / "sure_master"
            / "archive/staged_axes/gpt-5-icefall-tedlium3-smoke-staged-axes-mixed.yaml"
        )
        sure_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["sure"]
        self.assertEqual(sure_config["staged_axes"]["start_phase"], "arch")
        self.assertFalse(sure_config["remote_training"]["draft_enabled"])
        arch_env = sure_config["staged_axes"]["axes"]["arch"]["rungs"][0][
            "execution_env"
        ]
        self.assertEqual(arch_env["SURE_STAGED_TARGET_EPOCH"], "1")
        self.assertEqual(arch_env["SURE_MAX_DURATION"], "auto")
        self.assertEqual(arch_env["SURE_DURATION_AUTOTUNE"], "1")
        self.assertEqual(arch_env["SURE_DURATION_PROBE_SUCCESS_BATCHES"], "1")
        self.assertFalse(
            any(key.startswith("SURE_STAGED_RESUME_") for key in arch_env)
        )
        self.assertEqual(sure_config["execution_env"]["SURE_BASELINE_CHECKPOINT_DIR"], "")
        self.assertEqual(sure_config["execution_env"]["SURE_BASELINE_USE_PRETRAINED"], "0")



    def test_mixed_local_icefall_python_does_not_override_remote_config(self):
        playground = SureMasterPlayground.__new__(SureMasterPlayground)
        playground.sure_config = {
            "execution_env": {
                "SURE_ICEFALL_PYTHON": "/opt/conda/envs/icefall/bin/python",
                "SURE_MAX_DURATION": "300",
            }
        }

        with patch.dict(
            os.environ,
            {
                "SURE_ICEFALL_PYTHON": "/hpc_stor03/local/anaconda3/envs/icefall/bin/python",
                "SURE_LOCAL_ICEFALL_PYTHON": "/hpc_stor03/local/anaconda3/envs/icefall/bin/python",
            },
            clear=True,
        ):
            env = playground._execution_env()

        self.assertEqual(env["SURE_ICEFALL_PYTHON"], "/opt/conda/envs/icefall/bin/python")
        self.assertEqual(
            env["SURE_LOCAL_ICEFALL_PYTHON"],
            "/hpc_stor03/local/anaconda3/envs/icefall/bin/python",
        )

    def test_remote_icefall_python_override_uses_remote_specific_env(self):
        playground = SureMasterPlayground.__new__(SureMasterPlayground)
        playground.sure_config = {
            "execution_env": {
                "SURE_ICEFALL_PYTHON": "/opt/conda/envs/icefall/bin/python",
            }
        }

        with patch.dict(
            os.environ,
            {"SURE_REMOTE_ICEFALL_PYTHON": "/custom/container/icefall/bin/python"},
            clear=True,
        ):
            env = playground._execution_env()

        self.assertEqual(env["SURE_ICEFALL_PYTHON"], "/custom/container/icefall/bin/python")














    def test_parse_vc_info_partitions_extracts_free_gpu(self):
        output = """
-----------------------------------------------------------------------------------------
Partition          | gpu(allocated/total) | cpu(allocated/total) | mem(allocated/total)
-----------------------------------------------------------------------------------------
pdgpu-3090         | 112/112              | 696/1080             | 3208Gi/6930.0Gi
pdgpu-4090         | 56/64                | 448/752              | 1792Gi/3452.0Gi
pdgpu-a10          | 184/184              | 1152/1748            | 3648Gi/11385.0Gi
"""
        snapshot = parse_vc_info_partitions(output)
        self.assertEqual(snapshot["pdgpu-3090"]["free_gpu"], 0)
        self.assertEqual(snapshot["pdgpu-4090"]["free_gpu"], 8)
        self.assertEqual(snapshot["pdgpu-a10"]["total_gpu"], 184)

    def test_remote_training_selects_partition_with_most_free_gpu(self):
        config = {
            "sure": {
                "execution_mode": "mixed_local_vc",
                "task_id": "tts_en_wer",
                "remote_training": {
                    "enabled": True,
                    "image": "docker.v2.aispeech.com/sjtu/sjtu_yukai-chaolei-suremaster_f5tts:v1.3",
                    "partition": "pdgpu-a10",
                    "qos": "30m",
                    "num_task": 1,
                    "gpu_per_task": 8,
                    "cpu_per_task": 64,
                    "mem_per_task": "256G",
                    "volumes": ["/host:/host"],
                    "workdir": "/hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster",
                    "sync": True,
                    "debug": True,
                },
            }
        }
        vc_info_output = """
Partition          | gpu(allocated/total) | cpu(allocated/total) | mem(allocated/total)
pdgpu-3090         | 112/112              | 696/1080             | 3208Gi/6930.0Gi
pdgpu-4090         | 56/64                | 448/752              | 1792Gi/3452.0Gi
pdgpu-a10          | 184/184              | 1152/1748            | 3648Gi/11385.0Gi
"""
        env = {
            "SURE_REMOTE_PARTITION": "",
            "SURE_REMOTE_PARTITIONS": "pdgpu-3090,pdgpu-4090,pdgpu-a10",
            "SURE_REMOTE_PARTITION_POLICY": "most_free_gpu",
            "SURE_REMOTE_GPU_PER_TASK": "8",
        }
        with patch.dict(os.environ, env, clear=False), patch(
            "playground.sure_master.core.utils.vc_remote.subprocess.run",
            return_value=subprocess.CompletedProcess(["vc", "info"], 0, stdout=vc_info_output),
        ):
            executor = VcRemoteTrainingExecutor(config, config_path=Path(__file__))
            command = executor._build_vc_command(
                workspace=Path("/hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster/playground/sure_master/workspace_f5tts_mixed/exp_1"),
                result_path=Path("/hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster/playground/sure_master/workspace_f5tts_mixed/exp_1/metric/remote_training_result.json"),
                exp_name="exp_1_improve",
                execution_timeout=21600,
            )

        command_text = " ".join(shlex.quote(part) for part in command)
        self.assertIn("--partition pdgpu-4090", command_text)
        self.assertFalse(executor.last_partition_selection["fallback_used"])
        self.assertEqual(executor.last_partition_selection["selected_partition"], "pdgpu-4090")
        self.assertEqual(
            executor.last_partition_selection["partition_snapshot"]["pdgpu-4090"]["free_gpu"],
            8,
        )

    def test_remote_training_partition_probe_failure_falls_back_to_first_candidate(self):
        config = {
            "sure": {
                "execution_mode": "mixed_local_vc",
                "task_id": "tts_en_wer",
                "remote_training": {
                    "enabled": True,
                    "image": "docker.v2.aispeech.com/sjtu/sjtu_yukai-chaolei-suremaster_f5tts:v1.3",
                    "partition": "pdgpu-a10",
                    "gpu_per_task": 8,
                    "workdir": "/hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster",
                },
            }
        }
        env = {
            "SURE_REMOTE_PARTITION": "",
            "SURE_REMOTE_PARTITIONS": "pdgpu-3090,pdgpu-4090,pdgpu-a10",
            "SURE_REMOTE_PARTITION_POLICY": "most_free_gpu",
            "SURE_REMOTE_PARTITION_FALLBACK": "queue_first",
            "SURE_REMOTE_GPU_PER_TASK": "8",
        }
        with patch.dict(os.environ, env, clear=False), patch(
            "playground.sure_master.core.utils.vc_remote.subprocess.run",
            return_value=subprocess.CompletedProcess(
                ["vc", "info"],
                1,
                stdout="PermissionError: [Errno 1] Operation not permitted",
            ),
        ):
            executor = VcRemoteTrainingExecutor(config, config_path=Path(__file__))
            command = executor._build_vc_command(
                workspace=Path("/hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster/playground/sure_master/workspace_f5tts_mixed/exp_1"),
                result_path=Path("/hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster/playground/sure_master/workspace_f5tts_mixed/exp_1/metric/remote_training_result.json"),
                exp_name="exp_1_improve",
                execution_timeout=21600,
            )

        command_text = " ".join(shlex.quote(part) for part in command)
        self.assertIn("--partition pdgpu-3090", command_text)
        self.assertTrue(executor.last_partition_selection["fallback_used"])
        self.assertIn("vc info exited non-zero", executor.last_partition_selection["partition_probe_error"])

    def test_remote_submit_timeout_recovers_late_result(self):
        config = {
            "sure": {
                "remote_training": {
                    "enabled": True,
                    "image": "image:test",
                    "partition": "pdgpu-a10",
                    "workdir": str(Path.cwd()),
                    "runner": "playground/sure_master/tools/run_vc_sure_candidate.py",
                    "submit_timeout": 1,
                    "result_recovery": {
                        "enabled": True,
                        "grace_seconds": 0.1,
                        "poll_seconds": 0.01,
                    },
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "exp"
            result_path = workspace / "metric" / "remote_training_result.json"
            result_path.parent.mkdir(parents=True)

            def timeout_after_result(*args, **kwargs):
                result_path.write_text(
                    json.dumps(
                        {
                            "success": True,
                            "score": 0.123,
                            "code": "print('ok')",
                            "details": {"reason_code": "success"},
                        }
                    ),
                    encoding="utf-8",
                )
                raise subprocess.TimeoutExpired(args[0], timeout=1, output="still running")

            executor = VcRemoteTrainingExecutor(config, config_path=Path(__file__))
            with patch(
                "playground.sure_master.core.utils.vc_remote.subprocess.run",
                side_effect=timeout_after_result,
            ):
                payload = executor.run(
                    workspace_path=workspace,
                    exp_name="exp_1",
                    execution_timeout=60,
                )

        self.assertTrue(payload["success"])
        self.assertEqual(payload["score"], 0.123)
        self.assertTrue(payload["execution_info"]["submit_timed_out"])
        self.assertTrue(payload["execution_info"]["recovered_after_submit_timeout"])

    def test_remote_submit_timeout_remains_structured_failure_without_late_result(self):
        config = {
            "sure": {
                "remote_training": {
                    "enabled": True,
                    "image": "image:test",
                    "partition": "pdgpu-a10",
                    "workdir": str(Path.cwd()),
                    "runner": "playground/sure_master/tools/run_vc_sure_candidate.py",
                    "submit_timeout": 1,
                    "result_recovery": {
                        "enabled": True,
                        "grace_seconds": 0.01,
                        "poll_seconds": 0.001,
                    },
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "exp"
            (workspace / "metric").mkdir(parents=True)
            executor = VcRemoteTrainingExecutor(config, config_path=Path(__file__))
            with patch(
                "playground.sure_master.core.utils.vc_remote.subprocess.run",
                side_effect=subprocess.TimeoutExpired(["vc", "submit"], timeout=1, output="queued"),
            ):
                payload = executor.run(
                    workspace_path=workspace,
                    exp_name="exp_1",
                    execution_timeout=60,
                )

        self.assertFalse(payload["success"])
        self.assertEqual(payload["reason_code"], "remote_submit_timeout")

    def test_remote_missing_result_can_recover_after_submit_returns(self):
        config = {
            "sure": {
                "remote_training": {
                    "enabled": True,
                    "image": "image:test",
                    "partition": "pdgpu-a10",
                    "workdir": str(Path.cwd()),
                    "runner": "playground/sure_master/tools/run_vc_sure_candidate.py",
                    "sync": True,
                    "debug": True,
                    "result_recovery": {
                        "enabled": True,
                        "grace_seconds": 0.1,
                        "poll_seconds": 0.01,
                        "recover_missing_result": True,
                    },
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "exp"
            result_path = workspace / "metric" / "remote_training_result.json"
            result_path.parent.mkdir(parents=True)

            def write_delayed_result(seconds):
                result_path.write_text(
                    json.dumps(
                        {
                            "success": True,
                            "score": 0.456,
                            "code": "print('late')",
                            "details": {"reason_code": "success"},
                        }
                    ),
                    encoding="utf-8",
                )

            executor = VcRemoteTrainingExecutor(config, config_path=Path(__file__))
            with patch(
                "playground.sure_master.core.utils.vc_remote.subprocess.run",
                return_value=subprocess.CompletedProcess(["vc", "submit"], 0, stdout="job done"),
            ), patch("playground.sure_master.core.utils.vc_remote.time.sleep", side_effect=write_delayed_result):
                payload = executor.run(
                    workspace_path=workspace,
                    exp_name="exp_1",
                    execution_timeout=60,
                )

        self.assertTrue(payload["success"])
        self.assertEqual(payload["score"], 0.456)
        self.assertTrue(payload["execution_info"]["recovered_after_missing_result"])

    def test_remote_training_missing_result_is_structured_failure(self):
        config = {
            "sure": {
                "remote_training": {
                    "enabled": True,
                    "image": "image:test",
                    "partition": "pdgpu-a10",
                    "workdir": str(Path.cwd()),
                    "runner": "playground/sure_master/tools/run_vc_sure_candidate.py",
                    "sync": True,
                    "debug": True,
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "exp"
            (workspace / "metric").mkdir(parents=True)
            executor = VcRemoteTrainingExecutor(config, config_path=Path(__file__))
            with patch(
                "playground.sure_master.core.utils.vc_remote.subprocess.run",
                return_value=subprocess.CompletedProcess(["vc", "submit"], 0, stdout="job done"),
            ):
                payload = executor.run(
                    workspace_path=workspace,
                    exp_name="exp_1",
                    execution_timeout=60,
                )
        self.assertFalse(payload["success"])
        self.assertEqual(payload["reason_code"], "remote_result_missing")
        self.assertEqual(payload["execution_info"]["vc_exit_code"], 0)
        self.assertIn("job done", payload["execution_info"]["vc_submit_log_tail"])

    def test_scheduler_allocation_overrides_stale_world_sizes(self):
        execution_env = {"ASR_WORLD_SIZE": "8", "SURE_BASELINE_WORLD_SIZE": "8"}
        env = {
            "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b",
            "SURE_VC_RESOURCE_PROFILE": "inference",
            "SURE_VC_REQUESTED_RESOURCES": '{"gpu_per_task":4}',
        }
        with patch.dict(os.environ, env, clear=True):
            allocation = run_vc_sure_candidate.apply_scheduler_allocation(
                execution_env, asr_workload=True
            )
        self.assertEqual(execution_env["ASR_WORLD_SIZE"], "2")
        self.assertEqual(execution_env["SURE_BASELINE_WORLD_SIZE"], "2")
        self.assertEqual(allocation["gpu_count"], 2)
        self.assertEqual(allocation["requested_resources"]["gpu_per_task"], 4)

    def test_scheduler_allocation_does_not_inject_asr_world_size_for_f5(self):
        execution_env = {"ASR_WORLD_SIZE": "8", "SURE_BASELINE_WORLD_SIZE": "8"}
        env = {"CUDA_VISIBLE_DEVICES": "0,1"}
        allocation = run_vc_sure_candidate.apply_scheduler_allocation(
            execution_env, asr_workload=False, env=env
        )
        self.assertEqual(allocation["gpu_count"], 2)
        self.assertNotIn("ASR_WORLD_SIZE", execution_env)
        self.assertNotIn("SURE_BASELINE_WORLD_SIZE", execution_env)

    def test_scheduler_allocation_removes_stale_world_sizes_without_gpu(self):
        execution_env = {"ASR_WORLD_SIZE": "8", "SURE_BASELINE_WORLD_SIZE": "8"}
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1"}, clear=True):
            allocation = run_vc_sure_candidate.apply_scheduler_allocation(execution_env)
        self.assertEqual(allocation["gpu_count"], 0)
        self.assertNotIn("ASR_WORLD_SIZE", execution_env)
        self.assertNotIn("SURE_BASELINE_WORLD_SIZE", execution_env)

    def test_scheduler_allocation_uses_exact_nvidia_visibility(self):
        execution_env = {"ASR_WORLD_SIZE": "8"}
        allocation = run_vc_sure_candidate.apply_scheduler_allocation(
            execution_env,
            env={
                "CUDA_VISIBLE_DEVICES": "",
                "NVIDIA_VISIBLE_DEVICES": "GPU-one,MIG-GPU-two/1/0",
            },
        )
        self.assertEqual(allocation["allocation_source"], "NVIDIA_VISIBLE_DEVICES")
        self.assertEqual(allocation["visibility_kind"], "exact")
        self.assertEqual(allocation["gpu_count"], 2)
        self.assertEqual(execution_env["CUDA_VISIBLE_DEVICES"], "GPU-one,MIG-GPU-two/1/0")
        self.assertEqual(execution_env["ASR_WORLD_SIZE"], "2")

    def test_scheduler_allocation_uses_gpu_per_task_after_empty_visibility(self):
        execution_env: dict[str, str] = {}
        allocation = run_vc_sure_candidate.apply_scheduler_allocation(
            execution_env,
            env={
                "CUDA_VISIBLE_DEVICES": "",
                "NVIDIA_VISIBLE_DEVICES": "all",
                "GPU_PER_TASK": "2",
                "GPU_NUM": "8",
            },
        )
        self.assertEqual(allocation["allocation_source"], "GPU_PER_TASK")
        self.assertEqual(allocation["visibility_kind"], "task_count")
        self.assertEqual(allocation["gpu_count"], 2)
        self.assertEqual(allocation["gpu_devices"], [])
        self.assertEqual(allocation["effective_visible_devices"], "")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", execution_env)
        self.assertEqual(execution_env["SURE_BASELINE_WORLD_SIZE"], "2")
        normalized = run_vc_sure_candidate.normalize_process_gpu_environment(
            allocation,
            {"CUDA_VISIBLE_DEVICES": "", "GPU_PER_TASK": "2"},
        )
        self.assertNotIn("CUDA_VISIBLE_DEVICES", normalized)

    def test_scheduler_allocation_uses_gpu_num_when_task_count_is_invalid(self):
        execution_env: dict[str, str] = {}
        allocation = run_vc_sure_candidate.apply_scheduler_allocation(
            execution_env,
            asr_workload=False,
            env={"CUDA_VISIBLE_DEVICES": "", "GPU_PER_TASK": "bad", "GPU_NUM": "1"},
        )
        self.assertEqual(allocation["allocation_source"], "GPU_NUM")
        self.assertEqual(allocation["gpu_count"], 1)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", execution_env)
        self.assertNotIn("ASR_WORLD_SIZE", execution_env)

    def test_scheduler_count_allocation_sets_world_size_without_fabricated_mask(self):
        execution_env = {"ASR_WORLD_SIZE": "8", "SURE_BASELINE_WORLD_SIZE": "8"}
        allocation = run_vc_sure_candidate.apply_scheduler_allocation(
            execution_env,
            env={"CUDA_VISIBLE_DEVICES": "", "GPU_PER_TASK": "2"},
        )
        self.assertEqual(allocation["gpu_count"], 2)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", execution_env)
        self.assertEqual(execution_env["ASR_WORLD_SIZE"], "2")
        self.assertEqual(execution_env["SURE_BASELINE_WORLD_SIZE"], "2")

    def test_scheduler_count_allocation_invalid_values_leave_no_gpu(self):
        for value in ("0", "-1", "bad", ""):
            execution_env = {"ASR_WORLD_SIZE": "8", "SURE_BASELINE_WORLD_SIZE": "8"}
            allocation = run_vc_sure_candidate.apply_scheduler_allocation(
                execution_env,
                env={
                    "CUDA_VISIBLE_DEVICES": "",
                    "GPU_PER_TASK": value,
                    "GPU_NUM": value,
                },
            )
            self.assertEqual(allocation["gpu_count"], 0)
            self.assertNotIn("ASR_WORLD_SIZE", execution_env)
            self.assertNotIn("SURE_BASELINE_WORLD_SIZE", execution_env)

    def test_remote_candidate_context_filters_local_icefall_python(self):
        sanitized = run_vc_sure_candidate.sanitize_remote_execution_env(
            {
                "SURE_ICEFALL_PYTHON": "/hpc/local/anaconda3/envs/icefall/bin/python",
                "SURE_LOCAL_ICEFALL_PYTHON": "/hpc/local/anaconda3/envs/icefall/bin/python",
                "SURE_REMOTE_ICEFALL_PYTHON": "/opt/conda/envs/icefall/bin/python",
                "SURE_MAX_DURATION": "300",
            }
        )

        self.assertEqual(sanitized["SURE_ICEFALL_PYTHON"], "/opt/conda/envs/icefall/bin/python")
        self.assertNotIn("SURE_LOCAL_ICEFALL_PYTHON", sanitized)
        self.assertEqual(sanitized["SURE_MAX_DURATION"], "300")

    def test_remote_candidate_cudnn_failure_gets_training_runtime_reason(self):
        output = "RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED"
        self.assertEqual(
            run_vc_sure_candidate.execution_failure_reason_code(output),
            "training_runtime_failed",
        )
        details = run_vc_sure_candidate.execution_failure_details({"exit_code": 1}, output)
        self.assertEqual(details["fatal_error"], "cudnn_status_execution_failed")

    def test_remote_candidate_duration_startup_failure_gets_specific_reason(self):
        output = (
            "[candidate:error] duration_autotune failed with exit code 1\n"
            "ModuleNotFoundError: No module named 'icefall'\n"
        )
        self.assertEqual(
            run_vc_sure_candidate.execution_failure_reason_code(output),
            "duration_probe_startup_failed",
        )
        details = run_vc_sure_candidate.execution_failure_details({"exit_code": 1}, output)
        self.assertEqual(details["fatal_error"], "duration_probe_startup_failed")

    def test_remote_candidate_duration_contract_markers_preserve_details_and_category(self):
        cases = (
            (
                "duration_probe_help_discovery_failed",
                {"error": "timeout", "timeout_seconds": 30, "train_py": "/recipe/train.py"},
                "system_failure",
            ),
            (
                "duration_probe_helper_cli_incompatible",
                {"unsupported_options": ["--max-duration"], "train_py": "/recipe/train.py"},
                "system_failure",
            ),
            (
                "duration_probe_candidate_cli_incompatible",
                {"unsupported_options": ["--encoder-dim"], "train_py": "/recipe/train.py"},
                "candidate_failure",
            ),
        )
        for marker, payload, category in cases:
            with self.subTest(marker=marker):
                output = f"[sure_runtime] {marker}: {json.dumps(payload, sort_keys=True)}"
                self.assertEqual(
                    run_vc_sure_candidate.execution_failure_reason_code(output),
                    marker,
                )
                details = run_vc_sure_candidate.execution_failure_details(
                    {"exit_code": 1},
                    output,
                )
                self.assertEqual(details["fatal_error"], marker)
                for key, value in payload.items():
                    self.assertEqual(details[key], value)
                self.assertEqual(
                    SureMasterPlayground._failure_category_from_reason(marker),
                    category,
                )

    def test_remote_candidate_duration_contract_marker_cannot_override_fatal_error(self):
        marker = "duration_probe_helper_cli_incompatible"
        output = f'{marker}: {{"fatal_error": "spoofed", "train_py": "/recipe/train.py"}}'
        details = run_vc_sure_candidate.execution_failure_details(
            {"exit_code": 1},
            output,
        )
        self.assertEqual(details["fatal_error"], marker)

    def test_remote_candidate_timeout_takes_precedence_over_duration_log_text(self):
        output = "export SURE_DURATION_CACHE_DIR=/tmp/duration_autotune"
        execution_info = {"exit_code": -1, "timed_out": True}

        self.assertEqual(
            run_vc_sure_candidate.execution_failure_reason_code(
                output,
                execution_info,
            ),
            "candidate_execution_timeout",
        )
        details = run_vc_sure_candidate.execution_failure_details(
            execution_info,
            output,
        )
        self.assertEqual(details["fatal_error"], "candidate_execution_timeout")
        self.assertEqual(
            SureMasterPlayground._failure_category_from_reason(
                "candidate_execution_timeout"
            ),
            "system_failure",
        )

    def test_remote_candidate_bpe_validation_failure_takes_precedence(self):
        output = (
            "export SURE_DURATION_CACHE_DIR=/tmp/duration_autotune\n"
            "ValueError: Zipformer train/decode BPE mismatch: "
            "train_decode candidates must decode with the same BPE model used for training.\n"
        )
        self.assertEqual(
            run_vc_sure_candidate.execution_failure_reason_code(output),
            "candidate_bpe_validation_failed",
        )
        details = run_vc_sure_candidate.execution_failure_details({"exit_code": 1}, output)
        self.assertEqual(details["fatal_error"], "candidate_bpe_validation_failed")

    def test_remote_child_count_only_allocation_removes_blank_cuda_mask(self):
        execution_env: dict[str, str] = {}
        inherited_env = {"CUDA_VISIBLE_DEVICES": "", "GPU_PER_TASK": "2"}
        allocation = run_vc_sure_candidate.apply_scheduler_allocation(
            execution_env,
            env=inherited_env,
        )
        child_env = run_vc_sure_candidate.normalize_process_gpu_environment(
            allocation,
            inherited_env,
        )
        proc = Mock()
        proc.wait.return_value = 0
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            run_vc_sure_candidate.subprocess,
            "Popen",
            return_value=proc,
        ) as popen_mock:
            run_vc_sure_candidate.run_logged(
                "python run_sure.py",
                cwd=Path(tmp),
                timeout=0,
                log_path=Path(tmp) / "run.log",
                env=child_env,
            )

        self.assertNotIn("CUDA_VISIBLE_DEVICES", child_env)
        self.assertEqual(child_env["GPU_PER_TASK"], "2")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", popen_mock.call_args.kwargs["env"])

    def test_count_only_allocation_drives_icefall_world_size(self):
        execution_env: dict[str, str] = {}
        inherited_env = {"CUDA_VISIBLE_DEVICES": "", "GPU_PER_TASK": "2"}
        allocation = run_vc_sure_candidate.apply_scheduler_allocation(
            execution_env,
            env=inherited_env,
        )
        process_env = run_vc_sure_candidate.normalize_process_gpu_environment(
            allocation,
            inherited_env,
        )
        process_env.update(execution_env)
        with patch.dict(os.environ, process_env, clear=True):
            self.assertEqual(run_icefall_zipformer_candidate.current_world_size(), 2)

    def test_count_only_f5_inference_defaults_to_one_worker(self):
        allocation = run_vc_sure_candidate.scheduler_gpu_allocation(
            {"CUDA_VISIBLE_DEVICES": "", "GPU_PER_TASK": "1"}
        )
        process_env = run_vc_sure_candidate.normalize_process_gpu_environment(
            allocation,
            {"CUDA_VISIBLE_DEVICES": "", "GPU_PER_TASK": "1"},
        )
        with patch.dict(os.environ, process_env, clear=True):
            self.assertEqual(
                run_f5tts_batch_infer.choose_worker_count("auto", 4, "cuda"),
                1,
            )

    def test_remote_child_zero_timeout_waits_without_deadline(self):
        proc = Mock()
        proc.wait.return_value = 0
        child_env = {"GPU_PER_TASK": "2"}
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            run_vc_sure_candidate.subprocess,
            "Popen",
            return_value=proc,
        ) as popen_mock:
            result = run_vc_sure_candidate.run_logged(
                "python run_sure.py",
                cwd=Path(tmp),
                timeout=0,
                log_path=Path(tmp) / "run.log",
                env=child_env,
            )

        proc.wait.assert_called_once_with(timeout=None)
        self.assertEqual(popen_mock.call_args.kwargs["env"], child_env)
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["timed_out"])

    def test_remote_training_source_snapshot_uses_snapshot_for_runner_not_workspace(self):
        project_root = Path(__file__).resolve().parents[4]
        workdir = Path("/hpc_stor03/sure-test-project")
        snapshot = workdir / "runs/demo/source_snapshot"
        config = {
            "sure": {
                "source_snapshot": {"enabled": True, "path": str(snapshot)},
                "remote_training": {
                    "enabled": True,
                    "image": "image:test",
                    "partition": "pdgpu-a10",
                    "workdir": str(workdir),
                    "runner": "playground/sure_master/tools/run_vc_sure_candidate.py",
                },
            }
        }
        executor = VcRemoteTrainingExecutor(config, config_path=Path(__file__))
        workspace = project_root / "runs/demo/workspaces/task_0/exp_1"
        command = executor._build_vc_command(
            workspace=workspace,
            result_path=workspace / "metric" / "remote_training_result.json",
            exp_name="exp_1",
            execution_timeout=60,
        )
        command_text = " ".join(shlex.quote(part) for part in command)
        self.assertIn(f"--dir {shlex.quote(str(snapshot))}", command_text)
        self.assertIn(f"cd {shlex.quote(str(snapshot))}", command_text)
        self.assertIn("playground/sure_master/tools/run_vc_sure_candidate.py", command_text)
        self.assertIn(str(workdir / "runs/demo/workspaces/task_0/exp_1"), command_text)

    def test_zipformer_baseline_normalizes_librispeech_cut_ids(self):
        self.assertEqual(
            normalize_librispeech_cut_id("1089-134686-0000-0"),
            "1089-134686-0000",
        )
        self.assertEqual(normalize_librispeech_cut_id("custom-utt"), "custom-utt")

    def test_asr_librispeech_ref_splitter_keeps_tiers_disjoint(self):
        rows = [
            RefRow("dev-clean", f"spk1-1-{index:04d}", "TEXT", "spk1")
            for index in range(2)
        ]
        rows += [
            RefRow("dev-clean", f"spk2-1-{index:04d}", "TEXT", "spk2")
            for index in range(3)
        ]
        rows += [
            RefRow("dev-clean", f"spk3-1-{index:04d}", "TEXT", "spk3")
            for index in range(4)
        ]

        selected = select_tiered_rows(
            rows,
            {"smoke": 2, "early": 3},
            seed=20260714,
            split="dev-clean",
        )

        self.assertEqual(len(selected["smoke"]), 2)
        self.assertEqual(len(selected["early"]), 3)
        self.assertEqual(len(selected["selection"]), 4)
        tier_keys = [
            {row.key for row in selected[tier]}
            for tier in ("smoke", "early", "selection")
        ]
        self.assertFalse(tier_keys[0] & tier_keys[1])
        self.assertFalse(tier_keys[0] & tier_keys[2])
        self.assertFalse(tier_keys[1] & tier_keys[2])
        tier_groups = [
            {row.group for row in selected[tier]}
            for tier in ("smoke", "early", "selection")
        ]
        self.assertFalse(tier_groups[0] & tier_groups[1])
        self.assertFalse(tier_groups[0] & tier_groups[2])
        self.assertFalse(tier_groups[1] & tier_groups[2])

    def test_asr_profile_ref_builder_uses_tedlium3_identity_keys(self):
        profile = get_asr_dataset_profile("tedlium3")
        with tempfile.TemporaryDirectory() as tmp:
            manifest_dir = Path(tmp) / "manifests"
            output_dir = Path(tmp) / "refs"
            manifest_dir.mkdir()
            rows = [
                {
                    "id": f"talk-{index}",
                    "recording": {"id": f"talk-{index // 2}"},
                    "supervisions": [
                        {
                            "id": f"speaker-{index}-segment-{index}",
                            "speaker": f"speaker-{index}",
                            "text": f"TEDLIUM TEXT {index}",
                        }
                    ],
                }
                for index in range(8)
            ]
            manifest_path = manifest_dir / profile.manifest_patterns["dev"]
            import gzip

            with gzip.open(manifest_path, "wt", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            counts = build_asr_refs.build_refs(
                "tedlium3",
                manifest_dir,
                output_dir,
                seed=20260714,
                tier_sizes={"smoke": 2, "early": 2, "regular": 2},
            )
            build_asr_refs.validate_refs(output_dir, profile)

            regular_ref = output_dir / "asr_tedlium3_regular_ref.txt"
            regular_text = regular_ref.read_text(encoding="utf-8")

        self.assertEqual(counts["regular"], 2)
        self.assertIn("-segment-", regular_text)
        self.assertEqual(normalize_asr_cut_id("speaker-1-segment-9", profile), "speaker-1-segment-9")

    def test_sure_auto_gpu_config_treats_null_as_all_visible_gpus(self):
        local_config = {
            "gpu_devices": None,
            "parallel": {
                "enabled": True,
                "max_parallel": 1,
            },
        }
        with patch.object(
            sure_playground_module,
            "_discover_gpu_devices",
            return_value=["0", "1", "2", "3"],
        ):
            resolved = sure_playground_module._apply_auto_gpu_config(
                local_config,
                sure_playground_module.logging.getLogger("test"),
            )

        self.assertEqual(resolved["gpu_devices"], ["0", "1", "2", "3"])
        self.assertEqual(resolved["parallel"]["max_parallel"], 1)
        self.assertEqual(resolved["parallel"]["gpus_per_exp"], 4)
        self.assertTrue(resolved["parallel"]["set_asr_world_size"])

    def test_sure_auto_gpu_config_derives_parallel_from_gpus_per_exp(self):
        local_config = {
            "gpu_devices": "auto",
            "parallel": {
                "enabled": True,
                "max_parallel": "auto",
                "gpus_per_exp": 4,
            },
        }
        with patch.object(
            sure_playground_module,
            "_discover_gpu_devices",
            return_value=["0", "1", "2", "3", "4", "5", "6", "7"],
        ):
            resolved = sure_playground_module._apply_auto_gpu_config(
                local_config,
                sure_playground_module.logging.getLogger("test"),
            )

        self.assertEqual(resolved["gpu_devices"], ["0", "1", "2", "3", "4", "5", "6", "7"])
        self.assertEqual(resolved["parallel"]["max_parallel"], 2)
        self.assertEqual(resolved["parallel"]["gpus_per_exp"], 4)

    def test_sure_idle_gpu_config_filters_busy_gpus_and_shrinks_parallel(self):
        local_config = {
            "gpu_devices": "idle",
            "idle_gpu_min_free_mib": 9000,
            "idle_gpu_max_utilization": 20,
            "parallel": {
                "enabled": True,
                "max_parallel": 4,
                "gpus_per_exp": 1,
                "set_asr_world_size": False,
            },
        }
        smi = "\n".join(
            [
                "0, 11264, 10426, 838, 100",
                "1, 11264, 4, 11260, 0",
                "2, 11264, 100, 11164, 5",
                "3, 11264, 5000, 6264, 0",
            ]
        )
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}, clear=False), patch.object(
            sure_playground_module.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0, stdout=smi, stderr=""),
        ):
            resolved = sure_playground_module._apply_auto_gpu_config(
                local_config,
                sure_playground_module.logging.getLogger("test"),
            )

        self.assertEqual(resolved["gpu_devices"], ["1", "2"])
        self.assertEqual(resolved["parallel"]["max_parallel"], 2)
        self.assertEqual(resolved["parallel"]["gpus_per_exp"], 1)
        self.assertFalse(resolved["parallel"]["set_asr_world_size"])
        self.assertTrue(resolved["parallel"]["refresh_idle_gpu_before_exec"])
        self.assertEqual(resolved["parallel"]["idle_gpu_min_free_mib"], 9000)
        self.assertEqual(resolved["parallel"]["idle_gpu_max_utilization"], 20)
        self.assertFalse(resolved["parallel"]["idle_gpu_allow_busy_fallback"])
        self.assertTrue(resolved["parallel"]["gpu_lock_enabled"])

    def test_sure_idle_gpu_config_fails_when_no_idle_gpu(self):
        local_config = {
            "gpu_devices": "idle",
            "idle_gpu_min_free_mib": 9000,
            "idle_gpu_max_utilization": 20,
            "parallel": {"enabled": True, "max_parallel": 4, "gpus_per_exp": 1},
        }
        smi = "\n".join(
            [
                "0, 11264, 10426, 838, 100",
                "1, 11264, 9000, 2264, 80",
            ]
        )
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=False), patch.object(
            sure_playground_module.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0, stdout=smi, stderr=""),
        ):
            with self.assertRaisesRegex(RuntimeError, "below min_count"):
                sure_playground_module._apply_auto_gpu_config(
                    local_config,
                    sure_playground_module.logging.getLogger("test"),
                )

    def test_local_resource_allocator_splits_comma_separated_gpu_string(self):
        allocator = ResourceAllocator(
            gpu_devices="0,1,2",
            cpu_devices=None,
            max_parallel=3,
            gpus_per_exp=1,
        )

        self.assertEqual(allocator.allocate_resources(0)[0], "0")
        self.assertEqual(allocator.allocate_resources(1)[0], "1")
        self.assertEqual(allocator.allocate_resources(2)[0], "2")

    def test_resource_allocator_reselects_idle_gpu_before_execution(self):
        allocator = ResourceAllocator(
            gpu_devices=["0", "1", "2"],
            cpu_devices=None,
            max_parallel=3,
            gpus_per_exp=1,
            refresh_idle_gpus=True,
            idle_gpu_min_free_mib=9000,
            idle_gpu_max_utilization=20,
            idle_gpu_allow_busy_fallback=False,
            gpu_lock_enabled=False,
        )
        smi = "\n".join(
            [
                "0, 11264, 10426, 838, 100",
                "1, 11264, 4, 11260, 0",
                "2, 11264, 100, 11164, 5",
            ]
        )
        with patch.object(
            evomaster_local_env_module.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0, stdout=smi, stderr=""),
        ):
            selected, handles = allocator.prepare_gpu_allocation("0")

        self.assertEqual(selected, "1")
        self.assertEqual(handles, [])

    def test_resource_allocator_skips_cross_process_locked_gpu(self):
        smi = "\n".join(
            [
                "1, 11264, 4, 11260, 0",
                "2, 11264, 100, 11164, 5",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            first = ResourceAllocator(
                gpu_devices=["1", "2"],
                cpu_devices=None,
                max_parallel=2,
                gpus_per_exp=1,
                refresh_idle_gpus=True,
                idle_gpu_min_free_mib=9000,
                idle_gpu_max_utilization=20,
                idle_gpu_allow_busy_fallback=False,
                gpu_lock_enabled=True,
                gpu_lock_dir=tmp,
                gpu_lock_wait_seconds=0,
            )
            second = ResourceAllocator(
                gpu_devices=["1", "2"],
                cpu_devices=None,
                max_parallel=2,
                gpus_per_exp=1,
                refresh_idle_gpus=True,
                idle_gpu_min_free_mib=9000,
                idle_gpu_max_utilization=20,
                idle_gpu_allow_busy_fallback=False,
                gpu_lock_enabled=True,
                gpu_lock_dir=tmp,
                gpu_lock_wait_seconds=0,
            )
            handles1 = []
            handles2 = []
            try:
                with patch.object(
                    evomaster_local_env_module.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0, stdout=smi, stderr=""),
                ):
                    selected1, handles1 = first.prepare_gpu_allocation("1")
                    selected2, handles2 = second.prepare_gpu_allocation("1")
            finally:
                first.release_gpu_locks(handles1)
                second.release_gpu_locks(handles2)

        self.assertEqual(selected1, "1")
        self.assertEqual(selected2, "2")
        self.assertTrue(handles1)
        self.assertTrue(handles2)

    def test_sure_serial_tasks_can_use_all_gpus(self):
        self.assertTrue(
            sure_playground_module._use_all_gpus_for_serial_tasks(
                {"serial_gpus_per_exp": "all"},
                max_workers=1,
            )
        )
        self.assertFalse(
            sure_playground_module._use_all_gpus_for_serial_tasks(
                {"serial_gpus_per_exp": "all"},
                max_workers=2,
            )
        )

    def test_zipformer_zero_train_timeout_means_unlimited(self):
        with patch.dict(
            zipformer_baseline.os.environ,
            {"SURE_BASELINE_TRAIN_TIMEOUT": "0"},
            clear=False,
        ):
            self.assertIsNone(
                zipformer_baseline.timeout_env(
                    "SURE_BASELINE_TRAIN_TIMEOUT",
                    43200,
                )
            )

        with patch.dict(
            zipformer_baseline.os.environ,
            {"SURE_BASELINE_TRAIN_TIMEOUT": "123"},
            clear=False,
        ):
            self.assertEqual(
                zipformer_baseline.timeout_env(
                    "SURE_BASELINE_TRAIN_TIMEOUT",
                    43200,
                ),
                123,
            )

        with patch.dict(zipformer_baseline.os.environ, {}, clear=True):
            self.assertEqual(
                zipformer_baseline.timeout_env(
                    "SURE_BASELINE_TRAIN_TIMEOUT",
                    43200,
                ),
                43200,
            )

    def test_zipformer_candidate_zero_train_timeout_reaches_subprocess_wait(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            run_icefall_zipformer_candidate,
            "selected_train_duration",
            return_value=300,
        ), patch.object(
            zipformer_baseline,
            "duration_retry_sequence",
            return_value=[300],
        ), patch.object(
            run_icefall_zipformer_candidate,
            "train_command",
            return_value=["python", "train.py"],
        ), patch.object(
            zipformer_baseline,
            "run_command",
            return_value=Path(tmp) / "train.log",
        ) as run_command, patch.object(
            zipformer_baseline,
            "parse_max_memory_mb",
            return_value=None,
        ), patch.dict(
            os.environ,
            {"SURE_BASELINE_TRAIN_TIMEOUT": "0"},
            clear=False,
        ):
            run_icefall_zipformer_candidate.run_training(
                exp_dir=Path(tmp),
                train_epochs=1,
                start_epoch=1,
                final_train_args=[],
            )

        run_command.assert_called_once_with(
            "train",
            ["python", "train.py"],
            timeout=None,
        )

    def test_runtime_duration_uses_a10_memory_bucket(self):
        self.assertEqual(runtime_env.duration_from_gpu_memory(24_000, use_fp16=True), 1000)
        self.assertEqual(runtime_env.duration_from_gpu_memory(16_000, use_fp16=True), 600)
        self.assertEqual(runtime_env.duration_from_gpu_memory(48_000, use_fp16=True), 2400)

    def test_runtime_duration_auto_uses_visible_min_memory(self):
        with patch.object(runtime_env, "visible_gpu_memory_mb", return_value=[24_000, 48_000]):
            with patch.dict(
                runtime_env.os.environ,
                {"SURE_USE_FP16": "1", "SURE_DURATION_AUTOTUNE": "0"},
                clear=False,
            ):
                self.assertEqual(runtime_env.resolve_max_duration(), 1000)

    def test_runtime_duration_autotune_returns_last_successful_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "SURE_USE_FP16": "1",
                "SURE_DURATION_AUTOTUNE": "1",
                "SURE_DURATION_AUTOTUNE_MAX": "1100",
                "SURE_DURATION_AUTOTUNE_STEP": "100",
                "SURE_DURATION_CACHE_DIR": tmp,
            }
            with patch.object(runtime_env, "visible_gpu_memory_mb", return_value=[24_000]):
                with patch.object(runtime_env, "_probe_duration", side_effect=[True, False]):
                    with patch.dict(runtime_env.os.environ, env, clear=False):
                        self.assertEqual(runtime_env.resolve_max_duration(), 1000)

    def test_runtime_duration_autotune_returns_upper_when_all_upward_probes_succeed(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "SURE_USE_FP16": "1",
                "SURE_DURATION_AUTOTUNE": "1",
                "SURE_DURATION_AUTOTUNE_MAX": "1100",
                "SURE_DURATION_AUTOTUNE_STEP": "100",
                "SURE_DURATION_CACHE_DIR": tmp,
            }
            with patch.object(runtime_env, "visible_gpu_memory_mb", return_value=[24_000]):
                with patch.object(runtime_env, "_probe_duration", side_effect=[True, True]) as probe:
                    with patch.dict(runtime_env.os.environ, env, clear=False):
                        self.assertEqual(runtime_env.resolve_max_duration(), 1100)
                        self.assertEqual(probe.call_count, 2)

    def test_runtime_duration_autotune_falls_back_to_lower_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "SURE_USE_FP16": "1",
                "SURE_DURATION_AUTOTUNE": "1",
                "SURE_DURATION_AUTOTUNE_MAX": "1000",
                "SURE_DURATION_AUTOTUNE_MIN": "800",
                "SURE_DURATION_AUTOTUNE_STEP": "100",
                "SURE_DURATION_CACHE_DIR": tmp,
            }
            with patch.object(runtime_env, "visible_gpu_memory_mb", return_value=[24_000]):
                with patch.object(runtime_env, "_probe_duration", side_effect=[False, False, True]):
                    with patch.dict(runtime_env.os.environ, env, clear=False):
                        self.assertEqual(runtime_env.resolve_max_duration(), 800)

    def test_runtime_duration_autotune_fails_closed_when_all_probes_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "SURE_USE_FP16": "1",
                "SURE_DURATION_AUTOTUNE": "1",
                "SURE_DURATION_AUTOTUNE_MAX": "1000",
                "SURE_DURATION_AUTOTUNE_MIN": "800",
                "SURE_DURATION_AUTOTUNE_STEP": "100",
                "SURE_DURATION_CACHE_DIR": tmp,
            }
            with patch.object(runtime_env, "visible_gpu_memory_mb", return_value=[24_000]):
                with patch.object(runtime_env, "_probe_duration", side_effect=[False, False, False]):
                    with patch.dict(runtime_env.os.environ, env, clear=False):
                        with self.assertRaisesRegex(RuntimeError, "all duration probes failed"):
                            runtime_env.resolve_max_duration()

    def test_runtime_duration_autotune_uses_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "SURE_USE_FP16": "1",
                "SURE_DURATION_AUTOTUNE": "1",
                "SURE_DURATION_AUTOTUNE_MAX": "1100",
                "SURE_DURATION_AUTOTUNE_STEP": "100",
                "SURE_DURATION_CACHE_DIR": tmp,
            }
            with patch.object(runtime_env, "visible_gpu_memory_mb", return_value=[24_000]):
                with patch.dict(runtime_env.os.environ, env, clear=False):
                    with patch.object(runtime_env, "_probe_duration", side_effect=[True, False]) as first_probe:
                        self.assertEqual(runtime_env.resolve_max_duration(), 1000)
                        self.assertEqual(first_probe.call_count, 2)
                    with patch.object(runtime_env, "_probe_duration") as second_probe:
                        self.assertEqual(runtime_env.resolve_max_duration(), 1000)
                        second_probe.assert_not_called()

    def test_runtime_duration_autotune_isolates_probe_dir_by_cache_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(runtime_env, "_probe_duration", return_value=True) as probe:
                self.assertEqual(
                    runtime_env._autotune_uncached(
                        start=300,
                        lower_limit=100,
                        upper_limit=300,
                        step=100,
                        timeout=30,
                        world_size=2,
                        use_fp16=True,
                        cache_dir=Path(tmp),
                        cache_key="candidate-cache-key",
                        probe_args=[],
                    ),
                    300,
                )

        self.assertEqual(probe.call_args.args[4], Path(tmp) / "probes" / "candidate-cache-key")

    def test_runtime_duration_probe_args_prefers_json_over_shell_string(self):
        env = {
            "SURE_DURATION_TRAIN_ARGS_JSON": json.dumps(["--encoder-dim", "160"]),
            "SURE_DURATION_TRAIN_ARGS": "--encoder-dim 768",
        }
        with patch.dict(runtime_env.os.environ, env, clear=False):
            self.assertEqual(runtime_env._duration_probe_args(), ["--encoder-dim", "160"])

    def test_runtime_duration_probe_env_includes_workspace_icefall_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "exp"
            recipe_dir = workspace / "base_model" / "recipe"
            root_dir = workspace / "base_model" / "root"
            recipe_dir.mkdir(parents=True)
            root_dir.mkdir(parents=True)
            with patch.dict(runtime_env.os.environ, {"PYTHONPATH": "/existing/path"}, clear=True):
                env = runtime_env._probe_command_env(recipe_dir)

        pythonpath = env["PYTHONPATH"].split(os.pathsep)
        self.assertIn(str(root_dir), pythonpath)
        self.assertIn(str(recipe_dir), pythonpath)
        self.assertIn("/existing/path", pythonpath)

    def test_runtime_duration_probe_raises_on_non_resource_startup_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "import_fail.py"
            script.write_text(
                "print(\"ModuleNotFoundError: No module named 'icefall'\", flush=True)\n"
                "raise SystemExit(1)\n",
                encoding="utf-8",
            )
            log_path = Path(tmp) / "probe.log"

            with self.assertRaisesRegex(RuntimeError, "before proving a resource limit"):
                runtime_env._run_logged_command(
                    [sys.executable, str(script)],
                    cwd=Path(tmp),
                    timeout=10,
                    log_path=log_path,
                )
            log_text = log_path.read_text(encoding="utf-8")

        self.assertIn("PYTHONPATH:", log_text)
        self.assertIn("ModuleNotFoundError", log_text)

    def test_runtime_duration_short_probe_accepts_after_training_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "emit_batches.py"
            script.write_text(
                "\n".join(
                    [
                        "import time",
                        "print('Epoch 1, batch 1, loss=1.0', flush=True)",
                        "print('Epoch 1, batch 2, loss=0.9', flush=True)",
                        "time.sleep(30)",
                    ]
                ),
                encoding="utf-8",
            )
            log_path = Path(tmp) / "probe.log"

            ok = runtime_env._run_logged_command(
                [sys.executable, str(script)],
                cwd=Path(tmp),
                timeout=10,
                log_path=log_path,
                success_batch_count=2,
            )
            log_text = log_path.read_text(encoding="utf-8")

        self.assertTrue(ok)
        self.assertIn("probe accepted after 2 unique training batch log", log_text)

    def test_runtime_duration_probe_counts_duplicate_ddp_batch_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "emit_duplicate_batch.py"
            script.write_text(
                "\n".join(
                    [
                        "import time",
                        "for _ in range(4):",
                        "    print('Epoch 1, batch 1, loss=1.0', flush=True)",
                        "time.sleep(30)",
                    ]
                ),
                encoding="utf-8",
            )
            log_path = Path(tmp) / "probe.log"

            ok = runtime_env._run_logged_command(
                [sys.executable, str(script)],
                cwd=Path(tmp),
                timeout=2,
                log_path=log_path,
                success_batch_count=2,
            )
            log_text = log_path.read_text(encoding="utf-8")

        self.assertFalse(ok)
        self.assertNotIn("probe accepted after", log_text)

    def test_runtime_duration_probe_rejects_low_memory_headroom(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "emit_memory.py"
            script.write_text(
                "\n".join(
                    [
                        "import time",
                        "print('Epoch 1, batch 1, max memory allocated so far is 23500 MiB', flush=True)",
                        "time.sleep(30)",
                    ]
                ),
                encoding="utf-8",
            )
            log_path = Path(tmp) / "probe.log"

            ok = runtime_env._run_logged_command(
                [sys.executable, str(script)],
                cwd=Path(tmp),
                timeout=10,
                log_path=log_path,
                success_batch_count=1,
                memory_headroom_mb=1000,
                gpu_memory_mb=[24000],
            )
            log_text = log_path.read_text(encoding="utf-8")

        self.assertFalse(ok)
        self.assertIn("peak memory 23500MB", log_text)

    def test_runtime_duration_probe_uses_real_args_and_disables_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            recipe_dir = Path(tmp) / "recipe"
            recipe_dir.mkdir()
            (recipe_dir / "train.py").write_text(
                "import sys\n"
                "if '--help' in sys.argv:\n"
                "    print('--world-size --master-port --num-epochs --start-epoch "
                "--tensorboard --use-fp16 --exp-dir --max-duration --enable-musan "
                "--manifest-dir --bpe-model --print-diagnostics --log-interval "
                "--use-cr-ctc --encoder-dim --full-libri')\n",
                encoding="utf-8",
            )
            env = {
                "SURE_DURATION_RECIPE_DIR": str(recipe_dir),
                "SURE_DURATION_TRAIN_ARGS": (
                    "--use-cr-ctc 1 --encoder-dim 192 "
                    "--full-libri 1 --max-duration 900 --log-interval 99"
                ),
                "SURE_DURATION_PRINT_DIAGNOSTICS": "0",
                "SURE_DURATION_PROBE_LOG_INTERVAL": "1",
            }
            with patch.dict(runtime_env.os.environ, env, clear=False):
                extra_args = runtime_env._duration_probe_args()
                with patch.object(runtime_env, "_run_logged_command", return_value=True) as run:
                    self.assertTrue(
                        runtime_env._probe_duration(
                            300,
                            2,
                            True,
                            30,
                            Path(tmp) / "probes",
                            extra_args,
                        )
                    )
            command = run.call_args.args[0]
            diagnostics_index = command.index("--print-diagnostics")
            self.assertEqual(command[diagnostics_index + 1], "false")
            self.assertIn("--use-cr-ctc", command)
            self.assertIn("--encoder-dim", command)
            full_libri_index = command.index("--full-libri")
            self.assertEqual(command[full_libri_index + 1], "1")
            self.assertEqual(command.count("--full-libri"), 1)
            max_duration_index = command.index("--max-duration")
            self.assertEqual(command[max_duration_index + 1], "300")
            self.assertEqual(command.count("--max-duration"), 1)
            log_interval_index = command.index("--log-interval")
            self.assertEqual(command[log_interval_index + 1], "1")
            self.assertEqual(command.count("--log-interval"), 1)

    def test_runtime_duration_probe_omits_unsupported_log_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            recipe_dir = Path(tmp) / "recipe"
            recipe_dir.mkdir()
            (recipe_dir / "train.py").write_text(
                "import sys\n"
                "if '--help' in sys.argv:\n"
                "    print('--world-size --master-port --num-epochs --start-epoch "
                "--tensorboard --use-fp16 --exp-dir --max-duration --enable-musan "
                "--manifest-dir --bpe-model --encoder-dim')\n",
                encoding="utf-8",
            )
            env = {
                "SURE_DURATION_RECIPE_DIR": str(recipe_dir),
                "SURE_DURATION_TRAIN_ARGS": "--encoder-dim 192 --log-interval 99",
                "SURE_DURATION_PROBE_LOG_INTERVAL": "1",
            }
            runtime_env._TRAIN_HELP_OPTION_CACHE.clear()
            with patch.dict(runtime_env.os.environ, env, clear=False):
                extra_args = runtime_env._duration_probe_args()
                with patch.object(runtime_env, "_run_logged_command", return_value=True) as run:
                    self.assertTrue(
                        runtime_env._probe_duration(
                            300,
                            2,
                            True,
                            30,
                            Path(tmp) / "probes",
                            extra_args,
                        )
                    )
            command = run.call_args.args[0]
            self.assertNotIn("--log-interval", command)
            self.assertNotIn("--print-diagnostics", command)
            self.assertIn("--encoder-dim", command)

    def test_runtime_duration_probe_tedlium_ignores_legacy_full_libri(self):
        with tempfile.TemporaryDirectory() as tmp:
            recipe_dir = Path(tmp) / "recipe"
            recipe_dir.mkdir()
            train_py = recipe_dir / "train.py"
            train_py.write_text("# fake\n", encoding="utf-8")
            supported = runtime_env._PROBE_REQUIRED_OPTIONS | {"--encoder-dim"}
            env = {
                "SURE_DURATION_RECIPE_DIR": str(recipe_dir),
                "SURE_DURATION_PROBE_FULL_LIBRI": "1",
            }
            runtime_env._DEPRECATED_FULL_LIBRI_WARNED = False
            with patch.dict(runtime_env.os.environ, env, clear=False), patch.object(
                runtime_env, "_train_supported_options", return_value=supported
            ), patch.object(runtime_env, "_run_logged_command", return_value=True) as run, patch.object(
                runtime_env, "_log"
            ) as log:
                self.assertTrue(
                    runtime_env._probe_duration(
                        300,
                        2,
                        True,
                        30,
                        Path(tmp) / "probes",
                        ["--encoder-dim", "192"],
                    )
                )
                runtime_env._warn_deprecated_full_libri()

            self.assertNotIn("--full-libri", run.call_args.args[0])
            warnings = [
                call for call in log.call_args_list if "deprecated and ignored" in str(call)
            ]
            self.assertEqual(len(warnings), 1)

    def test_runtime_duration_probe_rejects_missing_helper_option_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            recipe_dir = Path(tmp) / "recipe"
            recipe_dir.mkdir()
            (recipe_dir / "train.py").write_text("# fake\n", encoding="utf-8")
            supported = runtime_env._PROBE_REQUIRED_OPTIONS - {"--max-duration"}
            with patch.dict(
                runtime_env.os.environ,
                {"SURE_DURATION_RECIPE_DIR": str(recipe_dir)},
                clear=False,
            ), patch.object(
                runtime_env, "_train_supported_options", return_value=supported
            ), patch.object(runtime_env, "_run_logged_command") as run:
                with self.assertRaisesRegex(
                    RuntimeError, "duration_probe_helper_cli_incompatible"
                ):
                    runtime_env._probe_duration(
                        300, 1, True, 30, Path(tmp) / "probes", []
                    )
            run.assert_not_called()

    def test_runtime_duration_probe_rejects_unsupported_candidate_option(self):
        with tempfile.TemporaryDirectory() as tmp:
            recipe_dir = Path(tmp) / "recipe"
            recipe_dir.mkdir()
            (recipe_dir / "train.py").write_text("# fake\n", encoding="utf-8")
            with patch.dict(
                runtime_env.os.environ,
                {"SURE_DURATION_RECIPE_DIR": str(recipe_dir)},
                clear=False,
            ), patch.object(
                runtime_env,
                "_train_supported_options",
                return_value=runtime_env._PROBE_REQUIRED_OPTIONS,
            ), patch.object(runtime_env, "_run_logged_command") as run:
                with self.assertRaisesRegex(
                    RuntimeError, "duration_probe_candidate_cli_incompatible"
                ):
                    runtime_env._probe_duration(
                        300,
                        1,
                        True,
                        30,
                        Path(tmp) / "probes",
                        ["--encoder-dim=192"],
                    )
            run.assert_not_called()

    def test_runtime_duration_help_discovery_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            recipe_dir = Path(tmp)
            train_py = recipe_dir / "train.py"
            train_py.write_text("# fake\n", encoding="utf-8")
            runtime_env._TRAIN_HELP_OPTION_CACHE.clear()
            result = subprocess.CompletedProcess([], 1, stdout="bad help")
            with patch.object(runtime_env.subprocess, "run", return_value=result):
                with self.assertRaisesRegex(
                    RuntimeError, "duration_probe_help_discovery_failed"
                ):
                    runtime_env._train_supported_options(
                        sys.executable, train_py, recipe_dir
                    )

    def test_runtime_duration_help_discovery_timeout_has_stable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            recipe_dir = Path(tmp)
            train_py = recipe_dir / "train.py"
            train_py.write_text("# fake\n", encoding="utf-8")
            runtime_env._TRAIN_HELP_OPTION_CACHE.clear()
            timeout = subprocess.TimeoutExpired(
                [sys.executable, str(train_py), "--help"],
                30,
            )
            with patch.object(runtime_env.subprocess, "run", side_effect=timeout):
                with self.assertRaises(RuntimeError) as raised:
                    runtime_env._train_supported_options(
                        sys.executable, train_py, recipe_dir
                    )

        marker, raw_payload = str(raised.exception).split(": ", 1)
        payload = json.loads(raw_payload)
        self.assertEqual(marker, "duration_probe_help_discovery_failed")
        self.assertEqual(payload["error"], "timeout")
        self.assertEqual(payload["timeout_seconds"], 30)
        self.assertEqual(payload["train_py"], str(train_py))

    def test_runtime_duration_help_discovery_oserror_has_stable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            recipe_dir = Path(tmp)
            train_py = recipe_dir / "train.py"
            train_py.write_text("# fake\n", encoding="utf-8")
            runtime_env._TRAIN_HELP_OPTION_CACHE.clear()
            with patch.object(
                runtime_env.subprocess,
                "run",
                side_effect=OSError(2, "not found"),
            ):
                with self.assertRaises(RuntimeError) as raised:
                    runtime_env._train_supported_options(
                        sys.executable, train_py, recipe_dir
                    )

        marker, raw_payload = str(raised.exception).split(": ", 1)
        payload = json.loads(raw_payload)
        self.assertEqual(marker, "duration_probe_help_discovery_failed")
        self.assertEqual(payload["error"], "launch_failed")
        self.assertEqual(payload["exception_type"], "FileNotFoundError")
        self.assertEqual(payload["errno"], 2)
        self.assertEqual(payload["train_py"], str(train_py))
        self.assertNotIn("detail", payload)

    def test_runtime_duration_help_discovery_rejects_success_without_long_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            recipe_dir = Path(tmp)
            train_py = recipe_dir / "train.py"
            train_py.write_text("# fake\n", encoding="utf-8")
            runtime_env._TRAIN_HELP_OPTION_CACHE.clear()
            result = subprocess.CompletedProcess(
                [],
                0,
                stdout="usage: train.py [-h]",
            )
            with patch.object(runtime_env.subprocess, "run", return_value=result) as run:
                with self.assertRaises(RuntimeError) as raised:
                    runtime_env._train_supported_options(
                        sys.executable, train_py, recipe_dir
                    )
                with self.assertRaises(RuntimeError):
                    runtime_env._train_supported_options(
                        sys.executable, train_py, recipe_dir
                    )

        marker, raw_payload = str(raised.exception).split(": ", 1)
        payload = json.loads(raw_payload)
        self.assertEqual(marker, "duration_probe_help_discovery_failed")
        self.assertEqual(payload["error"], "no_long_options")
        self.assertEqual(payload["train_py"], str(train_py))
        self.assertEqual(run.call_count, 2)

    def test_runtime_duration_help_cache_invalidates_when_train_script_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            recipe_dir = Path(tmp)
            train_py = recipe_dir / "train.py"
            train_py.write_text("# version one\n", encoding="utf-8")
            runtime_env._TRAIN_HELP_OPTION_CACHE.clear()
            results = (
                subprocess.CompletedProcess([], 0, stdout="--first-option"),
                subprocess.CompletedProcess([], 0, stdout="--second-option"),
            )
            with patch.object(runtime_env.subprocess, "run", side_effect=results) as run:
                first = runtime_env._train_supported_options(
                    sys.executable, train_py, recipe_dir
                )
                train_py.write_text("# version two\n", encoding="utf-8")
                second = runtime_env._train_supported_options(
                    sys.executable, train_py, recipe_dir
                )

        self.assertEqual(first, frozenset({"--first-option"}))
        self.assertEqual(second, frozenset({"--second-option"}))
        self.assertEqual(run.call_count, 2)

    def test_runtime_duration_cache_key_changes_with_policy_and_train_script(self):
        def cache_key() -> str:
            return runtime_env._duration_cache_key(
                baseline=300,
                memories=[24_000],
                use_fp16=True,
                world_size=1,
                step=100,
                lower_limit=100,
                upper_limit=300,
                probe_args=["--encoder-dim", "192"],
            )

        with tempfile.TemporaryDirectory() as tmp:
            recipe_dir = Path(tmp)
            train_py = recipe_dir / "train.py"
            env = {"SURE_DURATION_RECIPE_DIR": str(recipe_dir)}
            with patch.dict(runtime_env.os.environ, env, clear=True):
                missing_script_key = cache_key()
                train_py.write_text("# version one\n", encoding="utf-8")
                first_script_key = cache_key()
                train_py.write_text("# version two\n", encoding="utf-8")
                second_script_key = cache_key()
                with patch.object(
                    runtime_env,
                    "_DURATION_PROBE_POLICY_VERSION",
                    "next-policy",
                ):
                    next_policy_key = cache_key()

        self.assertNotEqual(missing_script_key, first_script_key)
        self.assertNotEqual(first_script_key, second_script_key)
        self.assertNotEqual(second_script_key, next_policy_key)

    def test_runtime_duration_cache_rejects_previous_probe_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "duration.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "schema_version": runtime_env._DURATION_CACHE_SCHEMA_VERSION,
                        "probe_policy": "previous-policy",
                        "duration": 700,
                    }
                ),
                encoding="utf-8",
            )
            self.assertIsNone(runtime_env._read_cached_duration(cache_path))

    def test_runtime_duration_cache_requires_current_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "duration.json"
            cache_path.write_text('{"duration": 700}\n', encoding="utf-8")
            self.assertIsNone(runtime_env._read_cached_duration(cache_path))
            runtime_env._write_cached_duration(cache_path, 700)
            self.assertEqual(runtime_env._read_cached_duration(cache_path), 700)

    def test_runtime_duration_probe_raises_on_incompatible_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "reject_args.py"
            script.write_text(
                "print('train.py: error: unrecognized arguments: --log-interval 1')\n",
                encoding="utf-8",
            )
            log_path = Path(tmp) / "probe.log"

            with self.assertRaisesRegex(RuntimeError, "incompatible with train.py"):
                runtime_env._run_logged_command(
                    [sys.executable, str(script)],
                    cwd=Path(tmp),
                    timeout=10,
                    log_path=log_path,
                )

    def test_asr_baseline_resolves_auto_duration_with_candidate_train_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            helper = Path(tmp) / "runtime_env.py"
            helper.write_text("# helper\n", encoding="utf-8")
            log_path = Path(tmp) / "duration_autotune.log"
            log_path.write_text("[sure_runtime] probe\n700\n", encoding="utf-8")
            env = {
                "SURE_MAX_DURATION": "auto",
                "SURE_RUNTIME_ENV_HELPER": str(helper),
            }
            with patch.dict(os.environ, env, clear=True):
                with patch.object(zipformer_baseline, "run_command", return_value=log_path) as run:
                    self.assertEqual(
                        zipformer_baseline.max_duration(
                            1400,
                            cap=1400,
                            train_args=["--encoder-dim", "160"],
                        ),
                        700,
                    )

            extra_env = run.call_args.kwargs["extra_env"]
            self.assertEqual(
                json.loads(extra_env["SURE_DURATION_TRAIN_ARGS_JSON"]),
                ["--encoder-dim", "160"],
            )

    def test_asr_baseline_auto_duration_fails_without_final_integer(self):
        with tempfile.TemporaryDirectory() as tmp:
            helper = Path(tmp) / "runtime_env.py"
            helper.write_text("# helper\n", encoding="utf-8")
            log_path = Path(tmp) / "duration_autotune.log"
            log_path.write_text("[sure_runtime] probe complete without value\n", encoding="utf-8")
            env = {
                "SURE_MAX_DURATION": "auto",
                "SURE_RUNTIME_ENV_HELPER": str(helper),
            }
            with patch.dict(os.environ, env, clear=True):
                with patch.object(zipformer_baseline, "run_command", return_value=log_path):
                    with self.assertRaisesRegex(RuntimeError, "standalone positive integer"):
                        zipformer_baseline.max_duration(
                            1400,
                            cap=1400,
                            train_args=["--encoder-dim", "160"],
                        )

    def test_asr_baseline_auto_duration_clamps_to_floor_for_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            helper = Path(tmp) / "runtime_env.py"
            helper.write_text("# helper\n", encoding="utf-8")
            log_path = Path(tmp) / "duration_autotune.log"
            log_path.write_text("[sure_runtime] probe\n16\n", encoding="utf-8")
            env = {
                "SURE_MAX_DURATION": "auto",
                "SURE_RUNTIME_ENV_HELPER": str(helper),
                "SURE_DURATION_AUTOTUNE_MIN": "100",
                "SURE_TRAIN_DURATION_MIN": "100",
            }
            with patch.dict(os.environ, env, clear=True):
                with patch.object(zipformer_baseline, "run_command", return_value=log_path):
                    self.assertEqual(
                        zipformer_baseline.max_duration(
                            1400,
                            cap=1400,
                            train_args=["--encoder-dim", "160"],
                            enforce_floor=True,
                        ),
                        100,
                    )

    def test_asr_baseline_duration_retry_sequence_descends_to_floor(self):
        env = {
            "SURE_TRAIN_DURATION_RETRY": "1",
            "SURE_TRAIN_DURATION_MIN": "300",
            "SURE_TRAIN_DURATION_RETRY_STEP": "100",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(duration_retry_sequence(600), [600, 500, 400, 300])

    def test_asr_baseline_duration_retry_can_be_disabled(self):
        with patch.dict(os.environ, {"SURE_TRAIN_DURATION_RETRY": "0"}, clear=True):
            self.assertEqual(duration_retry_sequence(600), [600])

    def test_asr_baseline_duration_retry_does_not_descend_below_floor(self):
        env = {
            "SURE_TRAIN_DURATION_RETRY": "1",
            "SURE_TRAIN_DURATION_MIN": "100",
            "SURE_TRAIN_DURATION_RETRY_VALUES": "80,100,120",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(duration_retry_sequence(200), [200, 120, 100])

    def test_asr_baseline_detects_cuda_oom_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "train.log"
            log_path.write_text("RuntimeError: CUDA out of memory.\n", encoding="utf-8")
            self.assertTrue(is_oom_failure(log_path))

    def test_asr_candidate_rejects_pipe_captured_subprocess_output(self):
        code = """
import subprocess
subprocess.Popen(["python", "train.py"], stdout=subprocess.PIPE)
subprocess.run(["python", "decode.py"], capture_output=True)
"""
        profile = BaseModelProfile(model_id="zipformer", framework="icefall")
        errors = validate_sure_candidate_boundary(
            code,
            "/tmp/sure",
            profile,
            canonical_task="asr",
        )
        self.assertTrue(any("stdout=subprocess.PIPE" in error for error in errors))
        self.assertTrue(any("capture_output=True" in error for error in errors))

    def test_asr_candidate_rejects_bounded_lhotse_suffix_length(self):
        code = """
def normalize_key(key):
    parts = key.split("-")
    if len(parts) == 4 and all(x.isdigit() for x in parts) and len(parts[-1]) <= 3:
        return "-".join(parts[:-1])
    return key

cmd = ["python", "base_model/recipe/decode.py"]
"""
        profile = BaseModelProfile(model_id="zipformer", framework="icefall")
        errors = validate_sure_candidate_boundary(
            code,
            "/tmp/sure",
            profile,
            canonical_task="asr",
        )
        self.assertTrue(any("Lhotse numeric suffix length" in error for error in errors))

    def test_asr_candidate_rejects_bounded_lhotse_suffix_variable(self):
        code = """
def normalize_key(key):
    parts = key.split("-")
    suffix = parts[-1]
    if len(parts) == 4 and suffix.isdigit() and len(suffix) < 4:
        return "-".join(parts[:-1])
    return key

cmd = ["python", "base_model/recipe/decode.py"]
"""
        profile = BaseModelProfile(model_id="zipformer", framework="icefall")
        errors = validate_sure_candidate_boundary(
            code,
            "/tmp/sure",
            profile,
            canonical_task="asr",
        )
        self.assertTrue(any("Lhotse numeric suffix length" in error for error in errors))

    def test_asr_candidate_allows_unbounded_lhotse_suffix_normalization(self):
        code = """
def normalize_key(key):
    parts = key.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return key

cmd = ["python", "base_model/recipe/decode.py"]
"""
        profile = BaseModelProfile(model_id="zipformer", framework="icefall")
        errors = validate_sure_candidate_boundary(
            code,
            "/tmp/sure",
            profile,
            canonical_task="asr",
        )
        self.assertFalse(any("Lhotse numeric suffix length" in error for error in errors))

    def test_tts_candidate_rejects_raw_f5_training_entrypoint(self):
        code = """
import subprocess
cmd = ["python", "base_model/root/src/f5_tts/train/finetune_cli.py", "--finetune"]
subprocess.run(cmd)
"""
        profile = BaseModelProfile(model_id="f5tts", framework="f5-tts")
        errors = validate_sure_candidate_boundary(
            code,
            "/tmp/sure",
            profile,
            canonical_task="tts",
        )
        self.assertTrue(any("raw F5-TTS training entrypoint" in error for error in errors))

    def test_tts_candidate_rejects_raw_f5_infer_cli(self):
        code = """
import subprocess
cmd = ["python", "base_model/root/src/f5_tts/infer/infer_cli.py", "--gen_text", "hello"]
subprocess.run(cmd)
"""
        profile = BaseModelProfile(model_id="f5tts", framework="f5-tts")
        errors = validate_sure_candidate_boundary(
            code,
            "/tmp/sure",
            profile,
            canonical_task="tts",
        )
        self.assertTrue(any("raw F5-TTS infer_cli.py" in error for error in errors))

    def test_candidate_changes_validator_requires_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            errors = validate_candidate_changes(workspace)
            self.assertTrue(errors)
            changes_path = workspace / "artifacts" / "candidate_changes.json"
            changes_path.parent.mkdir(parents=True)
            changes_path.write_text(
                json.dumps(
                    {
                        "candidate_type": "inference",
                        "idea_text": "adjust nfe",
                        "changed_fields": ["inference_config.nfe_step"],
                        "arch_config": {},
                        "training_config": {},
                        "inference_config": {"nfe_step": 32},
                        "defaults": {},
                        "diff_from_defaults": {},
                        "produced_artifacts": {"samples_jsonl": "artifacts/samples.jsonl"},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(validate_candidate_changes(workspace), [])

    def test_arch_candidate_changes_validator_requires_arch_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            changes_path = workspace / "artifacts" / "candidate_changes.json"
            changes_path.parent.mkdir(parents=True)
            changes_path.write_text(
                json.dumps(
                    {
                        "candidate_type": "arch",
                        "idea_text": "change depth",
                        "changed_fields": ["arch_config.depth"],
                        "arch_config": {"depth": 20},
                        "training_config": {},
                        "inference_config": {},
                        "defaults": {"arch_config": {"depth": 22}},
                        "diff_from_defaults": {
                            "arch_config": {"depth": {"default": 22, "used": 20}}
                        },
                        "produced_artifacts": {},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(validate_candidate_changes(workspace), [])
            self.assertEqual(validate_arch_candidate_changes(workspace), [])

    def test_arch_candidate_changes_validator_backfills_legacy_nested_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            changes_path = workspace / "artifacts" / "candidate_changes.json"
            changes_path.parent.mkdir(parents=True)
            changes_path.write_text(
                json.dumps(
                    {
                        "candidate_type": "arch",
                        "idea_text": "legacy arch change",
                        "changed_fields": ["arch_config.depth"],
                        "arch_config": {"depth": 20},
                        "training_config": {},
                        "inference_config": {},
                        "defaults": {"arch_config": {"depth": 22}},
                        "diff_from_defaults": {},
                        "produced_artifacts": {},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(validate_arch_candidate_changes(workspace), [])
            payload = json.loads(changes_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["diff_from_defaults"]["arch_config"]["depth"]["default"],
                22,
            )
            self.assertEqual(
                payload["diff_from_defaults"]["arch_config"]["depth"]["value"],
                20,
            )

    def test_run_vc_candidate_setup_failure_writes_remote_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "exp"
            (workspace / "metric").mkdir(parents=True)
            result = workspace / "metric" / "remote_training_result.json"
            argv = [
                "run_vc_sure_candidate.py",
                "--config",
                str(Path(tmp) / "missing.yaml"),
                "--workspace",
                str(workspace),
                "--result",
                str(result),
                "--timeout",
                "60",
            ]
            with patch.object(sys, "argv", argv), patch.object(
                run_vc_sure_candidate,
                "ensure_project_imports",
                side_effect=RuntimeError("import failed"),
            ):
                with self.assertRaises(RuntimeError):
                    run_vc_sure_candidate.main()
            payload = json.loads(result.read_text(encoding="utf-8"))
            self.assertFalse(payload["success"])
            self.assertEqual(payload["reason_code"], "remote_exception")
            self.assertEqual(payload["details"]["exception_type"], "RuntimeError")
            status = json.loads((workspace / "artifacts" / "candidate_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["reason_code"], "remote_exception")

    def test_f5tts_batch_worker_count_respects_visible_gpus(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,5"}, clear=False):
            self.assertEqual(run_f5tts_batch_infer.choose_worker_count("8", 20, "cuda"), 2)
            self.assertEqual(run_f5tts_batch_infer.choose_worker_count("auto", 1, "cuda:0"), 1)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=False):
            self.assertEqual(run_f5tts_batch_infer.choose_worker_count("8", 20, "cuda"), 1)
        self.assertEqual(run_f5tts_batch_infer.choose_worker_count("3", 20, "cpu"), 3)

    def test_f5tts_batch_shards_balance_cost_deterministically(self):
        rows = [
            {"index": 1, "sample_id": "short", "gen_text": "x" * 10},
            {"index": 2, "sample_id": "long", "gen_text": "x" * 100},
            {"index": 3, "sample_id": "medium", "gen_text": "x" * 60},
            {"index": 4, "sample_id": "small", "gen_text": "x" * 20},
        ]
        first = run_f5tts_batch_infer.split_shards(rows, 2)
        second = run_f5tts_batch_infer.split_shards(rows, 2)
        self.assertEqual(first, second)
        self.assertEqual([[row["sample_id"] for row in shard] for shard in first], [["long"], ["short", "medium", "small"]])
        self.assertEqual([sum(run_f5tts_batch_infer.row_cost(row) for row in shard) for shard in first], [100, 90])

    def test_f5tts_batch_chunking_prefers_punctuation_and_oom_detection_is_narrow(self):
        text = "First sentence is here. Second sentence is somewhat longer! Final words."
        chunks = run_f5tts_batch_infer.split_text_chunks(text, max_chars=35, min_chars=10)
        self.assertEqual(" ".join(chunks), text)
        self.assertTrue(all(len(chunk) <= 35 for chunk in chunks))
        self.assertTrue(chunks[0].endswith("."))
        self.assertTrue(run_f5tts_batch_infer.is_oom_error(RuntimeError("CUDA out of memory")))
        self.assertFalse(run_f5tts_batch_infer.is_oom_error(RuntimeError("invalid model input")))

    def test_f5tts_batch_infer_preserves_existing_arch_candidate_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            artifacts = workspace / "artifacts"
            artifacts.mkdir(parents=True)
            (artifacts / "candidate_changes.json").write_text(
                json.dumps(
                    {
                        "candidate_type": "arch",
                        "idea_text": "reduce depth",
                        "changed_fields": ["arch_config.depth"],
                        "arch_config": {"depth": 20},
                        "training_config": {"action": "arch_finetune_short"},
                        "inference_config": {},
                        "defaults": {"arch_config": {"depth": 22}},
                        "diff_from_defaults": {
                            "arch_config": {"depth": {"default": 22, "used": 20}}
                        },
                        "produced_artifacts": {"checkpoint": "models/f5tts_arch/final_checkpoint.pt"},
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                candidate_type="auto",
                idea_text="arch inference",
                training_action="no_train",
                arch_action="no_arch",
                nfe_step=16,
                cfg_strength=2.0,
                sway_sampling_coef=-1.0,
                speed=1.0,
                remove_silence="0",
                text_cleanup="none",
                model="F5TTS_v1_Base",
                ckpt_file="models/f5tts_arch/final_checkpoint.pt",
                vocab_file="",
                model_cfg="models/f5tts_arch/model_cfg.yaml",
            )
            with patch.object(run_f5tts_batch_infer, "WORKSPACE", workspace), patch.object(
                run_f5tts_batch_infer,
                "ARTIFACTS",
                artifacts,
            ):
                run_f5tts_batch_infer.write_candidate_changes(
                    args,
                    [{"reference_audio": "base_model/eval_data/ref.wav"}],
                    workers=1,
                )

            payload = json.loads((artifacts / "candidate_changes.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["candidate_type"], "arch")
            self.assertEqual(payload["diff_from_defaults"]["arch_config"]["depth"]["used"], 20)
            self.assertIn("arch_config.depth", payload["changed_fields"])
            self.assertEqual(payload["produced_artifacts"]["checkpoint"], "models/f5tts_arch/final_checkpoint.pt")
            self.assertEqual(validate_arch_candidate_changes(workspace), [])

    def test_f5tts_arch_candidate_changes_records_init_mode(self):
        args = SimpleNamespace(
            action="arch_finetune_short",
            init_mode="scratch",
            train_manifest="libritts_train_clean_100_1h",
            max_steps=1000,
            learning_rate="3e-6",
            effective_batch_size=8,
            early_stop=True,
            idea_text="scratch screen smaller depth",
            vocab_file=Path("/models/vocab.txt"),
        )
        arch_config = {
            **run_f5tts_arch_finetune.DEFAULT_ARCH,
            "depth": 20,
        }

        payload = run_f5tts_arch_finetune.candidate_changes_payload(
            args=args,
            arch_config=arch_config,
            output_dir=Path("models/f5tts_arch"),
            model_cfg_path=Path("models/f5tts_arch/model_cfg.yaml"),
            final_checkpoint=Path("models/f5tts_arch/final_checkpoint.pt"),
        )

        self.assertEqual(payload["candidate_type"], "arch")
        self.assertEqual(payload["training_config"]["init_mode"], "scratch")
        self.assertIn("training_config.init_mode", payload["changed_fields"])
        self.assertEqual(
            payload["diff_from_defaults"]["training_config"]["init_mode"]["default"],
            "partial_load",
        )
        self.assertEqual(payload["diff_from_defaults"]["arch_config"]["depth"]["value"], 20)

    def test_f5tts_arch_force_init_mode_overrides_candidate_arg(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            f5_root = root / "F5-TTS"
            manifest_dir = root / "f5tts_train_manifests" / "libritts_train_clean_100_1h"
            f5_root.mkdir()
            manifest_dir.mkdir(parents=True)
            (manifest_dir / "metadata.csv").write_text("audio_path|text\n", encoding="utf-8")
            args = SimpleNamespace(
                action="arch_finetune_short",
                init_mode="partial_load",
                depth=22,
                ff_mult=2,
                conv_layers=4,
                qk_norm="none",
                attn_mask_enabled=False,
                checkpoint_activations=False,
                match_threshold=0.70,
                early_stop_ema_alpha=0.05,
                early_stop_min_relative_delta=0.002,
                base_ckpt=None,
                vocab_file=None,
                train_manifest="libritts_train_clean_100_1h",
                max_steps=1000,
                effective_batch_size=8,
                learning_rate="3e-6",
                f5_root=f5_root,
                train_data_root=root / "f5tts_train_manifests",
            )

            with patch.dict(os.environ, {"SURE_TTS_ARCH_FORCE_INIT_MODE": "scratch"}):
                learning_rate = run_f5tts_arch_finetune.validate_args(args)

        self.assertEqual(learning_rate, 3e-6)
        self.assertEqual(args.init_mode, "scratch")

    def test_f5tts_official_baseline_code_is_inference_candidate(self):
        code = Path(f5tts_official_baseline.__file__).read_text(encoding="utf-8")
        self.assertEqual(candidate_type_from_code(code, default=INFERENCE), INFERENCE)
        self.assertNotIn("SURE_TTS_TRAIN_ACTION", code)
        self.assertNotIn("SURE_TTS_FINETUNE_WRAPPER", code)

    def test_f5tts_official_draft_uses_inference_resource_profile(self):
        code = Path(f5tts_official_baseline.__file__).read_text(encoding="utf-8")
        exp = object.__new__(SureRunExp)
        exp.candidate_type_hint = FINE_TUNE
        exp.candidate_stage_name = "stage0_draft"
        exp.stage = "draft"
        exp.code = code

        self.assertEqual(exp._remote_workload_profile(), INFERENCE)

        config = {
            "sure": {
                "remote_training": {
                    "gpu_per_task": 8,
                    "cpu_per_task": 64,
                    "mem_per_task": "256G",
                    "resource_profiles": {
                        "inference": {
                            "gpu_per_task": 1,
                            "cpu_per_task": 8,
                            "mem_per_task": "32G",
                        },
                        "training": {
                            "gpu_per_task": 8,
                            "cpu_per_task": 64,
                            "mem_per_task": "256G",
                        },
                    },
                }
            }
        }
        resources, diagnostics = remote_resource_config_from(
            config,
            candidate_type=exp.candidate_type_hint,
            stage=exp.candidate_stage_name,
            workload_profile=exp._remote_workload_profile(),
        )
        self.assertEqual(diagnostics["profile_name"], INFERENCE)
        self.assertEqual(resources["gpu_per_task"], 1)
        self.assertEqual(resources["cpu_per_task"], 8)
        self.assertEqual(resources["mem_per_task"], "32G")

    def test_f5tts_official_baseline_builds_batch_wrapper_command(self):
        env = {
            "SURE_TTS_PYTHON": "/env/f5/bin/python",
            "SURE_TTS_BATCH_INFER_WRAPPER": "/repo/run_f5tts_batch_infer.py",
            "SURE_TTS_ROOT": "base_model/root",
            "SURE_TTS_EVAL_DATA": "base_model/eval_data/prompts.jsonl",
            "SURE_TTS_MAX_SAMPLES": "20",
            "SURE_TTS_CKPT_FILE": "/models/F5TTS_v1_Base/model_1250000.safetensors",
            "SURE_TTS_VOCAB_FILE": "/models/F5TTS_v1_Base/vocab.txt",
            "SURE_RUN_TIMEOUT": "1234",
        }
        with patch.dict(os.environ, env, clear=True):
            command = f5tts_official_baseline.build_batch_infer_command()

        self.assertEqual(command[0], "/env/f5/bin/python")
        self.assertEqual(command[1], "/repo/run_f5tts_batch_infer.py")
        self.assertIn("--candidate-type", command)
        self.assertEqual(command[command.index("--candidate-type") + 1], "inference")
        self.assertIn("--training-action", command)
        self.assertEqual(command[command.index("--training-action") + 1], "no_train")
        self.assertEqual(command[command.index("--timeout") + 1], "1234")
        self.assertEqual(command[command.index("--max-chunk-chars") + 1], "300")
        self.assertEqual(command[command.index("--min-chunk-chars") + 1], "40")

    def test_f5tts_official_baseline_writes_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp) / "artifacts"
            env = {
                "SURE_TTS_MODEL": "F5TTS_v1_Base",
                "SURE_TTS_CKPT_FILE": "/models/model_1250000.safetensors",
                "SURE_TTS_VOCAB_FILE": "/models/vocab.txt",
                "SURE_TTS_EVAL_DATA": "base_model/eval_data/prompts.jsonl",
            }
            with patch.object(f5tts_official_baseline, "ARTIFACTS_DIR", artifacts), patch.dict(
                os.environ,
                env,
                clear=True,
            ):
                f5tts_official_baseline.write_official_baseline_record(
                    command=["python", "wrapper.py"],
                    sample_count=3,
                    elapsed_seconds=1.25,
                )

            payload = json.loads((artifacts / "official_baseline.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["baseline_type"], "official")
        self.assertEqual(payload["task_id"], "tts_en_wer")
        self.assertEqual(payload["model_id"], "SWivid/F5-TTS/F5TTS_v1_Base")
        self.assertEqual(payload["sample_count"], 3)

    def test_local_env_timeout_kills_child_process_group_and_unregisters(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = LocalSessionConfig(
                workspace_path=tmp,
                timeout=10,
                parallel={"enabled": True, "max_parallel": 1},
            )
            env = LocalEnv(LocalEnvConfig(session_config=config))
            env.setup()
            marker = Path(tmp) / "child_done"
            child_command = f"sleep 2; touch {shlex.quote(str(marker))}"
            python_code = (
                "import subprocess, time; "
                f"subprocess.Popen(['sh', '-c', {child_command!r}]); "
                "time.sleep(10)"
            )
            command = f"{shlex.quote(sys.executable)} -c {shlex.quote(python_code)}"
            result = env.local_exec(command, timeout=1, parallel_index=0)
            time.sleep(2.5)
            self.assertEqual(result["exit_code"], -1)
            self.assertFalse(marker.exists())
            assert env._resource_allocator is not None
            self.assertEqual(env._resource_allocator._active_executions, {})

    def test_local_env_nonzero_exit_kills_leftover_child_process_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = LocalSessionConfig(workspace_path=tmp, timeout=10)
            env = LocalEnv(LocalEnvConfig(session_config=config))
            env.setup()
            marker = Path(tmp) / "nonzero_child_done"
            child_command = f"sleep 2; touch {shlex.quote(str(marker))}"
            python_code = (
                "import subprocess, sys; "
                f"subprocess.Popen(['sh', '-c', {child_command!r}]); "
                "sys.exit(7)"
            )
            command = f"{shlex.quote(sys.executable)} -c {shlex.quote(python_code)}"
            result = env.local_exec(command, timeout=5)
            time.sleep(2.5)
            self.assertEqual(result["exit_code"], 7)
            self.assertFalse(marker.exists())

    def test_metric_direction_defaults(self):
        self.assertEqual(infer_metric_direction("WER"), "lower")
        self.assertEqual(infer_metric_direction("tts_cer"), "lower")
        self.assertEqual(infer_metric_direction("accuracy"), "higher")
        self.assertEqual(infer_metric_direction("BLEU"), "higher")

    def test_load_repo_task_cards(self):
        path = Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml"
        cards = load_task_cards(path)
        self.assertIn("asr_en_wer", cards)
        self.assertIn("tts_en_wer", cards)
        self.assertTrue(cards["asr_en_wer"].is_lower_better)
        self.assertFalse(cards["classification_accuracy"].is_lower_better)

    def test_required_roles_have_artifact_contracts(self):
        path = Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml"
        cards = load_task_cards(path)
        for task_id, card in cards.items():
            with self.subTest(task_id=task_id):
                for role in card.required_roles:
                    self.assertIn(role, card.artifact_contract)
        self.assertEqual(cards["slu_accuracy"].required_roles, ["ref", "hyp", "prompt_jsonl"])

    def test_resolve_unknown_task_card_reports_available(self):
        path = Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml"
        with self.assertRaisesRegex(KeyError, "Available task cards"):
            resolve_task_card(path, "missing_task")

    def test_asr_en_wer_declares_required_zipformer_base_model(self):
        path = Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml"
        card = resolve_task_card(path, "asr_en_wer")
        self.assertIsNotNone(card.base_model)
        assert card.base_model is not None
        self.assertEqual(card.base_model.model_id, "icefall_zipformer_asr")
        self.assertEqual(card.base_model.framework, "icefall")
        self.assertTrue(card.base_model.is_required)
        self.assertIn("dataset profile", card.base_model.prompt_guidance)
        self.assertIn("recipe profile", card.base_model.prompt_guidance)
        self.assertNotIn("LibriSpeech", card.base_model.prompt_guidance)
        self.assertEqual(card.base_model.required_paths["recipe"], "base_model/recipe")
        self.assertEqual(card.base_model.entrypoints["infer"], "base_model/recipe/decode.py")

    def test_base_model_config_override_adds_source_paths(self):
        path = Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml"
        card = resolve_task_card(path, "asr_en_wer")
        profile = merge_base_model_profile(
            card.base_model,
            {
                "source_paths": {
                    "recipe": "/tmp/icefall/zipformer",
                    "data": "/tmp/icefall/data",
                    "root": "/tmp/icefall",
                }
            },
        )
        validate_base_model_profile(profile, "asr_en_wer")
        assert profile is not None
        self.assertEqual(profile.source_paths["recipe"], "/tmp/icefall/zipformer")
        self.assertEqual(profile.required_paths["recipe"], "base_model/recipe")

    def test_required_base_model_without_source_paths_fails_validation(self):
        path = Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml"
        card = resolve_task_card(path, "asr_en_wer")
        with self.assertRaisesRegex(ValueError, "requires base_model source_paths"):
            validate_base_model_profile(card.base_model, "asr_en_wer")

    def test_playground_requires_base_model_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "sure": {
                            "task_cards_path": str(
                                Path(__file__).resolve().parents[2]
                                / "task_cards"
                                / "sure_tasks.yaml"
                            ),
                            "task_id": "slu_accuracy",
                        },
                        "session": {"type": "local", "local": {"working_dir": str(Path(tmp) / "workspace")}},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "requires a base_model profile"):
                SureMasterPlayground(config_path=config_path)

    def test_playground_accepts_configured_base_model_for_other_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "sure": {
                            "task_cards_path": str(
                                Path(__file__).resolve().parents[2]
                                / "task_cards"
                                / "sure_tasks.yaml"
                            ),
                            "task_id": "slu_accuracy",
                            "base_models": {
                                "slu_accuracy": {
                                    "model_id": "example_slu_model",
                                    "model_type": "classification_model",
                                    "framework": "custom",
                                    "usage_policy": "required",
                                    "required_paths": {"model": "base_model/model"},
                                    "source_paths": {"model": "/tmp/example_slu_model"},
                                    "entrypoints": {"infer": "base_model/model/infer.py"},
                                }
                            },
                        },
                        "session": {"type": "local", "local": {"working_dir": str(Path(tmp) / "workspace")}},
                    }
                ),
                encoding="utf-8",
            )
            pg = SureMasterPlayground(config_path=config_path)
            assert pg.base_model_profile is not None
            self.assertEqual(pg.base_model_profile.model_id, "example_slu_model")


class WorkspaceCleanupTest(unittest.TestCase):
    def test_workspace_cleanup_config_allows_env_overrides(self):
        config = {
            "sure": {
                "workspace_cleanup": {
                    "enabled": False,
                    "on_failure": False,
                    "log_tail_bytes": 100,
                }
            }
        }
        env = {
            "SURE_WORKSPACE_CLEANUP_ENABLED": "1",
            "SURE_WORKSPACE_CLEANUP_ON_FAILURE": "1",
            "SURE_WORKSPACE_CLEANUP_LOG_TAIL_BYTES": "2048",
        }
        with patch.dict(os.environ, env, clear=True):
            cleanup_config = workspace_cleanup_config(config)

        self.assertTrue(cleanup_config.enabled)
        self.assertTrue(cleanup_config.on_failure)
        self.assertEqual(cleanup_config.log_tail_bytes, 2048)

    def test_cleanup_candidate_workspace_compacts_heavy_candidate_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "exp"
            (workspace / "models").mkdir(parents=True)
            (workspace / "working" / "decode").mkdir(parents=True)
            (workspace / ".sure_runtime" / "duration_autotune").mkdir(parents=True)
            (workspace / ".sure_runtime" / "duration_autotune" / "probes" / "k" / "duration_100").mkdir(parents=True)
            (workspace / "artifacts").mkdir(parents=True)
            (workspace / "metric").mkdir(parents=True)

            (workspace / "run_sure.py").write_text("print('ok')\n", encoding="utf-8")
            (workspace / "models" / "epoch-1.pt").write_bytes(b"x" * 1024)
            (workspace / "working" / "decode" / "train.log").write_text(
                "start\n" + ("middle\n" * 100) + "final-error\n",
                encoding="utf-8",
            )
            (workspace / ".sure_runtime" / "duration_autotune" / "probe.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            (
                workspace
                / ".sure_runtime"
                / "duration_autotune"
                / "probes"
                / "k"
                / "duration_100"
                / "probe.log"
            ).write_text(
                "probe-start\nModuleNotFoundError: No module named icefall\n",
                encoding="utf-8",
            )
            (workspace / "artifacts" / "hyp.txt").write_text("utt\ttext\n", encoding="utf-8")
            (workspace / "artifacts" / "candidate_status.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            (workspace / "metric" / "score_summary.json").write_text(
                '{"score": 12.3}\n',
                encoding="utf-8",
            )
            (workspace / "metric" / "remote_training_vc_submit.log").write_text(
                "submit-start\n" + ("submit-middle\n" * 100) + "submit-end\n",
                encoding="utf-8",
            )

            manifest = cleanup_candidate_workspace(
                workspace,
                WorkspaceCleanupConfig(
                    enabled=True,
                    on_success=True,
                    on_failure=True,
                    log_tail_bytes=64,
                ),
                success=True,
                reason_code="success",
                score=12.3,
            )

            self.assertTrue(manifest["cleaned"])
            self.assertGreater(manifest["removed_bytes"], 0)
            self.assertTrue((workspace / "run_sure.py").is_file())
            self.assertTrue((workspace / "artifacts" / "hyp.txt").is_file())
            self.assertTrue((workspace / "artifacts" / "candidate_status.json").is_file())
            self.assertTrue((workspace / "metric" / "score_summary.json").is_file())
            self.assertTrue((workspace / "metric" / "cleanup_manifest.json").is_file())
            self.assertFalse((workspace / "models").exists())
            self.assertFalse((workspace / "working").exists())
            self.assertFalse((workspace / ".sure_runtime").exists())
            self.assertFalse((workspace / "metric" / "remote_training_vc_submit.log").exists())

            tail_files = sorted((workspace / "metric" / "log_tails").glob("*.tail.log"))
            self.assertGreaterEqual(len(tail_files), 2)
            tail_text = "\n".join(path.read_text(encoding="utf-8") for path in tail_files)
            self.assertIn("final-error", tail_text)
            self.assertIn("submit-end", tail_text)
            self.assertIn("ModuleNotFoundError", tail_text)

    def test_cleanup_workspace_finder_includes_outer_heavy_candidate_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            outer = root / "workspaces" / "task_0" / "exp_1_draft"
            inner = outer / "exp_1_draft"
            (outer / "artifacts").mkdir(parents=True)
            (outer / "models" / "zipformer_candidate").mkdir(parents=True)
            (outer / "working").mkdir()
            (outer / "artifacts" / "hyp.txt").write_text("utt\ttext\n", encoding="utf-8")
            (inner / "artifacts").mkdir(parents=True)
            (inner / "run_sure.py").write_text("print('ok')\n", encoding="utf-8")

            workspaces = find_candidate_workspaces(root)

            self.assertIn(outer, workspaces)
            self.assertIn(inner, workspaces)


class SureMetricRunnerTest(unittest.TestCase):
    def test_local_env_creates_icefall_data_compatibility_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            recipe = workspace / "base_model" / "recipe"
            data = workspace / "base_model" / "data"
            recipe.mkdir(parents=True)
            data.mkdir(parents=True)

            env = SureMasterLocalEnv(
                LocalEnvConfig(
                    session_config=LocalSessionConfig(workspace_path=str(workspace))
                )
            )
            env._ensure_base_data_links(workspace)

            recipe_data = recipe / "data"
            workspace_data = workspace / "data"
            self.assertTrue(recipe_data.is_symlink())
            self.assertEqual(recipe_data.readlink(), Path("../data"))
            self.assertTrue(workspace_data.is_symlink())
            self.assertEqual(workspace_data.readlink(), Path("base_model/data"))

    def test_run_exp_copies_configured_required_input_to_workspace_default(self):
        card = resolve_task_card(
            Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml",
            "asr_en_wer",
        )
        runner = SureMetricRunner("/tmp/sure", pythonpath="/tmp/sure/src")

        class DummySession:
            class Config:
                workspace_path = ""

            config = Config()

        class DummyAgent:
            session = DummySession()

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "exp"
            source = Path(tmp) / "ref_source.txt"
            source.write_text("utt1\tHELLO WORLD\n", encoding="utf-8")
            DummyAgent.session.config.workspace_path = str(Path(tmp))

            exp = SureRunExp(
                stage="draft",
                main_agent=DummyAgent(),
                debug_agent=DummyAgent(),
                config={},
                exp_name="exp",
                task_card=card,
                base_model_profile=None,
                metric_runner=runner,
            )
            exp.workspace_path = str(workspace)
            exp._ensure_workspace_dirs()
            exp._prepare_workspace_inputs({"ref": str(source), "hyp": "artifacts/hyp.txt"})

            copied = workspace / "input" / "ref.txt"
            self.assertTrue(copied.exists())
            self.assertEqual(copied.read_text(encoding="utf-8"), "utt1\tHELLO WORLD\n")

    def test_run_exp_does_not_duplicate_session_exp_workspace(self):
        runner = SureMetricRunner("/tmp/sure", pythonpath="/tmp/sure/src")
        card = SureTaskCard(
            task_id="asr_en_wer",
            canonical_task="asr",
            task_alias="asr",
            primary_metric="WER",
            required_roles=["hyp", "ref"],
            artifact_contract={"hyp": "artifacts/hyp.txt", "ref": "input/ref.txt"},
        )

        class DummySession:
            class Config:
                workspace_path = ""

            config = Config()

        class DummyAgent:
            session = DummySession()

        with tempfile.TemporaryDirectory() as tmp:
            exp_workspace = Path(tmp) / "task_0" / "exp_1_draft"
            DummyAgent.session.config.workspace_path = str(exp_workspace)
            exp = SureRunExp(
                stage="draft",
                main_agent=DummyAgent(),
                debug_agent=DummyAgent(),
                config={},
                exp_name="exp_1_draft",
                task_card=card,
                base_model_profile=None,
                metric_runner=runner,
            )

            self.assertEqual(exp._resolve_workspace_path(), str(exp_workspace))

            with patch.object(exp, "_ensure_workspace_dirs"), patch.object(
                exp,
                "_prepare_base_model_source_overrides",
            ), patch.object(exp, "_prepare_workspace_inputs"), patch.object(
                exp,
                "_execute_and_score",
                return_value=(False, None, {"stub": True}),
            ):
                exp.run_existing_code(code="print('ok')", role_paths={"hyp": "artifacts/hyp.txt"})

            self.assertEqual(exp.workspace_path, str(exp_workspace))

    def test_run_exp_applies_base_model_source_override_symlink(self):
        runner = SureMetricRunner("/tmp/sure", pythonpath="/tmp/sure/src")
        profile = BaseModelProfile(
            model_id="f5tts",
            usage_policy="required",
            required_paths={"eval_data": "base_model/eval_data"},
            source_paths={"eval_data": "/tmp/default_eval_data"},
        )
        card = SureTaskCard(
            task_id="tts_en_wer",
            canonical_task="tts",
            task_alias="tts",
            primary_metric="tts_wer",
            required_roles=["samples_jsonl"],
            artifact_contract={"samples_jsonl": "artifacts/samples.jsonl"},
        )

        class DummySession:
            class Config:
                workspace_path = ""

            config = Config()

        class DummyAgent:
            session = DummySession()

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "exp"
            selection_eval = Path(tmp) / "selection_eval"
            selection_eval.mkdir()
            DummyAgent.session.config.workspace_path = str(Path(tmp))
            exp = SureRunExp(
                stage="draft",
                main_agent=DummyAgent(),
                debug_agent=DummyAgent(),
                config={},
                exp_name="exp",
                task_card=card,
                base_model_profile=profile,
                metric_runner=runner,
            )
            exp.workspace_path = str(workspace)
            exp.base_model_source_overrides = {"eval_data": str(selection_eval)}
            exp._ensure_workspace_dirs()
            exp._prepare_base_model_source_overrides()

            link = workspace / "base_model" / "eval_data"
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), selection_eval.resolve())

    def test_run_exp_replaces_directory_of_symlinks_for_base_model_override(self):
        runner = SureMetricRunner("/tmp/sure", pythonpath="/tmp/sure/src")
        profile = BaseModelProfile(
            model_id="f5tts",
            usage_policy="required",
            required_paths={"eval_data": "base_model/eval_data"},
            source_paths={"eval_data": "/tmp/default_eval_data"},
        )
        card = SureTaskCard(
            task_id="tts_en_wer",
            canonical_task="tts",
            task_alias="tts",
            primary_metric="tts_wer",
            required_roles=["samples_jsonl"],
            artifact_contract={"samples_jsonl": "artifacts/samples.jsonl"},
        )

        class DummySession:
            class Config:
                workspace_path = ""

            config = Config()

        class DummyAgent:
            session = DummySession()

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "exp"
            default_eval = Path(tmp) / "default_eval"
            selection_eval = Path(tmp) / "selection_eval"
            default_eval.mkdir()
            selection_eval.mkdir()
            (default_eval / "prompts.jsonl").write_text("default\n", encoding="utf-8")
            (selection_eval / "prompts.jsonl").write_text("selection\n", encoding="utf-8")
            target = workspace / "base_model" / "eval_data"
            target.mkdir(parents=True)
            (target / "prompts.jsonl").symlink_to(default_eval / "prompts.jsonl")
            DummyAgent.session.config.workspace_path = str(Path(tmp))
            exp = SureRunExp(
                stage="draft",
                main_agent=DummyAgent(),
                debug_agent=DummyAgent(),
                config={},
                exp_name="exp",
                task_card=card,
                base_model_profile=profile,
                metric_runner=runner,
            )
            exp.workspace_path = str(workspace)
            exp.base_model_source_overrides = {"eval_data": str(selection_eval)}

            exp._prepare_base_model_source_overrides()

            self.assertTrue(target.is_symlink())
            self.assertEqual(target.resolve(), selection_eval.resolve())
            self.assertEqual((target / "prompts.jsonl").read_text(encoding="utf-8"), "selection\n")

    def test_run_exp_preserves_real_directory_for_base_model_override(self):
        runner = SureMetricRunner("/tmp/sure", pythonpath="/tmp/sure/src")
        profile = BaseModelProfile(
            model_id="f5tts",
            usage_policy="required",
            required_paths={"eval_data": "base_model/eval_data"},
            source_paths={"eval_data": "/tmp/default_eval_data"},
        )
        card = SureTaskCard(
            task_id="tts_en_wer",
            canonical_task="tts",
            task_alias="tts",
            primary_metric="tts_wer",
        )

        class DummySession:
            class Config:
                workspace_path = ""

            config = Config()

        class DummyAgent:
            session = DummySession()

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "exp"
            selection_eval = Path(tmp) / "selection_eval"
            selection_eval.mkdir()
            target = workspace / "base_model" / "eval_data"
            target.mkdir(parents=True)
            real_file = target / "local.txt"
            real_file.write_text("keep\n", encoding="utf-8")
            DummyAgent.session.config.workspace_path = str(Path(tmp))
            exp = SureRunExp(
                stage="draft",
                main_agent=DummyAgent(),
                debug_agent=DummyAgent(),
                config={},
                exp_name="exp",
                task_card=card,
                base_model_profile=profile,
                metric_runner=runner,
            )
            exp.workspace_path = str(workspace)
            exp.base_model_source_overrides = {"eval_data": str(selection_eval)}

            exp._prepare_base_model_source_overrides()

            self.assertFalse(target.is_symlink())
            self.assertEqual(real_file.read_text(encoding="utf-8"), "keep\n")

    def test_run_vc_candidate_ensures_base_model_profile_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_recipe = tmp_path / "icefall" / "recipe"
            source_data = tmp_path / "icefall" / "data"
            source_root = tmp_path / "icefall" / "root"
            for source in (source_recipe, source_data, source_root):
                source.mkdir(parents=True)
            profile = BaseModelProfile(
                model_id="zipformer",
                usage_policy="required",
                required_paths={
                    "recipe": "base_model/recipe",
                    "data": "base_model/data",
                    "root": "base_model/root",
                },
                source_paths={
                    "recipe": str(source_recipe),
                    "data": str(source_data),
                    "root": str(source_root),
                },
            )
            workspace = tmp_path / "workspace"
            workspace.mkdir()

            missing = ensure_base_model_paths(workspace, profile)

            self.assertEqual(missing, [])
            self.assertEqual((workspace / "base_model" / "recipe").resolve(), source_recipe.resolve())
            self.assertEqual((workspace / "base_model" / "data").resolve(), source_data.resolve())
            self.assertEqual((workspace / "base_model" / "root").resolve(), source_root.resolve())
            self.assertTrue((workspace / "base_model" / "recipe" / "data").is_symlink())
            self.assertTrue((workspace / "data").is_symlink())

    def test_coordinator_gpu_policy_skips_discovery_only_when_remote_or_disabled(self):
        all_remote = {
            "sure": {
                "coordinator": {"local_gpu_policy": "disabled"},
                "remote_training": {
                    "enabled": True,
                    "draft_enabled": True,
                    "candidate_types": ["inference", "fine_tune", "arch"],
                },
            }
        }
        with patch.object(
            sure_playground_module,
            "_discover_gpu_devices",
            side_effect=AssertionError("GPU discovery must be skipped"),
        ):
            resolved = sure_playground_module._apply_coordinator_gpu_policy(
                all_remote,
                {"gpu_devices": "auto", "parallel": {"gpus_per_exp": 8}},
                Mock(),
            )
        self.assertIsNone(resolved["gpu_devices"])
        for settings in (resolved, resolved["parallel"]):
            self.assertFalse(settings["refresh_idle_gpu_before_exec"])
            self.assertFalse(settings["gpu_lock_enabled"])
            self.assertFalse(settings["set_asr_world_size"])
            self.assertEqual(settings["gpus_per_exp"], 1)
            self.assertIsNone(settings["serial_gpus_per_exp"])

        auto_remote = {
            "sure": {
                "coordinator": {"local_gpu_policy": "auto"},
                "remote_training": dict(all_remote["sure"]["remote_training"]),
            }
        }
        with patch.object(
            sure_playground_module,
            "_discover_gpu_devices",
            side_effect=AssertionError("auto must skip discovery with complete coverage"),
        ):
            auto_resolved = sure_playground_module._apply_coordinator_gpu_policy(
                auto_remote, {"gpu_devices": "auto", "parallel": {}}, Mock()
            )
        self.assertIsNone(auto_resolved["gpu_devices"])

        arch_entry_remote = {
            "sure": {
                "coordinator": {"local_gpu_policy": "disabled"},
                "staged_axes": {"start_phase": "arch"},
                "remote_training": {
                    "enabled": True,
                    "draft_enabled": False,
                    "candidate_types": ["inference", "fine_tune", "arch"],
                },
            }
        }
        with patch.object(
            sure_playground_module,
            "_discover_gpu_devices",
            side_effect=AssertionError("arch entry must skip local GPU discovery"),
        ):
            arch_entry_resolved = sure_playground_module._apply_coordinator_gpu_policy(
                arch_entry_remote, {"gpu_devices": "auto", "parallel": {}}, Mock()
            )
        self.assertIsNone(arch_entry_resolved["gpu_devices"])

        incomplete_disabled = {
            "sure": {
                "coordinator": {"local_gpu_policy": "disabled"},
                "remote_training": {
                    "enabled": True,
                    "draft_enabled": True,
                    "candidate_types": ["fine_tune", "arch"],
                },
            }
        }
        with self.assertRaisesRegex(ValueError, "complete remote coverage"):
            sure_playground_module._apply_coordinator_gpu_policy(
                incomplete_disabled, {"gpu_devices": "auto"}, Mock()
            )

        partial_remote = {
            "sure": {
                "coordinator": {"local_gpu_policy": "auto"},
                "remote_training": {
                    "enabled": True,
                    "draft_enabled": True,
                    "candidate_types": ["fine_tune", "arch"],
                },
            }
        }
        with patch.object(sure_playground_module, "_discover_gpu_devices", return_value=["0"]):
            resolved = sure_playground_module._apply_coordinator_gpu_policy(
                partial_remote,
                {"gpu_devices": "auto"},
                Mock(),
            )
        self.assertEqual(resolved["gpu_devices"], ["0"])

    def test_run_exp_writes_remote_candidate_context(self):
        runner = SureMetricRunner("/tmp/sure", pythonpath="/tmp/sure/src")
        card = SureTaskCard(
            task_id="asr_en_wer",
            canonical_task="asr",
            task_alias="asr",
            primary_metric="WER",
        )

        class DummySession:
            class Config:
                workspace_path = ""

            config = Config()

        class DummyAgent:
            session = DummySession()

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "exp"
            DummyAgent.session.config.workspace_path = str(Path(tmp))
            exp = SureRunExp(
                stage="draft",
                main_agent=DummyAgent(),
                debug_agent=DummyAgent(),
                config={},
                exp_name="exp",
                task_card=card,
                base_model_profile=None,
                metric_runner=runner,
                execution_env={"SURE_RUN_TIMEOUT": "123"},
            )
            exp.workspace_path = str(workspace)
            exp._ensure_workspace_dirs()
            exp._current_role_paths = {"ref": "/selection/ref.txt", "hyp": "artifacts/hyp.txt"}
            exp.base_model_source_overrides = {"eval_data": "/selection/prompts"}
            exp.candidate_type_hint = ARCH
            exp._write_remote_candidate_context()

            payload = load_remote_candidate_context(workspace)
            self.assertEqual(payload["execution_env"]["SURE_RUN_TIMEOUT"], "123")
            self.assertEqual(payload["role_paths"]["ref"], "/selection/ref.txt")
            self.assertEqual(payload["base_model_source_overrides"]["eval_data"], "/selection/prompts")
            self.assertEqual(payload["candidate_type_hint"], ARCH)

    def test_run_exp_enforces_staged_candidate_type_before_execution(self):
        runner = SureMetricRunner("/tmp/sure", pythonpath="/tmp/sure/src")
        card = SureTaskCard(
            task_id="tts_en_wer",
            canonical_task="tts",
            task_alias="tts",
            primary_metric="tts_wer",
            required_roles=["samples_jsonl"],
            artifact_contract={"samples_jsonl": "artifacts/samples.jsonl"},
        )

        class DummySession:
            class Config:
                workspace_path = ""

            config = Config()

            def get_workspace_path(self):
                return None

        class DummyAgent:
            session = DummySession()

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "exp"
            DummyAgent.session.config.workspace_path = str(Path(tmp))
            exp = SureRunExp(
                stage="improve",
                main_agent=DummyAgent(),
                debug_agent=DummyAgent(),
                config={},
                exp_name="exp",
                task_card=card,
                base_model_profile=None,
                metric_runner=runner,
            )
            exp.workspace_path = str(workspace)
            exp.enforce_candidate_type = True
            code = "import os\nprint(os.environ.get('SURE_TTS_FINETUNE_WRAPPER'))\n"
            ok, score, _uid, _code, details = exp.run_existing_code(
                code=code,
                role_paths={"samples_jsonl": "artifacts/samples.jsonl"},
                candidate_type_hint=INFERENCE,
            )

            self.assertFalse(ok)
            self.assertIsNone(score)
            self.assertEqual(details["candidate_type_error"]["required"], INFERENCE)
            self.assertEqual(details["candidate_type_error"]["detected"], FINE_TUNE)

    def test_run_exp_rejects_asr_hypothesis_that_copies_reference(self):
        card = resolve_task_card(
            Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml",
            "asr_en_wer",
        )
        runner = SureMetricRunner("/tmp/sure", pythonpath="/tmp/sure/src")

        class DummySession:
            class Config:
                workspace_path = ""

            config = Config()

        class DummyAgent:
            session = DummySession()

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "exp"
            DummyAgent.session.config.workspace_path = str(Path(tmp))
            exp = SureRunExp(
                stage="draft",
                main_agent=DummyAgent(),
                debug_agent=DummyAgent(),
                config={},
                exp_name="exp",
                task_card=card,
                base_model_profile=None,
                metric_runner=runner,
            )
            exp.workspace_path = str(workspace)
            (workspace / "input").mkdir(parents=True)
            (workspace / "artifacts").mkdir(parents=True)
            lines = "".join(f"utt-{i}\tTHE SAME TEXT {i}\n" for i in range(120))
            (workspace / "input" / "ref.txt").write_text(lines, encoding="utf-8")
            (workspace / "artifacts" / "hyp.txt").write_text(lines, encoding="utf-8")

            errors = exp._artifact_guard_errors(
                {"ref": "input/ref.txt", "hyp": "artifacts/hyp.txt"}
            )

            self.assertTrue(errors)
            self.assertIn("copy the reference transcript", errors[0])

    def test_run_exp_rejects_low_diversity_asr_placeholder(self):
        card = resolve_task_card(
            Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml",
            "asr_en_wer",
        )
        runner = SureMetricRunner("/tmp/sure", pythonpath="/tmp/sure/src")

        class DummySession:
            class Config:
                workspace_path = ""

            config = Config()

        class DummyAgent:
            session = DummySession()

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "exp"
            DummyAgent.session.config.workspace_path = str(Path(tmp))
            exp = SureRunExp(
                stage="draft",
                main_agent=DummyAgent(),
                debug_agent=DummyAgent(),
                config={},
                exp_name="exp",
                task_card=card,
                base_model_profile=None,
                metric_runner=runner,
            )
            exp.workspace_path = str(workspace)
            (workspace / "input").mkdir(parents=True)
            (workspace / "artifacts").mkdir(parents=True)
            ref_lines = "".join(f"utt-{i}\tREAL REFERENCE TEXT {i}\n" for i in range(120))
            hyp_lines = "".join(f"utt-{i}\tTHE\n" for i in range(120))
            (workspace / "input" / "ref.txt").write_text(ref_lines, encoding="utf-8")
            (workspace / "artifacts" / "hyp.txt").write_text(hyp_lines, encoding="utf-8")

            errors = exp._artifact_guard_errors(
                {"ref": "input/ref.txt", "hyp": "artifacts/hyp.txt"}
            )

            self.assertTrue(errors)
            self.assertIn("low transcript diversity", errors[0])

    def test_run_exp_allows_blank_asr_hypothesis_from_decoder(self):
        card = resolve_task_card(
            Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml",
            "asr_en_wer",
        )
        runner = SureMetricRunner("/tmp/sure", pythonpath="/tmp/sure/src")

        class DummySession:
            class Config:
                workspace_path = ""

            config = Config()

        class DummyAgent:
            session = DummySession()

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "exp"
            DummyAgent.session.config.workspace_path = str(Path(tmp))
            exp = SureRunExp(
                stage="draft",
                main_agent=DummyAgent(),
                debug_agent=DummyAgent(),
                config={},
                exp_name="exp",
                task_card=card,
                base_model_profile=None,
                metric_runner=runner,
            )
            exp.workspace_path = str(workspace)
            (workspace / "input").mkdir(parents=True)
            (workspace / "artifacts").mkdir(parents=True)
            ref_lines = "".join(f"utt-{i}\tREAL REFERENCE TEXT {i}\n" for i in range(120))
            hyp_lines = "".join(
                f"utt-{i}\t{'HYPOTHESIS ' + str(i) if i != 37 else ''}\n"
                for i in range(120)
            )
            (workspace / "input" / "ref.txt").write_text(ref_lines, encoding="utf-8")
            (workspace / "artifacts" / "hyp.txt").write_text(hyp_lines, encoding="utf-8")

            errors = exp._artifact_guard_errors(
                {"ref": "input/ref.txt", "hyp": "artifacts/hyp.txt"}
            )

            self.assertEqual(errors, [])

    def test_run_exp_rejects_asr_training_candidate_with_baseline_checkpoint_only(self):
        card = resolve_task_card(
            Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml",
            "asr_en_wer",
        )
        runner = SureMetricRunner("/tmp/sure", pythonpath="/tmp/sure/src")

        class DummySession:
            class Config:
                workspace_path = ""

            config = Config()

        class DummyAgent:
            session = DummySession()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "exp"
            baseline_dir = root / "baseline_exp"
            baseline_dir.mkdir(parents=True)
            baseline_checkpoint = baseline_dir / "epoch-50.pt"
            baseline_checkpoint.write_bytes(b"official baseline checkpoint")
            candidate_dir = workspace / "models" / "candidate"
            candidate_dir.mkdir(parents=True)
            (candidate_dir / "epoch-50.pt").symlink_to(baseline_checkpoint)
            DummyAgent.session.config.workspace_path = str(root)
            exp = SureRunExp(
                stage="draft",
                main_agent=DummyAgent(),
                debug_agent=DummyAgent(),
                config={},
                exp_name="exp",
                task_card=card,
                base_model_profile=None,
                metric_runner=runner,
                execution_env={
                    "SURE_ASR_REQUIRE_LOCAL_CHECKPOINT": "1",
                    "SURE_ASR_FORBID_BASELINE_CHECKPOINT_FOR_TRAINING": "1",
                    "SURE_BASELINE_CHECKPOINT_DIR": str(baseline_dir),
                },
            )
            exp.workspace_path = str(workspace)
            exp.candidate_type_hint = ARCH
            exp.code = "cmd = ['train.py', '--encoder-dim', '192']"

            errors = exp._artifact_guard_errors(
                {"ref": "input/ref.txt", "hyp": "artifacts/hyp.txt"}
            )

        self.assertTrue(errors)
        self.assertIn("workspace-local epoch checkpoint", errors[0])

    def test_run_exp_rejects_asr_training_candidate_with_copied_baseline_checkpoint(self):
        card = resolve_task_card(
            Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml",
            "asr_en_wer",
        )
        runner = SureMetricRunner("/tmp/sure", pythonpath="/tmp/sure/src")

        class DummySession:
            class Config:
                workspace_path = ""

            config = Config()

        class DummyAgent:
            session = DummySession()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "exp"
            baseline_dir = root / "baseline_exp"
            baseline_dir.mkdir(parents=True)
            payload = b"official baseline checkpoint"
            (baseline_dir / "epoch-50.pt").write_bytes(payload)
            candidate_dir = workspace / "models" / "candidate"
            candidate_dir.mkdir(parents=True)
            (candidate_dir / "epoch-50.pt").write_bytes(payload)
            DummyAgent.session.config.workspace_path = str(root)
            exp = SureRunExp(
                stage="draft",
                main_agent=DummyAgent(),
                debug_agent=DummyAgent(),
                config={},
                exp_name="exp",
                task_card=card,
                base_model_profile=None,
                metric_runner=runner,
                execution_env={
                    "SURE_ASR_REQUIRE_LOCAL_CHECKPOINT": "1",
                    "SURE_ASR_FORBID_BASELINE_CHECKPOINT_FOR_TRAINING": "1",
                    "SURE_BASELINE_CHECKPOINT_DIR": str(baseline_dir),
                },
            )
            exp.workspace_path = str(workspace)
            exp.candidate_type_hint = ARCH
            exp.code = "cmd = ['train.py', '--encoder-dim', '192']"

            errors = exp._artifact_guard_errors(
                {"ref": "input/ref.txt", "hyp": "artifacts/hyp.txt"}
            )

        self.assertTrue(errors)
        self.assertIn("workspace-local epoch checkpoint", errors[0])

    def test_run_exp_allows_asr_training_candidate_with_local_checkpoint(self):
        card = resolve_task_card(
            Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml",
            "asr_en_wer",
        )
        runner = SureMetricRunner("/tmp/sure", pythonpath="/tmp/sure/src")

        class DummySession:
            class Config:
                workspace_path = ""

            config = Config()

        class DummyAgent:
            session = DummySession()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "exp"
            baseline_dir = root / "baseline_exp"
            baseline_dir.mkdir(parents=True)
            (baseline_dir / "epoch-50.pt").write_bytes(b"official baseline checkpoint")
            candidate_dir = workspace / "models" / "candidate"
            candidate_dir.mkdir(parents=True)
            (candidate_dir / "epoch-1.pt").write_bytes(b"candidate checkpoint")
            DummyAgent.session.config.workspace_path = str(root)
            exp = SureRunExp(
                stage="draft",
                main_agent=DummyAgent(),
                debug_agent=DummyAgent(),
                config={},
                exp_name="exp",
                task_card=card,
                base_model_profile=None,
                metric_runner=runner,
                execution_env={
                    "SURE_ASR_REQUIRE_LOCAL_CHECKPOINT": "1",
                    "SURE_ASR_FORBID_BASELINE_CHECKPOINT_FOR_TRAINING": "1",
                    "SURE_BASELINE_CHECKPOINT_DIR": str(baseline_dir),
                },
            )
            exp.workspace_path = str(workspace)
            exp.candidate_type_hint = ARCH
            exp.code = "cmd = ['train.py', '--encoder-dim', '192']"

            errors = exp._artifact_guard_errors(
                {"ref": "input/ref.txt", "hyp": "artifacts/hyp.txt"}
            )

        self.assertEqual(errors, [])

    def test_run_exp_rejects_asr_training_duration_below_floor(self):
        card = resolve_task_card(
            Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml",
            "asr_en_wer",
        )
        runner = SureMetricRunner("/tmp/sure", pythonpath="/tmp/sure/src")

        class DummySession:
            class Config:
                workspace_path = ""

            config = Config()

        class DummyAgent:
            session = DummySession()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "exp"
            artifacts = workspace / "artifacts"
            artifacts.mkdir(parents=True)
            (artifacts / "candidate_changes.json").write_text(
                json.dumps(
                    {
                        "candidate_type": "arch",
                        "training_config": {
                            "actual_train_max_duration": 16,
                            "train_epochs": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            DummyAgent.session.config.workspace_path = str(root)
            exp = SureRunExp(
                stage="draft",
                main_agent=DummyAgent(),
                debug_agent=DummyAgent(),
                config={},
                exp_name="exp",
                task_card=card,
                base_model_profile=None,
                metric_runner=runner,
                execution_env={
                    "SURE_DURATION_AUTOTUNE_MIN": "100",
                    "SURE_ASR_REQUIRE_LOCAL_CHECKPOINT": "1",
                },
            )
            exp.workspace_path = str(workspace)
            exp.candidate_type_hint = ARCH
            exp.code = "cmd = ['train.py', '--encoder-dim', '192']"

            errors = exp._artifact_guard_errors(
                {"ref": "input/ref.txt", "hyp": "artifacts/hyp.txt"}
            )

        self.assertTrue(errors)
        self.assertIn("max-duration below the configured floor", errors[0])

    def test_run_exp_allows_official_asr_baseline_draft(self):
        card = resolve_task_card(
            Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml",
            "asr_en_wer",
        )
        runner = SureMetricRunner("/tmp/sure", pythonpath="/tmp/sure/src")

        class DummySession:
            class Config:
                workspace_path = ""

            config = Config()

        class DummyAgent:
            session = DummySession()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "exp"
            baseline_dir = root / "baseline_exp"
            baseline_dir.mkdir(parents=True)
            payload = b"official baseline checkpoint"
            (baseline_dir / "epoch-50.pt").write_bytes(payload)
            candidate_dir = workspace / "models" / "candidate"
            candidate_dir.mkdir(parents=True)
            (candidate_dir / "epoch-50.pt").write_bytes(payload)
            (workspace / "artifacts").mkdir(parents=True)
            (workspace / "artifacts" / "official_baseline.json").write_text(
                json.dumps({"baseline_type": "official"}),
                encoding="utf-8",
            )
            DummyAgent.session.config.workspace_path = str(root)
            exp = SureRunExp(
                stage="draft",
                main_agent=DummyAgent(),
                debug_agent=DummyAgent(),
                config={},
                exp_name="exp",
                task_card=card,
                base_model_profile=None,
                metric_runner=runner,
                execution_env={
                    "SURE_ASR_REQUIRE_LOCAL_CHECKPOINT": "1",
                    "SURE_ASR_FORBID_BASELINE_CHECKPOINT_FOR_TRAINING": "1",
                    "SURE_BASELINE_CHECKPOINT_DIR": str(baseline_dir),
                },
            )
            exp.workspace_path = str(workspace)
            exp.candidate_type_hint = FINE_TUNE

            errors = exp._artifact_guard_errors(
                {"ref": "input/ref.txt", "hyp": "artifacts/hyp.txt"}
            )

        self.assertEqual(errors, [])

    def test_run_exp_rejects_asr_training_fatal_without_later_checkpoint(self):
        card = resolve_task_card(
            Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml",
            "asr_en_wer",
        )
        runner = SureMetricRunner("/tmp/sure", pythonpath="/tmp/sure/src")

        class DummySession:
            class Config:
                workspace_path = ""

            config = Config()

        class DummyAgent:
            session = DummySession()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "exp"
            (workspace / "working").mkdir(parents=True)
            (workspace / "working" / "train.log").write_text(
                "RuntimeError: CUDA out of memory\n",
                encoding="utf-8",
            )
            DummyAgent.session.config.workspace_path = str(root)
            exp = SureRunExp(
                stage="draft",
                main_agent=DummyAgent(),
                debug_agent=DummyAgent(),
                config={},
                exp_name="exp",
                task_card=card,
                base_model_profile=None,
                metric_runner=runner,
                execution_env={"SURE_ASR_FAIL_ON_TRAIN_FATAL_WITHOUT_CHECKPOINT": "1"},
            )
            exp.workspace_path = str(workspace)
            exp.candidate_type_hint = FINE_TUNE

            errors = exp._artifact_guard_errors(
                {"ref": "input/ref.txt", "hyp": "artifacts/hyp.txt"}
            )

        self.assertTrue(errors)
        self.assertIn("fatal CUDA/CUBLAS training failure", errors[0])

    def test_run_exp_rejects_tts_prediction_that_copies_reference_audio(self):
        card = resolve_task_card(
            Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml",
            "tts_en_wer",
        )
        runner = SureMetricRunner("/tmp/sure", pythonpath="/tmp/sure/src")

        class DummySession:
            class Config:
                workspace_path = ""

            config = Config()

        class DummyAgent:
            session = DummySession()

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "exp"
            DummyAgent.session.config.workspace_path = str(Path(tmp))
            exp = SureRunExp(
                stage="draft",
                main_agent=DummyAgent(),
                debug_agent=DummyAgent(),
                config={},
                exp_name="exp",
                task_card=card,
                base_model_profile=None,
                metric_runner=runner,
            )
            exp.workspace_path = str(workspace)
            (workspace / "artifacts" / "wavs").mkdir(parents=True)
            (workspace / "input").mkdir(parents=True)
            payload = b"RIFF" + (b"0" * 2048)
            (workspace / "artifacts" / "wavs" / "out.wav").write_bytes(payload)
            (workspace / "input" / "ref.wav").write_bytes(payload)
            row = {
                "sample_id": "tts-1",
                "prediction_audio": "wavs/out.wav",
                "reference_audio": "../input/ref.wav",
                "reference_text": "hello world",
                "language": "en",
            }
            (workspace / "artifacts" / "samples.jsonl").write_text(
                json.dumps(row) + "\n",
                encoding="utf-8",
            )

            errors = exp._artifact_guard_errors({"samples_jsonl": "artifacts/samples.jsonl"})

            self.assertTrue(errors)
            self.assertTrue(any("byte-identical" in error for error in errors))

    def test_run_exp_rejects_tts_reusing_one_prediction_for_all_samples(self):
        card = resolve_task_card(
            Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml",
            "tts_en_wer",
        )
        runner = SureMetricRunner("/tmp/sure", pythonpath="/tmp/sure/src")

        class DummySession:
            class Config:
                workspace_path = ""

            config = Config()

        class DummyAgent:
            session = DummySession()

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "exp"
            DummyAgent.session.config.workspace_path = str(Path(tmp))
            exp = SureRunExp(
                stage="draft",
                main_agent=DummyAgent(),
                debug_agent=DummyAgent(),
                config={},
                exp_name="exp",
                task_card=card,
                base_model_profile=None,
                metric_runner=runner,
            )
            exp.workspace_path = str(workspace)
            (workspace / "artifacts" / "wavs").mkdir(parents=True)
            (workspace / "artifacts" / "wavs" / "shared.wav").write_bytes(
                b"RIFF" + (b"1" * 2048)
            )
            rows = [
                {
                    "sample_id": "tts-1",
                    "prediction_audio": "wavs/shared.wav",
                    "reference_text": "first target",
                    "language": "en",
                },
                {
                    "sample_id": "tts-2",
                    "prediction_audio": "wavs/shared.wav",
                    "reference_text": "second target",
                    "language": "en",
                },
            ]
            (workspace / "artifacts" / "samples.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            errors = exp._artifact_guard_errors({"samples_jsonl": "artifacts/samples.jsonl"})

            self.assertTrue(errors)
            self.assertTrue(any("reuses the same prediction_audio" in error for error in errors))

    def test_run_exp_requires_tts_candidate_changes_when_configured(self):
        card = resolve_task_card(
            Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml",
            "tts_en_wer",
        )
        runner = SureMetricRunner("/tmp/sure", pythonpath="/tmp/sure/src")

        class DummySession:
            class Config:
                workspace_path = ""

            config = Config()

        class DummyAgent:
            session = DummySession()

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "exp"
            DummyAgent.session.config.workspace_path = str(Path(tmp))
            exp = SureRunExp(
                stage="draft",
                main_agent=DummyAgent(),
                debug_agent=DummyAgent(),
                config={},
                exp_name="exp",
                task_card=card,
                base_model_profile=None,
                metric_runner=runner,
                execution_env={"SURE_REQUIRE_CANDIDATE_CHANGES": "1"},
            )
            exp.workspace_path = str(workspace)
            (workspace / "artifacts" / "wavs").mkdir(parents=True)
            (workspace / "artifacts" / "wavs" / "out.wav").write_bytes(
                b"RIFF" + (b"2" * 2048)
            )
            row = {
                "sample_id": "tts-1",
                "prediction_audio": "wavs/out.wav",
                "reference_text": "target",
                "language": "en",
            }
            (workspace / "artifacts" / "samples.jsonl").write_text(
                json.dumps(row) + "\n",
                encoding="utf-8",
            )

            errors = exp._artifact_guard_errors({"samples_jsonl": "artifacts/samples.jsonl"})
            self.assertTrue(any("candidate change record is missing" in error for error in errors))

            (workspace / "artifacts" / "candidate_changes.json").write_text(
                json.dumps(
                    {
                        "candidate_type": "inference",
                        "idea_text": "baseline inference",
                        "changed_fields": [],
                        "arch_config": {},
                        "training_config": {},
                        "inference_config": {},
                        "defaults": {},
                        "diff_from_defaults": {},
                        "produced_artifacts": {"samples_jsonl": "artifacts/samples.jsonl"},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(exp._artifact_guard_errors({"samples_jsonl": "artifacts/samples.jsonl"}), [])

    def test_run_exp_uses_configured_execution_timeout(self):
        card = resolve_task_card(
            Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml",
            "classification_accuracy",
        )
        runner = SureMetricRunner("/tmp/sure", pythonpath="/tmp/sure/src")

        class DummySession:
            class Config:
                workspace_path = ""

            config = Config()

        class DummyAgent:
            session = DummySession()

        exp = SureRunExp(
            stage="draft",
            main_agent=DummyAgent(),
            debug_agent=DummyAgent(),
            config={},
            exp_name="exp",
            task_card=card,
            base_model_profile=None,
            metric_runner=runner,
            execution_env={"SURE_RUN_TIMEOUT": "123.0"},
        )

        self.assertEqual(exp._execution_timeout(), "123")

    def test_run_exp_zero_execution_timeout_means_unlimited(self):
        card = resolve_task_card(
            Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml",
            "classification_accuracy",
        )
        runner = SureMetricRunner("/tmp/sure", pythonpath="/tmp/sure/src")
        agent = SimpleNamespace(
            session=SimpleNamespace(config=SimpleNamespace(workspace_path=""))
        )
        exp = SureRunExp(
            stage="draft",
            main_agent=agent,
            debug_agent=agent,
            config={},
            exp_name="exp",
            task_card=card,
            base_model_profile=None,
            metric_runner=runner,
            execution_env={"SURE_RUN_TIMEOUT": "0"},
        )

        self.assertEqual(exp._execution_timeout(), "0")

    def test_run_exp_exports_runtime_helper_without_pre_resolving_duration(self):
        card = resolve_task_card(
            Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml",
            "asr_en_wer",
        )
        runner = SureMetricRunner("/tmp/sure", pythonpath="/tmp/sure/src")

        class DummySession:
            class Config:
                workspace_path = ""

            config = Config()

        class DummyAgent:
            session = DummySession()

        with tempfile.TemporaryDirectory() as tmp:
            DummyAgent.session.config.workspace_path = str(Path(tmp))
            exp = SureRunExp(
                stage="draft",
                main_agent=DummyAgent(),
                debug_agent=DummyAgent(),
                config={},
                exp_name="exp",
                task_card=card,
                base_model_profile=None,
                metric_runner=runner,
                execution_env={"SURE_MAX_DURATION": "auto"},
            )
            exp.code = "print('candidate')\n"
            command = exp._execution_command()

        self.assertIn("SURE_RUNTIME_ENV_HELPER", command)
        self.assertIn("SURE_MAX_DURATION=auto", command)
        self.assertNotIn("resolve-max-duration", command)

    def test_run_exp_uses_configured_initial_solution_for_draft(self):
        card = resolve_task_card(
            Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml",
            "classification_accuracy",
        )
        runner = SureMetricRunner("/tmp/sure", pythonpath="/tmp/sure/src")

        class DummySession:
            class Config:
                workspace_path = ""

            config = Config()

        class DummyAgent:
            session = DummySession()

        with tempfile.TemporaryDirectory() as tmp:
            solution = Path(tmp) / "baseline.py"
            solution.write_text("print('fixed draft')\n", encoding="utf-8")
            DummyAgent.session.config.workspace_path = str(Path(tmp))

            exp = SureRunExp(
                stage="draft",
                main_agent=DummyAgent(),
                debug_agent=DummyAgent(),
                config={"sure": {"initial_solution_path": str(solution)}},
                exp_name="exp",
                task_card=card,
                base_model_profile=None,
                metric_runner=runner,
            )

            response = exp._initial_solution_response()

        self.assertIsNotNone(response)
        self.assertIn("print('fixed draft')", response or "")

    def test_run_exp_initial_solution_failure_does_not_enter_debug_loop(self):
        card = resolve_task_card(
            Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml",
            "classification_accuracy",
        )
        runner = SureMetricRunner("/tmp/sure", pythonpath="/tmp/sure/src")

        class DummySession:
            class Config:
                workspace_path = ""

            config = Config()

        class DummyAgent:
            session = DummySession()

        with tempfile.TemporaryDirectory() as tmp:
            solution = Path(tmp) / "baseline.py"
            solution.write_text("print('fixed draft')\n", encoding="utf-8")
            DummyAgent.session.config.workspace_path = str(Path(tmp))

            exp = SureRunExp(
                stage="draft",
                main_agent=DummyAgent(),
                debug_agent=DummyAgent(),
                config={"sure": {"initial_solution_path": str(solution)}},
                exp_name="exp",
                task_card=card,
                base_model_profile=None,
                metric_runner=runner,
            )
            with patch.object(
                exp,
                "_execute_and_score",
                return_value=(False, None, {"error": "baseline failed"}),
            ) as execute_mock, patch.object(
                exp,
                "_run_debug_agent",
                side_effect=AssertionError("debug should not run for locked initial solution"),
            ):
                result = exp.run(
                    task_description="task",
                    data_preview="preview",
                    role_paths={"hyp": "artifacts/hyp.txt", "ref": "input/ref.txt"},
                )

        self.assertFalse(result[0])
        self.assertEqual(result[4], {"error": "baseline failed"})
        execute_mock.assert_called_once()

    def test_run_exp_initial_solution_keeps_candidate_type_hint(self):
        card = resolve_task_card(
            Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml",
            "asr_en_wer",
        )
        runner = SureMetricRunner("/tmp/sure", pythonpath="/tmp/sure/src")

        class DummySession:
            class Config:
                workspace_path = ""

            config = Config()

        class DummyAgent:
            session = DummySession()

        exp = SureRunExp(
            stage="draft",
            main_agent=DummyAgent(),
            debug_agent=DummyAgent(),
            config={"sure": {"initial_solution_path": "/tmp/baseline.py"}},
            exp_name="exp",
            task_card=card,
            base_model_profile=None,
            metric_runner=runner,
        )
        exp.candidate_type_hint = INFERENCE
        exp._current_response_is_initial_solution = True

        code = "cmd = ['base_model/recipe/train.py', '--num-encoder-layers', '2,2,4,5,4,2']"
        self.assertEqual(exp._candidate_type_from_code(code), INFERENCE)

        exp.candidate_type_hint = FINE_TUNE
        self.assertEqual(exp._candidate_type_from_code(code), FINE_TUNE)

    def test_run_exp_does_not_use_initial_solution_for_improve(self):
        card = resolve_task_card(
            Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml",
            "classification_accuracy",
        )
        runner = SureMetricRunner("/tmp/sure", pythonpath="/tmp/sure/src")

        class DummySession:
            class Config:
                workspace_path = ""

            config = Config()

        class DummyAgent:
            session = DummySession()

        with tempfile.TemporaryDirectory() as tmp:
            solution = Path(tmp) / "baseline.py"
            solution.write_text("print('fixed draft')\n", encoding="utf-8")
            DummyAgent.session.config.workspace_path = str(Path(tmp))

            exp = SureRunExp(
                stage="improve",
                main_agent=DummyAgent(),
                debug_agent=DummyAgent(),
                config={"sure": {"initial_solution_path": str(solution)}},
                exp_name="exp",
                task_card=card,
                base_model_profile=None,
                metric_runner=runner,
            )

            response = exp._initial_solution_response()

        self.assertIsNone(response)

    def test_build_run_args_resolves_workspace_relative_paths(self):
        path = Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml"
        card = resolve_task_card(path, "asr_en_wer")
        runner = SureMetricRunner("/tmp/sure", pythonpath="/tmp/sure/src")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            args = runner._build_run_args(
                card,
                workspace,
                {"ref": "input/ref.txt", "hyp": "artifacts/hyp.txt"},
            )
        self.assertEqual(set(args), {"ref_file", "hyp_file"})
        self.assertTrue(args["ref_file"].endswith("input/ref.txt"))
        self.assertTrue(args["hyp_file"].endswith("artifacts/hyp.txt"))

    def test_run_uses_fake_sure_module_without_touching_sure_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved_sure_modules = {
                name: module
                for name, module in sys.modules.items()
                if name == "sure_eval" or name.startswith("sure_eval.")
            }
            for name in list(saved_sure_modules):
                sys.modules.pop(name, None)

            root = Path(tmp) / "sure"
            src = root / "src"
            pkg = src / "sure_eval" / "evaluation"
            pkg.mkdir(parents=True)
            (src / "sure_eval" / "__init__.py").write_text("", encoding="utf-8")
            (pkg / "__init__.py").write_text("", encoding="utf-8")
            (pkg / "cli_adapters.py").write_text(
                """
def build_pipeline_spec(task, language=None, metric=None, output_path=None):
    return {"task": task, "language": language, "metric": metric, "pipeline_id": "fake"}

def run_pipeline_spec(pipeline, output_dir, **kwargs):
    return {
        "status": "ok",
        "task": pipeline["task"],
        "metric": pipeline["metric"],
        "score": 12.5,
        "pipeline_id": "fake",
        "output_dir": output_dir,
        "report_path": output_dir + "/report.json",
        "pipeline_description_path": output_dir + "/pipeline_description.json",
    }
""",
                encoding="utf-8",
            )
            workspace = Path(tmp) / "workspace"
            (workspace / "input").mkdir(parents=True)
            (workspace / "artifacts").mkdir()
            (workspace / "input" / "ref.txt").write_text("utt1\tHELLO\n", encoding="utf-8")
            (workspace / "artifacts" / "hyp.txt").write_text("utt1\tHELLO\n", encoding="utf-8")

            card = resolve_task_card(
                Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml",
                "asr_en_wer",
            )
            try:
                result = SureMetricRunner(root, pythonpath=src, device="cpu").run(
                    task_card=card,
                    workspace_path=workspace,
                    output_dir=workspace / "metric",
                    role_paths={},
                )
            finally:
                for name in [
                    name
                    for name in list(sys.modules)
                    if name == "sure_eval" or name.startswith("sure_eval.")
                ]:
                    sys.modules.pop(name, None)
                sys.modules.update(saved_sure_modules)

            self.assertTrue(result.success)
            self.assertEqual(result.score, 12.5)
            self.assertTrue((workspace / "metric" / "pipeline_spec.json").exists())
            self.assertTrue((workspace / "metric" / "score_summary.json").exists())

    def test_metric_gpu_same_reuses_uuid_allocation_without_polling(self):
        allocator = MetricGpuAllocator(
            {
                "enabled": True,
                "devices": "same",
                "gpus_per_metric": 1,
                "wait_timeout_sec": 900,
            }
        )
        visible = "GPU-226dd387-6c9c-803b-7fd4-5f6e55ea0cb3"

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": visible}, clear=False), patch.object(
            allocator,
            "_query_gpu_snapshots",
            side_effect=AssertionError("same allocation must not query nvidia-smi"),
        ), patch("playground.sure_master.core.utils.metric.time.sleep") as sleep:
            with allocator.allocate(attempt=1) as allocation:
                self.assertEqual(allocation.mode, "same")
                self.assertEqual(allocation.devices, [visible])
                self.assertEqual(allocation.cuda_visible_devices, visible)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], visible)

        sleep.assert_not_called()

    def test_metric_gpu_idle_numeric_snapshots_do_not_match_uuid_visibility(self):
        allocator = MetricGpuAllocator(
            {
                "enabled": True,
                "devices": "idle",
                "gpus_per_metric": 1,
                "min_free_mib": 9500,
                "max_utilization": 20,
            }
        )
        snapshots = {
            "0": MetricGpuSnapshot(
                index="0",
                total_mib=24576,
                used_mib=512,
                free_mib=24064,
                utilization=0,
            )
        }

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-226dd387"}, clear=False):
            self.assertEqual(allocator._candidate_groups(snapshots, set()), [])

    def test_remote_f5_configs_reuse_child_metric_gpu(self):
        root = Path(__file__).resolve().parents[4]
        for relative_path in (
            "configs/sure_master/archive/staged_axes/gpt-5-f5tts-staged-axes-mixed.yaml",
            "configs/sure_master/gpt-5-f5tts-smoke.yaml",
        ):
            config = yaml.safe_load((root / relative_path).read_text(encoding="utf-8"))
            metric_gpu = config["sure"]["metric_gpu"]
            self.assertTrue(metric_gpu["enabled"])
            self.assertEqual(metric_gpu["devices"], "same")
            self.assertFalse(metric_gpu["oom_retry"])

    def test_metric_gpu_sets_cuda_visible_devices_and_restores_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved_sure_modules = {
                name: module
                for name, module in sys.modules.items()
                if name == "sure_eval" or name.startswith("sure_eval.")
            }
            for name in list(saved_sure_modules):
                sys.modules.pop(name, None)

            root = Path(tmp) / "sure"
            src = root / "src"
            pkg = src / "sure_eval" / "evaluation"
            pkg.mkdir(parents=True)
            (src / "sure_eval" / "__init__.py").write_text("", encoding="utf-8")
            (pkg / "__init__.py").write_text("", encoding="utf-8")
            (pkg / "cli_adapters.py").write_text(
                """
import os

def build_pipeline_spec(task, language=None, metric=None, output_path=None):
    return {"task": task, "language": language, "metric": metric, "pipeline_id": "fake"}

def run_pipeline_spec(pipeline, output_dir, **kwargs):
    return {
        "status": "ok",
        "metric": pipeline["metric"],
        "score": 7.0,
        "pipeline_id": "fake",
        "seen_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
""",
                encoding="utf-8",
            )
            workspace = Path(tmp) / "workspace"
            (workspace / "input").mkdir(parents=True)
            (workspace / "artifacts").mkdir()
            (workspace / "input" / "ref.txt").write_text("utt1\tHELLO\n", encoding="utf-8")
            (workspace / "artifacts" / "hyp.txt").write_text("utt1\tHELLO\n", encoding="utf-8")
            card = resolve_task_card(
                Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml",
                "asr_en_wer",
            )

            try:
                with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,2"}, clear=False):
                    result = SureMetricRunner(
                        root,
                        pythonpath=src,
                        device="cuda",
                        metric_gpu={
                            "enabled": True,
                            "devices": "2",
                            "lock_dir": str(Path(tmp) / "locks"),
                        },
                    ).run(
                        task_card=card,
                        workspace_path=workspace,
                        output_dir=workspace / "metric",
                        role_paths={},
                    )
                    self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "0,2")
            finally:
                for name in [
                    name
                    for name in list(sys.modules)
                    if name == "sure_eval" or name.startswith("sure_eval.")
                ]:
                    sys.modules.pop(name, None)
                sys.modules.update(saved_sure_modules)

            self.assertTrue(result.success)
            summary = json.loads((workspace / "metric" / "score_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["seen_cuda_visible_devices"], "2")
            self.assertTrue((workspace / "metric" / "metric_gpu_attempts.json").exists())

    def test_metric_gpu_retries_cuda_oom_on_next_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved_sure_modules = {
                name: module
                for name, module in sys.modules.items()
                if name == "sure_eval" or name.startswith("sure_eval.")
            }
            for name in list(saved_sure_modules):
                sys.modules.pop(name, None)

            root = Path(tmp) / "sure"
            src = root / "src"
            pkg = src / "sure_eval" / "evaluation"
            pkg.mkdir(parents=True)
            (src / "sure_eval" / "__init__.py").write_text("", encoding="utf-8")
            (pkg / "__init__.py").write_text("", encoding="utf-8")
            (pkg / "cli_adapters.py").write_text(
                """
import os

attempts = []

def build_pipeline_spec(task, language=None, metric=None, output_path=None):
    return {"task": task, "language": language, "metric": metric, "pipeline_id": "fake"}

def run_pipeline_spec(pipeline, output_dir, **kwargs):
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    attempts.append(cuda)
    if cuda == "0":
        raise RuntimeError("CUDA out of memory while loading scorer")
    return {
        "status": "ok",
        "metric": pipeline["metric"],
        "score": 6.0,
        "pipeline_id": "fake",
        "seen_cuda_visible_devices": cuda,
        "attempt_count": len(attempts),
    }
""",
                encoding="utf-8",
            )
            workspace = Path(tmp) / "workspace"
            (workspace / "input").mkdir(parents=True)
            (workspace / "artifacts").mkdir()
            (workspace / "input" / "ref.txt").write_text("utt1\tHELLO\n", encoding="utf-8")
            (workspace / "artifacts" / "hyp.txt").write_text("utt1\tHELLO\n", encoding="utf-8")
            card = resolve_task_card(
                Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml",
                "asr_en_wer",
            )

            try:
                with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=False):
                    result = SureMetricRunner(
                        root,
                        pythonpath=src,
                        device="cuda",
                        metric_gpu={
                            "enabled": True,
                            "devices": "0,1",
                            "oom_retry": True,
                            "max_retries": 1,
                            "lock_dir": str(Path(tmp) / "locks"),
                        },
                    ).run(
                        task_card=card,
                        workspace_path=workspace,
                        output_dir=workspace / "metric",
                        role_paths={},
                    )
                    self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "0,1")
            finally:
                for name in [
                    name
                    for name in list(sys.modules)
                    if name == "sure_eval" or name.startswith("sure_eval.")
                ]:
                    sys.modules.pop(name, None)
                sys.modules.update(saved_sure_modules)

            self.assertTrue(result.success)
            summary = json.loads((workspace / "metric" / "score_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["seen_cuda_visible_devices"], "1")
            self.assertEqual(summary["attempt_count"], 2)
            attempts = json.loads(
                (workspace / "metric" / "metric_gpu_attempts.json").read_text(encoding="utf-8")
            )
            starts = [item["metric_gpu"]["cuda_visible_devices"] for item in attempts if item["event"] == "attempt_start"]
            self.assertEqual(starts, ["0", "1"])
            self.assertTrue(any(item.get("is_cuda_oom") for item in attempts))

    def test_candidate_boundary_rejects_direct_sure_access(self):
        errors = validate_sure_candidate_boundary(
            """
import sure_eval.evaluation.cli_adapters
SURE = "/hpc_stor03/sjtu_home/chaolei.liu/sure"
""",
            "/hpc_stor03/sjtu_home/chaolei.liu/sure",
        )
        self.assertTrue(any("imports SURE module" in error for error in errors))
        self.assertTrue(any("read-only SURE root" in error for error in errors))

    def test_candidate_boundary_allows_workspace_model_code(self):
        errors = validate_sure_candidate_boundary(
            """
from pathlib import Path
import torch

class Model(torch.nn.Module):
    pass

Path("models").mkdir(exist_ok=True)
Path("artifacts").mkdir(exist_ok=True)
""",
            "/hpc_stor03/sjtu_home/chaolei.liu/sure",
        )
        self.assertEqual(errors, [])

    def test_run_sure_script_validation_rejects_comment_only_script(self):
        errors = validate_run_sure_script(
            """
# Corrected run_sure.py has been written to the current workspace.
# It writes artifacts/hyp.txt and candidate_changes.json.
""",
            {"ref": "input/ref.txt", "hyp": "artifacts/hyp.txt"},
        )
        self.assertTrue(any("no executable Python statements" in error for error in errors))

    def test_run_sure_script_validation_allows_top_level_artifact_writer(self):
        errors = validate_run_sure_script(
            """
from pathlib import Path
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/hyp.txt").write_text("utt1\\tHELLO\\n")
""",
            {"ref": "input/ref.txt", "hyp": "artifacts/hyp.txt"},
        )
        self.assertEqual(errors, [])

    def test_candidate_boundary_requires_base_model_reference(self):
        path = Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml"
        card = resolve_task_card(path, "asr_en_wer")
        profile = merge_base_model_profile(
            card.base_model,
            {
                "source_paths": {
                    "recipe": "/tmp/icefall/zipformer",
                    "data": "/tmp/icefall/data",
                    "root": "/tmp/icefall",
                }
            },
        )
        errors = validate_sure_candidate_boundary(
            """
from pathlib import Path
Path("artifacts/hyp.txt").write_text("utt1\\tHELLO\\n")
""",
            "/hpc_stor03/sjtu_home/chaolei.liu/sure",
            profile,
        )
        self.assertTrue(any("required base model profile is not referenced" in error for error in errors))

    def test_candidate_boundary_allows_joined_base_model_reference(self):
        path = Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml"
        card = resolve_task_card(path, "asr_en_wer")
        profile = merge_base_model_profile(
            card.base_model,
            {
                "source_paths": {
                    "recipe": "/tmp/icefall/zipformer",
                    "data": "/tmp/icefall/data",
                    "root": "/tmp/icefall",
                }
            },
        )
        errors = validate_sure_candidate_boundary(
            """
from pathlib import Path
WORK = Path.cwd()
RECIPE = WORK / "base_model" / "recipe"
DATA = WORK / "base_model" / "data"
ROOT = WORK / "base_model" / "root"
Path("artifacts/hyp.txt").write_text("utt1\\tHELLO\\n")
""",
            "/hpc_stor03/sjtu_home/chaolei.liu/sure",
            profile,
        )
        self.assertEqual(errors, [])

    def test_candidate_boundary_rejects_lang_dir_for_zipformer_train(self):
        path = Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml"
        card = resolve_task_card(path, "asr_en_wer")
        profile = merge_base_model_profile(
            card.base_model,
            {
                "source_paths": {
                    "recipe": "/tmp/icefall/zipformer",
                    "data": "/tmp/icefall/data",
                    "root": "/tmp/icefall",
                }
            },
        )
        errors = validate_sure_candidate_boundary(
            """
cmd = ["python", "base_model/recipe/train.py", "--lang-dir", "base_model/data/lang_bpe_500"]
""",
            "/hpc_stor03/sjtu_home/chaolei.liu/sure",
            profile,
        )
        self.assertTrue(any("passes --lang-dir to zipformer train.py" in error for error in errors))

    def test_candidate_boundary_allows_lang_dir_for_zipformer_decode(self):
        path = Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml"
        card = resolve_task_card(path, "asr_en_wer")
        profile = merge_base_model_profile(
            card.base_model,
            {
                "source_paths": {
                    "recipe": "/tmp/icefall/zipformer",
                    "data": "/tmp/icefall/data",
                    "root": "/tmp/icefall",
                }
            },
        )
        errors = validate_sure_candidate_boundary(
            """
cmd = ["python", "base_model/recipe/decode.py", "--lang-dir", "base_model/data/lang_bpe_500"]
""",
            "/hpc_stor03/sjtu_home/chaolei.liu/sure",
            profile,
        )
        self.assertEqual(errors, [])

    def test_candidate_boundary_rejects_train_only_args_for_zipformer_decode(self):
        path = Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml"
        card = resolve_task_card(path, "asr_en_wer")
        profile = merge_base_model_profile(
            card.base_model,
            {
                "source_paths": {
                    "recipe": "/tmp/icefall/zipformer",
                    "data": "/tmp/icefall/data",
                    "root": "/tmp/icefall",
                }
            },
        )
        errors = validate_sure_candidate_boundary(
            """
cmd = [
    "python",
    "base_model/recipe/decode.py",
    "--ctc-loss-scale",
    "0.1",
    "--time-mask-ratio",
    "2.5",
]
""",
            "/hpc_stor03/sjtu_home/chaolei.liu/sure",
            profile,
            canonical_task="asr",
        )
        self.assertTrue(
            any("training-only argument(s) to decode.py" in error for error in errors)
        )

    def test_candidate_boundary_rejects_failed_duration_log_recovery(self):
        errors = validate_sure_candidate_boundary(
            r"""
import re

def parse_any_integer_from_log(log_path):
    text = log_path.read_text()
    return int(re.search(r"(?:resolved|max[-_ ]duration|selected|final)[^\d]{0,80}(\d{2,5})", text).group(1))

def resolve_duration():
    try:
        cmd = ["python", "runtime_env.py", "resolve-max-duration"]
        raise RuntimeError("duration_autotune failed")
    except Exception:
        return parse_any_integer_from_log(Path("working/duration_autotune.log"))

cmd = ["python", "base_model/recipe/train.py", "--encoder-dim", "192"]
""",
            "/hpc_stor03/sjtu_home/chaolei.liu/sure",
            canonical_task="asr",
        )

        self.assertTrue(any("failed helper logs" in error for error in errors))

    def test_candidate_boundary_rejects_direct_duration_helper_invocation(self):
        errors = validate_sure_candidate_boundary(
            """
import os
import subprocess

helper = os.environ["SURE_RUNTIME_ENV_HELPER"]
subprocess.run([helper, "resolve-max-duration"])
cmd = ["python", "base_model/recipe/train.py", "--encoder-dim", "192"]
""",
            "/hpc_stor03/sjtu_home/chaolei.liu/sure",
            canonical_task="asr",
        )

        self.assertTrue(any("invokes SURE_RUNTIME_ENV_HELPER directly" in error for error in errors))

    def test_candidate_boundary_allows_official_zipformer_baseline_source(self):
        profile = BaseModelProfile(model_id="zipformer", framework="icefall")
        baseline_source = Path(zipformer_baseline.__file__).read_text(encoding="utf-8")
        errors = validate_sure_candidate_boundary(
            baseline_source,
            "/hpc_stor03/sjtu_home/chaolei.liu/sure",
            profile,
            canonical_task="asr",
            require_asr_wrapper=True,
        )
        self.assertEqual(errors, [])

    def test_candidate_boundary_requires_asr_zipformer_wrapper_when_configured(self):
        direct_errors = validate_sure_candidate_boundary(
            """
import subprocess
cmd = ["python", "base_model/recipe/train.py", "--encoder-dim", "192"]
subprocess.run(cmd)
""",
            "/hpc_stor03/sjtu_home/chaolei.liu/sure",
            canonical_task="asr",
            require_asr_wrapper=True,
        )
        self.assertTrue(any("SURE_ASR_ZIPFORMER_WRAPPER" in error for error in direct_errors))

        wrapper_errors = validate_sure_candidate_boundary(
            """
import os
import subprocess
cmd = [os.environ["SURE_ASR_ZIPFORMER_WRAPPER"], "--candidate-type", "arch"]
subprocess.run(cmd)
""",
            "/hpc_stor03/sjtu_home/chaolei.liu/sure",
            canonical_task="asr",
            require_asr_wrapper=True,
        )
        self.assertEqual(wrapper_errors, [])

    def test_candidate_boundary_rejects_zipformer_wrapper_plus_duration_helper(self):
        errors = validate_sure_candidate_boundary(
            """
import os
import subprocess
helper_cmd = [os.environ["SURE_RUNTIME_ENV_HELPER"], "resolve-max-duration"]
subprocess.run(helper_cmd)
cmd = [os.environ["SURE_ASR_ZIPFORMER_WRAPPER"], "--candidate-type", "arch"]
subprocess.run(cmd)
""",
            "/hpc_stor03/sjtu_home/chaolei.liu/sure",
            canonical_task="asr",
            require_asr_wrapper=True,
        )
        self.assertTrue(any("must not call the duration helper directly" in error for error in errors))

    def test_candidate_boundary_rejects_direct_base_model_source_path(self):
        path = Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml"
        card = resolve_task_card(path, "asr_en_wer")
        profile = merge_base_model_profile(
            card.base_model,
            {
                "source_paths": {
                    "recipe": "/tmp/icefall/zipformer",
                    "data": "/tmp/icefall/data",
                    "root": "/tmp/icefall",
                }
            },
        )
        errors = validate_sure_candidate_boundary(
            """
from pathlib import Path
recipe = Path("base_model/recipe")
external = "/tmp/icefall/zipformer"
""",
            "/hpc_stor03/sjtu_home/chaolei.liu/sure",
            profile,
        )
        self.assertTrue(any("base model source path directly" in error for error in errors))

    def test_generic_classification_pipeline_compatibility_with_sure(self):
        sure_root = Path("/hpc_stor03/sjtu_home/chaolei.liu/sure")
        sure_src = sure_root / "src"
        if not sure_src.exists():
            self.skipTest(f"SURE source tree is not available: {sure_src}")

        runner = SureMetricRunner(sure_root, pythonpath=sure_src, device="cpu")
        old_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            runner._prepare_import_path()
            from sure_eval.evaluation.cli_adapters import build_pipeline_spec  # type: ignore
        finally:
            sys.dont_write_bytecode = old_dont_write_bytecode

        card = resolve_task_card(
            Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml",
            "classification_accuracy",
        )
        spec = runner._build_pipeline_spec(build_pipeline_spec, card)
        self.assertEqual(spec["task"], "classification")
        self.assertEqual(spec["task_alias"], "classification")
        self.assertEqual(spec["required_roles"], ["hyp", "ref"])
        self.assertEqual(
            spec["pipeline_id"],
            "classification.any.accuracy.classify_v1",
        )

    def test_all_repo_task_cards_match_sure_adapter_required_roles(self):
        sure_root = Path("/hpc_stor03/sjtu_home/chaolei.liu/sure")
        sure_src = sure_root / "src"
        if not sure_src.exists():
            self.skipTest(f"SURE source tree is not available: {sure_src}")

        runner = SureMetricRunner(sure_root, pythonpath=sure_src, device="cpu")
        old_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            runner._prepare_import_path()
            from sure_eval.evaluation.cli_adapters import build_pipeline_spec  # type: ignore
        finally:
            sys.dont_write_bytecode = old_dont_write_bytecode

        cards = load_task_cards(Path(__file__).resolve().parents[2] / "task_cards" / "sure_tasks.yaml")
        self.assertEqual(len(cards), 15)
        for task_id, card in cards.items():
            with self.subTest(task_id=task_id):
                spec = runner._build_pipeline_spec(build_pipeline_spec, card)
                self.assertEqual(set(card.required_roles), set(spec["required_roles"]))
                self.assertTrue(spec["pipeline_id"])
                self.assertTrue(spec["metric"])


if __name__ == "__main__":
    unittest.main()
