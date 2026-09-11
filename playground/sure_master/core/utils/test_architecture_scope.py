"""Architecture-only ordinary research: guidance, batch contracts and execution boundaries."""

from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from playground.sure_master.core.contracts import (
    IdeaRequest,
    MetricSpec,
    validate_idea_batch,
)
from playground.sure_master.core.providers import FakeXlabIdeaProvider
from playground.sure_master.core.search_scope import (
    execution_contract,
    validate_candidate_parameters,
)
from playground.sure_master.core.utils import test_multitask as fixtures
from playground.sure_master.tasks import get_adapter
from playground.sure_master.tools.run_icefall_zipformer_candidate import (
    validate_candidate_args,
)
import yaml


class ArchitectureScopeTests(unittest.TestCase):
    def request(self):
        return IdeaRequest(
            request_id="request",
            sure_run_id="run",
            task_id="tts_zh_cer",
            task_description="Improve CER",
            search_mode="ordinary",
            round_index=1,
            requested_idea_count=4,
            metric=MetricSpec("CER", "lower"),
            input_digest="digest",
            execution_contract=execution_contract(get_adapter("tts").context(), {}),
        )

    def test_guidance_only_exposes_structure_without_reintroducing_axes(self):
        for task in ("asr", "tts", "sd"):
            with self.subTest(task=task):
                contract = execution_contract(
                    get_adapter(task).context(),
                    {
                        "execution_contract": {
                            "candidate_types": ["fine_tune", "inference"]
                        },
                        "task": {
                            "training": {"max_steps": 1000},
                            "inference": {"speed": 1.0},
                        },
                    },
                )
                self.assertEqual(contract["candidate_types"], ["arch"])
                self.assertEqual(contract["allowed_change_domains"], ["arch"])
                self.assertNotIn("inference", contract["candidate_parameters"])
                self.assertNotIn("training", contract["candidate_parameters"])
                self.assertEqual(contract["fixed_training"]["max_steps"], 1000)
                self.assertIn("exactly four", contract["research_guidance"])
        request = self.request()
        self.assertIsNone(request.axis)
        self.assertEqual(request.search_mode, "ordinary")

    def test_batch_rejects_training_inference_and_mixed_changes(self):
        request = self.request()
        batch = FakeXlabIdeaProvider().generate(request)
        validate_idea_batch(batch, request)
        first = batch.ideas[0]
        for domains, kind in (
            (["train"], "fine_tune"),
            (["inference"], "inference"),
            (["arch", "train"], "arch"),
            (["arch", "inference"], "arch"),
        ):
            changes = [
                {"domain": domain, "target": "module", "description": "changed"}
                for domain in domains
            ]
            spec = replace(
                first.spec,
                change_domains=domains,
                change_set=changes,
                requires_training=domains != ["inference"],
            )
            invalid = replace(
                batch,
                ideas=[
                    replace(first, candidate_type=kind, spec=spec),
                    *batch.ideas[1:],
                ],
            )
            with (
                self.subTest(domains=domains),
                self.assertRaisesRegex(ValueError, "Architecture-only"),
            ):
                validate_idea_batch(invalid, request)

    def test_worker_keeps_training_and_inference_parameters_fixed(self):
        for adapter in ("tts.f5tts", "sd.diarizen"):
            env = {
                "SURE_SEARCH_SCOPE": "architecture_only",
                "SURE_TASK_ADAPTER": adapter,
                "SURE_CANDIDATE_PHASE": "search",
            }
            validate_candidate_parameters("arch", {"architecture": {"depth": 20}}, env)
            for action, parameters in [
                ("arch", {"architecture": {"checkpoint_activations": True}}),
                ("fine_tune", {}),
                ("infer", {}),
                (
                    "arch",
                    {"architecture": {"depth": 20}, "training": {"learning_rate": 0.1}},
                ),
                ("arch", {"architecture": {"depth": 20}, "inference": {"speed": 0.8}}),
            ]:
                with (
                    self.subTest(adapter=adapter, action=action),
                    self.assertRaisesRegex(ValueError, "Architecture-only"),
                ):
                    validate_candidate_parameters(action, parameters, env)
            validate_candidate_parameters("baseline", {}, env)
            validate_candidate_parameters("infer", {}, env, frozen=True)
            validate_candidate_parameters(
                "fine_tune", {}, {**env, "SURE_CANDIDATE_PHASE": "full_training"}
            )

    def test_asr_legacy_wrapper_rejects_nonstructural_arguments(self):
        env = {
            "SURE_SEARCH_SCOPE": "architecture_only",
            "SURE_CANDIDATE_PHASE": "search",
        }
        with patch.dict(os.environ, env):
            validate_candidate_args(
                candidate_type="arch",
                action="train_decode",
                train_extra_args=["--encoder-dim", "128"],
                decode_extra_args=[],
            )
            for extra in (["--lr-factor", ".1"], ["--enable-spec-aug", "false"]):
                with self.assertRaisesRegex(ValueError, "Architecture-only"):
                    validate_candidate_args(
                        candidate_type="arch",
                        action="train_decode",
                        train_extra_args=["--encoder-dim", "128", *extra],
                        decode_extra_args=[],
                    )
            with self.assertRaisesRegex(ValueError, "Architecture-only"):
                validate_candidate_args(
                    candidate_type="arch",
                    action="train_decode",
                    train_extra_args=["--encoder-dim", "128"],
                    decode_extra_args=["--beam", "8"],
                )

    def test_two_rounds_execute_four_architectures_for_each_task(self):
        for task in ("asr", "tts", "sd"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = fixtures.MultiTaskControllerTests().config(root, task)
                config = yaml.safe_load(path.read_text())
                config["sure"]["search_scope"] = "architecture_only"
                path.write_text(yaml.safe_dump(config))
                provider = FakeXlabIdeaProvider()
                controller = fixtures.Controller(
                    config_path=path, xlab_provider=provider
                )
                controller.set_run_dir(root / "run", task_id=task)
                with patch(
                    "playground.sure_master.core.playground.KnowledgePromotionExp.run",
                    return_value="findings",
                ):
                    result = controller.run("Only change model structure")
                self.assertEqual(result["status"], "completed", result)
                self.assertEqual(
                    [kind for kind, _, _ in controller.executed[1:]], ["arch"] * 8
                )
                self.assertTrue(
                    all(r.axis is None for r in provider.generated_requests)
                )
                self.assertEqual(
                    [len(r.candidates) for r in provider.summarized_results], [4, 4]
                )
                self.assertTrue(controller.replays)


if __name__ == "__main__":
    unittest.main()
