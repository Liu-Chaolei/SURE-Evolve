from __future__ import annotations

import unittest

from playground.sure_master.core.contracts import (
    CandidateResult,
    IdeaBatch,
    IdeaItem,
    IdeaRequest,
    IdeaSpec,
    MetricSpec,
    RoundResult,
    validate_idea_batch,
)
from playground.sure_master.core.providers import FakeXlabIdeaProvider


class FakeXlabProviderTests(unittest.TestCase):
    def _request(self, mode: str = "ordinary", axis: str | None = None) -> IdeaRequest:
        return IdeaRequest(
            request_id=f"request-{mode}-{axis or 'none'}",
            sure_run_id="sure-test",
            task_id="asr-test",
            task_description="test speech task",
            search_mode=mode,
            axis=axis,
            round_index=1,
            requested_idea_count=4,
            metric=MetricSpec(name="wer", direction="lower"),
            input_digest="sha256:request",
        )

    def test_request_requires_exactly_four_ideas(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly four"):
            IdeaRequest(
                request_id="request-invalid-count",
                sure_run_id="sure-test",
                task_id="asr-test",
                task_description="test speech task",
                search_mode="ordinary",
                round_index=1,
                requested_idea_count=3,
                metric=MetricSpec(name="wer", direction="lower"),
                input_digest="sha256:request",
            )

    def test_ordinary_generation_is_deterministic_and_validated(self) -> None:
        provider = FakeXlabIdeaProvider()
        first = provider.generate(self._request())
        second = provider.generate(self._request())
        self.assertEqual([idea.idea_id for idea in first.ideas], [idea.idea_id for idea in second.ideas])
        self.assertEqual(first.batch_digest, second.batch_digest)
        self.assertTrue(all(idea.axis is None for idea in first.ideas))

    def test_staged_generation_preserves_axis_and_candidate_type(self) -> None:
        provider = FakeXlabIdeaProvider()
        batch = provider.generate(self._request("staged_axes", "arch"))
        self.assertEqual(len(batch.ideas), 4)
        self.assertTrue(all(idea.axis == "arch" for idea in batch.ideas))
        self.assertTrue(all(idea.candidate_type == "arch" for idea in batch.ideas))

    def test_batch_validation_rejects_count_digest_and_axis_mismatches(self) -> None:
        request = self._request("staged_axes", "arch")
        idea = IdeaItem(
            idea_id="idea-1",
            artifact_id="artifact-1",
            artifact_digest="sha256:artifact",
            title="idea-1",
            axis="arch",
            candidate_type="arch",
            hypothesis="hypothesis",
            mechanism="mechanism",
            spec=IdeaSpec(implementation_instructions="implement"),
        )
        batch = IdeaBatch(
            status="success",
            request_digest=request.input_digest,
            xlab_run_id="xlab",
            workspace="fake",
            ideas=[idea, IdeaItem(**{**idea.__dict__, "idea_id": "idea-2"})],
            batch_digest="sha256:batch",
        )
        with self.assertRaisesRegex(ValueError, "count"):
            validate_idea_batch(batch, request)

        mismatched = IdeaBatch(
            status="success",
            request_digest="sha256:other",
            xlab_run_id="xlab",
            workspace="fake",
            ideas=[idea, IdeaItem(**{**idea.__dict__, "idea_id": "idea-2"})],
            batch_digest="sha256:batch",
        )
        with self.assertRaisesRegex(ValueError, "request digest"):
            validate_idea_batch(mismatched, request)

    def test_batch_validation_rejects_ordinary_axis_and_wrong_staged_type(self) -> None:
        ordinary = self._request()
        idea = IdeaItem(
            idea_id="idea-1",
            artifact_id="artifact-1",
            artifact_digest="sha256:artifact",
            title="idea-1",
            axis="arch",
            candidate_type="arch",
            hypothesis="hypothesis",
            mechanism="mechanism",
            spec=IdeaSpec(implementation_instructions="implement"),
        )
        ordinary_batch = IdeaBatch(
            status="success",
            request_digest=ordinary.input_digest,
            xlab_run_id="xlab",
            workspace="fake",
            ideas=[
                idea,
                IdeaItem(**{**idea.__dict__, "idea_id": "idea-2"}),
                IdeaItem(**{**idea.__dict__, "idea_id": "idea-3"}),
                IdeaItem(**{**idea.__dict__, "idea_id": "idea-4"}),
            ],
            batch_digest="sha256:batch",
        )
        with self.assertRaisesRegex(ValueError, "ordinary"):
            validate_idea_batch(ordinary_batch, ordinary)

        staged = self._request("staged_axes", "train")
        staged_batch = IdeaBatch(
            status="success",
            request_digest=staged.input_digest,
            xlab_run_id="xlab",
            workspace="fake",
            ideas=[
                IdeaItem(
                    idea_id=f"idea-{index}",
                    artifact_id=f"artifact-{index}",
                    artifact_digest=f"sha256:artifact-{index}",
                    title=f"idea-{index}",
                    axis="train",
                    candidate_type="arch",
                    hypothesis="hypothesis",
                    mechanism="mechanism",
                    spec=IdeaSpec(implementation_instructions="implement"),
                )
                for index in range(1, 5)
            ],
            batch_digest="sha256:batch",
        )
        with self.assertRaisesRegex(ValueError, "candidate type"):
            validate_idea_batch(staged_batch, staged)

    def test_round_result_keeps_candidate_lineage_fields(self) -> None:
        provider = FakeXlabIdeaProvider()
        result = RoundResult(
            sure_run_id="sure-test",
            search_mode="ordinary",
            round_index=1,
            baseline_digest="sha256:baseline",
            idea_batch_digest="sha256:batch",
            result_digest="sha256:result",
            candidates=[
                CandidateResult(
                    idea_id="idea-1",
                    idea_artifact_id="artifact-1",
                    final_status="success",
                    code_digest="sha256:code",
                    workspace_ref="exp_1",
                    candidate_type="inference",
                )
            ],
            parent_artifacts=["sha256:summary"],
        )
        summary = provider.summarize(result)
        self.assertEqual(summary.attempted_idea_fingerprints, ["idea-1"])
        self.assertEqual(result.candidates[0].idea_artifact_id, "artifact-1")
        self.assertEqual(result.candidates[0].workspace_ref, "exp_1")

        provider = FakeXlabIdeaProvider()
        result = RoundResult(
            sure_run_id="sure-test",
            search_mode="ordinary",
            round_index=1,
            baseline_digest="sha256:baseline",
            idea_batch_digest="sha256:batch",
            result_digest="sha256:result",
            candidates=[
                CandidateResult(
                    idea_id="idea-1",
                    idea_artifact_id="artifact-1",
                    final_status="failed",
                    failure_category="candidate_failure",
                ),
                CandidateResult(
                    idea_id="idea-2",
                    idea_artifact_id="artifact-2",
                    final_status="failed",
                    failure_category="system_failure",
                ),
            ],
        )
        summary = provider.summarize(result)
        self.assertEqual(summary.failure_counts, {"candidate_failure": 1, "system_failure": 1})
        self.assertEqual(provider.reconcile("summarize:sure-test:1")["status"], "published")


if __name__ == "__main__":
    unittest.main()
