from __future__ import annotations

import unittest

from playground.sure_master.core.contracts import (
    CandidateResult,
    IdeaRequest,
    MetricSpec,
    RoundResult,
    RungResult,
)
from playground.sure_master.core.providers import FakeXlabIdeaProvider
from playground.sure_master.core.utils.fingerprints import digest


class FakeXlabEndToEndGateTests(unittest.TestCase):
    """CPU-only proof of the staged XLab/SURE research-round contract."""

    def test_three_axes_four_rounds_four_ideas_and_all_rungs(self) -> None:
        provider = FakeXlabIdeaProvider()
        history: list[dict[str, object]] = []
        summaries = []
        axes = ("arch", "train", "inference")
        rung_names = ("short", "medium", "final")
        best_score = 1.0
        generated = 0

        for axis_index, axis in enumerate(axes):
            for round_index in range(1, 5):
                request_payload = {
                    "axis": axis,
                    "axis_index": axis_index,
                    "round_index": round_index,
                    "history": history,
                    "best_score": best_score,
                }
                request = IdeaRequest(
                    request_id=f"gate-{axis}-r{round_index}",
                    sure_run_id="gate-run",
                    task_id="gate-task",
                    task_description="fake speech-model self-evolution",
                    search_mode="staged_axes",
                    axis=axis,
                    axis_index=axis_index,
                    round_index=round_index,
                    requested_idea_count=4,
                    metric=MetricSpec(name="wer", direction="lower"),
                    input_digest=digest(request_payload),
                    current_best={"score": best_score},
                    prior_rounds=list(history),
                    history_digest=digest(history),
                )
                batch = provider.generate(request)
                self.assertEqual(len(batch.ideas), 4)
                generated += len(batch.ideas)

                candidates: list[CandidateResult] = []
                for index, idea in enumerate(batch.ideas):
                    score = best_score - (axis_index + round_index + index + 1) / 1000
                    rungs = [
                        RungResult(
                            name=rung,
                            success=True,
                            score=score + rung_index / 10000,
                            promoted=rung == "final",
                            checkpoint_artifact=(
                                f"checkpoints/{axis}/r{round_index}/{idea.idea_id}/{rung}.pt"
                            ),
                        )
                        for rung_index, rung in enumerate(rung_names)
                    ]
                    # Keep one candidate failure in the typed result without
                    # removing it from the round; SURE must publish all four.
                    failure = index == 3 and round_index == 2
                    if failure:
                        rungs[-1] = RungResult(
                            name="final",
                            success=False,
                            score=None,
                            reason_code="candidate_validation_failed",
                            failure_category="candidate_failure",
                        )
                    candidates.append(
                        CandidateResult(
                            idea_id=idea.idea_id,
                            idea_artifact_id=idea.artifact_id,
                            final_status="failed" if failure else "success",
                            code_digest=digest({"idea": idea.idea_id, "axis": axis}),
                            workspace_ref=f"workspaces/{idea.idea_id}",
                            candidate_type=idea.candidate_type,
                            rungs=rungs,
                            failure_category="candidate_failure" if failure else None,
                            reason_code="candidate_validation_failed" if failure else "success",
                        )
                    )
                    if not failure:
                        best_score = min(best_score, score)

                result = RoundResult(
                    sure_run_id="gate-run",
                    search_mode="staged_axes",
                    axis=axis,
                    round_index=round_index,
                    baseline_digest=digest({"axis": axis, "round": round_index - 1}),
                    idea_batch_digest=batch.batch_digest,
                    result_digest=digest(
                        {"axis": axis, "round": round_index, "candidates": [c.idea_id for c in candidates]}
                    ),
                    candidates=candidates,
                    ranking=[c.idea_id for c in candidates if c.final_status == "success"],
                    rung_names=list(rung_names),
                    current_best={"score": best_score},
                    parent_artifacts=[idea.artifact_id for idea in batch.ideas],
                )
                summary = provider.summarize(result)
                summaries.append(summary)
                history.append(
                    {
                        "axis": axis,
                        "round_index": round_index,
                        "batch_digest": batch.batch_digest,
                        "result_digest": result.result_digest,
                        "summary_digest": summary.summary_digest,
                    }
                )

        self.assertEqual(generated, 3 * 4 * 4)
        self.assertEqual(len(provider.generated_requests), 12)
        self.assertEqual(len(provider.summarized_results), 12)
        self.assertEqual(len(summaries), 12)
        self.assertEqual(
            [(request.axis, request.round_index) for request in provider.generated_requests],
            [(axis, round_index) for axis in axes for round_index in range(1, 5)],
        )
        self.assertTrue(all(request.requested_idea_count == 4 for request in provider.generated_requests))
        self.assertTrue(all(len(result.candidates) == 4 for result in provider.summarized_results))
        self.assertTrue(all(len(candidate.rungs) == 3 for result in provider.summarized_results for candidate in result.candidates))
        self.assertTrue(any(summary.failure_counts.get("candidate_failure") for summary in summaries))
        self.assertEqual(len(provider.generated_requests[-1].prior_rounds), 11)
        self.assertEqual(len({item["batch_digest"] for item in history}), 12)


if __name__ == "__main__":
    unittest.main()
