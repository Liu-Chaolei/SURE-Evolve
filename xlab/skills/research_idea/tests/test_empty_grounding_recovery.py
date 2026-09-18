"""A failed rollout after empty retrieval remains an error, not a failed mode."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from research_idea_lib.algorithm.contracts import OperatorAttemptRecord, RolloutBlockerRecord
from research_idea_lib.algorithm.workflow import (
    BudgetUsage, ModeSearchOutput, WorkflowContractError, _validate_rollout_records,
)


class EmptyGroundingRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.attempt = OperatorAttemptRecord(0, 'mechanism-commit-innovation', 'plan', 'error', True)
        self.blocker = RolloutBlockerRecord(0, 'mechanism-commit-innovation', 'plan',
                                          'generation', 'ProviderError', 'generation failed')
        self.output = ModeSearchOutput('moonshot_inventor', {}, (), 'root', 'evidence', BudgetUsage(),
                                       operator_attempts=(self.attempt,), rollout_blockers=(self.blocker,))

    def test_error_with_empty_retrieval_and_correlated_blocker_is_preserved(self):
        _validate_rollout_records(self.output, self.output.mode)
        self.assertEqual(self.output.operator_attempts[0].outcome, 'error')
        self.assertTrue(self.output.operator_attempts[0].grounding_explicit_empty)

    def test_error_without_blocker_is_rejected(self):
        with self.assertRaisesRegex(WorkflowContractError, 'corresponding rollout blocker'):
            _validate_rollout_records(replace(self.output, rollout_blockers=()), self.output.mode)

    def test_other_operators_cannot_claim_explicit_empty(self):
        attempt = replace(self.attempt, operator='theory-transfer-injection')
        with self.assertRaisesRegex(WorkflowContractError, 'mechanism commit attempt'):
            _validate_rollout_records(replace(self.output, operator_attempts=(attempt,)), self.output.mode)


if __name__ == '__main__':
    unittest.main()
