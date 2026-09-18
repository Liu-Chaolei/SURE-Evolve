from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from research_idea_lib.algorithm.contracts import OperatorAttemptRecord, RolloutBlockerRecord
from research_idea_lib.algorithm.operator_grounding import MECHANISM_COMMIT_OPERATOR
from research_idea_lib.algorithm.workflow import _validate_rollout_records, WorkflowContractError


class RolloutErrorTests(unittest.TestCase):
    def test_empty_retrieval_and_generation_error_remain_a_failure_record(self):
        attempt=OperatorAttemptRecord(parent_node_id=0,operator=MECHANISM_COMMIT_OPERATOR,
            plan_digest='fixture-plan',outcome='error',grounding_explicit_empty=True)
        blocker=RolloutBlockerRecord(parent_node_id=0,operator=MECHANISM_COMMIT_OPERATOR,
            plan_digest='fixture-plan',stage='generation',error_type='AdapterOutputError',message='Invalid generated node')
        _validate_rollout_records(SimpleNamespace(operator_attempts=(attempt,),rollout_blockers=(blocker,)),'steady_engineer')
        self.assertEqual(attempt.outcome,'error')
        with self.assertRaisesRegex(WorkflowContractError,'corresponding rollout blocker'):
            _validate_rollout_records(SimpleNamespace(operator_attempts=(attempt,),rollout_blockers=()),'steady_engineer')


if __name__=='__main__':unittest.main()
