from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from research_idea_lib.algorithm.workflow import OP_BACKGROUND, OP_QUERY, OP_RANKING, WorkflowRequest, run_workflow
from test_workflow import FakeProvider, FakeRetrieval, FakeSearchFactory, FakeFusion


class RankingProjectionTests(unittest.TestCase):
    def test_scalar_answer_wrappers_keep_existing_type_validation(self):
        for invalid in [False,True]:
            class ScalarWrapper(FakeProvider):
                def execute(self, operation):
                    output=super().execute(operation)
                    key={OP_BACKGROUND:'background',OP_QUERY:'query'}.get(operation.name)
                    if key:return replace(output,value={'answer':[] if invalid else output.value[key]})
                    return output
            result=run_workflow(WorkflowRequest('fixture','Research',{}),search_factory=FakeSearchFactory(),
                fusion_runner=FakeFusion(),provider=ScalarWrapper(),retrieval=FakeRetrieval())
            self.assertEqual(result.succeeded,not invalid)
            if not invalid:self.assertEqual(len(result.artifact['run']['response_projections']),2)

    def run_case(self, variant):
        class WrappedRanking(FakeProvider):
            def execute(self, operation):
                output=super().execute(operation)
                if operation.name==OP_RANKING:
                    ids=deepcopy(output.value['evidence_ids'])
                    value={'answer':ids}
                    if variant=='duplicate':value['answer']=[ids[0],ids[0]]
                    if variant=='unknown':value['answer'][0]='unknown-evidence'
                    if variant=='invalid_explicit':value['evidence_ids']=None
                    return replace(output,value=value)
                return output
        provider=WrappedRanking()
        result=run_workflow(WorkflowRequest('fixture','Research',{}),search_factory=FakeSearchFactory(),
            fusion_runner=FakeFusion(),provider=provider,retrieval=FakeRetrieval())
        return result,provider

    def test_answer_wrapper_preserves_exact_ranking(self):
        result,provider=self.run_case('valid')
        self.assertTrue(result.succeeded,result.error)
        payload=next(value for operation,value in provider.calls if operation==OP_RANKING)
        self.assertEqual(result.artifact['retrieval']['evidence_ids'],
                         list(reversed([item['evidence_id'] for item in payload['evidence']])))
        self.assertEqual(result.artifact['retrieval']['ranking_projection']['source_key'],'answer')
        self.assertEqual(payload['output_contract']['required_key'],'evidence_ids')

    def test_invalid_rankings_still_rejected(self):
        for variant in ['duplicate','unknown','invalid_explicit']:
            with self.subTest(variant=variant):
                result,_=self.run_case(variant)
                self.assertFalse(result.succeeded)


if __name__=='__main__':unittest.main()
