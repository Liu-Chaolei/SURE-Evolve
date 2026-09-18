from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from research_idea_lib.algorithm.response_projection import unwrap_answer_object
from research_idea_lib.algorithm.workflow import WorkflowRequest, run_workflow, OP_BACKGROUND
from research_idea_lib.algorithm.provider_adapter import AdapterOutputError, ProviderAdapter
from research_idea_lib.algorithm.fusion import FusionRequest, REFEREE_METRICS, fuse_five_modes
from test_workflow import FakeProvider, FakeRetrieval, FakeSearchFactory, FakeFusion
from test_fusion_source_contract import QueueProvider
from test_provider_adapter import fused_output, mode_inputs


class ResponseProjectionTests(unittest.TestCase):
    def test_nested_and_encoded_objects_keep_all_fields_and_raw_response(self):
        content={'background':'Flow matching speech synthesis.',
                 'key_questions':['How can adaptation improve CER?'],
                 'canonical_methods':['DiT','CFM']}
        raw={'answer':json.dumps({'answer':content})}
        original=deepcopy(raw)
        output,depth=unwrap_answer_object(raw)
        self.assertEqual(depth,2)
        self.assertEqual(output,content)
        output['key_questions'].append('Changed only in projected copy')
        self.assertEqual(raw,original)

    def test_explicit_invalid_fields_are_never_replaced_by_an_answer(self):
        raw={'background':None,'answer':{'background':'Must not replace null'}}
        self.assertEqual(unwrap_answer_object(raw),(raw,0))
        for value in ['plain text', [], None, True, '123']:
            raw={'answer':value}
            self.assertEqual(unwrap_answer_object(raw),(raw,0))

    def test_depth_is_bounded_and_missing_content_is_not_created(self):
        content={'background':'Text'}
        for _ in range(4):content={'answer':content}
        result,depth=unwrap_answer_object(content)
        self.assertEqual(depth,3)
        self.assertEqual(result,{'answer':{'background':'Text'}})
        self.assertEqual(unwrap_answer_object({'answer':{}}),({},1))

    def test_whole_workflow_accepts_object_wrappers_and_records_projection(self):
        class WrappedProvider(FakeProvider):
            def execute(self,operation):
                output=super().execute(operation)
                return replace(output,value={'answer':deepcopy(output.value)})
        result=run_workflow(WorkflowRequest('fixture','Research',{}),search_factory=FakeSearchFactory(),
                            fusion_runner=FakeFusion(),provider=WrappedProvider(),retrieval=FakeRetrieval())
        self.assertTrue(result.succeeded,result.error)
        self.assertTrue(result.artifact['run']['response_projections'])
        self.assertTrue(all(p['wrapper_depth']==1 for p in result.artifact['run']['response_projections']))

    def test_background_wrapper_with_explicit_null_still_fails(self):
        class InvalidProvider(FakeProvider):
            def execute(self,operation):
                output=super().execute(operation)
                if operation.name==OP_BACKGROUND:
                    return replace(output,value={'background':None,'answer':deepcopy(output.value)})
                return output
        result=run_workflow(WorkflowRequest('fixture','Research',{}),search_factory=FakeSearchFactory(),
                            fusion_runner=FakeFusion(),provider=InvalidProvider(),retrieval=FakeRetrieval())
        self.assertFalse(result.succeeded)
        self.assertIn('background must be a non-empty string',result.error)

    def test_wrapped_fusion_and_referee_still_use_strict_source_and_metric_checks(self):
        provider=QueueProvider([{'answer':fused_output()},{'answer':{k:3 for k in REFEREE_METRICS}}])
        adapter=ProviderAdapter(provider,model='glm-5.3-flash')
        result=fuse_five_modes(FusionRequest(mode_inputs(),max_repair_steps=0),generator=adapter,evaluator=adapter)
        self.assertEqual(result.idea['components'],['core','validator'])
        invalid=fused_output()
        invalid['selected_components'][0]['evidence']=['fabricated evidence']
        provider=QueueProvider([{'answer':invalid}])
        with self.assertRaisesRegex(AdapterOutputError,'evidence does not belong'):
            ProviderAdapter(provider,model='glm-5.3-flash').generate({'mode_inputs':mode_inputs()})


if __name__=='__main__':
    unittest.main()
