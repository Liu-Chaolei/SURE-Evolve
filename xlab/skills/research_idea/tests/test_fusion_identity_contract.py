from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from research_idea_lib.algorithm.fusion import FusionRequest, REFEREE_METRICS, fuse_five_modes
from research_idea_lib.algorithm.provider_adapter import AdapterOutputError, ProviderAdapter, _format_fusion_prompt
from test_fusion_source_contract import QueueProvider
from test_provider_adapter import fused_output, mode_inputs


class FusionIdentityContractTests(unittest.TestCase):
    def test_formal_mature_payload_is_shown_as_exact_identity(self):
        context={'mature_idea': True, 'mature_idea_payload': {'tags': ['speech', 'fine-tuning'], 'root_domains': []}}
        prompt=_format_fusion_prompt({'mode_inputs':mode_inputs(),'context':context})
        self.assertIn('Frozen fused-idea identity',prompt)
        self.assertIn('"tags": ["speech", "fine-tuning"], "root_domains": []',prompt)
        self.assertIn('not a summary of your new',prompt)

    def test_changed_tags_are_retried_before_referee_instead_of_failing_workflow(self):
        valid=fused_output()
        invalid=deepcopy(valid)
        invalid['idea']['tags']=['scheduler', 'cosine-decay']
        provider=QueueProvider([invalid,valid,{x:3 for x in REFEREE_METRICS}])
        adapter=ProviderAdapter(provider,model='glm-5.3-flash')
        context={'mature_idea':True,'mature_idea_payload':deepcopy(valid['idea'])}
        result=fuse_five_modes(FusionRequest(mode_inputs(),context=context,max_repair_steps=0),
                              generator=adapter,evaluator=adapter)
        self.assertEqual(result.metadata['fusion_draft_attempts'],2)
        self.assertEqual(result.idea['tags'],valid['idea']['tags'])
        retry=provider.requests[1].structured_input
        self.assertEqual(retry['previous_draft'],invalid)
        self.assertEqual(retry['validation_issues'][0]['required_identity'],
                         {'tags':['agents'],'root_domains':['computer science']})

    def test_explicit_changes_and_reordering_are_not_silently_overwritten(self):
        for key,values in [('tags',['second','first']),('root_domains',['second','first'])]:
            with self.subTest(key=key):
                original=fused_output()
                original['idea'][key]=['first','second']
                wrong=deepcopy(original)
                wrong['idea'][key]=values
                provider=QueueProvider([wrong])
                adapter=ProviderAdapter(provider,model='glm-5.3-flash')
                with self.assertRaisesRegex(AdapterOutputError,'explicitly changed frozen identity'):
                    adapter.generate({'mode_inputs':mode_inputs(),'context':{'mature_idea_payload':original['idea']}})
                self.assertEqual(provider.outputs,[])
                self.assertEqual(wrong['idea'][key],values)

    def test_empty_identity_arrays_are_preserved(self):
        value=fused_output()
        value['idea'].update(tags=[],root_domains=[])
        provider=QueueProvider([value])
        adapter=ProviderAdapter(provider,model='glm-5.3-flash')
        result=adapter.generate({'mode_inputs':mode_inputs(),'context':{'mature_idea_payload':value['idea']}})
        self.assertEqual(result['idea']['tags'],[])
        self.assertEqual(result['idea']['root_domains'],[])


if __name__=='__main__':
    unittest.main()
