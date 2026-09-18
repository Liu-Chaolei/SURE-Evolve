"""Regression for omitted mature-root metadata in real TTS analysis output."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from research_idea_lib.algorithm.workflow import (
    OP_ANALYSIS, OP_REPLAN, CANONICAL_MODES, WorkflowRequest, WorkflowContractError,
    _select_root, _validate_mature_root, stable_signature, run_workflow,
)
from research_idea_lib.algorithm.contracts import RefinementBoundary
from research_idea_lib.research_policy import ResearchPolicy
from research_idea_lib.algorithm.runtime_adapters import _idea_state, _workflow_prompt
from test_workflow import FakeProvider, FakeRetrieval, FakeSearchFactory, FakeFusion, evidence_fixture


class RootIdentityTests(unittest.TestCase):
    def setUp(self):
        data=json.loads((Path(__file__).parent/'fixtures/tts_missing_root_identity.json').read_text())
        self.parent=data['mature']
        self.raw=data['root_idea']
        self.request=WorkflowRequest('fixture','TTS',{},mature_idea=self.parent)

    def test_real_response_inherits_missing_metadata_without_mutation(self):
        original=deepcopy(self.raw)
        parent=deepcopy(self.parent)
        receipt={}
        root=_select_root(self.request,{'root_idea':self.raw},inheritance=receipt)
        _idea_state(root)
        self.assertEqual(set(receipt['fields']),{'tags','root_domains'})
        self.assertEqual(receipt['raw_root_signature'],stable_signature(original))
        self.assertEqual(receipt['normalized_root_signature'],stable_signature(root))
        self.assertEqual(receipt['stage'],'advanced_analysis.root_idea')
        self.assertTrue(all(root[key]==value for key,value in original.items()))
        root['tags'].append('local mutation')
        self.assertEqual(self.raw,original)
        self.assertEqual(self.parent,parent)

    def test_single_missing_and_present_metadata(self):
        for missing in [None,'tags','root_domains']:
            with self.subTest(missing=missing):
                raw={**deepcopy(self.raw),**{k:deepcopy(self.parent[k]) for k in ['tags','root_domains']}}
                if missing:del raw[missing]
                receipt={}
                root=_select_root(self.request,{'root_idea':raw},inheritance=receipt)
                self.assertEqual(root['tags'],self.parent['tags'])
                self.assertEqual(root['root_domains'],self.parent['root_domains'])
                self.assertEqual(receipt.get('fields',[]),[missing] if missing else [])

    def test_explicit_changes_and_invalid_values_rejected(self):
        for key in ['tags','root_domains']:
            for value in [None,'tts',{},[1],[],['different'],list(reversed(self.parent[key]))]:
                with self.subTest(key=key,value=value):
                    raw={**deepcopy(self.raw),key:value}
                    with self.assertRaisesRegex(WorkflowContractError,'advanced_analysis.root_idea.'+key):
                        _select_root(self.request,{'root_idea':raw})

    def test_empty_parent_lists_and_no_mature_parent(self):
        parent={**self.parent,'tags':[],'root_domains':[]}
        root=_select_root(replace(self.request,mature_idea=parent),{'root_idea':self.raw})
        self.assertEqual(root['tags'],[])
        self.assertEqual(root['root_domains'],[])
        root=_select_root(replace(self.request,mature_idea=None),{'root_idea':self.raw})
        self.assertEqual(root,self.raw)

    def test_replan_diagnostic_names_actual_stage(self):
        request=replace(self.request,experiment_feedback={'records':[]})
        receipt={}
        _select_root(request,{'replan':{'root_idea':self.raw}},inheritance=receipt)
        self.assertEqual(receipt['stage'],'re_analysis_replan.root_idea')
        with self.assertRaisesRegex(WorkflowContractError,'re_analysis_replan.root_idea.tags'):
            _select_root(request,{'replan':{'root_idea':{**self.raw,'tags':['changed']}}})

    def test_fusion_does_not_silently_inherit_or_accept_changed_identity(self):
        with self.assertRaisesRegex(WorkflowContractError,'idea_fusion.idea.tags: missing'):
            _validate_mature_root(self.parent,self.raw,None,context='idea_fusion.idea')

    def test_invalid_parent_identity_is_rejected(self):
        request=replace(self.request,mature_idea={**self.parent,'tags':None})
        with self.assertRaisesRegex(WorkflowContractError,'mature_idea.tags: invalid type'):
            _select_root(request,{'root_idea':self.raw})

    def test_inheritance_does_not_relax_component_boundary(self):
        raw=deepcopy(self.parent)
        del raw['tags']
        raw['components'][1]['description']='Disallowed component change'
        boundary=RefinementBoundary(allowed_component_ids=(self.parent['components'][0]['name'],),
                                    allowed_fields=('components',),allowed_edit_kinds=('REPLACE_COMPONENT',))
        with self.assertRaisesRegex(WorkflowContractError,'outside refinement_boundary'):
            _select_root(replace(self.request,refinement_boundary=boundary),{'root_idea':raw})

    def test_five_modes_and_feedback_paths_preserve_root_signatures(self):
        parent=self.parent
        class MissingMetadataProvider(FakeProvider):
            def execute(self, operation):
                output=super().execute(operation)
                value=deepcopy(output.value)
                if operation.name in {OP_ANALYSIS,OP_REPLAN}:
                    root=deepcopy(parent)
                    root.pop('tags');root.pop('root_domains')
                    if operation.name==OP_ANALYSIS:value['root_idea']=root
                    else:value['replan']['root_idea']=root
                return replace(output,value=value)
        class IdentityFusion(FakeFusion):
            def run(self, request):
                output=super().run(request)
                return replace(output,idea={**output.idea,**{k:deepcopy(request.root[k]) for k in ['tags','root_domains']}})
        for native, feedback in [(False,None),(False,{'records':[]}),(True,{'records':[]})]:
            with self.subTest(native=native,feedback=feedback):
                provider=MissingMetadataProvider();factory=FakeSearchFactory()
                request=replace(self.request,experiment_feedback=feedback,
                    research_policy=asdict(ResearchPolicy()) if native else {},
                    evidence=evidence_fixture(),resource_ids=('fixture-resource',))
                result=run_workflow(request,search_factory=factory,fusion_runner=IdentityFusion(),
                                    provider=provider,retrieval=FakeRetrieval())
                self.assertTrue(result.succeeded,result.error)
                self.assertEqual(len(factory.created),len(CANONICAL_MODES))
                self.assertEqual(len({s.requests[0].root_signature for s in factory.created}),1)
                for search in factory.created:
                    self.assertEqual(search.requests[0].root['tags'],parent['tags'])
                self.assertIn('root_identity_inheritance',result.artifact['analysis'])
                for operation,payload in provider.calls:
                    if operation in {OP_ANALYSIS,OP_REPLAN}:
                        prompt=_workflow_prompt(operation,payload)
                        self.assertIn('root_identity_contract',prompt)
                        self.assertIn('Preserve values and order',prompt)


if __name__=='__main__':
    unittest.main()
