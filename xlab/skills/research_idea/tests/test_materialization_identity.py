from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from research_idea_lib.materialization_validation import align_public_materialization, materialization_contract
from research_idea_lib.algorithm.runtime_adapters import GenericWorkflowProvider
from research_idea_lib.algorithm.workflow import ProviderOperation, OP_MATERIALIZATION
from research_idea_lib.pipeline import merge_workflow_retrieval_context
from test_ordered_api_routing import environment, provider, success


def input_data():
    fusion={'title':'Exact title','abstract':'Description','core_contribution':'Exact contribution',
            'hypothesis':'Exact hypothesis','method':'Exact method','risks':'Exact risk',
            'components':[{'name':'core','description':'Mechanism'}],'root_domains':['speech'],'tags':['TTS']}
    evidence=[{'evidence_id':'e1','paper_ids':['p1'],'text':'Actual evidence'}]
    refs=[{'paper_id':'p1','title':'Actual paper'}]
    contract=materialization_contract({'idea':fusion,'evidence_ids':['e1']},evidence,refs)
    modes=['moonshot_inventor','bridge_builder','steady_engineer','ambitious_realist','evidence_first']
    idea={**fusion,**contract,'introduction':'Problem and evidence','research_question':'Can this help?',
          'experiment_plan':['Train within the fixed budget'],'data_requirements':['Declared training split'],
          'baselines':['Frozen baseline'],'metrics':['CER'],'algorithm':['Run the core mechanism'],'source_modes':modes}
    return {'fusion':fusion,'evidence':evidence,'references':refs,'evidence_ids':['e1'],'source_modes':modes},idea


class MaterializationIdentityTests(unittest.TestCase):
    def test_graph_citations_require_exact_attribution(self):
        data,idea=input_data()
        evidence=[{'evidence_id':'e1','paper_ids':['p2'],'kind':'graph_neighbor',
                   'provenance':{'title':'Graph paper','resource':{'role':'graph','neighbor_paper_id':'p2','descriptor_digest':'frozen-resource-digest'}}}]
        contract=materialization_contract({'idea':data['fusion'],'evidence_ids':['e1']},evidence,data['references'])
        self.assertEqual(contract['reference_ids'],['p2'])
        self.assertEqual(contract['reference_papers'],['Graph paper'])
        context = merge_workflow_retrieval_context({'references':data['references']}, {'retrieval':{'evidence':evidence}})
        self.assertEqual(context['references'][-1]['paper_id'],'p2')
        del evidence[0]['provenance']['resource']['descriptor_digest']
        with self.assertRaisesRegex(ValueError,'exact resource attribution'):
            materialization_contract({'idea':data['fusion'],'evidence_ids':['e1']},evidence,data['references'])

    def test_drift_falls_through_before_success_cache(self):
        for field in ['title','core_contribution','hypothesis','method','risks','reference_ids','reference_papers']:
            with self.subTest(field=field):
                data,idea=input_data();calls=[]
                def transport(url,headers,raw,timeout):
                    calls.append(headers['Authorization']);value=deepcopy(idea)
                    if len(calls)==1:value[field]=['incorrect'] if isinstance(value[field],list) else 'incorrect'
                    return success(json.loads(raw),{'idea_result':value})
                with patch.dict(os.environ,environment(),clear=True):
                    result=GenericWorkflowProvider(provider(transport),SimpleNamespace(fusion_model='gpt-6-astra')).execute(ProviderOperation(OP_MATERIALIZATION,data))
                self.assertEqual(len(calls),2)
                projected=deepcopy(result.value['idea_result'])
                align_public_materialization(projected,{'idea':data['fusion'],'evidence_ids':data['evidence_ids']},data['evidence'],data['references'])
                self.assertEqual(projected,idea)

    def test_contract_is_input_guidance_not_silent_output_repair(self):
        data,idea=input_data();idea['title']='Renamed title';original=deepcopy(idea)
        with self.assertRaisesRegex(ValueError,'title drifted'):
            align_public_materialization(idea,{'idea':data['fusion'],'evidence_ids':['e1']},data['evidence'],data['references'])
        self.assertEqual(idea,original)
