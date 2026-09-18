import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from research_idea_lib.algorithm.prompts.idea_result_alignment import IDEA_RESULT_ALIGNMENT_PROMPT
from research_idea_lib.algorithm.runtime_adapters import _REQUIRED_RESULT_TEXT, _REQUIRED_RESULT_LISTS, _workflow_prompt
from research_idea_lib.algorithm.workflow import OP_MATERIALIZATION


class MaterializationPromptContractTests(unittest.TestCase):
    def test_template_declares_all_runtime_required_public_fields(self):
        prompt=IDEA_RESULT_ALIGNMENT_PROMPT.format(topic='TTS',mature_idea='baseline',
            refinement_scope='all',idea='{}',papers='[]')
        schema=json.loads(prompt.split('Return STRICT JSON only:',1)[1])
        self.assertEqual(set(schema),{'idea_result'})
        fields=schema['idea_result']
        self.assertTrue(set(_REQUIRED_RESULT_TEXT).issubset(fields))
        self.assertTrue(set(_REQUIRED_RESULT_LISTS).issubset(fields))
        for key in _REQUIRED_RESULT_TEXT:self.assertIsInstance(fields[key],str)
        for key in _REQUIRED_RESULT_LISTS:self.assertIsInstance(fields[key],list)
        self.assertTrue({'source_modes','evidence_ids','reference_ids','root_domains'}.issubset(fields))

    def test_runtime_prompt_preserves_exact_structural_and_evidence_input(self):
        source=[{'name':'scheduler_hook','description':'Cosine decay with a fixed floor.'}]
        prompt=_workflow_prompt(OP_MATERIALIZATION,{
            'topic':'TTS','fusion':{'components':source},'source_modes':['steady_engineer'],
            'evidence_ids':['evidence:fixture'],'research_policy':{'evidence_mode':'task_only'}})
        self.assertIn('exact deep copy',prompt)
        self.assertIn('fusion.components',prompt)
        self.assertIn('Cosine decay with a fixed floor.',prompt)
        self.assertIn('evidence:fixture',prompt)
        self.assertIn('both must instead be empty',prompt)


if __name__=='__main__':
    unittest.main()
