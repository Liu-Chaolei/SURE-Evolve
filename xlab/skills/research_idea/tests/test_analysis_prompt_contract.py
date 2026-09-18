import json
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from research_idea_lib.algorithm.prompts.advanced_analysis import ADVANCED_ANALYSIS_PROMPT
from research_idea_lib.algorithm.workflow import _require_analysis


class AnalysisPromptContractTests(unittest.TestCase):
    def test_prompt_schema_matches_workflow_validator(self):
        prompt = ADVANCED_ANALYSIS_PROMPT.format(**{name:'' for name in [
            'topic','mature_idea','mature_idea_source','refinement_scope','refinement_scope_source',
            'survey_contents','papers','experiment_findings']})
        schema = prompt.split('with the schema:\n',1)[1].split('== Rules',1)[0]
        value = json.loads(re.sub(r'//[^\n]*','',schema))
        _require_analysis(value,mature=False)
        self.assertIsInstance(value['analysis'],dict)
        self.assertTrue(value['root_idea']['components'])


if __name__=='__main__':
    unittest.main()
