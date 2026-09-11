import json
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from literature_survey_lib.xcientist.modules.survey_generator import SurveyGenerator


class OutlineBatchesTest(unittest.TestCase):
    def generator(self):
        generator = SurveyGenerator.__new__(SurveyGenerator)
        options = SimpleNamespace(outline_generation_assign_max_retry=3, outline_generation_assign_batch_size=20,
                                  include_other_relevant_papers_RAG_in_outline=False,
                                  llm_max_context_overhead_length_outline_generation=1000, outline_generation_temperature=0.1)
        generator.config = SimpleNamespace(ModuleInfo=SimpleNamespace(SurveyGenerator=options),
                                           APIInfo=SimpleNamespace(llm_max_context_length=100000), BasicInfo=SimpleNamespace(debug=False, error_conservatism_mode=True))
        generator.logger = Mock()
        generator.outline_fast_mode = True
        generator.omit_error_preserve_retry_time = 0
        generator.format_papers_analysis = Mock(return_value="Analysis")
        generator.work_analyzer = SimpleNamespace(get_paper_keynote=Mock(return_value="Evidence"),
                                                 work_collector=SimpleNamespace(get_paper_title=Mock(return_value="Paper"), graph_paper_ids=set()))
        generator.chat_agent = Mock()
        generator.chat_agent.estimate_tokens.return_value = 100
        return generator

    def test_every_batch_is_assigned_in_fast_mode(self):
        generator = self.generator()
        papers = [f"p{i}" for i in range(45)]
        batches = []

        def respond(prompts, validator, **kwargs):
            responses = []
            for index, prompt in enumerate(prompts):
                ids = re.findall(r"Paper ID: (p\d+)\b", prompt)
                batches.append(ids)
                value = json.dumps([{"paper_id": pid, "assignment": {"Methods": ["Modeling"]}} for pid in ids])
                info = {**kwargs["info_dict"], "idx": index}
                valid, result = validator(value, info)
                self.assertTrue(valid)
                responses.append(result)
            return responses

        generator.chat_agent.batch_remote_chat_with_retry.side_effect = respond
        outline = {"title": "Survey", "sections": [{"title": "Methods", "subsections": [{"title": "Modeling"}]}]}
        result = generator.generate_outline_assign_papers(outline, [], "", papers)
        self.assertEqual([len(batch) for batch in batches], [20, 20, 5])
        self.assertEqual(result["sections"][0]["subsections"][0]["papers_to_use"], papers)
        self.assertEqual(result["assignment_version"], 2)

    def test_omitted_papers_and_unknown_sections_cannot_pass(self):
        generator = self.generator()
        info = {"papers": ["p"], "batches": [["p"]], "outline": {"sections": [{"title": "Methods", "subsections": []}]}}
        with self.assertRaisesRegex(ValueError, "omitted"):
            generator._validate_assignment("[]", info)
        with self.assertRaisesRegex(ValueError, "unknown section"):
            generator._validate_assignment('[{"paper_id":"p","assignment":{"Invented":[]}}]', info)
        valid, _ = generator._validate_assignment('[{"paper_id":"p","assignment":{},"exclusion_reason":"Outside speech research"}]', info)
        self.assertTrue(valid)

    def test_unambiguous_identifier_prefix_is_restored(self):
        generator = self.generator()
        info = {"papers": ["title:abc123"], "batches": [["title:abc123"]]}
        valid, result = generator._validate_assignment('[{"paper_id":"abc123","assignment":{"Methods":[]}}]', info)
        self.assertTrue(valid)
        self.assertEqual(json.loads(result)[0]["paper_id"], "title:abc123")
        with self.assertRaises(ValueError):
            generator._validate_assignment('[{"paper_id":"abc123","assignment":{}}]',
                                           {"papers": ["title:abc123", "s2:abc123"], "batches": [["title:abc123", "s2:abc123"]]})

    def test_unique_subsection_parent_can_be_restored(self):
        generator = self.generator()
        info = {"papers": ["p"], "batches": [["p"]], "outline": {"sections": [{"title": "Methods", "subsections": [{"title": "Modeling"}]}]}}
        valid, result = generator._validate_assignment('[{"paper_id":"p","assignment":{"Modeling":["Modeling"]}}]', info)
        self.assertTrue(valid)
        self.assertEqual(json.loads(result)[0]["assignment"], {"Methods": ["Modeling"]})
        info["outline"]["sections"].append({"title": "Applications", "subsections": [{"title": "Modeling"}]})
        with self.assertRaises(ValueError):
            generator._validate_assignment('[{"paper_id":"p","assignment":{"Modeling":["Modeling"]}}]', info)


if __name__ == "__main__":
    unittest.main()
