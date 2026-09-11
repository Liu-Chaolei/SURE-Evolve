import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from literature_survey_lib.xcientist.modules.survey_generator import SurveyGenerator
from literature_survey_lib.xcientist.modules.pe import SECTION_REVIEW


class ReviewScopeTest(unittest.TestCase):
    def test_parallel_review_preserves_section_order_neighbors_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            generator = SurveyGenerator.__new__(SurveyGenerator)
            generator.config = SimpleNamespace(BasicInfo=SimpleNamespace(base_dir=temporary, topic="Speech", debug=False),
                                               APIInfo=SimpleNamespace(llm_model_name="test", llm_api_base_url="https://example.invalid"),
                                               ModuleInfo=SimpleNamespace(SurveyGenerator=SimpleNamespace(enable_review_and_revise=True)))
            generator.logger = Mock()
            generator.chat_agent = SimpleNamespace(batch_workers=2)
            generator.review_and_revise_section = Mock(side_effect=lambda text, **kwargs: text + " Reviewed.")
            outline = {"title": "Speech", "sections": [{"title": "Methods"}, {"title": "Evaluation"}]}
            draft = {"section_drafts": ["Methods body", "Evaluation body"]}
            result = generator.review_and_revise_survey_in_parts(copy.deepcopy(draft), outline)
            self.assertEqual(result["section_drafts"], ["Methods body Reviewed.", "Evaluation body Reviewed."])
            calls = {call.kwargs["section_outline"]["title"]: call.kwargs for call in generator.review_and_revise_section.call_args_list}
            self.assertEqual(calls["Methods"]["next_section_text"], "Evaluation body")
            self.assertEqual(calls["Evaluation"]["previous_section_text"], "Methods body")
            generator.review_and_revise_survey_in_parts(copy.deepcopy(draft), outline)
            self.assertEqual(generator.review_and_revise_section.call_count, 2)

    def test_headers_and_ambiguous_revision_targets_are_preserved(self):
        generator = SurveyGenerator.__new__(SurveyGenerator)
        generator.logger = Mock()
        generator.config = SimpleNamespace(BasicInfo=SimpleNamespace(debug=False))
        text = "### Methods\nRepeated. Repeated."
        self.assertEqual(generator._apply_revision_to_text(text, {"originalText": text, "newText": "Short summary"}), text)
        self.assertEqual(generator._apply_revision_to_text(text, {"originalText": "Repeated.", "newText": "Changed."}), text)
        self.assertIn("applies ONLY to the introduction", SECTION_REVIEW)
        self.assertIn("Preserve every subsection", SECTION_REVIEW)


if __name__ == "__main__":
    unittest.main()
