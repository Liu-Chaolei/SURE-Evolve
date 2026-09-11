import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from literature_survey_lib.xcientist.modules.database import Database
from literature_survey_lib.xcientist.modules.survey_generator import SurveyGenerator


class CitationIdentityTest(unittest.TestCase):
    def test_named_toolkit_is_not_misattributed_to_similar_selected_paper(self):
        with tempfile.TemporaryDirectory() as temporary:
            papers = {
                "sb": {"id": "sb", "title": "SpeechBrain: A General-Purpose Speech Toolkit", "abstract": "Toolkit"},
                "esp": {"id": "esp", "title": "ESPnet: End-to-End Speech Processing Toolkit", "abstract": "Toolkit"},
            }
            collector = SimpleNamespace(data_manager=SimpleNamespace(xlab_paper_by_id=papers), expand_in_local_paper_graph=True,
                                        get_paper_title_abstract=lambda pid: (papers[pid]["title"], papers[pid]["abstract"]))
            config = SimpleNamespace(BasicInfo=SimpleNamespace(base_dir=temporary), ModuleInfo=SimpleNamespace(WorkCollector=SimpleNamespace(sentence_transformer_model="test", sentence_transformer_batch_size=2)))
            model = Mock()
            model.encode.return_value = torch.ones((1, 2))
            with patch("literature_survey_lib.xcientist.modules.database.load_sentence_transformer_auto", return_value=(model, "cpu")):
                database = Database(config, collector)
            database.build(["esp"])
            self.assertEqual(database.resolve_title_to_paper_id("SpeechBrain: A Comprehensive Toolkit for End-to-End Speech Processing")[0], "sb")
            self.assertEqual(model.encode.call_count, 2)
            self.assertIn("sb", database.resolved_reference_papers)
            database.query_titles = Mock(return_value=(["esp"], [0.6]))
            model.encode.side_effect = [torch.tensor([[1.0, 0.0]]), torch.tensor([[0.6, 0.8]])]
            database.resolve_title_to_paper_id("Paraphrased toolkit comparison", 0.5, require_verified_identity=False)
            with self.assertRaises(ValueError):
                database.resolve_title_to_paper_id("Paraphrased toolkit comparison", 0.7, require_verified_identity=False)
            with self.assertRaisesRegex(ValueError, "requires identity verification"):
                database.resolve_title_to_paper_id("Paraphrased toolkit comparison", 0.5)
            self.assertFalse(database.resolution_records["paraphrased toolkit comparison"]["accepted"])
            self.assertEqual(database.query_titles.call_count, 1)

    def test_unresolved_citations_are_recorded_instead_of_silently_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            generator = SurveyGenerator.__new__(SurveyGenerator)
            generator.config = SimpleNamespace(BasicInfo=SimpleNamespace(base_dir=temporary, debug=False), ModuleInfo=SimpleNamespace(SurveyGenerator=SimpleNamespace(valid_title_min_similarity=0.5)))
            generator.database = SimpleNamespace(resolve_title_to_paper_id=Mock(side_effect=ValueError("unknown")))
            with self.assertRaisesRegex(ValueError, "remain unresolved"):
                generator.extract_and_process_citations("A claim <Unknown paper>.")
            errors = json.loads((Path(temporary) / "state/xcientist/citation_errors.json").read_text())
            self.assertEqual(errors["unresolved_titles"], ["Unknown paper"])


if __name__ == "__main__":
    unittest.main()
