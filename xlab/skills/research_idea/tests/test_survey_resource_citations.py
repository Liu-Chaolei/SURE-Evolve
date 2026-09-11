import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from research_idea_lib.resources.retrieval import CitationRegistry, parse_survey_markdown


class SurveyResourceCitationsTest(unittest.TestCase):
    def test_native_survey_markers_resolve_without_indexing_bibliography(self):
        registry = CitationRegistry.from_payloads({"references": [{"paper_id": "s2:abc"}, {"paper_id": "paper:kept"}]},
                                                 {"keynotes": {"s2:abc": "Extracted evidence"}})
        self.assertEqual(registry.resolve_markdown("Evidence [paper:s2:abc] and [paper:kept]."), ("s2:abc", "paper:kept"))
        with self.assertRaises(ValueError):
            registry.resolve_markdown("Unknown [paper:s2:missing].")
        paragraphs = parse_survey_markdown("# Topic\n\n## Findings\n\nA sufficiently long scientific claim [paper:s2:abc].\n\n## References\n\nBibliography text must never become retrieval evidence.", registry)
        self.assertTrue(any("s2:abc" in p.paper_ids for p in paragraphs))
        self.assertFalse(any("Bibliography text" in p.text for p in paragraphs))


if __name__ == "__main__":
    unittest.main()
