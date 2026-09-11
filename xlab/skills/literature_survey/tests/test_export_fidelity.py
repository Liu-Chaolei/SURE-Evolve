import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from literature_survey_lib.xcientist.engine import first_sentence, gaps_from_sections, sections_from_outline_and_markdown, ensure_markdown_citations, normalize_outline, paper_assignment_from_outline, normalize_xlab_markdown
from literature_survey_lib.pipeline import render_survey_agent_markdown


class ExportFidelityTest(unittest.TestCase):
    def test_only_unique_explicit_paper_titles_gain_citation_markers(self):
        context = SimpleNamespace(paper_by_id={"p": {"title": "Actual Paper"}, "q": {"title": "Ambiguous"}, "r": {"title": "Ambiguous"}})
        body = "## Heading\n\n**Actual Paper**\n\nThe method in **Actual Paper** improves quality. **Unknown Work** is unverified. **Ambiguous** cannot be resolved."
        result = normalize_xlab_markdown(body, [], context)
        self.assertIn("method in **Actual Paper** [paper:p] improves", result)
        self.assertIn("\n\n**Actual Paper**\n\n", result)
        self.assertEqual(result.count("[paper:"), 1)

    def test_excluded_papers_are_not_attached_to_first_section_or_bibliography(self):
        outline = normalize_outline({"title": "Survey", "sections": [{"title": "Methods", "papers_to_use": ["p"], "subsections": []}],
                                     "excluded_papers": [{"paper_id": "q", "exclusion_reason": "Outside scope"},
                                                         {"paper_id": "p", "exclusion_reason": "Auxiliary disagreement"}]}, ["p", "q"])
        self.assertEqual(outline["sections"][0]["paper_ids"], ["p"])
        self.assertEqual([paper["paper_id"] for paper in outline["excluded_papers"]], ["q"])
        assignment = paper_assignment_from_outline(outline, [{"id": "p"}, {"id": "q"}])["assignments"]
        self.assertEqual(assignment[1]["section_ids"], [])
        self.assertEqual(assignment[1]["exclusion_reason"], "Outside scope")
        text = render_survey_agent_markdown("Survey", "Evidence [paper:p]", [
            {"paper_id": "p", "title": "Cited work", "authors": ["Author"], "year": 2024, "venue": "Conference", "url": "https://example.org/p"},
            {"paper_id": "q", "title": "Excluded work"},
        ])
        self.assertIn("Author", text)
        self.assertIn("Conference", text)
        self.assertNotIn("Excluded work", text)

    def test_real_prose_and_citations_are_preserved(self):
        body = "## 1. Methods\n\nASR converts speech into text. [paper:p]\n\n### Streaming\n\nStreaming improves responsiveness. [paper:q]\n\n## 2. Outlook\n\nAn open challenge is robust recognition under noise. [paper:p]"
        outline = {"sections": [{"id": "s1", "title": "Methods", "paper_ids": ["uncited"]}, {"id": "s2", "title": "Outlook"}]}
        context = SimpleNamespace(paper_by_id={"p": {}, "q": {}, "uncited": {}})
        sections = sections_from_outline_and_markdown(outline, body, context)
        self.assertIn("Streaming improves", sections[0]["content_markdown"])
        self.assertEqual(sections[0]["paper_ids"], ["p", "q"])
        self.assertEqual(sections[0]["summary"], "ASR converts speech into text.")
        gaps = gaps_from_sections(sections, "ASR")
        self.assertEqual(gaps, [{"id": "gap-1", "gap": "An open challenge is robust recognition under noise. [paper:p]", "paper_ids": ["p"]}])

    def test_missing_sections_and_citations_are_not_fabricated(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            sections_from_outline_and_markdown({"sections": [{"title": "Missing"}]}, "## Other\nText", SimpleNamespace(paper_by_id={}))
        with self.assertRaisesRegex(ValueError, "no traceable citations"):
            ensure_markdown_citations("Uncited body", [{"paper_ids": ["p"]}], ["p"], 8)
        self.assertEqual(gaps_from_sections([{"content_markdown": "Ordinary discussion.", "paper_ids": ["p"]}], "ASR"), [])


if __name__ == "__main__":
    unittest.main()
