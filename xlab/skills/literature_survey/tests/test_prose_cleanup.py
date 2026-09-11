import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from literature_survey_lib.xcientist.utils.prose_cleanup import remove_repeated_prose


class ProseCleanupTest(unittest.TestCase):
    def test_repeated_evidence_is_preserved_once_per_subsection(self):
        paragraph = "A cited explanation <Primary paper>. " * 10
        unique = "An independent discussion with additional evidence. " * 6
        text = f"## Topic\n\n{paragraph}\n\n{unique}\n\n{paragraph}\n\n### Other\n\n{paragraph}"
        cleaned, count = remove_repeated_prose(text)
        self.assertEqual(count, 1)
        self.assertEqual(cleaned.count(paragraph), 2)
        self.assertIn(unique, cleaned)
        self.assertIn("### Other", cleaned)

    def test_tables_and_fenced_code_are_not_changed(self):
        row = "| measurement | 1.0 | " * 20
        prose = "An intentionally repeated code string. " * 10
        text = f"{row}\n\n{row}\n\n```text\n{prose}\n\n{prose}\n```"
        self.assertEqual(remove_repeated_prose(text), (text, 0))


if __name__ == "__main__":
    unittest.main()
