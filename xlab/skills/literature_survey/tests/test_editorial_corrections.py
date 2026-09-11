import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from literature_survey_lib.xcientist.utils.editorial_corrections import apply_editorial_corrections


class EditorialCorrectionsTest(unittest.TestCase):
    def test_exact_correction_preserves_surrounding_content_and_records_evidence(self):
        correction = {"before": "Wrong task result.", "after": "Verified task result <Primary>.",
                      "reason": "Source evaluated verification rather than diarization", "evidence_paper_ids": ["p1"], "occurrences": 1}
        text, report = apply_editorial_corrections("## Chapter\n\nWrong task result.\n\nOther content.", [correction], {"p1"})
        self.assertEqual(text, "## Chapter\n\nVerified task result <Primary>.\n\nOther content.")
        self.assertEqual(report[0]["evidence_paper_ids"], ["p1"])
        with self.assertRaises(ValueError):
            apply_editorial_corrections("Wrong task result. Wrong task result.", [correction], {"p1"})
        with self.assertRaises(ValueError):
            apply_editorial_corrections("Wrong task result.", [correction], set())
        with self.assertRaises(ValueError):
            apply_editorial_corrections("Wrong task result.", [{**correction, "after": "## Replacement chapter"}], {"p1"})


if __name__ == "__main__":
    unittest.main()
