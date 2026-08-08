import unittest

from playground.asr_master.core.playground import ASRMasterPlayground
from playground.asr_master.core.utils.metric import (
    extract_wer_from_metric,
    normalize_wer_score,
)


class TestASRMetric(unittest.TestCase):
    def test_extract_normalizes_decimal_wer(self):
        ok, wer = extract_wer_from_metric(r"\boxed{0.1577}")
        self.assertTrue(ok)
        self.assertAlmostEqual(wer, 15.77)

    def test_extract_keeps_percentage_number_wer(self):
        ok, wer = extract_wer_from_metric(r"\boxed{15.66}")
        self.assertTrue(ok)
        self.assertAlmostEqual(wer, 15.66)

        ok, wer = extract_wer_from_metric(r"\boxed{{15.66%}}")
        self.assertTrue(ok)
        self.assertAlmostEqual(wer, 15.66)

    def test_extract_rejects_missing_or_invalid_wer(self):
        self.assertEqual(extract_wer_from_metric("Not found"), (False, None))
        self.assertEqual(extract_wer_from_metric(r"\boxed{abc}"), (False, None))
        self.assertIsNone(normalize_wer_score(101.0))

    def test_compare_score_uses_normalized_wer(self):
        playground = object.__new__(ASRMasterPlayground)
        self.assertTrue(playground.compare_score(15.77, 15.66))
        self.assertTrue(playground.compare_score(0.1577, 15.66))
        self.assertFalse(playground.compare_score(0.1566, 15.77))


if __name__ == "__main__":
    unittest.main()
