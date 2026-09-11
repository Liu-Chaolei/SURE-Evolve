from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from playground.sure_master.core.history import XlabHistoryJournal


class XlabHistoryTests(unittest.TestCase):
    def test_round_history_survives_restart_and_redacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifacts" / "history.json"
            journal = XlabHistoryJournal(path)
            journal.append(
                "summary_published",
                {
                    "summary_digest": "sha256:summary",
                    "api_key": "secret",
                    "candidates": [{"idea_id": "idea-1", "status": "success"}],
                },
            )
            restored = XlabHistoryJournal(path)
            self.assertEqual(restored.summary_references(), ["sha256:summary"])
            self.assertEqual(restored.round_history()[0]["api_key"], "[REDACTED]")

    def test_duplicate_event_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            journal = XlabHistoryJournal(path)
            first = journal.append("request_accepted", {"request_id": "request-1"})
            second = journal.append("request_accepted", {"request_id": "request-1"})
            self.assertEqual(first, second)
            self.assertEqual(len(journal.snapshot()), 1)

    def test_invalid_digest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            path.write_text(
                '{"schema_version":"sure.xlab_history.v1","records":[{"event":"x","payload":{},"record_digest":"bad"}]}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                XlabHistoryJournal(path)


if __name__ == "__main__":
    unittest.main()
