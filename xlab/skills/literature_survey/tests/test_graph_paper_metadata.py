import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from literature_survey_lib.paper_sources import graph_db_papers_and_context


class GraphPaperMetadataTest(unittest.TestCase):
    def test_paper_after_semantic_row_limit_keeps_nested_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "graph.db"
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE nodes (id TEXT, node_type TEXT, paper_id TEXT, name TEXT, raw_json TEXT)")
                connection.executemany(
                    "INSERT INTO nodes VALUES (?, 'Baseline', 'unrelated', 'Baseline', '{}')",
                    [(str(index),) for index in range(50)],
                )
                raw = {"metadata": {"abstract": "Actual paper evidence.", "authors": [{"name": "Author"}], "year": 2024, "venue": "Conference", "url": "https://example.org/paper"}}
                connection.execute("INSERT INTO nodes VALUES ('p', 'Paper', 's2:paper', 'Paper title', ?)", (json.dumps(raw),))
            papers, _context = graph_db_papers_and_context(path, 1)
            self.assertEqual(len(papers), 1)
            self.assertEqual(papers[0]["id"], "s2:paper")
            self.assertEqual(papers[0]["abstract"], "Actual paper evidence.")
            self.assertEqual(papers[0]["authors"], [{"name": "Author"}])
            self.assertEqual(papers[0]["year"], 2024)
            self.assertEqual(papers[0]["venue"], "Conference")
            self.assertEqual(papers[0]["url"], "https://example.org/paper")


if __name__ == "__main__":
    unittest.main()
