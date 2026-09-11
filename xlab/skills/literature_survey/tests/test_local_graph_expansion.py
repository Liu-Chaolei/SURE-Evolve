import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from literature_survey_lib.xcientist.adapters import SeededDataManager, load_local_graph_papers, materialize_context_papers
from literature_survey_lib.xcientist.modules.work_analyzer import WorkAnalyzer


class LocalGraphExpansionTest(unittest.TestCase):
    def test_bibliography_uses_local_authors_without_network(self):
        analyzer = WorkAnalyzer.__new__(WorkAnalyzer)
        analyzer.reference_graph = None
        analyzer.work_collector = SimpleNamespace(data_manager=SimpleNamespace(xlab_paper_by_id={
            "p": {"title": "Speech paper", "authors": [{"name": "Author"}, "Coauthor"], "year": 2024, "venue": "Conference"}
        }))
        analyzer.semantic_scholar_api = Mock()
        self.assertEqual(analyzer.generate_mla("p"), 'Author and Coauthor. "Speech paper." *Conference*, 2024.')
        analyzer.semantic_scholar_api.get_paper_details.assert_not_called()

    def test_cached_expanded_authors_are_normalized_on_resume(self):
        context = SimpleNamespace(paper_by_id={"p": {"id": "p", "title": "Paper", "authors": [{"name": "Author"}, "Coauthor"]}})
        result = materialize_context_papers(context, ["p"])
        self.assertEqual(result[0]["authors"], ["Author", "Coauthor"])
        self.assertEqual(context.paper_by_id["p"]["authors"], ["Author", "Coauthor"])

    def test_expanded_paper_metadata_is_available_without_provider_lookup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "state" / "graph.db"
            path.parent.mkdir()
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE nodes (id TEXT, node_type TEXT, paper_id TEXT, name TEXT, paper_title TEXT, summary TEXT, pub_year INTEGER, source_venue TEXT, raw_json TEXT)")
                raw = {"metadata": {"abstract": "Evidence from the expanded paper.", "year": 2024, "venue": "Conference", "authors": [{"name": "Author"}]}}
                connection.execute("INSERT INTO nodes VALUES ('p', 'Paper', 's2:expanded', 'Expanded paper', '', '', NULL, '', ?)", (json.dumps(raw),))
            config = SimpleNamespace(BasicInfo=SimpleNamespace(base_dir=str(root)), ModuleInfo=SimpleNamespace(PaperGraphRetriever=SimpleNamespace(db_path=str(path))))
            papers = load_local_graph_papers(config)
            manager = SeededDataManager.__new__(SeededDataManager)
            manager.xlab_paper_by_id = papers
            result = manager.get_paper_with_title("Expanded paper")
            self.assertEqual(result["paperId"], "s2:expanded")
            self.assertEqual(result["abstract"], "Evidence from the expanded paper.")
            with sqlite3.connect(path) as connection:
                self.assertEqual(connection.execute("SELECT paper_title, pub_year FROM nodes WHERE paper_title = 'Expanded paper'").fetchone(), ("Expanded paper", 2024))

    def test_upstream_graph_is_not_modified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "upstream.db"
            path.write_bytes(b"upstream must not be opened for mutation")
            config = SimpleNamespace(BasicInfo=SimpleNamespace(base_dir=str(root / "run")), ModuleInfo=SimpleNamespace(PaperGraphRetriever=SimpleNamespace(db_path=str(path))))
            self.assertEqual(load_local_graph_papers(config), {})
            self.assertEqual(path.read_bytes(), b"upstream must not be opened for mutation")


if __name__ == "__main__":
    unittest.main()
