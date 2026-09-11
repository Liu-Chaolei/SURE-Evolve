import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_resource_bundle import collect_graph_resources, package_keynotes
from literature_survey_lib.manifest import write_final_manifest


class ResourcePackagingTest(unittest.TestCase):
    def test_components_keep_paper_identity_and_source_database_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.db"
            destination = Path(temporary) / "resource.db"
            with sqlite3.connect(source) as db:
                db.execute("CREATE TABLE nodes(id TEXT,node_type TEXT,paper_id TEXT,name TEXT,raw_json TEXT,paper_title TEXT,summary TEXT)")
                paper = {"metadata": {"abstract": "Actual paper abstract"}}
                core = {"name": "Encoder", "summary": "Encodes acoustic frames", "paper_title": "Actual paper",
                        "components": [{"name": "Attention", "summary": "Models temporal context", "quote": "Source quotation"}]}
                db.executemany("INSERT INTO nodes VALUES(?,?,?,?,?,'','')", [
                    ("paper1", "Paper", "p1", "Actual paper", json.dumps(paper)),
                    ("core1", "Core", "p1", "Encoder", json.dumps(core)),
                    ("fallback", "Core", "p2", "Unextracted", json.dumps({**core, "fallback_from_title": True})),
                ])
            papers, notes, components = collect_graph_resources(source, destination)
            self.assertEqual([c["component"] for c in components], ["Encoder", "Attention"])
            self.assertTrue(all(c["paper_ids"] == ["p1"] and c["node_id"] == "core1" for c in components))
            self.assertEqual(components[1]["quote"], "Source quotation")
            with sqlite3.connect(source) as db:
                self.assertEqual(db.execute("SELECT paper_title FROM nodes WHERE id='paper1'").fetchone()[0], "")
            with sqlite3.connect(destination) as db:
                self.assertEqual(db.execute("SELECT summary FROM nodes WHERE id='paper1'").fetchone()[0], "Actual paper abstract")

    def test_missing_extractions_are_explicit_and_resource_link_survives_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "state/xcientist").mkdir(parents=True)
            (run / "state/xcientist/keynotes.json").write_text(json.dumps({"keynotes": {"p1": "Paper Title: Unknown\nActual findings"}}))
            papers = {"p1": {"title": "Paper one"}, "p2": {"title": "Metadata only"}}
            payload, counts = package_keynotes(run, papers, {}, {"references": []})
            self.assertIn("Paper Title: Paper one", payload["keynotes"]["p1"]["keynote"])
            self.assertEqual(payload["keynotes"]["p2"]["evidence_level"], "metadata_only")
            self.assertIn("no extracted scientific findings", payload["keynotes"]["p2"]["keynote"])
            manifest = run / "manifest.json"
            write_final_manifest(manifest, {"resource_manifest": "resources/resource_manifest.json"})
            write_final_manifest(manifest, {"status": "success"})
            self.assertEqual(json.loads(manifest.read_text())["resource_manifest"], "resources/resource_manifest.json")


if __name__ == "__main__":
    unittest.main()
