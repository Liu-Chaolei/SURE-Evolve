import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from literature_survey_lib.xcientist.config import load_survey_agent_config


class VerifiedReferenceTest(unittest.TestCase):
    def test_verified_pdf_identity_and_checksum_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph = root / "graph-run/artifacts/graph.db"
            graph.parent.mkdir(parents=True)
            graph.write_bytes(b"graph fixture")
            pdf = root / "paper.pdf"
            pdf.write_bytes(b"%PDF fixture")
            digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
            metadata = {"title": "Primary Paper", "authors": ["Author"], "year": 2024, "venue": "Conference", "url": "https://example.org/p"}
            fetch = root / "fetch.json"
            fetch.write_text(json.dumps({"schema_version": "xlab.paper_fetch_manifest.v1", "papers": [
                {**metadata, "id": "arxiv:example", "download_status": "downloaded", "pdf_path": str(pdf), "sha256": digest}
            ]}))
            aliases = root / "aliases.json"
            aliases.write_text(json.dumps({"schema_version": "xlab.citation_aliases.v1", "graph_run_id": "graph-run", "aliases": [],
                                          "verified_references": [{**metadata, "paper_id": "arxiv:example", "fetch_manifest": str(fetch), "pdf_sha256": digest}]}))
            document = MagicMock()
            document.__len__.return_value = 1
            document.__getitem__.return_value.get_textpage.return_value.get_text_range.return_value = "Primary Paper\nAuthor\nVerified source text."
            with patch.dict(os.environ, {"XLAB_LITERATURE_SURVEY_CITATION_ALIASES": str(aliases)}), patch("literature_survey_lib.xcientist.config.pdfium.PdfDocument", return_value=document):
                load_survey_agent_config(run_dir=root / "run", topic="Speech", graph_db=graph)
                imported = json.loads((root / "run/state/xcientist/citation_aliases.json").read_text())
                self.assertIn("Verified source text", imported["verified_references"][0]["verified_reference_text"])
                pdf.write_bytes(b"changed PDF")
                with self.assertRaisesRegex(ValueError, "does not match"):
                    load_survey_agent_config(run_dir=root / "run", topic="Speech", graph_db=graph)


if __name__ == "__main__":
    unittest.main()
