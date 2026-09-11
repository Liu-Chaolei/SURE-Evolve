from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


SKILL_DIR = Path(__file__).resolve().parents[1]
SCRIPT = SKILL_DIR / "scripts" / "build_graph.py"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class LlmHandler(BaseHTTPRequestHandler):
    calls: list[dict[str, object]] = []
    unavailable = False
    lock = threading.Lock()

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        if self.unavailable:
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b'{"error":"temporarily unavailable"}')
            return
        user = payload["messages"][-1]["content"]
        with self.lock:
            self.calls.append(
                {
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                    "user": user,
                    "enable_thinking": payload.get("chat_template_kwargs", {}).get("enable_thinking"),
                }
            )
        is_second = "GCNPlus, a new graph model" in user
        if "PASS: MAIN" in user:
            if is_second:
                result = {
                    "metadata": {
                        "domain": ["graph learning"],
                        "paper_type": ["method"],
                        "structured_summary": {
                            "problem": "Graph learning needs improved propagation.",
                            "method": "GCNPlus is introduced.",
                            "results": "It is evaluated on Citeseer.",
                        },
                        "code_url": None,
                    },
                    "ideation_resource": {
                        "problems": [],
                        "core_contributions": [
                            {
                                "name": "GCNPlus",
                                "acronym": None,
                                "type": "Method",
                                "keywords": ["propagation"],
                                "summary": "A new graph model.",
                                "insight": None,
                                "quote": "We introduce GCNPlus, a new graph model.",
                            }
                        ],
                        "core_relations": [],
                        "components": [],
                        "innovations": [],
                        "limitations": [],
                        "future_work": [],
                    },
                }
            else:
                result = {
                    "metadata": {
                        "domain": ["robust graph learning"],
                        "paper_type": ["method"],
                        "structured_summary": {
                            "problem": "Noisy topology is challenging.",
                            "method": "GraphMixer uses adaptive aggregation.",
                            "results": "It is evaluated on Cora.",
                        },
                        "code_url": None,
                    },
                    "ideation_resource": {
                        "problems": [
                            {
                                "keywords": ["noisy topology"],
                                "summary": "Noisy topology is the challenge.",
                                "insight": None,
                                "quote": "The challenge is noisy topology.",
                                "related_to_core": "GraphMixer",
                            }
                        ],
                        "core_contributions": [
                            {
                                "name": "GraphMixer",
                                "acronym": None,
                                "type": "Method",
                                "keywords": ["robust graph learning"],
                                "summary": "A robust graph learning framework.",
                                "insight": None,
                                "quote": "We propose GraphMixer, a novel framework for robust graph learning.",
                            }
                        ],
                        "core_relations": [],
                        "components": [
                            {
                                "name": "Adaptive Aggregation Module",
                                "keywords": ["aggregation"],
                                "summary": "GraphMixer contains adaptive aggregation.",
                                "insight": None,
                                "quote": "GraphMixer contains an Adaptive Aggregation Module.",
                                "related_to_core": "GraphMixer",
                            }
                        ],
                        "innovations": [],
                        "limitations": [
                            {
                                "keywords": ["memory"],
                                "summary": "Additional memory is required.",
                                "insight": None,
                                "quote": "A limitation is additional memory.",
                                "related_to_core": "GraphMixer",
                            }
                        ],
                        "future_work": [],
                    },
                }
        elif "PASS: GRAPH" in user:
            if is_second:
                result = {
                    "graph_data": {
                        "baselines": [
                            {
                                "name": "GraphMixer",
                                "acronym": None,
                                "related_to_core": "GCNPlus",
                                "keywords": ["comparison"],
                                "summary": "Compared against GraphMixer.",
                                "metrics": ["F1"],
                                "insight": None,
                                "quote": "We compare against GraphMixer and evaluate on Citeseer benchmark using F1.",
                            }
                        ],
                        "datasets": [
                            {
                                "name": "Citeseer",
                                "acronym": None,
                                "related_to_core": "GCNPlus",
                                "keywords": ["benchmark"],
                                "summary": "Evaluated on Citeseer.",
                                "metrics": ["F1"],
                                "insight": None,
                                "quote": "We compare against GraphMixer and evaluate on Citeseer benchmark using F1.",
                            }
                        ],
                    }
                }
            else:
                result = {
                    "graph_data": {
                        "baselines": [
                            {
                                "name": "GCN",
                                "acronym": None,
                                "related_to_core": "GraphMixer",
                                "keywords": ["comparison"],
                                "summary": "Compared against GCN.",
                                "metrics": ["accuracy"],
                                "insight": None,
                                "quote": "We compare against GCN and evaluate on Cora dataset using accuracy.",
                            }
                        ],
                        "datasets": [
                            {
                                "name": "Cora",
                                "acronym": None,
                                "related_to_core": "GraphMixer",
                                "keywords": ["dataset"],
                                "summary": "Evaluated on Cora.",
                                "metrics": ["accuracy"],
                                "insight": None,
                                "quote": "We compare against GCN and evaluate on Cora dataset using accuracy.",
                            }
                        ],
                    }
                }
        else:
            if "GCNPlus" in user and '"kind": "baseline"' in user:
                result = {
                    "calibrations": [
                        {
                            "kind": "baseline",
                            "name": "GraphMixer",
                            "keep": True,
                            "canonical_name": "GraphMixer",
                            "acronym": None,
                            "related_to_core": "GCNPlus",
                            "citation_paper_id": "paper-a",
                            "reason": "Explicit comparison and local citation.",
                        },
                        {
                            "kind": "dataset",
                            "name": "Citeseer",
                            "keep": True,
                            "canonical_name": "Citeseer",
                            "acronym": None,
                            "related_to_core": "GCNPlus",
                            "citation_paper_id": None,
                            "reason": "Explicit evaluation dataset.",
                        },
                    ]
                }
            else:
                result = {
                    "calibrations": [
                        {
                            "kind": "baseline",
                            "name": "GCN",
                            "keep": True,
                            "canonical_name": "GCN",
                            "acronym": None,
                            "related_to_core": "GraphMixer",
                            "citation_paper_id": None,
                            "reason": "Explicit comparison.",
                        },
                        {
                            "kind": "dataset",
                            "name": "Cora",
                            "keep": True,
                            "canonical_name": "Cora",
                            "acronym": None,
                            "related_to_core": "GraphMixer",
                            "citation_paper_id": None,
                            "reason": "Explicit evaluation dataset.",
                        },
                    ]
                }
        response = json.dumps(
            {
                "id": f"response-{len(self.calls)}",
                "model": payload["model"],
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(result),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 100,
                    "total_tokens": 200,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


class SemanticScholarHandler(BaseHTTPRequestHandler):
    calls: list[str] = []
    lock = threading.Lock()

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        with self.lock:
            self.calls.append(self.path)
        references = [
            {
                "paperId": "s2-ref-1",
                "title": "Spectral Graph Convolution",
                "venue": "ICLR",
                "year": 2017,
                "externalIds": {"DOI": "10.1000/gcn"},
                "url": "https://example.test/gcn",
                "authors": [{"name": "Kipf"}],
            }
        ]
        if parsed.path.startswith("/graph/v1/paper/DOI%3A") or parsed.path.startswith(
            "/graph/v1/paper/DOI:"
        ):
            result = {
                "paperId": "s2-paper-a",
                "title": "GraphMixer",
                "references": references,
            }
        elif parsed.path == "/graph/v1/paper/search":
            query = parse_qs(parsed.query).get("query", [""])[0]
            result = {
                "data": [
                    {
                        "paperId": "s2-search",
                        "title": query,
                        "references": references,
                    }
                ]
            }
        else:
            self.send_response(404)
            self.end_headers()
            return
        response = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


class KnowledgeGraphPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.collection = self.root / "collection with spaces"
        self.run_dir = self.root / "run"
        self.fake_bin = self.root / "bin"
        self.fake_bin.mkdir()
        self._write_fake_mineru()
        LlmHandler.calls = []
        LlmHandler.unavailable = False
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), LlmHandler)
        self.server_thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.server_thread.start()
        self.environment = dict(os.environ)
        self.environment.update(
            {
                "PATH": f"{self.fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
                "KNOWLEDGE_GRAPH_LLM_API_URL": (
                    f"http://127.0.0.1:{self.server.server_port}/v1"
                ),
                "KNOWLEDGE_GRAPH_LLM_API_KEY": "test-secret-key",
                "KNOWLEDGE_GRAPH_LLM_MODEL": "test-model",
                "KNOWLEDGE_GRAPH_DISABLE_SEMANTIC_SCHOLAR": "1",
                "MINERU_TOOLS_CONFIG_JSON": str(self.root / "mineru-config.json"),
            }
        )

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=5)
        self.temporary.cleanup()

    def _write_fake_mineru(self) -> None:
        executable = self.fake_bin / "mineru"
        executable.write_text(
            """#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-p", required=True)
parser.add_argument("-o", required=True)
parser.add_argument("-b")
parser.add_argument("-m")
parser.add_argument("-l")
args = parser.parse_args()
sources = sorted(Path(args.p).glob("*.pdf"))
if len(sources) > 1 and any("[POISON_BATCH]" in source.read_text(encoding="utf-8", errors="replace") for source in sources):
    raise SystemExit(5)
for source in sources:
    stem = source.stem
    text = source.read_text(encoding="utf-8", errors="replace").split("\\n", 1)[-1].strip()
    output = Path(args.o) / stem / args.b
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{stem}.md").write_text("# Introduction\\n\\n" + text + "\\n", encoding="utf-8")
    (output / f"{stem}_content_list.json").write_text(json.dumps([
        {"type": "text", "text": "Introduction", "text_level": 1, "bbox": [0, 0, 100, 20], "page_idx": 0},
        {"type": "text", "text": text, "bbox": [0, 30, 500, 700], "page_idx": 0}
    ]), encoding="utf-8")
    (output / f"{stem}_middle.json").write_text(json.dumps({
        "pdf_info": [{"para_blocks": [{"type": "text", "bbox": [0, 30, 500, 700]}], "discarded_blocks": []}],
        "_backend": args.b,
        "_version_name": "3.4.4"
    }), encoding="utf-8")
    (output / f"{stem}_model.json").write_text("{}", encoding="utf-8")
    (output / "images").mkdir()
""",
            encoding="utf-8",
        )
        executable.chmod(0o755)

    def command(
        self, *arguments: str, expected: int = 0
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), *arguments],
            capture_output=True,
            text=True,
            timeout=120,
            env=self.environment,
        )
        if result.returncode != expected:
            self.fail(
                f"command returned {result.returncode}, expected {expected}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return result

    def make_collection(
        self,
        *,
        status: str = "success",
        poison_batch: bool = False,
        local_references: bool = False,
        high_recall_extra: bool = False,
    ) -> Path:
        artifacts = self.collection / "artifacts"
        pdfs = artifacts / "pdfs"
        metadata = artifacts / "metadata"
        pdfs.mkdir(parents=True)
        metadata.mkdir(parents=True)
        first_text = (
            "We propose GraphMixer, a novel framework for robust graph learning. "
            "The challenge is noisy topology. "
            "GraphMixer contains an Adaptive Aggregation Module. "
            "We compare against GCN and evaluate on Cora dataset using accuracy. "
            "A limitation is additional memory."
        )
        if local_references:
            first_text += (
                "\n\n# References\n\n"
                "[1] Jane Doe. Local Graph Baseline. NeurIPS 2020.\n"
            )
        if high_recall_extra:
            first_text += " We also compare against HiddenNet and evaluate on PubMed dataset using accuracy."
        if poison_batch:
            first_text += " [POISON_BATCH]"
        second_text = (
            "We introduce GCNPlus, a new graph model. "
            "We compare against GraphMixer and evaluate on Citeseer benchmark using F1."
        )
        paper_a_pdf = pdfs / "paper-a.pdf"
        paper_b_pdf = pdfs / "paper-b.pdf"
        paper_a_pdf.write_text(f"%PDF-FAKE\n{first_text}\n", encoding="utf-8")
        paper_b_pdf.write_text(f"%PDF-FAKE\n{second_text}\n", encoding="utf-8")
        paper_a_sha256 = hashlib.sha256(paper_a_pdf.read_bytes()).hexdigest()
        paper_b_sha256 = hashlib.sha256(paper_b_pdf.read_bytes()).hexdigest()
        (metadata / "edges.jsonl").write_text(
            json.dumps(
                {
                    "schema_version": "xlab.paper_edge.v1",
                    "source_paper_id": "paper-b",
                    "target_paper_id": "paper-a",
                    "relation": "cites",
                    "discovered_from": "test",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        write_json(
            artifacts / "papers.manifest.json",
            {
                "schema_version": "xlab.paper_set.v3",
                "files": {"edges": "artifacts/metadata/edges.jsonl"},
                "papers": [
                    {
                        "id": "paper-a",
                        "title": "GraphMixer",
                        "year": 2025,
                        "authors": [{"name": "A"}],
                        "external_ids": {"DOI": "10.1000/graphmixer"},
                        "download": {
                            "status": "downloaded",
                            "path": "paper-a.pdf",
                            "sha256": paper_a_sha256,
                        },
                    },
                    {
                        "id": "paper-b",
                        "title": "GCNPlus",
                        "year": 2026,
                        "authors": [{"name": "B"}],
                        "download": {
                            "status": "downloaded",
                            "path": "paper-b.pdf",
                            "sha256": paper_b_sha256,
                        },
                    },
                ],
            },
        )
        write_json(
            self.collection / "manifest.json",
            {
                "schema_version": "2",
                "run_id": "paper-collect-test-run",
                "skill_name": "paper_collect",
                "skill_version": "3.0.0",
                "status": status,
                "outputs": {
                    "paper_collection": "collection with spaces/artifacts/papers.manifest.json",
                    "edges": "collection with spaces/artifacts/metadata/edges.jsonl",
                    "pdf_dir": "collection with spaces/artifacts/pdfs",
                },
                "validation": {"passed": status == "success"},
                "artifacts": [
                    {
                        "type": "paper_set",
                        "schema_version": "3",
                        "path": "collection with spaces/artifacts/papers.manifest.json",
                    },
                    {
                        "type": "paper_edges",
                        "schema_version": "1",
                        "path": "collection with spaces/artifacts/metadata/edges.jsonl",
                    },
                ],
            },
        )
        return self.collection

    def init(self, directory: Path) -> None:
        invocation = (
            f'"{directory}" --mineru-backend pipeline --mineru-workers 1 --llm-workers 2 '
            "--min-papers 2 --min-nodes 4 --min-edges 3 "
            "--min-mineru-coverage 1 --min-structure-success 1 "
            "--min-extraction-success 1 --min-core-coverage 1 "
            "--min-evidence-coverage 1 --desired-baseline-coverage 0 "
            "--desired-dataset-coverage 0"
        )
        self.command(
            "init",
            "--arguments",
            invocation,
            "--run-id",
            "test-run",
            "--run-dir",
            str(self.run_dir),
            "--cwd",
            str(self.root),
        )

    def test_end_to_end_mineru_papergraph_and_graph(self) -> None:
        self.init(self.make_collection())
        result = self.command("run", "--run-dir", str(self.run_dir))
        report = json.loads(
            (self.run_dir / "artifacts" / "graph_report.json").read_text()
        )
        run_manifest = json.loads((self.run_dir / "manifest.json").read_text())
        self.assertTrue(report["passed"], result.stdout)
        self.assertEqual(run_manifest["skill_version"], "3.0.0")
        self.assertEqual(report["counts"]["parsed_papers"], 2)
        self.assertEqual(report["counts"]["structured_papers"], 2)
        self.assertEqual(report["counts"]["extracted_papers"], 2)
        self.assertGreaterEqual(report["counts"]["nodes"], 4)
        self.assertGreaterEqual(report["counts"]["edges"], 3)
        documents = json.loads(
            (self.run_dir / "artifacts" / "paper_documents.json").read_text()
        )
        self.assertEqual(documents["prior_manifest_path"], str(self.collection / "manifest.json"))
        self.assertRegex(documents["prior_manifest_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(documents["paper_set_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(documents["paper_edges_sha256"], r"^[0-9a-f]{64}$")
        for document in documents["documents"]:
            self.assertEqual(document["parser"], "mineru")
            self.assertEqual(document["source_sha256"], document["source_download_sha256"])
            self.assertEqual(document["source_manifest_sha256"], documents["paper_set_sha256"])
            self.assertEqual(document["paper_collect_manifest_sha256"], documents["prior_manifest_sha256"])
            self.assertTrue(Path(document["markdown_path"]).is_file())
            self.assertTrue(Path(document["content_list_path"]).is_file())
            self.assertTrue(Path(document["middle_json_path"]).is_file())
        self.assertEqual(len(LlmHandler.calls), 6)
        self.assertTrue(
            all(call["path"] == "/v1/chat/completions" for call in LlmHandler.calls)
        )
        self.assertTrue(
            all(
                call["authorization"] == "Bearer test-secret-key"
                for call in LlmHandler.calls
            )
        )
        artifact_text = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in (self.run_dir / "artifacts").rglob("*")
            if path.is_file() and path.suffix in {".json", ".jsonl", ".log"}
        )
        self.assertNotIn("test-secret-key", artifact_text)
        nodes = [
            json.loads(line)
            for line in (self.run_dir / "artifacts" / "nodes.jsonl").read_text().splitlines()
            if line.strip()
        ]
        edges = [
            json.loads(line)
            for line in (self.run_dir / "artifacts" / "edges.jsonl").read_text().splitlines()
            if line.strip()
        ]
        graphmixer_core = next(
            node
            for node in nodes
            if node["node_type"] == "Core" and node["name"] == "GraphMixer"
        )
        self.assertFalse(
            any(
                node["node_type"] == "Baseline" and node["name"] == "GraphMixer"
                for node in nodes
            )
        )
        self.assertTrue(
            any(
                edge["relation"] == "baseline_comparison"
                and edge["target"] == graphmixer_core["id"]
                and edge["edge_type"] == "baseline_comparison"
                for edge in edges
            )
        )
        self.assertTrue((self.run_dir / "artifacts" / "visualizations" / "overview.html").is_file())
        with sqlite3.connect(self.run_dir / "artifacts" / "graph.db") as database:
            relations = {
                row[0]
                for row in database.execute("SELECT DISTINCT relation FROM edges")
            }
            edge_types = {
                row[0]
                for row in database.execute("SELECT DISTINCT edge_type FROM edges")
            }
            columns = {
                row[1]
                for row in database.execute("PRAGMA table_info(nodes)")
            }
            fts_hit = database.execute(
                "SELECT id FROM node_fts WHERE node_fts MATCH ? LIMIT 1",
                ("GraphMixer",),
            ).fetchone()
        self.assertIn("introduces", relations)
        self.assertIn("evaluated_on", relations)
        self.assertIn("cites", relations)
        self.assertEqual(relations, edge_types)
        self.assertIn("full_name", columns)
        self.assertIn("paper_title", columns)
        self.assertIn("citation_paperId", columns)
        self.assertIsNotNone(fts_hit)

    def test_step1_enriches_references_from_semantic_scholar(self) -> None:
        SemanticScholarHandler.calls = []
        s2_server = ThreadingHTTPServer(("127.0.0.1", 0), SemanticScholarHandler)
        s2_thread = threading.Thread(target=s2_server.serve_forever, daemon=True)
        s2_thread.start()
        try:
            self.environment.pop("KNOWLEDGE_GRAPH_DISABLE_SEMANTIC_SCHOLAR", None)
            self.environment["KNOWLEDGE_GRAPH_SEMANTIC_SCHOLAR_API_URL"] = (
                f"http://127.0.0.1:{s2_server.server_port}/graph/v1"
            )
            self.init(self.make_collection())
            self.command("mineru", "--run-dir", str(self.run_dir))
            self.command("structure", "--run-dir", str(self.run_dir))
            structures = json.loads(
                (self.run_dir / "artifacts" / "paper_structures.manifest.json").read_text()
            )
            self.assertGreaterEqual(
                structures["references"]["semantic_scholar_documents"], 1
            )
            graphmixer_path = next(
                Path(record["path"])
                for record in structures["structures"]
                if record["paper_id"] == "paper-a"
            )
            graphmixer = json.loads(graphmixer_path.read_text())
            references = graphmixer["semantic_references"]
            self.assertTrue(
                any(reference["paperId"] == "s2-ref-1" for reference in references)
            )
            self.assertTrue(
                any("/paper/DOI" in value for value in SemanticScholarHandler.calls)
            )
        finally:
            s2_server.shutdown()
            s2_server.server_close()
            s2_thread.join(timeout=5)

    def test_step1_uses_local_references_when_semantic_scholar_is_disabled(self) -> None:
        self.init(self.make_collection(local_references=True))
        self.command("mineru", "--run-dir", str(self.run_dir))
        self.command("structure", "--run-dir", str(self.run_dir))
        structures = json.loads(
            (self.run_dir / "artifacts" / "paper_structures.manifest.json").read_text()
        )
        self.assertGreaterEqual(structures["references"]["local_reference_documents"], 1)
        graphmixer_path = next(
            Path(record["path"])
            for record in structures["structures"]
            if record["paper_id"] == "paper-a"
        )
        graphmixer = json.loads(graphmixer_path.read_text())
        self.assertTrue(
            any(
                reference["reference_source"] == "local_references_section"
                and reference["title"] == "Local Graph Baseline"
                for reference in graphmixer["semantic_references"]
            )
        )

    def test_step2_adds_grounded_regex_candidates(self) -> None:
        self.init(self.make_collection(high_recall_extra=True))
        self.command("run", "--run-dir", str(self.run_dir))
        extraction_manifest = json.loads(
            (self.run_dir / "artifacts" / "extractions.manifest.json").read_text()
        )
        graphmixer_path = next(
            Path(record["path"])
            for record in extraction_manifest["extractions"]
            if record["paper_id"] == "paper-a"
        )
        graphmixer = json.loads(graphmixer_path.read_text())
        baselines = graphmixer["graph_data"]["baselines"]
        datasets = graphmixer["graph_data"]["datasets"]
        self.assertTrue(any(item["name"] == "GCN" for item in baselines))
        self.assertTrue(any(item["name"] == "HiddenNet" for item in baselines))
        self.assertTrue(any(item["name"] == "Cora" for item in datasets))
        self.assertTrue(any(item["name"] == "PubMed" for item in datasets))
        self.assertRegex(
            graphmixer["provenance"]["candidate_hints_sha256"], r"^[0-9a-f]{64}$"
        )

    def test_numbered_references_heading_and_quoted_title(self) -> None:
        collection = self.make_collection(local_references=True)
        pdf = collection / "artifacts/pdfs/paper-a.pdf"
        content = pdf.read_text().replace("# References", "## 8. REFERENCES").replace(
            "Jane Doe. Local Graph Baseline. NeurIPS 2020.",
            "Jane Doe, “Local Graph Baseline,” NeurIPS 2020.",
        )
        pdf.write_text(content)
        paper_set_path = collection / "artifacts/papers.manifest.json"
        paper_set = json.loads(paper_set_path.read_text())
        paper_set["papers"][0]["download"]["sha256"] = hashlib.sha256(pdf.read_bytes()).hexdigest()
        write_json(paper_set_path, paper_set)
        self.init(collection)
        self.command("mineru", "--run-dir", str(self.run_dir))
        self.command("structure", "--run-dir", str(self.run_dir))
        structures = json.loads((self.run_dir / "artifacts/paper_structures.manifest.json").read_text())
        first = next(record for record in structures["structures"] if record["paper_id"] == "paper-a")
        references = json.loads(Path(first["path"]).read_text())["semantic_references"]
        self.assertTrue(any(reference["title"] == "Local Graph Baseline" and reference["year"] == 2020 for reference in references))

    def test_stage_resume_reuses_all_checkpoints(self) -> None:
        self.init(self.make_collection())
        self.command("mineru", "--run-dir", str(self.run_dir))
        mineru_result = json.loads(
            self.command("mineru", "--run-dir", str(self.run_dir)).stdout
        )
        self.assertEqual(mineru_result["reused"], 2)
        self.command("structure", "--run-dir", str(self.run_dir))
        structure_result = json.loads(
            self.command("structure", "--run-dir", str(self.run_dir)).stdout
        )
        self.assertEqual(structure_result["reused"], 2)
        self.command("extract", "--run-dir", str(self.run_dir))
        extraction_paths = sorted(
            (self.run_dir / "artifacts" / "extractions").glob("*.json")
        )
        timestamps = {path: path.stat().st_mtime_ns for path in extraction_paths}
        call_count = len(LlmHandler.calls)
        time.sleep(0.01)
        extraction_result = json.loads(
            self.command("extract", "--run-dir", str(self.run_dir)).stdout
        )
        self.assertEqual(extraction_result["reused"], 2)
        self.assertEqual(call_count, len(LlmHandler.calls))
        self.assertEqual(
            timestamps, {path: path.stat().st_mtime_ns for path in extraction_paths}
        )

    def test_failed_batch_is_bisected_into_isolated_retries(self) -> None:
        self.init(self.make_collection(poison_batch=True))
        result = json.loads(
            self.command("mineru", "--run-dir", str(self.run_dir)).stdout
        )
        self.assertEqual(result["parsed"], 2)
        self.assertEqual(result["failed"], 0)
        recovery_logs = list(
            (self.run_dir / "artifacts" / "logs").glob("mineru-recovery-*.log")
        )
        self.assertGreaterEqual(len(recovery_logs), 3)

    def test_shared_cache_reuses_pdf_bundles_across_runs(self) -> None:
        collection = self.make_collection()
        self.init(collection)
        self.command("mineru", "--run-dir", str(self.run_dir))
        self.run_dir = self.root / "second run"
        self.init(collection)
        result = json.loads(self.command("mineru", "--run-dir", str(self.run_dir)).stdout)
        self.assertEqual(result["reused"], 2)
        self.assertEqual(result["parsed"], 2)
        self.assertEqual(list((self.run_dir / "artifacts" / "mineru").rglob("*.md")), [])
        self.command("structure", "--run-dir", str(self.run_dir))
        self.command("extract", "--run-dir", str(self.run_dir))
        self.command("build", "--run-dir", str(self.run_dir))
        self.command("audit", "--run-dir", str(self.run_dir))

    def test_damaged_shared_cache_is_reparsed(self) -> None:
        collection = self.make_collection()
        self.init(collection)
        self.command("mineru", "--run-dir", str(self.run_dir))
        documents = json.loads((self.run_dir / "artifacts" / "paper_documents.json").read_text())
        Path(documents["documents"][0]["markdown_path"]).write_text("damaged cache")
        self.run_dir = self.root / "second run"
        self.init(collection)
        result = json.loads(self.command("mineru", "--run-dir", str(self.run_dir)).stdout)
        self.assertEqual(result["reused"], 1)
        self.assertEqual(result["parsed"], 2)

    def test_batches_checkpoint_all_papers(self) -> None:
        self.init(self.make_collection())
        request_path = self.run_dir / "artifacts" / "request.json"
        request = json.loads(request_path.read_text())
        request["mineru_batch_size"] = 1
        write_json(request_path, request)
        result = json.loads(self.command("mineru", "--run-dir", str(self.run_dir)).stdout)
        self.assertEqual(result["pending"], 0)
        events = [json.loads(line) for line in (self.run_dir / "artifacts" / "logs" / "pipeline.jsonl").read_text().splitlines()]
        self.assertEqual([event["summary"]["parsed"] for event in events], [1, 2])

    def test_llm_outage_preserves_pending_papers_for_resume(self) -> None:
        self.init(self.make_collection())
        self.command("mineru", "--run-dir", str(self.run_dir))
        self.command("structure", "--run-dir", str(self.run_dir))
        request_path = self.run_dir / "artifacts" / "request.json"
        request = json.loads(request_path.read_text())
        request["llm_retries"] = 0
        write_json(request_path, request)
        LlmHandler.unavailable = True
        result = self.command("extract", "--run-dir", str(self.run_dir), expected=1)
        self.assertIn("remaining papers left pending", result.stderr)
        manifest = json.loads((self.run_dir / "artifacts" / "extractions.manifest.json").read_text())
        self.assertEqual(manifest["summary"]["pending"], 2)
        self.assertEqual(manifest["summary"]["failed"], 0)
        LlmHandler.unavailable = False
        result = json.loads(self.command("extract", "--run-dir", str(self.run_dir)).stdout)
        self.assertEqual(result["extracted"], 2)
        self.assertEqual(result["pending"], 0)

    def test_cache_invalidates_when_parse_configuration_changes(self) -> None:
        self.init(self.make_collection())
        self.command("mineru", "--run-dir", str(self.run_dir))
        self.environment["MINERU_TABLE_ENABLE"] = "false"
        result = json.loads(self.command("mineru", "--run-dir", str(self.run_dir)).stdout)
        self.assertEqual(result["reused"], 0)
        self.assertEqual(result["parsed"], 2)

    def test_metadata_only_papers_remain_in_final_graph(self) -> None:
        collection = self.make_collection()
        paper_set_path = collection / "artifacts/papers.manifest.json"
        paper_set = json.loads(paper_set_path.read_text())
        paper_set["papers"].append({"id": "paper-no-pdf", "title": "Unavailable full text", "download": {"status": "unavailable"}})
        write_json(paper_set_path, paper_set)
        self.init(collection)
        self.command("run", "--run-dir", str(self.run_dir))
        graph = json.loads((self.run_dir / "artifacts/method_graph.json").read_text())
        paper_nodes = [node for node in graph["nodes"] if node["node_type"] == "Paper"]
        self.assertEqual(len(paper_nodes), 3)
        self.assertFalse(any(node["paper_id"] == "paper-no-pdf" and node["node_type"] != "Paper" for node in graph["nodes"]))

    def test_pending_extraction_blocks_successful_audit(self) -> None:
        self.init(self.make_collection())
        self.command("run", "--run-dir", str(self.run_dir))
        extraction_path = self.run_dir / "artifacts/extractions.manifest.json"
        extractions = json.loads(extraction_path.read_text())
        extractions["extractions"].append({"paper_id": "still-processing", "path": None, "status": "pending"})
        write_json(extraction_path, extractions)
        self.command("audit", "--run-dir", str(self.run_dir), expected=2)
        report = json.loads((self.run_dir / "artifacts/graph_report.json").read_text())
        self.assertIn("LLM extraction still has pending papers.", report["blocking_errors"])

    def test_thinking_mode_is_sent_and_invalidates_extraction_cache(self) -> None:
        self.init(self.make_collection())
        self.command("run", "--run-dir", str(self.run_dir))
        previous_calls = len(LlmHandler.calls)
        request_path = self.run_dir / "artifacts/request.json"
        request = json.loads(request_path.read_text())
        request["llm_disable_thinking"] = True
        write_json(request_path, request)
        result = json.loads(self.command("extract", "--run-dir", str(self.run_dir)).stdout)
        self.assertEqual(result["reused"], 0)
        self.assertEqual(len(LlmHandler.calls) - previous_calls, 6)
        self.assertTrue(all(call["enable_thinking"] is False for call in LlmHandler.calls[previous_calls:]))

    def test_endpoint_handoff_preserves_completed_records_but_restarts_partial_passes(self) -> None:
        self.init(self.make_collection())
        self.command("run", "--run-dir", str(self.run_dir))
        paths = sorted((self.run_dir / "artifacts/extractions").glob("*.json"))
        retained, partial = paths
        original = retained.read_bytes()
        old_endpoint = json.loads(original)["quality"]["endpoint"]
        partial.unlink()
        target = self.environment["KNOWLEDGE_GRAPH_LLM_API_URL"] + "/migrated"
        result = json.loads(self.command("migrate-endpoint", "--run-dir", str(self.run_dir), "--api-url", target, "--model", "test-model").stdout)
        self.assertEqual(result["retained_completed"], 1)
        self.environment["KNOWLEDGE_GRAPH_LLM_API_URL"] = target
        calls = len(LlmHandler.calls)
        result = json.loads(self.command("extract", "--run-dir", str(self.run_dir)).stdout)
        self.assertEqual(result["reused"], 1)
        self.assertEqual(len(LlmHandler.calls) - calls, 3)
        self.assertEqual(retained.read_bytes(), original)
        self.assertEqual(json.loads(retained.read_text())["quality"]["endpoint"], old_endpoint)
        self.assertEqual(json.loads(partial.read_text())["quality"]["endpoint"], target + "/chat/completions")
        self.command("build", "--run-dir", str(self.run_dir))
        self.command("audit", "--run-dir", str(self.run_dir))

    def test_endpoint_handoff_rejects_modified_retained_record(self) -> None:
        self.init(self.make_collection())
        self.command("run", "--run-dir", str(self.run_dir))
        target = self.environment["KNOWLEDGE_GRAPH_LLM_API_URL"] + "/migrated"
        self.command("migrate-endpoint", "--run-dir", str(self.run_dir), "--api-url", target, "--model", "test-model")
        path = next((self.run_dir / "artifacts/extractions").glob("*.json"))
        value = json.loads(path.read_text())
        value["quality"]["rejected_items"] += 1
        write_json(path, value)
        self.environment["KNOWLEDGE_GRAPH_LLM_API_URL"] = target
        calls = len(LlmHandler.calls)
        result = json.loads(self.command("extract", "--run-dir", str(self.run_dir)).stdout)
        self.assertEqual(result["reused"], 1)
        self.assertEqual(len(LlmHandler.calls) - calls, 3)

    def test_endpoint_handoff_rejects_model_change(self) -> None:
        self.init(self.make_collection())
        result = self.command("migrate-endpoint", "--run-dir", str(self.run_dir), "--api-url", "http://localhost:9999/v1", "--model", "other-model", expected=1)
        self.assertIn("same configured model", result.stderr)

    def test_init_rejects_incomplete_upstream_run(self) -> None:
        directory = self.make_collection(status="incomplete")
        invocation = f'"{directory}" --mineru-backend pipeline'
        result = self.command(
            "init",
            "--arguments",
            invocation,
            "--run-dir",
            str(self.run_dir),
            "--cwd",
            str(self.root),
            expected=1,
        )
        self.assertIn("not complete", result.stderr)

    def test_llm_worker_capacity_supports_128_and_rejects_larger_counts(self) -> None:
        directory = self.make_collection()
        arguments = ["init", "--input-directory", str(directory), "--run-id", "test-run",
                     "--run-dir", str(self.run_dir), "--cwd", str(self.root),
                     "--mineru-backend", "pipeline", "--mineru-workers", "1", "--llm-workers"]
        for workers in (64, 96, 128):
            self.command(*arguments, str(workers))
        self.command("run", "--run-dir", str(self.run_dir))
        report = json.loads((self.run_dir / "artifacts/graph_report.json").read_text())
        self.assertTrue(report["passed"])
        result = self.command(*arguments, "129", expected=1)
        self.assertIn("--llm-workers must be between 1 and 128", result.stderr)
        request = json.loads((self.run_dir / "artifacts/request.json").read_text())
        self.assertEqual(request["llm_workers"], 128)

    def test_init_requires_successful_prior_manifest(self) -> None:
        directory = self.make_collection()
        (directory / "manifest.json").unlink()
        result = self.command(
            "init",
            "--arguments",
            f'"{directory}" --mineru-backend pipeline',
            "--run-dir",
            str(self.run_dir),
            "--cwd",
            str(self.root),
            expected=1,
        )
        self.assertIn("manifest.json", result.stderr)

    def test_init_rejects_stale_pdf_sha256(self) -> None:
        directory = self.make_collection()
        manifest_path = directory / "artifacts" / "papers.manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["papers"][0]["download"]["sha256"] = "0" * 64
        write_json(manifest_path, manifest)
        result = self.command(
            "init",
            "--arguments",
            f'"{directory}" --mineru-backend pipeline',
            "--run-dir",
            str(self.run_dir),
            "--cwd",
            str(self.root),
            expected=1,
        )
        self.assertIn("sha256", result.stderr)

    def test_init_rejects_noncanonical_pdf_path(self) -> None:
        directory = self.make_collection()
        outside_pdf = self.root / "outside.pdf"
        outside_pdf.write_text("%PDF-FAKE\noutside\n", encoding="utf-8")
        manifest_path = directory / "artifacts" / "papers.manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["papers"][0]["download"]["path"] = str(outside_pdf)
        manifest["papers"][0]["download"]["sha256"] = hashlib.sha256(
            outside_pdf.read_bytes()
        ).hexdigest()
        write_json(manifest_path, manifest)
        result = self.command(
            "init",
            "--arguments",
            f'"{directory}" --mineru-backend pipeline',
            "--run-dir",
            str(self.run_dir),
            "--cwd",
            str(self.root),
            expected=1,
        )
        self.assertIn("safe relative path", result.stderr)

    def test_audit_rejects_upstream_hash_drift(self) -> None:
        self.init(self.make_collection())
        self.command("run", "--run-dir", str(self.run_dir))
        (self.collection / "artifacts" / "metadata" / "edges.jsonl").write_text(
            "", encoding="utf-8"
        )
        self.command("audit", "--run-dir", str(self.run_dir), expected=2)
        report = json.loads(
            (self.run_dir / "artifacts" / "graph_report.json").read_text()
        )
        self.assertFalse(report["passed"])
        self.assertTrue(
            any("edge file" in value for value in report["blocking_errors"])
        )

    def test_audit_rejects_invalid_jsonl_endpoint(self) -> None:
        self.init(self.make_collection())
        self.command("run", "--run-dir", str(self.run_dir))
        edges_path = self.run_dir / "artifacts" / "edges.jsonl"
        with edges_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "id": "edge:broken",
                        "source": "missing",
                        "target": "also-missing",
                        "relation": "cites",
                        "evidence_paper_id": "paper-a",
                        "evidence": {},
                        "provenance": {},
                    }
                )
                + "\n"
            )
        self.command("audit", "--run-dir", str(self.run_dir), expected=2)
        report = json.loads(
            (self.run_dir / "artifacts" / "graph_report.json").read_text()
        )
        self.assertFalse(report["passed"])
        self.assertTrue(
            any("missing endpoints" in value for value in report["blocking_errors"])
        )


if __name__ == "__main__":
    unittest.main()
