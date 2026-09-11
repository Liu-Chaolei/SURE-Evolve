from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1]
SCRIPT = SKILL_DIR / "scripts" / "run_survey_phase.py"
FIXTURE = SKILL_DIR / "examples" / "paper-collect-manifest.example.json"
PROJECT_ROOT = SKILL_DIR.parents[2]
FAKE_LLM_ENV = {
    "OPENAI_API_KEY": "test-openai-key",
    "SEMANTIC_SCHOLAR_API_KEY": "test-semantic-key",
    "XLAB_LITERATURE_SURVEY_FAKE_LLM": "1",
    "XLAB_LITERATURE_SURVEY_ENABLE_LLM_REFINEMENT": "1",
}


def write_graph_fixture(base_dir: Path) -> Path:
    graph_dir = base_dir / "knowledge-graph-run"
    artifacts = graph_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    method_graph = {
        "schema_version": "xlab.method_graph.v2",
        "nodes": [
            {
                "id": "paper:kg-1",
                "node_type": "Paper",
                "name": "Retrieval-Augmented Scientific Agents",
                "paper_id": "kg-1",
                "aliases": [],
                "provenance": {"paper_id": "kg-1"},
                "metadata": {
                    "title": "Retrieval-Augmented Scientific Agents",
                    "abstract": "Retrieval augmented generation grounds scientific agents in indexed paper evidence and tool traces.",
                    "year": 2024,
                    "venue": "XLab Conference",
                    "authors": ["Example Author"],
                    "url": "https://example.invalid/kg-1",
                },
            },
            {
                "id": "paper:kg-2",
                "node_type": "Paper",
                "name": "Evaluating Literature Review Agents",
                "paper_id": "kg-2",
                "aliases": [],
                "provenance": {"paper_id": "kg-2"},
                "metadata": {
                    "title": "Evaluating Literature Review Agents",
                    "abstract": "Evaluation of scientific literature agents benefits from citation traceability and reproducible evidence graphs.",
                    "year": 2025,
                    "venue": "Agent Evaluation Workshop",
                    "authors": ["Example Reviewer"],
                    "url": "https://example.invalid/kg-2",
                },
            },
            {
                "id": "paper:kg-3",
                "node_type": "Paper",
                "name": "Knowledge Graph Context for Research Planning",
                "paper_id": "kg-3",
                "aliases": [],
                "provenance": {"paper_id": "kg-3"},
                "metadata": {
                    "title": "Knowledge Graph Context for Research Planning",
                    "abstract": "Method graphs connect paper nodes, core concepts, and relations so downstream research workflows can reuse structured context.",
                    "year": 2023,
                    "venue": "Graph Systems",
                    "authors": ["Example Graph"],
                    "url": "https://example.invalid/kg-3",
                },
            },
            {
                "id": "core:retrieval",
                "node_type": "Core",
                "name": "retrieval grounding",
                "paper_id": "kg-1",
                "aliases": ["RAG"],
                "provenance": {"paper_id": "kg-1"},
            },
        ],
        "edges": [
            {"source": "core:retrieval", "target": "paper:kg-1", "relation": "supports"},
            {"source": "paper:kg-2", "target": "paper:kg-3", "relation": "extends"},
        ],
        "counts": {"nodes": 4, "edges": 2, "papers": 3},
    }
    (artifacts / "method_graph.json").write_text(json.dumps(method_graph, indent=2) + "\n", encoding="utf-8")
    (graph_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "2",
                "run_id": "kg-run",
                "skill_name": "knowledge_graph",
                "skill_version": "3.0.0",
                "status": "success",
                "created_at": "2026-01-01T00:00:00Z",
                "inputs": {},
                "outputs": {},
                "validation": {"passed": True},
                "artifacts": [
                    {
                        "type": "method_graph",
                        "schema_version": "xlab.method_graph.v2",
                        "path": str(artifacts / "method_graph.json"),
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return graph_dir


class LiteratureSurveyRuntimeTest(unittest.TestCase):
    def run_cli(
        self,
        *args: str,
        expected_status: int | tuple[int, ...] = 0,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.pop("LLM_MODEL", None)
        env.pop("LLM_BASE_URL", None)
        env.pop("LLM_API_BASE", None)
        env.pop("LLM_CONTEXT_WINDOW", None)
        env.pop("XLAB_LITERATURE_SURVEY_MODEL", None)
        env.pop("XLAB_LITERATURE_SURVEY_LLM_BASE_URL", None)
        env.pop("OPENAI_API_KEY", None)
        env.pop("LLM_API_KEY", None)
        env.pop("SEMANTIC_SCHOLAR_API_KEY", None)
        env.pop("S2_API_KEY", None)
        env.pop("XLAB_LITERATURE_SURVEY_FAKE_LLM", None)
        env.pop("XLAB_LITERATURE_SURVEY_ENABLE_LLM_REFINEMENT", None)
        if extra_env:
            env.update(extra_env)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=SKILL_DIR,
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
        expected = (expected_status,) if isinstance(expected_status, int) else expected_status
        if result.returncode not in expected:
            self.fail(
                f"Expected exit status {expected}, got {result.returncode}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        return result

    def test_smoke_generates_v2_manifest_and_traceable_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.run_cli(
                "smoke",
                "--run-id",
                "test-run",
                "--run-dir",
                str(run_dir),
                "--input",
                str(FIXTURE),
                "--topic",
                "Retrieval augmented generation for scientific agents",
                "--cwd",
                str(PROJECT_ROOT),
                extra_env=FAKE_LLM_ENV,
            )

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            survey = json.loads((run_dir / "artifacts" / "survey.json").read_text(encoding="utf-8"))
            citations = json.loads((run_dir / "artifacts" / "citations.json").read_text(encoding="utf-8"))
            report = json.loads((run_dir / "artifacts" / "survey_report.json").read_text(encoding="utf-8"))

            self.assertEqual(manifest["schema_version"], "2")
            self.assertEqual(manifest["skill_name"], "literature_survey")
            self.assertEqual(manifest["skill_version"], "1.0.0")
            self.assertEqual(manifest["status"], "success")
            self.assertTrue(report["passed"], report)
            self.assertEqual(survey["schema_version"], "xlab.literature_survey.v1")
            self.assertEqual(citations["schema_version"], "xlab.citation_trace.v1")
            self.assertGreaterEqual(survey["paper_count"], 3)
            artifact_types = {artifact["type"] for artifact in manifest["artifacts"]}
            self.assertEqual(
                artifact_types,
                {"literature_survey_document", "literature_survey_json", "citation_trace", "survey_report"},
            )
            reference_ids = {reference["paper_id"] for reference in citations["references"]}
            used_ids = {paper_id for trace in citations["traces"] for paper_id in trace["paper_ids"]}
            self.assertTrue(used_ids)
            self.assertTrue(used_ids.issubset(reference_ids))

    def test_graph_fixture_generates_successful_survey(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base_dir = Path(temporary)
            graph_dir = write_graph_fixture(base_dir)
            run_dir = base_dir / "run"

            self.run_cli(
                "smoke",
                "--run-id",
                "graph-run",
                "--run-dir",
                str(run_dir),
                "--graph",
                str(graph_dir),
                "--topic",
                "Retrieval augmented generation for scientific agents",
                "--cwd",
                str(PROJECT_ROOT),
                extra_env=FAKE_LLM_ENV,
            )

            survey = json.loads((run_dir / "artifacts" / "survey.json").read_text(encoding="utf-8"))
            report = json.loads((run_dir / "artifacts" / "survey_report.json").read_text(encoding="utf-8"))
            references = survey["references"]

            self.assertTrue(report["passed"], report)
            self.assertEqual(report["source"]["source_type"], "knowledge_graph")
            self.assertEqual(survey["source"]["source_type"], "knowledge_graph")
            self.assertTrue(survey["graph_context"]["available"])
            self.assertIn("retrieval grounding", survey["graph_context"]["core_terms"])
            self.assertEqual({reference["paper_id"] for reference in references}, {"kg-1", "kg-2", "kg-3"})
            self.assertEqual({reference["source"] for reference in references}, {"knowledge_graph"})

    def test_committed_graph_artifact_is_preserved_as_parent_lineage(self) -> None:
        from xlab.skills.literature_survey.scripts.literature_survey_lib.paper_sources import (
            xlab_file_artifact_digest,
        )

        with tempfile.TemporaryDirectory() as temporary:
            base_dir = Path(temporary)
            graph_dir = write_graph_fixture(base_dir)
            graph_artifact = graph_dir / "artifacts" / "method_graph.json"
            run_dir = base_dir / "run"
            self.run_cli(
                "smoke",
                "--run-id",
                "lineage-run",
                "--run-dir",
                str(run_dir),
                "--graph",
                str(graph_dir),
                "--topic",
                "Retrieval augmented generation for scientific agents",
                "--cwd",
                str(PROJECT_ROOT),
                extra_env=FAKE_LLM_ENV,
            )

            expected_parent = f"sha256:{xlab_file_artifact_digest(graph_artifact)}"
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["artifacts"])
            for artifact in manifest["artifacts"]:
                self.assertEqual(artifact["parents"], [expected_parent])

    def test_plain_local_input_does_not_invent_parent_artifact_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base_dir = Path(temporary)
            input_path = base_dir / "papers.json"
            input_path.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
            run_dir = base_dir / "run"
            self.run_cli(
                "smoke",
                "--run-id",
                "local-input-run",
                "--run-dir",
                str(run_dir),
                "--input",
                str(input_path),
                "--topic",
                "Retrieval augmented generation for scientific agents",
                "--cwd",
                str(PROJECT_ROOT),
                extra_env=FAKE_LLM_ENV,
            )

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["artifacts"])
            self.assertTrue(all("parents" not in artifact for artifact in manifest["artifacts"]))

    def test_survey_agent_path_uses_integrated_engine_and_writes_agent_state(self) -> None:
        from xlab.skills.literature_survey.scripts.literature_survey_lib.config import RuntimeConfig
        from xlab.skills.literature_survey.scripts.literature_survey_lib.inputs import parse_request_args
        from xlab.skills.literature_survey.scripts.literature_survey_lib.pipeline import run_pipeline
        from xlab.skills.literature_survey.scripts.literature_survey_lib.survey_agent import AGENT_SCHEMA_VERSION

        with tempfile.TemporaryDirectory() as temporary:
            base_dir = Path(temporary)
            graph_dir = write_graph_fixture(base_dir)
            run_dir = base_dir / "run"
            runtime = RuntimeConfig(
                model="fake-survey-agent",
                llm_base_url="https://example.invalid/v1",
                llm_api_key_set=True,
                semantic_scholar_api_key_set=True,
                llm_context_window=512000,
                max_workers=1,
                request_timeout_seconds=10,
                max_retries=0,
                default_max_papers=24,
                max_papers_limit=200,
                default_min_papers=3,
                default_language="en",
                default_depth="standard",
                cluster_limit=5,
                inline_citation_limit=8,
                evidence_chars=500,
                graph_context_limit=24,
                allow_synthetic=False,
                llm_refinement_enabled=True,
                full_text_enabled=False,
            )
            request = parse_request_args(
                f'"Retrieval augmented generation for scientific agents" --graph "{graph_dir}"',
                PROJECT_ROOT,
                runtime,
            )
            original_env = {key: os.environ.get(key) for key in FAKE_LLM_ENV}
            try:
                os.environ.update(FAKE_LLM_ENV)
                result = run_pipeline(cwd=PROJECT_ROOT, run_dir=run_dir, run_id="agent-run", request=request, runtime=runtime)
            finally:
                for key, value in original_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

            self.assertEqual(result["status"], "success", result["report"])
            agent_state = json.loads((run_dir / "state" / "survey_agent_result.json").read_text(encoding="utf-8"))
            survey = json.loads((run_dir / "artifacts" / "survey.json").read_text(encoding="utf-8"))
            markdown = (run_dir / "artifacts" / "survey.md").read_text(encoding="utf-8")
            self.assertEqual(agent_state["schema_version"], AGENT_SCHEMA_VERSION)
            self.assertEqual(agent_state["engine"]["mode"], "integrated_xcientist_survey_agent")
            self.assertEqual(survey["runtime"]["mode"], "survey-agent")
            self.assertEqual(survey["runtime"]["survey_agent_engine"], "integrated_xcientist_survey_agent")
            self.assertIn("[paper:kg-1]", markdown)
            self.assertIn("survey_agent", result["report"]["stages"])
            xcientist_state = run_dir / "state" / "xcientist"
            for name in (
                "seed_papers.json",
                "expanded_papers.json",
                "collected_papers.json",
                "keynotes.json",
                "clustering_result.json",
                "analysis_context.json",
                "outline.raw.json",
                "draft.raw.json",
                "draft.raw.md",
                "refined.raw.md",
                "references.raw.json",
                "evaluation.json",
                "engine_result.json",
            ):
                self.assertTrue((xcientist_state / name).exists(), name)

    def test_runtime_config_uses_pyagent_api_contract_without_persisting_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.run_cli(
                "init",
                "--run-id",
                "api-run",
                "--run-dir",
                str(run_dir),
                "--arguments",
                f'"Retrieval augmented generation for scientific agents" --input "{FIXTURE}"',
                "--cwd",
                str(PROJECT_ROOT),
            )
            runtime = json.loads((run_dir / "state" / "runtime_config.json").read_text(encoding="utf-8"))
            request = json.loads((run_dir / "state" / "request.json").read_text(encoding="utf-8"))

            self.assertEqual(runtime["model"], "MiniMax-M3")
            self.assertEqual(runtime["llm_base_url"], "https://api.minimaxi.com/v1")
            self.assertEqual(runtime["llm_context_window"], 512000)
            self.assertNotIn("survey_agent_profile", runtime)
            self.assertNotIn("use_llm", request)
            self.assertIn("llm_api_key_set", runtime)
            self.assertIn("semantic_scholar_api_key_set", runtime)
            self.assertFalse(any(key == "api_key" or key.endswith("_api_key") for key in runtime))
            self.assertIn("OPENAI_API_KEY", " ".join(request["compatibility_warnings"]))

    def test_runtime_config_accepts_llm_and_s2_secret_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.run_cli(
                "init",
                "--run-id",
                "alias-run",
                "--run-dir",
                str(run_dir),
                "--arguments",
                f'"Retrieval augmented generation for scientific agents" --input "{FIXTURE}"',
                "--cwd",
                str(PROJECT_ROOT),
                extra_env={
                    "LLM_API_KEY": "llm-alias-secret",
                    "S2_API_KEY": "s2-alias-secret",
                },
            )
            runtime_text = (run_dir / "state" / "runtime_config.json").read_text(encoding="utf-8")
            runtime = json.loads(runtime_text)

            self.assertTrue(runtime["llm_api_key_set"])
            self.assertTrue(runtime["semantic_scholar_api_key_set"])
            self.assertNotIn("llm-alias-secret", runtime_text)
            self.assertNotIn("s2-alias-secret", runtime_text)

    def test_integrated_survey_config_is_run_local_and_secret_safe(self) -> None:
        from omegaconf import OmegaConf
        from xlab.skills.literature_survey.scripts.literature_survey_lib.xcientist.config import (
            load_survey_agent_config,
        )

        with tempfile.TemporaryDirectory() as temporary:
            base_dir = Path(temporary)
            run_dir = base_dir / "run"
            graph_db = base_dir / "graph.db"
            with sqlite3.connect(graph_db) as connection:
                connection.execute("create table Core (paper_id text, title text, abstract text)")
                connection.commit()

            original_openai_key = os.environ.pop("OPENAI_API_KEY", None)
            original_semantic_key = os.environ.pop("SEMANTIC_SCHOLAR_API_KEY", None)
            try:
                config = load_survey_agent_config(
                    run_dir=run_dir,
                    topic="Physics-informed neural networks",
                    graph_db=graph_db,
                )
                resolved = OmegaConf.to_container(config, resolve=True)
            finally:
                if original_openai_key is not None:
                    os.environ["OPENAI_API_KEY"] = original_openai_key
                if original_semantic_key is not None:
                    os.environ["SEMANTIC_SCHOLAR_API_KEY"] = original_semantic_key

            run_local_graph_db = run_dir.resolve() / "state" / "xcientist" / "graph.db"
            self.assertEqual(config.BasicInfo.topic, "Physics-informed neural networks")
            self.assertEqual(Path(config.ModuleInfo.PaperGraphRetriever.db_path), run_local_graph_db)
            self.assertTrue(run_local_graph_db.exists())
            self.assertTrue(config.ModuleInfo.WorkCollector.expand_in_local_paper_graph)
            self.assertTrue(config.ModuleInfo.WorkAnalyzer.use_local_paper_graph_keynotes)
            self.assertTrue(config.ModuleInfo.SurveyGenerator.include_relation_graph)
            self.assertTrue(config.ModuleInfo.SurveyGenerator.include_relation_table)
            self.assertTrue(config.ModuleInfo.SurveyGenerator.enable_review_and_revise)
            self.assertTrue(config.ModuleInfo.Judge.rubrics_eval_10_dimensions)
            self.assertTrue(config.ModuleInfo.Judge.citation_eval)
            self.assertTrue(Path(config.BasicInfo.cache_path).is_absolute())
            self.assertTrue(str(Path(config.BasicInfo.cache_path)).startswith(str(run_dir.resolve())))
            self.assertTrue(str(config.BasicInfo.save_path).startswith(str((run_dir / "artifacts").resolve())))

            snapshot_path = run_dir / "state" / "xcientist" / "config.snapshot.yaml"
            snapshot = snapshot_path.read_text(encoding="utf-8")
            self.assertIn("${oc.env:OPENAI_API_KEY,''}", snapshot)
            self.assertIn("${oc.env:SEMANTIC_SCHOLAR_API_KEY,''}", snapshot)
            self.assertNotIn("test-openai-key", snapshot)
            self.assertEqual(resolved["APIInfo"]["llm_api_key"], "")
            self.assertEqual(resolved["APIInfo"]["semantic_scholar_api_key"], "")

            with self.assertRaises(ValueError):
                load_survey_agent_config(
                    run_dir=run_dir / "blocked",
                    topic="Physics-informed neural networks",
                    graph_db=graph_db,
                    overrides={"APIInfo.llm_api_key": "must-not-persist"},
                )

    def test_topic_only_normal_run_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.run_cli(
                "init",
                "--run-id",
                "topic-only",
                "--run-dir",
                str(run_dir),
                "--arguments",
                "Retrieval augmented generation for scientific agents",
                "--cwd",
                str(PROJECT_ROOT),
            )
            self.run_cli(
                "synthesize",
                "--run-id",
                "topic-only",
                "--run-dir",
                str(run_dir),
                "--cwd",
                str(PROJECT_ROOT),
                expected_status=2,
            )

            report = json.loads((run_dir / "artifacts" / "survey_report.json").read_text(encoding="utf-8"))
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

            self.assertFalse(report["passed"])
            self.assertEqual(manifest["status"], "incomplete")
            joined = " ".join(report["blocking_errors"] + report["warnings"])
            self.assertIn("--graph", joined)
            self.assertIn("--input", joined)

    def test_resume_reuses_checkpoint_and_restores_missing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.run_cli(
                "init",
                "--run-id",
                "resume-run",
                "--run-dir",
                str(run_dir),
                "--arguments",
                f'"Retrieval augmented generation for scientific agents" --input "{FIXTURE}"',
                "--cwd",
                str(PROJECT_ROOT),
                extra_env=FAKE_LLM_ENV,
            )
            self.run_cli(
                "synthesize",
                "--run-id",
                "resume-run",
                "--run-dir",
                str(run_dir),
                "--cwd",
                str(PROJECT_ROOT),
                extra_env=FAKE_LLM_ENV,
            )
            survey_markdown = run_dir / "artifacts" / "survey.md"
            survey_markdown.unlink()

            self.run_cli(
                "resume",
                "--run-id",
                "resume-run",
                "--run-dir",
                str(run_dir),
                "--cwd",
                str(PROJECT_ROOT),
                extra_env=FAKE_LLM_ENV,
            )
            report = json.loads((run_dir / "artifacts" / "survey_report.json").read_text(encoding="utf-8"))
            checkpoint = json.loads((run_dir / "state" / "checkpoint.json").read_text(encoding="utf-8"))

            self.assertTrue(survey_markdown.exists())
            self.assertTrue(report["passed"], report)
            self.assertIn("ingest_sources", checkpoint["completed_stages"])
            self.assertIn("render_artifacts", checkpoint["completed_stages"])


    def test_changed_source_file_invalidates_checkpoint_and_prunes_downstream(self) -> None:
        from xlab.skills.literature_survey.scripts.literature_survey_lib.config import RuntimeConfig
        from xlab.skills.literature_survey.scripts.literature_survey_lib.inputs import parse_request_args
        from xlab.skills.literature_survey.scripts.literature_survey_lib.pipeline import run_pipeline

        with tempfile.TemporaryDirectory() as temporary:
            base_dir = Path(temporary)
            manifest_path = base_dir / "papers.json"
            manifest = json.loads(FIXTURE.read_text(encoding="utf-8"))
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            run_dir = base_dir / "run"
            runtime = RuntimeConfig(
                model="fake-survey-agent",
                llm_base_url="https://example.invalid/v1",
                llm_api_key_set=True,
                semantic_scholar_api_key_set=True,
                llm_context_window=512000,
                max_workers=1,
                request_timeout_seconds=10,
                max_retries=0,
                default_max_papers=24,
                max_papers_limit=200,
                default_min_papers=3,
                default_language="en",
                default_depth="standard",
                cluster_limit=5,
                inline_citation_limit=8,
                evidence_chars=500,
                graph_context_limit=24,
                allow_synthetic=False,
                llm_refinement_enabled=True,
                full_text_enabled=False,
            )
            request = parse_request_args(
                f'"Retrieval augmented generation for scientific agents" --input "{manifest_path}"',
                PROJECT_ROOT,
                runtime,
            )
            original_env = {key: os.environ.get(key) for key in FAKE_LLM_ENV}
            try:
                os.environ.update(FAKE_LLM_ENV)
                result = run_pipeline(cwd=PROJECT_ROOT, run_dir=run_dir, run_id="source-run", request=request, runtime=runtime)
                self.assertEqual(result["status"], "success", result["report"])
                checkpoint_before = json.loads((run_dir / "state" / "checkpoint.json").read_text(encoding="utf-8"))
                source_signature_before = checkpoint_before["source_signatures"]["input"]

                manifest["papers"][0]["abstract"] = manifest["papers"][0]["abstract"] + " Updated source evidence."
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                stale_survey = run_dir / "artifacts" / "survey.md"
                stale_survey.write_text("stale downstream artifact", encoding="utf-8")

                result = run_pipeline(cwd=PROJECT_ROOT, run_dir=run_dir, run_id="source-run", request=request, runtime=runtime)
            finally:
                for key, value in original_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

            checkpoint_after = json.loads((run_dir / "state" / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "success", result["report"])
            self.assertNotEqual(source_signature_before, checkpoint_after["source_signatures"]["input"])
            self.assertIn("render_artifacts", checkpoint_after["completed_stages"])
            self.assertNotEqual(stale_survey.read_text(encoding="utf-8"), "stale downstream artifact")

    def test_provider_unavailable_records_blocked_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.run_cli(
                "init",
                "--run-id",
                "blocked-run",
                "--run-dir",
                str(run_dir),
                "--arguments",
                f'"Retrieval augmented generation for scientific agents" --input "{FIXTURE}"',
                "--cwd",
                str(PROJECT_ROOT),
            )
            self.run_cli(
                "synthesize",
                "--run-id",
                "blocked-run",
                "--run-dir",
                str(run_dir),
                "--cwd",
                str(PROJECT_ROOT),
                expected_status=2,
            )

            checkpoint = json.loads((run_dir / "state" / "checkpoint.json").read_text(encoding="utf-8"))
            report = json.loads((run_dir / "artifacts" / "survey_report.json").read_text(encoding="utf-8"))
            self.assertIn("survey_agent", checkpoint["attempted_stages"])
            self.assertIn("survey_agent", checkpoint["blocked_stages"])
            self.assertNotIn("survey_agent", checkpoint["completed_stages"])
            self.assertIn("survey_agent", report["attempted_stages"])
            self.assertIn("survey_agent", report["blocked_stages"])

    def test_audit_blocks_secret_leak_and_external_checkout_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            secret = "test-openai-key"
            self.run_cli(
                "smoke",
                "--run-id",
                "leak-run",
                "--run-dir",
                str(run_dir),
                "--input",
                str(FIXTURE),
                "--topic",
                "Retrieval augmented generation for scientific agents",
                "--cwd",
                str(PROJECT_ROOT),
                extra_env=FAKE_LLM_ENV,
            )
            external_checkout = "/" + "hpc_stor03/sjtu_home/example/agent_workspace/" + "Xcientist-2/src/agents/survey_agent"
            (run_dir / "logs" / "leak.log").write_text(
                f"{secret} {external_checkout}\n",
                encoding="utf-8",
            )

            self.run_cli(
                "audit",
                "--run-id",
                "leak-run",
                "--run-dir",
                str(run_dir),
                "--cwd",
                str(PROJECT_ROOT),
                expected_status=2,
                extra_env=FAKE_LLM_ENV,
            )
            report = json.loads((run_dir / "artifacts" / "survey_report.json").read_text(encoding="utf-8"))
            self.assertFalse(report["checks"]["no_api_secret_leak"])
            self.assertFalse(report["checks"]["no_external_checkout_reference"])
            self.assertIn("API secret", " ".join(report["blocking_errors"]))
            self.assertIn("external research-agent checkouts", " ".join(report["blocking_errors"]))

    def test_audit_requires_integrated_survey_agent_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.run_cli(
                "smoke",
                "--run-id",
                "provenance-run",
                "--run-dir",
                str(run_dir),
                "--input",
                str(FIXTURE),
                "--topic",
                "Retrieval augmented generation for scientific agents",
                "--cwd",
                str(PROJECT_ROOT),
                extra_env=FAKE_LLM_ENV,
            )
            agent_path = run_dir / "state" / "survey_agent_result.json"
            agent_state = json.loads(agent_path.read_text(encoding="utf-8"))
            agent_state["engine"]["mode"] = "external_survey_agent"
            agent_path.write_text(json.dumps(agent_state, indent=2) + "\n", encoding="utf-8")

            self.run_cli(
                "audit",
                "--run-id",
                "provenance-run",
                "--run-dir",
                str(run_dir),
                "--cwd",
                str(PROJECT_ROOT),
                expected_status=2,
                extra_env=FAKE_LLM_ENV,
            )
            report = json.loads((run_dir / "artifacts" / "survey_report.json").read_text(encoding="utf-8"))
            self.assertFalse(report["checks"]["survey_agent_engine_integrated"])
            self.assertIn("integrated_xcientist_survey_agent", " ".join(report["blocking_errors"]))

    def test_audit_rejects_schema_invalid_survey_and_citation_trace(self) -> None:
        for artifact_name, mutation, expected_check in (
            (
                "survey.json",
                lambda value: value.pop("chronology"),
                "survey_schema",
            ),
            (
                "citations.json",
                lambda value: value["traces"][0].pop("claim_id"),
                "citation_trace_schema",
            ),
        ):
            with self.subTest(artifact=artifact_name), tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary) / "run"
                self.run_cli(
                    "smoke",
                    "--run-id",
                    "schema-run",
                    "--run-dir",
                    str(run_dir),
                    "--input",
                    str(FIXTURE),
                    "--topic",
                    "Retrieval augmented generation for scientific agents",
                    "--cwd",
                    str(PROJECT_ROOT),
                    extra_env=FAKE_LLM_ENV,
                )
                artifact_path = run_dir / "artifacts" / artifact_name
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                mutation(artifact)
                artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

                self.run_cli(
                    "audit",
                    "--run-id",
                    "schema-run",
                    "--run-dir",
                    str(run_dir),
                    "--cwd",
                    str(PROJECT_ROOT),
                    expected_status=2,
                    extra_env=FAKE_LLM_ENV,
                )
                report = json.loads((run_dir / "artifacts" / "survey_report.json").read_text(encoding="utf-8"))
                self.assertFalse(report["passed"])
                self.assertFalse(report["checks"][expected_check])

    def test_generated_report_satisfies_bundled_schema(self) -> None:
        from xlab.skills.literature_survey.scripts.literature_survey_lib.audit import (
            validate_schema_file,
        )

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.run_cli(
                "smoke",
                "--run-id",
                "report-schema-run",
                "--run-dir",
                str(run_dir),
                "--input",
                str(FIXTURE),
                "--topic",
                "Retrieval augmented generation for scientific agents",
                "--cwd",
                str(PROJECT_ROOT),
                extra_env=FAKE_LLM_ENV,
            )
            report = json.loads((run_dir / "artifacts" / "survey_report.json").read_text(encoding="utf-8"))
            self.assertEqual(
                validate_schema_file(
                    report,
                    "survey_report.schema.json",
                    "artifacts/survey_report.json",
                ),
                [],
            )
            self.assertTrue(report["checks"]["survey_report_schema"])

    def test_audit_blocks_unresolved_citations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self.run_cli(
                "smoke",
                "--run-id",
                "test-run",
                "--run-dir",
                str(run_dir),
                "--input",
                str(FIXTURE),
                "--topic",
                "Retrieval augmented generation for scientific agents",
                "--cwd",
                str(PROJECT_ROOT),
                extra_env=FAKE_LLM_ENV,
            )
            citations_path = run_dir / "artifacts" / "citations.json"
            citations = json.loads(citations_path.read_text(encoding="utf-8"))
            citations["traces"][0]["paper_ids"].append("missing-paper")
            citations_path.write_text(json.dumps(citations, indent=2) + "\n", encoding="utf-8")

            self.run_cli(
                "audit",
                "--run-id",
                "test-run",
                "--run-dir",
                str(run_dir),
                "--cwd",
                str(PROJECT_ROOT),
                "--status",
                "success",
                expected_status=2,
            )
            report = json.loads((run_dir / "artifacts" / "survey_report.json").read_text(encoding="utf-8"))
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

            self.assertFalse(report["passed"])
            self.assertEqual(manifest["status"], "incomplete")
            self.assertIn("missing-paper", " ".join(report["blocking_errors"]))
            self.assertIn("Requested status success was ignored", " ".join(report["warnings"]))


if __name__ == "__main__":
    unittest.main()
