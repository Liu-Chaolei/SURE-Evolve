from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import patch

import numpy as np
from jsonschema import Draft202012Validator

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE_DIR / "scripts" / "run_idea_phase.py"
TOPIC = "Survey-grounded scientific agents"
SECRET = "sk-live-unique-secret-abcdef"
FAKE_LLM_ENV = "XLAB_RESEARCH_IDEA_" + "FAKE_LLM"
TINY_SEARCH_PROFILE = {
    "max_iterations": 1,
    "max_depth": 1,
    "branching_factor": 1,
    "exploration_constant": 1.2,
    "max_children": None,
    "max_evaluator_calls": None,
    "max_tokens": None,
    "max_cost": None,
}


class ResearchIdeaSkillTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="xlab-research-idea-test-"))
        self.env_patcher = patch.dict(
            os.environ,
            {
                name: ""
                for name in (
                    "OPENAI_CHAT_COMPLETIONS_URL",
                    "LLM_API_KEY",
                    "LLM_BASE_URL",
                    "LLM_API_BASE",
                    "LLM_MODEL",
                    "XLAB_RESEARCH_IDEA_OPENAI_BASE_URL",
                    "XLAB_RESEARCH_IDEA_CHAT_COMPLETIONS_URL",
                )
            },
        )
        self.env_patcher.start()

    def tearDown(self) -> None:
        self.env_patcher.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_cli(self, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        merged_env = os.environ.copy()
        for key in [
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "OPENAI_CHAT_COMPLETIONS_URL",
            "LLM_API_KEY",
            "LLM_BASE_URL",
            "LLM_API_BASE",
            "LLM_MODEL",
            "XLAB_RESEARCH_IDEA_OPENAI_BASE_URL",
            "XLAB_RESEARCH_IDEA_CHAT_COMPLETIONS_URL",
            FAKE_LLM_ENV,
            "XLAB_RESEARCH_IDEA_ENABLE_LLM",
        ]:
            merged_env.pop(key, None)
        merged_env["PYTHONDONTWRITEBYTECODE"] = "1"
        if env:
            merged_env.update(env)
        return subprocess.run(
            ["python", str(SCRIPT), *args],
            cwd=str(PACKAGE_DIR),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=merged_env,
            check=False,
        )

    def test_diagnostics_fixture_command_is_removed(self) -> None:
        run_dir = self.tmp / "run"
        removed_command = "smo" + "ke"
        result = self.run_cli(removed_command, "--run-dir", str(run_dir), "--run-id", "idea-diagnostics", "--topic", TOPIC)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((run_dir / "manifest.json").exists())
        self.assertIn("invalid choice", result.stderr)

    def test_runtime_snapshot_excludes_live_provider_capability(self) -> None:
        survey_run = write_survey_fixture(self.tmp / "survey-run")
        run_dir = self.tmp / "idea-init"
        result = self.run_cli(
            "init",
            "--run-dir",
            str(run_dir),
            "--run-id",
            "idea-init",
            "--arguments",
            f"--survey {survey_run} --topic {TOPIC}",
            "--cwd",
            str(self.tmp),
            env={"OPENAI_API_KEY": SECRET},
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        runtime = read_json(run_dir / "state" / "runtime_config.json")
        self.assertNotIn("openai_api_key_set", runtime)
        self.assertNotIn("llm_enabled", runtime)
        initialized = json.loads(result.stdout)
        self.assertNotIn("provider_available", initialized["api"])
        self.assertNotIn("openai_api_key_set", initialized["api"])
        self.assertEqual(initialized["api"]["algorithm_spec"], "xlab.research_idea.algorithm.v2")

    def test_endpoint_defaults_to_declared_openai_origin(self) -> None:
        survey_run = write_survey_fixture(self.tmp / "default-endpoint-survey")
        run_dir = self.tmp / "default-endpoint"
        result = self.run_cli(
            "init",
            "--run-dir",
            str(run_dir),
            "--run-id",
            "default-endpoint",
            "--arguments",
            f"--survey {survey_run} --topic endpoint-default",
            "--cwd",
            str(self.tmp),
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        runtime = read_json(run_dir / "state" / "runtime_config.json")
        self.assertEqual(runtime["openai_base_url"], "https://api.openai.com/v1")
        self.assertEqual(runtime["chat_completions_url"], "https://api.openai.com/v1/chat/completions")

    def test_endpoint_rejects_unsafe_urls_without_persisting_them(self) -> None:
        unsafe_urls = (
            "http://provider.example/v1",
            "https://user:password@provider.example/v1",
            "https://provider.example/v1?api_key=secret",
            "https://provider.example/v1#token=secret",
            "https://provider.example/v1/%73k-secret",
        )
        for index, endpoint in enumerate(unsafe_urls):
            with self.subTest(endpoint=endpoint):
                run_dir = self.tmp / f"unsafe-endpoint-{index}"
                result = self.run_cli(
                    "init",
                    "--run-dir",
                    str(run_dir),
                    "--run-id",
                    f"unsafe-endpoint-{index}",
                    "--arguments",
                    "--topic unsafe-endpoint",
                    "--cwd",
                    str(self.tmp),
                    env={"OPENAI_BASE_URL": endpoint},
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((run_dir / "state" / "runtime_config.json").exists())
                self.assertNotIn(endpoint, result.stderr)

    def test_endpoint_allows_loopback_http_but_not_non_loopback_http(self) -> None:
        survey_run = write_survey_fixture(self.tmp / "loopback-survey")
        for host in ("127.0.0.1", "localhost", "[::1]"):
            with self.subTest(host=host):
                run_dir = self.tmp / ("loopback-" + host.replace(":", "-").strip("[]"))
                endpoint = f"http://{host}:8080/v1"
                result = self.run_cli(
                    "init",
                    "--run-dir",
                    str(run_dir),
                    "--run-id",
                    "loopback-endpoint",
                    "--arguments",
                    f"--survey {survey_run} --topic loopback-endpoint",
                    "--cwd",
                    str(self.tmp),
                    env={"OPENAI_BASE_URL": endpoint},
                )
                self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
                runtime = read_json(run_dir / "state" / "runtime_config.json")
                self.assertEqual(runtime["openai_base_url"], endpoint)

    def test_undeclared_endpoint_and_key_aliases_are_ignored(self) -> None:
        survey_run = write_survey_fixture(self.tmp / "legacy-alias-survey")
        run_dir = self.tmp / "legacy-aliases"
        aliases = {
            "LLM_API_KEY": "legacy-key",
            "LLM_BASE_URL": "https://legacy.example/v1",
            "LLM_API_BASE": "https://legacy.example/v1",
            "LLM_MODEL": "legacy-model",
            "OPENAI_CHAT_COMPLETIONS_URL": "https://legacy.example/v1/chat/completions",
            "XLAB_RESEARCH_IDEA_OPENAI_BASE_URL": "https://legacy.example/v1",
            "XLAB_RESEARCH_IDEA_CHAT_COMPLETIONS_URL": "https://legacy.example/v1/chat/completions",
        }
        result = self.run_cli(
            "init",
            "--run-dir",
            str(run_dir),
            "--run-id",
            "legacy-aliases",
            "--arguments",
            f"--survey {survey_run} --topic alias-ignored",
            "--cwd",
            str(self.tmp),
            env=aliases,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        runtime = read_json(run_dir / "state" / "runtime_config.json")
        initialized = json.loads(result.stdout)
        self.assertEqual(runtime["openai_base_url"], "https://api.openai.com/v1")
        self.assertEqual(runtime["agent_model"], "gpt-5-mini")
        self.assertNotIn("legacy", json.dumps(runtime))
        self.assertNotIn("provider_available", initialized["api"])

    def test_actual_cli_succeeds_with_local_provider_and_tiny_declared_resources(self) -> None:
        add_scripts_to_path()
        from research_idea_lib.resources.evidence import RetrievalBackends
        from research_idea_lib.survey_repository import SurveyArtifactRepository
        import run_idea_phase

        survey_run = write_survey_fixture(self.tmp / "survey-run")
        resource_paths = write_cli_resource_bundle(survey_run)
        run_dir = self.tmp / ".xlab" / "runs" / "idea-cli-success"
        expected_modes = [
            "moonshot_inventor",
            "bridge_builder",
            "steady_engineer",
            "ambitious_realist",
            "evidence_first",
        ]

        with DeterministicOpenAIServer() as provider:
            environment = {
                "OPENAI_API_KEY": SECRET,
                "OPENAI_BASE_URL": provider.base_url,
                "XLAB_RESEARCH_IDEA_AGENT_MODEL": "fixture-agent",
                "XLAB_RESEARCH_IDEA_GENERATION_MODEL": "fixture-generation",
                "XLAB_RESEARCH_IDEA_EVALUATION_MODEL": "fixture-evaluation",
                "XLAB_RESEARCH_IDEA_FUSION_MODEL": "fixture-fusion",
                "XLAB_RESEARCH_IDEA_MAX_RETRIES": "0",
                "XLAB_RESEARCH_IDEA_TIMEOUT_SECONDS": "10",
            }
            initialized = self.run_cli(
                "init",
                "--run-dir",
                str(run_dir),
                "--run-id",
                "idea-cli-success",
                "--arguments",
                f"--survey {survey_run} --topic {json.dumps(TOPIC)}",
                "--cwd",
                str(self.tmp),
                env=environment,
            )
            self.assertEqual(initialized.returncode, 0, initialized.stderr or initialized.stdout)

            runtime_path = run_dir / "state" / "runtime_config.json"
            runtime = read_json(runtime_path)
            runtime.update(TINY_SEARCH_PROFILE)
            write_json(runtime_path, runtime)

            original_from_request = SurveyArtifactRepository.from_request.__func__

            def repository_with_test_backends(cls, request, cwd, *, model_cache_root=None):
                repository = original_from_request(cls, request, cwd, model_cache_root=model_cache_root)
                repository.retrieval_backends = RetrievalBackends(
                    HermeticEmbedding(384),
                    HermeticEmbedding(3),
                    HermeticFaiss(dimension=3, count=1),
                )
                return repository

            stdout = io.StringIO()
            stderr = io.StringIO()
            argv = [
                str(SCRIPT),
                "synthesize",
                "--run-dir",
                str(run_dir),
                "--run-id",
                "idea-cli-success",
                "--cwd",
                str(self.tmp),
            ]
            with patch.object(SurveyArtifactRepository, "from_request", classmethod(repository_with_test_backends)), patch(
                "research_idea_lib.algorithm.runtime_adapters.build_component_novelty_runtime",
                side_effect=lambda resolution: build_hermetic_novelty_runtime(resolution),
            ), patch.dict(os.environ, environment, clear=False), patch.object(sys, "argv", argv), redirect_stdout(stdout), redirect_stderr(stderr):
                returncode = run_idea_phase.main()
            synthesized = subprocess.CompletedProcess(argv, returncode, stdout.getvalue(), stderr.getvalue())

        failure_detail = synthesized.stderr or synthesized.stdout
        if (run_dir / "artifacts" / "idea_report.json").is_file():
            failure_detail += "\n" + json.dumps(read_json(run_dir / "artifacts" / "idea_report.json"), sort_keys=True)
        self.assertEqual(synthesized.returncode, 0, failure_detail)
        self.assertEqual(json.loads(synthesized.stdout)["status"], "success")
        self.assertTrue(provider.calls)
        self.assertTrue(all(call["authorization"] == f"Bearer {SECRET}" for call in provider.calls))

        manifest = read_json(run_dir / "manifest.json")
        report = read_json(run_dir / "artifacts" / "idea_report.json")
        idea = read_json(run_dir / "artifacts" / "research_idea.json")
        idea_result = read_json(run_dir / "artifacts" / "idea_result.json")
        trace = read_json(run_dir / "artifacts" / "idea_trace.json")
        preflight = read_json(run_dir / "state" / "resource_preflight.json")
        research_idea = read_json(run_dir / "state" / "research_idea" / "artifact.json")

        self.assertEqual(manifest["status"], "success")
        self.assertTrue(manifest["validation"]["passed"])
        self.assertEqual(
            [(artifact["type"], artifact["schema_version"]) for artifact in manifest["artifacts"]],
            [
                ("research_idea", "2"),
                ("research_idea_result", "2"),
                ("research_idea_trace", "2"),
                ("research_idea_report", "2"),
            ],
        )
        survey_parent_id = xlab_file_artifact_id(survey_run / "artifacts" / "survey.json")
        self.assertTrue(all(survey_parent_id in artifact["parents"] for artifact in manifest["artifacts"]))
        self.assertTrue(all("parent_artifact_ids" not in artifact for artifact in manifest["artifacts"]))
        self.assertEqual(
            [artifact["path"] for artifact in manifest["artifacts"]],
            [
                "artifacts/research_idea.json",
                "artifacts/idea_result.json",
                "artifacts/idea_trace.json",
                "artifacts/idea_report.json",
            ],
        )
        self.assertTrue(all(not Path(artifact["path"]).is_absolute() for artifact in manifest["artifacts"]))

        self.assertTrue(report["passed"])
        self.assertEqual(report["blocking_errors"], [])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["counts"]["research_idea_modes"], len(expected_modes))
        self.assertEqual(report["counts"]["mcts_iterations"], len(expected_modes))
        self.assertGreater(report["counts"]["provider_success_ops"], 3)
        self.assertTrue(preflight["passed"])
        self.assertTrue(all(preflight["checks"].values()))
        self.assertEqual(
            set(preflight["resources"]),
            {"survey", "graph", "component_index", "keynotes", "outcome_model", "component_model"},
        )
        self.assertTrue(preflight["resources"]["outcome_model"]["logical_uri"].startswith("xlab-cache://models/"))
        self.assertTrue(preflight["resources"]["component_model"]["logical_uri"].startswith("xlab-cache://models/"))

        self.assertEqual(set(research_idea), {"run", "retrieval", "analysis", "ideation", "persistence"})
        self.assertTrue(research_idea["retrieval"]["rag_hits"])
        self.assertTrue(research_idea["retrieval"]["references"])
        self.assertTrue(research_idea["analysis"]["entries"])
        self.assertTrue(research_idea["analysis"]["root_idea"])
        candidates = research_idea["ideation"]["mode_candidates"]
        self.assertEqual([candidate["mode"] for candidate in candidates], expected_modes)
        self.assertTrue(research_idea["ideation"]["fusion_result"])
        for mode in expected_modes:
            search = read_json(run_dir / "state" / "research_idea" / f"search.{mode}.json")
            self.assertEqual(search["mode"], mode)
            self.assertEqual(search["usage"]["iterations"], 1)

        self.assertEqual(idea["status"], "success")
        self.assertTrue(idea["source_evidence"])
        self.assertEqual(idea_result["status"], "success")
        self.assertEqual(idea_result["idea_source"], "fused")
        self.assertEqual(idea_result["source_modes"], expected_modes)
        self.assertEqual(idea_result["algorithm_provenance"]["runtime"], "package-native")
        self.assertEqual(set(idea_result["mcts_evolution"]["pareto_front"]), set(expected_modes))
        self.assertEqual(trace["status"], "success")
        self.assertEqual(
            [event["stage"] for event in trace["workflow_trace"]],
            ["knowledge_acquisition", "advanced_analysis", "idea_generation"],
        )
        successful_operations = {
            event["op_name"]
            for event in trace["operation_trace"]
            if event.get("event") == "llm_call" and event.get("status") == "success"
        }
        self.assertTrue({
            "xlab.research_idea.analysis.generate.v1",
            "xlab.research_idea.fusion.generate.v1",
            "xlab.research_idea.idea.materialize.v1",
        }.issubset(successful_operations))
        assert_public_artifacts_validate(run_dir)

        pipeline_events = [json.loads(line) for line in (run_dir / "logs" / "pipeline.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(
            {"ingest_survey", "resource_preflight", "materialize_xlab_artifact", "audit"}.issubset(
                {event["stage"] for event in pipeline_events if event["status"] == "success"}
            )
        )
        serialized = "\n".join(
            path.read_text(encoding="utf-8")
            for path in run_dir.rglob("*")
            if path.is_file() and path.suffix in {".json", ".jsonl", ".log"}
        )
        self.assertNotIn(SECRET, serialized)
        self.assertNotIn(SECRET[:8], serialized)
        self.assertNotIn(str(resource_paths["outcome_model"]), serialized)
        self.assertNotIn(str(resource_paths["component_model"]), serialized)
        self.assertNotIn(str(run_dir / "state" / "memory" / "resource-cache" / "models"), serialized)

    def test_completed_success_resume_without_credentials_preserves_result(self) -> None:
        add_scripts_to_path()
        from research_idea_lib.checkpoints import write_checkpoint
        from research_idea_lib.config import load_runtime_config
        from research_idea_lib.inputs import IdeaRequest
        from research_idea_lib.pipeline import run_pipeline

        run_dir = self.tmp / "completed-resume"
        artifacts = run_dir / "artifacts"
        artifacts.mkdir(parents=True)
        write_json(artifacts / "idea_result.json", {"status": "success", "title": "Preserved result"})
        write_json(artifacts / "research_idea.json", {"status": "success", "research_question": "Preserved question"})
        write_json(artifacts / "idea_trace.json", {"status": "success"})
        write_json(artifacts / "idea_report.json", {"passed": True, "status": "success"})
        write_json(run_dir / "manifest.json", {"run_id": "completed-resume", "status": "success"})
        write_checkpoint(run_dir, "audit", completed_stage="audit")
        paths = [run_dir / "manifest.json", *(artifacts / name for name in ("idea_result.json", "research_idea.json", "idea_trace.json", "idea_report.json"))]
        preserved = {path: path.read_bytes() for path in paths}

        result = run_pipeline(
            cwd=self.tmp,
            run_dir=run_dir,
            run_id="completed-resume",
            request=IdeaRequest(survey_path=self.tmp / "missing-survey.json", topic=TOPIC),
            runtime=load_runtime_config(),
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual({path: path.read_bytes() for path in paths}, preserved)

    def test_successful_resume_clears_transient_last_error(self) -> None:
        add_scripts_to_path()
        from research_idea_lib.checkpoints import checkpoint_signature, read_checkpoint, write_checkpoint
        from research_idea_lib.config import load_runtime_config
        from research_idea_lib.inputs import IdeaRequest
        from research_idea_lib.pipeline import runtime_contract_json, run_research_idea
        from research_idea_lib.survey_repository import SurveyArtifactRepository

        survey_run = write_survey_fixture(self.tmp / "survey-run")
        run_dir = self.tmp / "transient-error-resume"
        request = IdeaRequest(survey_path=survey_run, topic=TOPIC)
        runtime = load_runtime_config()
        repository = SurveyArtifactRepository.from_request(request, self.tmp, model_cache_root=run_dir / "state" / "memory" / "resource-cache")
        request_json = request.to_json(self.tmp)
        runtime_json = runtime_contract_json(runtime.to_json())
        request_signature, runtime_signature, _ = checkpoint_signature(request_json, runtime_json, repository.source.signatures)
        from research_idea_lib.common import run_paths
        paths = run_paths(run_dir)
        for path in (paths["idea_result_json"], paths["research_idea_json"], paths["idea_trace_json"], paths["workflow_artifact"]):
            write_json(path, {"preserved": True})
        write_checkpoint(
            run_dir,
            "materialize_xlab_artifact",
            completed_stage="materialize_xlab_artifact",
            request_signature=request_signature,
            runtime_config_signature=runtime_signature,
            source_signatures=repository.source.signatures,
            last_error="temporary provider failure",
        )

        run_research_idea(self.tmp, run_dir, "transient-error-resume", request, runtime, request_json, runtime_json, repository, {})

        self.assertIsNone(read_checkpoint(run_dir)["last_error"])

    def test_feedback_compatibility_alias_is_rejected(self) -> None:
        survey_run = write_survey_fixture(self.tmp / "survey-run")
        result = self.run_cli(
            "init",
            "--run-dir",
            str(self.tmp / "idea-feedback-alias"),
            "--run-id",
            "idea-feedback-alias",
            "--arguments",
            f"--survey {survey_run} --feedback legacy",
            "--cwd",
            str(self.tmp),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsupported research_idea arguments", result.stderr)

    def test_fake_provider_env_cannot_succeed(self) -> None:
        survey_run = write_survey_fixture(self.tmp / "survey-run")
        run_dir = self.tmp / "idea-fake-env"
        result = self.run_cli(
            "synthesize",
            "--run-dir",
            str(run_dir),
            "--run-id",
            "idea-fake-env",
            "--arguments",
            f"--survey {survey_run} --topic {TOPIC}",
            "--cwd",
            str(self.tmp),
            env={FAKE_LLM_ENV: "1"},
        )
        self.assertEqual(result.returncode, 2, result.stderr or result.stdout)
        manifest = read_json(run_dir / "manifest.json")
        report = read_json(run_dir / "artifacts" / "idea_report.json")
        self.assertEqual(manifest["status"], "incomplete")
        self.assertFalse(report["passed"])
        self.assertIn("OPENAI_API_KEY", json.dumps(report, ensure_ascii=False))

    def test_resolves_direct_survey_manifest_and_run_dir_but_requires_full_contract(self) -> None:
        survey_run = write_survey_fixture(self.tmp / "survey-run")
        for index, survey_arg in enumerate(
            [
                survey_run / "artifacts" / "survey.json",
                survey_run / "artifacts" / "survey.md",
                survey_run / "manifest.json",
                survey_run,
            ]
        ):
            run_dir = self.tmp / f"idea-{index}"
            result = self.run_cli(
                "synthesize",
                "--run-dir",
                str(run_dir),
                "--run-id",
                f"idea-{index}",
                "--arguments",
                f"--survey {survey_arg} --topic {TOPIC}",
                "--cwd",
                str(self.tmp),
            )
            self.assertEqual(result.returncode, 2, result.stderr or result.stdout)
            manifest = read_json(run_dir / "manifest.json")
            report = read_json(run_dir / "artifacts" / "idea_report.json")
            self.assertEqual(manifest["status"], "incomplete")
            self.assertFalse(report["passed"])

    def test_incomplete_survey_does_not_succeed_or_emit_lightweight_success(self) -> None:
        survey_run = write_survey_fixture(self.tmp / "bad-survey", passed=False, runtime_mode="survey-agent-unavailable")
        run_dir = self.tmp / "idea-incomplete"
        result = self.run_cli(
            "synthesize",
            "--run-dir",
            str(run_dir),
            "--run-id",
            "idea-incomplete",
            "--arguments",
            f"--survey {survey_run}",
            "--cwd",
            str(self.tmp),
            env={FAKE_LLM_ENV: "1"},
        )
        self.assertEqual(result.returncode, 2, result.stderr or result.stdout)
        manifest = read_json(run_dir / "manifest.json")
        idea_result = read_json(run_dir / "artifacts" / "idea_result.json")
        report = read_json(run_dir / "artifacts" / "idea_report.json")
        self.assertEqual(manifest["status"], "incomplete")
        self.assertFalse(report["passed"])
        self.assertFalse(idea_result["title"])
        self.assertEqual(idea_result["status"], "incomplete")
        self.assertEqual(report["status"], "incomplete")
        self.assertIn("SurveyAgent", idea_result["incomplete_reason"])
        self.assertIn("SurveyAgent", " ".join(report["blocking_errors"]))
        assert_public_artifacts_validate(run_dir)

    def test_string_reference_shape_is_supported_but_still_requires_resource_preflight(self) -> None:
        survey_run = write_survey_fixture(self.tmp / "string-reference-survey", string_references=True)
        run_dir = self.tmp / "idea-string-references"
        result = self.run_cli(
            "synthesize",
            "--run-dir",
            str(run_dir),
            "--run-id",
            "idea-string-references",
            "--arguments",
            f"--survey {survey_run} --topic {TOPIC}",
            "--cwd",
            str(self.tmp),
            env={"OPENAI_API_KEY": SECRET},
        )
        self.assertEqual(result.returncode, 2, result.stderr or result.stdout)
        survey_context = read_json(run_dir / "state" / "survey_context.json")
        report = read_json(run_dir / "artifacts" / "idea_report.json")
        self.assertEqual(survey_context["reference_count"], 3)
        self.assertEqual(survey_context["evidence_count"], 4)
        preflight = read_json(run_dir / "state" / "resource_preflight.json")
        self.assertFalse(preflight["passed"])
        self.assertIn("resource preflight", json.dumps(report, ensure_ascii=False).lower())
        self.assertNotIn("Citation traces must resolve", " ".join(report["blocking_errors"]))

    def test_resume_reprobes_provider_after_key_is_added(self) -> None:
        survey_run = write_survey_fixture(self.tmp / "survey-run")
        run_dir = self.tmp / "idea-provider-resume"
        initialized = self.run_cli(
            "init",
            "--run-dir",
            str(run_dir),
            "--run-id",
            "idea-provider-resume",
            "--arguments",
            f"--survey {survey_run} --topic {TOPIC}",
            "--cwd",
            str(self.tmp),
        )
        self.assertEqual(initialized.returncode, 0, initialized.stderr or initialized.stdout)
        resumed = self.run_cli(
            "resume",
            "--run-dir",
            str(run_dir),
            "--run-id",
            "idea-provider-resume",
            "--cwd",
            str(self.tmp),
            env={"OPENAI_API_KEY": SECRET},
        )
        self.assertEqual(resumed.returncode, 2, resumed.stderr or resumed.stdout)
        preflight = read_json(run_dir / "state" / "resource_preflight.json")
        self.assertTrue(preflight["checks"]["provider_available"])
        report = read_json(run_dir / "artifacts" / "idea_report.json")
        self.assertNotIn("OPENAI_API_KEY is missing", " ".join(report["blocking_errors"]))

    def test_audit_reuses_finalization_and_emits_portable_paths(self) -> None:
        survey_run = write_survey_fixture(self.tmp / "survey-run")
        run_dir = self.tmp / "idea-audit"
        synthesized = self.run_cli(
            "synthesize",
            "--run-dir",
            str(run_dir),
            "--run-id",
            "idea-audit",
            "--arguments",
            f"--survey {survey_run}",
            "--cwd",
            str(self.tmp),
        )
        self.assertEqual(synthesized.returncode, 2, synthesized.stderr or synthesized.stdout)
        audited = self.run_cli(
            "audit",
            "--run-dir",
            str(run_dir),
            "--run-id",
            "idea-audit",
            "--cwd",
            str(self.tmp),
        )
        self.assertEqual(audited.returncode, 2, audited.stderr or audited.stdout)
        manifest = read_json(run_dir / "manifest.json")
        self.assertEqual(manifest["status"], "incomplete")
        self.assertTrue(all(not Path(artifact["path"]).is_absolute() for artifact in manifest["artifacts"]))
        assert_public_artifacts_validate(run_dir)

    def test_finalization_downgrades_schema_invalid_generated_artifacts(self) -> None:
        add_scripts_to_path()
        from research_idea_lib.manifest import finalize_run

        mutations = {
            "research_idea.json": lambda payload: payload.update(topic=7),
            "idea_result.json": lambda payload: payload["source_modes"].append(7),
            "idea_trace.json": lambda payload: payload.update(topic=7),
        }
        for filename, mutate in mutations.items():
            with self.subTest(filename=filename):
                run_dir = self.tmp / f"invalid-{filename}"
                write_success_audit_fixture(run_dir)
                path = run_dir / "artifacts" / filename
                payload = read_json(path)
                mutate(payload)
                write_json(path, payload)
                self.assertTrue(audit_report(run_dir)["passed"], "legacy audit must not catch this schema-only defect")

                result = finalize_run(
                    cwd=self.tmp,
                    run_dir=run_dir,
                    run_id=f"invalid-{filename}",
                    skill_version="2.0.0",
                    request={},
                )

                self.assertEqual(result["status"], "incomplete")
                self.assertFalse(result["manifest"]["validation"]["passed"])
                self.assertFalse(result["report"]["passed"])
                self.assertFalse(result["report"]["checks"]["generated_public_artifact_schemas"])
                self.assertTrue(result["report"]["checks"]["emitted_public_artifact_schemas"])
                self.assertIn(filename, " ".join(result["report"]["blocking_errors"]))
                assert_public_artifacts_validate(run_dir)

    def test_finalization_validates_generated_report_schema(self) -> None:
        add_scripts_to_path()
        from unittest.mock import patch

        from research_idea_lib.manifest import finalize_run

        run_dir = self.tmp / "invalid-generated-report"
        write_success_audit_fixture(run_dir)
        generated_report = audit_report(run_dir)
        self.assertTrue(generated_report["passed"])
        generated_report["counts"] = []

        with patch("research_idea_lib.manifest.audit_artifacts", return_value=generated_report):
            result = finalize_run(
                cwd=self.tmp,
                run_dir=run_dir,
                run_id="invalid-generated-report",
                skill_version="2.0.0",
                request={},
            )

        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["manifest"]["validation"]["passed"])
        self.assertIn("idea_report.json.counts", " ".join(result["report"]["blocking_errors"]))
        assert_public_artifacts_validate(run_dir)

    def test_manifest_inputs_allowlist_omits_raw_arguments_and_external_paths(self) -> None:
        survey_run = write_survey_fixture(self.tmp / "survey-run")
        run_dir = self.tmp / "idea-public-inputs"
        unique_host_path = self.tmp / "host-only" / "private-survey"
        result = self.run_cli(
            "synthesize",
            "--run-dir",
            str(run_dir),
            "--run-id",
            "idea-public-inputs",
            "--arguments",
            f"--survey {survey_run} --topic {json.dumps(str(unique_host_path))}",
            "--cwd",
            str(self.tmp),
        )
        self.assertEqual(result.returncode, 2, result.stderr or result.stdout)
        manifest = read_json(run_dir / "manifest.json")
        self.assertEqual(
            set(manifest["inputs"]),
            {"topic", "mature_idea", "refinement_scope", "discussion", "experiment_feedback", "resume", "survey"},
        )
        self.assertNotIn("raw_args", manifest["inputs"])
        self.assertEqual(manifest["inputs"]["survey"]["path"], "survey-run")
        self.assertTrue(manifest["inputs"]["survey"]["artifact_ids"])
        self.assertNotIn(str(survey_run.resolve()), json.dumps(manifest, ensure_ascii=False))

    def test_finalization_sanitizes_absolute_paths_in_success_and_incomplete_projection(self) -> None:
        add_scripts_to_path()
        from research_idea_lib.manifest import finalize_run

        unique_host_path = str((self.tmp / "host-only" / "disclosure-sentinel").resolve())
        for status in ("success", "incomplete"):
            with self.subTest(status=status):
                run_dir = self.tmp / f"projection-{status}"
                write_success_audit_fixture(run_dir)
                idea_path = run_dir / "artifacts" / "research_idea.json"
                idea = read_json(idea_path)
                idea["method"] = f"Generated safely despite private source {unique_host_path}"
                write_json(idea_path, idea)
                report = audit_report(run_dir)
                report["passed"] = status == "success"
                report["blocking_errors"] = [] if status == "success" else [f"blocked by {unique_host_path}"]
                with patch("research_idea_lib.manifest.audit_artifacts", return_value=report):
                    finalize_run(
                        cwd=self.tmp,
                        run_dir=run_dir,
                        run_id=f"projection-{status}",
                        skill_version="2.0.0",
                        request={"topic": "portable topic"},
                    )
                serialized = "\n".join(
                    path.read_text(encoding="utf-8")
                    for path in [run_dir / "manifest.json", *(run_dir / "artifacts").glob("*.json")]
                )
                self.assertNotIn(unique_host_path, serialized)
                self.assertIn("[redacted-path]", serialized)

    def test_preflight_persists_portable_resource_blocker_not_host_paths(self) -> None:
        survey_run = write_survey_fixture(self.tmp / "survey-run")
        run_dir = self.tmp / "idea-portable-preflight"
        result = self.run_cli(
            "synthesize",
            "--run-dir",
            str(run_dir),
            "--run-id",
            "idea-portable-preflight",
            "--arguments",
            f"--survey {survey_run}",
            "--cwd",
            str(self.tmp),
            env={"OPENAI_API_KEY": SECRET},
        )
        self.assertEqual(result.returncode, 2, result.stderr or result.stdout)
        preflight = read_json(run_dir / "state" / "resource_preflight.json")
        self.assertNotIn("resource_paths", preflight)
        self.assertEqual(preflight["resources"], {})
        self.assertIn("No explicit XLab resource manifest", " ".join(preflight["blocking_errors"]))
        serialized = json.dumps(preflight, ensure_ascii=False)
        self.assertNotIn(str(self.tmp), serialized)

    def test_provider_missing_is_incomplete(self) -> None:
        survey_run = write_survey_fixture(self.tmp / "survey-run")
        run_dir = self.tmp / "idea-provider-missing"
        result = self.run_cli(
            "synthesize",
            "--run-dir",
            str(run_dir),
            "--run-id",
            "idea-provider-missing",
            "--arguments",
            f"--survey {survey_run}",
            "--cwd",
            str(self.tmp),
        )
        self.assertEqual(result.returncode, 2, result.stderr or result.stdout)
        manifest = read_json(run_dir / "manifest.json")
        report = read_json(run_dir / "artifacts" / "idea_report.json")
        self.assertEqual(manifest["status"], "incomplete")
        self.assertIn("OPENAI_API_KEY", json.dumps(report, ensure_ascii=False))

    def test_feedback_records_replan_workflow_intent_without_lightweight_success(self) -> None:
        survey_run = write_survey_fixture(self.tmp / "survey-run")
        run_dir = self.tmp / "idea-feedback"
        feedback = "ablation showed the retrieval memory component regressed long-context tasks"
        result = self.run_cli(
            "synthesize",
            "--run-dir",
            str(run_dir),
            "--run-id",
            "idea-feedback",
            "--arguments",
            f"--survey {survey_run} --topic {TOPIC} --experiment-feedback {json.dumps(feedback)}",
            "--cwd",
            str(self.tmp),
            env={FAKE_LLM_ENV: "1"},
        )
        self.assertEqual(result.returncode, 2, result.stderr or result.stdout)
        manifest = read_json(run_dir / "manifest.json")
        context = read_json(run_dir / "state" / "context.json")
        self.assertEqual(manifest["status"], "incomplete")
        self.assertEqual(context["workflow"], ["advanced_analysis", "re_analysis_replan", "idea_generation"])

    def test_secret_value_prefix_and_length_are_not_persisted(self) -> None:
        survey_run = write_survey_fixture(self.tmp / "survey-run")
        run_dir = self.tmp / "idea-secret"
        result = self.run_cli(
            "synthesize",
            "--run-dir",
            str(run_dir),
            "--run-id",
            "idea-secret",
            "--arguments",
            f"--survey {survey_run} --topic {TOPIC}",
            "--cwd",
            str(self.tmp),
            env={"OPENAI_API_KEY": SECRET},
        )
        self.assertEqual(result.returncode, 2, result.stderr or result.stdout)
        serialized = "\n".join(path.read_text(encoding="utf-8") for path in run_dir.rglob("*") if path.is_file() and path.suffix in {".json", ".jsonl", ".log"})
        self.assertNotIn(SECRET, serialized)
        self.assertNotIn(SECRET[:8], serialized)
        self.assertNotIn("api_key_" + "len" + "gth", serialized)
        self.assertNotIn("api_key_" + "pre" + "view", serialized)

    def test_package_has_no_runtime_import_of_external_xcientist_code(self) -> None:
        forbidden_imports = [
            "from " + "src.agents.idea_agent",
            "import " + "src.agents.idea_agent",
            "from " + "memory",
            "import " + "memory",
        ]
        forbidden_paths = ["/" + "hpc" + "_stor", "/" + "aistor" + "/" + "hpc" + "_stor", "agent_workspace/" + "Xcientist-2"]
        for path in PACKAGE_DIR.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".ts", ".json", ".md", ".lock"}:
                continue
            text = path.read_text(encoding="utf-8")
            for needle in forbidden_imports:
                self.assertNotIn(needle, text, f"{path} contains external import {needle}")
            for needle in forbidden_paths:
                self.assertNotIn(needle, text, f"{path} contains host-specific path {needle}")

    def test_artifact_path_audit_is_structural_not_generic_text_matching(self) -> None:
        import sys

        scripts_dir = PACKAGE_DIR / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from research_idea_lib.audit import scan_artifact_safety

        run_dir = self.tmp / "path-audit"
        write_json(run_dir / "state" / "portable.json", {"path": "artifacts/result.json", "note": "/outside/in-prose"})
        findings = scan_artifact_safety(run_dir)
        self.assertEqual(findings["nonportable_paths"], [])

        write_json(run_dir / "state" / "nonportable.json", {"artifact_path": "/outside/result.json"})
        findings = scan_artifact_safety(run_dir)
        self.assertEqual(len(findings["nonportable_paths"]), 1)
        self.assertIn("artifact_path", findings["nonportable_paths"][0])

    def test_artifact_audit_rejects_absolute_paths_even_inside_run_or_project(self) -> None:
        import sys

        scripts_dir = PACKAGE_DIR / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from research_idea_lib.audit import scan_artifact_safety

        run_dir = self.tmp / ".xlab" / "runs" / "absolute-path-audit"
        write_json(run_dir / "manifest.json", {"path": str((run_dir / "artifacts" / "idea.json").resolve())})
        write_json(run_dir / "state" / "checkpoint.json", {"data": {"manifest_path": str((self.tmp / "manifest.json").resolve())}})

        findings = scan_artifact_safety(run_dir)

        self.assertEqual(len(findings["nonportable_paths"]), 2)
        self.assertTrue(any("manifest.json:path" in finding for finding in findings["nonportable_paths"]))
        self.assertTrue(any("checkpoint.json:data.manifest_path" in finding for finding in findings["nonportable_paths"]))

    def test_artifact_audit_rejects_rotated_secret_without_live_environment_key(self) -> None:
        import sys

        scripts_dir = PACKAGE_DIR / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from unittest.mock import patch

        from research_idea_lib.audit import scan_artifact_safety

        run_dir = self.tmp / "rotated-secret-audit"
        write_json(run_dir / "artifacts" / "idea_trace.json", {"trace": "sk-proj-rotated-secret-1234567890"})
        write_json(run_dir / "state" / "checkpoint.json", {"data": {"api_key": "provider-key-without-known-prefix"}})
        with patch.dict(os.environ, {}, clear=True):
            findings = scan_artifact_safety(run_dir)

        self.assertEqual(len(findings["secrets"]), 2)
        self.assertTrue(any("idea_trace.json" in finding for finding in findings["secrets"]))
        self.assertTrue(any("checkpoint.json" in finding for finding in findings["secrets"]))

    def test_artifact_audit_avoids_secret_and_path_prose_false_positives(self) -> None:
        import sys

        scripts_dir = PACKAGE_DIR / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from research_idea_lib.audit import scan_artifact_safety

        run_dir = self.tmp / "safety-false-positive-audit"
        write_json(
            run_dir / "artifacts" / "idea_trace.json",
            {
                "note": "Use OPENAI_API_KEY and /tmp/example only as documentation.",
                "token": "token budget",
                "path": ["root", "candidate"],
                "logical_path": "models/snapshot.bin",
                "api_key": "<redacted>",
            },
        )

        self.assertEqual(scan_artifact_safety(run_dir), {"nonportable_paths": [], "secrets": []})

    def test_public_artifact_trace_and_manifest_writes_fail_closed(self) -> None:
        import sys

        scripts_dir = PACKAGE_DIR / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from research_idea_lib.common import write_portable_json

        run_dir = self.tmp / "public-write-boundary"
        cases = {
            run_dir / "artifacts" / "research_idea.json": {"api_key": "rotated-provider-key"},
            run_dir / "artifacts" / "idea_trace.json": {"trace": "sk-proj-rotated-secret-1234567890"},
            run_dir / "manifest.json": {"path": str((run_dir / "artifacts" / "research_idea.json").resolve())},
        }
        for path, payload in cases.items():
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "Refusing to publish unsafe"):
                write_portable_json(path, payload)
            self.assertFalse(path.exists())

    def test_full_contract_audit_rejects_raw_single_mode_artifacts(self) -> None:
        run_dir = self.tmp / "raw-single-mode"
        artifacts = run_dir / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        source_context = {
            "references": [{"paper_id": "p1", "title": "Survey-Grounded Research Ideation"}],
            "selected_evidence": [{"paper_ids": ["p1"], "title": "Survey evidence"}],
        }
        workflow_trace = full_workflow_trace()
        idea_result = full_contract_idea_result()
        idea_result["idea_source"] = "raw_mode"
        idea_result["source_modes"] = ["moonshot_inventor"]
        idea_result["algorithm_provenance"]["completed_modes"] = ["moonshot_inventor"]
        idea_result["algorithm_provenance"]["all_modes_completed"] = False
        idea_result["algorithm_provenance"]["fusion_used"] = False
        idea_result.pop("fusion_metadata", None)
        idea_result.pop("fusion_evolution", None)
        write_json(artifacts / "idea_result.json", idea_result)
        write_json(artifacts / "research_idea.json", full_research_idea(source_context, workflow_trace))
        write_json(artifacts / "idea_trace.json", full_idea_trace(source_context, workflow_trace, idea_result))
        write_full_state_provenance(run_dir, idea_result, source_context, workflow_trace)
        report = audit_report(run_dir)
        self.assertFalse(report["passed"])
        serialized_errors = " ".join(report["blocking_errors"])
        self.assertIn("research idea must complete all idea taste modes", serialized_errors)
        self.assertIn("fused output", serialized_errors)

    def test_full_contract_audit_rejects_missing_mcts_evaluation_or_pareto(self) -> None:
        run_dir = self.tmp / "missing-mcts-provenance"
        artifacts = run_dir / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        source_context = {
            "references": [{"paper_id": "p1", "title": "Survey-Grounded Research Ideation"}],
            "selected_evidence": [{"paper_ids": ["p1"], "title": "Survey evidence"}],
        }
        workflow_trace = full_workflow_trace()
        idea_result = full_contract_idea_result()
        idea_result["mcts_evolution"] = {
            "total_iterations": 1,
            "iterations": [{"iteration": 1, "title": "Candidate without evaluation"}],
            "pareto_front": {},
        }
        idea_result["algorithm_provenance"]["pareto_front_present"] = False
        write_json(artifacts / "idea_result.json", idea_result)
        write_json(artifacts / "research_idea.json", full_research_idea(source_context, workflow_trace))
        write_json(artifacts / "idea_trace.json", full_idea_trace(source_context, workflow_trace, idea_result))
        write_full_state_provenance(run_dir, idea_result, source_context, workflow_trace)
        report = audit_report(run_dir)
        self.assertFalse(report["passed"])
        serialized_errors = " ".join(report["blocking_errors"])
        self.assertIn("evaluation payloads", serialized_errors)
        self.assertIn("Pareto candidate provenance", serialized_errors)

    def test_full_contract_audit_rejects_missing_state_provenance(self) -> None:
        run_dir = self.tmp / "missing-state-provenance"
        artifacts = run_dir / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        source_context = {
            "references": [{"paper_id": "p1", "title": "Survey-Grounded Research Ideation"}],
            "selected_evidence": [{"paper_ids": ["p1"], "title": "Survey evidence"}],
        }
        workflow_trace = full_workflow_trace()
        idea_result = full_contract_idea_result()
        write_json(artifacts / "idea_result.json", idea_result)
        write_json(artifacts / "research_idea.json", full_research_idea(source_context, workflow_trace))
        write_json(artifacts / "idea_trace.json", full_idea_trace(source_context, workflow_trace, idea_result))
        report = audit_report(run_dir)
        self.assertFalse(report["passed"])
        serialized_errors = " ".join(report["blocking_errors"])
        self.assertIn("state/resource_preflight.json", serialized_errors)
        self.assertIn("state/research_idea/artifact.json", serialized_errors)

    def test_full_contract_audit_rejects_missing_provider_trace(self) -> None:
        run_dir = self.tmp / "missing-provider-trace"
        artifacts = run_dir / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        source_context = {
            "references": [{"paper_id": "p1", "title": "Survey-Grounded Research Ideation"}],
            "selected_evidence": [{"paper_ids": ["p1"], "title": "Survey evidence"}],
        }
        workflow_trace = full_workflow_trace()
        idea_result = full_contract_idea_result()
        write_json(artifacts / "idea_result.json", idea_result)
        write_json(artifacts / "research_idea.json", full_research_idea(source_context, workflow_trace))
        write_json(artifacts / "idea_trace.json", full_idea_trace(source_context, workflow_trace, idea_result))
        write_full_state_provenance(run_dir, idea_result, source_context, workflow_trace, provider_trace=[])
        report = audit_report(run_dir)
        self.assertFalse(report["passed"])
        self.assertIn("provider-call operation trace", " ".join(report["blocking_errors"]))

    def test_full_contract_audit_rejects_missing_component_novelty_trace(self) -> None:
        run_dir = self.tmp / "missing-component-novelty-trace"
        write_success_audit_fixture(run_dir)
        trace_without_novelty = [
            item
            for item in provider_operation_trace()
            if item.get("op_name") != "xlab.research_idea.component_novelty.evaluate.v1"
        ]
        idea_result = full_contract_idea_result()
        source_context = {
            "references": [
                {"paper_id": "p1", "title": "Survey-Grounded Research Ideation"}
            ],
            "selected_evidence": [
                {"paper_ids": ["p1"], "title": "Survey evidence"}
            ],
        }
        write_full_state_provenance(
            run_dir,
            idea_result,
            source_context,
            full_workflow_trace(),
            provider_trace=trace_without_novelty,
        )
        trace_path = run_dir / "artifacts" / "idea_trace.json"
        public_trace = read_json(trace_path)
        public_trace["operation_trace"] = trace_without_novelty
        write_json(trace_path, public_trace)

        report = audit_report(run_dir)

        self.assertFalse(report["passed"])
        self.assertIn(
            "xlab.research_idea.component_novelty.evaluate.v1",
            " ".join(report["blocking_errors"]),
        )

    def test_full_contract_audit_rejects_tampered_component_novelty_trace(self) -> None:
        run_dir = self.tmp / "tampered-component-novelty-trace"
        write_success_audit_fixture(run_dir)
        tampered_trace = provider_operation_trace()
        novelty = next(
            item
            for item in tampered_trace
            if item.get("op_name") == "xlab.research_idea.component_novelty.evaluate.v1"
        )
        novelty["provider_operation"] = "xlab.research_idea.idea.evaluate.v2"
        idea_result = full_contract_idea_result()
        source_context = {
            "references": [
                {"paper_id": "p1", "title": "Survey-Grounded Research Ideation"}
            ],
            "selected_evidence": [
                {"paper_ids": ["p1"], "title": "Survey evidence"}
            ],
        }
        write_full_state_provenance(
            run_dir,
            idea_result,
            source_context,
            full_workflow_trace(),
            provider_trace=tampered_trace,
        )
        trace_path = run_dir / "artifacts" / "idea_trace.json"
        public_trace = read_json(trace_path)
        public_trace["operation_trace"] = tampered_trace
        write_json(trace_path, public_trace)

        report = audit_report(run_dir)

        self.assertFalse(report["passed"])
        self.assertIn(
            "mismatched native provider operation",
            " ".join(report["blocking_errors"]),
        )

    def test_research_idea_accessors_route_final_result_and_latest_analysis(self) -> None:
        add_scripts_to_path()
        from research_idea_lib.research_idea_artifacts import (
            final_idea_result,
            latest_analysis_entry,
            persistence_namespace,
        )
        artifact = {
            "run": {},
            "retrieval": {},
            "analysis": {},
            "ideation": {},
            "persistence": {},
        }
        expected_result = full_contract_idea_result()
        artifact["persistence"]["idea_result"] = expected_result
        artifact["ideation"]["idea_result"] = {"title": "wrong namespace"}
        artifact["analysis"]["analysis"] = {"replan": "obsolete nested value"}
        artifact["analysis"]["entries"] = [
            {"replan": "older plan"},
            {"replan": "latest plan"},
        ]

        self.assertIs(final_idea_result(artifact), expected_result)
        self.assertIs(persistence_namespace(artifact)["idea_result"], expected_result)
        self.assertEqual(latest_analysis_entry(artifact), {"replan": "latest plan"})

    def test_map_to_research_idea_uses_latest_analysis_entry_for_feedback(self) -> None:
        add_scripts_to_path()
        from research_idea_lib.artifact_mapping import map_to_research_idea
        from research_idea_lib.inputs import IdeaRequest
        from research_idea_lib.research_idea_artifacts import final_idea_result
        artifact = {
            "run": {},
            "retrieval": {},
            "analysis": {},
            "ideation": {},
            "persistence": {},
        }
        idea_result = full_contract_idea_result()
        idea_result["reference_papers"] = []
        idea_result["root_domains"] = ["machine learning"]
        artifact["persistence"]["idea_result"] = idea_result
        artifact["ideation"]["fusion_result"] = {
            "idea": {
                field: idea_result[field]
                for field in (
                    "title",
                    "core_contribution",
                    "hypothesis",
                    "method",
                    "components",
                    "root_domains",
                )
            }
            | {"risks": idea_result["risks"][0]},
            "evidence_ids": ["ev-1"],
        }
        artifact["retrieval"]["evidence"] = [
            {
                "evidence_id": "ev-1",
                "kind": "survey_evidence",
                "text": "Attributed evidence",
                "provenance": {"title": "Attributed evidence", "source": "survey"},
                "paper_ids": ["p1"],
            }
        ]
        artifact["analysis"]["analysis"] = {"replan": "obsolete nested value"}
        artifact["analysis"]["entries"] = [
            {"replan": "older plan"},
            {"replan": "latest plan"},
        ]
        request = IdeaRequest(
            survey_path=self.tmp / "survey.json",
            topic=TOPIC,
            experiment_feedback="ablation feedback",
        )

        mapped = map_to_research_idea(
            run_id="routing-regression",
            request=request,
            idea_result=final_idea_result(artifact),
            workflow_artifact=artifact,
            source_context={
                "topic": TOPIC,
                "selected_evidence": [],
                "references": [{"paper_id": "p1", "title": "Attributed Paper"}],
            },
        )

        self.assertEqual(mapped["idea_result"]["title"], idea_result["title"])
        self.assertEqual(mapped["replanning_trigger"], "latest plan")

    def test_artifact_mapping_projects_validated_fused_evidence(self) -> None:
        add_scripts_to_path()
        from research_idea_lib.artifact_mapping import build_source_evidence

        registry = [
            {
                "evidence_id": "native-1",
                "kind": "claim",
                "text": "Attributed evidence",
                "provenance": {"title": "Attributed", "source": "survey", "rank": 1},
                "paper_ids": ["p1"],
            }
        ]
        self.assertEqual(
            build_source_evidence(registry),
            [
                {
                    "kind": "claim",
                    "id": "native-1",
                    "title": "Attributed",
                    "summary": "Attributed evidence",
                    "paper_ids": ["p1"],
                    "source": "survey",
                    "rank": 1,
                    "score": None,
                }
            ],
        )

    def test_materialization_alignment_rejects_scientific_drift(self) -> None:
        add_scripts_to_path()
        from research_idea_lib.materialization_validation import align_public_materialization

        fused = {
            "idea": {
                "title": "Fused title",
                "core_contribution": "Fused contribution",
                "hypothesis": "Fused hypothesis",
                "method": "Fused method",
                "risks": "Fused risk",
                "root_domains": ["machine learning"],
                "components": [{"name": "evidence memory", "description": "Keeps identities."}],
            },
            "evidence_ids": ["ev-1"],
        }
        base = {
            **fused["idea"],
            "risks": ["Fused risk"],
            "reference_papers": ["Attributed Paper"],
        }
        registry = [
            {
                "evidence_id": "ev-1",
                "kind": "claim",
                "text": "Evidence",
                "provenance": {"source": "survey"},
                "paper_ids": ["p1"],
            }
        ]
        references = [{"paper_id": "p1", "title": "Attributed Paper"}]
        mutations = {
            "title": "A different scientific idea",
            "core_contribution": "A different contribution",
            "hypothesis": "A different hypothesis",
            "method": "A different method",
            "risks": ["A different risk"],
            "root_domains": ["biology"],
            "components": [{"name": "fabricated", "description": "Different mechanism."}],
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                drifted = {**base, field: value}
                with self.assertRaisesRegex(ValueError, f"{field} drifted"):
                    align_public_materialization(drifted, fused, registry, references)

    def test_materialization_alignment_rejects_fabricated_evidence_and_references(self) -> None:
        add_scripts_to_path()
        from research_idea_lib.materialization_validation import align_public_materialization

        fused_idea = {
            "title": "Fused title",
            "core_contribution": "Fused contribution",
            "hypothesis": "Fused hypothesis",
            "method": "Fused method",
            "risks": "Fused risk",
            "root_domains": ["machine learning"],
            "components": ["evidence memory"],
        }
        idea_result = {
            **fused_idea,
            "risks": ["Fused risk"],
            "reference_papers": ["Invented Paper"],
        }
        registry = [
            {
                "evidence_id": "ev-1",
                "kind": "claim",
                "text": "Evidence",
                "provenance": {"source": "survey"},
                "paper_ids": ["p1"],
            }
        ]
        references = [
            {"paper_id": "p1", "title": "Attributed Paper"},
            {"paper_id": "p2", "title": "Available But Unattributed"},
        ]
        with self.assertRaisesRegex(ValueError, "without attributed evidence"):
            align_public_materialization(
                idea_result,
                {"idea": fused_idea, "evidence_ids": ["ev-1"]},
                registry,
                references,
            )
        with self.assertRaisesRegex(ValueError, "unknown evidence identities"):
            align_public_materialization(
                {**idea_result, "reference_papers": []},
                {"idea": fused_idea, "evidence_ids": ["fabricated-evidence"]},
                registry,
                references,
            )
        with self.assertRaisesRegex(ValueError, "evidence identities drifted"):
            align_public_materialization(
                {**idea_result, "reference_papers": [], "evidence_ids": ["fabricated-evidence"]},
                {"idea": fused_idea, "evidence_ids": ["ev-1"]},
                registry,
                references,
            )

    def test_materialization_alignment_projects_reference_identities_deterministically(self) -> None:
        add_scripts_to_path()
        from research_idea_lib.materialization_validation import align_public_materialization

        fused_idea = {
            "title": "Fused title",
            "core_contribution": "Fused contribution",
            "hypothesis": "Fused hypothesis",
            "method": "Fused method",
            "risks": "Fused risk",
            "root_domains": [],
            "components": ["evidence memory"],
        }
        idea_result = {**fused_idea, "risks": ["Fused risk"], "reference_papers": []}
        registry = [
            {
                "evidence_id": "ev-2",
                "kind": "claim",
                "text": "Second",
                "provenance": {"source": "survey"},
                "paper_ids": ["p2", "p1"],
            },
            {
                "evidence_id": "ev-1",
                "kind": "claim",
                "text": "First",
                "provenance": {"source": "survey"},
                "paper_ids": ["p1"],
            },
        ]
        references = [
            {"paper_id": "p1", "title": "First Paper"},
            {"paper_id": "p2", "title": "Second Paper"},
        ]

        aligned = align_public_materialization(
            idea_result,
            {"idea": fused_idea, "evidence_ids": ["ev-2", "ev-1"]},
            registry,
            references,
        )

        self.assertEqual([item["evidence_id"] for item in aligned], ["ev-2", "ev-1"])
        self.assertEqual(idea_result["evidence_ids"], ["ev-2", "ev-1"])
        self.assertEqual(idea_result["reference_ids"], ["p1", "p2"])
        self.assertEqual(idea_result["reference_papers"], ["First Paper", "Second Paper"])

    def test_artifact_mapping_rejects_malformed_generated_list_items(self) -> None:
        add_scripts_to_path()
        from research_idea_lib.artifact_mapping import require_generated_list

        for malformed in ({"invented": "entry"}, 7, True, None):
            with self.subTest(malformed=malformed):
                with self.assertRaisesRegex(ValueError, "only text entries"):
                    require_generated_list({"metrics": ["valid", malformed]}, "metrics")

    def test_reference_papers_require_attributed_native_evidence(self) -> None:
        add_scripts_to_path()
        from research_idea_lib.artifact_mapping import validate_reference_papers

        context = {
            "references": [
                {"paper_id": "p1", "title": "Attributed Paper"},
                {"paper_id": "p2", "title": "Available But Unattributed"},
            ]
        }
        evidence = [{"id": "ev-1", "paper_ids": ["p1"]}]
        self.assertEqual(
            validate_reference_papers({"reference_papers": ["Attributed Paper"]}, context, evidence),
            ["Attributed Paper"],
        )
        for invented in ("Invented Paper", "Available But Unattributed"):
            with self.subTest(reference=invented):
                with self.assertRaisesRegex(ValueError, "without attributed evidence"):
                    validate_reference_papers({"reference_papers": [invented]}, context, evidence)

    def test_reference_papers_reject_malformed_provider_items(self) -> None:
        add_scripts_to_path()
        from research_idea_lib.artifact_mapping import validate_reference_papers

        with self.assertRaisesRegex(ValueError, "only text entries"):
            validate_reference_papers(
                {"reference_papers": [{"title": "Attributed Paper"}]},
                {"references": [{"paper_id": "p1", "title": "Attributed Paper"}]},
                [{"paper_ids": ["p1"]}],
            )

    def test_research_idea_algorithm_spec_freezes_tastes_and_metric_champions(self) -> None:
        add_scripts_to_path()
        from research_idea_lib.research_idea_spec import (
            ALGORITHM_ID,
            ALGORITHM_SPEC_VERSION,
            IDEA_TASTE_WEIGHTS,
            METRIC_CHAMPIONS,
        )
        expected_weights = {
            "moonshot_inventor": (0.14, 0.06, 0.27, 0.22, 0.18, 0.05, 0.03, 0.02, 0.02, 0.01),
            "bridge_builder": (0.16, 0.05, 0.17, 0.10, 0.18, 0.13, 0.06, 0.02, 0.07, 0.06),
            "steady_engineer": (0.17, 0.04, 0.16, 0.07, 0.18, 0.16, 0.08, 0.02, 0.07, 0.05),
            "ambitious_realist": (0.15, 0.05, 0.22, 0.14, 0.20, 0.08, 0.06, 0.03, 0.05, 0.02),
            "evidence_first": (0.16, 0.04, 0.17, 0.04, 0.16, 0.15, 0.08, 0.02, 0.11, 0.07),
        }
        weight_fields = (
            "alignment_weight",
            "complexity_weight",
            "novelty_weight",
            "surprise_weight",
            "impact_weight",
            "feasibility_weight",
            "clarity_weight",
            "conciseness_weight",
            "risk_weight",
            "protocol_weight",
        )

        self.assertEqual(ALGORITHM_ID, "xlab.research_idea.algorithm.v2")
        self.assertEqual(ALGORITHM_SPEC_VERSION, "xlab.research_idea.algorithm-spec.v2")
        for schema_filename in (
            "idea.schema.json",
            "idea_result.schema.json",
            "idea_trace.schema.json",
            "idea_report.schema.json",
        ):
            schema = read_json(PACKAGE_DIR / "schemas" / schema_filename)
            self.assertEqual(schema["$defs"]["algorithmProvenance"]["properties"]["algorithm"]["const"], ALGORITHM_ID)
        self.assertEqual(tuple(IDEA_TASTE_WEIGHTS), tuple(expected_weights))
        for mode, expected in expected_weights.items():
            frozen = tuple(IDEA_TASTE_WEIGHTS[mode][field] for field in weight_fields)
            self.assertEqual(frozen, expected)

        self.assertEqual(
            [(champion.label, champion.metric, champion.objective) for champion in METRIC_CHAMPIONS],
            [("novel", "novelty", "max"), ("feasible", "feasibility", "max"), ("concise", "conciseness", "max")],
        )

    def test_success_contract_requires_complete_payloads(self) -> None:
        add_scripts_to_path()

        source_context = {
            "references": [{"paper_id": "p1", "title": "Survey-Grounded Research Ideation"}],
            "selected_evidence": [{"paper_ids": ["p1"], "title": "Survey evidence"}],
        }
        workflow_trace = full_workflow_trace()
        provenance = {
            "algorithm": "xlab.research_idea.algorithm.v2",
            "runtime_profile": "xlab.research_idea.runtime.v1",
            "evidence_profile": "xlab.research_idea.evidence.v1",
            "success_profile": "xlab.research_idea.success.v1",
        }
        completed_stages = ["knowledge_acquisition", "advanced_analysis", "idea_generation"]
        idea_result = full_contract_idea_result()
        idea_result.update(
            schema_version="xlab.research_idea.result.v2",
            status="success",
            blockers=[],
            completed_stages=completed_stages,
            algorithm_provenance=provenance,
        )
        idea = full_research_idea(source_context, workflow_trace)
        idea.update(
            schema_version="xlab.research_idea.v2",
            status="success",
            blockers=[],
            completed_stages=completed_stages,
            algorithm_provenance=provenance,
            idea_result=idea_result,
        )
        trace = full_idea_trace(source_context, workflow_trace, idea_result)
        trace.update(
            schema_version="xlab.research_idea.trace.v2",
            status="success",
            blockers=[],
            completed_stages=completed_stages,
            algorithm_provenance=provenance,
        )
        for value, schema_filename in (
            (idea, "idea.schema.json"),
            (idea_result, "idea_result.schema.json"),
            (trace, "idea_trace.schema.json"),
        ):
            Draft202012Validator(read_json(PACKAGE_DIR / "schemas" / schema_filename)).validate(value)
        idea_result["title"] = ""
        errors = list(Draft202012Validator(read_json(PACKAGE_DIR / "schemas" / "idea_result.schema.json")).iter_errors(idea_result))
        self.assertTrue(errors)

    def test_scoped_sources_use_only_canonical_research_idea_naming(self) -> None:
        forbidden = ("lig" + "agent").casefold()
        repository_root = PACKAGE_DIR.parents[2]
        roots = (
            PACKAGE_DIR,
            repository_root / "xlab" / "skills" / "research_workflow",
            repository_root / "packages" / "coding-agent" / "src" / "core" / "xlab",
            repository_root / "packages" / "coding-agent" / "test" / "suite" / "xlab-research-idea.test.ts",
        )
        excluded_parts = {
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "__pycache__",
            "coverage",
            "dist",
            "node_modules",
            "runs",
        }
        findings: list[str] = []
        for root in roots:
            paths = (root,) if root.is_file() else root.rglob("*")
            for path in paths:
                if any(part in excluded_parts for part in path.parts):
                    continue
                relative = path.relative_to(repository_root)
                for part in relative.parts:
                    if forbidden in part.casefold():
                        findings.append(f"path: {relative}")
                if not path.is_file():
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                for line_number, line in enumerate(text.splitlines(), start=1):
                    if forbidden in line.casefold():
                        findings.append(f"content: {relative}:{line_number}: {line.strip()}")
        self.assertEqual(findings, [], "Noncanonical research-idea naming remains:\n" + "\n".join(findings))

    def test_runtime_has_no_lightweight_success_markers(self) -> None:
        forbidden = [
            "XLAB_RESEARCH_IDEA_" + "FAKE_LLM",
            "MCTS-" + "style",
            "Deterministic, survey-" + "grounded",
            "score_" + "candidate(",
            "smo" + "ke_" + "sur" + "vey(",
            "api_key_" + "pre" + "view",
            "api_key_" + "len" + "gth",
            "FULL_" + "PARITY_RUNTIME_READY",
        ]
        runtime_roots = [PACKAGE_DIR / "scripts", PACKAGE_DIR / "SKILL.md", PACKAGE_DIR / "examples" / "README.md"]
        for root in runtime_roots:
            paths = [root] if root.is_file() else list(root.rglob("*"))
            for path in paths:
                if not path.is_file() or path.suffix not in {".py", ".json", ".md", ".lock"}:
                    continue
                text = path.read_text(encoding="utf-8")
                for needle in forbidden:
                    self.assertNotIn(needle, text, f"{path} contains lightweight marker {needle}")


def provider_environment(provider: DeterministicOpenAIServer) -> dict[str, str]:
    return {
        "OPENAI_API_KEY": SECRET,
        "OPENAI_BASE_URL": provider.base_url,
        "XLAB_RESEARCH_IDEA_AGENT_MODEL": "fixture-agent",
        "XLAB_RESEARCH_IDEA_GENERATION_MODEL": "fixture-generation",
        "XLAB_RESEARCH_IDEA_EVALUATION_MODEL": "fixture-evaluation",
        "XLAB_RESEARCH_IDEA_FUSION_MODEL": "fixture-fusion",
        "XLAB_RESEARCH_IDEA_MAX_RETRIES": "0",
        "XLAB_RESEARCH_IDEA_TIMEOUT_SECONDS": "10",
    }


def initialize_cli_run(
    test: ResearchIdeaSkillTest,
    run_dir: Path,
    survey_run: Path,
    run_id: str,
    environment: dict[str, str],
) -> None:
    initialized = test.run_cli(
        "init",
        "--run-dir",
        str(run_dir),
        "--run-id",
        run_id,
        "--arguments",
        f"--survey {survey_run} --topic {json.dumps(TOPIC)}",
        "--cwd",
        str(test.tmp),
        env=environment,
    )
    test.assertEqual(initialized.returncode, 0, initialized.stderr or initialized.stdout)
    runtime_path = run_dir / "state" / "runtime_config.json"
    runtime = read_json(runtime_path)
    runtime.update(TINY_SEARCH_PROFILE)
    write_json(runtime_path, runtime)


def write_survey_fixture(run_dir: Path, *, passed: bool = True, runtime_mode: str = "survey-agent", string_references: bool = False) -> Path:
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    references = [
        {"paper_id": "p1", "title": "Survey-Grounded Research Ideation", "year": 2026, "venue": "XLab"},
        {"paper_id": "p2", "title": "Memory Guided Search for Scientific Agents", "year": 2025, "venue": "Agents"},
        {"paper_id": "p3", "title": "Evidence Traceability in Experimental Design", "year": 2024, "venue": "Evaluation"},
    ]
    reference_payload = [reference["paper_id"] for reference in references] if string_references else references
    survey = {
        "schema_version": "xlab.literature_survey.v1",
        "run_id": "survey-run",
        "topic": TOPIC,
        "paper_count": len(references),
        "sections": [
            {"id": "section:grounding", "title": "Grounded ideation", "summary": "Idea agents should ground candidate generation in traceable survey evidence.", "paper_ids": ["p1", "p3"]},
        ],
        "clusters": [{"id": "cluster:agents", "name": "Scientific agents", "paper_ids": ["p1", "p2"]}],
        "key_claims": [
            {"id": "claim:trace", "claim": "Evidence traceability improves reliability of downstream experiment plans.", "paper_ids": ["p3"]},
        ],
        "research_gaps": [
            {"id": "gap:search", "gap": "Few systems connect literature survey memory with tree-search idea evolution and executable evaluation plans.", "paper_ids": ["p1", "p2", "p3"]},
        ],
        "references": reference_payload,
        "runtime": {"mode": runtime_mode, "survey_agent_engine": "integrated_xcientist_survey_agent"},
        "generated_at": "2026-01-01T00:00:00Z",
    }
    citations = {
        "schema_version": "xlab.citation_trace.v1",
        "run_id": "survey-run",
        "references": reference_payload,
        "traces": [
            {"claim_id": "claim:trace", "claim": "Evidence traceability improves reliability of downstream experiment plans.", "paper_ids": ["p3"], "evidence": ["Fixture evidence."]},
        ],
    }
    survey_md = "\n".join(
        [
            f"# {TOPIC}",
            "",
            "## Grounded ideation",
            "Idea agents should ground candidate generation in traceable survey evidence [1][3].",
            "",
            "## Research gaps",
            "Few systems connect literature survey memory with tree-search idea evolution and executable evaluation plans [1][2][3].",
            "",
            "## References",
            "1. [p1] Survey-Grounded Research Ideation (2026).",
            "2. [p2] Memory Guided Search for Scientific Agents (2025).",
            "3. [p3] Evidence Traceability in Experimental Design (2024).",
            "",
        ]
    )
    report = {
        "schema_version": "xlab.literature_survey.report.v1",
        "generated_at": "2026-01-01T00:00:00Z",
        "passed": passed,
        "blocking_errors": [] if passed else ["Survey Agent generation failed."],
        "warnings": [],
        "counts": {"papers": len(references), "sections": 1, "claims": 1, "gaps": 1, "traces": 1},
        "checks": {"survey_agent_mode": runtime_mode == "survey-agent"},
    }
    write_json(artifacts / "survey.json", survey)
    (artifacts / "survey.md").write_text(survey_md, encoding="utf-8")
    write_json(artifacts / "citations.json", citations)
    write_json(artifacts / "survey_report.json", report)
    write_json(
        run_dir / "manifest.json",
        {
            "schema_version": "2",
            "run_id": "survey-run",
            "skill_name": "literature_survey",
            "skill_version": "1.0.0",
            "status": "success" if passed else "incomplete",
            "created_at": "2026-01-01T00:00:00Z",
            "inputs": {},
            "outputs": {"topic": TOPIC},
            "validation": {"passed": passed},
            "artifacts": [
                {"type": "literature_survey_document", "schema_version": "1", "path": str(artifacts / "survey.md")},
                {"type": "literature_survey_json", "schema_version": "1", "path": str(artifacts / "survey.json")},
                {"type": "citation_trace", "schema_version": "1", "path": str(artifacts / "citations.json")},
                {"type": "survey_report", "schema_version": "1", "path": str(artifacts / "survey_report.json")},
            ],
        },
    )
    return run_dir


class DeterministicOpenAIServer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                structured_input = json.loads(request["messages"][-1]["content"])
                operation = provider_operation(structured_input)
                fixture.calls.append(
                    {
                        "path": self.path,
                        "authorization": self.headers.get("Authorization"),
                        "model": request["model"],
                        "operation": operation,
                    }
                )
                payload = provider_payload(operation, structured_input)
                response = json.dumps(
                    {
                        "id": f"fixture-{len(fixture.calls)}",
                        "object": "chat.completion",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(payload)}}],
                        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, format: str, *args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def __enter__(self) -> "DeterministicOpenAIServer":
        self.thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def provider_operation(data: dict[str, Any]) -> str:
    if data.get("diagnostic") is True:
        return "diagnostic"
    if data.get("diagnostic") is False:
        return "evaluate"
    if "paper_id" in data and "keynote" in data and "relevance" not in data:
        return "keynote_score"
    if "paper_id" in data and "keynote" in data and "relevance" in data:
        return "keynote_compress"
    if "papers" in data and "rag_query" in data:
        return "keynote_rollup"
    if data.get("operation") == "xlab.research_idea.component_novelty.evaluate.v1":
        return "component_novelty"
    if "idea" in data and "prompt_mode" in data and "algorithm" in data:
        return "operator_query"
    if "parent" in data and "plan" in data and "seed" in data:
        return "generate"
    if "best_idea" in data and "allowed_operations" in data:
        return "repair"
    if "mode_inputs" in data:
        return "fusion"
    if "candidate" in data and data.get("algorithm") == "xlab.research_idea.algorithm.v2":
        return "referee"
    if "fusion" in data and "source_modes" in data:
        return "materialization"
    if "evidence" in data and "background" in data and "mature_idea" in data:
        return "analysis"
    if "evidence" in data and "query" in data:
        return "ranking"
    if "background" in data and "topic" in data:
        return "query"
    if "topic" in data and "source_context" in data:
        return "background"
    raise AssertionError(f"Unknown fixture provider operation: {sorted(data)}")


def provider_payload(operation: str, data: dict[str, Any]) -> dict[str, Any]:
    if operation == "keynote_score":
        return {"score": 86}
    if operation == "keynote_compress":
        return {
            "summary": "Evidence-grounded ideation preserves auditable citations.",
            "insight": "Carry stable paper provenance through search.",
        }
    if operation == "keynote_rollup":
        return {"summary": "Remaining cited keynotes support auditable search."}
    if operation == "component_novelty":
        return {
            "retrieval_similarity": 3,
            "perceived_novelty": 4,
            "rubric_score": 4,
            "rationale": "Candidate differs from the retrieved declared component.",
            "provenance": {"fixture": "hermetic-cli-component-novelty"},
        }
    if operation == "background":
        return {
            "background": "Survey-grounded scientific agents require auditable evidence throughout idea search.",
            "key_questions": ["How can evidence identities survive search and fusion?"],
            "canonical_methods": ["Typed Monte Carlo tree search"],
        }
    if operation == "query":
        return {"query": "auditable survey evidence scientific agent search"}
    if operation == "ranking":
        return {"evidence_ids": [item["evidence_id"] for item in data["evidence"]]}
    if operation == "analysis":
        return {
            "analysis": {"gap": "Current scientific agents can lose source identities during iterative idea search."},
            "root_idea": fixture_idea("Evidence-rooted scientific search", "evidence-memory"),
        }
    if operation in {"diagnostic", "evaluate"}:
        return {
            "metrics": {
                "novelty": 4,
                "surprise": 4,
                "feasibility": 4,
                "clarity": 4,
                "impact": 4,
                "conciseness": 4,
                "alignment_score": 4,
                "protocol_score": 4,
                "risk": 1,
                "complexity_penalty": 1,
            },
            "confidence": 0.9,
            "detected_defects": ["unclear_mechanism"],
            "feedback": "Bound the evidence-preserving mechanism with an explicit intervention.",
        }
    if operation == "operator_query":
        return {
            "query": "auditable evidence mechanism",
            "mechanism_gap": "The evidence mechanism needs an explicit gate.",
            "expected_role": "Preserve evidence identities during search.",
        }
    if operation == "generate":
        parent = dict(data["parent"])
        mode = data["idea_taste_mode"]
        current = parent["components"][0]["name"]
        replacement = f"{mode}-evidence-memory"
        parent.update(
            title=f"{mode} evidence candidate",
            abstract="A bounded candidate that preserves survey evidence identities during search.",
            core_contribution="Stable evidence identities across typed search edits.",
            method="Apply one evidence-preserving edit and compare it with an ungrounded ablation.",
            risks="Provider judgments can vary across bounded search calls.",
            components=[{"name": replacement, "description": "Carries explicit survey evidence provenance."}],
        )
        return {
            "state": parent,
            "component_mapping": {
                "weak_internal_component": current,
                "refined_internal_component": replacement,
            },
            "component_role_explanations": {
                replacement: "Carries explicit survey evidence provenance."
            },
        }
    if operation == "fusion":
        mode_input = data["mode_inputs"][0]
        components = [dict(mode_input["idea"]["components"][0])]
        return {
            "idea": {
                **fixture_idea("Fused evidence-grounded idea", "unused"),
                "abstract": "Five native searches fused with explicit survey evidence provenance.",
                "core_contribution": "Auditable five-mode scientific ideation.",
                "hypothesis": "Five evidence-grounded searches improve traceability without sacrificing novelty.",
                "method": "Fuse typed MCTS candidates while preserving stable evidence identities.",
                "risks": "Provider judgments can vary across bounded search calls.",
                "components": components,
                "root_domains": ["machine learning"],
            },
            "selected_components": [
                {
                    "component": components[0]["name"],
                    "source_mode": mode_input["mode"],
                    "evidence": mode_input["evidence"],
                }
            ],
            "rejected_components": [],
            "conflict_resolutions": [],
        }
    if operation == "referee":
        return {
            "score": 4.2,
            "metrics": {
                "novelty": 4,
                "surprise": 4,
                "feasibility": 4,
                "clarity": 4,
                "impact": 4,
                "risk": 1,
                "conciseness": 4,
                "alignment_score": 4,
                "complexity_penalty": 1,
                "protocol_score": 4,
            },
        }
    if operation == "repair":
        return {"stop": True}
    if operation == "materialization":
        fusion = data["fusion"]
        return {
            "idea_result": {
                **fusion,
                "research_question": "Can stable survey evidence improve scientific-agent idea traceability?",
                "hypothesis": "Five evidence-grounded searches improve traceability without sacrificing novelty.",
                "experiment_plan": ["Compare five-mode native search with a single-mode ungrounded ablation."],
                "data_requirements": ["A completed SurveyAgent artifact with citation traces."],
                "baselines": ["Single-mode ungrounded scientific ideation."],
                "metrics": ["Evidence traceability, novelty, and feasibility."],
                "risks": [fusion["risks"]],
                "introduction": "Scientific agents need auditable evidence during idea generation.",
                "algorithm": [{"name": "Native five-mode MCTS fusion", "steps": ["retrieve", "search", "fuse"]}],
                "reference_papers": ["Survey-Grounded Research Ideation"],
                "source_modes": data["source_modes"],
            }
        }
    raise AssertionError(operation)


def fixture_idea(title: str, component: str) -> dict[str, Any]:
    return {
        "title": title,
        "abstract": "A complete scientific idea grounded in declared survey evidence.",
        "core_contribution": "Evidence remains stable throughout idea search.",
        "method": "Run typed MCTS over evidence-grounded candidate ideas.",
        "risks": "Provider judgments may vary under repeated evaluation.",
        "components": [{"name": component, "description": "Carries explicit survey evidence provenance."}],
        "tags": ["scientific agents"],
        "root_domains": ["machine learning"],
    }


class HermeticEmbedding:
    """Explicit lightweight backend for the in-process CLI integration test only."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype="float32")
        vectors[:, 0] = 1.0
        return vectors

    def usage(self) -> dict[str, object]:
        return {
            "adapter_algorithm": "explicit.hermetic_embedding.v1",
            "embedding_dimension": self.dimension,
            "transformer_inference": False,
            "local_files_only": True,
        }


class HermeticFaissIndex:
    def __init__(self, dimension: int, count: int) -> None:
        self.d = dimension
        self.ntotal = count

    def search(self, vectors: Any, limit: int) -> tuple[np.ndarray, np.ndarray]:
        if tuple(vectors.shape) != (1, self.d):
            raise AssertionError("unexpected hermetic query embedding shape")
        size = min(limit, self.ntotal)
        return (
            np.asarray([[1.0] * size], dtype="float32"),
            np.asarray([list(range(size))], dtype="int64"),
        )


class HermeticFaiss:
    def __init__(self, *, dimension: int, count: int) -> None:
        self.dimension = dimension
        self.count = count

    def read_index(self, path: str) -> HermeticFaissIndex:
        if not Path(path).is_file():
            raise AssertionError("declared hermetic FAISS fixture is missing")
        return HermeticFaissIndex(self.dimension, self.count)


def build_hermetic_novelty_runtime(resolution: Any) -> Any:
    from research_idea_lib.resources.component_novelty import build_component_novelty_runtime

    return build_component_novelty_runtime(
        resolution,
        embedding_backend=HermeticEmbedding(3),
        faiss_backend=HermeticFaiss(dimension=3, count=1),
    )


def write_cli_resource_bundle(survey_run: Path) -> dict[str, Path]:
    survey_json = survey_run / "artifacts" / "survey.json"
    root = survey_run / "resources"
    graph_path = root / "graph" / "graph.db"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(graph_path)
    try:
        connection.execute("create table nodes (id text primary key, paper_id text, paper_title text, summary text)")
        connection.execute("create table edges (source text, target text, relation text, summary text)")
        connection.execute(
            "insert into nodes values (?, ?, ?, ?)",
            ("node:p1", "p1", "Survey-Grounded Research Ideation", "Evidence-grounded ideation."),
        )
        connection.commit()
    finally:
        connection.close()

    resources = {
        "survey": write_resource_descriptor(
            root,
            "survey",
            resource_id="survey:v1",
            kind="survey",
            schema_version="xlab.literature_survey.v1",
            algorithm_version="survey-agent.v1",
            files={"survey.json": survey_json.read_bytes()},
            capabilities=["citation_traces"],
            parent_artifact_ids=[xlab_file_artifact_id(survey_json)],
        ),
        "graph": write_resource_descriptor(
            root,
            "graph",
            resource_id="graph:v1",
            kind="graph",
            schema_version="xlab.paper_graph.v1",
            algorithm_version="sqlite-paper-graph.v1",
            existing_files=[graph_path],
            capabilities=["paper_neighbors"],
        ),
        "component_index": write_resource_descriptor(
            root,
            "component-index",
            resource_id="components:v1",
            kind="component_index",
            schema_version="xlab.component_index.v1",
            algorithm_version="faiss-flat-ip.v1",
            files={
                "faiss.index": b"tiny-index",
                "meta.json": (
                    b'{"meta":{"component-1":{'
                    b'"node_id":"core-1",'
                    b'"title":"Evidence memory",'
                    b'"description":"An auditable evidence-preserving gate.",'
                    b'"domain":"Information Science",'
                    b'"paper_ids":["p1"]}}}'
                ),
            },
            capabilities=["component_similarity"],
            dimensions={"embedding": 3},
        ),
        "keynotes": write_resource_descriptor(
            root,
            "keynotes",
            resource_id="keynotes:v1",
            kind="keynotes",
            schema_version="xlab.paper_keynotes.v1",
            algorithm_version="keynote-cache.v1",
            files={"keynotes.json": b'{"p1":"Evidence-grounded ideation"}'},
            capabilities=["paper_keynotes"],
        ),
        "models": [
            write_resource_descriptor(
                root,
                "models/outcome-private",
                resource_id="model:outcome:v1",
                kind="model",
                schema_version="xlab.model_snapshot.v1",
                algorithm_version="sentence-transformers.v1",
                files={"config.json": b"{}", "weights.bin": b"tiny-outcome-model"},
                capabilities=["sentence_embedding"],
                dimensions={"embedding": 384},
                provenance={
                    "role": "outcome_model",
                    "model_id": "sentence-transformers/all-MiniLM-L6-v2",
                },
            ),
            write_resource_descriptor(
                root,
                "models/component-private",
                resource_id="model:component:v1",
                kind="model",
                schema_version="xlab.model_snapshot.v1",
                algorithm_version="sentence-transformers.v1",
                files={"config.json": b"{}", "weights.bin": b"tiny-component-model"},
                capabilities=["sentence_embedding"],
                dimensions={"embedding": 3},
                provenance={"role": "component_model"},
            ),
        ],
    }
    write_json(
        root / "resource-manifest.json",
        {"schema_version": "xlab.research_idea.resources.v1", "bundle_id": "cli-success-fixture:v1", "resources": resources},
    )
    survey_manifest_path = survey_run / "manifest.json"
    survey_manifest = read_json(survey_manifest_path)
    survey_manifest["resource_manifest"] = "resources/resource-manifest.json"
    survey_manifest["artifacts"].append(
        {"type": "research_idea_resource_manifest", "schema_version": "1", "path": "resources/resource-manifest.json"}
    )
    write_json(survey_manifest_path, survey_manifest)
    return {
        "outcome_model": root / resources["models"][0]["path"],
        "component_model": root / resources["models"][1]["path"],
    }


def write_resource_descriptor(
    root: Path,
    relative: str,
    *,
    resource_id: str,
    kind: str,
    schema_version: str,
    algorithm_version: str,
    capabilities: list[str],
    files: dict[str, bytes] | None = None,
    existing_files: list[Path] | None = None,
    dimensions: dict[str, int] | None = None,
    parent_artifact_ids: list[str] | None = None,
    provenance: dict[str, str] | None = None,
) -> dict[str, Any]:
    resource_root = root / relative
    resource_root.mkdir(parents=True, exist_ok=True)
    for name, content in (files or {}).items():
        path = resource_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    paths = list(resource_root.rglob("*")) if existing_files is None else existing_files
    descriptors = [
        {
            "path": path.relative_to(resource_root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }
        for path in paths
        if path.is_file()
    ]
    descriptor_digest = hashlib.sha256(
        json.dumps(sorted(descriptors, key=lambda item: item["path"]), separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "resource_id": resource_id,
        "kind": kind,
        "schema_version": schema_version,
        "algorithm_version": algorithm_version,
        "path": relative,
        "digest": descriptor_digest,
        "files": descriptors,
        "parent_artifact_ids": parent_artifact_ids or [],
        "capabilities": capabilities,
        "dimensions": dimensions or {},
        "provenance": provenance or {"producer": "deterministic-cli-test-fixture"},
    }


def xlab_file_artifact_id(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(b"file\0")
    digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()}"


def add_scripts_to_path() -> None:
    import sys

    scripts_dir = PACKAGE_DIR / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def assert_public_artifacts_validate(run_dir: Path) -> None:
    for filename, schema_filename in (
        ("research_idea.json", "idea.schema.json"),
        ("idea_result.json", "idea_result.schema.json"),
        ("idea_trace.json", "idea_trace.schema.json"),
        ("idea_report.json", "idea_report.schema.json"),
    ):
        schema = read_json(PACKAGE_DIR / "schemas" / schema_filename)
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(read_json(run_dir / "artifacts" / filename))


def audit_report(run_dir: Path) -> dict[str, Any]:
    add_scripts_to_path()
    from research_idea_lib.audit import audit_artifacts

    return audit_artifacts(run_dir)


def full_contract_idea_result() -> dict[str, Any]:
    required_modes = [
        "moonshot_inventor",
        "bridge_builder",
        "steady_engineer",
        "ambitious_realist",
        "evidence_first",
    ]
    iterations = [
        {
            "iteration": index,
            "node_id": index,
            "depth": 1,
            "title": f"Candidate {mode}",
            "operator": "memory_guided_edit",
            "defects": ["underspecified_evaluation"],
            "score": 0.8,
            "visits": 1,
            "path": ["root", mode],
            "evaluation": {"novelty": 4, "feasibility": 4, "composite": 0.8},
            "idea_taste_mode": mode,
        }
        for index, mode in enumerate(required_modes, start=1)
    ]
    return {
        "title": "Fused survey-grounded idea",
        "abstract": "A provider-generated fused idea grounded in survey evidence.",
        "core_contribution": "Memory-guided survey evidence fusion for scientific agents.",
        "research_question": "How can scientific agents use survey evidence for idea search?",
        "hypothesis": "Survey-grounded MCTS improves idea traceability.",
        "method": "Run survey-grounded retrieval, research idea MCTS, and fused evaluation.",
        "experiment_plan": ["Compare provider-generated ideas against non-grounded ideation baselines."],
        "data_requirements": ["Completed SurveyAgent literature survey with citation traces."],
        "baselines": ["Non-grounded ideation baseline."],
        "metrics": ["Traceability, novelty, feasibility, and protocol completeness."],
        "risks": ["Provider and evidence coverage limitations."],
        "introduction": "Survey evidence motivates a memory-guided scientific-agent idea.",
        "components": [{"component": "Evidence memory", "explanation": "Grounds each idea component."}],
        "root_domains": ["machine learning"],
        "algorithm": [{"name": "MCTS fusion", "steps": ["retrieve", "search", "fuse"]}],
        "evidence_ids": ["ev-1"],
        "reference_ids": ["p1"],
        "reference_papers": ["Survey-Grounded Research Ideation"],
        "mcts_evolution": {
            "total_iterations": len(iterations),
            "iterations": iterations,
            "pareto_front": {mode: {"score": 0.8, "title": f"Candidate {mode}"} for mode in required_modes},
        },
        "idea_source": "fused",
        "source_modes": required_modes,
        "fusion_metadata": {"host_idea_mode": "fusion_agent", "selected_components": ["Evidence memory"]},
        "fusion_evolution": {"source_modes": required_modes, "selected_components": ["Evidence memory"]},
        "algorithm_provenance": {
            "algorithm": "xlab.research_idea.algorithm.v2",
            "runtime_profile": "xlab.research_idea.runtime.v1",
            "evidence_profile": "xlab.research_idea.evidence.v1",
            "success_profile": "xlab.research_idea.success.v1",
            "required_modes": required_modes,
            "completed_modes": required_modes,
            "all_modes_completed": True,
            "fusion_required": True,
            "fusion_used": True,
            "fusion_metadata_present": True,
            "mcts_iterations": len(iterations),
            "pareto_front_present": True,
            "search_trace_present": True,
        },
    }


def full_research_idea(source_context: dict[str, Any], workflow_trace: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "xlab.research_idea.v2",
        "topic": TOPIC,
        "research_question": "How can scientific agents use survey evidence for idea search?",
        "hypothesis": "Survey-grounded MCTS improves idea traceability.",
        "method": "Run evidence retrieval, MCTS, and fusion.",
        "expected_contribution": "A traceable research ideation method.",
        "experiment_plan": ["Compare against non-grounded ideation baselines."],
        "data_requirements": ["completed literature survey"],
        "baselines": ["non-grounded ideation"],
        "metrics": ["traceability", "novelty", "feasibility"],
        "risks": ["retrieval coverage"],
        "source_evidence": [{"paper_ids": ["p1"], "summary": "Survey evidence"}],
        "source_context": source_context,
        "workflow_trace": workflow_trace,
        "operation_trace": [],
    }


def full_idea_trace(source_context: dict[str, Any], workflow_trace: list[dict[str, Any]], idea_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "xlab.research_idea.trace.v2",
        "run_id": "audit-fixture",
        "topic": TOPIC,
        "workflow_trace": workflow_trace,
        "operation_trace": provider_operation_trace(),
        "source_context": source_context,
        "mcts_evolution": idea_result.get("mcts_evolution", {}),
        "fusion_evolution": idea_result.get("fusion_evolution", {}),
        "fusion_metadata": idea_result.get("fusion_metadata", {}),
    }


def full_workflow_trace() -> list[dict[str, Any]]:
    return [
        {"workflow": "xlab.research_idea.workflow.v1", "stage": "knowledge_acquisition", "status": "success"},
        {"workflow": "xlab.research_idea.workflow.v1", "stage": "advanced_analysis", "status": "success"},
        {"workflow": "xlab.research_idea.workflow.v1", "stage": "idea_generation", "status": "success"},
    ]


def provider_operation_trace() -> list[dict[str, Any]]:
    return [
        {"event": "llm_call", "stage": "advanced_analysis", "op_name": "xlab.research_idea.analysis.generate.v1", "status": "success", "model": "gpt-5-mini"},
        {"event": "llm_call", "stage": "idea_generation", "op_name": "xlab.research_idea.idea.materialize.v1", "status": "success", "model": "gpt-5.4"},
        {"event": "llm_call", "stage": "idea_fusion", "op_name": "xlab.research_idea.fusion.generate.v1", "status": "success", "model": "gpt-5.4"},
        {"event": "llm_call", "stage": "mcts_evaluation", "op_name": "xlab.research_idea.idea.evaluate.v1", "status": "success", "model": "gpt-5.2"},
        {"event": "llm_call", "stage": "mcts_evaluation", "op_name": "xlab.research_idea.component_novelty.evaluate.v1", "provider_operation": "xlab.research_idea.component_novelty.evaluate.v1", "status": "success", "model": "gpt-5.2"},
    ]


def full_resource_preflight() -> dict[str, Any]:
    checks = {
        "provider_available": True,
        "required_python_modules": True,
        "graph_db_present": True,
        "graph_db_has_nodes": True,
        "component_index_dir_present": True,
        "component_faiss_present": True,
        "component_metadata_present": True,
        "component_metadata_nonempty": True,
        "outcome_sentence_transformer_present": True,
        "component_novelty_model_present": True,
        "keynote_cache_present": True,
        "survey_resource_paths_resolved": True,
    }
    resources = {
        role: {
            "resource_id": f"fixture-{role}",
            "logical_uri": (
                f"xlab-cache://models/fixture-{role}"
                if role in {"outcome_model", "component_model"}
                else f"xlab-resource://fixture/{role}"
            ),
        }
        for role in (
            "survey",
            "graph",
            "component_index",
            "keynotes",
            "outcome_model",
            "component_model",
        )
    }
    return {
        "schema_version": "xlab.research_idea.resource_preflight.v1",
        "implementation": {
            "name": "xlab-native-research-idea",
            "algorithm_spec": "xlab.research_idea.algorithm.v2",
            "algorithm_spec_version": "xlab.research_idea.algorithm-spec.v2",
            "success_policy": "xlab.research_idea.success.v1",
            "resource_profile": "xlab.research_idea.evidence.v1",
            "lightweight_success_paths": False,
        },
        "passed": True,
        "checks": checks,
        "blocking_errors": [],
        "warnings": [],
        "resources": resources,
    }


def full_workflow_artifact(
    idea_result: dict[str, Any],
    source_context: dict[str, Any],
    workflow_trace: list[dict[str, Any]],
    provider_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "run": {"workflow_trace": workflow_trace, "operation_trace": provider_trace},
        "retrieval": {
            "rag_hits": [{"query": "survey evidence", "hits": source_context.get("selected_evidence", [])}],
            "references": [source_context.get("references", [])],
            "evidence": [
                {
                    "evidence_id": "ev-1",
                    "kind": "survey_evidence",
                    "text": "Survey evidence",
                    "provenance": {"title": "Survey evidence", "source": "survey"},
                    "paper_ids": ["p1"],
                }
            ],
        },
        "analysis": {
            "entries": [{"tldr": "Provider-generated analysis."}],
            "root_idea": {"title": "Root idea", "method": "Survey-grounded MCTS."},
        },
        "ideation": {
            "latest_candidate": {"title": idea_result.get("title")},
            "mode_candidates": [{"idea_taste_mode": mode} for mode in idea_result.get("source_modes", [])],
            "fusion_result": {
                "idea": {
                    field: idea_result.get(field)
                    for field in (
                        "title",
                        "core_contribution",
                        "hypothesis",
                        "method",
                        "risks",
                        "root_domains",
                        "components",
                    )
                },
                "evidence_ids": list(idea_result.get("evidence_ids", [])),
            },
        },
        "persistence": {"idea_result": idea_result},
    }


def write_success_audit_fixture(run_dir: Path) -> None:
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    source_context = {
        "references": [{"paper_id": "p1", "title": "Survey-Grounded Research Ideation"}],
        "selected_evidence": [{"paper_ids": ["p1"], "title": "Survey evidence"}],
    }
    workflow_trace = full_workflow_trace()
    idea_result = full_contract_idea_result()
    write_json(artifacts / "idea_result.json", idea_result)
    write_json(artifacts / "research_idea.json", full_research_idea(source_context, workflow_trace))
    write_json(artifacts / "idea_trace.json", full_idea_trace(source_context, workflow_trace, idea_result))
    write_full_state_provenance(run_dir, idea_result, source_context, workflow_trace)


def write_full_state_provenance(
    run_dir: Path,
    idea_result: dict[str, Any],
    source_context: dict[str, Any],
    workflow_trace: list[dict[str, Any]],
    *,
    provider_trace: list[dict[str, Any]] | None = None,
) -> None:
    trace = provider_operation_trace() if provider_trace is None else provider_trace
    write_json(run_dir / "state" / "resource_preflight.json", full_resource_preflight())
    write_json(run_dir / "state" / "research_idea" / "artifact.json", full_workflow_artifact(idea_result, source_context, workflow_trace, trace))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
