from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PACKAGE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from research_idea_lib.checkpoints import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
    STATE_LAYOUT_ID,
    checkpoint_signature,
    read_checkpoint,
    stage_completed,
    stage_dependency_signature,
    write_checkpoint,
)
from research_idea_lib.common import atomic_write_json, run_paths  # noqa: E402


class CheckpointV2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="xlab-research-idea-checkpoint-test-"))
        self.run_dir = self.tmp / "run"
        (self.run_dir / "state").mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_writes_v2_run_relative_outputs_and_verifies_digest(self) -> None:
        output = run_paths(self.run_dir)["survey_context"]
        atomic_write_json(output, {"papers": ["p1"]})
        dependency_signature = stage_dependency_signature("ingest_survey", {"survey": "digest-1"})

        checkpoint = write_checkpoint(
            self.run_dir,
            "ingest_survey",
            completed_stage="ingest_survey",
            dependency_signature=dependency_signature,
            output_paths=[output],
        )

        self.assertEqual(checkpoint["schema_version"], CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(checkpoint["state_layout_id"], STATE_LAYOUT_ID)
        self.assertEqual(checkpoint["outputs"]["survey_context"], "state/survey_context.json")
        self.assertEqual(
            set(checkpoint["stages"]["ingest_survey"]["output_digests"]),
            {"state/survey_context.json"},
        )
        self.assertTrue(
            stage_completed(
                self.run_dir,
                "ingest_survey",
                [output],
                dependency_signature=dependency_signature,
            )
        )

        atomic_write_json(output, {"papers": ["changed"]})
        self.assertFalse(
            stage_completed(
                self.run_dir,
                "ingest_survey",
                [output],
                dependency_signature=dependency_signature,
            )
        )

    def test_stage_dependency_signatures_are_stage_specific_and_ignore_live_capability(self) -> None:
        base = {"model": "gpt", "provider_available": False, "nested": {"api_key_set": False}}
        live = {"model": "gpt", "provider_available": True, "nested": {"api_key_set": True}}

        self.assertEqual(
            stage_dependency_signature("resource_preflight", base),
            stage_dependency_signature("resource_preflight", live),
        )
        self.assertNotEqual(
            stage_dependency_signature("resource_preflight", base),
            stage_dependency_signature("idea_generation", base),
        )
        self.assertEqual(
            checkpoint_signature({}, base, {}),
            checkpoint_signature({}, live, {}),
        )

    def test_compatibility_wrappers_record_and_check_stage_scoped_signature(self) -> None:
        output = run_paths(self.run_dir)["context"]
        atomic_write_json(output, {"topic": "agents"})
        request, runtime, source = checkpoint_signature(
            {"topic": "agents"},
            {"model": "gpt"},
            {"survey": "digest-1"},
        )
        write_checkpoint(
            self.run_dir,
            "organize_context",
            completed_stage="organize_context",
            request_signature=request,
            runtime_config_signature=runtime,
            source_signatures={"survey": "digest-1"},
        )

        self.assertTrue(
            stage_completed(
                self.run_dir,
                "organize_context",
                [output],
                request_signature=request,
                runtime_config_signature=runtime,
                source_signature=source,
            )
        )
        self.assertFalse(
            stage_completed(
                self.run_dir,
                "organize_context",
                [output],
                request_signature="changed-request",
                runtime_config_signature=runtime,
                source_signature=source,
            )
        )

    def test_explicit_none_clears_last_error_but_omission_preserves_it(self) -> None:
        failed = write_checkpoint(self.run_dir, "idea_generation", last_error="temporary provider failure")
        self.assertEqual(failed["last_error"], "temporary provider failure")

        preserved = write_checkpoint(self.run_dir, "idea_generation")
        self.assertEqual(preserved["last_error"], "temporary provider failure")

        succeeded = write_checkpoint(self.run_dir, "idea_generation", last_error=None)
        self.assertIsNone(succeeded["last_error"])

    def test_v1_checkpoint_is_rejected_without_translation(self) -> None:
        output = run_paths(self.run_dir)["workflow_retrieval"]
        atomic_write_json(output, {"hits": []})
        self._write_v1(
            completed_stages=["knowledge_aquisition"],
            outputs={"workflow_retrieval": str(output.resolve())},
            phase="knowledge_aquisition",
        )

        self.assertIsNone(read_checkpoint(self.run_dir))

    def test_v1_checkpoint_with_external_paths_is_rejected_without_translation(self) -> None:
        external = self.tmp / "external.json"
        atomic_write_json(external, {"not": "owned by run"})
        self._write_v1(
            completed_stages=["ingest_survey", "organize_context"],
            outputs={
                "survey_context": str(external),
                "context": "../outside-context.json",
            },
            phase="organize_context",
        )

        self.assertIsNone(read_checkpoint(self.run_dir))

    def test_v2_checkpoint_without_canonical_state_layout_is_rejected(self) -> None:
        checkpoint = write_checkpoint(self.run_dir, "idea_generation")
        checkpoint.pop("state_layout_id")
        run_paths(self.run_dir)["checkpoint"].write_text(json.dumps(checkpoint), encoding="utf-8")

        self.assertIsNone(read_checkpoint(self.run_dir))

    def test_checkpoint_write_rejects_secret_values_without_creating_file(self) -> None:
        path = run_paths(self.run_dir)["checkpoint"]

        with self.assertRaisesRegex(ValueError, "secret-shaped value at data.api_key"):
            write_checkpoint(
                self.run_dir,
                "idea_generation",
                {"api_key": "provider-key-without-known-prefix"},
            )

        self.assertFalse(path.exists())

    def test_checkpoint_write_rejects_absolute_host_paths_without_creating_file(self) -> None:
        path = run_paths(self.run_dir)["checkpoint"]

        with self.assertRaisesRegex(ValueError, "absolute or unsafe path at data.artifact_path"):
            write_checkpoint(
                self.run_dir,
                "idea_generation",
                {"artifact_path": str((self.tmp / "host" / "artifact.json").resolve())},
            )

        self.assertFalse(path.exists())

    def test_checkpoint_write_allows_nonsecret_token_metadata_and_relative_paths(self) -> None:
        checkpoint = write_checkpoint(
            self.run_dir,
            "idea_generation",
            {"token": "token budget", "artifact_path": "artifacts/idea.json"},
        )

        self.assertEqual(checkpoint["data"]["token"], "token budget")
        self.assertTrue(run_paths(self.run_dir)["checkpoint"].exists())

    def test_rejects_required_output_outside_run_even_when_it_exists(self) -> None:
        output = run_paths(self.run_dir)["survey_context"]
        external = self.tmp / "external.json"
        atomic_write_json(output, {"papers": []})
        atomic_write_json(external, {"external": True})
        signature = stage_dependency_signature("ingest_survey", {})
        write_checkpoint(
            self.run_dir,
            "ingest_survey",
            completed_stage="ingest_survey",
            dependency_signature=signature,
            output_paths=[output],
        )

        self.assertFalse(
            stage_completed(
                self.run_dir,
                "ingest_survey",
                [external],
                dependency_signature=signature,
            )
        )

    def _write_v1(self, *, completed_stages: list[str], outputs: dict[str, str], phase: str) -> None:
        checkpoint = {
            "schema_version": "xlab.research_idea.checkpoint.v1",
            "phase": phase,
            "current_phase": phase,
            "updated_at": "2026-01-01T00:00:00Z",
            "resumable": True,
            "completed_stages": completed_stages,
            "request_signature": "request",
            "runtime_config_signature": "runtime",
            "source_signatures": {"survey": "source"},
            "outputs": outputs,
            "counts": {},
            "warnings": [],
            "last_error": None,
            "data": {},
        }
        path = run_paths(self.run_dir)["checkpoint"]
        path.write_text(json.dumps(checkpoint), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
