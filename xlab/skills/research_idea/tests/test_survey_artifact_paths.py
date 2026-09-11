"""Real filesystem regression for workspace-relative Survey manifests."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from research_idea_lib.survey_repository import resolve_artifact_path


class SurveyArtifactPathTests(unittest.TestCase):
    def test_workspace_relative_manifest_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            cwd = Path(temporary)
            run = cwd / ".xlab/runs/survey"
            artifact = run / "artifacts/survey.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("{}")
            self.assertEqual(resolve_artifact_path(".xlab/runs/survey/artifacts/survey.json", cwd, run), artifact)
            self.assertEqual(resolve_artifact_path("artifacts/survey.json", cwd, run), artifact)

    def test_missing_path_and_external_symlink(self):
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as external:
            cwd = Path(temporary)
            run = cwd / "run"
            run.mkdir()
            self.assertEqual(resolve_artifact_path("missing.json", cwd, run), run / "missing.json")
            (run / "escape").symlink_to(external, target_is_directory=True)
            (cwd / "escape").symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "escapes"):
                resolve_artifact_path("escape/secret", cwd, run)


if __name__ == "__main__":
    unittest.main()
