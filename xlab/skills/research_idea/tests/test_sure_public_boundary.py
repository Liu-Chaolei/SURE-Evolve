"""SURE source context stays private while strict operation schemas stay intact."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from research_idea_lib.manifest import sanitize_public_paths, validate_public_projection
from research_idea_lib.research_policy import ResearchPolicy


class SurePublicBoundaryTests(unittest.TestCase):
    def test_private_source_is_not_embedded_in_public_artifact(self):
        original = {"source_context": {"task_context": {"base_model_profile": {
            "implementation_task": "Use /shared/project/source.py; width = size // 2"}},
            "research_policy": {"evidence_mode": "task_only"}}}
        before = copy.deepcopy(original)
        public = sanitize_public_paths(original)
        context = public["source_context"]["task_context"]
        self.assertEqual(set(context), {"private_context_digest"})
        self.assertTrue(context["private_context_digest"].startswith("sha256:"))
        self.assertEqual(original, before)
        self.assertEqual(public, sanitize_public_paths(public))
        validate_public_projection({"research_idea.json": public}, {})

    def test_source_excerpt_redaction_is_idempotent_and_guard_remains(self):
        original = {"summary": "width = size // 2; load /shared/secret/source.py"}
        public = sanitize_public_paths(original)
        self.assertEqual(public, sanitize_public_paths(public))
        validate_public_projection({"research_idea.json": public}, {})
        with self.assertRaisesRegex(ValueError, "PUBLIC_PATH_DISCLOSURE"):
            validate_public_projection({"research_idea.json": original}, {})

    def test_policy_preserves_exact_operation_schemas(self):
        for mode in ("local_literature", "task_only"):
            prompt = ResearchPolicy(evidence_mode=mode).prompt()
            self.assertIn("exact output schema", prompt)
            self.assertNotIn("return ablation=[]", prompt)
            self.assertNotIn("reference_papers must be []", prompt)
            self.assertIn("No auxiliary experiments", prompt)


if __name__ == "__main__":
    unittest.main()
