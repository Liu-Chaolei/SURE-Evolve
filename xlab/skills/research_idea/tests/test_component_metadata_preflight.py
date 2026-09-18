"""Resource preflight must accept the same metadata formats as retrieval."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from research_idea_lib.pipeline import component_metadata_nonempty


class ComponentMetadataPreflightTests(unittest.TestCase):
    def test_supported_forms_preserve_resource_bytes(self):
        for payload in ({'meta': [{'id': 'c1', 'description': 'attention'}]},
                        {'meta': {'c1': {'description': 'attention'}}},
                        [{'id': 'c1'}]):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'meta.json'
                path.write_text(json.dumps(payload))
                original = path.read_bytes()
                self.assertTrue(component_metadata_nonempty(path))
                self.assertEqual(path.read_bytes(), original)

    def test_empty_or_malformed_records_are_rejected(self):
        for payload in ({'meta': []}, {'meta': {}}, {'meta': ['invalid']}, None):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'meta.json'
                path.write_text(json.dumps(payload))
                self.assertFalse(component_metadata_nonempty(path))


if __name__ == '__main__':
    unittest.main()
