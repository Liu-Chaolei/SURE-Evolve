import shlex
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from research_idea_lib.inputs import parse_request_args


class EqualsInputTests(unittest.TestCase):
    def parse(self, tokens):
        return parse_request_args(shlex.join(tokens), Path('/tmp'), SimpleNamespace(provider_available=False))

    def test_explicit_values_are_not_reinterpreted(self):
        for text in ['candidate_type=arch, seed=42', '--literal=a=b', 'line1\n{"x":"y=z"}', "it's quoted"]:
            request = self.parse(['--survey', '/tmp/survey.json', '--topic', text, '--discussion', text])
            self.assertEqual(request.topic, text)
            self.assertEqual(request.discussion, text)

    def test_supported_shorthand(self):
        request = self.parse(['survey=/tmp/survey.json', 'topic=seed=42', 'discussion=a=b', 'resume=true'])
        self.assertEqual(request.topic, 'seed=42')
        self.assertEqual(request.discussion, 'a=b')
        self.assertTrue(request.resume)

    def test_unknown_key_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            self.parse(['--survey', '/tmp/survey.json', 'unsupported=1'])


if __name__ == '__main__':
    unittest.main()
