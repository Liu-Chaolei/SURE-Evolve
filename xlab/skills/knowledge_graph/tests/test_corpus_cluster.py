from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import corpus_cluster


class StopMonitor(Exception):
    pass


class ClusterTests(unittest.TestCase):
    def test_parser_allocation_and_launch_count_are_frozen_together(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'resource-plan.json').write_text('{"parser_cards":2}')
            response = subprocess.CompletedProcess([], 0, '12345\n', '')
            with patch.object(corpus_cluster.subprocess, 'run', return_value=response) as submit:
                self.assertEqual(corpus_cluster.submit(root, 'parser', after_job='100'), '12345')
            args = submit.call_args.args[0]
            self.assertIn('--gres=gpu:ascend910b3:2', args)
            self.assertIn('-c32', args)
            self.assertIn('--mem=128G', args)
            self.assertIn('--dependency=afterany:100', args)
            script = (root / 'operations/parser-0.sbatch').read_text()
            self.assertIn('--parser-cards 2', script)
            child = Mock()
            child.wait.return_value = 0
            with patch.dict(os.environ, {'SPEECH_PIPELINE_KEY': 'test-only'}), patch.object(corpus_cluster.subprocess, 'Popen', return_value=child) as launch:
                self.assertEqual(corpus_cluster.run_allocated(root, 'parser', 0, parser_cards=2), 0)
            self.assertEqual(launch.call_count, 2)

    def test_parse_completion_adds_only_the_remaining_replica(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'pilot.json').write_text('{"passed":true}')
            (root / 'parse-complete.json').write_text('{}')
            jobs = [{'id': str(index), 'role': role, 'replica': replica} for index, (role, replica) in enumerate(
                [('main', 0), ('flow', 0), ('parser', 0), ('replica', 1), ('replica', 2)], 1)]
            (root / 'jobs.json').write_text(json.dumps(jobs))
            response = subprocess.CompletedProcess([], 0, 'RUNNING\n', '')
            with patch.object(corpus_cluster.subprocess, 'run', return_value=response), patch.object(corpus_cluster, 'submit', return_value='6') as submit, patch.object(corpus_cluster.time, 'sleep', side_effect=StopMonitor):
                with self.assertRaises(StopMonitor):
                    corpus_cluster.supervise(root, 'unused')
            submit.assert_called_once_with(root, 'replica', 3)
            updated = json.loads((root / 'jobs.json').read_text())
            self.assertEqual({row['replica'] for row in updated if row['role'] == 'replica'}, {1, 2, 3})


if __name__ == '__main__':
    unittest.main()
