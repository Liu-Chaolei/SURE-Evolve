import json
import os
from pathlib import Path
import tempfile
import unittest
import fcntl
from unittest.mock import patch

from .slurm_allocations import owned_running, run_in_allocations


class ExistingAllocationsTests(unittest.TestCase):
    def test_owned_allocation_required(self):
        with patch('playground.sure_master.core.utils.slurm.command',
                   return_value=f'UserId=other({os.getuid()+1}) JobState=RUNNING '):
            with self.assertRaises(ValueError):
                owned_running('123')

    def test_step_publishes_result_and_reconnect_does_not_rerun(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / 'result.json'
            fake = root / 'srun'
            fake.write_text('#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\n'
                            'Path(sys.argv[-1]).write_text(json.dumps(sys.argv[1:]))\n')
            fake.chmod(0o700)
            settings = {'existing_allocations': ['123'], 'allocation_lock_dir': str(root / 'locks')}
            with patch('playground.sure_master.core.utils.slurm_allocations.owned_running', return_value=True):
                self.assertTrue(run_in_allocations(settings, [str(fake), str(result)], root, result))
                args = json.loads(result.read_text())
                self.assertIn('--jobid=123', args)
                self.assertIn('--exact', args)
                self.assertIn('--cpus-per-task=64', args)
                fake.unlink()
                self.assertTrue(run_in_allocations(settings, ['missing'], root, result))

    def test_expired_allocations_allow_standard_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = {'existing_allocations': ['123'], 'allocation_lock_dir': str(root / 'locks')}
            with patch('playground.sure_master.core.utils.slurm_allocations.owned_running', return_value=False):
                self.assertFalse(run_in_allocations(settings, ['unused'], root, root / 'result.json'))

    def test_busy_lease_can_queue_other_candidates_normally(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pool = root / 'locks'
            pool.mkdir()
            settings = {'existing_allocations': ['123'], 'allocation_lock_dir': str(pool),
                        'allocation_fallback_when_busy': True}
            with (pool / '123.lock').open('a') as lease:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch('playground.sure_master.core.utils.slurm_allocations.owned_running', return_value=True):
                    self.assertFalse(run_in_allocations(settings, ['must-not-launch'], root, root/'result.json'))

    def test_handoff_step_requests_exclusive_resources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / 'result.json'
            fake = root / 'srun'
            fake.write_text('#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\n'
                            'Path(sys.argv[-1]).write_text(json.dumps(sys.argv[1:]))\n')
            fake.chmod(0o700)
            settings = {'existing_allocations': ['123'], 'allocation_lock_dir': str(root/'locks'),
                        'allocation_overlap': False}
            with patch('playground.sure_master.core.utils.slurm_allocations.owned_running', return_value=True):
                self.assertTrue(run_in_allocations(settings, [str(fake), str(result)], root, result))
            args = json.loads(result.read_text())
            self.assertIn('--exclusive', args)
            self.assertNotIn('--overlap', args)
