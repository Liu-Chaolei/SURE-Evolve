import fcntl
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .allocation_handoff import eligible, ready_allocation
from .slurm_allocations import run_in_allocations


class RecoveryAllocationTests(unittest.TestCase):
    def test_second_slot_runs_when_first_is_leased(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / 'srun'
            fake.write_text('#!/usr/bin/env python3\nimport sys,json\nfrom pathlib import Path\nPath(sys.argv[-1]).write_text(json.dumps(sys.argv[1:]))\n')
            fake.chmod(0o700)
            settings = dict(existing_allocations=['123'], allocation_lock_dir=tmp,
                            allocation_slots_per_job=2, allocation_cpus=32, allocation_overlap=False)
            result = root / 'result.json'
            with (root/'123.slot-0.lock').open('a') as lease:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch('playground.sure_master.core.utils.slurm_allocations.owned_running', return_value=True):
                    self.assertTrue(run_in_allocations(settings, [str(fake), str(result)], root, result))
            receipt = json.loads((root/'allocation.json').read_text())
            self.assertEqual(receipt['slot'], 1)
            self.assertIn('--cpus-per-task=32', receipt['command'])
            self.assertNotIn('--overlap', receipt['command'])

    def test_no_takeover_before_demand(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = {'authorized_service_takeover': {'job_id': '10028', 'receipt_dir': tmp}}
            with patch('playground.sure_master.core.utils.allocation_handoff.command') as command:
                self.assertIsNone(ready_allocation(settings))
                command.assert_not_called()

    def test_takeover_identity_and_protected_nodes(self):
        info = f'UserId=user({os.getuid()}) JobState=RUNNING NodeList=n03 NumCPUs=64 mem=256G gres/gpu:ascend910b3:8 '
        self.assertTrue(eligible(info, '10028'))
        self.assertFalse(eligible(info.replace('RUNNING', 'PENDING'), '10028'))
        for node in ['n11', 'n13']:
            with self.assertRaises(ValueError):
                eligible(info.replace('n03', node), '10028')
        with self.assertRaises(ValueError):
            eligible(info, '10029')
        with self.assertRaises(ValueError):
            eligible(info.replace(f'({os.getuid()})', f'({os.getuid()+1})'), '10028')

    def test_acquired_receipt_replays_without_signalling(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, '10028.json').write_text(json.dumps({'status': 'ready'}))
            info = f'JobName=asr-v2-allocation UserId=user({os.getuid()}) JobState=RUNNING NodeList=n03 NumCPUs=64 mem=256G gres/gpu:ascend910b3:8 '
            with patch('playground.sure_master.core.utils.allocation_handoff.command', return_value=info) as cmd:
                self.assertEqual(ready_allocation({'authorized_service_takeover': {'job_id':'10028','receipt_dir':tmp}}, acquire=True), '10028')
                self.assertEqual(cmd.call_count, 1)

    def test_handoff_stops_only_service_step_and_preserves_holder(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = f'JobName=qwen38-27b-w8a8-tp8 UserId=user({os.getuid()}) JobState=RUNNING NodeList=n03 NumCPUs=64 mem=256G gres/gpu:ascend910b3:8 '
            step = 'StepId=10028.0 SrunHost:Pid=n03:42 '
            stats = '42 (srun) ' + ' '.join(['S'] + ['0'] * 18 + ['identity'])
            steps = iter([step, step, 'StepId=10028.batch '])
            def command(argv):
                if argv[:3] == ['scontrol', 'show', 'job']: return info
                if argv[:3] == ['scontrol', 'show', 'step']: return next(steps)
                return ''
            def remote(job, argv):
                if argv[0] == 'ps': return 'srun' if argv[-1] == 'comm=' else 'T'
                if argv[0] == 'cat': return '/slurm/uid_1/job_10028/step_batch' if argv[1].endswith('cgroup') else stats
                return ''
            with patch('playground.sure_master.core.utils.allocation_handoff.command', side_effect=command) as cmd, \
                 patch('playground.sure_master.core.utils.allocation_handoff.remote', side_effect=remote) as rem, \
                 patch('playground.sure_master.core.utils.allocation_handoff.time.sleep'):
                self.assertEqual(ready_allocation({'authorized_service_takeover': {'job_id':'10028','receipt_dir':tmp}}, acquire=True), '10028')
            self.assertIn(['scancel','--signal=TERM','10028.0'], [c.args[0] for c in cmd.call_args_list])
            self.assertNotIn(['scancel','10028'], [c.args[0] for c in cmd.call_args_list])
            self.assertIn(('10028',['kill','-STOP','42']), [c.args for c in rem.call_args_list])
            self.assertEqual(json.loads(Path(tmp,'10028.json').read_text())['status'], 'ready')

    def test_prepared_handoff_rejects_reused_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, '10028.json').write_text(json.dumps({'status':'prepared','holder_pid':42,'holder_identity':'old'}))
            info = f'JobName=qwen38-27b-w8a8-tp8 UserId=user({os.getuid()}) JobState=RUNNING NodeList=n03 NumCPUs=64 mem=256G gres/gpu:ascend910b3:8 '
            stats = '42 (srun) ' + ' '.join(['S'] + ['0'] * 18 + ['new'])
            with patch('playground.sure_master.core.utils.allocation_handoff.command',return_value=info), \
                 patch('playground.sure_master.core.utils.allocation_handoff.remote',return_value=stats) as rem:
                with self.assertRaisesRegex(ValueError,'PID was reused'):
                    ready_allocation({'authorized_service_takeover': {'job_id':'10028','receipt_dir':tmp}}, acquire=True)
                self.assertEqual(rem.call_args.args[1], ['cat','/proc/42/stat'])
