"""Handoff authorization and preparation gates, with no real Slurm mutations."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import subprocess
from unittest.mock import patch

import yaml

from playground.sure_master.tools import launch_asr_corrected as launcher


class AsrActivationTests(unittest.TestCase):
    def test_stopped_batch_client_is_not_needed_for_service_termination(self):
        with patch.object(launcher, 'command', side_effect=['', subprocess.CalledProcessError(1, ['scontrol'])]) as command:
            launcher.stop_service_step('9980.0')
        self.assertEqual(command.call_args_list[0].args[0], ['scancel', '--signal=TERM', '9980.0'])

    def test_protected_allocations_never_reach_a_slurm_command(self):
        for job in ('9991', '9992', '12345'):
            with patch.object(launcher, 'command') as command, self.assertRaises(ValueError):
                launcher.verify_handoff(job)
            command.assert_not_called()

    def test_handoff_requires_exact_service_owner_node_and_step(self):
        info = f'JobId=9980 JobName=qwen38-27b-w8a8-tp8 UserId=user({os.getuid()}) JobState=RUNNING NodeList=n09 '
        step = f'StepId=9980.0 UserId={os.getuid()} SrunHost:Pid=bms-4821-0009:42 '
        with patch.object(launcher, 'command', side_effect=[info, step]):
            result = launcher.verify_handoff('9980')
            self.assertEqual(result['batch_client_pid'], 42)
            self.assertEqual(result['training_npus'], 4)
        for wrong in (info.replace('n09', 'n13'), info.replace(f'({os.getuid()})', f'({os.getuid()+1})'),
                      info.replace('qwen38-27b-w8a8-tp8', 'another-task')):
            with patch.object(launcher, 'command', return_value=wrong), self.assertRaises(ValueError):
                launcher.verify_handoff('9980')
        with patch.object(launcher, 'command', return_value=info.replace('RUNNING', 'COMPLETED')):
            self.assertIsNone(launcher.verify_handoff('9980'))

    def test_failed_checks_leave_service_and_controller_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'runs').mkdir()
            marker = root/'prepared.json'
            marker.write_text(json.dumps({'features_ready': True}))
            config = root/'config.yaml'
            config.write_text(yaml.safe_dump({'sure': {'execution_env': {'SURE_ASR_PREPARATION': str(marker)}}}))
            checks = root/'checks.json'
            checks.write_text(json.dumps({'exit_code': 1}))
            args = SimpleNamespace(output=root/'runs/new', config=config, preparation_job='',
                                   checks_receipt=checks, handoff_job='9980')
            with (patch.object(launcher, 'PROJECT', root), patch.object(launcher, 'command') as command,
                  patch.object(launcher.subprocess, 'Popen') as popen):
                with self.assertRaisesRegex(RuntimeError, 'checks have not passed'):
                    launcher.activate(args)
            command.assert_not_called()
            popen.assert_not_called()


if __name__ == '__main__':
    unittest.main()
