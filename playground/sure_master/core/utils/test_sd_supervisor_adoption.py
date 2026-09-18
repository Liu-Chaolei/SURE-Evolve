"""Supervisor replacement must retain the cross-deployment execution boundary."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from playground.sure_master.tools.sd_deployments import launch, register


class SupervisorAdoptionTests(unittest.TestCase):
    def test_adoption_keeps_existing_arm_and_starts_only_supervisor(self):
        with tempfile.TemporaryDirectory() as temporary:
            runs = Path(temporary)
            root = runs / 'current'
            root.mkdir()
            register(runs, 'current', {'root': str(root), 'status': 'running'})
            process = Mock()
            process.wait.return_value = 0
            with patch('playground.sure_master.tools.sd_deployments.live_processes', side_effect=[[123], []]), \
                 patch('playground.sure_master.tools.sd_deployments.subprocess.Popen', return_value=process) as popen:
                launch(runs, 'current', adopt_running=True)
                popen.assert_called_once_with(['bash', str(root / 'launch.engine.sh')])

    def test_adoption_never_ignores_another_deployment(self):
        with tempfile.TemporaryDirectory() as temporary:
            runs = Path(temporary)
            root = runs / 'current'
            root.mkdir()
            register(runs, 'other', {'root': str(runs / 'other'), 'status': 'running'})
            register(runs, 'current', {'root': str(root), 'status': 'running'})
            with patch('playground.sure_master.tools.sd_deployments.live_processes', return_value=[123]), \
                 patch('playground.sure_master.tools.sd_deployments.subprocess.Popen') as popen:
                with self.assertRaisesRegex(RuntimeError, 'other still has live'):
                    launch(runs, 'current', adopt_running=True)
                popen.assert_not_called()


if __name__ == '__main__':
    unittest.main()
