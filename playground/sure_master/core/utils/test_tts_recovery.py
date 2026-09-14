import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from .test_free_idea_generation import TrainingController, TrainingProvider
from . import test_xlab_controller_smoke as smoke


class RecoveryTests(unittest.TestCase):
    def test_ideas_only_imports_without_experiments_and_replays_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, ref = root/'baseline.py', root/'ref.txt'
            baseline.write_text('# baseline\n')
            ref.write_text('utt-1\thello\n')
            path = smoke.ControllerSmokeTests()._write_config(root, baseline, ref)
            config = yaml.safe_load(path.read_text())
            config['sure'].pop('staged_axes')
            config['sure'].update(search_strategy='ordinary', search_scope='all',
                initial_baseline_run=str(root/'old'), search_budget={'ideas_per_round':3})
            config['xlab']['idea_generation']={'max_attempts':8}
            path.write_text(yaml.safe_dump(config))
            provider = TrainingProvider()
            imported = ({'idea_id':'baseline','score':0.8,'code':'# baseline','model_artifact':None},{})
            with patch('playground.sure_master.core.baseline_import.import_baseline',return_value=imported), \
                 patch.object(TrainingController,'execute_parallel_tasks',side_effect=AssertionError('must not execute')):
                first = TrainingController(config_path=path,xlab_provider=provider)
                result=first.run('recover',ideas_only=True)
                self.assertEqual(result['status'],'ideas_ready')
                self.assertEqual(result['candidate_count'],3)
                state=json.loads((Path(first.session.config.workspace_path)/'metric/controller_state.json').read_text())
                self.assertEqual(state['completed_rounds'],0)
                self.assertEqual(state['candidates'],[])
                second=TrainingController(config_path=path,xlab_provider=provider)
                self.assertEqual(second.run('recover',ideas_only=True)['status'],'ideas_ready')
                self.assertEqual(len(provider.generated_requests),1)


if __name__ == '__main__':
    unittest.main()
