"""Scientific baseline identity survives path/research changes, not recipe changes."""
from copy import deepcopy
from pathlib import Path
import shutil
import json
import tempfile
import unittest

from playground.sure_master.core.sd_baseline_import import baseline_identity
from playground.sure_master.core.training import SD_TRAINING
from playground.sure_master.tools.sd_deployments import inventory, register, retry_failed, prepare_resume
from playground.sure_master.core.utils.fingerprints import digest


class DeploymentMigrationTests(unittest.TestCase):
    def config(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        for name in ('wavlm.bin', 'wavlm.bin.provenance.json', 'embedding.bin', 'baseline.py', 'preparation.json'):
            (root/name).write_text('fixture')
        model = root/'model'
        model.mkdir()
        (model/'model.py').write_text('class Model: pass\n')
        scoring = root/'score/sure_eval/evaluation'
        for name in ('core', 'tasks/sd', 'nodes/scoring/meeteval'):
            (scoring/name).mkdir(parents=True)
            (scoring/name/'implementation.py').write_text('FIXTURE = True\n')
        (scoring/'pipeline_identity.py').write_text('VERSION = 1\n')
        splits = {}
        for name in ('train','train_validation','search','selection','holdout'):
            (root/(name+'.jsonl')).write_text('{}\n')
            splits[name] = {'manifest':str(root/(name+'.jsonl'))}
        return {'task_id':'sd_der', 'adapter':'sd.diarizen',
            'task':{'training':{**deepcopy(SD_TRAINING), 'recipe':'diarizen.evolution.v1.bf16',
                                'batch_size':48, 'candidate_options':{}},
                    'resources':{'source':str(model), 'wavlm':str(root/'wavlm.bin'), 'embedding':str(root/'embedding.bin')},
                    'inference':{'ahc_threshold':0.7}},
            'runtime':{'accelerator':'npu','world_size':4,'precision':'fp32','training_precision':'bf16'},
            'datasets':splits, 'pythonpath':str(root/'score'), 'data_preparation':str(root/'preparation.json'),
            'initial_solution_path':str(root/'baseline.py'), 'slurm':{'image':'registry/image@sha256:fixture'}}

    def test_relocation_and_research_changes_preserve_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = self.config(root/'first'), self.config(root/'second')
            first['execution_contract'] = {'constraints':['free research']}
            second['execution_contract'] = {'constraints':['native MCTS refinement']}
            self.assertEqual(baseline_identity(first), baseline_identity(second))

    def test_training_scoring_or_model_changes_do_not_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary)/'fixture')
            before = baseline_identity(config)
            altered = deepcopy(config)
            altered['task']['training']['batch_size'] = 16
            with self.assertRaises(ValueError):
                baseline_identity(altered)
            altered = deepcopy(config)
            altered['task']['inference']['ahc_threshold'] = .9
            self.assertNotEqual(before, baseline_identity(altered))
            (Path(config['task']['resources']['source'])/'model.py').write_text('class Changed: pass\n')
            self.assertNotEqual(before, baseline_identity(config))

    def test_archive_inventory_tracks_symlinks_without_copying_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'record.json').write_text('{}')
            (root/'.env').write_text('DO_NOT_ARCHIVE=secret')
            (root/'dataset').symlink_to('/nonexistent/external/data')
            result = inventory(root)
            self.assertEqual(set(result['files']), {'record.json'})
            self.assertEqual(result['links'], {'dataset':'/nonexistent/external/data'})

    def test_registered_names_cannot_silently_change_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            register(root, 'direct24', {'root':'/old', 'status':'archived_incomplete'})
            register(root, 'direct24', {'root':'/old', 'status':'running'})
            with self.assertRaises(ValueError):
                register(root, 'direct24', {'root':'/new'})

    def test_explicit_retry_preserves_cached_response_and_receipt_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root/'A/search/workspace/artifacts/xlab_operations.json'
            path.parent.mkdir(parents=True)
            base = {'schema_version':'sure.xlab_receipts.v1', 'operations':{'request-1':{'status':'incomplete'}}}
            path.write_text(json.dumps({**base, 'document_digest':digest(base)}))
            cached = root/'analysis.json'
            cached.write_text('{"preserved":true}')
            (root/'control.json').write_text('{"active_groups":["A"]}')
            with self.assertRaisesRegex(RuntimeError, 'reconcile'):
                prepare_resume(root)
            retry_failed(root, 'A', 'request-1')
            result = json.loads(path.read_text())
            self.assertEqual(result['operations'], {})
            self.assertEqual(result['document_digest'], digest({key:result[key] for key in ('schema_version','operations')}))
            self.assertEqual(cached.read_text(), '{"preserved":true}')
            self.assertEqual(len(list((root/'recovery').glob('*.json'))), 1)


if __name__ == '__main__':
    unittest.main()
