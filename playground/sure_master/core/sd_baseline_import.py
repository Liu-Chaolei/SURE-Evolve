"""Verify and relocate an SD baseline independently of its research engine."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import shutil

import yaml

from .artifacts import bundle_resources, file_digest
from .training import require_completion, validate_training_config, training_precision, canonical_digest
from .utils.slurm import atomic_json

PACKAGE = Path(__file__).resolve().parents[1]


def code_identity(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return {p.relative_to(root).as_posix(): file_digest(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.toml', '.yaml', '.json'}
            and not set(p.relative_to(root).parts) & {'.git', '.cache', '__pycache__'}}


def scoring_identity(sure: dict) -> dict:
    root = Path(sure['pythonpath']) / 'sure_eval/evaluation'
    result = {name: code_identity(root/name) for name in ('core', 'tasks/sd', 'nodes/scoring/meeteval')}
    result['pipeline_identity'] = file_digest(root/'pipeline_identity.py')
    return result


def baseline_identity(sure: dict) -> dict:
    training = validate_training_config('sd.diarizen', sure['task']['training'], sure['runtime'])
    resources = sure['task']['resources']
    splits = {name: {'manifest': file_digest(Path(spec['manifest'])),
                    'roles': {key: file_digest(Path(value)) for key, value in spec.get('roles', {}).items()}}
              for name, spec in sure['datasets'].items()}
    return {'adapter': sure['adapter'], 'task_id': sure['task_id'], 'training': training,
            'backend': sure['runtime']['accelerator'], 'world_size': sure['runtime']['world_size'],
            'training_precision': training_precision('sd.diarizen', training),
            'inference_precision': sure['runtime'].get('precision', 'fp32'),
            'inference': sure['task']['inference'], 'splits': splits,
            'preparation': file_digest(Path(sure['data_preparation'])),
            'initialization': file_digest(Path(resources['wavlm'])),
            'initialization_provenance': file_digest(Path(resources['wavlm']).with_suffix(Path(resources['wavlm']).suffix+'.provenance.json')),
            'embedding': file_digest(Path(resources['embedding'])),
            'model_source': code_identity(Path(resources['source'])),
            'baseline_code': file_digest(Path(sure['initial_solution_path'])),
            'scoring': scoring_identity(sure), 'worker_image': sure['slurm']['image']}


def verify_baseline(previous_run: Path, source_config: Path, sure: dict):
    source_config.resolve().relative_to(previous_run.resolve())
    original = yaml.safe_load(source_config.read_text())['sure']
    old_identity, new_identity = baseline_identity(original), baseline_identity(sure)
    differences = [key for key in old_identity if old_identity[key] != new_identity[key]]
    if differences:
        raise ValueError('SD baseline scientific settings changed: '+', '.join(differences))
    state_path = previous_run/'search/workspace/metric/controller_state.json'
    state = json.loads(state_path.read_text())
    if state.get('completed_rounds') != 0 or state.get('candidates'):
        raise ValueError('SD import requires a baseline-only run')
    baseline = state['baseline']
    if type(baseline.get('score')) not in (int,float) or not math.isfinite(baseline['score']):
        raise ValueError('Invalid baseline score')
    if Path(sure['initial_solution_path']).read_text().strip() != baseline['code'].strip():
        raise ValueError('SD baseline implementation changed')
    manifest, resources = bundle_resources(baseline['model_artifact'])
    if manifest['adapter'] != 'sd.diarizen':
        raise ValueError('Expected an SD model bundle')
    completion = require_completion(resources['training_evidence']/'training_completion.json')
    contract = completion['contract']
    expected_data = {key: new_identity['splits'][key]['manifest'] for key in ('train','train_validation')}
    expected_data['preparation'] = new_identity['preparation']
    if contract['data'] != expected_data or contract['training'] != new_identity['training']:
        raise ValueError('Baseline completion does not match training/data identity')
    if contract['initial_checkpoint_sha256'] != new_identity['initialization']:
        raise ValueError('Baseline SSL checkpoint changed')
    if contract['precision'] != new_identity['training_precision']:
        raise ValueError('Baseline training precision changed')
    dependencies = {}
    for key, expected in contract['source_files'].items():
        if key.startswith('sure:'):
            path = PACKAGE/key[5:]
            if not path.is_file() or file_digest(path) != expected:
                raise ValueError('Baseline training implementation changed: '+key)
            dependencies[key] = expected
    reports = list((previous_run/'search/workspace').glob('exp_*_draft/metric/report.json'))
    if len(reports) != 1:
        raise ValueError('Expected one baseline score report')
    report = json.loads(reports[0].read_text())
    if report['score'] != baseline['score'] or report['metric'].lower() != 'der':
        raise ValueError('Baseline score differs from its report')
    if json.loads((previous_run/'data_fingerprints.json').read_text()) != {k:v['manifest'] for k,v in new_identity['splits'].items()}:
        raise ValueError('Baseline split fingerprint changed')
    verification = {'scientific_identity': canonical_digest(new_identity), 'identity': new_identity,
                    'training_dependencies': dependencies, 'source_run': str(previous_run),
                    'source_state_sha256': file_digest(state_path), 'source_report_sha256': file_digest(reports[0]),
                    'score': baseline['score'], 'pipeline_id': report['pipeline_id'],
                    'epochs_completed': completion['epochs_completed'], 'training_performed': False}
    return dict(baseline), verification, reports[0]


def import_sd_baseline(previous_run: Path, sure: dict, workspace: Path):
    baseline, verification, _ = verify_baseline(previous_run, Path(sure['initial_baseline_config']), sure)
    atomic_json(workspace/'metric/baseline_import.json', {**verification, 'model_artifact': baseline['model_artifact']})
    return baseline, {'model_artifact': baseline['model_artifact']}


def copy_baseline(previous_run: Path, destination: Path, sure: dict) -> dict:
    baseline, verification, report = verify_baseline(previous_run, previous_run/'deployment.yaml', sure)
    original_bundle = Path(baseline['model_artifact'])
    bundle = destination/'search/workspace/model_artifacts'/original_bundle.parent.name
    if bundle.exists():
        raise ValueError('Baseline destination already exists')
    shutil.copytree(original_bundle.parent, bundle, symlinks=False)
    baseline['model_artifact'] = str(bundle/'manifest.json')
    bundle_resources(baseline['model_artifact'])
    metric = destination/'search/workspace/exp_imported_draft/metric'
    metric.mkdir(parents=True)
    for name in ('report.json','score_summary.json','pipeline_description.json','pipeline_spec.json'):
        path = report.parent/name
        if path.exists():
            shutil.copy2(path, metric/name)
    artifacts = report.parent.parent/'artifacts'
    if artifacts.is_dir():
        shutil.copytree(artifacts, metric.parent/'artifacts')
    atomic_json(destination/'search/workspace/metric/controller_state.json', {
        'completed_rounds':0, 'candidates':[], 'baseline':baseline,
        'attributes':{'best_model_artifact':{'model_artifact':baseline['model_artifact']}},
        'imported':True})
    atomic_json(destination/'data_fingerprints.json', {name:value['manifest'] for name,value in verification['identity']['splits'].items()})
    atomic_json(destination/'baseline_import.json', verification)
    result = {'status':'baseline_ready','baseline':baseline,'origin':{'kind':'imported',**verification}}
    atomic_json(destination/'result.json', result)
    return result
