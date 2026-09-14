"""Verify a completed TTS baseline before importing it into a fresh search."""
import json
import math
from pathlib import Path

import yaml

from .artifacts import bundle_resources, file_digest
from .training import require_completion
from .utils.slurm import atomic_json


def import_tts_baseline(previous_run: Path, sure: dict, workspace: Path):
    config_path = Path(sure['initial_baseline_config']).resolve()
    config_path.relative_to(previous_run.resolve())
    original = yaml.safe_load(config_path.read_text())['sure']
    for key in ('task_id', 'adapter', 'task', 'datasets', 'execution_contract', 'data_preparation', 'runtime'):
        if original.get(key) != sure.get(key):
            raise ValueError(f'TTS baseline scientific setting changed: {key}')
    state_path = previous_run / 'search/workspace/metric/controller_state.json'
    state = json.loads(state_path.read_text())
    if state.get('completed_rounds') != 0 or state.get('candidates'):
        raise ValueError('TTS import requires a baseline-only run')
    baseline = state['baseline']
    if type(baseline.get('score')) not in (float, int) or not math.isfinite(baseline['score']):
        raise ValueError('Invalid TTS baseline score')
    if Path(sure['initial_solution_path']).read_text().strip() != baseline['code'].strip():
        raise ValueError('TTS baseline implementation changed')
    manifest_path = Path(baseline['model_artifact'])
    manifest, resources = bundle_resources(manifest_path)
    if manifest['adapter'] != 'tts.f5tts':
        raise ValueError('Expected TTS F5 artifact')
    completion = require_completion(resources['training_evidence'] / 'training_completion.json')
    training = sure['task']['training']
    data_paths = {'preparation': sure['data_preparation'],
                  'train': sure['datasets']['train']['manifest'],
                  'train_validation': sure['datasets']['train_validation']['manifest'],
                  'train_csv': training['manifest']}
    if completion['contract']['data'] != {key:file_digest(Path(path)) for key,path in data_paths.items()}:
        raise ValueError('TTS training data content changed')
    if completion['contract']['initial_checkpoint_sha256'] != file_digest(Path(sure['task']['resources']['checkpoint'])):
        raise ValueError('TTS initialization checkpoint changed')
    if completion['epochs_completed'] != training['epochs'] or completion['contract']['training'] != {k:v for k,v in training.items() if k != 'manifest'}:
        raise ValueError('TTS baseline training completion differs')
    if file_digest(resources['vocab']) != file_digest(Path(sure['task']['resources']['vocab'])):
        raise ValueError('TTS baseline vocabulary changed')
    drafts = list((previous_run/'search/workspace').glob('exp_*_draft/metric/report.json'))
    if len(drafts) != 1:
        raise ValueError('Expected one scored TTS baseline')
    report = json.loads(drafts[0].read_text())
    summary = json.loads(drafts[0].with_name('score_summary.json').read_text())
    if report['score'] != baseline['score'] or summary['score'] != baseline['score']:
        raise ValueError('TTS baseline score differs from its report')
    if report['metric'] != 'cer' or report['pipeline_id'] != summary['pipeline_id']:
        raise ValueError('TTS scoring protocol differs')
    data = [json.loads(line) for line in Path(sure['datasets']['search']['manifest']).read_text().splitlines() if line.strip()]
    scored = {row['sample_id']: row for row in report['details']['rows']}
    if len(data) != 400 or len(scored) != 400 or {r['sample_id'] for r in data} != set(scored):
        raise ValueError('TTS baseline must cover all 400 search samples')
    for row in data:
        item = scored[row['sample_id']]
        if item['reference_text'] != row['target_text'] or Path(item['reference_audio']).resolve() != Path(row['reference_audio']).resolve():
            raise ValueError('TTS search conditioning changed')
    preparation = json.loads(Path(sure['data_preparation']).read_text())
    for split in ('search', 'selection', 'holdout'):
        if file_digest(Path(sure['datasets'][split]['manifest'])) != preparation['digests'][split]:
            raise ValueError(f'TTS dataset changed: {split}')
    if state['attributes']['best_model_artifact']['model_artifact'] != str(manifest_path):
        raise ValueError('Imported best model is not the baseline')
    atomic_json(workspace/'metric/baseline_import.json', {
        'previous_run':str(previous_run), 'source_state_sha256':file_digest(state_path),
        'source_config':str(config_path), 'model_artifact':str(manifest_path),
        'score':baseline['score'], 'training_performed':False,
        'epochs_completed':completion['epochs_completed'], 'optimizer_updates':completion['optimizer_updates'],
        'search_manifest_sha256':file_digest(Path(sure['datasets']['search']['manifest'])),
        'score_report_sha256':file_digest(drafts[0]), 'pipeline_id':report['pipeline_id']})
    return dict(baseline), dict(state['attributes']['best_model_artifact'])
