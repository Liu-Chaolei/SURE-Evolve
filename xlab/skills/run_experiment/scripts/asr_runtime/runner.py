"""Frozen Zipformer condition execution and SURE scoring inside allocated Slurm jobs.

No experiment scheduling or model-provider calls live in this module.
"""
from __future__ import annotations

import argparse
import ast
import fcntl
import gzip
import hashlib
import importlib.util
import json
import math
import os
import pickle
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(root.rglob('*')):
        if file.is_file() and '__pycache__' not in file.parts:
            digest.update(str(file.relative_to(root)).encode())
            digest.update(file.read_bytes())
    return digest.hexdigest()


def prepare_recipe(source: Path, target: Path) -> None:
    shutil.copytree(source, target, symlinks=False, ignore=shutil.ignore_patterns('__pycache__', 'exp', 'log'))
    for name in ('train.py', 'decode.py', 'model.py'):
        path = target / name
        text = path.read_text()
        if name == 'train.py':
            text = text.replace('from icefall.dist import cleanup_dist, setup_dist',
                                'from asr_runtime.distributed import cleanup_dist, setup_dist')
            text = text.replace('torch.set_num_threads(1)', 'torch.set_num_threads(8)')
            text = text.replace('from torch.cuda.amp import GradScaler', 'from torch_npu.npu.amp import GradScaler')
            text = text.replace('k2.RaggedTensor(y).to(device)', 'k2.RaggedTensor(y)')
        text = text.replace('torch.cuda', 'torch.npu').replace('torch.device("cuda",', 'torch.device("npu",')
        text = 'import torch_npu\n' + text
        if name == 'model.py':
            text = text.replace('import k2\n', 'import k2\nfrom asr_runtime import npu_k2\n')
            for function in ('rnnt_loss_smoothed', 'rnnt_loss_pruned', 'get_rnnt_prune_ranges'):
                if f'k2.{function}(' not in text:
                    raise ValueError(f'Unsupported native recipe: missing {function}')
                text = text.replace(f'k2.{function}(', f'npu_k2.{function}(')
            text = text.replace('self.decoder(sos_y_padded)', 'self.decoder(sos_y_padded.to(encoder_out.device))')
        if name == 'decode.py':
            old = '''    dev_cuts = tedlium.dev_cuts()
    test_cuts = tedlium.test_cuts()

    dev_dl = tedlium.test_dataloaders(dev_cuts)
    test_dl = tedlium.test_dataloaders(test_cuts)

    test_sets = ["dev", "test"]
    test_dls = [dev_dl, test_dl]
'''
            new = '''    import os
    from lhotse import CutSet
    ref_ids = set()
    with open(os.environ["XLAB_EVAL_REF"]) as source:
        for line in source:
            ref_ids.add(line.split("\\t", 1)[0])
    split = os.environ["XLAB_EVAL_SPLIT"]
    cuts = tedlium.test_cuts() if split == "test" else tedlium.dev_cuts()
    cuts = CutSet.from_cuts(cut for cut in cuts if cut.id in ref_ids)
    if set(cut.id for cut in cuts) != ref_ids:
        raise ValueError("Evaluation manifest does not cover the frozen reference IDs")
    test_sets = [split]
    test_dls = [tedlium.test_dataloaders(cuts)]
'''
            if old not in text:
                raise ValueError('Native TEDLIUM decode split binding has changed')
            text = text.replace(old, new)
        path.write_text(text)
    scaling = target / 'scaling.py'
    scaling.write_text(scaling.read_text().replace('import k2\n', 'from asr_runtime import npu_k2 as k2\n'))


def prepare_project(config: dict, project: Path) -> dict:
    project.mkdir(parents=True, exist_ok=True)
    marker = project / 'preparation.json'
    if marker.exists():
        return json.loads(marker.read_text())
    recipe = project / 'recipe'
    if recipe.exists():
        raise RuntimeError('Partial preparation exists; refusing to overwrite editable model sources')
    prepare_recipe(Path(config['recipe']), recipe)
    shutil.copytree(Path(config['icefall']) / 'icefall', project / 'library' / 'icefall',
                    ignore=shutil.ignore_patterns('__pycache__'))
    (project / 'method.py').write_text('''"""Scientific configuration; edit this and recipe modules for the canonical Idea."""
import json
import os

def component_enabled(name):
    return name not in json.loads(os.environ.get("XLAB_DISABLED_COMPONENTS", "[]"))

def configuration(disabled_components):
    return {"train_args": ["--enable-musan", "0"], "decode_args": [], "decoding_method": "greedy_search"}
''')
    result = {'status': 'prepared', 'data': config['data'], 'recipe_source': config['recipe'],
              'recipe_digest': tree_digest(recipe), 'library_digest': tree_digest(project / 'library'),
              'method_digest': hashlib.sha256((project / 'method.py').read_bytes()).hexdigest(),
              'startup': 'direct_formal', 'runtime_checks': 'deferred_to_formal_science'}
    write_json(marker, result)
    return result


def static_check(project: Path, components: list[str]) -> dict:
    for path in [*project.joinpath('recipe').rglob('*.py'), project / 'method.py']:
        ast.parse(path.read_text(), filename=str(path))
    if len(set(components)) != len(components) or any(not name.strip() for name in components):
        raise ValueError('Canonical components must have unique nonempty names')
    if not components:
        baseline = json.loads((project / 'preparation.json').read_text())
        if tree_digest(project / 'recipe') != baseline['recipe_digest'] or hashlib.sha256((project / 'method.py').read_bytes()).hexdigest() != baseline['method_digest']:
            raise ValueError('The zero-component baseline must preserve its declared prepared recipe and method configuration exactly')
    source = '\n'.join(path.read_text() for path in [*project.joinpath('recipe').rglob('*.py'), project / 'method.py'])
    for name in components:
        if name not in source:
            raise ValueError(f'No implementation binding found for canonical component {name}')
    return {'status': 'success', 'syntax': True, 'component_contract': components,
            'recipe_digest': tree_digest(project / 'recipe'),
            'method_digest': hashlib.sha256((project / 'method.py').read_bytes()).hexdigest(),
            'runtime_evidence': 'deferred_to_formal_science', 'smoke_executed': False}


def run_command(argv: list[str], log: Path, cwd: Path) -> None:
    print(json.dumps({'command': argv, 'log': str(log)}), flush=True)
    with log.open('a') as stream:
        subprocess.run(argv, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=True)


def references(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        key, text = line.split('\t', 1)
        if not key or key in result:
            raise ValueError('Duplicate or empty reference ID')
        result[key] = text
    if not result:
        raise ValueError('Empty reference')
    return result


def score(config: dict, ref: Path, hyp: Path, output: Path) -> dict:
    sys.path.insert(0, config['sure_pythonpath'])
    from sure_eval.evaluation.cli_adapters import build_pipeline_spec, run_pipeline_spec
    expected = references(ref)
    actual = references(hyp)
    if set(expected) != set(actual):
        raise ValueError('Prediction IDs must match all frozen reference IDs exactly')
    pipeline = build_pipeline_spec('asr', language='en', metric='wer')
    if pipeline['pipeline_id'] != config['pipeline_id']:
        raise ValueError('SURE scoring pipeline changed')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'pipeline.json', pipeline)
    result = run_pipeline_spec(pipeline, output_dir=str(output), device='cpu',
                               ref_file=str(ref), hyp_file=str(hyp))
    value = float(result['score'])
    if not math.isfinite(value) or value < 0:
        raise ValueError('Nonfinite or negative WER')
    result.update(score=value, coverage=len(expected), reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),
                  hypothesis_sha256=hashlib.sha256(hyp.read_bytes()).hexdigest())
    write_json(output / 'score.json', result)
    return result


def stage_data(config: dict, output: Path, split: str) -> Path:
    """Bind train and only the authorized dev/test reference IDs for this condition."""
    source = Path(config['data'])
    data = output / 'data'
    data.mkdir(exist_ok=True)
    for entry in source.iterdir():
        if entry.name == 'fbank':
            continue
        target = data / entry.name
        if not target.exists():
            target.symlink_to(entry, target_is_directory=entry.is_dir())
    fbank = data / 'fbank'
    fbank.mkdir(exist_ok=True)
    allowed = set(references(Path(config['refs'][split])))
    selected = 'test' if split == 'test' else 'dev'
    for entry in (source / 'fbank').iterdir():
        target = fbank / entry.name
        if entry.name == f'tedlium_cuts_{selected}.jsonl.gz':
            found = set()
            with gzip.open(entry, 'rt') as reader, gzip.open(target.with_suffix('.tmp'), 'wt') as writer:
                for line in reader:
                    cut = json.loads(line)
                    if cut['id'] in allowed:
                        if cut['id'] in found:
                            raise ValueError('Duplicate evaluation cut ID')
                        found.add(cut['id'])
                        writer.write(line)
            if found != allowed:
                raise ValueError('Prepared evaluation cuts do not cover frozen reference IDs')
            target.with_suffix('.tmp').replace(target)
        elif entry.name.startswith('tedlium_cuts_') and ('dev' in entry.name or 'test' in entry.name):
            continue
        elif not target.exists():
            target.symlink_to(entry, target_is_directory=entry.is_dir())
    return data


def execute(request: dict) -> dict:
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('ASR execution requires a Slurm allocation')
    import torch
    import torch_npu  # noqa: F401
    config = request['config']
    project, output = Path(request['project']), Path(request['output'])
    output.mkdir(parents=True, exist_ok=True)
    if torch.npu.device_count() != 4:
        raise RuntimeError('Expected exactly four allocated NPUs')
    disabled = request['disabled_components']
    inherited_disabled = request.get('inherited_disabled_components', [])
    effective_disabled = sorted(set(disabled + inherited_disabled))
    os.environ['XLAB_DISABLED_COMPONENTS'] = json.dumps(effective_disabled)
    sys.path.insert(0, str(project))
    spec = importlib.util.spec_from_file_location('xlab_method', project / 'method.py')
    method = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(method)
    settings = method.configuration(effective_disabled)
    train_args, decode_args = settings['train_args'], settings['decode_args']
    forbidden = {'--world-size', '--num-epochs', '--start-epoch', '--start-batch', '--seed', '--use-fp16',
                 '--exp-dir', '--bpe-model', '--max-duration', '--manifest-dir', '--master-port', '--epoch', '--avg', '--use-averaged-model'}
    for arguments in (train_args, decode_args):
        if not isinstance(arguments, list) or any(not isinstance(arg, str) for arg in arguments):
            raise ValueError('Method arguments must be string arrays')
        if any(arg.split('=', 1)[0] in forbidden for arg in arguments):
            raise ValueError('Method cannot override frozen resource, data or training budgets')
    identity = {'source_digest': request['source_digest'], 'disabled_components': disabled,
                'inherited_disabled_components': inherited_disabled,
                'training': config['training']}
    binding = output / 'training_binding.json'
    if binding.exists() and json.loads(binding.read_text()) != identity:
        raise ValueError('Cannot reuse checkpoints across changed scientific conditions')
    write_json(binding, identity)
    ports = Path(config['ports_dir'])
    ports.mkdir(parents=True, exist_ok=True)
    lease = None
    for offset in range(128):
        port = 20000 + (int(os.environ['SLURM_JOB_ID']) + offset) % 30000
        candidate = (ports / str(port)).open('a')
        try:
            fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', port))
            lease = candidate
            break
        except (OSError, BlockingIOError):
            candidate.close()
    if lease is None:
        raise RuntimeError('No rendezvous port available')
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), PYTHONDONTWRITEBYTECODE='1',
                      OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                      XDG_CACHE_HOME='/local/job/.cache', TORCH_HOME='/local/job/.cache/torch',
                      ASCEND_CACHE_PATH='/local/job/ascend/cache', ASCEND_PROCESS_LOG_PATH='/local/job/ascend/log')
    for name in ('XDG_CACHE_HOME', 'TORCH_HOME', 'ASCEND_CACHE_PATH', 'ASCEND_PROCESS_LOG_PATH'):
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
    library = project / 'library'
    runtime = Path(__file__).resolve().parents[1]
    os.environ['PYTHONPATH'] = os.pathsep.join(map(str, [runtime, project, library]))
    split = request.get('split', 'regular')
    data = stage_data(config, output, split)
    models = Path(request.get('checkpoint_dir') or output / 'models')
    models.mkdir(parents=True, exist_ok=True)
    if request.get('evaluation_only') is not True:
        completed = 0
        for checkpoint in sorted(models.glob('epoch-*.pt'), key=lambda p: int(p.stem.split('-')[1]), reverse=True):
            try:
                value = torch.load(checkpoint, map_location='cpu', weights_only=False)
                epoch = int(checkpoint.stem.split('-')[1])
                if all(key in value for key in ('model', 'optimizer', 'scheduler', 'batch_idx_train')) and int(value['cur_epoch']) == epoch:
                    completed = epoch
                    break
            except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError):
                continue
        if completed < 30:
            run_command([sys.executable, str(project / 'recipe/train.py'), '--world-size', '4',
                         '--num-epochs', '30', '--start-epoch', str(completed + 1), '--seed', '42',
                         '--use-fp16', '0', '--exp-dir', str(models), '--bpe-model', str(data / 'lang_bpe_500/bpe.model'),
                         '--master-port', str(port), '--max-duration', '900', *train_args], output / 'train.log', output)
    checkpoint = models / 'epoch-30.pt'
    trained = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if int(trained['cur_epoch']) != 30 or int(trained.get('batch_idx_train', 0)) <= 0 or not all(key in trained for key in ('model', 'optimizer', 'scheduler')):
        raise ValueError('Formal training did not produce a complete epoch-30 checkpoint')
    del trained
    ref = Path(config['refs'][split])
    os.environ.update(XLAB_EVAL_REF=str(ref), XLAB_EVAL_SPLIT='test' if split == 'test' else 'dev')
    # Each evaluation uses an isolated directory while linking the exact frozen checkpoint.
    decoding = output / f'decode_{split}'
    decoding.mkdir(exist_ok=True)
    linked = decoding / checkpoint.name
    if not linked.exists():
        linked.symlink_to(checkpoint)
    run_command([sys.executable, '-m', 'asr_runtime.npu_decode', str(project / 'recipe/decode.py'), '--epoch', '30', '--avg', '1',
                 '--use-averaged-model', 'false', '--exp-dir', str(decoding), '--bpe-model',
                 str(data / 'lang_bpe_500/bpe.model'), '--max-duration', '30', '--decoding-method',
                 settings['decoding_method'], *decode_args], output / f'decode_{split}.log', output)
    hypotheses = {}
    for recogs in decoding.rglob('recogs-*.txt'):
        for line in recogs.read_text().splitlines():
            match = re.match(r'^(.*?):\thyp=(.*)$', line)
            if not match:
                continue
            key, raw = match.groups()
            if key in hypotheses:
                raise ValueError(f'Duplicate prediction {key}; decoder must select one output stream')
            try:
                words = ast.literal_eval(raw)
                hypotheses[key] = ' '.join(map(str, words)) if isinstance(words, (list, tuple)) else str(words)
            except (ValueError, SyntaxError):
                hypotheses[key] = raw
    expected = references(ref)
    if set(hypotheses) != set(expected):
        raise ValueError('Decoded hypotheses do not exactly cover the requested split')
    hyp = output / f'hyp_{split}.txt'
    hyp.write_text(''.join(f'{key}\t{hypotheses[key]}\n' for key in expected))
    result = score(config, ref, hyp, output / f'sure_{split}')
    report = {'status': 'success', 'job_id': os.environ['SLURM_JOB_ID'], 'condition_id': request['condition_id'],
              'disabled_components': disabled, 'enabled_components': request['enabled_components'],
              'inherited_disabled_components': inherited_disabled,
              'source_digest': request['source_digest'], 'checkpoint': str(checkpoint), 'checkpoint_dir': str(models),
              'epoch': 30, 'split': split, 'metric': 'wer', 'score': result['score'], 'coverage': result['coverage'],
              'sure_report': str(output / f'sure_{split}/score.json'), 'output': str(output), 'project': str(project)}
    write_json(output / 'result.json', report)
    lease.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['prepare', 'static', 'execute'])
    parser.add_argument('request')
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    if args.operation == 'prepare':
        result = prepare_project(request['config'], Path(request['project']))
    elif args.operation == 'static':
        result = static_check(Path(request['project']), request['components'])
    else:
        result = execute(request)
    if request.get('receipt'):
        write_json(Path(request['receipt']), result)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
