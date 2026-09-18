#!/usr/bin/env python3
"""Operator entrypoint: schedule only this pipeline's Slurm jobs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from knowledge_graph_lib.common import atomic_write_json


MODEL_IMAGE = 'registry.cluster.local:5000/users/hanqi-li/vllm-ascend@sha256:49d22ba9fcb1151be41282da46615260bb327d1e1adf791522d1cda2d040b576'
PARSER_IMAGE = 'registry.cluster.local:5000/users/chaolei-liu/speech-parser@sha256:d1a69e579b4717e1c8d4b10fba91c4589a7ff9072ad12cd17a2b6ca33fd1cece'
MODEL_PATH = '/shared/models/Qwen3.8-27B-W8A8'
SCRIPTS = Path(__file__).resolve().parent


def container_args(image: str, key: str, *, model: bool = False, cpu: bool = False) -> list[str]:
    args = ['sudo', '-n', 'slurm-docker-run', '--pull', 'missing', '--network', 'host', '--shm-size', '32g' if model else '16g',
            '--mount', '/shared/chaolei.liu:/shared/chaolei.liu', '--env', 'SPEECH_PIPELINE_KEY=' + key,
            '--workdir', '/local/job', '--env', 'PYTHONUNBUFFERED=1', '--env', 'OMP_NUM_THREADS=' + ('8' if model else '4'),
            '--env', 'HCCL_BUFFSIZE=512', '--env', 'PYTORCH_NPU_ALLOC_CONF=expandable_segments:True']
    if model or cpu:
        args += ['--mount', MODEL_PATH + ':/models/Qwen3.8-27B-W8A8:ro']
    if model:
        args += ['--env', 'VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000']
    if cpu:
        args += ['--env', 'USE_TORCH=0', '--env', 'USE_TF=0', '--env', 'TOKENIZERS_PARALLELISM=false']
    return args + [image]


def pilot_client(root: Path) -> int:
    key = os.environ['SPEECH_PIPELINE_KEY']
    args = ['srun', '-p', 'mineru', '-N1', '-n1', '--nodelist=n03', '-c16', '--mem=64G', '--time=06:00:00', '--gres=none']
    args += container_args(MODEL_IMAGE, key, cpu=True)
    args += ['python3', str(SCRIPTS / 'corpus_pipeline.py'), 'pilot', '--corpus', str(root.parent), '--run-dir', str(root)]
    return subprocess.call(args)


def parser_card_count(root: Path) -> int:
    path = root / 'resource-plan.json'
    cards = json.loads(path.read_text())['parser_cards'] if path.exists() else 6
    if not isinstance(cards, int) or cards not in (2, 4, 6):
        raise ValueError('Parser allocation must be 2, 4 or 6 cards')
    return cards


def run_allocated(root: Path, role: str, replica: int, parser_cards: int = 6) -> int:
    key = os.environ['SPEECH_PIPELINE_KEY']
    if role in ('main', 'replica'):
        args = ['srun', '-n1', '-c64' if role == 'main' else '-c32', '--mem=256G', '--gres=gpu:ascend910b3:2']
        return subprocess.call(args + container_args(MODEL_IMAGE, key, model=True) + ['python3', str(SCRIPTS / 'corpus_serve.py'), '--run-dir', str(root), '--replica', str(replica)])
    if role == 'flow':
        args = ['srun', '-n1', '-c16', '--mem=64G', '--gres=none']
        return subprocess.call(args + container_args(MODEL_IMAGE, key, cpu=True) + ['python3', str(SCRIPTS / 'corpus_pipeline.py'), 'run', '--corpus', str(root.parent), '--run-dir', str(root)])
    children = []
    for card in range(parser_cards):
        command = ['srun', '--exclusive', '--exact', '-n1', '-c16', '--mem=48G', '--gres=gpu:ascend910b3:1']
        command += container_args(PARSER_IMAGE, key)
        command += ['bash', str(SCRIPTS / 'corpus_parser_card.sh'), str(root), str(card)]
        children.append(subprocess.Popen(command))
    return int(any(child.wait() != 0 for child in children))


def submit(root: Path, role: str, replica: int = 0, *, launcher: str | None = None, after_job: str | None = None) -> str:
    parser_cards = parser_card_count(root)
    resources = {'main': (2, 64, 256), 'flow': (0, 16, 64), 'parser': (parser_cards, parser_cards * 16, parser_cards * 48 + 32), 'replica': (2, 32, 256)}
    cards, cpus, memory = resources[role]
    operation = root / 'operations'
    operation.mkdir(exist_ok=True)
    script = Path(launcher).resolve() if launcher else operation / f'{role}-{replica}.sbatch'
    if launcher and not script.is_relative_to(operation.resolve()):
        raise ValueError('Recovery launcher must belong to this run operations directory')
    # Secrets are inherited through Slurm's private job environment, not script text.
    command = [sys.executable, str(Path(__file__).resolve()), 'allocated', '--run-dir', str(root), '--role', role, '--replica', str(replica), '--parser-cards', str(parser_cards)]
    if not launcher:
        script.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec ' + shlex.join(command) + '\n')
    result = subprocess.run(['sbatch', '--parsable', '-p', 'mineru', '-N1', '-n1', '--nodelist=n03', f'-c{cpus}',
                             f'--mem={memory}G', f'--gres=gpu:ascend910b3:{cards}' if cards else '--gres=none', '--tmp=1T' if role == 'parser' else '--tmp=100G',
                             '--time=2-00:00:00', '--export=ALL', f'--job-name=speech-{role}-{replica}',
                             f'--output={root}/{role}-{replica}-%j.log',
                             *(['--dependency=afterany:' + after_job] if after_job else []), str(script)], capture_output=True, text=True, check=True)
    return result.stdout.strip().split(';')[0]


def supervise(root: Path, old_job: str) -> None:
    pilot = json.loads((root / 'pilot.json').read_text())
    if not pilot.get('passed'):
        raise RuntimeError('The end-to-end pilot must pass before production migration')
    ledger_path = root / 'jobs.json'
    jobs = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    if not jobs:
        main = submit(root, 'main')
        parser = submit(root, 'parser')
        flow = submit(root, 'flow')
        jobs = [{'id': main, 'role': 'main', 'replica': 0}, {'id': parser, 'role': 'parser', 'replica': 0}, {'id': flow, 'role': 'flow', 'replica': 0}]
        atomic_write_json(ledger_path, jobs)
        subprocess.run(['tmux', 'kill-session', '-t', 'speech-parse-supervisor-20260912'], check=False)
        subprocess.run(['scancel', old_job], check=True)
    while True:
        jobs = json.loads(ledger_path.read_text())
        status_path = root / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get('phase') in ('complete', 'incomplete'):
            for job in jobs:
                subprocess.run(['scancel', job['id']], check=False)
            return
        blocked = False
        for job in jobs:
            queue = subprocess.run(['squeue', '-h', '-j', job['id'], '-o', '%T'], capture_output=True, text=True).stdout.strip()
            if queue:
                continue
            if job['role'] == 'parser' and (root / 'parse-complete.json').exists():
                continue
            if job['role'] == 'replica' and (root / 'extraction-complete.json').exists():
                continue
            control = subprocess.run(['scontrol', 'show', 'job', '-o', job['id']], capture_output=True, text=True)
            state_match = re.search(r'\bJobState=(\S+)', control.stdout)
            accounting = state_match.group(1) if state_match else ''
            exit_path = root / 'operations' / f"job-{job['id']}-exit.json"
            if not accounting and exit_path.exists():
                accounting = 'COMPLETED' if json.loads(exit_path.read_text())['exit_code'] == 0 else 'FAILED'
            if any(value in accounting for value in ('TIMEOUT', 'NODE_FAIL', 'PREEMPTED')):
                job['previous_id'] = job['id']
                job['id'] = submit(root, job['role'], job['replica'], launcher=job.get('launcher'))
                atomic_write_json(ledger_path, jobs)
            else:
                atomic_write_json(root / 'deployment-blocker.json', {'job': job, 'state': accounting or 'UNKNOWN',
                                  'reason': 'Unexpected job termination; inspect job logs and replace its ledger entry after repair', 'time': time.time()})
                blocked = True
        if blocked:
            time.sleep(30)
            continue
        if (root / 'parse-complete.json').exists() and not (root / 'extraction-complete.json').exists():
            existing = {job['replica'] for job in jobs if job['role'] == 'replica'}
            for replica in (1, 2, 3):
                if replica not in existing:
                    jobs.append({'id': submit(root, 'replica', replica), 'role': 'replica', 'replica': replica})
            atomic_write_json(ledger_path, jobs)
        if (root / 'extraction-complete.json').exists():
            for job in jobs:
                if job['role'] == 'replica':
                    subprocess.run(['scancel', job['id']], check=False)
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['pilot-client', 'allocated', 'supervise', 'implement'])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--role', choices=['main', 'flow', 'parser', 'replica'])
    parser.add_argument('--replica', type=int, default=0)
    parser.add_argument('--parser-cards', type=int, choices=(2, 4, 6), default=6)
    parser.add_argument('--old-job', default='9945')
    args = parser.parse_args()
    if args.command == 'implement':
        pilot_path = args.run_dir / 'pilot.json'
        if not pilot_path.exists() or not json.loads(pilot_path.read_text()).get('passed') or 'survey_exit' not in json.loads(pilot_path.read_text()):
            code = pilot_client(args.run_dir)
            if code:
                raise SystemExit(code)
        supervise(args.run_dir, args.old_job)
        return
    if args.command == 'pilot-client':
        raise SystemExit(pilot_client(args.run_dir))
    if args.command == 'allocated':
        code = 1
        try:
            code = run_allocated(args.run_dir, args.role, args.replica, args.parser_cards)
        finally:
            atomic_write_json(args.run_dir / 'operations' / f"job-{os.environ['SLURM_JOB_ID']}-exit.json",
                              {'role': args.role, 'replica': args.replica, 'exit_code': code, 'time': time.time()})
        raise SystemExit(code)
    supervise(args.run_dir, args.old_job)


if __name__ == '__main__':
    main()
