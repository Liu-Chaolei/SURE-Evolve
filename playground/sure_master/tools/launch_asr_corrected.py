"""Activate one corrected SURE ASR workflow after resources and checks complete.

The optional n09 handoff queues an exclusive training step before stopping the
old service step, while its stopped batch client keeps the Slurm allocation alive.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import yaml

from playground.sure_master.core.utils.slurm import atomic_json, job_state, ACTIVE
from playground.sure_master.core.utils.slurm_allocations import process_identity
from playground.sure_master.tools.freeze_tedlium_run import freeze

PROJECT = Path(__file__).resolve().parents[3]
XLAB_RUNS = Path('/shared/chaolei.liu/XLab/.xlab/runs')
CONTROLLER = '/shared/chaolei.liu/data/sure_asr_controller/bin/python'
PROTECTED_JOBS = {'9991', '9992'}


def command(argv: list[str]) -> str:
    return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT).strip()


def verify_handoff(job: str) -> dict | None:
    if job != '9980' or job in PROTECTED_JOBS:
        raise ValueError('Only the explicitly authorized n09 job 9980 can be replaced')
    try:
        info = command(['scontrol', 'show', 'job', job, '-o'])
    except subprocess.CalledProcessError:
        return None
    if 'JobState=RUNNING ' not in info:
        return None
    owner = re.search(r'UserId=\S+\((\d+)\)', info)
    node = re.search(r'\bNodeList=(\S+)', info)
    if not owner or int(owner[1]) != os.getuid() or not node or node[1] != 'n09':
        raise ValueError('Handoff job owner or node changed')
    if 'JobName=qwen38-27b-w8a8-tp8 ' not in info:
        raise ValueError('Expected the original qwen service before a fresh handoff')
    steps = command(['scontrol', 'show', 'step', job, '-o'])
    step = next((line for line in steps.splitlines() if line.startswith(f'StepId={job}.0 ')), None)
    pid = re.search(r'SrunHost:Pid=\S+:(\d+)', step or '')
    if not pid:
        raise ValueError('Cannot identify the original service batch client')
    return {'job_id': job, 'node': 'n09', 'service_step': job+'.0', 'batch_client_pid': int(pid[1]),
            'cpu_limit': 64, 'memory_limit': '256G', 'allocated_npus': 8, 'training_npus': 4}


def remote(job: str, argv: list[str]) -> str:
    return command(['srun', f'--jobid={job}', '--overlap', '--exact', '--cpus-per-task=1',
                    '--gres=none', '--ntasks=1', *argv])


def stop_service_step(step: str) -> None:
    # Default scancel can rely on the srun client; ours is deliberately stopped.
    command(['scancel', '--signal=TERM', step])
    for attempt in range(50):
        try:
            info = command(['scontrol', 'show', 'step', step, '-o'])
        except subprocess.CalledProcessError:
            return
        if f'StepId={step} ' not in info:
            return
        if attempt == 39:
            command(['scancel', '--signal=KILL', step])
        time.sleep(3)
    raise RuntimeError('Original service step did not release its resources; allocation preserved')


def retire_old_workflows(new_run: Path) -> None:
    """Retire identified ASR controllers, never jobs selected by a name prefix."""
    old_runs = []
    for root in (PROJECT/'runs').iterdir():
        if not root.is_dir() or root.resolve() == new_run.resolve():
            continue
        config = root/'deployment.yaml'
        if not config.exists():
            continue
        try:
            sure = yaml.safe_load(config.read_text()).get('sure', {})
        except (ValueError, AttributeError, yaml.YAMLError):
            continue
        if str(sure.get('task_id', '')).startswith('asr_'):
            old_runs.append(root.resolve())
    retired = []
    for root in old_runs:
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit() or int(proc.name) == os.getpid():
                continue
            try:
                if proc.stat().st_uid != os.getuid():
                    continue
                args = (proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
            except OSError:
                continue
            clean = [arg for arg in args if '\n' not in arg]
            if (any('tedlium_workflow' in arg for arg in clean)
                    and any(arg == str(root) or arg == str(root/'deployment.yaml') for arg in clean)):
                os.kill(int(proc.name), signal.SIGTERM)
                retired.append({'controller_pid': int(proc.name), 'run_dir': str(root)})
        for receipt in root.glob('search/workspace/exp_*/metric/slurm/*/job.json'):
            record = json.loads(receipt.read_text())
            job = str(record.get('job_id', ''))
            if not job.isdigit() or job in PROTECTED_JOBS or job == '9980':
                continue
            try:
                info = command(['scontrol', 'show', 'job', job, '-o'])
            except subprocess.CalledProcessError:
                continue
            owner = re.search(r'UserId=\S+\((\d+)\)', info)
            if (owner and int(owner[1]) == os.getuid() and f'Command={root}/' in info
                    and ('JobState=RUNNING ' in info or 'JobState=PENDING ' in info)):
                command(['scancel', job])
                retired.append({'job_id': job, 'run_dir': str(root)})
        atomic_json(root/'SUPERSEDED.json', {'replacement_run': str(new_run), 'at': time.time()})
    for state_path in XLAB_RUNS.glob('*/evolution.json'):
        state = json.loads(state_path.read_text())
        if not str(state.get('schema_version', '')).startswith('xlab.asr_evolution_state.'):
            continue
        root = state_path.parent
        atomic_json(root/'cancel_requested.json', {'requested_at': datetime.now(timezone.utc).isoformat(),
                                                   'reason': 'superseded by corrected SURE-Evolve'})
        atomic_json(state_path, {**state, 'status': 'cancelled', 'replacement_run': str(new_run)})
        record = root/'run.json'
        if record.exists():
            data = json.loads(record.read_text())
            atomic_json(record, {**data, 'status': 'cancelled'})
        retired.append({'xlab_run': root.name})
    atomic_json(new_run/'retired_workflows.json', {'entries': retired, 'protected_jobs': sorted(PROTECTED_JOBS)})


def activate(args) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (PROJECT/'runs/asr_activation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = output/'activation.json'
        if state_path.exists():
            previous = json.loads(state_path.read_text())
            pid = previous.get('controller_pid')
            if pid and process_identity(pid) == previous.get('controller_identity'):
                print('The corrected controller is already active', flush=True)
                return
        config = yaml.safe_load(args.config.read_text())
        marker = Path(config['sure']['execution_env']['SURE_ASR_PREPARATION'])
        atomic_json(state_path, {'stage': 'waiting_for_resources', 'preparation_job': args.preparation_job,
                                 'run_dir': str(output), 'protected_jobs': sorted(PROTECTED_JOBS)})
        while not marker.exists() or not json.loads(marker.read_text()).get('features_ready'):
            if args.preparation_job:
                status = job_state(args.preparation_job)
                if status and status not in ACTIVE:
                    raise RuntimeError(f'Data preparation not ready; job {args.preparation_job}: {status}')
            time.sleep(15)
        checks = json.loads(args.checks_receipt.read_text())
        if checks['exit_code'] != 0:
            raise RuntimeError('Offline protocol checks have not passed')
        config = yaml.safe_load(args.config.read_text())
        if Path(config['sure']['execution_env']['SURE_ASR_PREPARATION']).resolve() != marker.resolve():
            raise RuntimeError('Data configuration changed during preparation; restart activation')
        handoff = verify_handoff(args.handoff_job) if args.handoff_job else None
        if handoff:
            config['sure']['slurm'].update(existing_allocations=[args.handoff_job],
                allocation_lock_dir='/shared/chaolei.liu/data/sure_asr_allocations',
                allocation_cpus=64, allocation_overlap=False, allocation_fallback_when_busy=True)
            config['sure']['execution_contract']['initial_allocation_limits'] = handoff
        configured = output/'launch_config.yaml'
        configured.write_text(yaml.safe_dump(config, sort_keys=False))
        source, deployment = freeze(configured, output)
        env = os.environ.copy()
        env['PYTHONPATH'] = str(source)
        log = output/'controller.log'
        with log.open('ab') as stream:
            child = subprocess.Popen([CONTROLLER, '-P', '-m', 'playground.sure_master.tools.tedlium_workflow',
                '--stage', 'search', '--config', str(deployment), '--output', str(output)], cwd=source,
                env=env, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        record = {'stage': 'controller_started', 'controller_pid': child.pid,
                  'controller_identity': process_identity(child.pid), 'source': str(source),
                  'deployment': str(deployment), 'run_dir': str(output), 'handoff': handoff}
        atomic_json(state_path, record)
        if handoff:
            deadline = time.monotonic() + 3600
            while not list(output.glob('search/workspace/exp_*/metric/slurm/*/allocation.json')):
                if child.poll() is not None:
                    raise RuntimeError(f'Controller exited before handoff; inspect {log}')
                if time.monotonic() > deadline:
                    raise RuntimeError('No new candidate step request; original service was preserved')
                time.sleep(10)
            current = verify_handoff(args.handoff_job)
            if current != handoff:
                raise RuntimeError('Service allocation changed before handoff; refusing to signal it')
            pid = handoff['batch_client_pid']
            name = remote(args.handoff_job, ['ps', '-p', str(pid), '-o', 'comm=']).strip()
            if name != 'srun':
                raise RuntimeError('Original batch client identity no longer matches')
            record['stage'] = 'handoff_requested'
            atomic_json(state_path, record)
            remote(args.handoff_job, ['kill', '-STOP', str(pid)])
            state = remote(args.handoff_job, ['ps', '-p', str(pid), '-o', 'stat=']).strip()
            if not state.startswith('T'):
                raise RuntimeError('Batch client did not stop; service step not cancelled')
            stop_service_step(handoff['service_step'])
            command(['scontrol', 'update', f'JobId={args.handoff_job}', 'JobName=asr-v2-allocation'])
            record['stage'] = 'allocation_handed_off'
            atomic_json(state_path, record)
        retire_old_workflows(output)
        record['stage'] = 'active'
        atomic_json(state_path, record)
        print(json.dumps(record), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT/'configs/sure_master/asr-formal-6x4-npu4-v2.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preparation-job', default='')
    parser.add_argument('--checks-receipt', type=Path, required=True)
    parser.add_argument('--handoff-job', default='')
    args = parser.parse_args()
    try:
        activate(args)
    except Exception as error:
        atomic_json(args.output/'activation_error.json', {'error': str(error), 'at': time.time()})
        raise


if __name__ == '__main__':
    main()
