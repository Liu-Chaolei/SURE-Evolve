#!/usr/bin/env python3
"""Slurm-scoped Qwen service; credentials stay in the inherited environment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from urllib.request import Request, urlopen

from knowledge_graph_lib.common import atomic_write_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--replica', type=int, default=0)
    args = parser.parse_args()
    run = args.run_dir
    run.mkdir(parents=True, exist_ok=True)
    model = Path('/models/Qwen3.8-27B-W8A8')
    key = os.environ['SPEECH_PIPELINE_KEY']
    port = 18100 + args.replica
    endpoint = f'http://127.0.0.1:{port}/v1'
    backend = run / f'backend-{args.replica}.json'
    atomic_write_json(backend, {'id': args.replica, 'url': endpoint, 'ready': False, 'time': time.time()})
    identity_path = run / 'model-identity.json'
    if not identity_path.exists():
        index = json.loads((model / 'quant_model_weights.safetensors.index.json').read_text())
        hashes = {name: sha256_file(model / name) for name in sorted(set(index['weight_map'].values()))}
        identity = {'model': 'Qwen3.8-27B-W8A8', 'config_sha256': sha256_file(model / 'config.json'),
                    'weight_hashes': hashes, 'tokenizer_sha256': sha256_file(model / 'tokenizer.json')}
        identity['sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        atomic_write_json(identity_path, identity)
    profiles = [(32768, 16, 8192, True), (16384, 8, 4096, False)]
    profile_path = run / 'service-profile.json'
    if profile_path.exists():
        selected = json.loads(profile_path.read_text())
        profiles = [tuple(selected['profile'])]
    for context, sequences, batch, mtp in profiles:
        command = ['vllm', 'serve', str(model), '--host', '127.0.0.1', '--port', str(port),
                   '--served-model-name', 'Qwen3.8-27B-W8A8', '--tensor-parallel-size', '2',
                   '--quantization', 'ascend', '--trust-remote-code', '--max-model-len', str(context),
                   '--max-num-seqs', str(sequences), '--max-num-batched-tokens', str(batch),
                   '--gpu-memory-utilization', '0.85', '--generation-config', 'vllm',
                   '--enable-prefix-caching', '--enable-chunked-prefill',
                   '--compilation-config', '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
                   '--additional-config', '{"enable_cpu_binding":true}', '--api-key', key]
        if mtp:
            command += ['--speculative-config', '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}']
        environment = dict(os.environ)
        environment.pop('USE_TORCH', None)
        cache = Path('/local/job') / f'qwen-{args.replica}'
        for name in ('triton', 'inductor', 'extensions', 'ascend-cache', 'ascend-log', 'cache', 'pycache'):
            (cache / name).mkdir(parents=True, exist_ok=True)
        environment.update(TRITON_CACHE_DIR=str(cache / 'triton'), TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'),
                           TORCH_EXTENSIONS_DIR=str(cache / 'extensions'), ASCEND_CACHE_PATH=str(cache / 'ascend-cache'),
                           ASCEND_PROCESS_LOG_PATH=str(cache / 'ascend-log'), XDG_CACHE_HOME=str(cache / 'cache'),
                           VLLM_CACHE_ROOT=str(cache / 'cache/vllm'), PYTHONPYCACHEPREFIX=str(cache / 'pycache'), HF_HUB_OFFLINE='1')
        process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)

        def copy_log() -> None:
            with (run / f'model-{args.replica}-{context}.log').open('a') as handle:
                for line in process.stdout:
                    handle.write(line.replace(key, '[redacted]'))
                    handle.flush()

        thread = threading.Thread(target=copy_log, daemon=True)
        thread.start()
        ready = False
        deadline = time.monotonic() + 1800
        while process.poll() is None and time.monotonic() < deadline:
            try:
                with urlopen(Request(endpoint + '/models', headers={'Authorization': f'Bearer {key}'}), timeout=5) as response:
                    models = json.load(response)
                if any(item['id'] == 'Qwen3.8-27B-W8A8' for item in models['data']):
                    ready = True
                    break
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(10)
        atomic_write_json(backend, {'id': args.replica, 'url': endpoint, 'ready': ready, 'context': context,
                                    'max_sequences': sequences, 'mtp': mtp, 'pid': process.pid, 'time': time.time()})
        if ready:
            if args.replica == 0:
                atomic_write_json(profile_path, {'profile': [context, sequences, batch, mtp]})
            print(json.dumps({'backend': args.replica, 'ready': True, 'context': context}), flush=True)
            code = process.wait()
            atomic_write_json(backend, {'id': args.replica, 'url': endpoint, 'ready': False, 'exit_code': code})
            raise SystemExit(code or 1)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        thread.join(timeout=5)
    raise RuntimeError('Both Qwen service profiles failed; see redacted model logs')


if __name__ == '__main__':
    main()
