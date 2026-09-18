#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys
from pathlib import Path
import time

from transformers import AutoTokenizer

from knowledge_graph_lib.corpus_extract import extract_paper
from knowledge_graph_lib.corpus_flow import Flow
from knowledge_graph_lib.corpus_graph import build_corpus_graph, select_core
from knowledge_graph_lib.corpus_llm import CorpusLlm
from knowledge_graph_lib.corpus_store import CorpusStore
from knowledge_graph_lib.common import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['import-corpus', 'run', 'pilot', 'build', 'select-core'])
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--database', type=Path, default=Path('/local/job/corpus-queue.sqlite'))
    parser.add_argument('--tokenizer', type=Path, default=Path('/models/Qwen3.8-27B-W8A8'))
    parser.add_argument('--pilot-per-domain', type=int, default=40)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run bulk corpus phases inside a Slurm allocation')
    store = CorpusStore(args.corpus, args.run_dir, args.database)
    atomic_write_json(args.run_dir / 'status.json', {'phase': 'importing', 'started_at': time.time()})
    pilot_selection = args.run_dir / 'pilot-selection.json'
    pilot_hashes = set(json.loads(pilot_selection.read_text())['hashes']) if args.command == 'pilot' and pilot_selection.exists() else None
    print(json.dumps(store.ingest(parse_limit=0 if pilot_hashes is not None else 1500 if args.command == 'pilot' else None,
                                  parse_hashes=pilot_hashes)), flush=True)
    if args.command == 'import-corpus':
        return
    if args.command == 'build':
        print(json.dumps(build_corpus_graph(store)))
        return
    if args.command == 'select-core':
        print(json.dumps(select_core(args.run_dir)))
        return
    deadline = time.time() + 2400
    while time.time() < deadline:
        path = args.run_dir / 'backend-0.json'
        if path.exists() and json.loads(path.read_text()).get('ready'):
            break
        time.sleep(10)
    else:
        raise RuntimeError('Local model did not become ready')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    identity = json.loads((args.run_dir / 'model-identity.json').read_text())
    llm = CorpusLlm(args.run_dir, tokenizer, identity)
    flow = Flow(store, llm)
    flow.refresh_backends()
    flow.restore_completed()
    if args.command == 'pilot':
        flow.start_api()
        if not (args.run_dir / 'benchmark.json').exists():
            flow.benchmark()
        result = flow.pilot(concurrency=flow.limit, per_domain=args.pilot_per_domain)
        print(json.dumps(result), flush=True)
        if not result['passed']:
            raise SystemExit(2)
        candidates = []
        for path in (args.run_dir / 'artifacts/basic').glob('*.json'):
            candidates.append(json.loads(path.read_text()))
        deep = set()
        deep_counts = {}
        for domain in ('ASR', 'TTS', 'SD'):
            matches = sorted((row for row in candidates if domain in row['domains']), key=lambda row: row['sha256'])[:10]
            deep_counts[domain] = len(matches)
            deep.update(row['sha256'] for row in matches)
        result.update(passed=False, deep_counts=deep_counts, deep_failures=[])
        atomic_write_json(args.run_dir / 'pilot.json', result)
        if any(count < 10 for count in deep_counts.values()):
            raise RuntimeError('Pilot requires ten deep papers in each domain')
        store.enable_deep(sorted(deep))
        finished = 0
        with ThreadPoolExecutor(max_workers=flow.limit) as pool:
            jobs = {pool.submit(extract_paper, store, llm, store.paper(sha), 'deep'): sha for sha in sorted(deep)}
            for future in as_completed(jobs):
                sha = jobs[future]
                try:
                    future.result()
                    store.finish(sha, 'deep')
                except Exception as error:
                    result['deep_failures'].append({'sha256': sha, 'error': f'{type(error).__name__}: {error}'})
                finished += 1
                atomic_write_json(args.run_dir / 'status.json', {'phase': 'pilot-deep', 'total': len(deep), 'finished': finished, 'failures': len(result['deep_failures'])})
                atomic_write_json(args.run_dir / 'pilot.json', result)
        if result['deep_failures']:
            raise SystemExit(2)
        build_corpus_graph(store)
        script = Path(__file__).parents[2] / 'literature_survey/scripts/run_survey_phase.py'
        code = subprocess.call([sys.executable, str(script), 'corpus-synthesize', '--pipeline-dir', str(args.run_dir), '--pilot'])
        result.update(deep_papers=len(deep), survey_exit=code, passed=code == 0)
        atomic_write_json(args.run_dir / 'pilot.json', result)
        flow.server.shutdown()
        raise SystemExit(0 if result['passed'] else 2)
    flow.run_forever()


if __name__ == '__main__':
    main()
