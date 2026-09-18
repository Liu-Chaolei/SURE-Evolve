#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import uuid

import httpx
from pypdf import PdfReader

from knowledge_graph_lib.common import atomic_write_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--worker', required=True)
    args = parser.parse_args()
    root = args.run_dir
    owner = args.worker + '-' + uuid.uuid4().hex[:12]
    local = Path('/local/job') / owner
    local.mkdir(parents=True, exist_ok=True)
    for directory in ('texts', 'bundles', 'parses', 'parser-logs', 'parser-status'):
        (root / 'artifacts' / directory).mkdir(parents=True, exist_ok=True)
    client = httpx.Client(base_url='http://127.0.0.1:18092', timeout=120, trust_env=False,
                          headers={'Authorization': 'Bearer ' + os.environ['SPEECH_PIPELINE_KEY']})
    stop = threading.Event()
    versions = {name: importlib.metadata.version(name) for name in ('mineru', 'torch', 'torch-npu')}

    def stopping(signum: int, frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)

    def post(path: str, body: dict) -> object:
        response = client.post(path, json=body)
        response.raise_for_status()
        return response.json()

    def batch(papers: list[dict], isolated: bool = False) -> list[dict]:
        label = owner + '-' + hashlib.sha256('|'.join(p['sha256'] for p in papers).encode()).hexdigest()[:16]
        folder = local / label
        inputs = folder / 'input'
        output = folder / 'output'
        inputs.mkdir(parents=True, exist_ok=True)
        pages = {}
        errors = {}
        for paper in papers:
            sha = paper['sha256']
            try:
                destination = inputs / f'{sha}.pdf'
                shutil.copyfile(paper['source_pdf'], destination)
                if sha256_file(destination) != sha:
                    raise ValueError('PDF checksum mismatch')
                pages[sha] = len(PdfReader(destination).pages)
            except Exception as error:
                errors[sha] = str(error)
                (inputs / f'{sha}.pdf').unlink(missing_ok=True)
        code = 0
        logpath = root / 'artifacts/parser-logs' / f'{label}.log'
        if pages:
            with logpath.open('w') as log:
                process = subprocess.Popen([sys.executable, str(Path(__file__).with_name('mineru_npu.py')), '-p', str(inputs), '-o', str(output), '-b', 'pipeline', '-m', 'auto', '-l', 'en'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + (1800 if isolated else 7200)
                while process.poll() is None and time.monotonic() < deadline:
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
                    try:
                        post('/queue/heartbeat', {'stage': 'parse', 'owner': owner})
                    except (httpx.HTTPError, ValueError):
                        pass
                    atomic_write_json(root / 'artifacts/parser-status' / f'{args.worker}.json', {'phase': 'parsing', 'owner': owner, 'time': time.time(), 'papers': len(papers), 'batch': label})
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                code = process.returncode
        valid = []
        missing = []
        for paper in papers:
            sha = paper['sha256']
            try:
                if sha in errors:
                    raise ValueError(errors[sha])
                md = next(output.rglob(f'{sha}.md'))
                content = md.parent / f'{sha}_content_list.json'
                if not content.exists():
                    content = md.parent / f'{sha}_content_list_v2.json'
                middle = md.parent / f'{sha}_middle.json'
                if not md.read_text().strip() or not json.loads(content.read_text()) or len(json.loads(middle.read_text())['pdf_info']) != pages[sha]:
                    raise ValueError('Incomplete MinerU parse')
                valid.append((paper, md, content, middle))
            except Exception as error:
                missing.append({**paper, 'error': f'{type(error).__name__}: {error}; parser_exit={code}'})
        if valid:
            archive_local = folder / 'bundle.tar.gz'
            with tarfile.open(archive_local, 'w:gz', compresslevel=1) as archive:
                for paper, md, content, middle in valid:
                    archive.add(md.parent, arcname=paper['sha256'], filter=lambda item: None if item.name.lower().endswith('.pdf') else item)
            archive_relative = f'artifacts/bundles/{label}.tar.gz'
            archive_sha = sha256_file(archive_local)
            temporary = root / (archive_relative + '.partial')
            shutil.copyfile(archive_local, temporary)
            if sha256_file(temporary) != archive_sha:
                raise ValueError('Published archive checksum mismatch')
            temporary.replace(root / archive_relative)
            for paper, md, content, middle in valid:
                sha = paper['sha256']
                text_relative = f'artifacts/texts/{sha}.md'
                temporary = root / (text_relative + '.' + owner + '.partial')
                shutil.copyfile(md, temporary)
                temporary.replace(root / text_relative)
                record = {'sha256': sha, 'paper_id': paper['paper_id'], 'asset_root': str(root), 'status': 'parsed',
                          'source_pdf': paper['source_pdf'], 'pages': pages[sha], 'markdown_path': text_relative,
                          'markdown_sha256': sha256_file(md), 'archive': archive_relative, 'archive_sha256': archive_sha,
                          'content_list_member': f'{sha}/{content.name}', 'middle_json_member': f'{sha}/{middle.name}',
                          'versions': versions, 'parsed_at': time.time()}
                # The durable artifact is authoritative even if the coordinator
                # disappears between publication and acknowledgement.
                atomic_write_json(root / 'artifacts/parses' / f'{sha}.json', record)
                try:
                    post('/queue/parse-complete', {'owner': owner, 'record': record})
                except httpx.HTTPError:
                    pass
        if isolated:
            for paper in missing:
                post('/queue/fail', {'stage': 'parse', 'owner': owner, 'sha256': paper['sha256'], 'error': paper['error']})
        shutil.rmtree(folder)
        return missing

    while not stop.is_set():
        try:
            papers = post('/queue/claim', {'stage': 'parse', 'owner': owner, 'limit': 6})
            if not papers:
                state = client.get('/status').json()
                if (root / 'parse-complete.json').exists() or state.get('phase') in ('deep', 'survey', 'complete'):
                    break
                stop.wait(10)
                continue
            missing = batch(papers)
            for paper in missing:
                batch([paper], isolated=True)
        except (httpx.HTTPError, ValueError) as error:
            atomic_write_json(root / 'artifacts/parser-status' / f'{args.worker}.json', {'phase': 'waiting', 'time': time.time(), 'error': str(error)[:500]})
            stop.wait(15)
    atomic_write_json(root / 'artifacts/parser-status' / f'{args.worker}.json', {'phase': 'complete' if not stop.is_set() else 'stopped', 'time': time.time()})


if __name__ == '__main__':
    main()
