from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid

from .common import atomic_write_json
from .corpus_extract import extract_paper, signature_for, validate_facts, FACETS
from .corpus_graph import build_corpus_graph, records, select_core
from .corpus_llm import CorpusLlm, ServiceUnavailable
from .corpus_store import CorpusStore, contained, fingerprint


class Admission:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.active = 0
        self.condition = threading.Condition()

    def resize(self, limit: int) -> None:
        with self.condition:
            self.limit = limit
            self.condition.notify_all()

    def __enter__(self) -> None:
        with self.condition:
            self.condition.wait_for(lambda: self.active < self.limit)
            self.active += 1

    def __exit__(self, *args: object) -> None:
        with self.condition:
            self.active -= 1
            self.condition.notify()


class Flow:
    def __init__(self, store: CorpusStore, llm: CorpusLlm) -> None:
        self.store = store
        self.llm = llm
        self.run = store.run
        self.stop = threading.Event()
        self.phase = 'basic'
        self.limit = 8
        benchmark = self.run / 'benchmark.json'
        if benchmark.exists():
            self.limit = json.loads(benchmark.read_text())['selected_concurrency']
        self.semaphore = Admission(self.limit)
        self.server = None
        self.backend_lock = threading.Lock()
        self.backend_checked = 0.0
        self.backend_cache: list[dict] = []
        atomic_write_json(self.run / 'taxonomy.json', FACETS)
        self.supplements: dict[str, set[str]] = {domain: set() for domain in ('ASR', 'TTS', 'SD')}
        self.supplement_path = self.run / 'supplements.json'
        if self.supplement_path.exists():
            self.supplements = {key: set(value) for key, value in json.loads(self.supplement_path.read_text()).items()}

    def refresh_backends(self) -> list[dict]:
        with self.backend_lock:
            if time.monotonic() - self.backend_checked < 5:
                return self.backend_cache
            return self._refresh_backends()

    def _refresh_backends(self) -> list[dict]:
        backends = [json.loads(path.read_text()) for path in sorted(self.run.glob('backend-*.json'))]
        primary = next((row for row in backends if row['id'] == 0 and row.get('ready')), None)
        if primary:
            self.llm.context = primary['context']
        healthy = []
        for row in backends:
            if not row.get('ready') or row.get('context') != self.llm.context:
                continue
            try:
                response = self.llm.client.get(row['url'].rstrip('/') + '/models', headers={'Authorization': 'Bearer ' + os.environ['SPEECH_PIPELINE_KEY']}, timeout=3)
                if response.status_code == 200:
                    healthy.append(row)
            except Exception:
                continue
        self.semaphore.resize(max(1, self.limit * len(healthy)))
        atomic_write_json(self.run / 'service.json', {'model': self.llm.identity, 'context': self.llm.context,
                                                    'front_url': 'http://127.0.0.1:18092', 'backends': healthy})
        self.backend_cache = healthy
        self.backend_checked = time.monotonic()
        return healthy

    def start_api(self) -> None:
        if self.server is None:
            self.server = ThreadingHTTPServer(('127.0.0.1', 18092), self.handler())
            threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def restore_completed(self) -> None:
        selection = self.run / 'core-selection.json'
        if selection.exists():
            self.store.enable_deep(json.loads(selection.read_text())['unique_hashes'])
        self.store.enable_deep(sorted(set(sha for values in self.supplements.values() for sha in values)))
        for level in ('basic', 'deep'):
            for path in (self.run / 'artifacts' / level).glob('*.json'):
                value = json.loads(path.read_text())
                paper = self.store.paper(value['sha256'])
                if not paper['parse_ref']:
                    continue
                if paper['parse_ref'] and value.get('signature') == signature_for(self.llm, paper, level):
                    parse = paper['parse_ref']
                    source = contained(Path(parse['asset_root']), parse['markdown_path'])
                    if self.store.digest(source) == parse['markdown_sha256']:
                        try:
                            validated = validate_facts(value, source.read_text())
                        except ValueError:
                            pass
                        else:
                            if value['facts'] != validated['facts'] or value.get('no_named_core') != validated['no_named_core']:
                                value.update(validated, validation_revision='speech-grounding-v3')
                                atomic_write_json(path, value)
                            self.store.finish(value['sha256'], level)
                            continue
                stale = self.run / 'artifacts/stale' / level / f"{value['sha256']}-{value.get('signature', 'unknown')}.json"
                stale.parent.mkdir(parents=True, exist_ok=True)
                path.replace(stale)
                self.store.finish(value['sha256'], level, error='Cached extraction requires revalidation', retry=True)

    def handler(self) -> type[BaseHTTPRequestHandler]:
        flow = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                pass

            def send_json(self, status: int, payload: object) -> None:
                data = json.dumps(payload, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def authorized(self) -> bool:
                expected = 'Bearer ' + os.environ['SPEECH_PIPELINE_KEY']
                return hmac.compare_digest(self.headers.get('Authorization', ''), expected)

            def do_GET(self) -> None:
                if not self.authorized():
                    self.send_json(401, {'error': 'Unauthorized'})
                    return
                self.send_json(200, {'phase': flow.phase, 'context': flow.llm.context, 'model': flow.llm.identity,
                                     'ready': bool(flow.refresh_backends()), **flow.store.summary()})

            def do_POST(self) -> None:
                if not self.authorized():
                    self.send_json(401, {'error': 'Unauthorized'})
                    return
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= 2 * 1024 * 1024:
                        raise ValueError('Invalid request size')
                    body = json.loads(self.rfile.read(size))
                    if self.path == '/queue/claim':
                        value = flow.store.claim(body['stage'], body['owner'], int(body.get('limit', 1)))
                    elif self.path == '/queue/heartbeat':
                        flow.store.heartbeat(body['stage'], body['owner'])
                        value = {'ok': True}
                    elif self.path == '/queue/parse-complete':
                        record = body['record']
                        with flow.store.lock:
                            task = flow.store.db.execute("SELECT state,owner FROM tasks WHERE sha=? AND stage='parse'", (record['sha256'],)).fetchone()
                            if not task or (task['state'] != 'done' and task['owner'] != body['owner']):
                                raise ValueError('Parse lease is not owned by this worker')
                            flow.store.register_parse(record)
                        value = {'ok': True}
                    elif self.path == '/queue/fail':
                        with flow.store.lock:
                            task = flow.store.db.execute('SELECT owner FROM tasks WHERE sha=? AND stage=?', (body['sha256'], body['stage'])).fetchone()
                            if not task or task['owner'] != body['owner']:
                                raise ValueError('Task lease is not owned by this worker')
                            flow.store.finish(body['sha256'], body['stage'], error=str(body['error'])[:2000], retry=bool(body.get('retry')))
                        value = {'ok': True}
                    elif self.path == '/queue/promote':
                        domain = body['domain']
                        if domain not in flow.supplements:
                            raise ValueError('Unknown survey domain')
                        hashes = set(body['hashes'])
                        if len(flow.supplements[domain] | hashes) > 100:
                            raise ValueError('Supplementary deep extraction budget exhausted')
                        for sha in hashes:
                            if not (flow.run / 'artifacts/basic' / f'{sha}.json').is_file():
                                raise ValueError('A supplement requires validated basic evidence')
                        flow.supplements[domain].update(hashes)
                        flow.store.enable_deep(sorted(hashes))
                        atomic_write_json(flow.supplement_path, {key: sorted(value) for key, value in flow.supplements.items()})
                        value = {'queued': len(hashes)}
                    elif self.path == '/research/count':
                        value = {'tokens': flow.llm.count(body['messages']), 'context': flow.llm.context}
                    elif self.path == '/v1/chat/completions':
                        messages = body['messages']
                        if len(messages) != 2 or messages[0]['role'] != 'system' or messages[1]['role'] != 'user':
                            raise ValueError('Research endpoint expects one system and one user message')
                        with flow.semaphore:
                            output, metadata = flow.llm.complete(messages[0]['content'], messages[1]['content'], maximum=int(body.get('max_tokens', 4096)), tag='survey', json_output=True)
                        value = {'id': metadata['request_sha256'], 'model': 'Qwen3.8-27B-W8A8', 'usage': metadata['usage'],
                                 'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': json.dumps(output, ensure_ascii=False)}}]}
                    else:
                        self.send_json(404, {'error': 'Unknown endpoint'})
                        return
                    self.send_json(200, value)
                except ServiceUnavailable as error:
                    self.send_json(503, {'error': str(error)})
                except (ValueError, KeyError, OSError) as error:
                    self.send_json(400, {'error': str(error)[:1000]})

        return Handler

    def worker(self, index: int) -> None:
        owner = f'extract-{os.getpid()}-{index}'
        while not self.stop.is_set():
            stage = 'basic' if self.phase in ('basic', 'pilot') else 'deep'
            if not self.refresh_backends():
                self.stop.wait(10)
                continue
            group = self.store.claim(stage, owner)
            if not group:
                self.stop.wait(5)
                continue
            paper = group[0]
            ended = threading.Event()

            def heartbeat() -> None:
                while not ended.wait(30):
                    self.store.heartbeat(stage, owner)

            heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
            heartbeat_thread.start()
            try:
                with self.semaphore:
                    extract_paper(self.store, self.llm, paper, stage)
                self.store.finish(paper['sha256'], stage)
            except ServiceUnavailable as error:
                self.store.finish(paper['sha256'], stage, error=str(error), retry=True)
                self.stop.wait(30)
            except Exception as error:
                self.store.finish(paper['sha256'], stage, error=f'{type(error).__name__}: {error}')
            finally:
                ended.set()
                heartbeat_thread.join(timeout=2)

    def pilot(self, concurrency: int = 8, per_domain: int = 40) -> dict:
        self.phase = 'pilot'
        selected = {}
        papers = [paper for paper in self.store.all_papers() if paper['parse_ref']]
        selection_path = self.run / 'pilot-selection.json'
        if selection_path.exists():
            previous = json.loads(selection_path.read_text())
            available = {paper['sha256']: paper for paper in papers}
            if previous['per_domain'] != per_domain or len(previous['hashes']) != 3 * per_domain:
                raise ValueError('Existing pilot selection has different sampling parameters')
            if any(sha not in available for sha in previous['hashes']):
                raise ValueError('An existing pilot paper is no longer parsed')
            selected = {sha: available[sha] for sha in previous['hashes']}
        else:
            for domain in ('ASR', 'TTS', 'SD'):
                candidates = sorted((paper for paper in papers if domain in paper['metadata'].get('tags', []) and paper['sha256'] not in selected), key=lambda p: (p['parse_ref']['pages'], p['sha256']))
                if len(candidates) < per_domain:
                    raise ValueError(f'Not enough parsed {domain} pilot papers')
                for i in range(per_domain):
                    paper = candidates[min(len(candidates) - 1, int((i + .5) * len(candidates) / per_domain))]
                    selected[paper['sha256']] = paper
            atomic_write_json(selection_path, {'hashes': list(selected), 'per_domain': per_domain})
        started = time.time()
        failures = []
        finished = 0
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            jobs = {pool.submit(extract_paper, self.store, self.llm, paper, 'basic'): sha for sha, paper in selected.items()}
            for future in as_completed(jobs):
                sha = jobs[future]
                try:
                    future.result()
                    self.store.finish(sha, 'basic')
                except Exception as error:
                    failures.append({'sha256': sha, 'error': f'{type(error).__name__}: {error}'})
                finished += 1
                atomic_write_json(self.run / 'status.json', {'phase': 'pilot-basic', 'total': len(selected), 'finished': finished,
                                                            'failures': len(failures), 'elapsed_seconds': time.time() - started})
        report = {'pilot_papers': len(selected), 'successes': len(selected) - len(failures), 'failures': failures,
                  'elapsed_seconds': time.time() - started, 'concurrency': concurrency,
                  'passed': len(failures) <= .05 * len(selected)}
        atomic_write_json(self.run / 'pilot.json', report)
        return report

    def benchmark(self) -> dict:
        papers = [paper for paper in self.store.all_papers() if paper['parse_ref']]
        chosen = {}
        for domain in ('ASR', 'TTS', 'SD'):
            candidates = sorted((p for p in papers if domain in p['metadata'].get('tags', []) and p['sha256'] not in chosen), key=lambda p: (p['parse_ref']['pages'], p['sha256']))
            if len(candidates) < 8:
                raise ValueError('Not enough benchmark papers')
            for index in range(8):
                paper = candidates[int((index + .5) * len(candidates) / 8)]
                chosen[paper['sha256']] = paper
        results = []
        for concurrency in (4, 8, 16):
            case_store = copy.copy(self.store)
            case_store.run = self.run / 'benchmarks' / (f'c{concurrency}-' + uuid.uuid4().hex[:8])
            case_llm = CorpusLlm(self.run, self.llm.tokenizer, self.llm.identity, context=self.llm.context)
            case_llm.benchmark_nonce = uuid.uuid4().hex
            started = time.time()
            atomic_write_json(self.run / 'status.json', {'phase': 'pilot-benchmark', 'concurrency': concurrency,
                                                        'total': len(chosen), 'started_at': started})
            errors = []
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                jobs = {pool.submit(extract_paper, case_store, case_llm, paper, 'basic'): sha for sha, paper in chosen.items()}
                for future in as_completed(jobs):
                    try:
                        future.result()
                    except Exception as error:
                        errors.append({'sha256': jobs[future], 'error': f'{type(error).__name__}: {error}'})
            elapsed = time.time() - started
            results.append({'concurrency': concurrency, 'papers': len(chosen), 'errors': errors, 'seconds': elapsed,
                            'papers_per_hour': (len(chosen) - len(errors)) * 3600 / elapsed, 'passed': len(errors) <= .05 * len(chosen)})
            atomic_write_json(self.run / 'benchmark-progress.json', results)
        acceptable = [row for row in results if row['passed']]
        if not acceptable:
            raise ValueError('No concurrency profile passed evidence validation')
        best = max(acceptable, key=lambda row: row['papers_per_hour'])
        report = {'selected_concurrency': best['concurrency'], 'cases': results, 'paper_hashes': list(chosen), 'context': self.llm.context}
        atomic_write_json(self.run / 'benchmark.json', report)
        self.limit = best['concurrency']
        self.refresh_backends()
        return report

    def run_forever(self) -> None:
        self.store.ingest()
        self.start_api()
        threads = [threading.Thread(target=self.worker, args=(index,), daemon=True) for index in range(64)]
        for thread in threads:
            thread.start()
        last_graph = 0.0
        survey: subprocess.Popen | None = None
        try:
            while not self.stop.is_set():
                self.store.ingest()
                backends = self.refresh_backends()
                summary = self.store.summary()
                try:
                    downloads = json.loads((self.store.corpus / 'status.json').read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    downloads = {'status': 'running'}
                parsed = summary['stages']['parse']
                basic = summary['stages']['basic']
                deep = summary['stages']['deep']
                parse_terminal = (self.store.catalog_is_current() and not parsed.get('pending', 0)
                                  and not parsed.get('running', 0) and downloads.get('status') != 'running')
                if parse_terminal:
                    atomic_write_json(self.run / 'parse-complete.json', {'time': time.time(), 'summary': parsed})
                else:
                    (self.run / 'parse-complete.json').unlink(missing_ok=True)
                if self.phase == 'basic' and parse_terminal and not basic.get('pending', 0) and not basic.get('running', 0):
                    if parsed.get('done', 0) < .95 * max(1, sum(parsed.values())) or basic.get('done', 0) < .95 * max(1, sum(basic.values())):
                        self.phase = 'incomplete'
                        break
                    selection_path = self.run / 'core-selection.json'
                    selection = json.loads(selection_path.read_text()) if selection_path.exists() else select_core(self.run)
                    if any(len(selection['domains'].get(domain, [])) < 500 for domain in ('ASR', 'TTS', 'SD')):
                        atomic_write_json(self.run / 'coverage-blocker.json', {'reason': 'Fewer than 500 validated core candidates in a domain', 'counts': {domain: len(values) for domain, values in selection['domains'].items()}})
                        self.phase = 'incomplete'
                        break
                    self.store.enable_deep(selection['unique_hashes'])
                    self.phase = 'deep'
                if time.time() - last_graph >= 1800 and basic.get('done', 0) >= 1000:
                    build_corpus_graph(self.store)
                    last_graph = time.time()
                if self.phase == 'deep' and deep and not deep.get('pending', 0) and not deep.get('running', 0):
                    if deep.get('done', 0) < .95 * sum(deep.values()):
                        self.phase = 'incomplete'
                        break
                    build_corpus_graph(self.store)
                    atomic_write_json(self.run / 'extraction-complete.json', {'time': time.time(), 'summary': summary})
                    self.phase = 'survey'
                    script = Path(__file__).parents[3] / 'literature_survey/scripts/run_survey_phase.py'
                    log = (self.run / 'survey.log').open('a')
                    survey = subprocess.Popen([os.environ.get('SPEECH_CPU_PYTHON', 'python3'), str(script), 'corpus-synthesize', '--pipeline-dir', str(self.run)], stdout=log, stderr=subprocess.STDOUT)
                    log.close()
                if survey and survey.poll() is not None:
                    self.phase = 'complete' if survey.returncode == 0 else 'incomplete'
                    build_corpus_graph(self.store)
                    break
                atomic_write_json(self.run / 'status.json', {'phase': self.phase, 'time': time.time(), 'backends': len(backends), **summary})
                self.stop.wait(60)
        finally:
            self.stop.set()
            self.server.shutdown()
            atomic_write_json(self.run / 'status.json', {'phase': self.phase, 'time': time.time(), **self.store.summary()})
