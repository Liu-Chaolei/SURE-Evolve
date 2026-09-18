"""Durable bulk-corpus records with a rebuildable, node-local work queue."""
from __future__ import annotations

import csv
from collections import OrderedDict
import errno
import hashlib
import io
import json
import os
import re
from pathlib import Path, PurePosixPath
import sqlite3
import tarfile
import threading
import time

from .common import atomic_write_json, sha256_file, compact_key
from .references import REFERENCE_ENTRY, YEAR, _guess_title, _references_section


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def contained(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe artifact path: {value}')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('Artifact escapes its declared root')
    return resolved


def archive_json(path: Path, member: str) -> object:
    name = PurePosixPath(member)
    if name.is_absolute() or '..' in name.parts:
        raise ValueError('Unsafe archive member')
    with tarfile.open(path, 'r:gz') as archive:
        info = archive.getmember(member)
        if not info.isfile() or info.size > 512 * 1024 * 1024:
            raise ValueError('Expected a bounded regular JSON member')
        handle = archive.extractfile(info)
        if handle is None:
            raise ValueError('Missing archive member')
        return json.load(handle)


class CorpusStore:
    def __init__(self, corpus: Path, run: Path, database: Path) -> None:
        self.corpus = corpus.resolve()
        self.run = run.resolve()
        self.lock = threading.RLock()
        self.verified: dict[tuple[str, int, int], str] = {}
        self.offsets: dict[str, int] = {}
        self.native_seen: set[str] = set()
        self.bundles: OrderedDict[str, dict[str, object]] = OrderedDict()
        self.metadata: dict[str, dict] = {}
        self.reference_titles: dict[str, list[str]] = {}
        self.reference_ids: dict[str, str] = {}
        self.catalog_cache: str | None = None
        self.catalog_stamp: tuple[int, int, int] | None = None
        self.catalog_current = False
        for name in ('parses', 'basic', 'deep', 'passes', 'failures', 'graphs', 'surveys'):
            (run / 'artifacts' / name).mkdir(parents=True, exist_ok=True)
        database.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(database, check_same_thread=False, isolation_level='DEFERRED')
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS papers (
                sha TEXT PRIMARY KEY, paper_id TEXT NOT NULL, metadata TEXT NOT NULL,
                pdf TEXT, parse_ref TEXT, active INTEGER NOT NULL DEFAULT 1);
            CREATE TABLE IF NOT EXISTS aliases (id TEXT PRIMARY KEY, sha TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS tasks (
                sha TEXT NOT NULL REFERENCES papers(sha), stage TEXT NOT NULL,
                state TEXT NOT NULL, owner TEXT, expires REAL, attempts INTEGER DEFAULT 0,
                error TEXT, PRIMARY KEY(sha,stage));
            CREATE INDEX IF NOT EXISTS task_state ON tasks(stage,state);
            CREATE INDEX IF NOT EXISTS parsed_presence ON papers(sha) WHERE parse_ref IS NOT NULL;
        ''')
        # A new coordinator reconstructs completions from artifacts; leases never
        # survive a coordinator restart as if their old workers were still alive.
        self.db.execute("UPDATE tasks SET state='pending',owner=NULL WHERE state='running'")
        self.db.commit()

    def digest(self, path: Path) -> str:
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self.verified:
            self.verified[key] = sha256_file(path)
        return self.verified[key]

    def read_catalog(self) -> str:
        path = self.corpus / 'catalog.csv'
        for attempt in range(5):
            try:
                before = path.stat()
                raw = path.read_bytes()
                after = path.stat()
                stamp = (after.st_ino, after.st_size, after.st_mtime_ns)
                if (before.st_ino, before.st_size, before.st_mtime_ns) != stamp or not raw.endswith(b'\n'):
                    time.sleep(.1 * (attempt + 1))
                    continue
                self.catalog_cache = raw.decode('utf-8')
                self.catalog_stamp = stamp
                self.catalog_current = True
                return self.catalog_cache
            except OSError as error:
                if error.errno not in (errno.ENOENT, errno.ESTALE):
                    raise
                time.sleep(.1 * (attempt + 1))
        self.catalog_current = False
        if self.catalog_cache is not None:
            return self.catalog_cache
        raise RuntimeError('No stable corpus catalog snapshot is available')

    def catalog_is_current(self) -> bool:
        try:
            stat = (self.corpus / 'catalog.csv').stat()
            return self.catalog_current and self.catalog_stamp == (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        except OSError:
            return False

    def ingest(self, parse_limit: int | None = None, parse_hashes: set[str] | None = None) -> dict:
        with self.lock:
            if not self.metadata:
                for parent in (self.corpus, self.corpus / 'arxiv'):
                    path = parent / 'artifacts/metadata/papers.jsonl'
                    if path.exists():
                        with path.open() as handle:
                            for line in handle:
                                paper = json.loads(line)
                                self.metadata[paper['id']] = {key: paper.get(key) for key in ('title', 'authors', 'year', 'venue', 'abstract', 'external_ids', 'source_records')}
                for identifier, metadata in self.metadata.items():
                    self.reference_titles.setdefault(compact_key(metadata.get('title')), []).append(identifier)
                    for key, value in (metadata.get('external_ids') or {}).items():
                        self.reference_ids[f'{key.lower()}:{str(value).lower()}'] = identifier
            # DPC may invalidate an open inode when the downloader atomically
            # replaces it. Parse a complete in-memory snapshot, never a live FD.
            with io.StringIO(self.read_catalog()) as handle:
                for row in csv.DictReader(handle):
                    downloaded = row['status'] in ('downloaded', 'existing')
                    sha = row['sha256'] if downloaded else 'metadata-' + fingerprint(row['id'])
                    if downloaded and (len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha)):
                        raise ValueError('Invalid downloaded PDF checksum')
                    pdf = str(contained(self.corpus, row['pdf_path'])) if downloaded else None
                    metadata = {**self.metadata.get(row['id'], {}), 'id': row['id'], 'title': row['title'],
                                'year': row.get('year'), 'venue': row.get('venue'), 'tags': row['tags'].split(';')}
                    existing = self.db.execute('SELECT metadata FROM papers WHERE sha=?', (sha,)).fetchone()
                    if existing:
                        original = json.loads(existing['metadata'])
                        original['tags'] = sorted(set(original.get('tags', [])) | set(metadata['tags']))
                        self.db.execute('UPDATE papers SET metadata=? WHERE sha=?', (json.dumps(original, ensure_ascii=False), sha))
                    previous = self.db.execute('SELECT sha FROM aliases WHERE id=?', (row['id'],)).fetchone()
                    if previous and previous['sha'] != sha and previous['sha'].startswith('metadata-'):
                        self.db.execute('UPDATE papers SET active=0 WHERE sha=?', (previous['sha'],))
                    self.db.execute('INSERT INTO papers(sha,paper_id,metadata,pdf) VALUES(?,?,?,?) ON CONFLICT(sha) DO UPDATE SET active=1',
                                    (sha, row['id'], json.dumps(metadata, ensure_ascii=False), pdf))
                    self.db.execute('INSERT OR REPLACE INTO aliases VALUES(?,?)', (row['id'], sha))
                    if downloaded:
                        self.db.execute("INSERT OR IGNORE INTO tasks(sha,stage,state) VALUES(?,'parse','pending')", (sha,))
            # Prefer durable canonical manifests. Their unchanged archive/PDF/
            # Markdown hashes allow reuse of the previously checked structure.
            for path in (self.run / 'artifacts/parses').glob('*.json'):
                if parse_hashes is not None and path.stem not in parse_hashes:
                    continue
                if str(path) in self.native_seen:
                    continue
                record = json.loads(path.read_text())
                current = self.db.execute('SELECT parse_ref FROM papers WHERE sha=?', (record['sha256'],)).fetchone()
                if current and not current['parse_ref']:
                    self.register_parse(record)
                if current:
                    self.native_seen.add(str(path))
            legacy = self.corpus / 'parsing_20260912'
            for path in sorted((legacy / 'checkpoints').glob('worker-*.jsonl')):
                if parse_limit is not None and self.db.execute('SELECT count(*) FROM papers WHERE parse_ref IS NOT NULL').fetchone()[0] >= parse_limit:
                    break
                offset = self.offsets.get(str(path), 0)
                with path.open() as handle:
                    if path.stat().st_size < offset:
                        offset = 0
                    handle.seek(offset)
                    while True:
                        start = handle.tell()
                        line = handle.readline()
                        if not line:
                            break
                        if not line.endswith('\n'):
                            handle.seek(start)
                            break
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if record.get('status') == 'parsed':
                            if not self.db.execute('SELECT 1 FROM papers WHERE sha=?', (record['sha256'],)).fetchone():
                                handle.seek(start)
                                break
                            self.register_parse({**record, 'asset_root': str(legacy)})
                            if parse_limit is not None and self.db.execute('SELECT count(*) FROM papers WHERE parse_ref IS NOT NULL').fetchone()[0] >= parse_limit:
                                break
                    self.offsets[str(path)] = handle.tell()
            snapshot = {'schema_version': 'xlab.corpus.v1', 'source_kind': 'bulk_archive',
                        'corpus_root': str(self.corpus), 'updated_at': time.time(), **self.summary()}
            atomic_write_json(self.run / 'corpus.json', snapshot)
            self.db.commit()
            return snapshot

    def register_parse(self, record: dict, *, verify_pdf: bool = True) -> None:
        with self.lock:
            sha = record['sha256']
            paper = self.db.execute('SELECT * FROM papers WHERE sha=?', (sha,)).fetchone()
            if paper is None:
                return
            if paper['parse_ref']:
                return
            asset_root = Path(record['asset_root']).resolve()
            if not asset_root.is_relative_to(self.corpus) and not asset_root.is_relative_to(self.run):
                raise ValueError('Unexpected parse asset root')
            md = contained(asset_root, record['markdown_path'])
            archive = contained(asset_root, record['archive'])
            if self.digest(md) != record['markdown_sha256'] or self.digest(archive) != record['archive_sha256']:
                raise ValueError('Parse artifact checksum mismatch')
            if verify_pdf and self.digest(Path(paper['pdf'])) != sha:
                raise ValueError('PDF checksum mismatch')
            cache_key = record['archive_sha256']
            for member in (record['middle_json_member'], record['content_list_member']):
                item_path = PurePosixPath(member)
                if item_path.is_absolute() or '..' in item_path.parts:
                    raise ValueError('Unsafe bundle member')
            structure_signature = fingerprint({key: record[key] for key in ('archive_sha256', 'middle_json_member', 'content_list_member', 'pages')})
            previous_path = self.run / 'artifacts/parses' / f'{sha}.json'
            previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
            checked_fields = ('sha256', 'asset_root', 'archive', 'archive_sha256', 'markdown_path', 'markdown_sha256',
                              'middle_json_member', 'content_list_member', 'pages')
            reusable = (previous.get('schema_version') == 'xlab.corpus_parse.v1'
                        and all(previous.get(key) == record.get(key) for key in checked_fields)
                        and previous.get('structure_signature', structure_signature) == structure_signature)
            if reusable:
                if not isinstance(record['pages'], int) or record['pages'] < 1 or not md.read_text().strip():
                    raise ValueError('Incomplete canonical parse checkpoint')
            elif cache_key not in self.bundles:
                contents = {}
                with tarfile.open(archive, 'r|gz') as handle:
                    for info in handle:
                        if info.name.endswith(('_content_list.json', '_content_list_v2.json', '_middle.json', '/content.json', '/middle.json')):
                            if not info.isfile() or info.size > 512 * 1024 * 1024:
                                raise ValueError('Unsafe JSON bundle member')
                            stream = handle.extractfile(info)
                            contents[info.name] = json.load(stream)
                self.bundles[cache_key] = contents
                while len(self.bundles) > 8:
                    self.bundles.popitem(last=False)
            if not reusable:
                self.bundles.move_to_end(cache_key)
                middle = self.bundles[cache_key][record['middle_json_member']]
                content = self.bundles[cache_key][record['content_list_member']]
                if not isinstance(middle, dict) or len(middle.get('pdf_info', [])) != record['pages'] or not content or not md.read_text().strip():
                    raise ValueError('Incomplete MinerU bundle')
            canonical = {**record, 'schema_version': 'xlab.corpus_parse.v1', 'paper_id': paper['paper_id'], 'structure_signature': structure_signature}
            atomic_write_json(self.run / 'artifacts/parses' / f'{sha}.json', canonical)
            self.db.execute('UPDATE papers SET parse_ref=? WHERE sha=?', (json.dumps(canonical), sha))
            self.db.execute("UPDATE tasks SET state='done',owner=NULL WHERE sha=? AND stage='parse'", (sha,))
            self.db.execute("INSERT OR IGNORE INTO tasks(sha,stage,state) VALUES(?,'basic','pending')", (sha,))
            self.db.commit()

    def claim(self, stage: str, owner: str, limit: int = 1) -> list[dict]:
        if stage not in ('parse', 'basic', 'deep') or not 1 <= limit <= 6 or not owner:
            raise ValueError('Invalid claim')
        with self.lock:
            now = time.time()
            self.db.execute("UPDATE tasks SET state='pending',owner=NULL WHERE state='running' AND expires<?", (now,))
            rows = self.db.execute("SELECT p.* FROM papers p JOIN tasks t ON p.sha=t.sha WHERE p.active=1 AND t.stage=? AND t.state='pending' ORDER BY p.sha LIMIT ?", (stage, limit)).fetchall()
            result = []
            for row in rows:
                self.db.execute("UPDATE tasks SET state='running',owner=?,expires=?,attempts=attempts+1 WHERE sha=? AND stage=?", (owner, now + 600, row['sha'], stage))
                result.append({'sha256': row['sha'], 'paper_id': row['paper_id'], 'metadata': json.loads(row['metadata']),
                               'source_pdf': row['pdf'], 'parse_ref': json.loads(row['parse_ref']) if row['parse_ref'] else None})
            self.db.commit()
            return result

    def heartbeat(self, stage: str, owner: str) -> None:
        with self.lock:
            self.db.execute("UPDATE tasks SET expires=? WHERE stage=? AND owner=? AND state='running'", (time.time() + 600, stage, owner))
            self.db.commit()

    def finish(self, sha: str, stage: str, *, error: str | None = None, retry: bool = False) -> None:
        with self.lock:
            state = 'pending' if retry else ('failed' if error else 'done')
            if error and not retry:
                atomic_write_json(self.run / 'artifacts/failures' / f'{stage}-{sha}.json', {'sha256': sha, 'stage': stage, 'error': error, 'time': time.time()})
            self.db.execute('UPDATE tasks SET state=?,error=?,owner=NULL WHERE sha=? AND stage=?', (state, error, sha, stage))
            self.db.commit()

    def enable_deep(self, hashes: list[str]) -> None:
        with self.lock:
            self.db.executemany("INSERT OR IGNORE INTO tasks(sha,stage,state) VALUES(?,'deep','pending')", [(sha,) for sha in hashes])
            self.db.commit()

    def paper(self, sha: str) -> dict:
        with self.lock:
            row = self.db.execute('SELECT * FROM papers WHERE sha=?', (sha,)).fetchone()
            if row is None:
                raise KeyError(sha)
            return {'sha256': sha, 'paper_id': row['paper_id'], 'metadata': json.loads(row['metadata']),
                    'source_pdf': row['pdf'], 'parse_ref': json.loads(row['parse_ref']) if row['parse_ref'] else None}

    def summary(self) -> dict:
        with self.lock:
            stages: dict[str, dict[str, int]] = {stage: {} for stage in ('parse', 'basic', 'deep')}
            for row in self.db.execute('SELECT stage,state,count(*) AS n FROM tasks GROUP BY stage,state'):
                stages[row['stage']][row['state']] = row['n']
            total = self.db.execute('SELECT count(*) FROM papers WHERE active=1').fetchone()[0]
            available = self.db.execute('SELECT count(*) FROM papers WHERE active=1 AND pdf IS NOT NULL').fetchone()[0]
            return {'papers': total, 'downloaded': available, 'stages': stages}

    def all_papers(self) -> list[dict]:
        with self.lock:
            return [self.paper(row[0]) for row in self.db.execute('SELECT sha FROM papers WHERE active=1')]

    def references(self, source: str) -> list[dict]:
        results = []
        for match in REFERENCE_ENTRY.finditer(_references_section(source)):
            raw = ' '.join(match.group(3).split())
            title = _guess_title(raw)
            year_match = YEAR.search(raw)
            year = year_match.group(1) if year_match else None
            target = None
            doi = re.search(r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+', raw)
            arxiv = re.search(r'(?:arxiv[: /]+)(\d{4}\.\d{4,5})', raw, re.I)
            if doi:
                target = self.reference_ids.get('doi:' + doi.group().rstrip('.,;').lower())
            if not target and arxiv:
                target = self.reference_ids.get('arxiv:' + arxiv.group(1))
            candidates = self.reference_titles.get(compact_key(title), [])
            if not target and len(candidates) == 1:
                metadata = self.metadata[candidates[0]]
                if not year or not metadata.get('year') or str(metadata['year']) == year:
                    target = candidates[0]
            results.append({'local_index': match.group(1) or match.group(2), 'raw_reference': raw,
                            'title': title, 'year': year, 'paper_id': target,
                            'status': 'resolved' if target else 'unresolved'})
        return results

    def close(self) -> None:
        self.db.close()
