from __future__ import annotations

import collections
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

from .common import atomic_write_json, atomic_write_jsonl, compact_key
from .corpus_store import CorpusStore, fingerprint


def records(run: Path, level: str) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted((run / 'artifacts' / level).glob('*.json'))]


def select_core(run: Path, count: int = 500) -> dict:
    papers = records(run, 'basic')
    citations = collections.Counter(ref['paper_id'] for paper in papers for ref in paper.get('references', []) if ref.get('paper_id'))
    selection = {}
    for domain in ('ASR', 'TTS', 'SD'):
        cells = collections.defaultdict(list)
        for paper in papers:
            if domain not in paper['domains']:
                continue
            raw_year = paper['metadata'].get('year')
            year = int(raw_year) if str(raw_year).isdigit() else 0
            era = 'unknown' if not year else ('pre2000' if year < 2000 else '2000s' if year < 2010 else '2010s' if year < 2020 else '2020-22' if year < 2023 else '2023+')
            cell = (paper['facets'].get(domain, 'other'), era)
            metrics = sum(fact.get('metric') is not None for fact in paper['facts'])
            priority = paper['paper_type'] in ('method', 'dataset', 'benchmark', 'system')
            rank = (-int(priority), -metrics, -citations[paper['paper_id']], -year, paper['paper_id'])
            cells[cell].append((rank, paper['sha256']))
        for values in cells.values():
            values.sort()
        offsets = {cell: 0 for cell in cells}
        chosen = []
        for cell in sorted(cells):
            limit = min(5, len(cells[cell]), count - len(chosen))
            chosen.extend(item[1] for item in cells[cell][:limit])
            offsets[cell] = limit
        while len(chosen) < count:
            available = [cell for cell in cells if offsets[cell] < len(cells[cell])]
            if not available:
                break
            cell = min(available, key=lambda key: (-math.sqrt(len(cells[key])) / (1 + max(0, offsets[key] - 5)), key))
            chosen.append(cells[cell][offsets[cell]][1])
            offsets[cell] += 1
        selection[domain] = chosen
    result = {'schema_version': 'xlab.core_selection.v1', 'per_domain_target': count,
              'domains': selection, 'unique_hashes': sorted(set(sha for values in selection.values() for sha in values)),
              'basic_snapshot': fingerprint([(paper['sha256'], paper['signature']) for paper in papers]), 'created_at': time.time()}
    atomic_write_json(run / 'core-selection.json', result)
    return result


def build_corpus_graph(store: CorpusStore) -> dict:
    run = store.run
    basic = {row['sha256']: row for row in records(run, 'basic')}
    deep = {row['sha256']: row for row in records(run, 'deep')}
    nodes = {}
    edges = {}
    paper_nodes = {}
    all_papers = store.all_papers()
    for paper in all_papers:
        identifier = 'paper:' + paper['sha256']
        paper_nodes[paper['paper_id']] = identifier
        evidence = deep.get(paper['sha256']) or basic.get(paper['sha256'])
        nodes[identifier] = {'id': identifier, 'type': 'Paper', 'label': paper['metadata']['title'],
                             'paper_id': paper['paper_id'], 'sha256': paper['sha256'], 'metadata': paper['metadata'],
                             'extraction_level': evidence['extraction_level'] if evidence else 'metadata_only',
                             'domains': evidence['domains'] if evidence else paper['metadata'].get('tags', []),
                             'facts': evidence['facts'] if evidence else []}
    core_names = collections.defaultdict(list)
    for sha, paper in {**basic, **deep}.items():
        parent = paper_nodes.get(paper['paper_id'])
        for fact in paper['facts']:
            if fact['kind'] == 'core' and fact.get('name'):
                identifier = 'core:' + fingerprint([sha, compact_key(fact['name'])])[:24]
                nodes[identifier] = {'id': identifier, 'type': 'Core', 'label': fact['name'], 'paper_id': paper['paper_id'], 'evidence': fact}
                core_names[compact_key(fact['name'])].append(identifier)
                edge = {'source': parent, 'target': identifier, 'relation': 'introduces', 'edge_type': 'introduces', 'evidence': fact, 'paper_id': paper['paper_id']}
                edges[fingerprint(edge)] = edge
    unresolved = 0
    for paper in {**basic, **deep}.values():
        parent = paper_nodes.get(paper['paper_id'])
        cited = {ref['paper_id'] for ref in paper.get('references', []) if ref.get('paper_id')}
        for ref in paper.get('references', []):
            target = paper_nodes.get(ref.get('paper_id'))
            if target and target != parent:
                edge = {'source': parent, 'target': target, 'relation': 'cites', 'edge_type': 'cites', 'reference': ref}
                edges[fingerprint(edge)] = edge
            elif not target:
                unresolved += 1
        for fact in paper['facts']:
            if fact['kind'] not in ('dataset', 'baseline', 'method') or not fact.get('name'):
                continue
            kind = 'Dataset' if fact['kind'] == 'dataset' else 'Baseline'
            key = compact_key(fact['name'])
            matches = [identifier for identifier in core_names.get(key, []) if nodes[identifier]['paper_id'] in cited]
            # Acronyms alone are not a cross-paper identity. A cited, unique
            # introduced Core can disambiguate a baseline; otherwise retain it.
            identifier = matches[0] if len(matches) == 1 and kind != 'Dataset' else kind.lower() + ':' + fingerprint([kind, key, paper['paper_id'] if len(key) < 8 else ''])[:24]
            nodes.setdefault(identifier, {'id': identifier, 'type': kind, 'label': fact['name'], 'evidence': fact})
            relation = 'uses_dataset' if kind == 'Dataset' else 'compares_with' if fact['kind'] == 'baseline' else 'uses_method'
            edge = {'source': parent, 'target': identifier, 'relation': relation, 'edge_type': relation, 'evidence': fact, 'paper_id': paper['paper_id']}
            edges[fingerprint(edge)] = edge
    output = run / 'artifacts/graphs'
    atomic_write_jsonl(output / 'nodes.jsonl', nodes.values())
    atomic_write_jsonl(output / 'edges.jsonl', edges.values())
    graph = {'schema_version': 'xlab.method_graph.v3', 'source_kind': 'bulk_archive', 'nodes': list(nodes.values()), 'edges': list(edges.values())}
    atomic_write_json(output / 'method_graph.json', graph)
    for domain in ('ASR', 'TTS', 'SD'):
        included = {key for key, value in nodes.items() if value['type'] == 'Paper' and domain in value['domains']}
        domain_edges = [edge for edge in edges.values() if edge['source'] in included]
        included.update(edge['target'] for edge in domain_edges)
        atomic_write_json(output / f'{domain.lower()}.json', {**graph, 'nodes': [nodes[key] for key in sorted(included)], 'edges': domain_edges})
    local = Path(os.environ.get('SLURM_TMPDIR', '/local/job')) if os.environ.get('SLURM_JOB_ID') else Path(tempfile.gettempdir())
    local.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='corpus-graph-', suffix='.sqlite', dir=local)
    os.close(descriptor)
    temporary = Path(name)
    db = sqlite3.connect(temporary)
    db.executescript('''PRAGMA foreign_keys=ON;
        CREATE TABLE nodes(id TEXT PRIMARY KEY,type TEXT,label TEXT,data TEXT);
        CREATE TABLE edges(id TEXT PRIMARY KEY,source TEXT REFERENCES nodes(id),target TEXT REFERENCES nodes(id),relation TEXT,data TEXT);
        CREATE VIRTUAL TABLE search USING fts5(id UNINDEXED,label,body);
        CREATE INDEX edge_source ON edges(source); CREATE INDEX edge_target ON edges(target);''')
    db.executemany('INSERT INTO nodes VALUES(?,?,?,?)', [(key, node['type'], node['label'], json.dumps(node, ensure_ascii=False)) for key, node in nodes.items()])
    db.executemany('INSERT INTO edges VALUES(?,?,?,?,?)', [(key, edge['source'], edge['target'], edge['relation'], json.dumps(edge, ensure_ascii=False)) for key, edge in edges.items()])
    db.executemany('INSERT INTO search VALUES(?,?,?)', [(key, node['label'], json.dumps(node.get('facts') or node.get('evidence', {}), ensure_ascii=False)) for key, node in nodes.items()])
    db.commit()
    assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    assert not db.execute('PRAGMA foreign_key_check').fetchall()
    assert db.execute('SELECT count(*) FROM nodes').fetchone()[0] == len(nodes)
    db.close()
    published = output / 'graph.db.partial'
    shutil.copyfile(temporary, published)
    published.replace(output / 'graph.db')
    temporary.unlink()
    report = {'nodes': len(nodes), 'edges': len(edges), 'basic': len(basic), 'deep': len(deep),
              'unresolved_references': unresolved, 'integrity_passed': True, 'updated_at': time.time()}
    atomic_write_json(output / 'graph-report.json', report)
    return report
