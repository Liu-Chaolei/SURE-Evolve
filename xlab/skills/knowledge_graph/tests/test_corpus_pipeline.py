from __future__ import annotations

import hashlib
import io
import json
import csv
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from knowledge_graph_lib.corpus_extract import validate_facts, quote_span, signature_for, evidence_blocks, numeric_supported, extract_paper
from knowledge_graph_lib.corpus_graph import select_core
from knowledge_graph_lib.corpus_flow import Flow
from knowledge_graph_lib.corpus_llm import CorpusLlm, OutputInvalid
from knowledge_graph_lib.corpus_store import CorpusStore, archive_json, contained


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(text)

    def apply_chat_template(self, messages, **kwargs):
        return list(json.dumps(messages))


class CorpusTests(unittest.TestCase):
    def test_catalog_replacement_retries_and_stale_snapshots_block_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / 'catalog.csv'
            content = 'id,title\np1,Paper one\n'
            catalog.write_text(content)
            store = CorpusStore(root, root / 'run', root / 'local/queue.sqlite')
            with patch.object(Path, 'read_bytes', side_effect=[FileNotFoundError(2, 'Catalog replaced'), content.encode()]), patch('knowledge_graph_lib.corpus_store.time.sleep'):
                self.assertEqual(store.read_catalog(), content)
            self.assertTrue(store.catalog_is_current())
            catalog.unlink()
            with patch('knowledge_graph_lib.corpus_store.time.sleep'):
                self.assertEqual(store.read_catalog(), content)
            self.assertFalse(store.catalog_is_current())
            catalog.write_text(content + 'p2,Paper two\n')
            self.assertFalse(store.catalog_is_current())
            self.assertIn('p2', store.read_catalog())
            self.assertTrue(store.catalog_is_current())
            store.close()

    def test_cached_false_core_is_corrected_durably_without_repeating_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = 'We use StarGAN-v2VC [19] for the anonymization experiments.'
            (root / 'source.md').write_text(source)
            store = CorpusStore(root, root / 'run', root / 'local/queue.sqlite')
            llm = CorpusLlm(root / 'run', CharacterTokenizer(), {})
            paper = {'sha256': 'a' * 64, 'paper_id': 'test', 'metadata': {'title': 'Test'},
                     'parse_ref': {'asset_root': str(root), 'markdown_path': 'source.md',
                                   'markdown_sha256': hashlib.sha256(source.encode()).hexdigest()}}
            output = root / 'run/artifacts/basic' / ('a' * 64 + '.json')
            output.write_text(json.dumps({'signature': signature_for(llm, paper, 'basic'), 'quality': {'grounded': True},
                'domains': ['TTS'], 'facets': {}, 'facts': [{'kind': 'core', 'name': 'StarGAN-v2VC',
                'quote': source, 'summary': 'The paper uses this model.'}], 'no_named_core': False}))
            with patch.object(llm, 'complete', side_effect=AssertionError('No new inference expected')):
                result = extract_paper(store, llm, paper, 'basic')
            self.assertEqual(result['facts'][0]['kind'], 'method')
            self.assertTrue(json.loads(output.read_text())['no_named_core'])
            store.close()

    def test_cited_predecessors_are_not_original_contribution_nodes(self):
        source = 'We propose two approaches. StarGAN-v2VC [19] is a prior voice conversion model. We introduce NewVoice for this task.'
        value = {'domains': ['TTS'], 'facets': {}, 'facts': [
            {'kind': 'core', 'name': 'StarGAN-v2VC', 'quote': source, 'summary': 'The system uses StarGAN-v2VC.'},
            {'kind': 'core', 'name': 'NewVoice', 'quote': source, 'summary': 'The paper introduces NewVoice.'}]}
        facts = validate_facts(value, source)['facts']
        self.assertEqual([fact['kind'] for fact in facts], ['method', 'core'])

    def test_pilot_resume_keeps_the_sample_when_catalog_grows(self):
        class PilotStore:
            def __init__(self, run):
                self.run = run
                self.papers = [{'sha256': str(i), 'metadata': {'tags': [domain]}, 'parse_ref': {'pages': 10}}
                               for i, domain in enumerate(('ASR', 'TTS', 'SD'))]
            def all_papers(self):
                return self.papers
            def finish(self, sha, stage):
                pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = PilotStore(root)
            flow = Flow(store, CorpusLlm(root, CharacterTokenizer(), {}))
            with patch('knowledge_graph_lib.corpus_flow.extract_paper', return_value={}) as extract:
                self.assertTrue(flow.pilot(per_domain=1)['passed'])
                selected = {call.args[2]['sha256'] for call in extract.call_args_list}
                extract.reset_mock()
                store.papers.extend({'sha256': str(i), 'metadata': {'tags': ['ASR', 'TTS', 'SD']}, 'parse_ref': {'pages': 1}} for i in range(3, 30))
                self.assertTrue(flow.pilot(per_domain=1)['passed'])
                self.assertEqual(selected, {call.args[2]['sha256'] for call in extract.call_args_list})

    def test_numeric_math_rendering_preserves_digits_and_reference_boundaries(self):
        self.assertTrue(numeric_supported('4.20 ± 0.06', r'MOS is $4.20 \pm 0.06$.'))
        self.assertFalse(numeric_supported('4.21 ± 0.06', r'MOS is $4.20 \pm 0.06$.'))
        self.assertFalse(numeric_supported('4.20', '14.20'))
        self.assertFalse(numeric_supported('4.20', '4.2 0'))
        source = 'A grounded conclusion remains usable.\n\n# References\nA cited paper is not experimental evidence.'
        blocks = evidence_blocks({'text': source})
        self.assertEqual(''.join(blocks.values()), source)
        self.assertFalse(any('conclusion' in block and 'References' in block for block in blocks.values()))

    def test_calibration_can_remove_a_whole_group_without_faking_a_paper(self):
        value = {'domains': ['ASR'], 'facets': {}, 'facts': []}
        self.assertEqual(validate_facts(value, 'source text', allow_empty=True)['facts'], [])
        with self.assertRaises(ValueError):
            validate_facts(value, 'source text')

    def test_evidence_ids_resolve_raw_formula_spans_and_reject_unknown_ids(self):
        source = 'The method scores $\\mathrm { WER } = 3.2$ on the test split.\n' * 60
        evidence = evidence_blocks({'text': source})
        self.assertEqual(''.join(evidence.values()), source)
        value = {'domains': ['ASR'], 'facets': {}, 'facts': [
            {'kind': 'result', 'quote_id': 'e0', 'quote': 'incorrectly reformatted', 'summary': 'The reported WER is 3.2.',
             'metric': {'name': 'WER', 'value': '3.2', 'context_quote_id': 'e0'}}]}
        result = validate_facts(value, source, evidence)
        fact = result['facts'][0]
        self.assertEqual(fact['quote'], evidence['e0'])
        self.assertEqual(source[slice(*fact['span'])], fact['quote'])
        value['facts'][0]['quote_id'] = 'invented'
        with self.assertRaisesRegex(ValueError, 'Unknown quote_id'):
            validate_facts(value, source, evidence)

    def test_failed_grounding_preserves_actionable_diagnostics(self):
        source = 'Speech perception varies across listeners.'
        invalid = {'domains': ['RELATED'], 'facets': {}, 'facts': [
            {'kind': 'result', 'quote': 'The model ... achieves a fabricated result.'}]}
        class InvalidLlm(CorpusLlm):
            def complete(self, system, user, **kwargs):
                self.last_prompt = user
                return invalid, {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            llm = InvalidLlm(root, CharacterTokenizer(), {})
            with self.assertRaisesRegex(OutputInvalid, 'not a verbatim'):
                llm.validated('extract', source, maximum=100, tag='test',
                              validator=lambda value: validate_facts(value, source),
                              cache_path=root / 'pass.json', signature='test')
            self.assertIn('fabricated result', llm.last_prompt)
            failure = json.loads((root / 'pass.failure.json').read_text())
            self.assertEqual(len(failure['failures']), 2)
            self.assertFalse((root / 'pass.json').exists())

    def test_counts_token_ids_not_batch_encoding_keys(self):
        class MappingTokenizer(CharacterTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return {'input_ids': list(range(2500)), 'attention_mask': [1] * 2500}
        with tempfile.TemporaryDirectory() as directory:
            llm = CorpusLlm(Path(directory), MappingTokenizer(), {})
            self.assertEqual(llm.count([{'role': 'user', 'content': 'text'}]), 2500)

    def test_import_reuses_verified_bundle_and_catalog_growth_is_local(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            legacy = base / 'parsing_20260912'
            (base / 'artifacts/pdfs').mkdir(parents=True)
            (base / 'artifacts/metadata').mkdir()
            (legacy / 'checkpoints').mkdir(parents=True)
            pdf = base / 'artifacts/pdfs/one.pdf'
            pdf.write_bytes(b'%PDF-1.7\none\n%%EOF')
            sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
            md = legacy / 'one.md'
            md.write_text('Speech perception varies across listeners.')
            archive = legacy / 'one.tar.gz'
            with tarfile.open(archive, 'w:gz') as handle:
                for name, data in [('doc/content.json', b'[{"type":"text","text":"Speech"}]'), ('doc/middle.json', b'{"pdf_info":[{}]}')]:
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    handle.addfile(info, io.BytesIO(data))
            record = {'sha256': sha, 'status': 'parsed', 'paper_id': 'one', 'markdown_path': 'one.md',
                      'markdown_sha256': hashlib.sha256(md.read_bytes()).hexdigest(), 'archive': 'one.tar.gz',
                      'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'pages': 1,
                      'content_list_member': 'doc/content.json', 'middle_json_member': 'doc/middle.json'}
            (legacy / 'checkpoints/worker-0.jsonl').write_text(json.dumps(record) + '\n')
            (base / 'artifacts/metadata/papers.jsonl').write_text(json.dumps({'id': 'one', 'title': 'First paper', 'year': 2020}) + '\n')
            fields = ['id', 'title', 'year', 'venue', 'tags', 'status', 'sha256', 'pdf_path']
            with (base / 'catalog.csv').open('w') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict(zip(fields, ['one', 'First paper', '2020', 'Test', 'ASR', 'downloaded', sha, 'artifacts/pdfs/one.pdf'])))
            store = CorpusStore(base, base / 'run', base / 'local/queue.db')
            store.ingest()
            llm = CorpusLlm(base / 'run', CharacterTokenizer(), {'model': 'fixed'})
            before = signature_for(llm, store.paper(sha), 'basic')
            with (base / 'catalog.csv').open('a') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writerow(dict(zip(fields, ['two', 'Unrelated paper', '2021', 'Test', 'TTS', 'unavailable', '', ''])))
            store.ingest()
            self.assertEqual(before, signature_for(llm, store.paper(sha), 'basic'))
            self.assertEqual(store.summary()['stages']['parse']['done'], 1)
            self.assertEqual(store.summary()['papers'], 2)
            resumed = CorpusStore(base, base / 'run', base / 'local/pilot.db')
            resumed.ingest(parse_limit=0, parse_hashes=set())
            self.assertEqual(resumed.summary()['stages']['parse'].get('done', 0), 0)
            with patch('knowledge_graph_lib.corpus_store.tarfile.open', side_effect=AssertionError('Validated unchanged archive must not be decompressed again')):
                resumed.ingest(parse_limit=0, parse_hashes={sha})
            self.assertEqual(resumed.summary()['stages']['parse']['done'], 1)
            resumed.close()
            manifest = base / 'run/artifacts/parses' / f'{sha}.json'
            canonical = json.loads(manifest.read_text())
            manifest.write_text(json.dumps({**canonical, 'pages': 2}))
            altered = CorpusStore(base, base / 'run', base / 'local/altered.db')
            with self.assertRaises(ValueError):
                altered.ingest(parse_limit=0, parse_hashes={sha})
            altered.close()
            manifest.write_text(json.dumps(canonical))
            md.write_text('Corrupted output')
            fresh = CorpusStore(base, base / 'another-run', base / 'local/fresh.db')
            with self.assertRaises(ValueError):
                fresh.ingest()
            store.close()
            fresh.close()

    def test_archive_rejects_links_and_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'bundle.tar.gz'
            with tarfile.open(archive, 'w:gz') as handle:
                info = tarfile.TarInfo('document/middle.json')
                data = b'{"pdf_info":[{}]}'
                info.size = len(data)
                handle.addfile(info, io.BytesIO(data))
                link = tarfile.TarInfo('document/link.json')
                link.type = tarfile.SYMTYPE
                link.linkname = '/etc/passwd'
                handle.addfile(link)
            self.assertEqual(len(archive_json(archive, 'document/middle.json')['pdf_info']), 1)
            for member in ('../outside.json', '/etc/passwd', 'document/link.json'):
                with self.assertRaises(ValueError):
                    archive_json(archive, member)
            with self.assertRaises(ValueError):
                contained(root, '../outside')

    def test_chunking_covers_all_text_and_preserves_raw_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            llm = CorpusLlm(Path(directory), CharacterTokenizer(), {}, context=4096)
            source = ('A sentence with evidence.\n\n' * 900) + 'Final conclusion must remain.'
            chunks = llm.chunks(source, reserve=2048, maximum=1800)
            self.assertGreater(len(chunks), 1)
            self.assertEqual(chunks[0]['start'], 0)
            self.assertEqual(chunks[-1]['end'], len(source))
            previous_end = 0
            for chunk in chunks:
                self.assertLessEqual(chunk['start'], previous_end)
                self.assertEqual(chunk['text'], source[chunk['start']:chunk['end']])
                self.assertLessEqual(len(chunk['text']), 1800)
                previous_end = chunk['end']

    def test_grounding_allows_no_named_core_but_not_reference_only_claims(self):
        source = 'Speech perception varies across listeners.\n\n# References\nA fictitious model achieves 99 percent.'
        basic = {'domains': ['RELATED'], 'facets': {}, 'paper_type': 'theory',
                 'facts': [{'kind': 'result', 'summary': 'Variation across listeners.', 'quote': 'Speech perception varies across listeners.'}]}
        value = validate_facts(basic, source)
        self.assertTrue(value['no_named_core'])
        self.assertEqual(value['facts'][0]['span'], [0, 42])
        basic['facts'][0]['quote'] = 'A fictitious model achieves 99 percent.'
        with self.assertRaises(ValueError):
            validate_facts(basic, source)
        self.assertIsNone(quote_span('Nonexistent evidence quote.', source))

    def test_leases_are_exclusive_and_restart_requeues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / 'local/queue.db'
            store = CorpusStore(root, root / 'run', db)
            with store.lock:
                store.db.execute('INSERT INTO papers(sha,paper_id,metadata,pdf) VALUES(?,?,?,?)', ('a' * 64, 'paper-a', '{}', '/source.pdf'))
                store.db.execute("INSERT INTO tasks(sha,stage,state) VALUES(?,'parse','pending')", ('a' * 64,))
            self.assertEqual(len(store.claim('parse', 'worker-one')), 1)
            self.assertEqual(store.claim('parse', 'worker-two'), [])
            store.close()
            resumed = CorpusStore(root, root / 'run', db)
            self.assertEqual(len(resumed.claim('parse', 'worker-two')), 1)
            resumed.close()

    def test_core_selection_is_deterministic_and_counts_overlap_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'artifacts/basic').mkdir(parents=True)
            for i in range(40):
                row = {'sha256': f'{i:064x}', 'paper_id': str(i), 'signature': str(i), 'domains': ['ASR', 'TTS'],
                       'facets': {'ASR': 'end-to-end', 'TTS': 'language-model'}, 'metadata': {'year': 1990 + i},
                       'paper_type': 'method', 'facts': [], 'references': []}
                (root / 'artifacts/basic' / f'{i}.json').write_text(json.dumps(row))
            first = select_core(root, 20)
            second = select_core(root, 20)
            self.assertEqual(first['domains'], second['domains'])
            self.assertEqual(len(first['domains']['ASR']), 20)
            self.assertEqual(len(first['unique_hashes']), 20)


if __name__ == '__main__':
    unittest.main()
