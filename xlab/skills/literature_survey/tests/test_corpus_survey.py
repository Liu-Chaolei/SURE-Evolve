import sys
import json
from pathlib import Path
import unittest
import tempfile
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from literature_survey_lib.corpus_survey import validate_section, complete_section_length, LocalSurveyClient, short_context_section, finish_sections


class CorpusSurveyTests(unittest.TestCase):
    def test_176_word_blocks_are_preserved_and_supplemented_with_two_reviews(self):
        evidence = [{'paper_id': 'p1', 'sha256': 'a' * 64, 'level': 'deep', 'title': 'Method', 'year': 2024,
                     'facts': [{'fact_index': 0, 'quote': 'Evaluated on test data.', 'span': [0, 23],
                                'summary': 'Evaluated on test data.', 'kind': 'method', 'name': 'Method'}]}]
        class Client:
            calls = 0
            def fits(self, system, user, maximum):
                return True
            def call(self, system, user, maximum):
                self.calls += 1
                sentence = {'text': ' '.join(['analysis'] * 22) + '.', 'paper_id': 'p1', 'fact_index': 0}
                return {'paragraphs': [{'sentences': [sentence] * 4}] * 2}
        client = Client()
        with tempfile.TemporaryDirectory() as tmp:
            result = short_context_section(client, {'title': 'Methods'}, evidence, Path(tmp), 0, 'signature')
            self.assertEqual(result['word_count'], 880)
            self.assertEqual(client.calls, 15)
            for path in Path(tmp).glob('*block*.json'):
                self.assertEqual(json.loads(path.read_text())['review_rounds'], 2)
            short_context_section(client, {'title': 'Methods'}, evidence, Path(tmp), 0, 'signature')
            self.assertEqual(client.calls, 15)

    def test_failed_section_does_not_discard_other_sections_and_only_failures_retry(self):
        attempts = {index: 0 for index in range(4)}
        def job(item):
            index, _ = item
            attempts[index] += 1
            if index == 0 and attempts[index] == 1:
                raise ValueError('temporary response failure')
            return {'index': index}
        with tempfile.TemporaryDirectory() as tmp:
            result = finish_sections(job, [(index, {}) for index in range(4)], Path(tmp))
            self.assertEqual(result, [{'index': index} for index in range(4)])
            self.assertEqual(attempts, {0: 2, 1: 1, 2: 1, 3: 1})
            self.assertEqual(json.loads((Path(tmp) / 'progress.json').read_text())['failed_sections'], {})

    def test_short_context_blocks_keep_reviews_grounding_and_resume(self):
        evidence = [{'paper_id': 'p1', 'sha256': 'a' * 64, 'level': 'deep', 'title': 'Method', 'year': 2024,
                     'facts': [{'fact_index': 0, 'quote': 'Evaluated on test data.', 'span': [0, 23],
                                'summary': 'Evaluated on test data.', 'kind': 'method', 'name': 'Method'}]}]
        class Client:
            calls = 0
            def fits(self, system, user, maximum):
                return True
            def call(self, system, user, maximum):
                self.calls += 1
                sentence = {'text': ' '.join(['analysis'] * 22) + '.', 'paper_id': 'p1', 'fact_index': 0}
                return {'paragraphs': [{'sentences': [sentence] * 5}] * 2}
        client = Client()
        with tempfile.TemporaryDirectory() as tmp:
            result = short_context_section(client, {'title': 'Methods'}, evidence, Path(tmp), 0, 'signature')
            self.assertEqual(result['word_count'], 880)
            self.assertEqual(client.calls, 12)
            self.assertEqual(len(result['claims']), 40)
            self.assertEqual(result['claims'][0]['source_span'], [0, 23])
            short_context_section(client, {'title': 'Methods'}, evidence, Path(tmp), 0, 'signature')
            self.assertEqual(client.calls, 12)

    def test_direct_endpoint_counts_template_and_enforces_8192_limit(self):
        def transport(request):
            self.assertEqual(request.url.path, '/tokenize')
            self.assertFalse(json.loads(request.content)['chat_template_kwargs']['enable_thinking'])
            return httpx.Response(200, json={'count': 6000})
        client = object.__new__(LocalSurveyClient)
        client.direct = True
        client.context = 8192
        with httpx.Client(base_url='http://test', transport=httpx.MockTransport(transport)) as session:
            client.client = session
            self.assertTrue(client.fits('system', 'user', 1500))
            self.assertFalse(client.fits('system', 'user', 2048))

    def test_truncated_local_responses_retry_with_a_verified_larger_budget(self):
        budgets = []
        def transport(request):
            if request.url.path == '/research/count':
                return httpx.Response(200, json={'tokens': 1000})
            budgets.append(json.loads(request.content)['max_tokens'])
            if len(budgets) == 1:
                return httpx.Response(400, json={'error': 'Incomplete response: length'})
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': '{"paragraphs":[]}'}}]})
        client = object.__new__(LocalSurveyClient)
        client.context = 32768
        with httpx.Client(base_url='http://test', transport=httpx.MockTransport(transport)) as session:
            client.client = session
            self.assertEqual(client.call('system', 'user'), {'paragraphs': []})
        self.assertEqual(budgets, [6144, 8192])

    def test_short_drafts_gain_evidence_linked_paragraphs_instead_of_padding(self):
        sentence = {'text': ' '.join(['analysis'] * 15) + '.', 'paper_id': 'p1', 'fact_index': 0}
        paragraph = {'sentences': [sentence] * 5}
        class Client:
            calls = 0
            def call(self, system, user):
                self.calls += 1
                return {'paragraphs': [paragraph] * 4}
        client = Client()
        value = complete_section_length(client, 'evidence', {'paragraphs': [paragraph] * 8})
        evidence = [{'paper_id': 'p1', 'sha256': 'a' * 64, 'level': 'deep',
                     'facts': [{'fact_index': 0, 'quote': 'We evaluate on the test set.', 'span': [0, 28]}]}]
        self.assertEqual(validate_section(value, evidence)['word_count'], 900)
        self.assertEqual(client.calls, 1)

    def test_structured_sentences_render_citations_and_source_traces(self):
        evidence = [{'paper_id': 'p1', 'sha256': 'a' * 64, 'level': 'deep',
                     'facts': [{'fact_index': 0, 'quote': 'We evaluate on the test set.', 'span': [0, 28]}]}]
        sentence = ' '.join(['analysis'] * 20) + '.'
        value = {'paragraphs': [{'sentences': [{'text': sentence, 'paper_id': 'p1', 'fact_index': 0}
                                              for _ in range(5)]} for _ in range(8)]}
        result = validate_section(value, evidence)
        self.assertEqual(result['markdown'].count('[paper:p1]'), 40)
        self.assertEqual(len(result['claims']), 40)
        self.assertEqual(result['claims'][0]['quote'], evidence[0]['facts'][0]['quote'])
        value['paragraphs'][0]['sentences'][0]['paper_id'] = 'invented'
        with self.assertRaises(ValueError):
            validate_section(value, evidence)

    def test_rejects_unknown_citations_and_untraced_paragraphs(self):
        statement = 'The method is evaluated on a clearly identified benchmark.'
        paragraph = statement + ' ' + ' '.join(['analysis'] * 795) + ' [paper:p1]'
        evidence = [{'paper_id': 'p1', 'sha256': 'a' * 64, 'level': 'deep',
                     'facts': [{'fact_index': 0, 'quote': 'We evaluate this method on the test set.', 'span': [12, 54]}]}]
        value = {'markdown': paragraph, 'claims': [{'statement': statement, 'paper_id': 'p1', 'fact_index': 0, 'quote': evidence[0]['facts'][0]['quote']}]}
        self.assertGreaterEqual(validate_section(value, evidence)['word_count'], 800)
        value['markdown'] = paragraph.replace('benchmark.', 'benchmark [paper:p1].')
        self.assertGreaterEqual(validate_section(value, evidence)['word_count'], 800)
        value['markdown'] = paragraph
        value['claims'][0].pop('quote')
        resolved = validate_section(value, evidence)['claims'][0]
        self.assertEqual(resolved['quote'], evidence[0]['facts'][0]['quote'])
        self.assertEqual(resolved['source_span'], [12, 54])
        value['claims'][0]['quote'] = 'Fabricated evidence sentence.'
        with self.assertRaises(ValueError):
            validate_section(value, evidence)
        value['claims'][0].pop('quote')
        value['markdown'] = paragraph.replace('[paper:p1]', '[paper:invented]')
        with self.assertRaises(ValueError):
            validate_section(value, evidence)
        value['markdown'] = paragraph
        value['claims'] = []
        with self.assertRaises(ValueError):
            validate_section(value, evidence)


if __name__ == '__main__':
    unittest.main()
