from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from research_idea_lib.algorithm.fusion import FusionRequest, REFEREE_METRICS, fuse_five_modes, validate_fusion_output
from research_idea_lib.algorithm.provider_adapter import (
    AdapterOutputError, FUSION_GENERATE_OPERATION, ProviderAdapter,
    _format_fusion_prompt, _format_repair_prompt, _normalize_fusion, _validate_fusion_sources,
)
from research_idea_lib.providers import ProviderResult, ProviderTrace, ProviderUsage
from test_provider_adapter import fused_output, mode_inputs


class QueueProvider:
    def __init__(self, outputs):
        self.outputs = deepcopy(outputs)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        value = self.outputs.pop(0)
        return ProviderResult(
            text=json.dumps(value), json_value=deepcopy(value), usage=ProviderUsage(),
            trace=ProviderTrace(provider='fixture', operation=request.operation,
                                input_digest=request.input_digest, output_kind='json',
                                model=request.model, attempts=1, status='success'),
        )


class FusionSourceContractTests(unittest.TestCase):
    def test_rejected_provenance_identity_includes_source_mode(self):
        inputs = mode_inputs()
        output = fused_output()
        output['rejected_components'] = [
            {'component': 'core-v2', 'source_mode': item['mode'], 'evidence': item['evidence']}
            for item in inputs
        ]
        for item in inputs:
            if 'core-v2' not in item['idea']['components']:
                item['idea']['components'].append('core-v2')
        validate_fusion_output(output, inputs)
        adapter = ProviderAdapter(QueueProvider([output]), model='glm-5.3-flash')
        adapter.generate({'mode_inputs': inputs})
        output['rejected_components'].append(deepcopy(output['rejected_components'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate component provenance'):
            validate_fusion_output(output, inputs)
        adapter = ProviderAdapter(QueueProvider([output]), model='glm-5.3-flash')
        with self.assertRaisesRegex(AdapterOutputError, 'Duplicate component provenance'):
            adapter.generate({'mode_inputs': inputs})

    def test_catalog_uses_top_level_sources_and_keeps_input_unchanged(self):
        inputs = mode_inputs()
        inputs[0]['trace'] = [{'idea': {'components': ['child-only component']},
                               'evidence': ['reviewer prose']}]
        original = deepcopy(inputs)
        prompt = _format_fusion_prompt({'mode_inputs': inputs})
        catalog = prompt.split('== Authoritative selectable source catalog ==')[1].splitlines()[1]
        parsed = json.loads(catalog)
        self.assertEqual(parsed['moonshot_inventor']['evidence_ids'], ['evidence:moonshot_inventor'])
        self.assertEqual([x['name'] for x in parsed['moonshot_inventor']['components']], ['core', 'core-v2'])
        self.assertNotIn('child-only component', catalog)
        self.assertNotIn('reviewer prose', catalog)
        self.assertEqual(inputs, original)
        self.assertIn('bare evidence IDs', prompt)
        self.assertIn('Authoritative selectable source catalog', _format_repair_prompt({'mode_inputs': inputs}))

    def test_bad_evidence_is_never_stripped_guessed_or_reassigned(self):
        for bad in ['evidence:moonshot_inventor (survey: explanation)',
                    'A reviewer found the component feasible.', 'evidence:evidence_first']:
            with self.subTest(bad=bad):
                output = fused_output()
                output['selected_components'][0]['evidence'] = [bad]
                original = deepcopy(output)
                with self.assertRaises(AdapterOutputError) as caught:
                    _validate_fusion_sources(_normalize_fusion(output), mode_inputs())
                issue = caught.exception.validation_issues[0]
                self.assertEqual(issue['received']['evidence'], [bad])
                self.assertEqual(issue['allowed_evidence_ids'], ['evidence:moonshot_inventor'])
                self.assertEqual(output, original)

    def test_all_source_errors_are_sent_with_the_previous_draft_on_retry(self):
        invalid = fused_output()
        invalid['selected_components'][0]['evidence'] = ['component description']
        invalid['rejected_components'][0]['component'] = 'whole idea title'
        invalid['conflict_resolutions'][0]['evidence'] = ['evidence:evidence_first']
        original = deepcopy(invalid)
        provider = QueueProvider([invalid, fused_output(), {x: 3 for x in REFEREE_METRICS}])
        adapter = ProviderAdapter(provider, model='glm-5.3-flash')
        result = fuse_five_modes(FusionRequest(mode_inputs(), max_repair_steps=0),
                                 generator=adapter, evaluator=adapter)
        self.assertEqual(result.metadata['fusion_draft_attempts'], 2)
        retry = provider.requests[1].structured_input
        self.assertEqual(retry['previous_draft'], original)
        self.assertEqual([x['path'] for x in retry['validation_issues']],
                         ['selected_components[0]', 'rejected_components[0]',
                          'conflict_resolutions[0].evidence'])
        self.assertEqual(json.loads(provider.requests[1].user_prompt)['validation_issues'], retry['validation_issues'])
        self.assertEqual(invalid, original)
        self.assertNotIn('previous_draft', provider.requests[0].structured_input)

    def test_retry_budget_still_fails_closed_without_referee_calls(self):
        invalid = fused_output()
        invalid['selected_components'][0]['evidence'] = ['wrong']
        provider = QueueProvider([invalid] * 3)
        adapter = ProviderAdapter(provider, model='glm-5.3-flash')
        with self.assertRaisesRegex(ValueError, 'exhausted semantic draft attempts'):
            fuse_five_modes(FusionRequest(mode_inputs(), max_fusion_draft_attempts=3),
                            generator=adapter, evaluator=adapter)
        self.assertEqual(len(provider.requests), 3)
        self.assertTrue(all(r.operation == FUSION_GENERATE_OPERATION for r in provider.requests))

    def test_identical_modes_allow_exact_shared_components_without_fabrication(self):
        inputs = mode_inputs()
        for item in inputs:
            item['idea'] = deepcopy(inputs[0]['idea'])
            item['evidence'] = ['evidence:shared']
        output = fused_output()
        output['idea']['components'] = ['core']
        output['selected_components'] = [{'component': 'core', 'source_mode': 'steady_engineer',
                                          'evidence': ['evidence:shared']}]
        output['rejected_components'] = []
        output['conflict_resolutions'] = []
        provider = QueueProvider([output, {x: 3 for x in REFEREE_METRICS}])
        adapter = ProviderAdapter(provider, model='glm-5.3-flash')
        result = fuse_five_modes(FusionRequest(inputs, max_repair_steps=0), generator=adapter, evaluator=adapter)
        self.assertEqual(result.idea['components'], ['core'])

    def test_recorded_tts_drafts_remain_rejected_with_actionable_feedback(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/tts_fusion_invalid_sources.json').read_text())
        self.assertEqual(len(fixture['drafts']), 5)
        for draft in fixture['drafts']:
            with self.subTest(draft=draft['cache_file']):
                provider = QueueProvider([draft['output']])
                adapter = ProviderAdapter(provider, model='glm-5.3-flash')
                with self.assertRaises(AdapterOutputError) as caught:
                    adapter.generate({'mode_inputs': fixture['mode_inputs']})
                self.assertTrue(caught.exception.validation_issues)
                self.assertEqual(caught.exception.previous_draft, draft['output'])


if __name__ == '__main__':
    unittest.main()
