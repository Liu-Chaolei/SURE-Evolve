"""Offline regressions for ASR's native reliability amendments."""
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from dataclasses import dataclass
from playground.sure_master.tools.asr_native_reliability import project_empty_ablation
from playground.sure_master.tools.asr_native_support.materialization_validation import align_public_materialization
from playground.sure_master.tools.asr_native_support.deferred_reference_audit import deferred_audit_report

@dataclass
class Result:
    json_value: dict
    text: str = 'retained raw response'

class NativeV2Tests(unittest.TestCase):
    def test_completed_handoff_never_pauses_old_and_waits_for_evaluation(self):
        from playground.sure_master.tools import asr_deployment_ops as ops
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / 'old'
            old.mkdir()
            (root / 'deployment.json').write_text(json.dumps({'name':'ASR-Native-MCTS-v2','old_run':str(old)}))
            (root / 'ideas_ready.json').write_text(json.dumps({'status':'ideas_ready','candidate_count':4}))
            state = old / 'workflow_state.json'
            state.write_text(json.dumps({'status':'running','pid':123}))
            with patch.object(ops, 'owned_running', return_value=True), \
                 patch.object(ops, 'process_identity', return_value=None), \
                 patch.object(ops, 'pause') as pause, \
                 patch.object(ops, 'register') as register, \
                 patch.object(ops, 'launch', return_value=456) as launch:
                with self.assertRaises(ops.CheckpointNotReady):
                    ops.handoff(root)
                launch.assert_not_called()
                state.write_text(json.dumps({'status':'completed','pid':123}))
                self.assertEqual(ops.handoff(root), 456)
                pause.assert_not_called()
                register.assert_any_call(root, 'ASR-Native-MCTS-v2', 'starting')
                self.assertTrue((root / 'TRAINING_ENABLED').exists())

    def test_summary_projection_preserves_raw_response(self):
        result = Result({'summary':'s','insight':'i','ablation':[]})
        projected = project_empty_ablation(result, {'summary','insight'})
        self.assertEqual(projected.json_value, {'summary':'s','insight':'i'})
        self.assertEqual(result.json_value['ablation'], [])
        self.assertEqual(projected.text, result.text)
        bad = Result({'summary':'s','ablation':['extra experiment']})
        self.assertIs(project_empty_ablation(bad, {'summary'}), bad)
        missing = Result({'ablation':[]})
        self.assertIs(project_empty_ablation(missing, {'summary'}), missing)

    def test_only_known_paper_ids_can_be_deferred(self):
        base = dict(title='title', core_contribution='c', hypothesis='h', method='m',
                    root_domains=['ASR'], components=['c'], risks=['r'])
        fusion = {'idea':base, 'evidence_ids':['e1']}
        evidence = [{'evidence_id':'e1','paper_ids':['known-paper']}]
        idea = dict(base, reference_papers=['known-paper'])
        align_public_materialization(idea, fusion, evidence, [])
        self.assertEqual(idea['reference_validation']['status'], 'deferred')
        self.assertEqual(idea['reference_papers'], ['known-paper'])
        with self.assertRaises(ValueError):
            align_public_materialization(dict(base, reference_papers=['Invented title']), fusion, evidence, [])
        with self.assertRaises(ValueError):
            align_public_materialization(dict(base), fusion, [], [])
        with self.assertRaises(ValueError):
            align_public_materialization(dict(base, method='changed'), fusion, evidence, [])

    def test_deferred_audit_keeps_unrelated_errors(self):
        warning = 'Unresolved source_evidence paper ids: p.'
        report = {'blocking_errors':[warning, 'scientific drift'], 'blockers':[warning, 'scientific drift'], 'checks':{}}
        idea = {'source_context':{'references':[]}, 'source_evidence':[{'paper_ids':['p']}]}
        result = deferred_audit_report(report, idea, {'reference_validation':{'status':'deferred','unresolved_ids':['p']}})
        self.assertEqual(result['blocking_errors'], ['scientific drift'])
        self.assertFalse(result['passed'])
        self.assertFalse(result['checks']['source_evidence_resolves'])

if __name__ == '__main__':
    unittest.main()
