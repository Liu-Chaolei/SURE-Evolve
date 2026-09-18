import unittest

from playground.sure_master.tools.sd_role_models import operation_model, GLM, OPENAI


class RoleModelTests(unittest.TestCase):
    def test_evaluation_and_fusion_are_separate(self):
        for role in ('idea.evaluate', 'idea.diagnostic', 'component_novelty.evaluate'):
            self.assertEqual(operation_model(f'xlab.research_idea.{role}.v1'), GLM)
        for role in ('idea.generate', 'analysis.generate', 'analysis.replan',
                     'fusion.generate', 'fusion.referee', 'fusion.repair'):
            self.assertEqual(operation_model(f'xlab.research_idea.{role}.v1'), OPENAI)

    def test_unknown_role_is_not_silently_routed(self):
        with self.assertRaises(ValueError):
            operation_model('unknown')
