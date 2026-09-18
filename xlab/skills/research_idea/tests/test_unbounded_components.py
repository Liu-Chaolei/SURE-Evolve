"""Offline regression coverage for complete, uncapped component inventories."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from research_idea_lib.algorithm.contracts import IdeaComponent, IdeaState
from research_idea_lib.algorithm.fusion import CANONICAL_MODES, REFEREE_METRICS, FusionRequest, fuse_five_modes


class UnboundedComponentsTest(unittest.TestCase):
    def test_state_keeps_eight_components_and_rejects_duplicates(self):
        components = tuple(IdeaComponent(f'component_{i}', f'Scientific mechanism {i}') for i in range(8))
        state = IdeaState('Title', 'Abstract', 'Contribution', 'Method', 'Risks', components)
        self.assertEqual(len(state.components), 8)
        with self.assertRaisesRegex(ValueError, 'unique'):
            IdeaState('Title', 'Abstract', 'Contribution', 'Method', 'Risks', (components[0], components[0]))

    def test_fusion_does_not_discard_components_above_five(self):
        names = [f'component_{i}' for i in range(8)]
        idea = dict(title='Title', abstract='Abstract', core_contribution='Contribution', method='Method', risks='Risks', components=names)
        inputs = [dict(mode=mode, idea=idea, evidence=['paper:1']) for mode in CANONICAL_MODES]

        class Generator:
            def generate(self, request):
                return dict(idea=idea, selected_components=[dict(component=name, source_mode=CANONICAL_MODES[0], evidence=['paper:1']) for name in names], rejected_components=[], conflict_resolutions=[])

        class Evaluator:
            def evaluate(self, candidate):
                return dict(metrics={name: 3.0 for name in REFEREE_METRICS})

        result = fuse_five_modes(FusionRequest(inputs, minimum_components=8, max_repair_steps=0), generator=Generator(), evaluator=Evaluator())
        self.assertEqual(result.idea['components'], names)


if __name__ == '__main__':
    unittest.main()
