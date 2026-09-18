import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from research_idea_lib.resources.component_novelty import _core_nodes
from research_idea_lib.resources.embedding import ResourceExecutionError


class CoreProjectionTests(unittest.TestCase):
    def test_components_share_one_stable_core(self):
        descriptor=SimpleNamespace(resource_id='index',digest='sha256:fixture')
        records=[('a',{'node_id':'core:one','paper_title':'Paper','label':'Model','component':'encoder','summary':'Encoder fact'}),
                 ('b',{'node_id':'core:one','paper_title':'Paper','label':'Model','component':'decoder','summary':'Decoder fact'})]
        first=_core_nodes(records,descriptor)['core:one']
        second=_core_nodes(list(reversed(records)),descriptor)['core:one']
        self.assertEqual(first,second)
        self.assertIn('Encoder fact',first.summary)
        self.assertIn('Decoder fact',first.summary)
        self.assertEqual(json.loads(first.provenance_json)['component_ids'],['a','b'])

    def test_conflicting_paper_is_rejected(self):
        descriptor=SimpleNamespace(resource_id='index',digest='sha256:fixture')
        with self.assertRaisesRegex(ResourceExecutionError,'Conflicting'):
            _core_nodes([('a',{'node_id':'core:one','paper_title':'First'}),
                         ('b',{'node_id':'core:one','paper_title':'Other'})],descriptor)


if __name__=='__main__':
    unittest.main()
