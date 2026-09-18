from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from research_idea_lib.algorithm.keynote_pipeline import _score_output, _exact_output, KeynotePipelineError


class KeynoteProjectionTests(unittest.TestCase):
    def test_glm_annotations_do_not_relax_other_provider_contracts(self):
        value={'score':'35','rationale':'Limited transfer to the task.'}
        result=SimpleNamespace(json_value=value,trace=SimpleNamespace(model='glm-5.3-flash'))
        self.assertEqual(_score_output(result),35)
        self.assertEqual(value,{'score':'35','rationale':'Limited transfer to the task.'})
        with self.assertRaises(KeynotePipelineError):
            _score_output(SimpleNamespace(json_value=value,trace=SimpleNamespace(model='other-provider')))
        with self.assertRaises(KeynotePipelineError):
            _score_output(SimpleNamespace(json_value={'score':'35.5'},trace=result.trace))
        explanation={'answer':'Relevant motivation, but it requires major adaptation.','score':38}
        self.assertEqual(_score_output(SimpleNamespace(json_value=explanation,trace=result.trace)),38)
        self.assertEqual(explanation['score'],38)
        self.assertEqual(_score_output(SimpleNamespace(json_value={'answer':{'score':35,'confidence':0.7}},trace=result.trace)),35)
        with self.assertRaises(KeynotePipelineError):
            _score_output(SimpleNamespace(json_value={'answer':'30','score':35},trace=result.trace))
        with self.assertRaises(KeynotePipelineError):
            _score_output(SimpleNamespace(json_value={'answer':{'confidence':0.7}},trace=result.trace))

    def test_actual_glm_answer_score_preserves_value_and_raw_response(self):
        raw={'answer':'45'};original=deepcopy(raw)
        self.assertEqual(_score_output(SimpleNamespace(json_value=raw)),45)
        self.assertEqual(raw,original)
        self.assertEqual(_score_output(SimpleNamespace(json_value={'score':30})),30)
        redundant={'answer':'15','score':15}
        self.assertEqual(_score_output(SimpleNamespace(json_value=redundant)),15)
        self.assertEqual(redundant,{'answer':'15','score':15})
        annotated={'answer':{'score':35,'rationale':'Only weakly related to this task.'}}
        original=deepcopy(annotated)
        self.assertEqual(_score_output(SimpleNamespace(json_value=annotated)),35)
        self.assertEqual(annotated,original)

    def test_score_range_and_type_stay_strict(self):
        for value in ['101','-1','45.5','true','forty-five',None,True,[],{'unexpected':45}]:
            with self.subTest(value=value),self.assertRaises(KeynotePipelineError):
                _score_output(SimpleNamespace(json_value={'answer':value}))
        with self.assertRaises(KeynotePipelineError):
            _score_output(SimpleNamespace(json_value={'score':'45'}))
        with self.assertRaises(KeynotePipelineError):
            _score_output(SimpleNamespace(json_value={'score':None,'answer':'45'}))
        with self.assertRaises(KeynotePipelineError):
            _score_output(SimpleNamespace(json_value={'score':16,'answer':'15'}))
        with self.assertRaises(KeynotePipelineError):
            _score_output(SimpleNamespace(json_value={'score':15,'unrelated':'15'}))
        with self.assertRaises(KeynotePipelineError):
            _score_output(SimpleNamespace(json_value={'score':15,'rationale':{'score':20}}))

    def test_summary_wrappers_require_exact_schema(self):
        raw={'answer':{'summary':'A factual summary.','insight':'A mechanism insight.'}}
        original=deepcopy(raw)
        self.assertEqual(_exact_output(SimpleNamespace(json_value=raw),{'summary','insight'}),raw['answer'])
        self.assertEqual(raw,original)
        self.assertEqual(_exact_output(SimpleNamespace(json_value={'answer':'Summary'}),{'summary'}),{'summary':'Summary'})
        with self.assertRaises(KeynotePipelineError):
            _exact_output(SimpleNamespace(json_value={'answer':{'summary':'Only summary'}}),{'summary','insight'})

    def test_glm_json_encoded_summary_preserves_both_fields(self):
        content={'summary':'Noise-robust adaptation uses a bottleneck.','insight':'Keep the clean text-to-acoustic mapping.'}
        wire={'answer':json.dumps(content)}
        result=SimpleNamespace(json_value=wire,trace=SimpleNamespace(model='glm-5.3-flash'))
        self.assertEqual(_exact_output(result,{'summary','insight'}),content)
        self.assertEqual(wire,{'answer':json.dumps(content)})
        with self.assertRaises(KeynotePipelineError):
            _exact_output(SimpleNamespace(json_value={'answer':'{"summary":"missing insight"}'},trace=result.trace),{'summary','insight'})

    def test_glm_partial_summary_alias_preserves_the_explicit_insight(self):
        raw={'answer':'WavLM jointly learns masked prediction and denoising.',
             'insight':'Corruption during training may improve robustness.'}
        original=deepcopy(raw)
        result=SimpleNamespace(json_value=raw,trace=SimpleNamespace(model='glm-5.3-flash'))
        self.assertEqual(_exact_output(result,{'summary','insight'}),
                         {'summary':raw['answer'],'insight':raw['insight']})
        self.assertEqual(raw,original)
        with self.assertRaises(KeynotePipelineError):
            _exact_output(SimpleNamespace(json_value=raw,trace=SimpleNamespace(model='other-provider')),{'summary','insight'})
        for bad in [{'answer':'Only a summary'}, {'answer':{'summary':'Nested'},'insight':'Explicit'},
                    {'answer':'Summary','insight':None}, {'summary':'Explicit summary','answer':'Ambiguous prose'}]:
            with self.subTest(bad=bad),self.assertRaises(KeynotePipelineError):
                _exact_output(SimpleNamespace(json_value=bad,trace=result.trace),{'summary','insight'})

    def test_glm_score_after_explanatory_object_is_read_without_guessing(self):
        first={'answer':'Paper on video dubbing has only marginal relevance to F5-TTS.'}
        raw=json.dumps(first)+'\n'+json.dumps({'score':12})
        result=SimpleNamespace(json_value=deepcopy(first),text=raw,trace=SimpleNamespace(model='glm-5.3-flash'))
        self.assertEqual(_score_output(result),12)
        self.assertEqual(result.json_value,first)
        self.assertEqual(result.text,raw)
        with self.assertRaises(KeynotePipelineError):
            _score_output(SimpleNamespace(json_value=first,text=raw,trace=SimpleNamespace(model='other-provider')))

    def test_conflicting_or_incomplete_json_sequences_are_not_combined(self):
        for tail in ['{"score":13}', '{"score":true}']:
            with self.subTest(tail=tail),self.assertRaisesRegex(KeynotePipelineError,'conflicting score'):
                _score_output(SimpleNamespace(json_value={'score':12},text='{"score":12}\n'+tail,
                    trace=SimpleNamespace(model='glm-5.3-flash')))
        for text in ['{"answer":"Explanation"}\n{"score":12',
                     '{"answer":"Explanation"}\nnot JSON\n{"score":12}']:
            with self.subTest(text=text),self.assertRaises(KeynotePipelineError):
                _score_output(SimpleNamespace(json_value={'answer':'Explanation'},text=text,
                    trace=SimpleNamespace(model='glm-5.3-flash')))


if __name__=='__main__':unittest.main()
