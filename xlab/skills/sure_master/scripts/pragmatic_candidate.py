"""Generate executable ASR ideas from validated survey evidence without deep-search gates."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path

from research_idea_lib.inputs import IdeaRequest
from research_idea_lib.survey_repository import SurveyArtifactRepository
from research_idea_lib.providers.contracts import ProviderRequest
from research_idea_lib.providers.openai_compatible import OpenAICompatibleConfig, OpenAICompatibleProvider
from pragmatic_tts_candidate import generate_tts_candidate

BASE_ARCH = {
    'num-encoder-layers': '2,2,3,4,3,2',
    'encoder-dim': '192,256,384,512,384,256',
    'feedforward-dim': '512,768,1024,1536,1024,768',
    'encoder-unmasked-dim': '192,192,256,256,256,192',
}


def validate_architecture(arguments, accepted):
    if not isinstance(arguments, dict) or set(arguments) != set(BASE_ARCH):
        raise ValueError('Provide exactly the four supported architecture arguments')
    values = {}
    for key, raw in arguments.items():
        if not isinstance(raw, str):
            raise ValueError('Architecture argument values must be comma-separated strings')
        parts = [int(x.strip()) for x in raw.split(',')]
        minimum = 0 if key == 'encoder-unmasked-dim' else 1
        if len(parts) != 6 or min(parts) < minimum:
            raise ValueError('Architecture requires six valid positive stage values')
        if key != 'num-encoder-layers' and any(x % 32 for x in parts):
            raise ValueError('Stage dimensions must be multiples of 32')
        values[key] = parts
    if any(a > b for a,b in zip(values['encoder-unmasked-dim'],values['encoder-dim'])):
        raise ValueError('Unmasked width exceeds encoder width')
    if sum(values['num-encoder-layers']) > 20 or max(values['encoder-dim']) > 512 or max(values['feedforward-dim']) > 1536:
        raise ValueError('Candidate exceeds the conservative first-round architecture envelope')
    normalized = {k:','.join(map(str,v)) for k,v in values.items()}
    if normalized == BASE_ARCH:
        raise ValueError('Candidate repeats the baseline architecture')
    if any(item.get('architecture_arguments') == normalized for item in accepted):
        raise ValueError('Duplicate architecture configuration')
    return normalized


def generate_candidate(payload, survey, branch_dir, runtime, accepted, rejected):
    if payload.get('task_id') == 'tts_zh_cer':
        return generate_tts_candidate(payload, survey, branch_dir, runtime, accepted, rejected)
    if payload.get('task_id') != 'asr_en_wer':
        raise ValueError('Pragmatic generation currently supports this ASR architecture task only')
    repository = SurveyArtifactRepository.from_request(IdeaRequest(survey_path=survey),Path(__file__).resolve().parents[4])
    evidence = repository.evidence_items[:24]
    allowed_ids = {str(item.get('evidence_id') or item.get('id') or '') for item in evidence}
    allowed_ids.discard('')
    if not allowed_ids:
        raise ValueError('Validated survey has no usable evidence IDs')
    def relevance(item):
        title = str(item.get('title') or '').lower()
        return sum(weight for term,weight in [('zipformer',10),('transducer',5),('conformer',4),('speech',2)] if term in title)
    references = sorted(repository.references,key=relevance,reverse=True)[:16]
    analysis = None
    for path in sorted(branch_dir.parent.rglob('provider_cache/*.json')):
        result = json.loads(path.read_text()).get('result',{})
        if result.get('trace',{}).get('operation') == 'xlab.research_idea.analysis.generate.v1':
            analysis = result.get('json_value')
            break
    context = {'task':payload['task_description'],'execution_contract':payload.get('execution_contract',{}),
               'current_best':payload.get('current_best',{}),'baseline_architecture':BASE_ARCH,
               'prior_rounds':payload.get('prior_rounds',payload.get('history',[])),
               'accepted_candidates':accepted,'rejected_attempts':rejected,
               'survey_evidence':evidence,'allowed_evidence_ids':sorted(allowed_ids),
               'references':references,'available_analysis':analysis,
               'research_mode':'survey_analysis_direct'}
    provider = OpenAICompatibleProvider(api_key=os.environ['OPENAI_API_KEY'],endpoint=runtime.chat_completions_url,
        config=OpenAICompatibleConfig(timeout_seconds=runtime.request_timeout_seconds,max_attempts=runtime.max_retries+1))
    prompt = (
        'Generate ONE executable Zipformer architecture candidate for SURE. Use supplied survey evidence and '
        'existing analysis. Prioritize a runnable, distinct architecture experiment, not publication-level novelty. '
        'Change ONLY the four supported architecture arguments; keep training, data, FP32, duration900, seed42, '
        '30 epochs, MUSAN, tokenizer and decoding fixed. Never invent measured results. '
        'Return all four architecture_arguments as comma-separated SIX-stage integer strings. Dimensions are '
        'multiples of32, encoder-unmasked-dim must not exceed encoder-dim. Keep sum(layers)<=20, '
        'max encoder width<=512 and max FFN width<=1536. Redistribute existing capacity conservatively. '
        'Distinct directions may redistribute depth, FFN capacity, unmasked bottlenecks or stage widths. '
        'Do not repeat accepted configurations or the unchanged baseline. '
        'Return JSON with nonempty string fields title, abstract, core_contribution, research_question, hypothesis, '
        'method, introduction; nonempty lists experiment_plan, data_requirements, baselines, metrics, risks, '
        'components, algorithm, reference_papers, evidence_ids; and architecture_arguments. '
        'Every evidence_id MUST come from allowed_evidence_ids. Explain the exact argument changes in method '
        'and algorithm. experiment_plan is only the fixed full training and regular-set WER comparison; '
        'do not schedule extra ablation runs. Components have names and descriptions. '
        'Use evidence as motivation; do not claim it proves the proposed candidate improves WER.'
    )
    response = provider.complete(ProviderRequest('xlab.sure.executable_architecture.v1',runtime.generation_model,
        context,prompt,json.dumps(context,ensure_ascii=False),output_kind='json'))
    candidate = dict(response.json_value or {})
    candidate['architecture_arguments'] = validate_architecture(candidate.get('architecture_arguments'),accepted)
    if not set(candidate.get('evidence_ids',[])).issubset(allowed_ids):
        raise ValueError('Candidate cites evidence outside the supplied survey')
    candidate.update(candidate_type='arch',source_modes=['survey_analysis_direct'],
        research_mode='survey_analysis_direct',research_warnings=[
            'Deep MCTS/fusion/novelty completion is advisory in this user-authorized flow-first mode.'])
    path=branch_dir/'advisory_candidate.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(candidate,indent=2,ensure_ascii=False))
    (branch_dir/'advisory_generation_trace.json').write_text(json.dumps(
        {'trace':response.trace.to_dict(),'usage':asdict(response.usage),'reused_analysis':analysis is not None},indent=2))
    return path
