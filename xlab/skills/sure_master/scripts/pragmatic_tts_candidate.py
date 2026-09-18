"""Evidence-backed free TTS ideas using the supplied execution contract."""
from dataclasses import asdict
import json
import os
from pathlib import Path

from research_idea_lib.inputs import IdeaRequest
from research_idea_lib.survey_repository import SurveyArtifactRepository
from research_idea_lib.providers.contracts import ProviderRequest
from research_idea_lib.providers.openai_compatible import OpenAICompatibleConfig, OpenAICompatibleProvider


def validate_tts_candidate(candidate, contract, evidence_ids):
    if not isinstance(candidate, dict):
        raise ValueError('TTS candidate must be an object')
    changes = candidate.get('change_set')
    if not isinstance(changes, list) or not changes:
        raise ValueError('TTS candidate requires concrete changes')
    domains = set()
    for change in changes:
        if not isinstance(change, dict) or change.get('domain') not in {'arch', 'train', 'inference'}:
            raise ValueError('Invalid TTS change domain')
        if any(not isinstance(change.get(key), str) or not change[key].strip() for key in ('target', 'description')):
            raise ValueError('TTS changes require target and description')
        domains.add(change['domain'])
    training = candidate.get('requires_training')
    if type(training) is not bool or training != bool(domains & {'arch', 'train'}):
        raise ValueError('TTS training declaration conflicts with changes')
    overrides = candidate.get('parameters', {})
    if not isinstance(overrides, dict) or set(overrides) - {'training', 'architecture', 'inference'}:
        raise ValueError('Invalid TTS parameter sections')
    allowed = contract.get('candidate_parameters', {})
    for section, domain in [('training', 'train'), ('architecture', 'arch'), ('inference', 'inference')]:
        values = overrides.get(section, {})
        if not isinstance(values, dict):
            raise ValueError('TTS parameter section must be an object')
        if values and domain not in domains:
            raise ValueError('TTS parameter changes must be declared')
        if section != 'architecture' and set(values) - set(allowed.get(section, [])):
            raise ValueError('TTS parameters exceed the execution contract')
    refs = candidate.get('evidence_ids')
    if not isinstance(refs, list) or not refs or any(not isinstance(x, str) or x not in evidence_ids for x in refs):
        raise ValueError('TTS candidate must cite supplied evidence')
    candidate['candidate_type'] = 'arch' if 'arch' in domains else 'fine_tune' if training else 'inference'
    return candidate


def generate_tts_candidate(payload, survey, branch_dir, runtime, accepted, rejected):
    repository = SurveyArtifactRepository.from_request(
        IdeaRequest(survey_path=survey), Path(__file__).resolve().parents[4])
    def relevance(item):
        text = json.dumps(item, ensure_ascii=False).lower()
        return sum(weight for term, weight in [('f5-tts', 12), ('flow matching', 8),
            ('text-to-speech', 6), ('diffusion', 3), ('speech', 2)] if term in text)
    selected = sorted(repository.evidence_items, key=relevance, reverse=True)[:24]
    evidence = [{key:item.get(key) for key in ('id', 'evidence_id', 'kind', 'title', 'paper_ids', 'paper_titles')}
                | {'summary':str(item.get('summary', ''))[:600], 'text':str(item.get('text', ''))[:1200]}
                for item in selected]
    ids = {str(item.get('evidence_id') or item.get('id') or '') for item in evidence} - {''}
    if not ids:
        raise ValueError('TTS survey has no usable evidence')
    context = {key: payload.get(key) for key in ('task_description', 'execution_contract',
               'current_best', 'prior_rounds', 'task_card', 'metric', 'base_model_profile')}
    context.update(survey_evidence=evidence, references=sorted(repository.references, key=relevance, reverse=True)[:16],
                   allowed_evidence_ids=sorted(ids), accepted_candidates=accepted,
                   rejected_attempts=rejected, research_mode='survey_analysis_direct')
    branch_dir.mkdir(parents=True, exist_ok=True)
    (branch_dir/'evidence_excerpts.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    prompt = (
        'Generate ONE distinct executable F5-TTS candidate for SURE using supplied survey evidence. '
        'Explore architecture, training and inference freely, including coordinated changes; no type quotas. '
        'Use the supplied execution_contract as the sole source of allowed capabilities and fixed settings. '
        'All weight or model source changes require complete training; inference-only uses the frozen parent. '
        'Do not change data splits, metric, fixed epoch budget, initialization, precision or rank count. '
        'Do not require deep MCTS, fusion or publication-level novelty; motivate an actual distinct experiment. '
        'Do not invent measured results, unsupported wrapper flags or auxiliary training runs. '
        'Return JSON with nonempty strings title, abstract, core_contribution, research_question, hypothesis, '
        'method, introduction; nonempty lists experiment_plan, data_requirements, baselines, metrics, risks, '
        'components, algorithm, reference_papers, evidence_ids. Every evidence_id must be supplied. '
        'Also return requires_training boolean, change_set (objects domain arch/train/inference, target, '
        'description), and parameters (training/architecture/inference objects; only allowed override keys). '
        'Explain precise implementation and any editable model source in method and algorithm. '
        'Use SURE_TASK_WRAPPER candidate/prepare_source interface from the contract. '
        'Plan only the existing full training/search evaluation; ablations can be future budgeted ideas. '
        'Do not repeat accepted experiments or disguise an unchanged configuration as a new idea.'
    )
    provider = OpenAICompatibleProvider(api_key=os.environ['OPENAI_API_KEY'], endpoint=runtime.chat_completions_url,
        config=OpenAICompatibleConfig(timeout_seconds=runtime.request_timeout_seconds,max_attempts=runtime.max_retries+1))
    response = provider.complete(ProviderRequest('xlab.sure.executable_tts.v1',runtime.generation_model,
        context,prompt,json.dumps(context,ensure_ascii=False),output_kind='json'))
    candidate = validate_tts_candidate(response.json_value, payload['execution_contract'], ids)
    candidate.update(source_modes=['survey_analysis_direct'], research_mode='survey_analysis_direct')
    path = branch_dir / 'advisory_candidate.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidate,ensure_ascii=False,indent=2))
    (branch_dir/'advisory_generation_trace.json').write_text(json.dumps(
        {'trace':response.trace.to_dict(),'usage':asdict(response.usage)},indent=2))
    return path
