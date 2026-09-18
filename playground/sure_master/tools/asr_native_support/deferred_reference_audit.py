"""Downgrade only attested missing-bibliography errors; retain every other check."""
from copy import deepcopy


def deferred_audit_report(report, idea, idea_result):
    result = deepcopy(report)
    context = idea.get('source_context', {})
    references = {r['paper_id'] for r in context.get('references', []) if isinstance(r,dict) and r.get('paper_id')}
    evidence = {p for e in idea.get('source_evidence', []) for p in e.get('paper_ids', [])}
    unresolved = sorted(evidence - references)
    declaration = idea_result.get('reference_validation', {})
    if not unresolved or declaration.get('status') != 'deferred':
        return result
    if set(unresolved) != set(declaration.get('unresolved_ids', [])):
        return result
    expected = "Unresolved source_evidence paper ids: " + ', '.join(unresolved) + '.'
    if expected not in result.get('blocking_errors', []):
        return result
    for key in ['blockers','blocking_errors']:
        result[key] = [message for message in result.get(key, []) if message != expected]
    warning = 'USER-DEFERRED bibliography lookup (not verified citations): ' + ', '.join(unresolved)
    result['warnings'] = sorted(set([*result.get('warnings', []), warning]))
    result['checks']['source_evidence_resolves'] = False
    result['checks']['bibliography_resolution_deferred'] = True
    result['passed'] = not result['blocking_errors']
    result['status'] = 'success' if result['passed'] else 'incomplete'
    return result
