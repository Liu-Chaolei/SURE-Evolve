from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

_STRUCTURAL_TEXT_FIELDS = ("title", "core_contribution", "hypothesis", "method")
_STRUCTURAL_LIST_FIELDS = ("root_domains", "components")


def materialization_contract(fusion_result, evidence_registry, references):
    """Supply exact immutable values as input instructions, never repair output."""
    idea = deepcopy(dict(fusion_result['idea']))
    if isinstance(idea.get('risks'), str):
        idea['risks'] = [idea['risks'].strip()]
    idea['reference_papers'] = []
    align_public_materialization(idea, fusion_result, evidence_registry, references)
    return {key:idea[key] for key in (*_STRUCTURAL_TEXT_FIELDS, *_STRUCTURAL_LIST_FIELDS,
                                     'risks','evidence_ids','reference_ids','reference_papers')}


def align_public_materialization(
    idea_result: dict[str, Any],
    fusion_result: Mapping[str, Any],
    evidence_registry: Sequence[Any],
    references: Sequence[Any],
) -> list[dict[str, Any]]:
    """Validate and project public structure from the persisted fused result."""
    fused_idea = fusion_result.get("idea")
    if not isinstance(fused_idea, Mapping):
        raise ValueError("persisted fusion result must contain a structured idea.")

    validate_materialization_identity(idea_result, fusion_result)
    fused_risks = fused_idea.get("risks")
    expected_risks = deepcopy(fused_risks) if isinstance(fused_risks, list) else [fused_risks.strip()]
    evidence_ids = _unique_strings(fusion_result.get("evidence_ids"), "fusion result evidence_ids")
    registry = _evidence_by_id(evidence_registry)
    unknown = [evidence_id for evidence_id in evidence_ids if evidence_id not in registry]
    if unknown:
        raise ValueError(f"persisted fusion result contains unknown evidence identities: {unknown!r}.")
    fused_evidence = [registry[evidence_id] for evidence_id in evidence_ids]
    reference_by_id, reference_order = _reference_registry(attributed_references(references, evidence_registry))
    reference_ids = list(dict.fromkeys(paper_id for item in fused_evidence
        for paper_id in _unique_strings(item.get("paper_ids"), "evidence registry paper_ids")))
    unresolved = [paper_id for paper_id in reference_ids if paper_id not in reference_by_id]
    if unresolved:
        raise ValueError(f"fused evidence contains unresolved reference identities: {unresolved!r}.")
    reference_ids = [paper_id for paper_id in reference_order if paper_id in reference_ids]
    supplied_reference_ids = idea_result.get("reference_ids")
    if supplied_reference_ids is not None and _unique_strings(supplied_reference_ids, "idea_result reference_ids") != reference_ids:
        raise ValueError("idea_result reference identities drifted from the fused evidence registry.")
    _validate_supplied_references(idea_result.get("reference_papers"),
                                 {paper_id:reference_by_id[paper_id] for paper_id in reference_ids})
    idea_result.update({**{field:deepcopy(fused_idea[field]) for field in (*_STRUCTURAL_TEXT_FIELDS,*_STRUCTURAL_LIST_FIELDS)},
                       "risks":expected_risks,"evidence_ids":evidence_ids,"reference_ids":reference_ids,
                       "reference_papers":[reference_by_id[paper_id] for paper_id in reference_ids]})
    return [deepcopy(item) for item in fused_evidence]


def validate_materialization_identity(idea_result: Mapping[str, Any], fusion_result: Mapping[str, Any]) -> None:
    """The same immutable-field contract before fallback and before publication."""
    fused_idea = fusion_result.get("idea")
    if not isinstance(fused_idea, Mapping):
        raise ValueError("persisted fusion result must contain a structured idea.")

    for field in _STRUCTURAL_TEXT_FIELDS:
        expected = _required_text(fused_idea.get(field), f"fused idea {field}")
        actual = _required_text(idea_result.get(field), f"idea_result {field}")
        if actual != expected:
            raise ValueError(f"idea_result {field} drifted from the persisted fused idea.")

    for field in _STRUCTURAL_LIST_FIELDS:
        expected = _required_json_list(
            fused_idea.get(field),
            f"fused idea {field}",
            nonempty=field == "components",
        )
        actual = _required_json_list(
            idea_result.get(field),
            f"idea_result {field}",
            nonempty=field == "components",
        )
        if actual != expected:
            raise ValueError(f"idea_result {field} drifted from the persisted fused idea.")

    fused_risks = fused_idea.get("risks")
    expected_risks = (
        _required_list(fused_risks, "fused idea risks")
        if isinstance(fused_risks, list)
        else [_required_text(fused_risks, "fused idea risks")]
    )
    actual_risks = _required_list(idea_result.get("risks"), "idea_result risks")
    if actual_risks != expected_risks:
        raise ValueError("idea_result risks drifted from the persisted fused idea.")

    evidence_ids = _unique_strings(fusion_result.get("evidence_ids"), "fusion result evidence_ids")
    if not evidence_ids:
        raise ValueError("persisted fusion result must contain evidence identities.")
    supplied_evidence_ids = idea_result.get("evidence_ids")
    if supplied_evidence_ids is not None and _unique_strings(
        supplied_evidence_ids, "idea_result evidence_ids"
    ) != evidence_ids:
        raise ValueError("idea_result evidence identities drifted from the persisted fusion result.")

def attributed_references(references, evidence_registry):
    """Extend citations only from exact, resource-attributed graph neighbors."""
    registry, _ = _reference_registry(references)
    result = deepcopy(list(references))
    added = {}
    for item in evidence_registry:
        provenance = item.get('provenance', {})
        resource = provenance.get('resource', {})
        if item.get('kind') != 'graph_neighbor' or resource.get('role') != 'graph':
            continue
        paper_id = resource.get('neighbor_paper_id')
        if not paper_id or item.get('paper_ids') != [paper_id] or not resource.get('descriptor_digest'):
            raise ValueError('Graph citation lacks exact resource attribution')
        title = _required_text(provenance.get('title'), 'graph citation title')
        if paper_id in added and added[paper_id] != title:
            raise ValueError('Graph citation title conflicts for the same paper identity')
        if paper_id not in registry:
            added[paper_id] = title
            registry[paper_id] = title
            result.append({'paper_id':paper_id,'title':title,'provenance':deepcopy(provenance)})
    return result


def _evidence_by_id(values: Sequence[Any]) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for raw in values:
        if not isinstance(raw, Mapping):
            raise ValueError("evidence registry must contain only mappings.")
        evidence_id = _required_text(raw.get("evidence_id"), "evidence registry evidence_id")
        if evidence_id in registry:
            raise ValueError(f"evidence registry contains duplicate identity {evidence_id!r}.")
        registry[evidence_id] = deepcopy(dict(raw))
    return registry


def _reference_registry(values: Sequence[Any]) -> tuple[dict[str, str], list[str]]:
    registry: dict[str, str] = {}
    order: list[str] = []
    for raw in values:
        if not isinstance(raw, Mapping):
            raise ValueError("source context references must contain only mappings.")
        paper_id = _required_text(raw.get("paper_id"), "source reference paper_id")
        title = _required_text(raw.get("title"), f"source reference {paper_id!r} title")
        if paper_id in registry:
            raise ValueError(f"source context contains duplicate reference identity {paper_id!r}.")
        registry[paper_id] = title
        order.append(paper_id)
    return registry, order


def _validate_supplied_references(value: Any, reference_by_id: Mapping[str, str]) -> None:
    supplied = _unique_strings(value, "reference_papers")
    supported = {paper_id.casefold() for paper_id in reference_by_id}
    supported.update(title.casefold() for title in reference_by_id.values())
    fabricated = [reference for reference in supplied if reference.casefold() not in supported]
    if fabricated:
        raise ValueError(
            f"research idea idea_result contains reference_papers without attributed evidence: {fabricated!r}."
        )


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text.")
    return value.strip()


def _required_list(value: Any, label: str) -> list[Any]:
    return _required_json_list(value, label, nonempty=True)


def _required_json_list(value: Any, label: str, *, nonempty: bool) -> list[Any]:
    if not isinstance(value, list) or (nonempty and not value):
        requirement = "a non-empty list" if nonempty else "a list"
        raise ValueError(f"{label} must be {requirement}.")
    return deepcopy(value)


def _unique_strings(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must contain only text entries.")
    result = [item.strip() for item in value]
    if any(not item for item in result) or len(result) != len(set(result)):
        raise ValueError(f"{label} must contain unique non-empty text entries.")
    return result
