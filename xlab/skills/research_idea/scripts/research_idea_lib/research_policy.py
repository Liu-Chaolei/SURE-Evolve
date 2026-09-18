"""Explicit policy for SURE's native, experiment-free research phase."""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Mapping

from .providers.contracts import ProviderRequest

PROFILE = "xlab.sure.native.v1"


@dataclass(frozen=True)
class ResearchPolicy:
    profile: str = PROFILE
    evidence_mode: str = "local_literature"
    feedback_kind: str = "candidate_outcome"
    refinement: str = "current_best"
    allow_auxiliary_experiments: bool = False

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "ResearchPolicy":
        result = cls(**dict(value))
        if (result.profile != PROFILE or result.evidence_mode not in {"local_literature", "task_only"}
                or result.feedback_kind != "candidate_outcome" or result.refinement != "current_best"
                or result.allow_auxiliary_experiments is not False):
            raise ValueError("Unsupported native SURE research policy")
        return result

    def prompt(self) -> str:
        text = (
            "Refine only the supplied current best method, within its functional slots and execution contract. "
            "Preserve its overall method, root domains and tags. Task context and accepted/rejected attempts "
            "are authoritative constraints throughout analysis, search, evaluation and fusion. "
            "Candidate outcomes describe complete interventions under recorded conditions, NOT component "
            "removal evidence. Do not attribute a joint intervention's outcome to one component. "
            "Execution failures do not refute a mechanism. Never invent measured results or significance. "
            "No auxiliary experiments or component ablations are allowed. The validation plan is exactly "
            "one normal SURE candidate execution under the supplied budget. "
            "Follow the current operation's exact output schema; never add policy-specific fields. "
            "Only when that schema requests an ablation field, return an empty list for it. "
            "Search remove/replace operations describe method edits, not additional experiment jobs. "
        )
        if self.evidence_mode == "task_only":
            text += (
                "External literature is unavailable. Use only supplied task/source/experiment records and "
                "model reasoning; mark reasoning as unverified. These records are not papers. "
                "Do not invent citations. Only when the operation schema requests reference_papers, "
                "return an empty list for it. Internal evidence IDs can support "
                "feasibility and provenance, but do not establish literature novelty."
            )
        return text


class PolicyProvider:
    """Apply the same research-only boundary to every native provider operation."""

    def __init__(self, provider, policy: ResearchPolicy, context: Mapping[str, Any]):
        self.provider = provider
        self.policy = policy
        self.context = context

    def complete(self, request: ProviderRequest):
        return self.provider.complete(replace(
            request,
            system_prompt=request.system_prompt + "\nSURE policy overrides conflicting template defaults:\n" + self.policy.prompt(),
            user_prompt=request.user_prompt + "\nAuthoritative SURE task context:\n" + json.dumps(self.context, ensure_ascii=False, sort_keys=True),
        ))
