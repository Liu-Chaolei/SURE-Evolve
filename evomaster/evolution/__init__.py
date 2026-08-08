"""Post-run self-evolution utilities for EvoMaster.

The evolution package is intentionally optional. Normal EvoMaster runs do not
import or execute it unless the user runs the evolution wrapper or passes
``--evolve`` to ``run.py``.
"""

from .manager import EvolutionManager, EvolutionRunConfig
from .models import (
    EvolutionCandidates,
    EvolutionOverlay,
    PromptPatchCandidate,
    SkillCandidate,
    ToolProposal,
    TraceDigest,
)

__all__ = [
    "EvolutionManager",
    "EvolutionRunConfig",
    "TraceDigest",
    "SkillCandidate",
    "PromptPatchCandidate",
    "ToolProposal",
    "EvolutionCandidates",
    "EvolutionOverlay",
]
