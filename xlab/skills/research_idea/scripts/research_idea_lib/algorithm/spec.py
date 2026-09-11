"""Observable metadata contract for native research idea searches."""

from __future__ import annotations

from ..research_idea_spec import ALGORITHM_SPEC_VERSION
from .contracts import ALGORITHM_ID
from .tastes import IDEA_TASTE_MODES

SPEC_VERSION = ALGORITHM_SPEC_VERSION
IMPLEMENTATION_METADATA = {
    "algorithm": ALGORITHM_ID,
    "spec_version": SPEC_VERSION,
    "runtime": "package-native",
    "native_reconstruction": True,
    "modes": IDEA_TASTE_MODES,
}
