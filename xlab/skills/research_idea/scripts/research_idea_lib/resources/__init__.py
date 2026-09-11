"""Research idea resource contracts and immutable model cache."""

from .component_novelty import (
    ComponentNoveltyRuntime,
    DeclaredFaissComponentRetriever,
    build_component_novelty_runtime,
)
from .ids import stable_evidence_id, stable_paper_id
from .manifest import (
    RESOURCE_MANIFEST_SCHEMA_VERSION,
    FileDigest,
    LoadedResourceBundle,
    ResourceDescriptor,
    ResourceManifest,
    ResourceRequirement,
    ResourceValidationError,
    descriptor_digest,
    load_resource_bundle,
    sha256_file,
    validate_resource_bundle,
)
from .model_cache import CachedModel, ModelCache, ModelCacheError, ModelFile, ModelSnapshot
from .resolver import ResourceResolution, resolve_survey_resources, survey_artifact_ids

__all__ = [
    "RESOURCE_MANIFEST_SCHEMA_VERSION",
    "CachedModel",
    "ComponentNoveltyRuntime",
    "DeclaredFaissComponentRetriever",
    "FileDigest",
    "LoadedResourceBundle",
    "ModelCache",
    "ModelCacheError",
    "ModelFile",
    "ModelSnapshot",
    "ResourceDescriptor",
    "ResourceManifest",
    "ResourceRequirement",
    "ResourceResolution",
    "ResourceValidationError",
    "build_component_novelty_runtime",
    "descriptor_digest",
    "load_resource_bundle",
    "resolve_survey_resources",
    "sha256_file",
    "stable_evidence_id",
    "stable_paper_id",
    "survey_artifact_ids",
    "validate_resource_bundle",
]
