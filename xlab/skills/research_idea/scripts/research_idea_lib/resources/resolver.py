"""Resolve explicit, survey-linked resources for research idea generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from .manifest import (
    LoadedResourceBundle,
    ResourceDescriptor,
    ResourceRequirement,
    ResourceValidationError,
    load_resource_bundle,
    validate_resource_bundle,
)
from .model_cache import ModelCache, ModelCacheError, ModelFile, ModelSnapshot

_RESOURCE_ARTIFACT_TYPES = {"research_idea_resource_manifest", "resource_manifest", "resource_bundle_manifest"}
_SURVEY_ARTIFACT_TYPES = {"literature_survey_json", "survey_json"}
_EXPECTED_SURVEY_SCHEMA = "xlab.literature_survey.v1"
_EXPECTED_SURVEY_ALGORITHM = "survey-agent.v1"
_EXPECTED_GRAPH_SCHEMA = "xlab.paper_graph.v1"
_EXPECTED_GRAPH_ALGORITHM = "sqlite-paper-graph.v1"
_EXPECTED_COMPONENT_SCHEMA = "xlab.component_index.v1"
_EXPECTED_COMPONENT_ALGORITHM = "faiss-flat-ip.v1"
_EXPECTED_KEYNOTES_SCHEMA = "xlab.paper_keynotes.v1"
_EXPECTED_KEYNOTES_ALGORITHM = "keynote-cache.v1"
_EXPECTED_MODEL_SCHEMA = "xlab.model_snapshot.v1"
_EXPECTED_MODEL_ALGORITHM = "sentence-transformers.v1"


@dataclass
class ResourceResolution:
    manifest_path: Path | None = None
    manifest_logical_path: str | None = None
    bundle: LoadedResourceBundle | None = None
    graph_db_path: Path | None = None
    component_index_dir: Path | None = None
    outcome_model_path: Path | None = None
    component_model_path: Path | None = None
    keynote_cache_path: Path | None = None
    model_uris: dict[str, str] = field(default_factory=dict)
    direct_parent_artifact_ids: tuple[str, ...] = ()
    blocking_errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.bundle is not None and not self.blocking_errors

    def portable_manifest(self) -> dict[str, Any] | None:
        if self.bundle is None or self.manifest_path is None:
            return None
        return {
            "bundle_id": self.bundle.manifest.bundle_id,
            "schema_version": self.bundle.manifest.schema_version,
            "digest": _sha256_file(self.manifest_path),
            "logical_path": self.manifest_logical_path,
        }

    def portable_resources(self) -> dict[str, dict[str, Any]]:
        if self.bundle is None:
            return {}
        resources: dict[str, dict[str, Any]] = {}
        for descriptor in self.bundle.manifest.resources:
            role = _model_role(descriptor, self.bundle.manifest.models) if descriptor.kind == "model" else descriptor.kind
            logical_uri = self.model_uris.get(role) if descriptor.kind == "model" else _resource_uri(
                self.bundle.manifest.bundle_id, descriptor.resource_id
            )
            resources[role] = {
                "resource_id": descriptor.resource_id,
                "kind": descriptor.kind,
                "schema_version": descriptor.schema_version,
                "algorithm_version": descriptor.algorithm_version,
                "digest": descriptor.digest,
                "capabilities": list(descriptor.capabilities),
                "dimensions": dict(descriptor.dimensions),
                "files": [
                    {"logical_path": file.path, "sha256": file.sha256, "size": file.size}
                    for file in descriptor.files
                ],
                "logical_uri": logical_uri,
            }
        return resources


def resolve_survey_resources(
    *,
    survey_manifest_path: Path | None,
    survey_manifest: dict[str, Any],
    survey: dict[str, Any],
    survey_json_path: Path | None,
    cwd: Path,
    model_cache_root: Path | None = None,
) -> ResourceResolution:
    """Resolve one explicit bundle declaration without searching neighboring locations."""

    result = ResourceResolution()
    result.direct_parent_artifact_ids = survey_artifact_ids(
        survey_manifest_path=survey_manifest_path,
        survey_manifest=survey_manifest,
        survey_json_path=survey_json_path,
        cwd=cwd,
    )
    candidates = _manifest_declarations(survey_manifest, survey)
    if not candidates:
        result.blocking_errors.append(
            "No explicit XLab resource manifest is declared by the Survey manifest or Survey metadata; "
            "declare resource_manifest (or resources.manifest) and resume."
        )
        return result
    unique_candidates = sorted(set(candidates))
    if len(unique_candidates) != 1:
        result.blocking_errors.append(
            "RESOURCE_MANIFEST_DECLARATION_AMBIGUOUS: Survey metadata must declare exactly one resource manifest."
        )
        return result
    if survey_manifest_path is None:
        result.blocking_errors.append(
            "An explicit resource manifest requires the source Survey manifest.json so its path and artifact lineage can be verified."
        )
        return result

    declaration = unique_candidates[0]
    try:
        manifest_path = _contained_declaration_path(survey_manifest_path.parent, declaration)
    except ResourceValidationError:
        result.blocking_errors.append(_public_resource_error("RESOURCE_MANIFEST_DECLARATION_INVALID", declaration))
        return result
    result.manifest_path = manifest_path
    result.manifest_logical_path = manifest_path.relative_to(survey_manifest_path.parent.resolve()).as_posix()

    try:
        bundle = load_resource_bundle(manifest_path)
    except ResourceValidationError:
        result.blocking_errors.append(_public_resource_error("RESOURCE_MANIFEST_INVALID", declaration))
        return result
    try:
        _validate_profile(bundle)
    except ResourceValidationError:
        result.blocking_errors.append(_public_resource_error("RESOURCE_PROFILE_INCOMPATIBLE", declaration))
        return result
    try:
        _validate_survey_lineage(bundle, result.direct_parent_artifact_ids)
    except ResourceValidationError:
        result.blocking_errors.append(_public_resource_error("RESOURCE_MANIFEST_LINEAGE_MISMATCH", declaration))
        return result
    try:
        outcome, component = _select_models(bundle.manifest.models)
    except ResourceValidationError:
        result.blocking_errors.append(_public_resource_error("RESOURCE_MODEL_DECLARATION_INVALID", declaration))
        return result

    result.bundle = bundle
    try:
        result.graph_db_path = _declared_file(bundle, bundle.manifest.graph, "graph.db")
    except ResourceValidationError:
        result.blocking_errors.append(_public_resource_error("RESOURCE_FILE_DECLARATION_INVALID", declaration))
        return result
    result.component_index_dir = bundle.path_for(bundle.manifest.component_index.resource_id)
    result.keynote_cache_path = bundle.path_for(bundle.manifest.keynotes.resource_id)
    if model_cache_root is not None:
        cache = ModelCache(model_cache_root)
        try:
            for role, descriptor in (("outcome_model", outcome), ("component_model", component)):
                snapshot = _model_snapshot(descriptor)
                cached = cache.resolve(snapshot, source=bundle.path_for(descriptor.resource_id))
                result.model_uris[role] = cached.uri
                if role == "outcome_model":
                    result.outcome_model_path = cached.path
                else:
                    result.component_model_path = cached.path
        except ModelCacheError:
            result.blocking_errors.append(_public_resource_error("RESOURCE_MODEL_CACHE_INVALID", declaration))
    return result


def survey_artifact_ids(
    *,
    survey_manifest_path: Path | None,
    survey_manifest: dict[str, Any],
    survey_json_path: Path | None,
    cwd: Path,
) -> tuple[str, ...]:
    """Return direct Survey artifact IDs using the XLab content-addressed identity."""

    paths: list[Path] = []
    if survey_manifest_path is not None:
        for artifact in survey_manifest.get("artifacts", []) if isinstance(survey_manifest.get("artifacts"), list) else []:
            if not isinstance(artifact, dict) or artifact.get("type") not in _SURVEY_ARTIFACT_TYPES:
                continue
            value = artifact.get("path")
            if isinstance(value, str) and value.strip():
                path = _resolve_survey_artifact_path(survey_manifest_path.parent, cwd, value)
                if path is not None:
                    paths.append(path)
    if survey_manifest_path is not None and not paths and survey_json_path is not None and survey_json_path.is_file():
        paths.append(survey_json_path.resolve())
    return tuple(sorted({f"sha256:{_xlab_file_digest(path)}" for path in paths}))


def _manifest_declarations(manifest: dict[str, Any], survey: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for container in (manifest, manifest.get("metadata"), survey, survey.get("metadata")):
        if not isinstance(container, dict):
            continue
        for key in ("resource_manifest", "resource_manifest_path"):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
        resources = container.get("resources")
        if isinstance(resources, dict):
            for key in ("manifest", "manifest_path"):
                value = resources.get(key)
                if isinstance(value, str) and value.strip():
                    values.append(value.strip())
    for artifact in manifest.get("artifacts", []) if isinstance(manifest.get("artifacts"), list) else []:
        if not isinstance(artifact, dict) or artifact.get("type") not in _RESOURCE_ARTIFACT_TYPES:
            continue
        value = artifact.get("path")
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    return values


def _public_resource_error(code: str, declaration: str) -> str:
    logical_path = _validated_logical_declaration(declaration)
    suffix = f" declaration={logical_path}" if logical_path is not None else ""
    return f"{code}:{suffix} Correct the declared resource bundle and resume."


def _validated_logical_declaration(value: str) -> str | None:
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        return None
    if "\\" in value:
        return None
    return pure.as_posix()


def _contained_declaration_path(root: Path, value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ResourceValidationError(f"path must be a normalized relative POSIX path: {value!r}")
    root = root.resolve()
    candidate = (root / Path(*pure.parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ResourceValidationError(f"path escapes the Survey run: {value!r}") from error
    if not candidate.is_file():
        raise ResourceValidationError(f"manifest does not exist: {value!r}")
    return candidate


def _validate_profile(bundle: LoadedResourceBundle) -> None:
    manifest = bundle.manifest
    requirements = (
        ResourceRequirement(manifest.survey.resource_id, (_EXPECTED_SURVEY_SCHEMA,), (_EXPECTED_SURVEY_ALGORITHM,), ("citation_traces",)),
        ResourceRequirement(manifest.graph.resource_id, (_EXPECTED_GRAPH_SCHEMA,), (_EXPECTED_GRAPH_ALGORITHM,), ("paper_neighbors",)),
        ResourceRequirement(
            manifest.component_index.resource_id,
            (_EXPECTED_COMPONENT_SCHEMA,),
            (_EXPECTED_COMPONENT_ALGORITHM,),
            ("component_similarity",),
        ),
        ResourceRequirement(manifest.keynotes.resource_id, (_EXPECTED_KEYNOTES_SCHEMA,), (_EXPECTED_KEYNOTES_ALGORITHM,), ("paper_keynotes",)),
    )
    validate_resource_bundle(bundle.root, manifest, requirements=requirements)
    outcome, component = _select_models(manifest.models)
    component_dimension = manifest.component_index.dimension_map.get("embedding")
    if component_dimension is None or component_dimension <= 0:
        raise ResourceValidationError("component index must declare a positive embedding dimension")
    outcome_dimension = outcome.dimension_map.get("embedding")
    if outcome_dimension != 384:
        raise ResourceValidationError(
            "OutcomeRAG all-MiniLM-L6-v2 model must declare embedding dimension 384"
        )
    if _declared_model_id(outcome) != "sentence-transformers/all-MiniLM-L6-v2":
        raise ResourceValidationError(
            "OutcomeRAG model provenance.model_id must be sentence-transformers/all-MiniLM-L6-v2"
        )
    for descriptor, dimension in ((outcome, outcome_dimension), (component, component_dimension)):
        model_requirement = ResourceRequirement(
            descriptor.resource_id,
            (_EXPECTED_MODEL_SCHEMA,),
            (_EXPECTED_MODEL_ALGORITHM,),
            ("sentence_embedding",),
            (("embedding", dimension),),
        )
        validate_resource_bundle(bundle.root, manifest, requirements=(model_requirement,))
    if component.dimension_map.get("embedding") != component_dimension:
        raise ResourceValidationError(
            "component model embedding dimension must match the component FAISS index"
        )


def _validate_survey_lineage(bundle: LoadedResourceBundle, survey_ids: tuple[str, ...]) -> None:
    if not survey_ids:
        raise ResourceValidationError("source Survey artifact identity could not be established from its manifest")
    declared = set(bundle.manifest.survey.parent_artifact_ids)
    if not declared:
        raise ResourceValidationError("survey resource must declare parent_artifact_ids for the source Survey artifact")
    if not declared.intersection(survey_ids):
        raise ResourceValidationError(
            "survey resource lineage does not include the selected Survey artifact ID " + ", ".join(survey_ids)
        )


def _select_models(models: tuple[ResourceDescriptor, ...]) -> tuple[ResourceDescriptor, ResourceDescriptor]:
    outcome = [model for model in models if _model_role(model, models) == "outcome_model"]
    component = [model for model in models if _model_role(model, models) == "component_model"]
    if len(outcome) != 1:
        raise ResourceValidationError(f"bundle must identify exactly one outcome model; found {len(outcome)}")
    if len(component) != 1:
        raise ResourceValidationError(f"bundle must identify exactly one component novelty model; found {len(component)}")
    if outcome[0].resource_id == component[0].resource_id:
        raise ResourceValidationError("outcome and component novelty models must be distinct immutable descriptors")
    return outcome[0], component[0]


def _model_role(descriptor: ResourceDescriptor, models: tuple[ResourceDescriptor, ...]) -> str:
    provenance_role = descriptor.provenance_map.get("role", "").strip().casefold().replace("-", "_")
    searchable = " ".join((provenance_role, descriptor.resource_id.casefold(), *descriptor.capabilities))
    if "outcome" in searchable:
        return "outcome_model"
    if "component" in searchable or "novelty" in searchable:
        return "component_model"
    if len(models) == 1:
        return "outcome_model"
    return f"model:{descriptor.resource_id}"


def _declared_model_id(descriptor: ResourceDescriptor) -> str:
    return descriptor.provenance_map.get("model_id", descriptor.resource_id).strip()


def _model_snapshot(descriptor: ResourceDescriptor) -> ModelSnapshot:
    return ModelSnapshot(
        model_id=descriptor.resource_id,
        revision=descriptor.digest,
        files=tuple(ModelFile(path=file.path, sha256=file.sha256, size=file.size) for file in descriptor.files),
        capabilities=descriptor.capabilities,
        dimensions=descriptor.dimensions,
    )


def _declared_file(bundle: LoadedResourceBundle, descriptor: ResourceDescriptor, filename: str) -> Path:
    matches = [file.path for file in descriptor.files if PurePosixPath(file.path).name == filename]
    if len(matches) != 1:
        raise ResourceValidationError(
            f"resource {descriptor.resource_id!r} must declare exactly one {filename}; found {len(matches)}"
        )
    return bundle.path_for(descriptor.resource_id) / Path(*PurePosixPath(matches[0]).parts)


def _resource_uri(bundle_id: str, resource_id: str) -> str:
    return f"xlab-resource://{quote(bundle_id, safe='')}/{quote(resource_id, safe='')}"


def _resolve_survey_artifact_path(manifest_root: Path, cwd: Path, value: str) -> Path | None:
    path = Path(value).expanduser()
    candidates = [path.resolve()] if path.is_absolute() else [(cwd / path).resolve(), (manifest_root / path).resolve()]
    manifest_root = manifest_root.resolve()
    cwd = cwd.resolve()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        if _is_contained(manifest_root, candidate) or _is_contained(cwd, candidate):
            return candidate
    return None


def _is_contained(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _xlab_file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(b"file\0")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
