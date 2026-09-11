"""Portable, versioned resource contracts for research idea generation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, Mapping

RESOURCE_MANIFEST_SCHEMA_VERSION = "xlab.research_idea.resources.v1"
RESOURCE_KINDS = ("survey", "graph", "component_index", "keynotes", "model")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

ResourceKind = Literal["survey", "graph", "component_index", "keynotes", "model"]


class ResourceValidationError(ValueError):
    """Raised when a resource bundle does not satisfy its contract."""


@dataclass(frozen=True)
class FileDigest:
    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ResourceDescriptor:
    resource_id: str
    kind: ResourceKind
    schema_version: str
    algorithm_version: str
    path: str
    digest: str
    files: tuple[FileDigest, ...]
    parent_artifact_ids: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    dimensions: tuple[tuple[str, int], ...] = ()
    provenance: tuple[tuple[str, str], ...] = ()

    @property
    def dimension_map(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self.dimensions))

    @property
    def provenance_map(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self.provenance))


@dataclass(frozen=True)
class ResourceManifest:
    schema_version: str
    bundle_id: str
    survey: ResourceDescriptor
    graph: ResourceDescriptor
    component_index: ResourceDescriptor
    keynotes: ResourceDescriptor
    models: tuple[ResourceDescriptor, ...]

    @property
    def resources(self) -> tuple[ResourceDescriptor, ...]:
        return (self.survey, self.graph, self.component_index, self.keynotes, *self.models)

    def by_id(self) -> Mapping[str, ResourceDescriptor]:
        return MappingProxyType({resource.resource_id: resource for resource in self.resources})


@dataclass(frozen=True)
class ResourceRequirement:
    resource_id: str
    schema_versions: tuple[str, ...]
    algorithm_versions: tuple[str, ...]
    capabilities: tuple[str, ...] = ()
    dimensions: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class LoadedResourceBundle:
    root: Path
    manifest: ResourceManifest

    def path_for(self, resource_id: str) -> Path:
        descriptor = self.manifest.by_id().get(resource_id)
        if descriptor is None:
            raise KeyError(resource_id)
        return _contained_path(self.root, descriptor.path, label=f"resource {resource_id}")


def load_resource_bundle(
    manifest_path: Path,
    *,
    requirements: tuple[ResourceRequirement, ...] = (),
) -> LoadedResourceBundle:
    """Load a portable bundle and validate containment, compatibility, and content."""

    manifest_path = manifest_path.resolve()
    root = manifest_path.parent
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResourceValidationError(f"Cannot load resource manifest {manifest_path}: {exc}") from exc
    manifest = resource_manifest_from_dict(payload)
    validate_resource_bundle(root, manifest, requirements=requirements)
    return LoadedResourceBundle(root=root, manifest=manifest)


def validate_resource_bundle(
    root: Path,
    manifest: ResourceManifest,
    *,
    requirements: tuple[ResourceRequirement, ...] = (),
) -> None:
    root = root.resolve()
    if manifest.schema_version != RESOURCE_MANIFEST_SCHEMA_VERSION:
        raise ResourceValidationError(
            f"Unsupported resource manifest schema {manifest.schema_version!r}; expected {RESOURCE_MANIFEST_SCHEMA_VERSION!r}"
        )

    resources_by_id = manifest.by_id()
    if len(resources_by_id) != len(manifest.resources):
        raise ResourceValidationError("Resource IDs must be unique")

    for resource in manifest.resources:
        _validate_descriptor(root, resource)

    for requirement in requirements:
        resource = resources_by_id.get(requirement.resource_id)
        if resource is None:
            raise ResourceValidationError(f"Missing required resource {requirement.resource_id!r}")
        if resource.schema_version not in requirement.schema_versions:
            raise ResourceValidationError(
                f"Resource {resource.resource_id!r} schema {resource.schema_version!r} is incompatible"
            )
        if resource.algorithm_version not in requirement.algorithm_versions:
            raise ResourceValidationError(
                f"Resource {resource.resource_id!r} algorithm {resource.algorithm_version!r} is incompatible"
            )
        missing = sorted(set(requirement.capabilities) - set(resource.capabilities))
        if missing:
            raise ResourceValidationError(
                f"Resource {resource.resource_id!r} is missing capabilities: {', '.join(missing)}"
            )
        declared_dimensions = dict(resource.dimensions)
        for name, expected in requirement.dimensions:
            actual = declared_dimensions.get(name)
            if actual != expected:
                raise ResourceValidationError(
                    f"Resource {resource.resource_id!r} dimension {name!r} is {actual!r}, expected {expected!r}"
                )


def resource_manifest_from_dict(payload: object) -> ResourceManifest:
    if not isinstance(payload, dict):
        raise ResourceValidationError("Resource manifest must be a JSON object")
    resources = payload.get("resources")
    if not isinstance(resources, dict):
        raise ResourceValidationError("Resource manifest must contain a resources object")

    survey = _descriptor_from_dict(resources.get("survey"), "survey")
    graph = _descriptor_from_dict(resources.get("graph"), "graph")
    component_index = _descriptor_from_dict(resources.get("component_index"), "component_index")
    keynotes = _descriptor_from_dict(resources.get("keynotes"), "keynotes")
    raw_models = resources.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ResourceValidationError("Resource manifest must contain at least one model")
    models = tuple(_descriptor_from_dict(value, "model") for value in raw_models)

    schema_version = _required_string(payload, "schema_version")
    bundle_id = _required_string(payload, "bundle_id")
    return ResourceManifest(
        schema_version=schema_version,
        bundle_id=bundle_id,
        survey=survey,
        graph=graph,
        component_index=component_index,
        keynotes=keynotes,
        models=models,
    )


def descriptor_digest(files: tuple[FileDigest, ...]) -> str:
    """Return the relocation-independent digest of a descriptor's ordered files."""

    canonical = [
        {"path": file.path, "sha256": file.sha256, "size": file.size}
        for file in sorted(files, key=lambda value: value.path)
    ]
    encoded = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _descriptor_from_dict(payload: object, expected_kind: ResourceKind) -> ResourceDescriptor:
    if not isinstance(payload, dict):
        raise ResourceValidationError(f"Resource {expected_kind!r} must be a JSON object")
    kind = _required_string(payload, "kind")
    if kind != expected_kind:
        raise ResourceValidationError(f"Expected resource kind {expected_kind!r}, got {kind!r}")

    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ResourceValidationError(f"Resource {expected_kind!r} must declare non-empty file digests")
    files: list[FileDigest] = []
    for raw_file in raw_files:
        if not isinstance(raw_file, dict):
            raise ResourceValidationError(f"Resource {expected_kind!r} has an invalid file digest")
        size = raw_file.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ResourceValidationError(f"Resource {expected_kind!r} has an invalid file size")
        files.append(
            FileDigest(
                path=_required_string(raw_file, "path"),
                sha256=_required_digest(raw_file, "sha256"),
                size=size,
            )
        )

    parent_artifact_ids = _string_tuple(payload.get("parent_artifact_ids", []), "parent_artifact_ids")
    capabilities = _string_tuple(payload.get("capabilities", []), "capabilities")
    dimensions = _integer_items(payload.get("dimensions", {}), "dimensions")
    provenance = _string_items(payload.get("provenance", {}), "provenance")
    return ResourceDescriptor(
        resource_id=_required_string(payload, "resource_id"),
        kind=expected_kind,
        schema_version=_required_string(payload, "schema_version"),
        algorithm_version=_required_string(payload, "algorithm_version"),
        path=_required_string(payload, "path"),
        digest=_required_digest(payload, "digest"),
        files=tuple(sorted(files, key=lambda value: value.path)),
        parent_artifact_ids=tuple(sorted(set(parent_artifact_ids))),
        capabilities=tuple(sorted(set(capabilities))),
        dimensions=dimensions,
        provenance=provenance,
    )


def _validate_descriptor(root: Path, resource: ResourceDescriptor) -> None:
    resource_path = _contained_path(root, resource.path, label=f"resource {resource.resource_id}")
    if not resource_path.exists():
        raise ResourceValidationError(f"Resource {resource.resource_id!r} does not exist: {resource.path}")

    seen: set[str] = set()
    actual_files: list[FileDigest] = []
    for expected in resource.files:
        if expected.path in seen:
            raise ResourceValidationError(f"Resource {resource.resource_id!r} repeats file {expected.path!r}")
        seen.add(expected.path)
        file_path = _contained_path(resource_path, expected.path, label=f"file in resource {resource.resource_id}")
        if not file_path.is_file():
            raise ResourceValidationError(f"Resource {resource.resource_id!r} is missing file {expected.path!r}")
        actual = FileDigest(path=expected.path, sha256=sha256_file(file_path), size=file_path.stat().st_size)
        if actual.sha256 != expected.sha256:
            raise ResourceValidationError(f"Resource {resource.resource_id!r} file {expected.path!r} has a digest mismatch")
        if actual.size != expected.size:
            raise ResourceValidationError(f"Resource {resource.resource_id!r} file {expected.path!r} has a size mismatch")
        actual_files.append(actual)

    actual_digest = descriptor_digest(tuple(actual_files))
    if actual_digest != resource.digest:
        raise ResourceValidationError(f"Resource {resource.resource_id!r} has a descriptor digest mismatch")


def _contained_path(root: Path, relative: str, *, label: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ResourceValidationError(f"{label.capitalize()} path must be a normalized relative POSIX path: {relative!r}")
    candidate = (root / Path(*pure.parts)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ResourceValidationError(f"{label.capitalize()} escapes its containing bundle: {relative!r}") from exc
    return candidate


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ResourceValidationError(f"{key} must be a non-empty string")
    return value.strip()


def _required_digest(payload: dict[str, Any], key: str) -> str:
    value = _required_string(payload, key).lower()
    if not _SHA256_RE.fullmatch(value):
        raise ResourceValidationError(f"{key} must be a lowercase SHA-256 digest")
    return value


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ResourceValidationError(f"{label} must be an array of non-empty strings")
    return tuple(item.strip() for item in value)


def _integer_items(value: object, label: str) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, dict):
        raise ResourceValidationError(f"{label} must be an object")
    items: list[tuple[str, int]] = []
    for key, item in value.items():
        if not isinstance(key, str) or not key or not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ResourceValidationError(f"{label} must map non-empty names to non-negative integers")
        items.append((key, item))
    return tuple(sorted(items))


def _string_items(value: object, label: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict):
        raise ResourceValidationError(f"{label} must be an object")
    items: list[tuple[str, str]] = []
    for key, item in value.items():
        if not isinstance(key, str) or not key or not isinstance(item, str) or not item.strip():
            raise ResourceValidationError(f"{label} must map non-empty names to non-empty strings")
        items.append((key, item.strip()))
    return tuple(sorted(items))
