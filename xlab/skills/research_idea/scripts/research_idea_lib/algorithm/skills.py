"""Immutable, package-native edit-operator skill resources."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

_RESOURCE_ROOT = Path(__file__).with_name("resources")
_TEMPLATE_FILE = "DEFAULT_SKILL_TEMPLATES.json"
_SKILLS_DIR = "edit_operator_skills"


class OperatorSkillValidationError(ValueError):
    """Raised when package-local operator skill resources are incomplete or inconsistent."""


@dataclass(frozen=True)
class SkillReference:
    path: str
    content: str


@dataclass(frozen=True)
class OperatorSkill:
    name: str
    description: str
    instructions: str
    references: tuple[SkillReference, ...]
    template: Mapping[str, Any]
    structural_mode: str
    scope_preference: str
    requires_control_centered_parent: bool
    content_digest: str


@dataclass(frozen=True)
class OperatorSkillCatalog(Mapping[str, OperatorSkill]):
    _skills: Mapping[str, OperatorSkill]
    content_digest: str

    def __getitem__(self, name: str) -> OperatorSkill:
        return self._skills[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._skills)

    def __len__(self) -> int:
        return len(self._skills)


def load_operator_skill_catalog(
    operator_names: Sequence[str],
    *,
    resource_root: Path | None = None,
) -> OperatorSkillCatalog:
    """Load and validate the operator catalog from package-local resources."""

    declared = tuple(operator_names)
    if not declared or any(not isinstance(name, str) or not name.strip() for name in declared):
        raise OperatorSkillValidationError("declared operator names must be non-empty strings")
    if len(set(declared)) != len(declared):
        raise OperatorSkillValidationError("declared operator names must be unique")
    expected = set(declared)

    root = (resource_root or _RESOURCE_ROOT).resolve()
    template_path = _contained_path(root, _TEMPLATE_FILE, label="template file")
    skills_root = _contained_path(root, _SKILLS_DIR, label="operator skills directory")
    templates = _load_templates(template_path)

    if not skills_root.is_dir():
        raise OperatorSkillValidationError(f"operator skills directory does not exist: {skills_root}")
    resource_dirs: dict[str, Path] = {}
    for child in skills_root.iterdir():
        if child.is_dir():
            contained = _require_contained(root, child, label=f"operator resource {child.name!r}")
            resource_dirs[child.name] = contained

    _validate_name_set("template", expected, set(templates))
    _validate_name_set("operator resource directory", expected, set(resource_dirs))

    catalog_files: list[tuple[str, bytes]] = [(_TEMPLATE_FILE, _read_bytes(template_path))]
    loaded: dict[str, OperatorSkill] = {}
    for name in sorted(expected):
        skill_root = resource_dirs[name]
        skill_path = _contained_path(skill_root, "SKILL.md", label=f"skill file for {name}")
        skill_bytes = _read_bytes(skill_path)
        skill_text = _decode_text(skill_bytes, skill_path)
        metadata, instructions = _parse_skill_markdown(skill_text, skill_path)
        skill_name = metadata.get("name", "").strip()
        description = metadata.get("description", "").strip()
        if skill_name != name:
            raise OperatorSkillValidationError(
                f"operator resource {name!r} declares mismatched skill name {skill_name!r}"
            )
        if not description:
            raise OperatorSkillValidationError(f"operator skill {name!r} has an empty description")
        if not instructions:
            raise OperatorSkillValidationError(f"operator skill {name!r} has empty instructions")

        template = templates[name]
        template_description = template.get("description")
        if not isinstance(template_description, str) or not template_description.strip():
            raise OperatorSkillValidationError(f"operator template {name!r} has an empty description")
        structural_mode = _required_template_string(template, "structural_mode", name)
        scope_preference = _required_template_string(template, "scope_preference", name)
        requires_control_centered_parent = template.get("requires_control_centered_parent")
        if not isinstance(requires_control_centered_parent, bool):
            raise OperatorSkillValidationError(
                f"operator template {name!r} requires_control_centered_parent must be boolean"
            )

        references_root = _contained_path(skill_root, "references", label=f"references for {name}")
        if not references_root.is_dir():
            raise OperatorSkillValidationError(f"operator skill {name!r} has no references directory")
        reference_files = sorted(
            (path for path in references_root.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(references_root).as_posix(),
        )
        if not reference_files:
            raise OperatorSkillValidationError(f"operator skill {name!r} has no references")

        references: list[SkillReference] = []
        skill_files: list[tuple[str, bytes]] = [(f"{_SKILLS_DIR}/{name}/SKILL.md", skill_bytes)]
        for reference_path in reference_files:
            contained = _require_contained(root, reference_path, label=f"reference for {name}")
            relative = contained.relative_to(references_root).as_posix()
            content_bytes = _read_bytes(contained)
            content = _decode_text(content_bytes, contained).strip()
            if not content:
                raise OperatorSkillValidationError(
                    f"operator skill {name!r} has empty reference {relative!r}"
                )
            references.append(SkillReference(path=relative, content=content))
            skill_files.append((f"{_SKILLS_DIR}/{name}/references/{relative}", content_bytes))

        template_bytes = _canonical_json_bytes(template)
        skill_files.append((f"{_TEMPLATE_FILE}#{name}", template_bytes))
        loaded[name] = OperatorSkill(
            name=name,
            description=description,
            instructions=instructions,
            references=tuple(references),
            template=_freeze_json(template),
            structural_mode=structural_mode,
            scope_preference=scope_preference,
            requires_control_centered_parent=requires_control_centered_parent,
            content_digest=_content_digest(skill_files),
        )
        catalog_files.extend(skill_files[:-1])

    return OperatorSkillCatalog(
        _skills=MappingProxyType(loaded),
        content_digest=_content_digest(catalog_files),
    )


def _load_templates(path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(_decode_text(_read_bytes(path), path))
    except json.JSONDecodeError as exc:
        raise OperatorSkillValidationError(f"invalid operator template JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise OperatorSkillValidationError("operator templates must be a JSON object")
    templates: dict[str, dict[str, Any]] = {}
    for name, template in payload.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(template, dict):
            raise OperatorSkillValidationError("operator templates must map names to objects")
        templates[name] = template
    return templates


def _required_template_string(template: Mapping[str, Any], field: str, name: str) -> str:
    value = template.get(field)
    if not isinstance(value, str) or not value.strip():
        raise OperatorSkillValidationError(f"operator template {name!r} has an empty {field}")
    return value.strip()


def _parse_skill_markdown(text: str, path: Path) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise OperatorSkillValidationError(f"operator skill lacks frontmatter: {path}")
    try:
        end = next(index for index, line in enumerate(lines[1:], 1) if line.strip() == "---")
    except StopIteration as exc:
        raise OperatorSkillValidationError(f"operator skill has unterminated frontmatter: {path}") from exc

    metadata: dict[str, str] = {}
    for line in lines[1:end]:
        key, separator, value = line.partition(":")
        if separator and key.strip():
            metadata[key.strip()] = value.strip()
    return metadata, "\n".join(lines[end + 1 :]).strip()


def _validate_name_set(label: str, expected: set[str], actual: set[str]) -> None:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unexpected {', '.join(extra)}")
        raise OperatorSkillValidationError(f"{label} catalog mismatch: {'; '.join(details)}")


def _contained_path(root: Path, relative: str, *, label: str) -> Path:
    return _require_contained(root, root / relative, label=label)


def _require_contained(root: Path, path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise OperatorSkillValidationError(f"{label} escapes package resource root: {path}") from exc
    return resolved


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise OperatorSkillValidationError(f"cannot read operator resource {path}: {exc}") from exc


def _decode_text(content: bytes, path: Path) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OperatorSkillValidationError(f"operator resource is not UTF-8: {path}") from exc


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(value[key]) for key in sorted(value)})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _content_digest(files: Sequence[tuple[str, bytes]]) -> str:
    descriptors = [
        {"path": path, "sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
        for path, content in sorted(files, key=lambda item: item[0])
    ]
    return hashlib.sha256(_canonical_json_bytes(descriptors)).hexdigest()
