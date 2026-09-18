"""Preserve mature-root identity independently of provider field omissions."""
from copy import deepcopy
from typing import Any, Mapping

IDENTITY_FIELDS = ("tags", "root_domains")
ROOT_IDENTITY_INSTRUCTION = (
    "Copy mature_idea.tags and mature_idea.root_domains exactly into the returned "
    "root_idea (inside replan for a replan response). Preserve values and order, "
    "including empty arrays. These identity fields are not scientific edits."
)


def inherit_root_identity(
    mature: Mapping[str, Any], root: Mapping[str, Any], *, context: str
) -> tuple[dict[str, Any], list[str]]:
    """Copy missing keys only; leave explicit changes for strict validation."""
    result = deepcopy(dict(root))
    inherited = []
    for field in IDENTITY_FIELDS:
        if field in mature:
            _validate_value(mature[field], f"mature_idea.{field}")
            if field not in result:
                result[field] = deepcopy(mature[field])
                inherited.append(field)
        if field in result:
            _validate_value(result[field], f"{context}.{field}")
    return result, inherited


def validate_root_identity(
    mature: Mapping[str, Any], root: Mapping[str, Any], *, context: str
) -> None:
    for field in IDENTITY_FIELDS:
        if field in mature:
            _validate_value(mature[field], f"mature_idea.{field}")
            if field not in root:
                raise ValueError(f"{context}.{field}: missing frozen identity field")
        if field in root:
            _validate_value(root[field], f"{context}.{field}")
        if (field in root) != (field in mature) or root.get(field) != mature.get(field):
            raise ValueError(f"{context}.{field}: explicitly changed frozen identity values or order")


def _validate_value(value: Any, label: str) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label}: invalid type; expected a list of strings")
