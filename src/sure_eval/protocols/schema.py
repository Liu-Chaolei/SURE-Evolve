"""Typed inference protocol catalog and resolution records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


PROTOCOL_CATALOG_SCHEMA = "sure.eval.inference_protocol_catalog.v2"
PROTOCOL_RESOLUTION_SCHEMA = "sure.eval.inference_protocol_resolution.v2"
ALLOWED_PROTOCOL_IDS = ("standard_system", "strict_core")
ALLOWED_ENFORCEMENT_MODES = ("parameter", "attestation", "not_applicable")


@dataclass(frozen=True)
class ProtocolParam:
    name: str
    description: str
    default: Any
    allowed_values: tuple[Any, ...]
    scope: str


@dataclass(frozen=True)
class ProtocolConstraint:
    id: str
    description: str
    category: str
    scope: str


@dataclass(frozen=True)
class ProtocolDefinition:
    id: str
    name: str
    version: str
    policy: str
    description: str
    is_default: bool
    params: dict[str, ProtocolParam]
    constraints: tuple[ProtocolConstraint, ...]

    @classmethod
    def from_dict(cls, protocol_id: str, data: dict[str, Any]) -> "ProtocolDefinition":
        params = {
            name: ProtocolParam(
                name=name,
                description=str(value.get("description", "")),
                default=value.get("default"),
                allowed_values=tuple(value.get("allowed_values") or ()),
                scope=str(value.get("scope", "inference")),
            )
            for name, value in (data.get("params") or {}).items()
        }
        constraints = tuple(
            ProtocolConstraint(
                id=str(value["id"]),
                description=str(value.get("description", "")),
                category=str(value.get("category", "required")),
                scope=str(value.get("scope", "inference")),
            )
            for value in data.get("constraints") or ()
        )
        return cls(
            id=protocol_id,
            name=str(data.get("name", protocol_id)),
            version=str(data.get("version", "")),
            policy=str(data.get("policy", "")),
            description=str(data.get("description", "")),
            is_default=bool(data.get("is_default", False)),
            params=params,
            constraints=constraints,
        )


@dataclass(frozen=True)
class ModelParamMapping:
    enforcement: str
    model_param: str | None = None
    value: Any = None
    mapping: dict[str, Any] = field(default_factory=dict)
    attestation: str | None = None
    note: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelParamMapping":
        return cls(
            enforcement=str(data.get("enforcement", "")),
            model_param=str(data["model_param"]) if data.get("model_param") else None,
            value=data.get("value"),
            mapping=dict(data.get("mapping") or {}),
            attestation=str(data["attestation"]) if data.get("attestation") else None,
            note=str(data.get("note", "")),
        )


@dataclass(frozen=True)
class ModelProtocolConfig:
    enabled: bool
    param_map: dict[str, ModelParamMapping]
    declared_modules: tuple[dict[str, Any], ...]
    attestation: dict[str, Any]
    official_defaults: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelProtocolConfig":
        return cls(
            enabled=bool(data.get("enabled", False)),
            param_map={
                name: ModelParamMapping.from_dict(value)
                for name, value in (data.get("param_map") or {}).items()
            },
            declared_modules=tuple(dict(item) for item in data.get("declared_modules") or ()),
            attestation=dict(data.get("attestation") or {}),
            official_defaults=dict(data.get("official_defaults") or {}),
        )


@dataclass(frozen=True)
class ResolvedParams:
    schema: str
    protocol_id: str
    protocol_version: str
    policy: str
    standard_params: dict[str, Any]
    model_params: dict[str, Any]
    enforcement: dict[str, dict[str, Any]]
    provenance: dict[str, Any]
    effective_params_sha256: str
    unmapped: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "protocol_id": self.protocol_id,
            "protocol_version": self.protocol_version,
            "policy": self.policy,
            "standard_params": self.standard_params,
            "model_params": self.model_params,
            "enforcement": self.enforcement,
            "provenance": self.provenance,
            "effective_params_sha256": self.effective_params_sha256,
            "unmapped": self.unmapped,
        }
