"""Resolve official-default or fail-closed strict inference protocols."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

import yaml

from .schema import (
    ALLOWED_ENFORCEMENT_MODES,
    ALLOWED_PROTOCOL_IDS,
    PROTOCOL_CATALOG_SCHEMA,
    PROTOCOL_RESOLUTION_SCHEMA,
    ModelProtocolConfig,
    ProtocolDefinition,
    ResolvedParams,
)


class ProtocolResolutionError(ValueError):
    """Raised when protocol evidence is missing or cannot be enforced."""


class ModelProtocolInfo(Protocol):
    name: str
    config: dict[str, Any]


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ProtocolResolver:
    def __init__(self, protocols_path: str | Path | None = None) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self.protocols_path = (
            Path(protocols_path) if protocols_path else repo_root / "config/protocols.yaml"
        )
        self._default_protocol_id = ""
        self._protocols = self._load_protocols()

    def _load_protocols(self) -> dict[str, ProtocolDefinition]:
        payload = yaml.safe_load(self.protocols_path.read_text(encoding="utf-8")) or {}
        if payload.get("schema") != PROTOCOL_CATALOG_SCHEMA:
            raise ProtocolResolutionError(
                f"unsupported protocol catalog schema: {payload.get('schema')!r}"
            )
        raw = payload.get("protocols") or {}
        if tuple(raw) != ALLOWED_PROTOCOL_IDS:
            raise ProtocolResolutionError(
                f"protocol catalog must contain exactly {ALLOWED_PROTOCOL_IDS}, got {tuple(raw)}"
            )
        self._default_protocol_id = str(payload.get("default_protocol_id", ""))
        protocols = {
            protocol_id: ProtocolDefinition.from_dict(protocol_id, value)
            for protocol_id, value in raw.items()
        }
        defaults = [item.id for item in protocols.values() if item.is_default]
        if defaults != [self._default_protocol_id] or self._default_protocol_id != "standard_system":
            raise ProtocolResolutionError("standard_system must be the single default protocol")
        return protocols

    def get_protocol(self, protocol_id: str) -> ProtocolDefinition | None:
        return self._protocols.get(protocol_id)

    def list_protocols(self) -> list[str]:
        return list(self._protocols)

    def get_default_protocol(self) -> str:
        return self._default_protocol_id

    def resolve(self, protocol_id: str | None, model_info: ModelProtocolInfo) -> ResolvedParams:
        selected = protocol_id or self.get_default_protocol()
        protocol = self.get_protocol(selected)
        if protocol is None:
            raise ProtocolResolutionError(
                f"unknown protocol {selected!r}; allowed values are {', '.join(ALLOWED_PROTOCOL_IDS)}"
            )
        raw_protocols = model_info.config.get("protocols")
        if not isinstance(raw_protocols, dict):
            raise ProtocolResolutionError(
                f"model {model_info.name!r} has no protocol declarations"
            )
        raw_model_protocol = raw_protocols.get(selected)
        if not isinstance(raw_model_protocol, dict):
            raise ProtocolResolutionError(
                f"model {model_info.name!r} does not declare protocol {selected!r}"
            )
        model_protocol = ModelProtocolConfig.from_dict(raw_model_protocol)
        if not model_protocol.enabled:
            raise ProtocolResolutionError(
                f"model {model_info.name!r} declares protocol {selected!r} as unsupported"
            )
        if selected == "standard_system":
            return self._resolve_standard(protocol, model_info, model_protocol)
        return self._resolve_strict(protocol, model_info, model_protocol)

    def _resolve_standard(
        self,
        protocol: ProtocolDefinition,
        model_info: ModelProtocolInfo,
        model_protocol: ModelProtocolConfig,
    ) -> ResolvedParams:
        official = model_protocol.official_defaults
        required_strings = ("repository", "commit", "weights_revision", "entrypoint")
        missing = [key for key in required_strings if not str(official.get(key, "")).strip()]
        if missing:
            raise ProtocolResolutionError(
                f"model {model_info.name!r} standard_system attestation is missing: {', '.join(missing)}"
            )
        if official.get("invocation_overrides") != {}:
            raise ProtocolResolutionError(
                "standard_system requires invocation_overrides to be an explicit empty mapping"
            )
        effective_params = official.get("effective_params")
        if not isinstance(effective_params, dict):
            raise ProtocolResolutionError(
                "standard_system requires an effective_params mapping captured from the official flow"
            )
        provenance = {
            "official_defaults": official,
            "declared_modules": list(model_protocol.declared_modules),
        }
        standard_params = {name: value.default for name, value in protocol.params.items()}
        enforcement = {
            name: {"mode": "attestation", "evidence": "official_defaults"}
            for name in protocol.params
        }
        digest_payload = {
            "model_id": model_info.name,
            "protocol_id": protocol.id,
            "protocol_version": protocol.version,
            "standard_params": standard_params,
            "effective_params": effective_params,
            "provenance": provenance,
        }
        return ResolvedParams(
            schema=PROTOCOL_RESOLUTION_SCHEMA,
            protocol_id=protocol.id,
            protocol_version=protocol.version,
            policy=protocol.policy,
            standard_params=standard_params,
            model_params={},
            enforcement=enforcement,
            provenance=provenance,
            effective_params_sha256=canonical_sha256(digest_payload),
        )

    def _resolve_strict(
        self,
        protocol: ProtocolDefinition,
        model_info: ModelProtocolInfo,
        model_protocol: ModelProtocolConfig,
    ) -> ResolvedParams:
        standard_params = {name: value.default for name, value in protocol.params.items()}
        model_params: dict[str, Any] = {}
        enforcement: dict[str, dict[str, Any]] = {}
        errors: list[str] = []

        for name, expected in standard_params.items():
            mapping = model_protocol.param_map.get(name)
            if mapping is None:
                errors.append(f"{name}: missing control declaration")
                continue
            if mapping.enforcement not in ALLOWED_ENFORCEMENT_MODES:
                errors.append(f"{name}: invalid enforcement mode {mapping.enforcement!r}")
                continue
            if mapping.enforcement == "parameter":
                if not mapping.model_param:
                    errors.append(f"{name}: parameter enforcement requires model_param")
                    continue
                value = mapping.value
                if value is None:
                    value = mapping.mapping.get(str(expected), expected)
                if value != expected and not mapping.mapping:
                    errors.append(
                        f"{name}: declared value {value!r} does not enforce required value {expected!r}"
                    )
                    continue
                model_params[mapping.model_param] = value
                enforcement[name] = {
                    "mode": "parameter",
                    "model_param": mapping.model_param,
                    "value": value,
                }
            elif mapping.enforcement == "attestation":
                key = mapping.attestation
                if not key or model_protocol.attestation.get(key) is not True:
                    errors.append(f"{name}: required attestation {key!r} is not true")
                    continue
                enforcement[name] = {"mode": "attestation", "evidence": key}
            else:
                if not mapping.note.strip():
                    errors.append(f"{name}: not_applicable requires a reason")
                    continue
                enforcement[name] = {"mode": "not_applicable", "reason": mapping.note}

        if errors:
            raise ProtocolResolutionError(
                f"strict_core cannot be enforced for model {model_info.name!r}: " + "; ".join(errors)
            )

        provenance = {
            "declared_modules": list(model_protocol.declared_modules),
            "attestation": model_protocol.attestation,
        }
        digest_payload = {
            "model_id": model_info.name,
            "protocol_id": protocol.id,
            "protocol_version": protocol.version,
            "standard_params": standard_params,
            "model_params": model_params,
            "enforcement": enforcement,
            "provenance": provenance,
        }
        return ResolvedParams(
            schema=PROTOCOL_RESOLUTION_SCHEMA,
            protocol_id=protocol.id,
            protocol_version=protocol.version,
            policy=protocol.policy,
            standard_params=standard_params,
            model_params=model_params,
            enforcement=enforcement,
            provenance=provenance,
            effective_params_sha256=canonical_sha256(digest_payload),
        )
