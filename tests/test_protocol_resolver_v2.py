from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from sure_eval.protocols import ProtocolResolutionError, ProtocolResolver


@dataclass
class _Model:
    name: str
    config: dict[str, Any]


def _standard_config() -> dict[str, Any]:
    return {
        "protocols": {
            "standard_system": {
                "enabled": True,
                "declared_modules": [{"name": "official-vad", "version": "1"}],
                "official_defaults": {
                    "repository": "https://example.invalid/official.git",
                    "commit": "a" * 40,
                    "weights_revision": "revision-1",
                    "entrypoint": "official.infer",
                    "invocation_overrides": {},
                    "effective_params": {"beam_size": 5, "temperature": 0.7},
                },
            }
        }
    }


def _strict_config() -> dict[str, Any]:
    parameter_controls = {
        "decode_mode": ("decode_mode", "greedy"),
        "sampling_enabled": ("do_sample", False),
        "temperature": ("temperature", 0.0),
        "beam_size": ("num_beams", 1),
        "num_hypotheses": ("num_return_sequences", 1),
        "max_batch_size": ("batch_size", 1),
    }
    param_map = {
        name: {"enforcement": "parameter", "model_param": target, "value": value}
        for name, (target, value) in parameter_controls.items()
    }
    for name, attestation in {
        "external_lm_enabled": "no_external_lm",
        "hotwords_enabled": "no_hotwords",
        "retrieval_enabled": "no_retrieval",
        "multi_pass_enabled": "single_pass",
    }.items():
        param_map[name] = {
            "enforcement": "attestation",
            "attestation": attestation,
        }
    return {
        "protocols": {
            "strict_core": {
                "enabled": True,
                "param_map": param_map,
                "attestation": {
                    "no_external_lm": True,
                    "no_hotwords": True,
                    "no_retrieval": True,
                    "single_pass": True,
                },
            }
        }
    }


def test_standard_system_is_the_default_and_records_official_provenance() -> None:
    resolver = ProtocolResolver()
    model = _Model("Org__Model", _standard_config())

    first = resolver.resolve(None, model)
    second = resolver.resolve("standard_system", model)

    assert resolver.get_default_protocol() == "standard_system"
    assert resolver.list_protocols() == ["standard_system", "strict_core"]
    assert first.protocol_id == "standard_system"
    assert first.policy == "official_defaults"
    assert first.model_params == {}
    assert first.provenance["official_defaults"]["invocation_overrides"] == {}
    assert first.effective_params_sha256 == second.effective_params_sha256


def test_standard_system_rejects_user_overrides() -> None:
    config = _standard_config()
    config["protocols"]["standard_system"]["official_defaults"]["invocation_overrides"] = {
        "temperature": 0
    }

    with pytest.raises(ProtocolResolutionError, match="explicit empty mapping"):
        ProtocolResolver().resolve(None, _Model("Org__Model", config))


def test_strict_core_resolves_every_required_control() -> None:
    resolved = ProtocolResolver().resolve(
        "strict_core",
        _Model("Org__Model", _strict_config()),
    )

    assert resolved.policy == "constrained"
    assert resolved.model_params["do_sample"] is False
    assert resolved.model_params["num_beams"] == 1
    assert resolved.unmapped == {}
    assert set(resolved.enforcement) == set(resolved.standard_params)


def test_strict_core_fails_closed_when_one_control_is_missing() -> None:
    config = _strict_config()
    del config["protocols"]["strict_core"]["param_map"]["beam_size"]

    with pytest.raises(ProtocolResolutionError, match="beam_size: missing"):
        ProtocolResolver().resolve("strict_core", _Model("Org__Model", config))


def test_strict_core_accepts_explicit_not_applicable_with_reason() -> None:
    config = _strict_config()
    config["protocols"]["strict_core"]["param_map"]["beam_size"] = {
        "enforcement": "not_applicable",
        "note": "The classifier has no decoding search.",
    }

    resolved = ProtocolResolver().resolve("strict_core", _Model("Org__Classifier", config))

    assert resolved.enforcement["beam_size"]["mode"] == "not_applicable"


def test_unknown_protocol_is_rejected() -> None:
    with pytest.raises(ProtocolResolutionError, match="allowed values"):
        ProtocolResolver().resolve("custom", _Model("Org__Model", _standard_config()))
