"""Strict structured input contract for the main-flow evaluation agent."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from sure_eval.protocols.schema import ALLOWED_PROTOCOL_IDS
from sure_eval.storage import validate_model_id


MAIN_FLOW_INPUT_SCHEMA = "sure.eval.main_flow_input.v2"
ALLOWED_EXECUTION_MODES = ("auto", "reuse_result", "reevaluate", "retest")
_IDENTITY_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class MainFlowInputError(ValueError):
    """Raised when an input could bypass a benchmark contract."""


def _only_keys(value: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise MainFlowInputError(f"unsupported field(s) at {location}: {', '.join(unknown)}")


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MainFlowInputError(f"{location} must be a mapping")
    return value


@dataclass(frozen=True, order=True)
class DatasetIdentity:
    name: str
    version: str
    split: str

    @property
    def id(self) -> str:
        return f"{self.name}__{self.version}"

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "name": self.name, "version": self.version, "split": self.split}


@dataclass(frozen=True)
class MainFlowInput:
    model_id: str
    datasets: tuple[DatasetIdentity, ...]
    protocol_id: str = "standard_system"
    execution_mode: str = "auto"
    schema: str = MAIN_FLOW_INPUT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "target": {"model_id": self.model_id},
            "datasets": [item.to_dict() for item in self.datasets],
            "inference": {"protocol_id": self.protocol_id},
            "execution": {"mode": self.execution_mode},
        }


def parse_main_flow_input(payload: dict[str, Any]) -> MainFlowInput:
    if not isinstance(payload, dict):
        raise MainFlowInputError("MAIN_FLOW_INPUT must be a mapping")
    _only_keys(payload, {"schema", "target", "datasets", "inference", "execution"}, "root")
    schema = payload.get("schema", MAIN_FLOW_INPUT_SCHEMA)
    if schema != MAIN_FLOW_INPUT_SCHEMA:
        raise MainFlowInputError(f"unsupported MAIN_FLOW_INPUT schema: {schema!r}")

    target = _mapping(payload.get("target"), "target")
    _only_keys(target, {"model_id"}, "target")
    try:
        model_id = validate_model_id(str(target.get("model_id", "")))
    except ValueError as exc:
        raise MainFlowInputError(str(exc)) from exc

    raw_datasets = payload.get("datasets")
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise MainFlowInputError("datasets must be a non-empty list")
    datasets: list[DatasetIdentity] = []
    for index, raw in enumerate(raw_datasets):
        dataset = _mapping(raw, f"datasets[{index}]")
        _only_keys(dataset, {"name", "version", "split"}, f"datasets[{index}]")
        name = str(dataset.get("name", "")).strip()
        version = str(dataset.get("version", "")).strip()
        split = str(dataset.get("split", "test")).strip()
        if "__" in name or not _IDENTITY_PART.fullmatch(name):
            raise MainFlowInputError(
                f"datasets[{index}].name must be the source dataset name without task/version suffixes"
            )
        if not _IDENTITY_PART.fullmatch(version):
            raise MainFlowInputError(f"datasets[{index}].version is invalid")
        if not _IDENTITY_PART.fullmatch(split):
            raise MainFlowInputError(f"datasets[{index}].split is invalid")
        datasets.append(DatasetIdentity(name=name, version=version, split=split))
    if len({item.id + "@" + item.split for item in datasets}) != len(datasets):
        raise MainFlowInputError("datasets contains duplicate identities")

    inference = _mapping(payload.get("inference", {}), "inference")
    _only_keys(inference, {"protocol_id"}, "inference")
    protocol_id = str(inference.get("protocol_id", "standard_system"))
    if protocol_id not in ALLOWED_PROTOCOL_IDS:
        raise MainFlowInputError(
            f"inference.protocol_id must be one of {', '.join(ALLOWED_PROTOCOL_IDS)}"
        )

    execution = _mapping(payload.get("execution", {}), "execution")
    _only_keys(execution, {"mode"}, "execution")
    execution_mode = str(execution.get("mode", "auto"))
    if execution_mode not in ALLOWED_EXECUTION_MODES:
        raise MainFlowInputError(
            f"execution.mode must be one of {', '.join(ALLOWED_EXECUTION_MODES)}"
        )

    return MainFlowInput(
        model_id=model_id,
        datasets=tuple(datasets),
        protocol_id=protocol_id,
        execution_mode=execution_mode,
    )
