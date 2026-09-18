from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Callable, Literal, Protocol, TypeAlias


JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)
OutputKind: TypeAlias = Literal["json", "text"]


@dataclass(frozen=True)
class ProviderRequest:
    operation: str
    model: str
    structured_input: JsonValue = field(repr=False)
    system_prompt: str = field(repr=False)
    user_prompt: str = field(repr=False)
    output_kind: OutputKind = "json"
    temperature: float | None = None
    validation_profile: str | None = None
    response_validator: Callable[["ProviderResult"], object] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.response_validator is not None and (
            not callable(self.response_validator) or not isinstance(self.validation_profile, str) or not self.validation_profile.strip()
        ):
            raise ValueError("A response validator requires a stable validation_profile")

    def cache_payload(self) -> dict[str, object]:
        return {item.name: deepcopy(getattr(self, item.name)) for item in fields(self)
                if item.name != "response_validator"
                and not (item.name == "validation_profile" and self.validation_profile is None)}

    @property
    def input_digest(self) -> str:
        return structured_input_digest(self.structured_input)


@dataclass(frozen=True)
class ProviderUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class ProviderTrace:
    provider: str
    operation: str
    input_digest: str
    output_kind: OutputKind
    model: str
    attempts: int
    status: Literal["success", "error"]
    error_code: str | None = None
    routing: dict[str, JsonValue] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, object]:
        result = {
            "provider": self.provider,
            "operation": self.operation,
            "input_digest": self.input_digest,
            "output_kind": self.output_kind,
            "model": self.model,
            "attempts": self.attempts,
            "status": self.status,
            "error_code": self.error_code,
        }
        if self.routing is not None:
            result["routing"] = deepcopy(self.routing)
        return result


@dataclass(frozen=True)
class ProviderResult:
    text: str
    json_value: dict[str, JsonValue] | None
    usage: ProviderUsage
    trace: ProviderTrace


class Provider(Protocol):
    def complete(self, request: ProviderRequest) -> ProviderResult: ...


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, trace: ProviderTrace) -> None:
        super().__init__(message)
        self.trace = trace


class ProviderExhaustedError(ProviderError):
    pass


class ProviderRecoveryBlockedError(ValueError):
    """Recovery needs reconciliation; never a scientific branch rejection."""


class ProviderContractError(ProviderExhaustedError, ValueError):
    """No route returned a usable response; retain semantic repair feedback."""

    def __init__(self, message: str, *, trace: ProviderTrace, previous_draft=None, validation_issues=None) -> None:
        super().__init__(message, trace=trace)
        self.previous_draft = deepcopy(previous_draft)
        self.validation_issues = deepcopy(validation_issues or [])


def structured_input_digest(value: JsonValue) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
