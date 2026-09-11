from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeAlias


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

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "operation": self.operation,
            "input_digest": self.input_digest,
            "output_kind": self.output_kind,
            "model": self.model,
            "attempts": self.attempts,
            "status": self.status,
            "error_code": self.error_code,
        }


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


def structured_input_digest(value: JsonValue) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
