from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .contracts import (
    JsonValue,
    ProviderError,
    ProviderRequest,
    ProviderResult,
    ProviderTrace,
    ProviderUsage,
)


FixtureKey = tuple[str, str]


@dataclass(frozen=True)
class FakeFixture:
    text: str
    json_value: dict[str, JsonValue] | None = None
    usage: ProviderUsage = ProviderUsage()


class DeterministicFakeProvider:
    def __init__(self, fixtures: Mapping[FixtureKey, FakeFixture]) -> None:
        self._fixtures = dict(fixtures)
        self.traces: list[ProviderTrace] = []

    def complete(self, request: ProviderRequest) -> ProviderResult:
        trace = ProviderTrace(
            provider="fake",
            operation=request.operation,
            input_digest=request.input_digest,
            output_kind=request.output_kind,
            model=request.model,
            attempts=1,
            status="success",
        )
        fixture = self._fixtures.get((request.operation, request.input_digest))
        if fixture is None:
            trace = ProviderTrace(
                provider="fake",
                operation=request.operation,
                input_digest=request.input_digest,
                output_kind=request.output_kind,
                model=request.model,
                attempts=1,
                status="error",
                error_code="fixture_not_found",
            )
            self.traces.append(trace)
            raise ProviderError("No deterministic provider fixture matched the request.", trace=trace)
        if request.output_kind == "json" and fixture.json_value is None:
            trace = ProviderTrace(
                provider="fake",
                operation=request.operation,
                input_digest=request.input_digest,
                output_kind=request.output_kind,
                model=request.model,
                attempts=1,
                status="error",
                error_code="fixture_kind_mismatch",
            )
            self.traces.append(trace)
            raise ProviderError("The matched fixture has no structured JSON result.", trace=trace)
        self.traces.append(trace)
        return ProviderResult(
            text=fixture.text,
            json_value=fixture.json_value,
            usage=fixture.usage,
            trace=trace,
        )
