from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

from .common import (
    JsonObject,
    append_jsonl,
    artifact_paths,
    as_list,
    as_mapping,
    read_json,
    read_jsonl,
    run_lock,
    text,
    utc_now,
)
from .metadata import _ingest_run_unlocked
from .providers import (
    S2_CITATIONS,
    S2_GET_PAPER,
    S2_RECOMMENDATIONS,
    S2_REFERENCES,
    S2_SEARCH,
    TAVILY_SEARCH,
    ProviderRequestError,
    SemanticScholarClient,
    McpWebSearchClient,
)


def _signature(operation: str, arguments: Mapping[str, object]) -> str:
    canonical = dict(arguments)
    if operation in {S2_CITATIONS, S2_REFERENCES, S2_RECOMMENDATIONS}:
        canonical.pop("fields", None)
    return f"{operation}\0{json.dumps(canonical, ensure_ascii=False, sort_keys=True)}"


def _latest_records_by_signature(
    records: list[object],
) -> dict[str, Mapping[str, object]]:
    latest: dict[str, Mapping[str, object]] = {}
    for raw_record in records:
        record = as_mapping(raw_record)
        operation = text(record.get("operation"))
        if operation:
            latest[_signature(operation, as_mapping(record.get("input")))] = record
    return latest


def _successful_signatures(records: list[object]) -> set[str]:
    return {
        signature
        for signature, record in _latest_records_by_signature(records).items()
        if record.get("is_error") is not True
    }


def _retryable_failure_signatures(records: list[object]) -> set[str]:
    return {
        signature
        for signature, record in _latest_records_by_signature(records).items()
        if record.get("is_error") is True
        and record.get("retryable") is not False
    }


def _terminal_failure_signatures(records: list[object]) -> set[str]:
    return {
        signature
        for signature, record in _latest_records_by_signature(records).items()
        if record.get("is_error") is True
        and record.get("retryable") is False
    }


class ProviderRunner:
    def __init__(
        self,
        run_dir: Path,
        *,
        tavily: McpWebSearchClient,
        semantic_scholar: SemanticScholarClient,
    ) -> None:
        self.paths = artifact_paths(run_dir)
        self.tavily = tavily
        self.semantic_scholar = semantic_scholar
        records = read_jsonl(self.paths["provider_results"])
        self.successful = _successful_signatures(records)
        self.retryable_failures = _retryable_failure_signatures(records)
        self.terminal_failures = _terminal_failure_signatures(records)
        self.api_calls = 0
        self.skipped_calls = 0
        self.failed_calls = 0

    def execute(self, action: Mapping[str, object]) -> None:
        operation = text(action.get("operation"))
        arguments = as_mapping(action.get("arguments"))
        signature = _signature(operation, arguments)
        if (
            signature in self.successful
            or signature in self.terminal_failures
            or signature in self.retryable_failures
        ):
            self.skipped_calls += 1
            return
        provider = text(action.get("provider")) or operation.split(".", 1)[0]
        try:
            if operation == TAVILY_SEARCH:
                response = self.tavily.call(operation, arguments)
            elif operation in {
                S2_SEARCH,
                S2_GET_PAPER,
                S2_CITATIONS,
                S2_REFERENCES,
                S2_RECOMMENDATIONS,
            }:
                response = self.semantic_scholar.call(operation, arguments)
            else:
                raise ValueError(f"Unsupported provider operation: {operation}")
        except ProviderRequestError as error:
            self.failed_calls += 1
            append_jsonl(
                self.paths["provider_results"],
                {
                    "captured_at": utc_now(),
                    "provider": provider,
                    "operation": operation,
                    "input": dict(arguments),
                    "is_error": True,
                    "http_status": error.status_code,
                    "attempts": error.attempts,
                    "retryable": error.retryable,
                    "error": str(error),
                },
            )
            if error.retryable:
                self.retryable_failures.add(signature)
            else:
                self.terminal_failures.add(signature)
            if error.auth_failure:
                raise
            return
        self.api_calls += 1
        self.successful.add(signature)
        self.retryable_failures.discard(signature)
        append_jsonl(
            self.paths["provider_results"],
            {
                "captured_at": utc_now(),
                "provider": provider,
                "operation": operation,
                "input": dict(arguments),
                "is_error": False,
                "http_status": response.status_code,
                "attempts": response.attempts,
                "request_id": response.request_id,
                "payload": response.payload,
            },
        )


def _clients_from_environment() -> tuple[McpWebSearchClient, SemanticScholarClient]:
    endpoint = os.environ.get("WEB_SEARCH_MCP_URL", "http://127.0.0.1:17890/mcp").strip()
    semantic_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    missing = [name for name, value in (("SEMANTIC_SCHOLAR_API_KEY", semantic_key),) if not value]
    if missing:
        raise ValueError("Missing required environment variables: " + ", ".join(missing))
    return McpWebSearchClient(endpoint), SemanticScholarClient(semantic_key)


def _collect_run_unlocked(
    run_dir: Path,
    *,
    tavily: McpWebSearchClient | None = None,
    semantic_scholar: SemanticScholarClient | None = None,
) -> JsonObject:
    paths = artifact_paths(run_dir)
    if not paths["request"].is_file() or not paths["query_plan"].is_file():
        raise ValueError("Run is not initialized; execute the init command first.")
    if tavily is None or semantic_scholar is None:
        environment_tavily, environment_semantic = _clients_from_environment()
        tavily = tavily or environment_tavily
        semantic_scholar = semantic_scholar or environment_semantic
    runner = ProviderRunner(
        run_dir,
        tavily=tavily,
        semantic_scholar=semantic_scholar,
    )
    latest_records = _latest_records_by_signature(
        read_jsonl(paths["provider_results"])
    )
    for record in latest_records.values():
        operation = text(record.get("operation"))
        arguments = as_mapping(record.get("input"))
        if (
            record.get("is_error") is not True
            or not operation
            or record.get("retryable") is False
        ):
            continue
        signature = _signature(operation, arguments)
        runner.retryable_failures.discard(signature)
        runner.execute(
            {
                "provider": record.get("provider"),
                "operation": operation,
                "arguments": arguments,
            }
        )
    plan = as_mapping(read_json(paths["query_plan"], {}))
    for raw_action in as_list(plan.get("queries")):
        runner.execute(as_mapping(raw_action))

    rounds = 0
    result: JsonObject = {}
    while rounds < 16:
        rounds += 1
        result = _ingest_run_unlocked(run_dir)
        followups = as_mapping(read_json(paths["followups"], {}))
        actions = [
            as_mapping(action)
            for action in as_list(followups.get("actions"))
            if as_mapping(action)
        ]
        if not actions:
            break
        before = (
            runner.api_calls,
            runner.skipped_calls,
            runner.failed_calls,
        )
        for action in actions:
            runner.execute(action)
        after = (
            runner.api_calls,
            runner.skipped_calls,
            runner.failed_calls,
        )
        if after == before:
            break
    else:
        raise RuntimeError("Provider expansion exceeded the 16-round safety limit.")

    return {
        "status": "collected",
        "rounds": rounds,
        "api_calls": runner.api_calls,
        "skipped_calls": runner.skipped_calls,
        "failed_calls": runner.failed_calls,
        **result,
    }


def collect_run(
    run_dir: Path,
    *,
    tavily: McpWebSearchClient | None = None,
    semantic_scholar: SemanticScholarClient | None = None,
) -> JsonObject:
    with run_lock(run_dir, "collect"):
        return _collect_run_unlocked(
            run_dir,
            tavily=tavily,
            semantic_scholar=semantic_scholar,
        )
