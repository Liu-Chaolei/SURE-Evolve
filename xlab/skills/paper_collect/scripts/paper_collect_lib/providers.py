from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import TypeAlias

from .common import JsonObject, as_list, integer, text


TAVILY_SEARCH = "tavily.search"
S2_SEARCH = "semantic_scholar.search_papers"
S2_GET_PAPER = "semantic_scholar.get_paper"
S2_CITATIONS = "semantic_scholar.get_paper_citations"
S2_REFERENCES = "semantic_scholar.get_paper_references"
S2_RECOMMENDATIONS = "semantic_scholar.get_recommendations"

PAPER_FIELDS = [
    "paperId",
    "corpusId",
    "title",
    "abstract",
    "authors",
    "year",
    "publicationDate",
    "venue",
    "publicationTypes",
    "fieldsOfStudy",
    "citationCount",
    "influentialCitationCount",
    "referenceCount",
    "externalIds",
    "url",
    "openAccessPdf",
    "tldr",
]
RELATION_FIELDS = [field for field in PAPER_FIELDS if field != "tldr"]
RECOMMENDATION_FIELDS = list(RELATION_FIELDS)

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)


@dataclass(frozen=True)
class ProviderResponse:
    payload: JsonValue
    status_code: int
    attempts: int
    request_id: str | None


class ProviderRequestError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        provider: str,
        operation: str,
        status_code: int | None,
        attempts: int,
        auth_failure: bool = False,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.operation = operation
        self.status_code = status_code
        self.attempts = attempts
        self.auth_failure = auth_failure
        self.retryable = retryable


class RequestRateLimiter:
    def __init__(
        self,
        interval_seconds: float,
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.interval_seconds = max(0.0, interval_seconds)
        self.sleep = sleep
        self.monotonic = monotonic
        self._lock = threading.Lock()
        self._last_request_at: float | None = None

    def wait(self) -> None:
        if self.interval_seconds == 0:
            return
        with self._lock:
            now = self.monotonic()
            if self._last_request_at is not None:
                remaining = self.interval_seconds - (
                    now - self._last_request_at
                )
                if remaining > 0:
                    self.sleep(remaining)
                    now = self.monotonic()
            self._last_request_at = now


class JsonApiClient:
    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        headers: Mapping[str, str],
        timeout_seconds: float = 60.0,
        max_attempts: int = 4,
        min_interval_seconds: float = 0.0,
        opener: Callable[..., object] = urllib.request.urlopen,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        rate_limiter: RequestRateLimiter | None = None,
    ) -> None:
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.headers = dict(headers)
        self.secret_values: set[str] = set()
        for key, value in self.headers.items():
            if not any(
                marker in key.lower()
                for marker in ("authorization", "api-key", "apikey", "token")
            ):
                continue
            self.secret_values.add(value)
            if value.lower().startswith("bearer "):
                self.secret_values.add(value[7:])
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max(1, max_attempts)
        self.opener = opener
        self.sleep = sleep
        self.rate_limiter = rate_limiter or RequestRateLimiter(
            min_interval_seconds,
            sleep=sleep,
            monotonic=monotonic,
        )

    def request(
        self,
        *,
        operation: str,
        method: str,
        path: str,
        query: Mapping[str, object] | None = None,
        body: Mapping[str, object] | None = None,
        expect_response: bool = True,
    ) -> ProviderResponse:
        query_items: list[tuple[str, str]] = []
        for key, value in (query or {}).items():
            if value is None or value == "":
                continue
            if isinstance(value, list):
                query_items.append((key, ",".join(text(item) for item in value)))
            elif isinstance(value, bool):
                query_items.append((key, str(value).lower()))
            else:
                query_items.append((key, text(value)))
        url = f"{self.base_url}{path}"
        if query_items:
            url = f"{url}?{urllib.parse.urlencode(query_items)}"
        encoded_body = (
            json.dumps(body, ensure_ascii=False).encode("utf-8")
            if body is not None
            else None
        )
        request_headers = {
            "Accept": "application/json",
            "User-Agent": "xlab-paper-collect/2.2.0",
            **self.headers,
        }
        if encoded_body is not None:
            request_headers["Content-Type"] = "application/json"

        last_error = ""
        last_status: int | None = None
        for attempt in range(1, self.max_attempts + 1):
            self.rate_limiter.wait()
            request = urllib.request.Request(
                url,
                data=encoded_body,
                headers=request_headers,
                method=method,
            )
            try:
                with self.opener(request, timeout=self.timeout_seconds) as response:
                    status = int(response.getcode())
                    raw = response.read()
                    request_id = (
                        response.headers.get("x-request-id")
                        or response.headers.get("X-Request-ID")
                    )
                    if not expect_response:
                        return ProviderResponse(None, status, attempt, request_id)
                    payload = self._decode_payload(raw, operation, status, attempt)
                    return ProviderResponse(payload, status, attempt, request_id)
            except urllib.error.HTTPError as error:
                last_status = error.code
                raw = error.read(64 * 1024)
                last_error = self._error_message(raw, error.reason)
                if error.code in {401, 403}:
                    raise ProviderRequestError(
                        f"{self.provider} authentication failed ({error.code}): {last_error}",
                        provider=self.provider,
                        operation=operation,
                        status_code=error.code,
                        attempts=attempt,
                        auth_failure=True,
                        retryable=True,
                    ) from error
                if error.code not in {408, 425, 429, 500, 502, 503, 504}:
                    raise ProviderRequestError(
                        f"{self.provider} request failed ({error.code}): {last_error}",
                        provider=self.provider,
                        operation=operation,
                        status_code=error.code,
                        attempts=attempt,
                    ) from error
                delay = self._retry_delay(attempt, error.headers.get("Retry-After"))
            except (TimeoutError, socket.timeout, urllib.error.URLError) as error:
                last_error = text(getattr(error, "reason", error))[:2000]
                delay = self._retry_delay(attempt, None)
            if attempt < self.max_attempts:
                self.sleep(delay)

        raise ProviderRequestError(
            (
                f"{self.provider} request failed after {self.max_attempts} attempts"
                f"{f' ({last_status})' if last_status is not None else ''}: {last_error}"
            ),
            provider=self.provider,
            operation=operation,
            status_code=last_status,
            attempts=self.max_attempts,
            retryable=True,
        )

    def _decode_payload(
        self,
        raw: bytes,
        operation: str,
        status: int,
        attempt: int,
    ) -> JsonValue:
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProviderRequestError(
                f"{self.provider} returned invalid JSON for {operation}.",
                provider=self.provider,
                operation=operation,
                status_code=status,
                attempts=attempt,
            ) from error
        return value

    def _error_message(self, raw: bytes, fallback: object) -> str:
        try:
            value = json.loads(raw.decode("utf-8"))
            if isinstance(value, dict):
                for key in ("message", "error", "detail"):
                    message = value.get(key)
                    if message:
                        return self._redact(text(message)[:2000])
            return self._redact(text(value)[:2000])
        except (UnicodeDecodeError, json.JSONDecodeError):
            message = (
                raw.decode("utf-8", errors="replace")[:2000]
                or text(fallback)[:2000]
            )
            return self._redact(message)

    def _redact(self, value: str) -> str:
        result = value
        for secret in sorted(self.secret_values, key=len, reverse=True):
            if len(secret) >= 4:
                result = result.replace(secret, "[redacted]")
        return result

    @staticmethod
    def _retry_delay(attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(60.0, max(0.0, float(retry_after)))
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    return min(
                        60.0,
                        max(0.0, retry_at.timestamp() - time.time()),
                    )
                except (TypeError, ValueError, OverflowError):
                    pass
        return min(30.0, float(2 ** (attempt - 1)))


class McpWebSearchClient:
    """Stateless HTTP MCP client for the configured web-search service."""

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:17890/mcp",
        *,
        timeout_seconds: float = 120.0,
        max_attempts: int = 4,
        opener: Callable[..., object] = urllib.request.urlopen,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not endpoint.strip():
            raise ValueError("WEB_SEARCH_MCP_URL is required.")
        parsed = urllib.parse.urlparse(endpoint.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("WEB_SEARCH_MCP_URL must be an HTTP(S) URL.")
        self.client = JsonApiClient(
            provider="web_search_mcp",
            base_url=urllib.parse.urlunparse(
                (parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", "")
            ),
            headers={},
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            opener=opener,
            sleep=sleep,
        )
        self._next_id = 0

    def call(
        self,
        operation: str,
        arguments: Mapping[str, object],
    ) -> ProviderResponse:
        if operation != TAVILY_SEARCH:
            raise ValueError(f"Unsupported web search operation: {operation}")
        query = text(arguments.get("query"))
        if not query:
            raise ValueError("web search requires query.")
        self._next_id += 1
        initialization = self._request(
            operation,
            {
                "jsonrpc": "2.0",
                "id": self._next_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "xlab-paper-collect", "version": "2.2.0"},
                },
            },
        )
        if not isinstance(initialization.payload, dict):
            raise self._protocol_error(operation, initialization, "initialize returned a non-object")
        self._validate_response(initialization, self._next_id, operation)
        result = initialization.payload.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("serverInfo"), dict):
            raise self._protocol_error(operation, initialization, "initialize omitted serverInfo")
        self.client.request(
            operation=operation,
            method="POST",
            path="",
            body={"jsonrpc": "2.0", "method": "notifications/initialized"},
            expect_response=False,
        )
        self._next_id += 1
        response = self._request(
            operation,
            {
                "jsonrpc": "2.0",
                "id": self._next_id,
                "method": "tools/call",
                "params": {"name": "search", "arguments": self._search_arguments(arguments)},
            },
        )
        payload = response.payload
        if not isinstance(payload, dict):
            raise self._protocol_error(operation, response, "tools/call returned a non-object")
        self._validate_response(response, self._next_id, operation)
        result = payload.get("result")
        if not isinstance(result, dict):
            raise self._protocol_error(operation, response, "tools/call omitted result")
        if result.get("isError") is True:
            raise ProviderRequestError(
                f"web_search_mcp tool failed: {self._redact(json.dumps(result, ensure_ascii=False))}",
                provider="web_search_mcp", operation=operation,
                status_code=response.status_code, attempts=response.attempts, retryable=True,
            )
        return ProviderResponse(
            payload={"results": self._normalize_results(result)},
            status_code=response.status_code,
            attempts=response.attempts,
            request_id=response.request_id,
        )

    def _request(self, operation: str, body: Mapping[str, object]) -> ProviderResponse:
        response = self.client.request(operation=operation, method="POST", path="", body=body)
        payload = response.payload
        if isinstance(payload, dict) and payload.get("error") is not None:
            raise self._protocol_error(operation, response, self._redact(json.dumps(payload["error"], ensure_ascii=False)))
        return response

    @staticmethod
    def _validate_response(response: ProviderResponse, request_id: int, operation: str) -> None:
        payload = response.payload
        if (
            not isinstance(payload, dict)
            or payload.get("jsonrpc") != "2.0"
            or payload.get("id") != request_id
        ):
            raise ProviderRequestError(
                f"web_search_mcp protocol error: invalid JSON-RPC response for {operation}",
                provider="web_search_mcp",
                operation=operation,
                status_code=response.status_code,
                attempts=response.attempts,
                retryable=False,
            )

    def _protocol_error(
        self,
        operation: str,
        response: ProviderResponse,
        detail: str,
    ) -> ProviderRequestError:
        return ProviderRequestError(
            f"web_search_mcp protocol error: {detail}",
            provider="web_search_mcp",
            operation=operation,
            status_code=response.status_code,
            attempts=response.attempts,
            retryable=False,
        )

    @staticmethod
    def _search_arguments(arguments: Mapping[str, object]) -> dict[str, object]:
        allowed = {"query", "search_context_size", "allowed_domains", "user_location", "max_sources"}
        result = {key: value for key, value in arguments.items() if key in allowed}
        if "allowed_domains" not in result and "include_domains" in arguments:
            result["allowed_domains"] = arguments["include_domains"]
        if "search_context_size" not in result and arguments.get("search_depth") == "advanced":
            result["search_context_size"] = "high"
        if "max_sources" not in result and "max_results" in arguments:
            result["max_sources"] = min(50, max(1, integer(arguments.get("max_results")) or 20))
        return result

    @staticmethod
    def _normalize_results(result: Mapping[str, object]) -> list[JsonObject]:
        content = result.get("structuredContent")
        if not isinstance(content, dict):
            content = result.get("structured_content")
        if not isinstance(content, dict):
            for item in as_list(result.get("content")):
                item_map = item if isinstance(item, dict) else {}
                if item_map.get("type") == "text":
                    try:
                        decoded = json.loads(text(item_map.get("text")))
                    except json.JSONDecodeError:
                        continue
                    if isinstance(decoded, dict):
                        content = decoded
                        break
        raw_results = as_list(content.get("results")) if isinstance(content, dict) else []
        normalized: list[JsonObject] = []
        for raw in raw_results:
            item = raw if isinstance(raw, dict) else {}
            url = text(item.get("url") or item.get("link"))
            title = text(item.get("title") or item.get("name"))
            if not url or not title:
                continue
            entry: JsonObject = {"title": title, "url": url}
            snippet = item.get("content") or item.get("snippet") or item.get("description")
            if snippet:
                entry["content"] = text(snippet)
            if isinstance(item.get("score"), (int, float)):
                entry["score"] = item["score"]
            normalized.append(entry)
        return normalized

    def _redact(self, value: str) -> str:
        return value[:2000]


class SemanticScholarClient:
    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 60.0,
        max_attempts: int = 4,
        min_interval_seconds: float = 1.0,
        graph_base_url: str = "https://api.semanticscholar.org/graph/v1",
        recommendations_base_url: str = "https://api.semanticscholar.org/recommendations/v1",
        opener: Callable[..., object] = urllib.request.urlopen,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not api_key.strip():
            raise ValueError("SEMANTIC_SCHOLAR_API_KEY is required.")
        rate_limiter = RequestRateLimiter(
            min_interval_seconds,
            sleep=sleep,
            monotonic=monotonic,
        )
        common = {
            "headers": {"x-api-key": api_key.strip()},
            "timeout_seconds": timeout_seconds,
            "max_attempts": max_attempts,
            "min_interval_seconds": min_interval_seconds,
            "opener": opener,
            "sleep": sleep,
            "monotonic": monotonic,
            "rate_limiter": rate_limiter,
        }
        self.graph = JsonApiClient(
            provider="semantic_scholar",
            base_url=graph_base_url,
            **common,
        )
        self.recommendations = JsonApiClient(
            provider="semantic_scholar",
            base_url=recommendations_base_url,
            **common,
        )

    def call(
        self,
        operation: str,
        arguments: Mapping[str, object],
    ) -> ProviderResponse:
        if operation == S2_SEARCH:
            query = {
                "query": arguments.get("query"),
                "limit": min(100, max(1, integer(arguments.get("limit")) or 100)),
                "offset": max(0, integer(arguments.get("offset")) or 0),
                "year": arguments.get("year"),
                "fields": self._fields(arguments, PAPER_FIELDS),
            }
            return self.graph.request(
                operation=operation,
                method="GET",
                path="/paper/search",
                query=query,
            )

        if operation in {S2_GET_PAPER, S2_CITATIONS, S2_REFERENCES}:
            paper_id = text(
                arguments.get("paperId") or arguments.get("paper_id")
            )
            if not paper_id:
                raise ValueError(f"{operation} requires paperId.")
            encoded_id = urllib.parse.quote(paper_id, safe="")
            suffix = {
                S2_GET_PAPER: "",
                S2_CITATIONS: "/citations",
                S2_REFERENCES: "/references",
            }[operation]
            allowed_fields = (
                PAPER_FIELDS if operation == S2_GET_PAPER else RELATION_FIELDS
            )
            query: JsonObject = {
                "fields": self._fields(arguments, allowed_fields)
            }
            if operation != S2_GET_PAPER:
                query.update(
                    {
                        "limit": min(
                            1000,
                            max(1, integer(arguments.get("limit")) or 200),
                        ),
                        "offset": max(0, integer(arguments.get("offset")) or 0),
                    }
                )
            return self.graph.request(
                operation=operation,
                method="GET",
                path=f"/paper/{encoded_id}{suffix}",
                query=query,
            )

        if operation == S2_RECOMMENDATIONS:
            positive = [
                text(value)
                for value in as_list(arguments.get("positivePaperIds"))
                if text(value)
            ]
            if not positive:
                raise ValueError(
                    "semantic_scholar.get_recommendations requires positivePaperIds."
                )
            return self.recommendations.request(
                operation=operation,
                method="POST",
                path="/papers/",
                query={
                    "limit": min(
                        500,
                        max(1, integer(arguments.get("limit")) or 100),
                    ),
                    "fields": self._fields(
                        arguments,
                        RECOMMENDATION_FIELDS,
                    ),
                },
                body={
                    "positivePaperIds": positive,
                    "negativePaperIds": [
                        text(value)
                        for value in as_list(arguments.get("negativePaperIds"))
                        if text(value)
                    ],
                },
            )

        raise ValueError(f"Unsupported Semantic Scholar operation: {operation}")

    @staticmethod
    def _fields(
        arguments: Mapping[str, object],
        allowed: list[str],
    ) -> list[str]:
        requested = {
            text(field)
            for field in as_list(arguments.get("fields"))
            if text(field)
        }
        if not requested:
            return list(allowed)
        selected = [field for field in allowed if field in requested]
        return selected or list(allowed)
