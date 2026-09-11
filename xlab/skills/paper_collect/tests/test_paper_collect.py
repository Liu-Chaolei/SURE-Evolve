from __future__ import annotations

import io
import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from unittest.mock import patch


SKILL_DIR = Path(__file__).resolve().parents[1]
SCRIPT = SKILL_DIR / "scripts" / "paper_collect.py"
sys.path.insert(0, str(SKILL_DIR / "scripts"))

from paper_collect_lib.common import (
    artifact_paths,
    atomic_write_json,
    atomic_write_jsonl,
    read_json,
    read_jsonl,
    run_lock,
)
from paper_collect_lib.download import (
    _CheckedHTTPConnection,
    _CheckedHTTPSConnection,
    _download_url,
    _public_host,
    _public_peer,
    download_run,
    validate_pdf,
)
from paper_collect_lib.pipeline import ProviderRunner, collect_run
from paper_collect_lib.providers import (
    PAPER_FIELDS,
    S2_CITATIONS,
    S2_GET_PAPER,
    S2_RECOMMENDATIONS,
    S2_REFERENCES,
    S2_SEARCH,
    TAVILY_SEARCH,
    ProviderRequestError,
    ProviderResponse,
    SemanticScholarClient,
    McpWebSearchClient,
)


def provider_record(
    provider: str,
    operation: str,
    input_value: dict[str, object],
    payload: object,
) -> dict[str, object]:
    return {
        "captured_at": "2026-07-19T00:00:00Z",
        "provider": provider,
        "operation": operation,
        "input": input_value,
        "is_error": False,
        "http_status": 200,
        "attempts": 1,
        "payload": payload,
    }


class FakeMcpClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def call(
        self,
        operation: str,
        arguments: dict[str, object],
    ) -> ProviderResponse:
        self.calls.append((operation, dict(arguments)))
        return ProviderResponse(
            payload={
                "results": [
                    {
                        "title": "Graph Neural Networks: A Review",
                        "url": "https://doi.org/10.1000/gnn.review",
                        "content": "A review of graph neural network methods.",
                        "score": 0.95,
                    }
                ]
            },
            status_code=200,
            attempts=1,
            request_id="tavily-test",
        )


class FakeSemanticScholarClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def call(
        self,
        operation: str,
        arguments: dict[str, object],
    ) -> ProviderResponse:
        self.calls.append((operation, dict(arguments)))
        if operation == S2_SEARCH:
            payload: object = {
                "data": [
                    {
                        "paperId": "S2-REVIEW",
                        "title": "Graph Neural Networks: A Review",
                        "authors": [{"authorId": "A1", "name": "Ada Author"}],
                        "year": 2021,
                        "abstract": "A review of graph neural network methods.",
                        "citationCount": 200,
                        "externalIds": {"DOI": "10.1000/gnn.review"},
                    },
                    {
                        "paperId": "S2-GCN",
                        "title": "Graph Convolutional Networks",
                        "authors": [{"authorId": "A2", "name": "Bob Author"}],
                        "year": 2017,
                        "abstract": "Graph convolution for node classification.",
                        "citationCount": 5000,
                        "externalIds": {"DOI": "10.1000/gcn"},
                    },
                ]
            }
        elif operation == S2_REFERENCES:
            payload = {
                "data": [
                    {
                        "citedPaper": {
                            "paperId": "S2-MPNN",
                            "title": "Message Passing Neural Networks",
                            "authors": [{"authorId": "A3", "name": "Carol Author"}],
                            "year": 2017,
                            "abstract": "Message passing for graph neural networks.",
                            "citationCount": 1500,
                            "externalIds": {"DOI": "10.1000/mpnn"},
                        }
                    }
                ]
            }
        elif operation == S2_CITATIONS:
            payload = {"data": []}
        elif operation == S2_RECOMMENDATIONS:
            payload = {"recommendedPapers": []}
        elif operation == S2_GET_PAPER:
            payload = {
                "paperId": arguments["paperId"],
                "title": "Resolved Graph Neural Network Paper",
                "authors": [{"authorId": "A4", "name": "Dana Author"}],
                "year": 2020,
                "abstract": "Resolved metadata.",
            }
        else:
            raise AssertionError(f"Unexpected operation: {operation}")
        return ProviderResponse(
            payload=payload,
            status_code=200,
            attempts=1,
            request_id="s2-test",
        )


class JsonResponse:
    def __init__(self, payload: object, status: int = 200) -> None:
        self.payload = json.dumps(payload).encode()
        self.status = status
        self.headers: dict[str, str] = {"x-request-id": "request-test"}

    def __enter__(self) -> JsonResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def read(self) -> bytes:
        return self.payload


class PaperCollectTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> dict[str, object]:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_quoted_invocation_arguments_are_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            output = self.run_cli(
                "init",
                "--run-dir",
                str(run_dir),
                "--run-id",
                "quoted-arguments",
                "--arguments",
                (
                    '"graph neural networks" --target-papers 200 '
                    '--max-papers 600 --facet "graph transformers" '
                    '--facet "message passing" --download-workers 6 '
                    '--download-per-host 2'
                ),
            )
            request = read_json(run_dir / "artifacts" / "request.json", {})
            self.assertEqual(output["query"], "graph neural networks")
            self.assertEqual(output["target_papers"], 200)
            self.assertEqual(output["max_papers"], 600)
            self.assertEqual(request["download_workers"], 6)
            self.assertEqual(request["download_per_host"], 2)
            self.assertNotIn("min_papers", request)
            self.assertEqual(
                request["facets"],
                ["graph transformers", "message passing"],
            )

    def test_invalid_invocation_arguments_are_rejected(self) -> None:
        invalid_arguments = [
            "graph neural networks --target-papers 200",
            '"graph neural networks" --target-papers 700 --max-papers 600',
            (
                '"graph neural networks" --download-workers 2 '
                "--download-per-host 3"
            ),
            '"graph neural networks" --unknown-option value',
        ]
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT),
                        "init",
                        "--run-dir",
                        directory,
                        "--arguments",
                        arguments,
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('"status": "failed"', result.stderr)

    def test_smoke_builds_deduplicated_graph_ready_collection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            output = self.run_cli(
                "smoke",
                "--run-dir",
                str(run_dir),
                "--run-id",
                "test-smoke",
            )
            self.assertTrue(output["audit"]["passed"])
            collection = read_json(
                run_dir / "artifacts" / "papers.manifest.json",
                {},
            )
            self.assertEqual(collection["collected_count"], 2)
            self.assertEqual(collection["edge_count"], 1)
            self.assertEqual(
                {paper["download"]["status"] for paper in collection["papers"]},
                {"unavailable"},
            )

    def test_low_core_ratio_is_a_non_blocking_warning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self.run_cli(
                "smoke",
                "--run-dir",
                str(run_dir),
                "--run-id",
                "low-core-ratio",
            )
            paths = artifact_paths(run_dir)
            collection = read_json(paths["collection"], {})
            for paper in collection["papers"]:
                paper["relevance"]["tier"] = "related"
            atomic_write_json(paths["collection"], collection)

            audit = self.run_cli(
                "audit",
                "--run-dir",
                str(run_dir),
                "--run-id",
                "low-core-ratio",
            )
            report = read_json(paths["report"], {})
            manifest = read_json(paths["manifest"], {})
            warning = "Core-paper ratio is 0.000; minimum is 0.150."

            self.assertTrue(audit["passed"])
            self.assertTrue(report["passed"])
            self.assertFalse(report["checks"]["minimum_core_ratio"]["passed"])
            self.assertEqual(report["metrics"]["core_ratio"], 0.0)
            self.assertNotIn(warning, report["blocking_errors"])
            self.assertIn(warning, report["warnings"])
            self.assertEqual(manifest["status"], "success")
            self.assertTrue(manifest["validation"]["passed"])
            self.assertIn(warning, manifest["validation"]["warnings"])

    def test_missing_abstract_is_partial_not_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self.run_cli(
                "init",
                "--run-dir",
                str(run_dir),
                "--run-id",
                "missing-abstract",
                "--query",
                "graph neural networks",
                "--target-papers",
                "1",
                "--max-papers",
                "10",
            )
            paths = artifact_paths(run_dir)
            atomic_write_jsonl(
                paths["provider_results"],
                [
                    provider_record(
                        "tavily",
                        TAVILY_SEARCH,
                        {"query": "graph neural networks survey", "max_results": 20},
                        {
                            "results": [
                                {
                                    "title": "Graph Neural Network Systems",
                                    "url": "https://doi.org/10.1000/gnn.systems",
                                    "content": "Graph neural network systems.",
                                    "score": 0.9,
                                }
                            ]
                        },
                    ),
                    provider_record(
                        "semantic_scholar",
                        S2_SEARCH,
                        {"query": "graph neural networks", "limit": 100},
                        {
                            "data": [
                                {
                                    "paperId": "S2-MISSING-ABSTRACT",
                                    "title": "Graph Neural Network Systems",
                                    "authors": [{"authorId": "A1", "name": "A. Author"}],
                                    "year": 2024,
                                    "abstract": None,
                                    "externalIds": {"DOI": "10.1000/gnn.systems"},
                                }
                            ]
                        },
                    ),
                ],
            )
            self.run_cli("ingest", "--run-dir", str(run_dir))
            self.run_cli("download", "--run-dir", str(run_dir))
            audit = self.run_cli(
                "audit",
                "--run-dir",
                str(run_dir),
                "--run-id",
                "missing-abstract",
            )
            collection = read_json(paths["collection"], {})
            self.assertEqual(collection["papers"][0]["metadata_status"], "partial")
            self.assertFalse(audit["passed"])
            self.assertTrue(
                any(
                    "at least one retained relation" in blocker
                    for blocker in audit["blocking_errors"]
                )
            )
            self.assertTrue(
                any("Abstract coverage" in warning for warning in audit["warnings"])
            )

    def test_exact_title_resolution_rejects_fuzzy_search_noise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self.run_cli(
                "init",
                "--run-dir",
                str(run_dir),
                "--query",
                "graph neural networks",
                "--target-papers",
                "2",
                "--max-papers",
                "10",
            )
            paths = artifact_paths(run_dir)
            expected_title = "Graph Neural Networks for Adaptive Navigation"
            atomic_write_jsonl(
                paths["provider_results"],
                [
                    provider_record(
                        "tavily",
                        TAVILY_SEARCH,
                        {"query": "graph neural networks survey"},
                        {
                            "results": [
                                {
                                    "title": expected_title,
                                    "url": "https://example.org/gnn-navigation",
                                    "content": "Graph neural networks for navigation.",
                                    "score": 0.9,
                                }
                            ]
                        },
                    ),
                    provider_record(
                        "semantic_scholar",
                        S2_SEARCH,
                        {
                            "query": expected_title,
                            "limit": 3,
                            "expectedTitle": expected_title,
                            "matchMode": "exact_title",
                        },
                        {
                            "data": [
                                {
                                    "paperId": "S2-UNRELATED",
                                    "title": (
                                        "Road Continuity-Aware Network for Road "
                                        "Extraction from Remote Sensing Images"
                                    ),
                                    "authors": [{"name": "Noise Author"}],
                                    "year": 2025,
                                    "abstract": "Road extraction from satellite images.",
                                },
                                {
                                    "paperId": "S2-EXPECTED",
                                    "title": expected_title,
                                    "authors": [{"name": "Graph Author"}],
                                    "year": 2024,
                                    "abstract": "Graph neural networks for adaptive navigation.",
                                },
                            ]
                        },
                    ),
                ],
            )
            output = self.run_cli("ingest", "--run-dir", str(run_dir))
            papers = read_jsonl(paths["papers_jsonl"])
            self.assertEqual(output["collected_count"], 1)
            self.assertEqual([paper["title"] for paper in papers], [expected_title])

    def test_relation_and_recommendation_requests_omit_tldr(self) -> None:
        urls: list[str] = []

        def opener(request: object, timeout: float) -> object:
            self.assertIsInstance(request, urllib.request.Request)
            urls.append(request.full_url)
            if "/recommendations/" in request.full_url:
                return JsonResponse({"recommendedPapers": []})
            return JsonResponse({"data": []})

        client = SemanticScholarClient(
            "test-key",
            opener=opener,
            sleep=lambda _: None,
            min_interval_seconds=0,
        )
        client.call(
            S2_CITATIONS,
            {"paperId": "S2-PAPER", "limit": 10, "fields": PAPER_FIELDS},
        )
        client.call(
            S2_REFERENCES,
            {"paperId": "S2-PAPER", "limit": 10, "fields": PAPER_FIELDS},
        )
        client.call(
            S2_RECOMMENDATIONS,
            {
                "positivePaperIds": ["S2-PAPER"],
                "limit": 10,
                "fields": PAPER_FIELDS,
            },
        )
        self.assertEqual(len(urls), 3)
        for url in urls:
            fields = urllib.parse.parse_qs(
                urllib.parse.urlparse(url).query
            )["fields"][0].split(",")
            self.assertNotIn("tldr", fields)

    def test_terminal_provider_failure_is_not_retried_forever(self) -> None:
        class TerminalSemanticScholarClient:
            def __init__(self) -> None:
                self.calls = 0

            def call(
                self,
                operation: str,
                arguments: dict[str, object],
            ) -> ProviderResponse:
                self.calls += 1
                raise ProviderRequestError(
                    "unsupported request",
                    provider="semantic_scholar",
                    operation=operation,
                    status_code=400,
                    attempts=1,
                    retryable=False,
                )

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self.run_cli(
                "init",
                "--run-dir",
                str(run_dir),
                "--query",
                "graph neural networks",
                "--target-papers",
                "2",
                "--max-papers",
                "10",
            )
            semantic = TerminalSemanticScholarClient()
            action = {
                "provider": "semantic_scholar",
                "operation": S2_CITATIONS,
                "arguments": {"paperId": "S2-PAPER", "limit": 10},
            }
            ProviderRunner(
                run_dir,
                tavily=FakeMcpClient(),
                semantic_scholar=semantic,
            ).execute(action)
            ProviderRunner(
                run_dir,
                tavily=FakeMcpClient(),
                semantic_scholar=semantic,
            ).execute(action)
            self.assertEqual(semantic.calls, 1)

    def test_collect_calls_providers_and_resume_skips_successes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self.run_cli(
                "init",
                "--run-dir",
                str(run_dir),
                "--query",
                "graph neural networks",
                "--target-papers",
                "3",
                "--max-papers",
                "10",
                "--seed-limit",
                "1",
                "--max-s2-calls",
                "10",
            )
            tavily = FakeMcpClient()
            semantic = FakeSemanticScholarClient()
            first = collect_run(
                run_dir,
                tavily=tavily,
                semantic_scholar=semantic,
            )
            self.assertEqual(first["collected_count"], 3)
            self.assertEqual(first["stop_reason"], "target_reached")
            self.assertGreater(first["api_calls"], 0)
            self.assertIn(S2_REFERENCES, [operation for operation, _ in semantic.calls])
            self.assertEqual(read_jsonl(artifact_paths(run_dir)["failures"]), [])

            resumed_tavily = FakeMcpClient()
            resumed_semantic = FakeSemanticScholarClient()
            second = collect_run(
                run_dir,
                tavily=resumed_tavily,
                semantic_scholar=resumed_semantic,
            )
            self.assertEqual(second["api_calls"], 0)
            self.assertEqual(resumed_tavily.calls, [])
            self.assertEqual(resumed_semantic.calls, [])

    def test_mcp_maps_legacy_search_arguments(self) -> None:
        captured: list[dict[str, object]] = []

        def opener(request: object, timeout: float) -> object:
            body = json.loads(request.data.decode())
            if body["method"] == "initialize":
                result = {"protocolVersion": "2025-11-25", "serverInfo": {"name": "search"}}
            elif body["method"] == "notifications/initialized":
                return JsonResponse({})
            else:
                captured.append(body["params"]["arguments"])
                result = {"structuredContent": {"results": []}}
            return JsonResponse({"jsonrpc": "2.0", "id": body.get("id"), "result": result})

        client = McpWebSearchClient(opener=opener, sleep=lambda _: None)
        client.call(TAVILY_SEARCH, {
            "query": "test",
            "include_domains": ["arxiv.org"],
            "max_results": 7,
        })
        self.assertEqual(captured, [{
            "query": "test",
            "allowed_domains": ["arxiv.org"],
            "max_sources": 7,
        }])

    def test_mcp_rejects_json_rpc_and_tool_errors(self) -> None:
        def error_opener(request: object, timeout: float) -> object:
            body = json.loads(request.data.decode())
            if body["method"] == "notifications/initialized":
                return JsonResponse({})
            if body["method"] == "initialize":
                result = {"protocolVersion": "2025-11-25", "serverInfo": {"name": "search"}}
                return JsonResponse({"jsonrpc": "2.0", "id": body["id"], "result": result})
            return JsonResponse({"jsonrpc": "2.0", "id": body["id"], "error": {"message": "bad search"}})

        client = McpWebSearchClient(opener=error_opener, sleep=lambda _: None)
        with self.assertRaises(ProviderRequestError):
            client.call(TAVILY_SEARCH, {"query": "test"})

        def tool_error_opener(request: object, timeout: float) -> object:
            body = json.loads(request.data.decode())
            if body["method"] == "notifications/initialized":
                return JsonResponse({})
            if body["method"] == "initialize":
                result = {"protocolVersion": "2025-11-25", "serverInfo": {"name": "search"}}
            else:
                result = {"isError": True, "content": [{"type": "text", "text": "failed"}]}
            return JsonResponse({"jsonrpc": "2.0", "id": body.get("id"), "result": result})

        client = McpWebSearchClient(opener=tool_error_opener, sleep=lambda _: None)
        with self.assertRaises(ProviderRequestError):
            client.call(TAVILY_SEARCH, {"query": "test"})

    def test_provider_retries_429_and_fails_fast_on_authentication(self) -> None:
        calls = 0
        delays: list[float] = []

        def retrying_opener(request: object, timeout: float) -> object:
            nonlocal calls
            calls += 1
            body = json.loads(request.data.decode())
            if body["method"] == "notifications/initialized":
                return JsonResponse({})
            if calls == 2:
                raise urllib.error.HTTPError(
                    "http://127.0.0.1:17890/mcp",
                    429,
                    "rate limited",
                    {"Retry-After": "0"},
                    io.BytesIO(b'{"message":"slow down"}'),
                )
            if body["method"] == "initialize":
                result = {"protocolVersion": "2025-11-25", "serverInfo": {"name": "web-search", "version": "0.3.0"}}
            else:
                result = {"structuredContent": {"results": []}}
            return JsonResponse({"jsonrpc": "2.0", "id": body["id"], "result": result})

        client = McpWebSearchClient(
            opener=retrying_opener,
            sleep=delays.append,
        )
        response = client.call(TAVILY_SEARCH, {"query": "graph neural networks", "max_results": 20})
        self.assertEqual(response.attempts, 1)
        self.assertEqual(calls, 3)
        self.assertEqual(delays, [])


        def mcp_opener(request: object, timeout: float) -> object:
            body = json.loads(request.data.decode())
            if body["method"] == "initialize":
                return JsonResponse({"jsonrpc": "2.0", "id": body["id"], "result": {"protocolVersion": "2025-11-25", "serverInfo": {"name": "web-search", "version": "0.3.0"}}})
            if body["method"] == "notifications/initialized":
                return JsonResponse({})
            return JsonResponse({"jsonrpc": "2.0", "id": body["id"], "result": {"structuredContent": {"results": [{"title": "A result", "url": "https://example.org/a", "snippet": "text"}]}}})

        client = McpWebSearchClient(opener=mcp_opener, sleep=lambda _: None)
        response = client.call(TAVILY_SEARCH, {"query": "test", "max_results": 3})
        self.assertEqual(response.payload["results"][0]["url"], "https://example.org/a")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            html = root / "login.pdf"
            html.write_bytes(b"<html>login</html>" * 100)
            self.assertFalse(validate_pdf(html)["valid"])

            pdf = root / "paper.pdf"
            pdf.write_bytes(b"%PDF-1.7\n" + b"0" * 2048 + b"\n%%EOF\n")
            validation = validate_pdf(pdf)
            self.assertTrue(validation["valid"])
            self.assertEqual(len(validation["sha256"]), 64)

    def test_download_rejects_non_global_hosts_and_rebound_peers(self) -> None:
        class FakeSocket:
            def __init__(self, peer: str) -> None:
                self.peer = peer
                self.closed = False

            def getpeername(self) -> tuple[str, int]:
                return (self.peer, 443)

        class ReboundConnection(_CheckedHTTPConnection):
            def __init__(self) -> None:
                super().__init__("example.org")
                self.sock = None

            def close(self) -> None:
                self.sock = None

        with patch(
            "paper_collect_lib.download.socket.getaddrinfo",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("100.64.0.1", 0))],
        ):
            self.assertFalse(_public_host("shared.example"))

        self.assertFalse(_public_peer(FakeSocket("100.64.0.1")))
        self.assertTrue(_public_peer(FakeSocket("8.8.8.8")))

        connection = ReboundConnection()
        with (
            patch.object(_CheckedHTTPConnection.__mro__[1], "connect", lambda self: setattr(self, "sock", FakeSocket("100.64.0.1"))),
            self.assertRaisesRegex(OSError, "non-public peer"),
        ):
            connection.connect()
        self.assertIsNone(connection.sock)

    def test_loopback_proxy_peers_are_allowed_for_http_and_https(self) -> None:
        class FakeSocket:
            def __init__(self, peer: tuple[str, int]) -> None:
                self.peer = peer

            def getpeername(self) -> tuple[str, int]:
                return self.peer

        class HttpConnection(_CheckedHTTPConnection):
            def __init__(self) -> None:
                super().__init__("example.org")
                self.sock = None

        class HttpsConnection(_CheckedHTTPSConnection):
            def __init__(self) -> None:
                super().__init__("example.org")
                self.sock = None

        with (
            patch(
                "paper_collect_lib.download.urllib.request.getproxies",
                return_value={
                    "http": "http://127.0.0.1:10809",
                    "https": "http://127.0.0.1:10809",
                },
            ),
            patch(
                "paper_collect_lib.download.socket.getaddrinfo",
                return_value=[
                    (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 10809))
                ],
            ),
            patch.object(
                _CheckedHTTPConnection.__mro__[1],
                "connect",
                lambda self: setattr(
                    self, "sock", FakeSocket(("127.0.0.1", 10809))
                ),
            ),
            patch.object(
                _CheckedHTTPSConnection.__mro__[1],
                "connect",
                lambda self: setattr(
                    self, "sock", FakeSocket(("127.0.0.1", 10809))
                ),
            ),
        ):
            HttpConnection().connect()
            HttpsConnection().connect()

    def test_private_peers_are_rejected_without_a_matching_loopback_proxy(self) -> None:
        class FakeSocket:
            def getpeername(self) -> tuple[str, int]:
                return ("127.0.0.1", 10809)

        class Connection(_CheckedHTTPConnection):
            def __init__(self) -> None:
                super().__init__("example.org")
                self.sock = None

            def close(self) -> None:
                self.sock = None

        connection = Connection()
        with (
            patch(
                "paper_collect_lib.download.urllib.request.getproxies",
                return_value={},
            ),
            patch.object(
                _CheckedHTTPConnection.__mro__[1],
                "connect",
                lambda self: setattr(self, "sock", FakeSocket()),
            ),
            self.assertRaisesRegex(OSError, "non-public peer"),
        ):
            connection.connect()
        self.assertIsNone(connection.sock)

        connection = Connection()
        with (
            patch(
                "paper_collect_lib.download.urllib.request.getproxies",
                return_value={"http": "http://192.0.2.10:10809"},
            ),
            patch(
                "paper_collect_lib.download.socket.getaddrinfo",
                return_value=[
                    (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.0.2.10", 10809))
                ],
            ),
            patch.object(
                _CheckedHTTPConnection.__mro__[1],
                "connect",
                lambda self: setattr(self, "sock", FakeSocket()),
            ),
            self.assertRaisesRegex(OSError, "non-public peer"),
        ):
            connection.connect()
        self.assertIsNone(connection.sock)

    def test_transient_download_failure_preserves_partial_file(self) -> None:
        class FailingOpener:
            def open(self, request: object, timeout: float) -> object:
                raise urllib.error.URLError("temporary network failure")

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "paper.pdf"
            part_path = destination.with_suffix(".pdf.part")
            part_path.write_bytes(b"%PDF-1.7\npartial")
            with (
                patch(
                    "paper_collect_lib.download.validate_public_url",
                    return_value=None,
                ),
                patch(
                    "paper_collect_lib.download.urllib.request.build_opener",
                    return_value=FailingOpener(),
                ),
                patch("paper_collect_lib.download.time.sleep", return_value=None),
            ):
                result = _download_url(
                    url="https://example.org/paper.pdf",
                    destination=destination,
                    timeout_seconds=1,
                    max_bytes=1024 * 1024,
                    retries=1,
                )
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["attempts"], 2)
            self.assertEqual(result["part_path"], str(part_path))
            self.assertTrue(part_path.exists())

    def test_run_lock_rejects_a_concurrent_mutation_phase(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            with run_lock(run_dir, "collect"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Another paper_collect phase is active",
                ):
                    with run_lock(run_dir, "download"):
                        self.fail("A concurrent phase acquired the run lock.")

    def test_download_limits_per_host_and_checkpoints_each_paper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self.run_cli(
                "init",
                "--run-dir",
                str(run_dir),
                "--query",
                "graph neural networks",
                "--target-papers",
                "1",
                "--max-papers",
                "10",
                "--download-workers",
                "6",
                "--download-per-host",
                "2",
            )
            paths = artifact_paths(run_dir)
            papers = [
                {
                    "id": f"s2:{index}",
                    "title": f"Graph Paper {index}",
                    "pdf_candidates": [
                        {
                            "url": f"https://arxiv.org/pdf/{index}.pdf",
                            "source": "test",
                            "priority": 1,
                        }
                    ],
                    "download": {"status": "not_requested"},
                }
                for index in range(6)
            ]
            atomic_write_json(
                paths["collection"],
                {
                    "schema_version": "xlab.paper_set.v3",
                    "papers": papers,
                },
            )
            active = 0
            peak = 0
            active_lock = threading.Lock()

            def fake_download(**arguments: object) -> dict[str, object]:
                nonlocal active, peak
                with active_lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.02)
                with active_lock:
                    active -= 1
                return {
                    "status": "failed",
                    "path": None,
                    "url": arguments["url"],
                    "sha256": None,
                    "bytes": None,
                    "error": "test_failure",
                    "attempts": 1,
                }

            with (
                patch(
                    "paper_collect_lib.download.validate_pdf",
                    return_value={"valid": False},
                ),
                patch(
                    "paper_collect_lib.download._download_url",
                    side_effect=fake_download,
                ),
            ):
                result = download_run(run_dir)

            self.assertEqual(peak, 2)
            self.assertEqual(result["failed"], 6)
            self.assertEqual(len(read_jsonl(paths["downloads"])), 6)
            self.assertEqual(len(read_jsonl(paths["download_results"])), 6)

    def test_download_isolates_worker_exceptions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self.run_cli(
                "init",
                "--run-dir",
                str(run_dir),
                "--query",
                "graph neural networks",
                "--target-papers",
                "1",
                "--max-papers",
                "10",
            )
            paths = artifact_paths(run_dir)
            papers = [
                {
                    "id": f"s2:{index}",
                    "title": f"Graph Paper {index}",
                    "pdf_candidates": [],
                    "download": {"status": "not_requested"},
                }
                for index in range(2)
            ]
            atomic_write_json(
                paths["collection"],
                {
                    "schema_version": "xlab.paper_set.v3",
                    "papers": papers,
                },
            )

            def worker(**arguments: object) -> tuple[dict[str, object], list[dict[str, object]]]:
                paper = dict(arguments["paper"])
                if paper["id"] == "s2:0":
                    raise RuntimeError("worker exploded")
                paper["download"] = {
                    "status": "unavailable",
                    "path": None,
                    "url": None,
                    "sha256": None,
                    "bytes": None,
                    "error": None,
                }
                return paper, []

            with patch(
                "paper_collect_lib.download._download_paper",
                side_effect=worker,
            ):
                result = download_run(run_dir)

            self.assertEqual(result["failed"], 1)
            self.assertEqual(result["unavailable"], 1)
            self.assertEqual(len(read_jsonl(paths["download_results"])), 2)

    def test_semantic_scholar_endpoints_share_one_rate_limiter(self) -> None:
        clock = 0.0
        delays: list[float] = []

        def monotonic() -> float:
            return clock

        def sleep(delay: float) -> None:
            nonlocal clock
            delays.append(delay)
            clock += delay

        client = SemanticScholarClient(
            "test-key",
            opener=lambda request, timeout: JsonResponse({"data": []}),
            sleep=sleep,
            monotonic=monotonic,
            min_interval_seconds=1,
        )
        client.call(S2_SEARCH, {"query": "graph neural networks"})
        client.call(
            S2_RECOMMENDATIONS,
            {"positivePaperIds": ["S2-PAPER"]},
        )
        self.assertEqual(delays, [1.0])


if __name__ == "__main__":
    unittest.main()
