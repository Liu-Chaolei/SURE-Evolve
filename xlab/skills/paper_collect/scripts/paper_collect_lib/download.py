from __future__ import annotations

import hashlib
import http.client
import ipaddress
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

from .common import (
    JsonObject,
    append_jsonl,
    artifact_paths,
    as_list,
    as_mapping,
    atomic_write_json,
    atomic_write_jsonl,
    integer,
    normalize_title,
    number,
    read_json,
    read_jsonl,
    run_lock,
    text,
    utc_now,
)


_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


def _public_host(hostname: str) -> bool:
    try:
        addresses = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    for address in addresses:
        raw_ip = address[4][0]
        try:
            ip = ipaddress.ip_address(raw_ip)
        except ValueError:
            return False
        if not ip.is_global or ip.is_multicast:
            return False
    return bool(addresses)


def validate_public_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only HTTP and HTTPS PDF URLs are allowed.")
    if not parsed.hostname or not _public_host(parsed.hostname):
        raise ValueError("PDF URL host is missing, private, reserved, or unresolved.")
    if parsed.username or parsed.password:
        raise ValueError("PDF URLs containing credentials are not allowed.")


def _public_peer(sock: object) -> bool:
    try:
        peer = sock.getpeername()  # type: ignore[attr-defined]
    except OSError:
        return False
    if not isinstance(peer, tuple) or not peer:
        return False
    try:
        ip = ipaddress.ip_address(str(peer[0]))
    except ValueError:
        return False
    return ip.is_global and not ip.is_multicast


def _configured_loopback_proxy(
    connection: http.client.HTTPConnection,
) -> set[tuple[str, int]]:
    scheme = "https" if isinstance(connection, http.client.HTTPSConnection) else "http"
    proxies = urllib.request.getproxies()
    proxy_url = proxies.get(scheme) or proxies.get("all")
    if not proxy_url:
        return set()
    try:
        parsed = urllib.parse.urlsplit(proxy_url)
        hostname = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return set()
    if parsed.scheme not in {"http", "https"} or not hostname:
        return set()
    try:
        addresses = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return set()
    loopback = {
        (str(ipaddress.ip_address(address[4][0])), port)
        for address in addresses
        if isinstance(address[4], tuple)
        and address[4]
        and _is_loopback(address[4][0])
    }
    return loopback


def _is_loopback(raw_ip: str) -> bool:
    try:
        return ipaddress.ip_address(raw_ip).is_loopback
    except ValueError:
        return False


def _peer_is_configured_loopback_proxy(
    connection: http.client.HTTPConnection,
) -> bool:
    if not connection.sock:
        return False
    try:
        peer = connection.sock.getpeername()
    except OSError:
        return False
    if not isinstance(peer, tuple) or len(peer) < 2:
        return False
    try:
        peer_address = str(ipaddress.ip_address(str(peer[0])))
        peer_port = int(peer[1])
    except (TypeError, ValueError):
        return False
    return (peer_address, peer_port) in _configured_loopback_proxy(connection)


def _check_connection_peer(connection: http.client.HTTPConnection) -> None:
    if connection.sock and (_public_peer(connection.sock) or _peer_is_configured_loopback_proxy(connection)):
        return
    connection.close()
    raise OSError("PDF URL resolved to a non-public peer address.")


class _CheckedHTTPConnection(http.client.HTTPConnection):
    def connect(self) -> None:
        super().connect()
        _check_connection_peer(self)


class _CheckedHTTPSConnection(http.client.HTTPSConnection):
    def connect(self) -> None:
        super().connect()
        _check_connection_peer(self)


class _SafeHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, request: urllib.request.Request) -> object:
        return self.do_open(_CheckedHTTPConnection, request)


class _SafeHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, request: urllib.request.Request) -> object:
        return self.do_open(
            _CheckedHTTPSConnection,
            request,
            context=self._context,  # type: ignore[attr-defined]
        )


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: object,
        code: int,
        message: str,
        headers: object,
        new_url: str,
    ) -> urllib.request.Request | None:
        validate_public_url(new_url)
        return super().redirect_request(
            request,
            file_pointer,
            code,
            message,
            headers,
            new_url,
        )


def validate_pdf(path: Path, minimum_bytes: int = 1024) -> JsonObject:
    if not path.is_file():
        return {"valid": False, "error": "file_missing"}
    size = path.stat().st_size
    if size < minimum_bytes:
        return {"valid": False, "error": "file_too_small", "bytes": size}
    with path.open("rb") as handle:
        head = handle.read(8)
        handle.seek(max(0, size - 65536))
        tail = handle.read()
    if not head.startswith(b"%PDF-"):
        return {"valid": False, "error": "missing_pdf_header", "bytes": size}
    if b"%%EOF" not in tail:
        return {"valid": False, "error": "missing_pdf_eof", "bytes": size}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "valid": True,
        "bytes": size,
        "sha256": digest.hexdigest(),
    }


def _safe_filename(paper: Mapping[str, object]) -> str:
    paper_id = text(paper.get("id")) or "paper"
    prefix = hashlib.sha1(paper_id.encode("utf-8")).hexdigest()[:12]
    slug = re.sub(r"[^a-z0-9]+", "-", normalize_title(paper.get("title"))).strip("-")
    return f"{prefix}--{slug[:80] or 'untitled'}.pdf"


def _retry_delay(error: urllib.error.HTTPError, attempt: int) -> float:
    retry_after = error.headers.get("Retry-After") if error.headers else None
    if retry_after:
        try:
            return min(60.0, max(0.0, float(retry_after)))
        except ValueError:
            pass
    return min(30.0, 2.0**attempt)


def _download_url(
    *,
    url: str,
    destination: Path,
    timeout_seconds: float,
    max_bytes: int,
    retries: int,
) -> JsonObject:
    validate_public_url(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_suffix(f"{destination.suffix}.part")
    opener = urllib.request.build_opener(
        _SafeHTTPHandler(),
        _SafeHTTPSHandler(),
        _SafeRedirectHandler(),
    )
    last_error = "download_failed"
    attempts_made = 0
    discard_part = False

    for attempt in range(retries + 1):
        attempts_made = attempt + 1
        part_size = part_path.stat().st_size if part_path.exists() else 0
        headers = {
            "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.1",
            "User-Agent": "XLab-Paper-Collect/2.0",
        }
        if part_size:
            headers["Range"] = f"bytes={part_size}-"
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with opener.open(request, timeout=timeout_seconds) as response:
                status = response.getcode()
                content_type = response.headers.get_content_type().lower()
                content_length = integer(response.headers.get("Content-Length"))
                if status == 206 and part_size:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.match(r"bytes\s+(\d+)-", content_range, flags=re.IGNORECASE)
                    if not match or int(match.group(1)) != part_size:
                        raise ValueError("Server returned an invalid resume range.")
                    mode = "ab"
                    downloaded = part_size
                else:
                    mode = "wb"
                    downloaded = 0
                if content_length is not None and content_length + downloaded > max_bytes:
                    raise ValueError("PDF exceeds configured maximum size.")
                with part_path.open(mode) as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            raise ValueError("PDF exceeds configured maximum size.")
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            if content_type.startswith("text/") or content_type in {
                "application/json",
                "application/xhtml+xml",
            }:
                raise ValueError(f"Unexpected content type: {content_type}")
            validation = validate_pdf(part_path)
            if validation.get("valid") is not True:
                raise ValueError(f"Invalid PDF: {validation.get('error')}")
            os.replace(part_path, destination)
            return {
                "status": "downloaded",
                "path": str(destination),
                "url": url,
                "sha256": validation.get("sha256"),
                "bytes": validation.get("bytes"),
                "error": None,
                "attempts": attempt + 1,
            }
        except urllib.error.HTTPError as error:
            last_error = f"http_{error.code}"
            if error.code == 416 and part_path.exists():
                validation = validate_pdf(part_path)
                if validation.get("valid") is True:
                    os.replace(part_path, destination)
                    return {
                        "status": "downloaded",
                        "path": str(destination),
                        "url": url,
                        "sha256": validation.get("sha256"),
                        "bytes": validation.get("bytes"),
                        "error": None,
                        "attempts": attempt + 1,
                    }
                discard_part = True
            if error.code not in _RETRYABLE_STATUS or attempt >= retries:
                if error.code not in _RETRYABLE_STATUS:
                    discard_part = True
                break
            time.sleep(_retry_delay(error, attempt))
        except (
            OSError,
            TimeoutError,
            ValueError,
            urllib.error.URLError,
        ) as error:
            last_error = f"{type(error).__name__}: {error}"
            if isinstance(error, ValueError):
                discard_part = True
                break
            if attempt >= retries:
                break
            time.sleep(min(30.0, 2.0**attempt))
    if discard_part:
        part_path.unlink(missing_ok=True)
    return {
        "status": "failed",
        "path": None,
        "part_path": str(part_path) if part_path.exists() else None,
        "url": url,
        "sha256": None,
        "bytes": None,
        "error": last_error,
        "attempts": attempts_made,
    }


def _download_paper(
    *,
    paper: Mapping[str, object],
    pdf_dir: Path,
    timeout_seconds: float,
    max_bytes: int,
    retries: int,
    per_host_workers: int,
    host_locks: dict[str, threading.Semaphore],
    host_locks_guard: threading.Lock,
) -> tuple[JsonObject, list[JsonObject]]:
    result = dict(paper)
    candidates = sorted(
        (as_mapping(item) for item in as_list(paper.get("pdf_candidates"))),
        key=lambda item: integer(item.get("priority")) or 100,
    )
    if not candidates:
        result["download"] = {
            "status": "unavailable",
            "path": None,
            "url": None,
            "sha256": None,
            "bytes": None,
            "error": None,
        }
        return result, []

    destination = pdf_dir / _safe_filename(paper)
    existing = validate_pdf(destination)
    if existing.get("valid") is True:
        result["download"] = {
            "status": "existing",
            "path": destination.name,
            "url": None,
            "sha256": existing.get("sha256"),
            "bytes": existing.get("bytes"),
            "error": None,
        }
        return result, []

    attempts: list[JsonObject] = []
    for candidate in candidates:
        url = text(candidate.get("url"))
        parsed = urllib.parse.urlparse(url)
        hostname = parsed.hostname or ""
        with host_locks_guard:
            host_lock = host_locks.setdefault(
                hostname,
                threading.Semaphore(per_host_workers),
            )
        try:
            with host_lock:
                download = _download_url(
                    url=url,
                    destination=destination,
                    timeout_seconds=timeout_seconds,
                    max_bytes=max_bytes,
                    retries=retries,
                )
        except (OSError, ValueError) as error:
            download = {
                "status": "failed",
                "path": None,
                "url": url,
                "sha256": None,
                "bytes": None,
                "error": f"{type(error).__name__}: {error}",
                "attempts": 0,
            }
        attempts.append(
            {
                "timestamp": utc_now(),
                "paper_id": paper.get("id"),
                "candidate_source": candidate.get("source"),
                **download,
            }
        )
        if download.get("status") == "downloaded":
            result["download"] = {
                **download,
                "path": destination.name,
            }
            return result, attempts

    last_error = attempts[-1].get("error") if attempts else "no_candidate_url"
    result["download"] = {
        "status": "failed",
        "path": None,
        "url": None,
        "sha256": None,
        "bytes": None,
        "error": last_error,
    }
    return result, attempts


def _primary_host(paper: Mapping[str, object]) -> str:
    candidates = sorted(
        (as_mapping(item) for item in as_list(paper.get("pdf_candidates"))),
        key=lambda item: integer(item.get("priority")) or 100,
    )
    if not candidates:
        return ""
    return urllib.parse.urlparse(text(candidates[0].get("url"))).hostname or ""


def _interleaved_work(
    papers: list[JsonObject],
) -> list[tuple[int, JsonObject]]:
    queues: dict[str, deque[tuple[int, JsonObject]]] = {}
    for index, paper in enumerate(papers):
        queues.setdefault(_primary_host(paper), deque()).append((index, paper))
    work: list[tuple[int, JsonObject]] = []
    active_hosts = list(queues)
    while active_hosts:
        next_hosts: list[str] = []
        for hostname in active_hosts:
            queue = queues[hostname]
            work.append(queue.popleft())
            if queue:
                next_hosts.append(hostname)
        active_hosts = next_hosts
    return work


def _apply_download_checkpoints(
    papers: list[JsonObject],
    checkpoint_records: list[object],
) -> None:
    latest: dict[str, JsonObject] = {}
    for raw_record in checkpoint_records:
        record = as_mapping(raw_record)
        paper_id = text(record.get("paper_id"))
        download = as_mapping(record.get("download"))
        if paper_id and download:
            latest[paper_id] = download
    for paper in papers:
        download = latest.get(text(paper.get("id")))
        if download:
            paper["download"] = download


def _worker_failure(
    paper: Mapping[str, object],
    error: Exception,
) -> tuple[JsonObject, list[JsonObject]]:
    result = dict(paper)
    message = f"{type(error).__name__}: {error}"
    result["download"] = {
        "status": "failed",
        "path": None,
        "url": None,
        "sha256": None,
        "bytes": None,
        "error": message,
    }
    return result, [
        {
            "timestamp": utc_now(),
            "paper_id": paper.get("id"),
            "candidate_source": "worker_exception",
            "status": "failed",
            "path": None,
            "url": None,
            "sha256": None,
            "bytes": None,
            "error": message,
            "attempts": 0,
        }
    ]


def _download_run_unlocked(
    run_dir: Path,
    *,
    workers: int | None = None,
    per_host_workers: int | None = None,
    timeout_seconds: float | None = None,
    max_bytes: int | None = None,
    retries: int | None = None,
) -> JsonObject:
    paths = artifact_paths(run_dir)
    request = as_mapping(read_json(paths["request"], {}))
    collection = as_mapping(read_json(paths["collection"], {}))
    papers = [
        as_mapping(item)
        for item in as_list(collection.get("papers"))
        if as_mapping(item)
    ]
    _apply_download_checkpoints(
        papers,
        read_jsonl(paths["download_results"]),
    )
    configured_workers = (
        workers
        if workers is not None
        else integer(request.get("download_workers")) or 8
    )
    configured_per_host_workers = (
        per_host_workers
        if per_host_workers is not None
        else integer(request.get("download_per_host")) or 2
    )
    configured_timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else number(request.get("download_timeout_seconds")) or 60.0
    )
    requested_max_bytes = (
        max_bytes
        if max_bytes is not None
        else integer(request.get("max_pdf_bytes"))
    )
    configured_max_bytes = (
        requested_max_bytes
        if requested_max_bytes is not None
        else 100 * 1024 * 1024
    )
    requested_retries = (
        retries
        if retries is not None
        else integer(request.get("download_retries"))
    )
    configured_retries = (
        requested_retries
        if requested_retries is not None
        else 3
    )
    if not 1 <= configured_workers <= 16:
        raise ValueError("Download workers must be between 1 and 16.")
    if not 1 <= configured_per_host_workers <= configured_workers:
        raise ValueError(
            "Per-host download workers must be between 1 and the global worker count."
        )
    if configured_timeout <= 0:
        raise ValueError("Download timeout must be greater than zero.")
    if configured_max_bytes < 1024:
        raise ValueError("Maximum PDF size must be at least 1024 bytes.")
    if configured_retries < 0:
        raise ValueError("Download retries must not be negative.")
    paths["pdfs"].mkdir(parents=True, exist_ok=True)

    updated: list[JsonObject | None] = [None] * len(papers)
    attempt_record_count = 0
    host_locks: dict[str, threading.Semaphore] = {}
    host_locks_guard = threading.Lock()
    work = iter(_interleaved_work(papers))
    queue_limit = configured_workers * 2
    pending: dict[
        Future[tuple[JsonObject, list[JsonObject]]],
        tuple[int, JsonObject],
    ] = {}

    with ThreadPoolExecutor(max_workers=configured_workers) as executor:
        exhausted = False
        while pending or not exhausted:
            while len(pending) < queue_limit and not exhausted:
                try:
                    index, paper = next(work)
                except StopIteration:
                    exhausted = True
                    break
                future = executor.submit(
                    _download_paper,
                    paper=paper,
                    pdf_dir=paths["pdfs"],
                    timeout_seconds=configured_timeout,
                    max_bytes=configured_max_bytes,
                    retries=configured_retries,
                    per_host_workers=configured_per_host_workers,
                    host_locks=host_locks,
                    host_locks_guard=host_locks_guard,
                )
                pending[future] = (index, paper)
            if not pending:
                continue
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                index, source_paper = pending.pop(future)
                try:
                    paper, attempts = future.result()
                except Exception as error:
                    paper, attempts = _worker_failure(source_paper, error)
                updated[index] = paper
                for attempt in attempts:
                    append_jsonl(paths["downloads"], attempt)
                    attempt_record_count += 1
                append_jsonl(
                    paths["download_results"],
                    {
                        "timestamp": utc_now(),
                        "paper_id": paper.get("id"),
                        "download": paper.get("download"),
                    },
                )

    completed_papers = [
        paper if paper is not None else dict(papers[index])
        for index, paper in enumerate(updated)
    ]
    collection["papers"] = completed_papers
    collection["generated_at"] = utc_now()
    collection["download"] = {
        "downloaded": sum(
            text(as_mapping(paper.get("download")).get("status"))
            in {"downloaded", "existing"}
            for paper in completed_papers
        ),
        "failed": sum(
            text(as_mapping(paper.get("download")).get("status")) == "failed"
            for paper in completed_papers
        ),
        "unavailable": sum(
            text(as_mapping(paper.get("download")).get("status")) == "unavailable"
            for paper in completed_papers
        ),
        "attempt_records": attempt_record_count,
        "workers": configured_workers,
        "per_host_workers": configured_per_host_workers,
    }
    atomic_write_json(paths["collection"], collection)
    atomic_write_jsonl(paths["papers_jsonl"], completed_papers)
    return {
        "paper_count": len(completed_papers),
        **as_mapping(collection.get("download")),
        "collection_path": str(paths["collection"]),
        "pdf_dir": str(paths["pdfs"]),
        "checkpoint_path": str(paths["download_results"]),
    }


def download_run(
    run_dir: Path,
    *,
    workers: int | None = None,
    per_host_workers: int | None = None,
    timeout_seconds: float | None = None,
    max_bytes: int | None = None,
    retries: int | None = None,
) -> JsonObject:
    with run_lock(run_dir, "download"):
        return _download_run_unlocked(
            run_dir,
            workers=workers,
            per_host_workers=per_host_workers,
            timeout_seconds=timeout_seconds,
            max_bytes=max_bytes,
            retries=retries,
        )
