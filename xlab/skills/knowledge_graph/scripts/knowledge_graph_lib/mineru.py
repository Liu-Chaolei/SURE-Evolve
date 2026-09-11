from __future__ import annotations

import hashlib
import fcntl
import os
import shutil
import signal
import subprocess
import sys
import threading
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .common import (
    JsonObject,
    append_jsonl,
    artifact_paths,
    as_list,
    as_mapping,
    atomic_write_json,
    read_json,
    sha256_file,
    text,
    utc_now,
)
from .mineru_cache import cache_key, cached_bundle, parser_context, publish_bundle


PARSER_VERSION = "mineru-batched-v1"


def _stem(paper_id: str) -> str:
    return hashlib.sha256(paper_id.encode("utf-8")).hexdigest()[:24]


def _stage_pdf(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _batch_key(
    documents: list[JsonObject], *, backend: str, method: str
) -> str:
    source = "\x1e".join(
        sorted(
            f"{text(document.get('paper_id'))}\x1f{text(document.get('source_sha256'))}"
            for document in documents
        )
    )
    return hashlib.sha256(
        f"{PARSER_VERSION}\x1f{backend}\x1f{method}\x1f{source}".encode("utf-8")
    ).hexdigest()[:20]


def _run_batch(
    *,
    command: Path,
    input_dir: Path,
    output_dir: Path,
    backend: str,
    method: str,
    language: str,
    timeout: int,
    gpu: str | None,
    log_path: Path,
    device: str = "cpu",
) -> tuple[int, str | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    if device == "npu":
        environment["ASCEND_RT_VISIBLE_DEVICES"] = gpu or "0"
        environment["MINERU_DEVICE_MODE"] = "npu"
    elif gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        environment["MINERU_DEVICE_MODE"] = "cuda"
    elif backend == "pipeline":
        environment["MINERU_DEVICE_MODE"] = "cpu"
    environment.setdefault("MINERU_API_MAX_CONCURRENT_REQUESTS", "1")
    environment.setdefault("OMP_NUM_THREADS", "16")
    environment.setdefault("MKL_NUM_THREADS", "16")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    arguments = [
        str(command),
        "-p",
        str(input_dir),
        "-o",
        str(output_dir),
        "-b",
        backend,
        "-m",
        method,
        "-l",
        language,
    ]
    if device == "npu":
        arguments[:1] = [sys.executable, str(Path(__file__).parents[1] / "mineru_npu.py")]
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                arguments,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                return process.wait(timeout=timeout), None
            except BaseException:
                # Give the CLI's context managers a chance to stop its detached
                # local API and prediction workers before escalating shutdown.
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise
    except subprocess.TimeoutExpired:
        return 124, f"MinerU batch exceeded {timeout} seconds"
    except OSError as error:
        return 127, str(error)


def _choose_content_list(parent: Path, stem: str) -> Path | None:
    preferred = parent / f"{stem}_content_list.json"
    if preferred.is_file():
        return preferred
    versioned = parent / f"{stem}_content_list_v2.json"
    return versioned if versioned.is_file() else None


def _locate_bundle(root: Path, stem: str) -> dict[str, Path | None] | None:
    candidates = sorted(root.rglob(f"{stem}.md"))
    for markdown in candidates:
        parent = markdown.parent
        content_list = _choose_content_list(parent, stem)
        middle = parent / f"{stem}_middle.json"
        if content_list is None or not middle.is_file():
            continue
        try:
            content = read_json(content_list)
            layout = as_mapping(read_json(middle))
            if not markdown.read_text(encoding="utf-8").strip() or not content or not as_list(layout.get("pdf_info")):
                continue
            if not isinstance(content, (list, dict)):
                continue
        except (OSError, ValueError, UnicodeError):
            continue
        model_candidates = sorted(parent.glob(f"{stem}*model*.json"))
        images = parent / "images"
        return {
            "markdown": markdown.resolve(),
            "content_list": content_list.resolve(),
            "middle": middle.resolve(),
            "model": model_candidates[0].resolve() if model_candidates else None,
            "images": images.resolve() if images.is_dir() else None,
        }
    return None


def _cache_valid(
    document: JsonObject,
    *,
    backend: str,
    method: str,
    language: str,
) -> bool:
    context = as_mapping(document.get("mineru_context"))
    if (
        document.get("status") != "parsed"
        or context.get("parser_version") != PARSER_VERSION
        or context.get("backend") != backend
        or context.get("method") != method
        or context.get("language") != language
        or text(context.get("source_sha256")) != text(document.get("source_sha256"))
    ):
        return False
    for path_key, sha_key in (
        ("markdown_path", "markdown_sha256"),
        ("content_list_path", "content_list_sha256"),
        ("middle_json_path", "middle_json_sha256"),
    ):
        path = Path(text(document.get(path_key)))
        if not path.is_file() or sha256_file(path) != text(document.get(sha_key)):
            return False
    return True


def _recover_missing(
    *,
    documents: list[JsonObject],
    worker: int,
    branch: str,
    gpu: str | None,
    paths: dict[str, Path],
    command: Path,
    backend: str,
    method: str,
    language: str,
    timeout: int,
    device: str = "cpu",
) -> tuple[dict[str, dict[str, Path | None]], dict[str, str]]:
    if not documents:
        return {}, {}
    label = f"worker-{worker}-{branch}"
    staging = paths["mineru_staging"] / "recovery" / label
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for document in documents:
        paper_id = text(document.get("paper_id"))
        _stage_pdf(
            Path(text(document.get("source_pdf"))),
            staging / f"{_stem(paper_id)}.pdf",
        )
    output = paths["mineru"] / "recovery" / label
    if output.exists():
        shutil.rmtree(output)
    return_code, batch_error = _run_batch(
        command=command,
        input_dir=staging,
        output_dir=output,
        backend=backend,
        method=method,
        language=language,
        timeout=timeout,
        gpu=gpu,
        device=device,
        log_path=paths["logs"].parent / f"mineru-recovery-{label}.log",
    )
    bundles: dict[str, dict[str, Path | None]] = {}
    missing: list[JsonObject] = []
    for document in documents:
        paper_id = text(document.get("paper_id"))
        bundle = _locate_bundle(output, _stem(paper_id))
        if bundle is None:
            missing.append(document)
        else:
            bundles[paper_id] = bundle
    if not missing:
        return bundles, {}
    if len(documents) == 1:
        paper_id = text(documents[0].get("paper_id"))
        error = batch_error or (
            f"MinerU isolated retry exited {return_code} without Markdown, "
            "content_list, and middle JSON outputs"
        )
        return bundles, {paper_id: error}
    midpoint = max(len(missing) // 2, 1)
    errors: dict[str, str] = {}
    for suffix, subset in (
        ("a", missing[:midpoint]),
        ("b", missing[midpoint:]),
    ):
        recovered, failed = _recover_missing(
            documents=subset,
            worker=worker,
            branch=branch + suffix,
            gpu=gpu,
            device=device,
            paths=paths,
            command=command,
            backend=backend,
            method=method,
            language=language,
            timeout=timeout,
        )
        bundles.update(recovered)
        errors.update(failed)
    return bundles, errors


def parse_with_mineru(
    run_dir: Path,
    document_manifest: JsonObject,
    resource_plan: JsonObject,
    *,
    method: str,
    language: str,
    timeout: int,
    batch_size: int = 24,
    cache_root: Path | None = None,
) -> JsonObject:
    paths = artifact_paths(run_dir)
    plan = as_mapping(resource_plan.get("mineru"))
    command = Path(text(plan.get("command")))
    backend = text(plan.get("backend"))
    device = text(plan.get("device")) or ("cuda" if plan.get("selected_gpus") else "cpu")
    workers = int(plan.get("workers") or 1)
    devices = [text(value) for value in as_list(plan.get("selected_npus" if device == "npu" else "selected_gpus"))]
    if device != "cpu" and len(devices) < workers:
        raise ValueError("MinerU worker count exceeds the allocated devices.")
    if not 1 <= batch_size <= 256:
        raise ValueError("MinerU batch size must be between 1 and 256.")
    context = parser_context(command, device, backend, method, language)
    cache_root = cache_root or paths["artifacts"] / ".mineru-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    documents = [as_mapping(value) for value in as_list(document_manifest.get("documents"))]
    pending: dict[str, list[JsonObject]] = {}
    reused = 0
    for document in documents:
        if not text(document.get("source_pdf")):
            continue
        source = Path(text(document.get("source_pdf")))
        if not source.is_file() or sha256_file(source) != document.get("source_sha256"):
            raise ValueError(f"Source PDF is missing or changed: {document.get('paper_id')}")
        previous_context = dict(as_mapping(document.get("mineru_context")))
        previous_context.pop("source_sha256", None)
        if previous_context == context and _cache_valid(document, backend=backend, method=method, language=language):
            reused += 1
            continue
        document.update(status="pending", blocker=None, failure_phase=None)
        key = cache_key(text(document.get("source_sha256")), context)
        pending.setdefault(key, []).append(document)
    state_lock = threading.Lock()

    def save() -> JsonObject:
        summary = {
            "total": len(documents),
            "downloaded_pdfs": sum(bool(text(doc.get("source_pdf"))) for doc in documents),
            "parsed": sum(doc.get("status") == "parsed" for doc in documents),
            "metadata_only": sum(doc.get("status") == "metadata_only" for doc in documents),
            "pending": sum(doc.get("status") == "pending" for doc in documents),
            "failed": sum(doc.get("status") == "failed" for doc in documents),
            "reused": reused,
        }
        document_manifest.update(documents=documents, generated_at=utc_now(), summary=summary, mineru=context)
        atomic_write_json(paths["documents"], document_manifest)
        manifest: JsonObject = {
            "schema_version": "xlab.mineru_manifest.v1",
            "generated_at": utc_now(),
            "documents_path": str(paths["documents"]),
            "backend": backend, "device": device, "method": method, "language": language,
            "records": [{key: doc.get(key) for key in ("paper_id", "status", "markdown_path", "content_list_path", "middle_json_path", "blocker")} for doc in documents],
            "summary": summary,
        }
        atomic_write_json(paths["mineru_manifest"], manifest)
        return manifest

    def accept(key: str, bundle: dict[str, Path | None], was_cached: bool) -> None:
        nonlocal reused
        updates: JsonObject = {"status": "parsed", "parser": "mineru", "blocker": None, "failure_phase": None}
        for kind, path_key, hash_key in (
            ("markdown", "markdown_path", "markdown_sha256"),
            ("content_list", "content_list_path", "content_list_sha256"),
            ("middle", "middle_json_path", "middle_json_sha256"),
        ):
            path = bundle[kind]
            if path is None:
                raise ValueError("Incomplete MinerU bundle")
            updates[path_key] = str(path)
            updates[hash_key] = sha256_file(path)
        updates["model_json_path"] = str(bundle["model"]) if bundle.get("model") else None
        updates["images_dir"] = str(bundle["images"]) if bundle.get("images") else None
        with state_lock:
            for document in pending[key]:
                document.update(updates)
                document["mineru_context"] = {**context, "source_sha256": document.get("source_sha256")}
            reused += len(pending[key]) if was_cached else len(pending[key]) - 1

    def worker(index: int, keys: list[str]) -> None:
        gpu = devices[index] if devices else None
        for offset in range(0, len(keys), batch_size):
            group_keys = keys[offset:offset + batch_size]
            # Stable ordering prevents deadlock when independent runs share PDFs.
            with ExitStack() as locks:
                missing_keys: list[str] = []
                for key in sorted(group_keys):
                    handle = locks.enter_context((cache_root / f"{key}.lock").open("a+"))
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    bundle = cached_bundle(cache_root, key)
                    if bundle is not None:
                        accept(key, bundle, True)
                    else:
                        missing_keys.append(key)
                if not missing_keys:
                    with state_lock:
                        save()
                    continue
                group = [pending[key][0] for key in missing_keys]
                batch = _batch_key(group, backend=backend, method=method)
                label = f"{batch}-{index}"
                staging = paths["mineru_staging"] / label
                output = paths["mineru"] / f"batch-{label}"
                staging.mkdir(parents=True, exist_ok=True)
                for document in group:
                    _stage_pdf(Path(text(document.get("source_pdf"))), staging / f"{_stem(text(document.get('paper_id')))}.pdf")
                # Output has its own context hash: stale partial outputs from another
                # parser configuration cannot be accepted on an interrupted rerun.
                output = output / cache_key("batch", context)[:16]
                return_code, batch_error = _run_batch(command=command, input_dir=staging, output_dir=output, backend=backend, method=method, language=language, timeout=timeout, gpu=gpu, device=device, log_path=paths["logs"].parent / f"mineru-{label}.log")
                missing = [doc for doc in group if _locate_bundle(output, _stem(text(doc.get("paper_id")))) is None]
                recovered, errors = _recover_missing(documents=missing, worker=index, branch=batch, gpu=gpu, device=device, paths=paths, command=command, backend=backend, method=method, language=language, timeout=timeout) if missing else ({}, {})
                for key, document in zip(missing_keys, group):
                    paper_id = text(document.get("paper_id"))
                    bundle = _locate_bundle(output, _stem(paper_id)) or recovered.get(paper_id)
                    if bundle is not None:
                        accept(key, publish_bundle(cache_root, key, bundle), False)
                    else:
                        error = errors.get(paper_id) or batch_error or f"MinerU exited {return_code} without a complete bundle"
                        with state_lock:
                            for duplicate in pending[key]:
                                duplicate.update(status="failed", blocker=error[:2000], failure_phase="mineru")
                                append_jsonl(paths["failures"], {"phase": "mineru", "paper_id": duplicate.get("paper_id"), "error": error[:2000], "recorded_at": utc_now()})
                with state_lock:
                    save()
                    append_jsonl(paths["logs"], {"phase": "mineru", "batch": batch, "worker": index, "papers": len(group), "recorded_at": utc_now(), "summary": document_manifest.get("summary")})

    save()
    keys = list(pending)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, index, keys[index::workers]) for index in range(min(workers, len(keys)))]
        for future in as_completed(futures):
            future.result()
    return save()
