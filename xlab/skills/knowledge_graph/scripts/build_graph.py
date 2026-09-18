#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from knowledge_graph_lib.audit import audit_run
from knowledge_graph_lib.common import (
    JsonObject,
    artifact_paths,
    as_mapping,
    atomic_write_json,
    read_json,
    run_lock,
    sha256_file,
    text,
    utc_now,
)
from knowledge_graph_lib.extract import extract_documents, migrate_endpoint
from knowledge_graph_lib.graph import build_graph
from knowledge_graph_lib.inputs import locate_collection, prepare_input_documents
from knowledge_graph_lib.llm import LlmClient
from knowledge_graph_lib.mineru import parse_with_mineru
from knowledge_graph_lib.resources import (
    build_resource_plan,
    public_api_url,
    require_llm_environment,
)
from knowledge_graph_lib.structure import structure_documents


SKILL_VERSION = "3.0.0"


def _resolved_path(value: str, cwd: Path) -> Path:
    candidate = Path(value).expanduser()
    return (cwd / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def _add_request_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mineru-backend",
        default="auto",
        choices=["auto", "pipeline", "vlm-engine", "hybrid-engine"],
    )
    parser.add_argument("--mineru-method", default="auto", choices=["auto", "txt", "ocr"])
    parser.add_argument("--mineru-language", default="en")
    parser.add_argument("--mineru-device", default="auto", choices=["auto", "cpu", "cuda", "npu"])
    parser.add_argument("--mineru-batch-size", type=int, default=24)
    parser.add_argument("--gpu", action="append", default=[])
    parser.add_argument("--mineru-workers", type=int, default=8)
    parser.add_argument("--mineru-timeout", type=int, default=21_600)
    parser.add_argument("--min-gpu-free-gb", type=float, default=24.0)
    parser.add_argument("--max-gpu-utilization", type=int, default=20)
    parser.add_argument("--structure-workers", type=int, default=8)
    parser.add_argument("--llm-workers", type=int, default=2)
    parser.add_argument("--llm-timeout", type=int, default=180)
    parser.add_argument("--llm-retries", type=int, default=4)
    parser.add_argument("--llm-validation-retries", type=int, default=2)
    parser.add_argument("--llm-rpm", type=int, default=0)
    parser.add_argument("--llm-max-input-chars", type=int, default=120_000)
    parser.add_argument("--llm-max-output-tokens", type=int, default=6_000)
    parser.add_argument("--llm-disable-thinking", action="store_true")
    parser.add_argument("--min-papers", type=int, default=1)
    parser.add_argument("--min-nodes", type=int, default=1)
    parser.add_argument("--min-edges", type=int, default=1)
    parser.add_argument("--min-mineru-coverage", type=float, default=0.9)
    parser.add_argument("--min-structure-success", type=float, default=0.98)
    parser.add_argument("--min-extraction-success", type=float, default=0.9)
    parser.add_argument("--min-core-coverage", type=float, default=0.9)
    parser.add_argument("--min-evidence-coverage", type=float, default=0.95)
    parser.add_argument("--desired-baseline-coverage", type=float, default=0.2)
    parser.add_argument("--desired-dataset-coverage", type=float, default=0.2)
    parser.add_argument("--max-ambiguous-alias-ratio", type=float, default=0.2)


def _parse_invocation(arguments: str) -> dict[str, object]:
    try:
        tokens = shlex.split(arguments, posix=True)
    except ValueError as error:
        raise ValueError(f"Cannot parse knowledge_graph arguments: {error}") from error
    if not tokens:
        raise ValueError(
            'Provide a completed paper_collect directory, for example: '
            '/xlab build-knowledge-graph "/path/to/paper-collect-run".'
        )
    parser = argparse.ArgumentParser(prog="/xlab build-knowledge-graph", add_help=False)
    parser.add_argument("input_directory")
    _add_request_options(parser)
    try:
        return vars(parser.parse_args(tokens))
    except SystemExit as error:
        raise ValueError("Invalid knowledge_graph invocation.") from error


def _validate_request(request: JsonObject) -> None:
    for key, minimum, maximum in (
        ("mineru_workers", 1, 32),
        ("mineru_batch_size", 1, 256),
        ("structure_workers", 1, 64),
        ("llm_workers", 1, 128),
        ("llm_retries", 0, 10),
        ("llm_validation_retries", 0, 5),
        ("max_gpu_utilization", 0, 100),
    ):
        value = int(request[key])
        if value < minimum or value > maximum:
            raise ValueError(
                f"--{key.replace('_', '-')} must be between {minimum} and {maximum}."
            )
    if int(request["mineru_timeout"]) < 300:
        raise ValueError("--mineru-timeout must be at least 300 seconds.")
    if int(request["llm_timeout"]) < 10:
        raise ValueError("--llm-timeout must be at least 10 seconds.")
    if int(request["llm_rpm"]) < 0:
        raise ValueError("--llm-rpm cannot be negative.")
    if int(request["llm_max_input_chars"]) < 20_000:
        raise ValueError("--llm-max-input-chars must be at least 20000.")
    if int(request["llm_max_output_tokens"]) < 1_000:
        raise ValueError("--llm-max-output-tokens must be at least 1000.")
    if float(request["min_gpu_free_gb"]) < 0:
        raise ValueError("--min-gpu-free-gb cannot be negative.")
    for key in ("min_papers", "min_nodes", "min_edges"):
        if int(request[key]) < 1:
            raise ValueError(f"--{key.replace('_', '-')} must be at least 1.")
    for key in (
        "min_mineru_coverage",
        "min_structure_success",
        "min_extraction_success",
        "min_core_coverage",
        "min_evidence_coverage",
        "desired_baseline_coverage",
        "desired_dataset_coverage",
        "max_ambiguous_alias_ratio",
    ):
        value = float(request[key])
        if not 0 <= value <= 1:
            raise ValueError(f"--{key.replace('_', '-')} must be between 0 and 1.")
    gpus = [text(value) for value in request.get("gpu", []) if text(value)]
    if len(gpus) != len(set(gpus)):
        raise ValueError("--gpu values must be unique.")
    request["gpu"] = gpus


def init_run(args: argparse.Namespace) -> JsonObject:
    run_dir = Path(args.run_dir).expanduser().resolve()
    cwd = Path(args.cwd).expanduser().resolve()
    if text(args.arguments):
        request = _parse_invocation(args.arguments)
    else:
        if not text(args.input_directory):
            raise ValueError("--input-directory or --arguments is required.")
        request = {
            key: value
            for key, value in vars(args).items()
            if key
            not in {
                "command",
                "arguments",
                "run_id",
                "run_dir",
                "cwd",
            }
        }
    _validate_request(request)
    input_directory = _resolved_path(text(request["input_directory"]), cwd)
    collection_root, paper_set_path, prior_manifest, prior_manifest_value = locate_collection(
        input_directory, cwd
    )
    api_url, _api_key, model = require_llm_environment()
    paths = artifact_paths(run_dir)
    for path in (
        paths["artifacts"],
        paths["mineru"],
        paths["structures"],
        paths["llm"],
        paths["extractions"],
        paths["logs"].parent,
    ):
        path.mkdir(parents=True, exist_ok=True)
    resource_plan = build_resource_plan(
        output_path=paths["resource_plan"],
        requested_backend=text(request["mineru_backend"]),
        requested_gpus=[text(value) for value in request["gpu"]],
        max_mineru_workers=int(request["mineru_workers"]),
        llm_workers=int(request["llm_workers"]),
        min_gpu_free_gib=float(request["min_gpu_free_gb"]),
        max_gpu_utilization=int(request["max_gpu_utilization"]),
        requested_device=text(request["mineru_device"]),
    )
    request.update(
        {
            "schema_version": "xlab.knowledge_graph_request.v2",
            "skill_version": SKILL_VERSION,
            "run_id": args.run_id or run_dir.name,
            "run_dir": str(run_dir),
            "cwd": str(cwd),
            "input_directory": str(collection_root),
            "input_path": str(paper_set_path),
            "input_schema_version": "xlab.paper_set.v3",
            "input_sha256": sha256_file(paper_set_path),
            "paper_set_sha256": sha256_file(paper_set_path),
            "paper_edges_path": str(collection_root / "artifacts" / "metadata" / "edges.jsonl"),
            "paper_edges_sha256": sha256_file(
                collection_root / "artifacts" / "metadata" / "edges.jsonl"
            ),
            "pdf_root": str(collection_root / "artifacts" / "pdfs"),
            "prior_manifest": str(prior_manifest),
            "prior_manifest_sha256": sha256_file(prior_manifest),
            "paper_collect_run_id": text(prior_manifest_value.get("run_id"))
            or collection_root.name,
            "paper_collect_skill_version": text(
                prior_manifest_value.get("skill_version")
            ),
            "llm": {
                "api_url": public_api_url(api_url),
                "model": model,
                "enable_thinking": False if request.get("llm_disable_thinking") else None,
                "key_environment": "KNOWLEDGE_GRAPH_LLM_API_KEY",
            },
            "resource_plan_path": str(paths["resource_plan"]),
            "resolved_mineru_backend": as_mapping(resource_plan.get("mineru")).get(
                "backend"
            ),
            "initialized_at": utc_now(),
        }
    )
    atomic_write_json(paths["request"], request)
    prepare_input_documents(
        run_dir, collection_root, paper_set_path, prior_manifest, prior_manifest_value
    )
    return request


def _request(run_dir: Path) -> JsonObject:
    request = as_mapping(read_json(artifact_paths(run_dir)["request"]))
    if not request:
        raise ValueError("Missing artifacts/request.json; run init first.")
    return request


def mineru_phase(run_dir: Path) -> JsonObject:
    paths = artifact_paths(run_dir)
    request = _request(run_dir)
    documents = as_mapping(read_json(paths["documents"]))
    if not documents:
        raise ValueError("Missing input documents or resource plan; run init first.")
    # A resumed scheduler job may own different devices or CPU affinity.
    plan = build_resource_plan(
        output_path=paths["resource_plan"],
        requested_backend=text(request["mineru_backend"]),
        requested_device=text(request.get("mineru_device")) or "auto",
        requested_gpus=[text(value) for value in request.get("gpu", [])],
        max_mineru_workers=int(request["mineru_workers"]),
        llm_workers=int(request["llm_workers"]),
        min_gpu_free_gib=float(request["min_gpu_free_gb"]),
        max_gpu_utilization=int(request["max_gpu_utilization"]),
    )
    return parse_with_mineru(
        run_dir,
        documents,
        plan,
        method=text(request["mineru_method"]),
        language=text(request["mineru_language"]),
        timeout=int(request["mineru_timeout"]),
        batch_size=int(request.get("mineru_batch_size", 24)),
        cache_root=Path(text(request["cwd"])) / ".xlab" / "cache" / "mineru",
    )


def structure_phase(run_dir: Path) -> JsonObject:
    paths = artifact_paths(run_dir)
    request = _request(run_dir)
    documents = as_mapping(read_json(paths["documents"]))
    if not documents:
        raise ValueError("Missing paper_documents.json; run mineru first.")
    return structure_documents(
        run_dir, documents, workers=int(request["structure_workers"])
    )


def extract_phase(run_dir: Path) -> JsonObject:
    paths = artifact_paths(run_dir)
    request = _request(run_dir)
    structures = as_mapping(read_json(paths["structure_manifest"]))
    if not structures:
        raise ValueError("Missing paper_structures.manifest.json; run structure first.")
    api_url, api_key, model = require_llm_environment()
    configured_llm = as_mapping(request.get("llm"))
    if (
        model != text(configured_llm.get("model"))
        or public_api_url(api_url) != text(configured_llm.get("api_url"))
    ):
        raise ValueError(
            "The knowledge-graph LLM URL or model changed after initialization; "
            "start a new run or restore the configured values."
        )
    client = LlmClient(
        api_url=api_url,
        api_key=api_key,
        model=model,
        timeout=int(request["llm_timeout"]),
        retries=int(request["llm_retries"]),
        requests_per_minute=int(request["llm_rpm"]),
        log_path=paths["logs"].parent / "llm_calls.jsonl",
        enable_thinking=False if request.get("llm_disable_thinking") else None,
    )
    return extract_documents(
        run_dir,
        structures,
        client=client,
        workers=int(request["llm_workers"]),
        max_input_chars=int(request["llm_max_input_chars"]),
        validation_retries=int(request["llm_validation_retries"]),
        max_tokens=int(request["llm_max_output_tokens"]),
    )


def build_phase(run_dir: Path) -> JsonObject:
    paths = artifact_paths(run_dir)
    documents = as_mapping(read_json(paths["documents"]))
    extractions = as_mapping(read_json(paths["extraction_manifest"]))
    if not documents or not extractions:
        raise ValueError("Missing document or extraction manifest; run earlier phases.")
    return build_graph(run_dir, documents, extractions)


def _summary(value: JsonObject) -> JsonObject:
    summary = value.get("summary")
    return as_mapping(summary) if isinstance(summary, dict) else value


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    corpus = commands.add_parser('import-corpus', help='Validate and import a bulk archive with existing MinerU bundles')
    corpus.add_argument('--corpus', required=True)
    corpus.add_argument('--run-dir', required=True)
    corpus.add_argument('--database', default='/local/job/corpus-queue.sqlite')
    init = commands.add_parser("init")
    init.add_argument("--arguments", default="")
    init.add_argument("--input-directory")
    init.add_argument("--run-id", default="")
    init.add_argument("--run-dir", required=True)
    init.add_argument("--cwd", default=".")
    _add_request_options(init)
    migration = commands.add_parser("migrate-endpoint")
    migration.add_argument("--run-dir", required=True)
    migration.add_argument("--api-url", required=True)
    migration.add_argument("--model", required=True)
    for name in ("mineru", "structure", "extract", "build", "audit", "run", "resume"):
        command = commands.add_parser(name)
        command.add_argument("--run-id", default="")
        command.add_argument("--run-dir", required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == 'import-corpus':
        return subprocess.call([sys.executable, str(Path(__file__).with_name('corpus_pipeline.py')), 'import-corpus',
                                '--corpus', args.corpus, '--run-dir', args.run_dir, '--database', args.database])
    run_dir = Path(args.run_dir).expanduser().resolve()
    try:
        if args.command == "init":
            result = init_run(args)
        else:
            with run_lock(run_dir, args.command):
                if args.command == "migrate-endpoint":
                    result = migrate_endpoint(run_dir, args.api_url, args.model)
                elif args.command == "mineru":
                    result = mineru_phase(run_dir)
                elif args.command == "structure":
                    result = structure_phase(run_dir)
                elif args.command == "extract":
                    result = extract_phase(run_dir)
                elif args.command == "build":
                    result = build_phase(run_dir)
                elif args.command == "audit":
                    result = audit_run(run_dir)
                else:
                    mineru_phase(run_dir)
                    structure_phase(run_dir)
                    extract_phase(run_dir)
                    build_phase(run_dir)
                    result = audit_run(run_dir)
        print(json.dumps(_summary(result), ensure_ascii=False, sort_keys=True))
        if args.command in {"audit", "run", "resume"}:
            return 0 if result.get("passed") is True else 2
        return 0
    except Exception as error:
        print(f"knowledge_graph: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
