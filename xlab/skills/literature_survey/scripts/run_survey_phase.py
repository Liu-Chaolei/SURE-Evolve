#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from literature_survey_lib.audit import audit_artifacts
from literature_survey_lib.checkpoints import write_checkpoint
from literature_survey_lib.common import atomic_write_json, project_relative, project_root_from_run_dir, read_json, resolve_project_path, run_paths
from literature_survey_lib.config import RuntimeConfig, load_runtime_config, runtime_config_from_json
from literature_survey_lib.inputs import SurveyRequest, parse_request_args, request_from_json
from literature_survey_lib.manifest import build_final_manifest, write_final_manifest
from literature_survey_lib.pipeline import SKILL_VERSION, run_pipeline
from literature_survey_lib.corpus_survey import run_corpus_surveys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the self-contained XLab literature survey runtime")
    subcommands = parser.add_subparsers(dest="command", required=True)
    corpus = subcommands.add_parser('corpus-synthesize', help='Write full local surveys from a validated tiered corpus pipeline')
    corpus.add_argument('--pipeline-dir', type=Path, required=True)
    corpus.add_argument('--pilot', action='store_true')

    init = subcommands.add_parser("init", help="Parse /xlab write-literature-survey arguments and initialize run state")
    init.add_argument("--arguments", default="")
    init.add_argument("--run-dir", required=True)
    init.add_argument("--run-id", required=True)
    init.add_argument("--cwd")

    synthesize = subcommands.add_parser("synthesize", help="Generate survey artifacts from a graph or paper set")
    synthesize.add_argument("--run-dir", required=True)
    synthesize.add_argument("--run-id", required=True)
    synthesize.add_argument("--arguments", default="")
    synthesize.add_argument("--cwd")
    synthesize.add_argument("--status", choices=("success", "incomplete"), default=None)

    resume = subcommands.add_parser("resume", help="Resume a staged survey run from checkpointed state")
    resume.add_argument("--run-dir", required=True)
    resume.add_argument("--run-id", required=True)
    resume.add_argument("--cwd")
    resume.add_argument("--status", choices=("success", "incomplete"), default=None)

    audit = subcommands.add_parser("audit", help="Audit generated artifacts and write a final manifest")
    audit.add_argument("--run-dir", required=True)
    audit.add_argument("--run-id", required=True)
    audit.add_argument("--cwd")
    audit.add_argument("--status", choices=("success", "incomplete"), default=None)

    resources = subcommands.add_parser("resources", help="Build and verify the explicit native idea resource bundle")
    resources.add_argument("--run-dir", required=True)
    resources.add_argument("--run-id", required=True)
    resources.add_argument("--cwd")
    resources.add_argument("--model-snapshot", required=True)

    smoke = subcommands.add_parser("smoke", help="Run init, synthesize, and audit in one process")
    smoke.add_argument("--run-dir", required=True)
    smoke.add_argument("--run-id", default="smoke-run")
    smoke.add_argument("--input")
    smoke.add_argument("--graph")
    smoke.add_argument("--topic", default="Literature survey smoke test")
    smoke.add_argument("--cwd")

    return parser.parse_args()


def cwd_from_args(run_dir: Path, cwd: str | None) -> Path:
    return Path(cwd).resolve() if cwd else project_root_from_run_dir(run_dir)


def load_request(run_dir: Path) -> dict[str, Any]:
    request_path = run_paths(run_dir)["request"]
    if not request_path.exists():
        raise ValueError("Run state is missing request.json; run init first.")
    request = read_json(request_path)
    if not isinstance(request, dict):
        raise ValueError("request.json must contain an object.")
    return request


def load_runtime(run_dir: Path) -> RuntimeConfig:
    path = run_paths(run_dir)["runtime_config"]
    if path.exists():
        value = read_json(path)
        if isinstance(value, dict):
            return runtime_config_from_json(value)
    runtime = load_runtime_config()
    atomic_write_json(path, runtime.to_json())
    return runtime


def command_init(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    cwd = cwd_from_args(run_dir, args.cwd)
    paths = run_paths(run_dir)
    for directory in (paths["artifacts"], paths["state"], paths["logs"]):
        directory.mkdir(parents=True, exist_ok=True)
    runtime = load_runtime_config()
    request = parse_request_args(args.arguments, cwd, runtime)
    request_json = request.to_json(cwd)
    runtime_json = runtime.to_json()
    atomic_write_json(paths["request"], request_json)
    atomic_write_json(paths["runtime_config"], runtime_json)
    checkpoint = write_checkpoint(run_dir, "initialized", {"topic": request.topic, "warnings": request.compatibility_warnings}, warnings=request.compatibility_warnings)
    print(
        json.dumps(
            {
                "schema_version": "xlab.literature_survey.init.v1",
                "run_id": args.run_id,
                "topic": request.topic,
                "input_path": request_json.get("input_path"),
                "graph_path": request_json.get("graph_path"),
                "max_papers": request.max_papers,
                "min_papers": request.min_papers,
                "full_text": request.full_text,
                "api": {
                    "llm_model": runtime.model,
                    "llm_base_url": runtime.llm_base_url,
                    "llm_context_window": runtime.llm_context_window,
                    "llm_api_key_set": runtime.llm_api_key_set,
                    "semantic_scholar_api_key_set": runtime.semantic_scholar_api_key_set,
                    "llm_refinement_enabled": runtime.llm_refinement_enabled,
                },
                "warnings": request.compatibility_warnings,
                "checkpoint": checkpoint,
            },
            ensure_ascii=False,
        )
    )
    return 0


def command_synthesize(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    cwd = cwd_from_args(run_dir, args.cwd)
    paths = run_paths(run_dir)
    for directory in (paths["artifacts"], paths["state"], paths["logs"]):
        directory.mkdir(parents=True, exist_ok=True)
    runtime = load_runtime(run_dir)
    if not paths["request"].exists() and args.arguments:
        request = parse_request_args(args.arguments, cwd, runtime)
        atomic_write_json(paths["request"], request.to_json(cwd))
    request = request_from_json(load_request(run_dir), cwd)
    result = run_pipeline(cwd=cwd, run_dir=run_dir, run_id=args.run_id, request=request, runtime=runtime, requested_status=args.status)
    print(json.dumps({"manifest_path": project_relative(cwd, paths["manifest_json"]), "status": result["status"], "passed": result["report"].get("passed")}, ensure_ascii=False))
    return 0 if result["status"] == "success" else 2


def command_resume(args: argparse.Namespace) -> int:
    return command_synthesize(argparse.Namespace(run_dir=args.run_dir, run_id=args.run_id, arguments="", cwd=args.cwd, status=args.status))


def command_audit(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    cwd = cwd_from_args(run_dir, args.cwd)
    paths = run_paths(run_dir)
    request_json = load_request(run_dir)
    report = audit_artifacts(run_dir, int(request_json.get("min_papers") or 3))
    status = "success" if report.get("passed") is True else "incomplete"
    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []
    if args.status == "success" and status != "success":
        warnings.append("Requested status success was ignored because audit did not pass.")
    elif args.status == "incomplete" and status == "success":
        warnings.append("Requested status incomplete downgraded an otherwise successful survey manifest.")
        status = "incomplete"
    report["warnings"] = sorted(set(str(warning) for warning in warnings))
    atomic_write_json(paths["report_json"], report)
    survey = read_json(paths["survey_json"]) if paths["survey_json"].exists() else {}
    citations = read_json(paths["citations_json"]) if paths["citations_json"].exists() else {}
    manifest = build_final_manifest(
        cwd=cwd,
        run_dir=run_dir,
        run_id=args.run_id,
        skill_version=SKILL_VERSION,
        status=status,
        request=request_json,
        survey=survey if isinstance(survey, dict) else {},
        citations=citations if isinstance(citations, dict) else {},
        report=report,
    )
    write_final_manifest(paths["manifest_json"], manifest)
    print(json.dumps({"manifest_path": project_relative(cwd, paths["manifest_json"]), "status": status, "passed": report["passed"]}, ensure_ascii=False))
    return 0 if status == "success" else 2


def command_smoke(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    cwd = cwd_from_args(run_dir, args.cwd)
    argument_parts = [shlex.quote(args.topic)]
    if args.graph:
        argument_parts.extend(["--graph", shlex.quote(args.graph)])
    if args.input:
        argument_parts.extend(["--input", shlex.quote(args.input)])
    init_args = argparse.Namespace(arguments=" ".join(argument_parts), run_dir=str(run_dir), run_id=args.run_id, cwd=str(cwd))
    command_init(init_args)
    runtime = load_runtime(run_dir)
    request = request_from_json(load_request(run_dir), cwd)
    result = run_pipeline(cwd=cwd, run_dir=run_dir, run_id=args.run_id, request=request, runtime=runtime, allow_synthetic=not bool(args.input or args.graph))
    print(json.dumps({"manifest_path": project_relative(cwd, run_paths(run_dir)["manifest_json"]), "status": result["status"], "passed": result["report"].get("passed")}, ensure_ascii=False))
    return 0 if result["status"] == "success" else 2


def main() -> int:
    args = parse_args()
    if args.command == 'corpus-synthesize':
        print(json.dumps(run_corpus_surveys(args.pipeline_dir, args.pilot), ensure_ascii=False))
        return 0
    if args.command == "init":
        return command_init(args)
    if args.command == "synthesize":
        return command_synthesize(args)
    if args.command == "resume":
        return command_resume(args)
    if args.command == "audit":
        return command_audit(args)
    if args.command == "smoke":
        return command_smoke(args)
    if args.command == "resources":
        command = [sys.executable, str(Path(__file__).with_name("build_resource_bundle.py")),
                   "--run-dir", args.run_dir, "--run-id", args.run_id, "--model-snapshot", args.model_snapshot]
        if args.cwd:
            command.extend(["--cwd", args.cwd])
        return subprocess.call(command)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
