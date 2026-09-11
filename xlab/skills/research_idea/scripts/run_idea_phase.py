#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research_idea_lib.checkpoints import write_checkpoint
from research_idea_lib.common import atomic_write_json, ensure_run_directories, project_relative, project_root_from_run_dir, read_json, run_paths
from research_idea_lib.config import RuntimeConfig, load_runtime_config, runtime_config_from_json
from research_idea_lib.inputs import parse_request_args, request_from_json
from research_idea_lib.manifest import finalize_run
from research_idea_lib.pipeline import SKILL_VERSION, run_pipeline
from research_idea_lib.research_idea_spec import ALGORITHM_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Run the package-owned XLab research idea implementation of {ALGORITHM_ID}"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    init = subcommands.add_parser(
        "init",
        help="Parse /xlab generate-research-ideas arguments and initialize Survey-linked run state",
    )
    init.add_argument("--arguments", default="", help="Quoted public XLab arguments; never include API keys or other secrets")
    init.add_argument("--run-dir", required=True, help="XLab-owned run directory")
    init.add_argument("--run-id", required=True, help="Stable XLab run identifier")
    init.add_argument("--cwd", help="Project root used to resolve XLab-owned paths")

    synthesize = subcommands.add_parser(
        "synthesize",
        help=f"Generate audited idea artifacts with package-native {ALGORITHM_ID} contract checks",
    )
    synthesize.add_argument("--run-dir", required=True, help="Initialized XLab-owned run directory")
    synthesize.add_argument("--run-id", required=True, help="Stable XLab run identifier")
    synthesize.add_argument("--arguments", default="", help="Public XLab arguments when init has not yet persisted them; never include secrets")
    synthesize.add_argument("--cwd", help="Project root used to resolve XLab-owned paths")
    synthesize.add_argument("--status", choices=("success", "incomplete"), default=None, help="Finalization request; success never overrides audit blockers")

    resume = subcommands.add_parser("resume", help="Resume the package-native runtime from XLab-owned checkpointed state")
    resume.add_argument("--run-dir", required=True, help="Initialized XLab-owned run directory")
    resume.add_argument("--run-id", required=True, help="Stable XLab run identifier")
    resume.add_argument("--cwd", help="Project root used to resolve XLab-owned paths")
    resume.add_argument("--status", choices=("success", "incomplete"), default=None, help="Finalization request; success never overrides audit blockers")

    audit = subcommands.add_parser("audit", help="Apply fail-closed audit semantics and write the final XLab manifest")
    audit.add_argument("--run-dir", required=True, help="XLab-owned run directory containing generated artifacts")
    audit.add_argument("--run-id", required=True, help="Stable XLab run identifier")
    audit.add_argument("--cwd", help="Project root used to resolve XLab-owned paths")
    audit.add_argument("--status", choices=("success", "incomplete"), default=None, help="Finalization request; success never overrides audit blockers")

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
    ensure_run_directories(run_dir)
    runtime = load_runtime_config()
    request = parse_request_args(args.arguments, cwd, runtime)
    request_json = request.to_json(cwd)
    runtime_json = runtime.to_json()
    paths = run_paths(run_dir)
    atomic_write_json(paths["request"], request_json)
    atomic_write_json(paths["runtime_config"], runtime_json)
    checkpoint = write_checkpoint(run_dir, "initialized", {"topic": request.topic, "warnings": request.compatibility_warnings}, warnings=request.compatibility_warnings)
    print(
        json.dumps(
            {
                "schema_version": "xlab.research_idea.init.v1",
                "run_id": args.run_id,
                "survey_path": request_json.get("survey_path"),
                "topic": request.topic,
                "mature_idea": bool(request.mature_idea),
                "refinement_scope": bool(request.refinement_scope),
                "discussion": bool(request.discussion),
                "experiment_feedback": bool(request.experiment_feedback),
                "api": {
                    "agent_model": runtime.agent_model,
                    "generation_model": runtime.generation_model,
                    "evaluation_model": runtime.evaluation_model,
                    "fusion_model": runtime.fusion_model,
                    "openai_base_url": runtime.openai_base_url,
                    "algorithm_spec": ALGORITHM_ID,
                },
                "mcts": {
                    "max_iterations": runtime.max_iterations,
                    "max_depth": runtime.max_depth,
                    "branching_factor": runtime.branching_factor,
                    "idea_taste_modes": runtime.idea_taste_modes,
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
    ensure_run_directories(run_dir)
    runtime = load_runtime(run_dir)
    if not run_paths(run_dir)["request"].exists() and args.arguments:
        request = parse_request_args(args.arguments, cwd, runtime)
        atomic_write_json(run_paths(run_dir)["request"], request.to_json(cwd))
    request = request_from_json(load_request(run_dir), cwd)
    result = run_pipeline(cwd=cwd, run_dir=run_dir, run_id=args.run_id, request=request, runtime=runtime, requested_status=args.status)
    print(json.dumps({"manifest_path": project_relative(cwd, run_paths(run_dir)["manifest_json"]), "status": result["status"], "passed": result["report"].get("passed")}, ensure_ascii=False))
    return 0 if result["status"] == "success" else 2


def command_resume(args: argparse.Namespace) -> int:
    return command_synthesize(argparse.Namespace(run_dir=args.run_dir, run_id=args.run_id, arguments="", cwd=args.cwd, status=args.status))


def command_audit(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    cwd = cwd_from_args(run_dir, args.cwd)
    request_json = load_request(run_dir)
    result = finalize_run(
        cwd=cwd,
        run_dir=run_dir,
        run_id=args.run_id,
        skill_version=SKILL_VERSION,
        request=request_json,
        requested_status=args.status,
    )
    print(
        json.dumps(
            {
                "manifest_path": project_relative(cwd, run_paths(run_dir)["manifest_json"]),
                "status": result["status"],
                "passed": result["report"].get("passed"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["status"] == "success" else 2


def main() -> int:
    args = parse_args()
    if args.command == "init":
        return command_init(args)
    if args.command == "synthesize":
        return command_synthesize(args)
    if args.command == "resume":
        return command_resume(args)
    if args.command == "audit":
        return command_audit(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
