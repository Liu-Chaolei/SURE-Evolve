#!/usr/bin/env python3
"""Thin, fail-closed launcher for the authorized SURE-Master checkout."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

MAX_OUTPUT = 32_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="/xlab run-sure-master")
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument("--task")
    task.add_argument("--task-file")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default="runs/sure_master")
    parser.add_argument("--timeout-seconds", type=int, default=0)
    return parser.parse_args()


def workspace_path(value: str, workspace: Path, *, must_exist: bool = False) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError("public paths must be workspace-relative")
    resolved = (workspace / candidate).resolve()
    if resolved == workspace or workspace not in resolved.parents:
        raise ValueError("path escapes the XLab workspace")
    if must_exist and not resolved.is_file():
        raise ValueError(f"file does not exist: {value}")
    return resolved


def bounded(value: str) -> str:
    if len(value) <= MAX_OUTPUT:
        return value
    return value[-MAX_OUTPUT:]


def public_result(*, status: str, returncode: int | None, stdout: str, stderr: str, run_dir: str, error: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "sure_master.result.v1",
        "status": status,
        "returncode": returncode,
        "run_dir": run_dir,
        "stdout": bounded(stdout),
        "stderr": bounded(stderr),
    }
    if error:
        result["error"] = error
    return result


def main() -> int:
    args = parse_args()
    root_value = os.environ.get("SURE_EVOLVE_ROOT", "").strip()
    if not root_value:
        print(json.dumps(public_result(status="incomplete", returncode=None, stdout="", stderr="", run_dir=args.run_dir, error="SURE_EVOLVE_ROOT is required")), flush=True)
        return 2
    root = Path(root_value).expanduser().resolve()
    if not root.is_dir() or not (root / "run.py").is_file():
        print(json.dumps(public_result(status="incomplete", returncode=None, stdout="", stderr="", run_dir=args.run_dir, error="SURE_EVOLVE_ROOT is not an authorized SURE checkout")), flush=True)
        return 2

    workspace = Path.cwd().resolve()
    try:
        config = workspace_path(args.config, workspace, must_exist=True)
        run_dir = workspace_path(args.run_dir, workspace)
        run_dir.mkdir(parents=True, exist_ok=True)
        task_file = workspace_path(args.task_file, workspace, must_exist=True) if args.task_file else None
        if args.task is not None and not args.task.strip():
            raise ValueError("--task cannot be empty")
    except (OSError, ValueError) as exc:
        print(json.dumps(public_result(status="incomplete", returncode=None, stdout="", stderr="", run_dir=args.run_dir, error=str(exc))), flush=True)
        return 2

    python_executable = os.environ.get("SURE_PYTHON", "python").strip() or "python"
    command = [python_executable, str(root / "run.py"), "--agent", "sure_master", "--config", str(config)]
    if args.task is not None:
        command.extend(("--task", args.task))
    else:
        command.extend(("--task", task_file.read_text(encoding="utf-8")))
    command.extend(("--run-dir", str(run_dir)))

    try:
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds or None,
            check=False,
        )
        status = "success" if completed.returncode == 0 else "incomplete"
        result = public_result(status=status, returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr, run_dir=args.run_dir)
        print(json.dumps(result, ensure_ascii=True), flush=True)
        return completed.returncode if completed.returncode else 0
    except subprocess.TimeoutExpired as exc:
        result = public_result(status="incomplete", returncode=None, stdout=exc.stdout or "", stderr=exc.stderr or "", run_dir=args.run_dir, error="SURE-Master timed out; reconcile before retrying")
        print(json.dumps(result, ensure_ascii=True), flush=True)
        return 124
    except KeyboardInterrupt:
        result = public_result(status="incomplete", returncode=None, stdout="", stderr="", run_dir=args.run_dir, error="SURE-Master cancelled; reconcile before retrying")
        print(json.dumps(result, ensure_ascii=True), flush=True)
        return 130
    except OSError as exc:
        result = public_result(status="incomplete", returncode=None, stdout="", stderr="", run_dir=args.run_dir, error=f"unable to start SURE-Master: {exc}")
        print(json.dumps(result, ensure_ascii=True), flush=True)
        return 126


if __name__ == "__main__":
    raise SystemExit(main())
