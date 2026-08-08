#!/usr/bin/env python3
"""Evaluate existing F5-TTS regular-search candidates on the selection split.

This script does not run SURE Master research. It copies fixed candidate
`run_sure.py` files into fresh selection workspaces, prepares the configured
base-model symlinks, submits each candidate through the VC child runner, and
writes an aggregate result table.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evomaster.config import ConfigManager
from playground.sure_master.core.utils.vc_remote import (
    VcRemoteTrainingExecutor,
    remote_training_max_parallel,
)
from playground.sure_master.tools.run_vc_sure_candidate import load_sure_objects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run",
        default="runs/f5tts_regular_search",
        help="Regular-search run directory containing workspaces/task_0/exp_*/run_sure.py.",
    )
    parser.add_argument(
        "--output-run",
        default="runs/f5tts_selection_from_regular",
        help="Output run directory for selection evaluation workspaces and aggregate results.",
    )
    parser.add_argument(
        "--config",
        default="configs/sure_master/gpt-5-f5tts-selection.yaml",
        help="Selection config. Its base_models eval_data must point to the selection split.",
    )
    parser.add_argument(
        "--candidates",
        nargs="*",
        help="Optional candidate names such as exp_16_improve. Defaults to all scored source candidates.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Maximum VC child jobs in parallel. Defaults to config/env remote_training max_parallel.",
    )
    parser.add_argument("--timeout", type=int, default=21600, help="Per-candidate execution timeout.")
    parser.add_argument("--force", action="store_true", help="Recreate existing candidate workspaces.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare workspaces and commands, but do not submit.")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def exp_sort_key(path: Path) -> tuple[int, str]:
    parts = path.name.split("_", 2)
    try:
        index = int(parts[1])
    except (IndexError, ValueError):
        index = 10**9
    return index, path.name


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def source_candidates(source_run: Path, requested: list[str] | None) -> list[dict[str, Any]]:
    task_root = source_run / "workspaces" / "task_0"
    if not task_root.is_dir():
        raise FileNotFoundError(f"source task workspace not found: {task_root}")

    requested_set = set(requested or [])
    rows: list[dict[str, Any]] = []
    for exp_dir in sorted(task_root.glob("exp_*"), key=exp_sort_key):
        if requested_set and exp_dir.name not in requested_set:
            continue
        run_sure = exp_dir / "run_sure.py"
        if not run_sure.is_file():
            continue
        result_path = exp_dir / "metric" / "remote_training_result.json"
        summary_path = exp_dir / "metric" / "score_summary.json"
        payload = load_json(result_path) or load_json(summary_path) or {}
        if requested_set or payload.get("success", payload.get("status") == "ok"):
            rows.append(
                {
                    "name": exp_dir.name,
                    "source_workspace": str(exp_dir),
                    "source_run_sure": str(run_sure),
                    "regular_score": payload.get("score"),
                    "regular_metric": payload.get("metric"),
                }
            )

    missing = sorted(requested_set - {row["name"] for row in rows})
    if missing:
        raise ValueError(f"requested candidates were not found or have no run_sure.py: {missing}")
    if not rows:
        raise ValueError(f"no source candidates found under {task_root}")
    return rows


def replace_with_symlink(source: Path, target: Path, *, force: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        current = target.readlink()
        if current == source:
            return
        target.unlink()
    elif target.exists():
        if not force:
            return
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.symlink_to(source)


def copy_if_present(source: Path, target: Path, *, force: bool) -> None:
    if not source.exists():
        return
    if target.exists():
        if not force:
            return
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, symlinks=True)
    else:
        shutil.copy2(source, target)


def copy_source_candidate_state(source_workspace: Path, workspace: Path, *, force: bool) -> dict[str, str]:
    copied: dict[str, str] = {}
    for relative in (
        Path("models/f5tts_finetune"),
        Path("models/f5tts_arch"),
        Path("artifacts/candidate_changes.json"),
    ):
        source = source_workspace / relative
        target = workspace / relative
        copy_if_present(source, target, force=force)
        if target.exists():
            copied[str(relative)] = str(target)
    return copied


def prepare_workspace(
    *,
    candidate: dict[str, Any],
    output_task_root: Path,
    base_model_profile: Any,
    force: bool,
) -> Path:
    workspace = output_task_root / candidate["name"]
    if workspace.exists() and force:
        shutil.rmtree(workspace)
    for name in ("artifacts", "metric", "models", "working"):
        (workspace / name).mkdir(parents=True, exist_ok=True)

    shutil.copy2(candidate["source_run_sure"], workspace / "run_sure.py")
    copied_state = copy_source_candidate_state(
        Path(candidate["source_workspace"]),
        workspace,
        force=force,
    )

    if base_model_profile is not None:
        for name, source_value in base_model_profile.source_paths.items():
            target_value = base_model_profile.required_paths.get(name)
            if not target_value:
                continue
            source = Path(str(source_value)).expanduser()
            if not source.is_absolute():
                source = PROJECT_ROOT / source
            replace_with_symlink(source, workspace / target_value, force=force)

    manifest = {
        **candidate,
        "selection_workspace": str(workspace),
        "copied_candidate_state": copied_state,
        "prepared_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (workspace / "selection_candidate.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return workspace


def write_aggregate(output_run: Path, rows: list[dict[str, Any]]) -> None:
    output_run.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(
        rows,
        key=lambda row: (
            row.get("selection_score") is None,
            float("inf") if row.get("selection_score") is None else float(row["selection_score"]),
            row["name"],
        ),
    )
    (output_run / "selection_results.json").write_text(
        json.dumps(rows_sorted, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = ["rank\tcandidate\tsuccess\tregular_score\tselection_score\tmetric\tworkspace\tresult_path"]
    for rank, row in enumerate(rows_sorted, start=1):
        lines.append(
            "\t".join(
                [
                    str(rank),
                    str(row.get("name", "")),
                    str(row.get("success", "")),
                    str(row.get("regular_score", "")),
                    str(row.get("selection_score", "")),
                    str(row.get("metric", "")),
                    str(row.get("workspace", "")),
                    str(row.get("result_path", "")),
                ]
            )
        )
    (output_run / "selection_results.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_one(
    *,
    config: Any,
    config_path: Path,
    candidate: dict[str, Any],
    workspace: Path,
    timeout: int,
    dry_run: bool,
) -> dict[str, Any]:
    result_path = workspace / "metric" / "remote_training_result.json"
    executor = VcRemoteTrainingExecutor(config, config_path=config_path)
    if dry_run:
        command = executor._build_vc_command(
            workspace=workspace,
            result_path=result_path,
            exp_name=f"selection-{candidate['name']}",
            execution_timeout=timeout,
        )
        command_path = workspace / "metric" / "selection_vc_command.json"
        command_path.parent.mkdir(parents=True, exist_ok=True)
        command_path.write_text(
            json.dumps({"command": command}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            **candidate,
            "success": None,
            "selection_score": None,
            "metric": None,
            "workspace": str(workspace),
            "result_path": str(result_path),
            "dry_run_command": str(command_path),
        }

    payload = executor.run(
        workspace_path=workspace,
        exp_name=f"selection-{candidate['name']}",
        execution_timeout=timeout,
    )
    return {
        **candidate,
        "success": bool(payload.get("success")),
        "selection_score": payload.get("score"),
        "metric": payload.get("metric"),
        "metric_feedback": payload.get("metric_feedback"),
        "error": payload.get("error"),
        "workspace": str(workspace),
        "result_path": str(result_path),
    }


def main() -> None:
    args = parse_args()
    source_run = resolve_path(args.source_run)
    output_run = resolve_path(args.output_run)
    config_path = resolve_path(args.config)
    output_task_root = output_run / "workspaces" / "task_0"

    config = ConfigManager(config_dir=config_path.parent, config_file=config_path.name).load()
    _, base_model_profile, _, _, _ = load_sure_objects(config)

    candidates = source_candidates(source_run, args.candidates)
    output_task_root.mkdir(parents=True, exist_ok=True)
    workspaces = {
        row["name"]: prepare_workspace(
            candidate=row,
            output_task_root=output_task_root,
            base_model_profile=base_model_profile,
            force=args.force,
        )
        for row in candidates
    }

    max_parallel = args.max_parallel or remote_training_max_parallel(config, default=1)
    max_parallel = max(1, min(max_parallel, len(candidates)))

    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as pool:
        future_to_name = {
            pool.submit(
                evaluate_one,
                config=config,
                config_path=config_path,
                candidate=row,
                workspace=workspaces[row["name"]],
                timeout=args.timeout,
                dry_run=args.dry_run,
            ): row["name"]
            for row in candidates
        }
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - aggregate all candidate failures.
                result = {
                    "name": name,
                    "success": False,
                    "selection_score": None,
                    "error": str(exc),
                    "workspace": str(workspaces[name]),
                }
            rows.append(result)
            write_aggregate(output_run, rows)
            print(
                f"[selection] {name}: success={result.get('success')} "
                f"score={result.get('selection_score')} error={result.get('error')}",
                flush=True,
            )

    write_aggregate(output_run, rows)
    print(f"Wrote {output_run / 'selection_results.tsv'}")
    print(f"Wrote {output_run / 'selection_results.json'}")


if __name__ == "__main__":
    main()
