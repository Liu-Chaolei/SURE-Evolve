#!/usr/bin/env python3
"""Compact existing SURE candidate workspaces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from playground.sure_master.core.utils.workspace_cleanup import (  # noqa: E402
    WorkspaceCleanupConfig,
    cleanup_candidate_workspace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Run directory to scan, e.g. runs/zipformer_staged_axes")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete heavy candidate outputs. Without this, only list matching workspaces.",
    )
    parser.add_argument(
        "--keep-failures",
        action="store_true",
        help="Do not clean workspaces whose candidate_status.json says success=false.",
    )
    parser.add_argument(
        "--log-tail-bytes",
        type=int,
        default=65536,
        help="Bytes to preserve from each candidate log before deleting full logs.",
    )
    parser.add_argument(
        "--max-workspaces",
        type=int,
        default=0,
        help="Stop after this many candidate workspaces. 0 means no limit.",
    )
    return parser.parse_args()


def find_candidate_workspaces(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"root does not exist: {root}")
    workspaces: set[Path] = {path.parent for path in root.rglob("run_sure.py")}
    for artifact_name in ("candidate_status.json", "candidate_changes.json", "hyp.txt"):
        for path in root.rglob(f"artifacts/{artifact_name}"):
            workspaces.add(path.parent.parent)
    for heavy_name in ("models", "working", ".sure_runtime"):
        for path in root.rglob(heavy_name):
            if not path.is_dir():
                continue
            candidate = path.parent
            if _looks_like_candidate_workspace(candidate):
                workspaces.add(candidate)
    return sorted(set(workspaces))


def _looks_like_candidate_workspace(path: Path) -> bool:
    if (path / "run_sure.py").is_file():
        return True
    if (path / "artifacts" / "candidate_status.json").is_file():
        return True
    if (path / "artifacts" / "candidate_changes.json").is_file():
        return True
    if (path / "artifacts" / "hyp.txt").is_file():
        return True
    return path.name.startswith("exp_") and (path / "artifacts").is_dir()


def read_status(workspace: Path) -> dict[str, Any]:
    path = workspace / "artifacts" / "candidate_status.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    workspaces = find_candidate_workspaces(root)
    if args.max_workspaces > 0:
        workspaces = workspaces[: args.max_workspaces]

    cleanup_config = WorkspaceCleanupConfig(
        enabled=bool(args.apply),
        on_success=True,
        on_failure=not bool(args.keep_failures),
        log_tail_bytes=max(1, int(args.log_tail_bytes)),
    )

    removed_total = 0
    cleaned_count = 0
    skipped_count = 0
    for workspace in workspaces:
        status = read_status(workspace)
        success = bool(status.get("success", False))
        reason_code = str(status.get("reason_code") or "unknown")
        score = status.get("score")
        score_value = float(score) if isinstance(score, (int, float)) else None
        rel = workspace.relative_to(root) if workspace.is_relative_to(root) else workspace
        if not args.apply:
            print(f"DRY-RUN {rel} success={success} reason={reason_code}")
            continue
        manifest = cleanup_candidate_workspace(
            workspace,
            cleanup_config,
            success=success,
            reason_code=reason_code,
            score=score_value,
            metadata={"cleanup_script_root": str(root)},
        )
        if manifest.get("cleaned"):
            cleaned_count += 1
            removed_total += int(manifest.get("removed_bytes") or 0)
            print(
                f"CLEANED {rel} removed_mib={int(manifest.get('removed_bytes') or 0) / (1024 * 1024):.2f}"
            )
        else:
            skipped_count += 1
            print(f"SKIPPED {rel} reason={manifest.get('reason')}")

    if args.apply:
        print(
            f"summary: cleaned={cleaned_count} skipped={skipped_count} "
            f"removed_mib={removed_total / (1024 * 1024):.2f}"
        )
    else:
        print(f"summary: matching_workspaces={len(workspaces)}; add --apply to clean")


if __name__ == "__main__":
    main()
