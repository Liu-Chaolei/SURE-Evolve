"""Collectors that convert run directories into compact trace digests."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from .models import RunMetrics, TraceDigest

logger = logging.getLogger(__name__)

_ISSUE_RE = re.compile(
    r"(traceback|exception|error|failed|failure|timeout|timed out|not found|"
    r"invalid|jsondecode|exit code|segmentation fault|oom|out of memory|"
    r"submission.*missing|no such file)",
    re.IGNORECASE,
)
_SCORE_RE = re.compile(
    r"(?:validation score|score|metric)\s*[:=]\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _truncate(text: Any, limit: int = 4000) -> str:
    value = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, default=str)
    if len(value) <= limit:
        return value
    half = limit // 2
    return value[:half] + "\n...[truncated]...\n" + value[-half:]


def _safe_load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Failed to load json %s: %s", path, exc)
        return None


def _read_text_tail(path: Path, limit: int = 80_000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)
    return ""


def _known_agents(config_path: Path) -> list[str]:
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    agents = raw.get("agents") or {}
    return list(agents.keys()) if isinstance(agents, dict) else []


def _tool_name_from_call(call: dict[str, Any]) -> str:
    return str((call.get("function") or {}).get("name") or "")


def _tool_args_from_call(call: dict[str, Any]) -> str:
    return str((call.get("function") or {}).get("arguments") or "")


def _tool_names_from_specs(specs: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for spec in specs or []:
        name = ((spec.get("function") or {}).get("name") if isinstance(spec, dict) else None)
        if name:
            names.append(str(name))
    return names


def _iter_trajectory_entries(run_dir: Path) -> list[tuple[Path, int, dict[str, Any]]]:
    entries: list[tuple[Path, int, dict[str, Any]]] = []
    for path in sorted((run_dir / "trajectories").glob("**/trajectory.json")):
        data = _safe_load_json(path)
        if not isinstance(data, list):
            continue
        for idx, entry in enumerate(data):
            if isinstance(entry, dict):
                entries.append((path, idx, entry))
    return entries


def _collect_logs(run_dir: Path) -> tuple[str, list[str], float | None]:
    excerpts: list[str] = []
    issues: list[str] = []
    score: float | None = None

    for path in sorted((run_dir / "logs").glob("*.log")):
        text = _read_text_tail(path)
        if not text:
            continue
        excerpts.append(f"## {path.relative_to(run_dir)}\n{_truncate(text, 25_000)}")
        for line in text.splitlines():
            if _ISSUE_RE.search(line):
                issues.append(f"{path.name}: {_truncate(line.strip(), 800)}")
            score_match = _SCORE_RE.search(line)
            if score_match:
                try:
                    score = float(score_match.group(1))
                except ValueError:
                    pass

    return "\n\n".join(excerpts), issues[:120], score


def _collect_artifacts(run_dir: Path) -> tuple[dict[str, Any], float | None]:
    artifacts: dict[str, Any] = {}
    score: float | None = None

    interesting = [
        *run_dir.glob("**/*grade*.json"),
        *run_dir.glob("**/*metric*.json"),
        *run_dir.glob("**/result.json"),
    ]
    seen: set[Path] = set()
    for path in interesting:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            if path.stat().st_size > 200_000:
                continue
        except OSError:
            continue
        data = _safe_load_json(path)
        if data is None:
            continue
        key = str(path.relative_to(run_dir))
        artifacts[key] = data
        if isinstance(data, dict):
            for score_key in ("score", "metric", "validation_score", "public_score", "private_score"):
                value = data.get(score_key)
                if isinstance(value, (int, float)):
                    score = float(value)

    return artifacts, score


def _workspace_manifest(run_dir: Path, limit: int = 250) -> list[str]:
    roots = [run_dir / "workspace", run_dir / "workspaces"]
    manifest: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if len(manifest) >= limit:
                manifest.append("...[manifest truncated]...")
                return manifest
            if path.is_dir():
                continue
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            manifest.append(f"{path.relative_to(run_dir)} ({size} bytes)")
    return manifest


def collect_trace_digest(run_dir: str | Path, config_path: str | Path) -> TraceDigest:
    """Collect a compact digest from an EvoMaster run directory."""

    run_dir = Path(run_dir)
    config_path = Path(config_path)
    known_agents = _known_agents(config_path)

    tool_call_counts: dict[str, int] = {}
    tool_error_counts: dict[str, int] = {}
    step_summaries: list[dict[str, Any]] = []
    issues: list[str] = []
    task_description: str | None = None
    status = "unknown"

    entries = _iter_trajectory_entries(run_dir)
    for path, idx, entry in entries:
        traj = entry.get("trajectory") or {}
        status = str(entry.get("status") or traj.get("status") or status)
        agent_name = str(traj.get("agent_name") or (traj.get("meta") or {}).get("agent_name") or "unknown")
        dialogs = traj.get("dialogs") or []
        prompt = dialogs[0] if dialogs else {}
        messages = prompt.get("messages") or []
        tools = _tool_names_from_specs(prompt.get("tools") or [])

        if task_description is None:
            for msg in messages:
                if msg.get("role") == "user":
                    task_description = _truncate(_extract_message_text(msg.get("content")), 5000)
                    break

        steps = traj.get("steps") or []
        step = steps[0] if steps else {}
        assistant = step.get("assistant_message") or {}
        tool_calls = assistant.get("tool_calls") or []
        tool_responses = step.get("tool_responses") or []
        call_names = [_tool_name_from_call(tc) for tc in tool_calls if _tool_name_from_call(tc)]

        for name in call_names:
            tool_call_counts[name] = tool_call_counts.get(name, 0) + 1

        response_issues: list[str] = []
        for response in tool_responses:
            name = str(response.get("name") or "unknown")
            content = str(response.get("content") or "")
            info = response.get("meta", {}).get("info", {}) if isinstance(response.get("meta"), dict) else {}
            has_error = False
            if isinstance(info, dict) and (info.get("error") or info.get("exit_code") not in (None, 0, "0")):
                has_error = True
            if _ISSUE_RE.search(content):
                has_error = True
            if has_error:
                tool_error_counts[name] = tool_error_counts.get(name, 0) + 1
                issue = f"{agent_name} step {traj.get('step')}: {name} -> {_truncate(content, 1000)}"
                issues.append(issue)
                response_issues.append(issue)

        summary = {
            "trajectory_file": str(path.relative_to(run_dir)),
            "entry_index": idx,
            "agent_name": agent_name,
            "exp_name": entry.get("exp_name"),
            "exp_index": entry.get("exp_index"),
            "step": traj.get("step") or entry.get("steps"),
            "status": status,
            "exposed_tools": tools,
            "tool_calls": [
                {
                    "name": _tool_name_from_call(tc),
                    "arguments": _truncate(_tool_args_from_call(tc), 1200),
                }
                for tc in tool_calls
            ],
            "assistant_content": _truncate(assistant.get("content") or "", 1200),
            "tool_response_summaries": [
                {
                    "name": response.get("name"),
                    "content": _truncate(response.get("content") or "", 1200),
                    "info": response.get("meta", {}).get("info", {}) if isinstance(response.get("meta"), dict) else {},
                }
                for response in tool_responses[:5]
            ],
            "issues": response_issues[:5],
        }
        step_summaries.append(summary)

    log_excerpt, log_issues, log_score = _collect_logs(run_dir)
    issues.extend(log_issues)
    artifacts, artifact_score = _collect_artifacts(run_dir)
    score = artifact_score if artifact_score is not None else log_score

    metrics = RunMetrics(
        status=status,
        step_count=max((int(entry.get("steps") or 0) for _, _, entry in entries), default=0),
        llm_call_count=len(entries),
        tool_call_count=sum(tool_call_counts.values()),
        tool_error_count=sum(tool_error_counts.values()),
        issue_count=len(issues),
        score=score,
    )

    return TraceDigest(
        run_dir=str(run_dir),
        config_path=str(config_path),
        known_agents=known_agents,
        task_description=task_description,
        metrics=metrics,
        tool_call_counts=tool_call_counts,
        tool_error_counts=tool_error_counts,
        step_summaries=step_summaries[:120],
        issues=issues[:160],
        artifacts=artifacts,
        workspace_manifest=_workspace_manifest(run_dir),
        log_excerpt=_truncate(log_excerpt, 80_000),
    )

