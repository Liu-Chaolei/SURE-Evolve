from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any


_SENSITIVE_VALUE_FIELDS = {
    "access_key",
    "access_token",
    "api_key",
    "auth_token",
    "bearer_token",
    "client_secret",
    "credential",
    "credentials",
    "password",
    "passwd",
    "private_key",
    "refresh_token",
    "secret_key",
}
_AMBIGUOUS_SENSITIVE_FIELDS = {"secret", "token"}
_REDACTED_VALUES = {"", "***", "<redacted>", "[redacted]", "redacted", "unset"}
_OPAQUE_SECRET_RE = re.compile(r"^[A-Za-z0-9_./+=:-]{12,}$")
_SECRET_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?<![A-Za-z0-9_-])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
    re.compile(r"(?<![A-Za-z0-9_-])(?:github_pat_|gh[pousr]_|glpat-|hf_|npm_)[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?<![A-Za-z0-9_-])xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"(?<![A-Za-z0-9_-])AIza[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")


def write_portable_json(path: Path, value: object) -> None:
    assert_portable_output(value, label=str(path))
    atomic_write_json(path, value)


def assert_portable_output(value: object, *, label: str = "output") -> None:
    findings = portable_output_findings(value)
    if findings["secrets"] or findings["nonportable_paths"]:
        details = [
            *(f"secret-shaped value at {field}" for field in findings["secrets"]),
            *(f"absolute or unsafe path at {field}" for field in findings["nonportable_paths"]),
        ]
        raise ValueError(f"Refusing to publish unsafe {label}: " + "; ".join(details))


def portable_output_findings(value: object) -> dict[str, list[str]]:
    secrets: list[str] = []
    nonportable_paths: list[str] = []
    for field, child in _walk_json(value):
        key = _field_name(field)
        if _contains_secret_shape(child, key):
            secrets.append(field or "<root>")
        if _is_path_field(key) and isinstance(child, str) and not _is_safe_relative_path(child):
            nonportable_paths.append(field)
    return {"secrets": secrets, "nonportable_paths": nonportable_paths}


def text_contains_secret_shape(text: str) -> bool:
    return any(pattern.search(text) for pattern in _SECRET_PATTERNS)


def _walk_json(value: object, prefix: str = "") -> Iterator[tuple[str, object]]:
    if isinstance(value, dict):
        for key, child in value.items():
            field = f"{prefix}.{key}" if prefix else str(key)
            yield field, child
            yield from _walk_json(child, field)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            field = f"{prefix}[{index}]"
            yield field, child
            yield from _walk_json(child, field)


def _field_name(field: str) -> str:
    return field.rsplit(".", 1)[-1].split("[", 1)[0].lower()


def _contains_secret_shape(value: object, key: str) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if text_contains_secret_shape(text):
        return True
    if text.lower() in _REDACTED_VALUES:
        return False
    if key in _SENSITIVE_VALUE_FIELDS:
        return True
    return key in _AMBIGUOUS_SENSITIVE_FIELDS and bool(_OPAQUE_SECRET_RE.fullmatch(text))


def _is_path_field(key: str) -> bool:
    return key == "path" or key.endswith("_path") or key.endswith("_dir")


def _is_safe_relative_path(value: str) -> bool:
    text = value.strip()
    if not text:
        return True
    path = Path(text).expanduser()
    windows_path = PureWindowsPath(text)
    return (
        not path.is_absolute()
        and not windows_path.is_absolute()
        and ".." not in path.parts
        and ".." not in windows_path.parts
    )


def atomic_write_jsonl(path: Path, values: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                yield value


def stable_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha1("\0".join(str(part) for part in parts if part is not None).encode("utf-8")).hexdigest()
    return f"{prefix}:{digest[:12]}"


def stable_signature(value: object) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def file_digest(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_root_from_run_dir(run_dir: Path) -> Path:
    resolved = run_dir.resolve()
    if resolved.parent.name == "runs" and resolved.parent.parent.name == ".xlab":
        return resolved.parent.parent.parent
    return Path.cwd().resolve()


def project_relative(cwd: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(cwd.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def resolve_project_path(cwd: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (cwd / path).resolve()


def run_paths(run_dir: Path) -> dict[str, Path]:
    artifacts = run_dir / "artifacts"
    state = run_dir / "state"
    logs = run_dir / "logs"
    research_idea = state / "research_idea"
    return {
        "artifacts": artifacts,
        "state": state,
        "logs": logs,
        "research_idea": research_idea,
        "memory": state / "memory",
        "request": state / "request.json",
        "runtime_config": state / "runtime_config.json",
        "checkpoint": state / "checkpoint.json",
        "survey_context": state / "survey_context.json",
        "context": state / "context.json",
        "resource_preflight": state / "resource_preflight.json",
        "workflow_artifact": research_idea / "artifact.json",
        "workflow_run": research_idea / "run.json",
        "workflow_retrieval": research_idea / "retrieval.json",
        "workflow_analysis": research_idea / "analysis.json",
        "workflow_ideation": research_idea / "ideation.json",
        "workflow_persistence": research_idea / "persistence.json",
        "workflow_keynote": research_idea / "keynote.json",
        **{
            f"workflow_search_{mode}": research_idea / f"search.{mode}.json"
            for mode in (
                "moonshot_inventor",
                "bridge_builder",
                "steady_engineer",
                "ambitious_realist",
                "evidence_first",
            )
        },
        "pipeline_log": logs / "pipeline.jsonl",
        "diagnostics_log": logs / "diagnostics.jsonl",
        "research_idea_log": logs / "research_idea.log",
        "idea_result_json": artifacts / "idea_result.json",
        "research_idea_json": artifacts / "research_idea.json",
        "idea_trace_json": artifacts / "idea_trace.json",
        "report_json": artifacts / "idea_report.json",
        "manifest_json": run_dir / "manifest.json",
    }


def ensure_run_directories(run_dir: Path) -> None:
    paths = run_paths(run_dir)
    for key in ("artifacts", "state", "logs", "research_idea", "memory"):
        paths[key].mkdir(parents=True, exist_ok=True)


def append_diagnostic(run_dir: Path, severity: str, message: str, *, code: str | None = None, data: object | None = None) -> None:
    entry: dict[str, object] = {"timestamp": utc_now(), "severity": severity, "message": message}
    if code:
        entry["code"] = code
    if data is not None:
        entry["data"] = data
    append_jsonl(run_paths(run_dir)["diagnostics_log"], entry)


def append_pipeline_event(run_dir: Path, stage: str, status: str, data: object | None = None) -> None:
    entry: dict[str, object] = {"timestamp": utc_now(), "stage": stage, "status": status}
    if data is not None:
        entry["data"] = data
    append_jsonl(run_paths(run_dir)["pipeline_log"], entry)
