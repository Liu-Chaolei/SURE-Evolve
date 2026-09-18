"""Publish external SURE locations as references, never as executable local paths."""
import hashlib
from pathlib import Path, PureWindowsPath


def portable_task_context(value, cwd, field=""):
    if isinstance(value, dict):
        return {key: portable_task_context(child, cwd, key) for key, child in value.items()}
    if isinstance(value, list):
        return [portable_task_context(child, cwd, field) for child in value]
    if isinstance(value, str) and (field == "path" or field.endswith(("_path", "_dir"))):
        path = Path(value).expanduser()
        if path.is_absolute() or PureWindowsPath(value).is_absolute() or ".." in path.parts:
            try:
                return path.resolve().relative_to(cwd.resolve()).as_posix()
            except ValueError:
                return {"external_artifact_identity": "sha256:" + hashlib.sha256(value.encode()).hexdigest(),
                        "display_name": path.name, "location_kind": "external_reference"}
    return value
