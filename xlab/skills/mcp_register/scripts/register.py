#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--server-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--approve", action="store_true")
    args = parser.parse_args()
    if not args.approve:
        raise PermissionError("registration requires explicit --approve")
    package = json.loads(Path(args.package).read_text(encoding="utf-8"))
    validation = json.loads(Path(args.validation).read_text(encoding="utf-8"))
    if validation.get("passed") is not True or validation.get("tool_name") != package.get("tool_name"):
        raise ValueError("matching successful MCP validation is required")

    config_path = Path(args.config).resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    if not isinstance(existing, dict):
        raise ValueError("target config must be a JSON object")
    servers = existing.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        raise ValueError("target mcpServers field must be an object")
    registration = {
        "command": "python",
        "args": [package["server"]],
    }
    changed = servers.get(args.server_name) != registration
    backup_path = None
    if changed:
        if config_path.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = config_path.with_name(f"{config_path.name}.{stamp}.bak")
            shutil.copy2(config_path, backup)
            backup_path = str(backup)
        servers[args.server_name] = registration
        handle, temporary_name = tempfile.mkstemp(prefix=f".{config_path.name}.", dir=config_path.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as temporary:
                json.dump(existing, temporary, indent=2)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, config_path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    result = {
        "schema_version": "xlab.mcp_registration.v1",
        "server_name": args.server_name,
        "config_path": str(config_path),
        "backup_path": backup_path,
        "changed": changed,
        "registered": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "changed": changed}))


if __name__ == "__main__":
    main()
