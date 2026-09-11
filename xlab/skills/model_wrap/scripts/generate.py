#!/usr/bin/env python3
import argparse
import json
import re
import shlex
from pathlib import Path


SERVER_TEMPLATE = '''#!/usr/bin/env python3
import json
import subprocess
import sys

PROTOCOL_VERSION = "2025-11-25"
SERVER_NAME = __SERVER_NAME__
TOOL_NAME = __TOOL_NAME__
DESCRIPTION = __DESCRIPTION__
INPUT_SCHEMA = __INPUT_SCHEMA__
RUNNER_COMMAND = __RUNNER_COMMAND__
TIMEOUT_SECONDS = __TIMEOUT_SECONDS__


def response(request_id, result=None, error=None):
    value = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        value["error"] = error
    else:
        value["result"] = result
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\\n")
    sys.stdout.flush()


def invoke(arguments):
    completed = subprocess.run(
        RUNNER_COMMAND,
        input=json.dumps(arguments) + "\\n",
        text=True,
        capture_output=True,
        timeout=TIMEOUT_SECONDS,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "content": [{"type": "text", "text": completed.stderr.strip() or f"runner exited {completed.returncode}"}],
            "isError": True,
        }
    output = completed.stdout.strip()
    try:
        structured = json.loads(output)
    except json.JSONDecodeError:
        structured = None
    result = {"content": [{"type": "text", "text": output}], "isError": False}
    if structured is not None:
        result["structuredContent"] = structured
    return result


def handle(message):
    method = message.get("method")
    request_id = message.get("id")
    if method == "notifications/initialized":
        return
    if method == "initialize":
        response(request_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": "1.0.0"},
        })
    elif method == "ping":
        response(request_id, {})
    elif method == "tools/list":
        response(request_id, {"tools": [{
            "name": TOOL_NAME,
            "description": DESCRIPTION,
            "inputSchema": INPUT_SCHEMA,
        }]})
    elif method == "tools/call":
        params = message.get("params", {})
        if params.get("name") != TOOL_NAME:
            response(request_id, {
                "content": [{"type": "text", "text": "Unknown tool"}],
                "isError": True,
            })
        else:
            response(request_id, invoke(params.get("arguments", {})))
    elif request_id is not None:
        response(request_id, error={"code": -32601, "message": f"Method not found: {method}"})


for line in sys.stdin:
    try:
        message = json.loads(line)
        if isinstance(message, dict):
            handle(message)
    except Exception as error:
        sys.stderr.write(f"mcp server error: {error}\\n")
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-name", required=True)
    parser.add_argument("--tool-name", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--input-schema", required=True)
    parser.add_argument("--runner-command", required=True)
    parser.add_argument("--example-input", required=True)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tool_name):
        raise ValueError("tool name must contain only letters, digits, underscore, and hyphen")
    schema = json.loads(Path(args.input_schema).read_text(encoding="utf-8"))
    example = json.loads(Path(args.example_input).read_text(encoding="utf-8"))
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("input schema must be a JSON Schema object with type=object")
    if not isinstance(example, dict):
        raise ValueError("example input must be an object")
    runner_command = shlex.split(args.runner_command)
    if not runner_command:
        raise ValueError("runner command is empty")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    schema_path = output / "input.schema.json"
    example_path = output / "example.input.json"
    server_path = output / "server.py"
    schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    example_path.write_text(json.dumps(example, indent=2) + "\n", encoding="utf-8")
    server = (
        SERVER_TEMPLATE
        .replace("__SERVER_NAME__", repr(args.server_name))
        .replace("__TOOL_NAME__", repr(args.tool_name))
        .replace("__DESCRIPTION__", repr(args.description))
        .replace("__INPUT_SCHEMA__", repr(schema))
        .replace("__RUNNER_COMMAND__", repr(runner_command))
        .replace("__TIMEOUT_SECONDS__", repr(args.timeout))
    )
    server_path.write_text(server, encoding="utf-8")
    server_path.chmod(0o755)
    (output / "requirements.lock").write_text("# Python standard library only.\\n", encoding="utf-8")
    config = {
        "mcpServers": {
            args.server_name: {
                "command": "python",
                "args": [str(server_path)],
            }
        }
    }
    config_path = output / "mcp.config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    package = {
        "schema_version": "xlab.mcp_package.v1",
        "protocol_version": "2025-11-25",
        "tool_name": args.tool_name,
        "server": str(server_path),
        "config": str(config_path),
        "input_schema": str(schema_path),
        "example_input": str(example_path),
        "runner_command": runner_command,
    }
    manifest_path = output / "mcp-package.json"
    manifest_path.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(manifest_path), "server": str(server_path)}))


if __name__ == "__main__":
    main()
