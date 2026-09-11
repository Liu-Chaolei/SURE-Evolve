#!/usr/bin/env python3
import argparse
import json
import selectors
import subprocess
from pathlib import Path


def send(process: subprocess.Popen, message: dict) -> None:
    process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
    process.stdin.flush()


def receive(process: subprocess.Popen, timeout: float) -> dict:
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    if not selector.select(timeout):
        raise TimeoutError(f"MCP server did not respond within {timeout} seconds")
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("MCP server closed stdout before responding")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("MCP response is not a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()

    package = json.loads(Path(args.package).read_text(encoding="utf-8"))
    example = json.loads(Path(package["example_input"]).read_text(encoding="utf-8"))
    process = subprocess.Popen(
        ["python", package["server"]],
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    responses = []
    try:
        send(process, {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "xlab-mcp-validate", "version": "1.0.0"},
            },
        })
        initialize = receive(process, args.timeout)
        responses.append(initialize)
        send(process, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        send(process, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        listed = receive(process, args.timeout)
        responses.append(listed)
        send(process, {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": package["tool_name"], "arguments": example},
        })
        called = receive(process, args.timeout)
        responses.append(called)
        initialized_ok = initialize.get("result", {}).get("protocolVersion") == "2025-11-25"
        tools = listed.get("result", {}).get("tools", [])
        listed_ok = any(tool.get("name") == package["tool_name"] for tool in tools)
        call_result = called.get("result", {})
        called_ok = isinstance(call_result.get("content"), list) and call_result.get("isError") is False
        report = {
            "schema_version": "xlab.mcp_validation.v1",
            "protocol_version": "2025-11-25",
            "tool_name": package["tool_name"],
            "initialized": initialized_ok,
            "listed": listed_ok,
            "called": called_ok,
            "passed": initialized_ok and listed_ok and called_ok,
            "responses": responses,
        }
    finally:
        process.terminate()
        try:
            _, stderr = process.communicate(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            _, stderr = process.communicate()
    report["stderr"] = stderr
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError(f"MCP smoke test failed; see {output}")
    print(json.dumps({"output": str(output), "passed": True}))


if __name__ == "__main__":
    main()
