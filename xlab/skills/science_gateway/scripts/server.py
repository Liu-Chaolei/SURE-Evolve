#!/usr/bin/env python3
import argparse
import json
import sys
from storage import Store


PROTOCOL_VERSION = "2025-11-25"


def tool_definitions() -> list[dict]:
    return [
        {
            "name": "forum_create",
            "description": "Create a durable research discussion thread.",
            "inputSchema": {
                "type": "object",
                "required": ["title", "sender"],
                "properties": {
                    "title": {"type": "string"},
                    "sender": {"type": "string"},
                    "task_id": {"type": "string"},
                },
            },
        },
        {
            "name": "forum_post",
            "description": "Post or reply to a durable forum thread.",
            "inputSchema": {
                "type": "object",
                "required": ["thread_id", "sender", "body"],
                "properties": {
                    "thread_id": {"type": "string"},
                    "sender": {"type": "string"},
                    "body": {"type": "string"},
                    "reply_to": {"type": "string"},
                    "task_id": {"type": "string"},
                    "run_id": {"type": "string"},
                    "artifacts": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        {
            "name": "direct_send",
            "description": "Send a durable direct message.",
            "inputSchema": {
                "type": "object",
                "required": ["recipient", "sender", "body"],
                "properties": {
                    "recipient": {"type": "string"},
                    "sender": {"type": "string"},
                    "body": {"type": "string"},
                    "task_id": {"type": "string"},
                    "run_id": {"type": "string"},
                    "artifacts": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        {
            "name": "message_list",
            "description": "List forum or direct messages.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "thread_id": {"type": "string"},
                    "recipient": {"type": "string"},
                },
            },
        },
    ]


def invoke(store: Store, name: str, values: dict) -> object:
    if name == "forum_create":
        return store.create_thread(values["title"], values["sender"], values.get("task_id"))
    if name == "forum_post":
        return store.post(
            "forum", values["sender"], values["body"], thread_id=values["thread_id"],
            reply_to=values.get("reply_to"), task_id=values.get("task_id"),
            run_id=values.get("run_id"), artifacts=values.get("artifacts"),
        )
    if name == "direct_send":
        return store.post(
            "direct", values["sender"], values["body"], recipient=values["recipient"],
            task_id=values.get("task_id"), run_id=values.get("run_id"), artifacts=values.get("artifacts"),
        )
    if name == "message_list":
        return store.messages(values.get("thread_id"), values.get("recipient"))
    raise ValueError(f"unknown tool: {name}")


def write_response(request_id: object, result: object = None, error: dict | None = None) -> None:
    value = {"jsonrpc": "2.0", "id": request_id}
    value["error" if error else "result"] = error if error else result
    sys.stdout.write(json.dumps(value, separators=(",", ":"), ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    args = parser.parse_args()
    store = Store(args.database)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            method = request.get("method")
            request_id = request.get("id")
            if method == "notifications/initialized":
                continue
            if method == "initialize":
                write_response(request_id, {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "xlab-science-gateway", "version": "1.0.0"},
                })
            elif method == "ping":
                write_response(request_id, {})
            elif method == "tools/list":
                write_response(request_id, {"tools": tool_definitions()})
            elif method == "tools/call":
                params = request.get("params", {})
                try:
                    result = invoke(store, params.get("name"), params.get("arguments", {}))
                    write_response(request_id, {
                        "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
                        "structuredContent": result,
                        "isError": False,
                    })
                except (KeyError, TypeError, ValueError) as error:
                    write_response(request_id, {
                        "content": [{"type": "text", "text": str(error)}],
                        "isError": True,
                    })
            elif request_id is not None:
                write_response(request_id, error={"code": -32601, "message": f"Method not found: {method}"})
        except Exception as error:
            sys.stderr.write(f"science gateway error: {error}\n")


if __name__ == "__main__":
    main()
