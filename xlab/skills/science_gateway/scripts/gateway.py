#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from storage import Store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    subparsers = parser.add_subparsers(dest="action", required=True)
    create = subparsers.add_parser("create-thread")
    create.add_argument("--title", required=True)
    create.add_argument("--sender", required=True)
    create.add_argument("--task-id")
    post = subparsers.add_parser("post")
    post.add_argument("--thread", required=True)
    post.add_argument("--sender", required=True)
    post.add_argument("--body", required=True)
    post.add_argument("--reply-to")
    post.add_argument("--task-id")
    post.add_argument("--run-id")
    post.add_argument("--artifact", action="append", default=[])
    direct = subparsers.add_parser("direct")
    direct.add_argument("--recipient", required=True)
    direct.add_argument("--sender", required=True)
    direct.add_argument("--body", required=True)
    direct.add_argument("--task-id")
    direct.add_argument("--run-id")
    direct.add_argument("--artifact", action="append", default=[])
    list_messages = subparsers.add_parser("list")
    list_messages.add_argument("--thread")
    list_messages.add_argument("--recipient")
    export = subparsers.add_parser("export")
    export.add_argument("--output", required=True)
    args = parser.parse_args()

    store = Store(args.database)
    if args.action == "create-thread":
        result = store.create_thread(args.title, args.sender, args.task_id)
    elif args.action == "post":
        result = store.post(
            "forum", args.sender, args.body, thread_id=args.thread, reply_to=args.reply_to,
            task_id=args.task_id, run_id=args.run_id, artifacts=args.artifact,
        )
    elif args.action == "direct":
        result = store.post(
            "direct", args.sender, args.body, recipient=args.recipient,
            task_id=args.task_id, run_id=args.run_id, artifacts=args.artifact,
        )
    elif args.action == "list":
        result = {"messages": store.messages(args.thread, args.recipient)}
    else:
        threads = store.threads()
        messages = store.messages()
        result = {
            "schema_version": "xlab.communication_manifest.v1",
            "database": store.path,
            "thread_count": len(threads),
            "message_count": len(messages),
            "threads": threads,
            "messages": messages,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
