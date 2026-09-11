#!/usr/bin/env python3
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: str):
        self.path = str(Path(path).resolve())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS threads (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                task_id TEXT,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                channel TEXT NOT NULL CHECK(channel IN ('forum', 'direct')),
                thread_id TEXT,
                reply_to TEXT,
                sender TEXT NOT NULL,
                recipient TEXT,
                body TEXT NOT NULL,
                task_id TEXT,
                run_id TEXT,
                artifacts_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(thread_id) REFERENCES threads(id),
                FOREIGN KEY(reply_to) REFERENCES messages(id)
            );
        """)
        self.connection.commit()

    def create_thread(self, title: str, created_by: str, task_id: str | None = None) -> dict:
        thread = {
            "id": f"thread-{uuid.uuid4().hex[:12]}",
            "title": title,
            "task_id": task_id,
            "created_by": created_by,
            "created_at": now(),
        }
        self.connection.execute(
            "INSERT INTO threads (id, title, task_id, created_by, created_at) VALUES (?, ?, ?, ?, ?)",
            tuple(thread.values()),
        )
        self.connection.commit()
        return thread

    def post(
        self,
        channel: str,
        sender: str,
        body: str,
        thread_id: str | None = None,
        recipient: str | None = None,
        reply_to: str | None = None,
        task_id: str | None = None,
        run_id: str | None = None,
        artifacts: list[str] | None = None,
    ) -> dict:
        if channel == "forum" and not thread_id:
            raise ValueError("forum message requires thread_id")
        if channel == "direct" and not recipient:
            raise ValueError("direct message requires recipient")
        if thread_id and self.connection.execute("SELECT 1 FROM threads WHERE id = ?", (thread_id,)).fetchone() is None:
            raise ValueError(f"thread does not exist: {thread_id}")
        message = {
            "id": f"message-{uuid.uuid4().hex[:12]}",
            "channel": channel,
            "thread_id": thread_id,
            "reply_to": reply_to,
            "sender": sender,
            "recipient": recipient,
            "body": body,
            "task_id": task_id,
            "run_id": run_id,
            "artifacts": artifacts or [],
            "created_at": now(),
        }
        self.connection.execute(
            """INSERT INTO messages
            (id, channel, thread_id, reply_to, sender, recipient, body, task_id, run_id, artifacts_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                message["id"], channel, thread_id, reply_to, sender, recipient, body,
                task_id, run_id, json.dumps(message["artifacts"]), message["created_at"],
            ),
        )
        self.connection.commit()
        stored = self.connection.execute("SELECT 1 FROM messages WHERE id = ?", (message["id"],)).fetchone()
        if stored is None:
            raise RuntimeError("message write was not readable after commit")
        return message

    def threads(self) -> list[dict]:
        return [dict(row) for row in self.connection.execute("SELECT * FROM threads ORDER BY created_at, id")]

    def messages(self, thread_id: str | None = None, recipient: str | None = None) -> list[dict]:
        query = "SELECT * FROM messages"
        values = []
        clauses = []
        if thread_id:
            clauses.append("thread_id = ?")
            values.append(thread_id)
        if recipient:
            clauses.append("recipient = ?")
            values.append(recipient)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at, id"
        result = []
        for row in self.connection.execute(query, values):
            value = dict(row)
            value["artifacts"] = json.loads(value.pop("artifacts_json"))
            result.append(value)
        return result
