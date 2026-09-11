"""Atomic durable history projection for SURE/XLab round boundaries."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .utils.fingerprints import canonical_json, digest, redact


class XlabHistoryJournal:
    """Persist compact, redacted round lineage without embedding secrets in state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.records: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid XLab history journal: {self.path}") from exc
        if not isinstance(value, dict) or value.get("schema_version") != "sure.xlab_history.v1":
            raise ValueError(f"unsupported XLab history journal: {self.path}")
        records = value.get("records")
        if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
            raise ValueError(f"invalid XLab history records: {self.path}")
        for record in records:
            expected = digest({"event": record.get("event"), "payload": record.get("payload")})
            if record.get("record_digest") != expected:
                raise ValueError(f"XLab history digest mismatch: {self.path}")
        self.records = list(records)

    def append(self, event: str, payload: dict[str, Any]) -> str:
        redacted_payload = redact(payload)
        record = {
            "event": event,
            "payload": redacted_payload,
            "record_digest": digest({"event": event, "payload": redacted_payload}),
        }
        if any(item.get("record_digest") == record["record_digest"] for item in self.records):
            return str(record["record_digest"])
        self.records.append(record)
        self.flush()
        return str(record["record_digest"])

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {"schema_version": "sure.xlab_history.v1", "records": self.records}
        fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(canonical_json(document))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def references(self) -> list[str]:
        return [str(item["record_digest"]) for item in self.records if item.get("record_digest")]

    def snapshot(self) -> list[dict[str, Any]]:
        return list(self.records)

    def round_history(self) -> list[dict[str, Any]]:
        return [
            dict(item["payload"])
            for item in self.records
            if item.get("event") == "summary_published" and isinstance(item.get("payload"), dict)
        ]

    def summary_references(self) -> list[str]:
        references: list[str] = []
        for record in self.round_history():
            reference = record.get("summary_digest")
            if reference and str(reference) not in references:
                references.append(str(reference))
        return references


__all__ = ["XlabHistoryJournal"]
