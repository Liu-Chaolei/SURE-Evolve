"""Explicit JSONL bridge for a configured XLab idea provider command."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .contracts import IdeaBatch, IdeaRequest, RoundResult, RoundSummary, validate_idea_batch
from .utils.fingerprints import canonical_json, digest, portable_path

_RECEIPT_SCHEMA = "sure.xlab_receipts.v1"
_PROTOCOL = "xlab.sure.jsonl.v1"
_RECEIPT_STATUSES = {"started", "incomplete", "published"}


class XlabIdeaClientError(RuntimeError):
    """Raised when the provider bridge cannot return a valid response."""


class XlabIdeaClient:
    """Call a configured provider using one request and one JSON response per process.

    The command is supplied by trusted deployment configuration and is never assembled
    from task text. A fresh process per operation keeps request boundaries explicit and
    avoids retrying an unconfirmed non-idempotent XLab execution.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: int = 3600,
        receipt_path: str | Path | None = None,
        workspace_root: str | Path | None = None,
        environment: dict[str, str] | None = None,
    ) -> None:
        if not command or any(not isinstance(part, str) or not part for part in command):
            raise ValueError("provider command must contain non-empty strings")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.command = tuple(command)
        self.environment = dict(environment or {})
        self.timeout_seconds = timeout_seconds
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else None
        self.receipt_path = self._resolve_receipt_path(receipt_path)
        self._operations: dict[str, dict[str, Any]] = {}
        self._load_receipts()

    def _resolve_receipt_path(self, receipt_path: str | Path | None) -> Path | None:
        if receipt_path is None:
            return None
        if self.workspace_root is None:
            raise ValueError("workspace_root is required when receipt_path is configured")
        relative = portable_path(receipt_path, self.workspace_root)
        return self.workspace_root / relative

    @staticmethod
    def _validate_operation(operation_id: str, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise XlabIdeaClientError("invalid XLab operation receipt entry")
        if value.get("operation_id") != operation_id:
            raise XlabIdeaClientError("XLab operation receipt identity mismatch")
        if value.get("operation") not in {"generate", "summarize"}:
            raise XlabIdeaClientError("invalid XLab operation receipt operation")
        if value.get("status") not in _RECEIPT_STATUSES:
            raise XlabIdeaClientError("invalid XLab operation receipt status")
        request_digest = value.get("request_digest")
        if not isinstance(request_digest, str) or not request_digest.startswith("sha256:"):
            raise XlabIdeaClientError("invalid XLab operation request digest")
        if value.get("status") == "published":
            response = value.get("response")
            response_digest = value.get("response_digest")
            if not isinstance(response, dict) or not isinstance(response_digest, str):
                raise XlabIdeaClientError("published XLab receipt has no replay response")
            if digest(response) != response_digest:
                raise XlabIdeaClientError("XLab receipt response digest mismatch")
            if (
                response.get("protocol") != _PROTOCOL
                or response.get("operation") != value.get("operation")
                or response.get("operation_id") != operation_id
                or response.get("status") != "success"
                or not isinstance(response.get("payload"), dict)
            ):
                raise XlabIdeaClientError("published XLab receipt response is invalid")
        return dict(value)

    def _load_receipts(self) -> None:
        if self.receipt_path is None or not self.receipt_path.exists():
            return
        try:
            document = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise XlabIdeaClientError("invalid XLab operation receipt") from exc
        if not isinstance(document, dict) or document.get("schema_version") != _RECEIPT_SCHEMA:
            raise XlabIdeaClientError("unsupported XLab operation receipt")
        receipts = document.get("operations")
        if not isinstance(receipts, dict):
            raise XlabIdeaClientError("invalid XLab operation receipts")
        expected_digest = digest({"schema_version": _RECEIPT_SCHEMA, "operations": receipts})
        if document.get("document_digest") != expected_digest:
            raise XlabIdeaClientError("XLab operation receipt digest mismatch")
        self._operations = {
            str(operation_id): self._validate_operation(str(operation_id), value)
            for operation_id, value in receipts.items()
        }

    def _flush_receipts(self) -> None:
        if self.receipt_path is None:
            return
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        base = {"schema_version": _RECEIPT_SCHEMA, "operations": self._operations}
        document = {**base, "document_digest": digest(base)}
        fd, temporary = tempfile.mkstemp(prefix=f".{self.receipt_path.name}.", dir=self.receipt_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(canonical_json(document) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.receipt_path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def _save_operation(self, operation_id: str, operation: str, payload: dict[str, Any], **extra: Any) -> None:
        self._operations[operation_id] = {
            "operation_id": operation_id,
            "operation": operation,
            "request_digest": digest(payload),
            **extra,
        }
        self._flush_receipts()

    @staticmethod
    def _operation_id(operation: str, payload: dict[str, Any]) -> str:
        identity_field = "request_id" if operation == "generate" else "result_digest"
        operation_id = payload.get(identity_field)
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise XlabIdeaClientError(f"{operation} requires a non-empty {identity_field}")
        return operation_id

    def _call(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        operation_id = self._operation_id(operation, payload)
        request_digest = digest(payload)
        previous = self._operations.get(operation_id)
        if previous:
            if previous.get("operation") != operation or previous.get("request_digest") != request_digest:
                raise XlabIdeaClientError("operation identity conflicts with an existing request")
            if previous.get("status") == "published":
                return self._validate_operation(operation_id, previous)["response"]
            raise XlabIdeaClientError(
                "XLab operation is incomplete; reconcile it before any retry"
            )

        envelope = {
            "protocol": _PROTOCOL,
            "operation": operation,
            "operation_id": operation_id,
            "payload": payload,
        }
        self._save_operation(operation_id, operation, payload, status="started")
        try:
            completed = subprocess.run(
                self.command,
                input=canonical_json(envelope) + "\n",
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
                env={**os.environ, **self.environment},
            )
        except subprocess.TimeoutExpired as exc:
            self._save_operation(operation_id, operation, payload, status="incomplete")
            raise XlabIdeaClientError("XLab provider timed out; reconcile the operation before retrying") from exc
        except OSError as exc:
            self._save_operation(operation_id, operation, payload, status="incomplete", error=str(exc))
            raise XlabIdeaClientError(f"unable to start XLab provider: {exc}") from exc
        if completed.returncode != 0:
            self._save_operation(operation_id, operation, payload, status="incomplete", exit_code=completed.returncode)
            raise XlabIdeaClientError(f"XLab provider failed with exit code {completed.returncode}")

        frames = [line for line in completed.stdout.splitlines() if line.strip()]
        if len(frames) != 1:
            self._save_operation(operation_id, operation, payload, status="incomplete")
            raise XlabIdeaClientError("XLab provider must return exactly one JSON response")
        try:
            response = json.loads(frames[0])
        except json.JSONDecodeError as exc:
            self._save_operation(operation_id, operation, payload, status="incomplete")
            raise XlabIdeaClientError("XLab provider returned no valid JSON response") from exc
        if (
            not isinstance(response, dict)
            or response.get("protocol") != _PROTOCOL
            or response.get("operation") != operation
            or response.get("operation_id") != operation_id
            or response.get("status") != "success"
            or not isinstance(response.get("payload"), dict)
        ):
            self._save_operation(operation_id, operation, payload, status="incomplete")
            raise XlabIdeaClientError("XLab provider response has invalid protocol, identity, or status")
        self._save_operation(
            operation_id,
            operation,
            payload,
            status="published",
            response_digest=digest(response),
            response=response,
        )
        return response

    def _invalidate_published(self, operation_id: str, operation: str, payload: dict[str, Any]) -> None:
        self._save_operation(operation_id, operation, payload, status="incomplete", error="contract_validation")

    def generate(self, request: IdeaRequest) -> IdeaBatch:
        payload = asdict(request)
        response = self._call("generate", payload)
        try:
            from .contracts import IdeaItem, IdeaSpec

            raw_ideas = response["payload"]["ideas"]
            if not isinstance(raw_ideas, list) or any(not isinstance(item, dict) for item in raw_ideas):
                raise TypeError("ideas must be a list of objects")
            ideas = [IdeaItem(**{**item, "spec": IdeaSpec(**item["spec"])}) for item in raw_ideas]
            batch = IdeaBatch(**{**response["payload"], "ideas": ideas})
            validate_idea_batch(batch, request)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            self._invalidate_published(request.request_id, "generate", payload)
            raise XlabIdeaClientError("XLab generate response violates sure.idea_batch.v1") from exc
        return batch

    def summarize(self, result: RoundResult) -> RoundSummary:
        payload = asdict(result)
        response = self._call("summarize", payload)
        try:
            summary = RoundSummary(**response["payload"])
            if summary.schema_version != "sure.round_summary.v1":
                raise ValueError("unsupported round summary schema")
            if (
                summary.sure_run_id != result.sure_run_id
                or summary.search_mode != result.search_mode
                or summary.round_index != result.round_index
                or summary.axis != result.axis
            ):
                raise ValueError("round summary lineage does not match result")
        except (KeyError, TypeError, ValueError) as exc:
            self._invalidate_published(result.result_digest, "summarize", payload)
            raise XlabIdeaClientError("XLab summarize response violates sure.round_summary.v1") from exc
        return summary

    def reconcile(self, operation_id: str) -> dict[str, Any]:
        operation = self._operations.get(operation_id)
        if operation is None:
            return {"status": "unknown", "operation_id": operation_id}
        return {key: value for key, value in operation.items() if key != "response"}

    def retry_summary(self, result: RoundResult) -> RoundSummary:
        """Explicitly replay only summary publication, never candidate generation."""
        previous = self._operations.get(result.result_digest)
        if previous and previous.get("status") != "published":
            if previous.get("operation") != "summarize" or previous.get("request_digest") != digest(asdict(result)):
                raise XlabIdeaClientError("summary retry conflicts with the original request")
            del self._operations[result.result_digest]
            self._flush_receipts()
        return self.summarize(result)

    def close(self) -> None:
        self._flush_receipts()


__all__ = ["XlabIdeaClient", "XlabIdeaClientError"]
