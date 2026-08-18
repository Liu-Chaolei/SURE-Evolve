"""Append immutable run records under one repository-local model result root."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator

from sure_eval.storage import StorageConfig, load_storage_config, validate_model_id

from .schema import RESULT_INDEX_SCHEMA, file_sha256, validate_result_index
from .views import refresh_derived_report_views


PUBLICATION_MANIFEST_SCHEMA = "sure.eval.publication_manifest.v2"
RESULT_DELTA_SCHEMA = "sure.eval.result_delta.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_run_id(run_id: str) -> str:
    return validate_model_id(run_id)


class AppendOnlyResultStore:
    """Keep immutable runs and atomically update only the model-level index."""

    def __init__(
        self,
        model_id: str,
        model_artifact_sha256: str,
        *,
        storage: StorageConfig | None = None,
    ) -> None:
        self.storage = storage or load_storage_config()
        self.model_id = validate_model_id(model_id)
        self.model_artifact_sha256 = model_artifact_sha256
        self.model_root = self.storage.staged_results(self.model_id, create=True)
        self.index_path = self.model_root / "result_index.json"
        self.lock_path = self.model_root / ".result_index.lock"
        self._initialize_index()

    def reserve_inference_run(self, run_id: str) -> Path:
        return self._reserve("inference_runs", run_id)

    def reserve_evaluation_run(self, run_id: str) -> Path:
        return self._reserve("evaluation_runs", run_id)

    def write_json_once(self, path: str | Path, payload: dict[str, Any]) -> Path:
        target = self.storage.assert_staging_write_path(Path(path))
        try:
            with target.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
        except FileExistsError as exc:
            raise FileExistsError(f"append-only artifact already exists: {target}") from exc
        return target

    def append_inference_entry(self, entry: dict[str, Any]) -> None:
        self._append_entry("inference_runs", entry)

    def append_evaluation_entry(self, entry: dict[str, Any]) -> None:
        self._append_entry("evaluation_runs", entry)

    def append_published_inference_evaluation(
        self,
        entry: dict[str, Any],
        *,
        published_index_path: str | Path,
    ) -> None:
        """Stage an evaluation whose inference already lives in verified NFS results."""

        index_path = self.storage.assert_registry_read_path(published_index_path)
        expected_path = self.storage.published_results(self.model_id) / "result_index.json"
        if index_path != expected_path.resolve(strict=True):
            raise ValueError("published inference must come from the canonical model result index")
        published = validate_result_index(
            expected_path.parent,
            expected_model_id=self.model_id,
            require_verified=True,
        )
        if published["model"]["artifact_sha256"] != self.model_artifact_sha256:
            raise ValueError("published result index belongs to a different model artifact")
        inference_id = str(entry.get("inference_id", ""))
        if not any(item.get("id") == inference_id for item in published["inference_runs"]):
            raise ValueError(f"published inference run does not exist: {inference_id}")

        with self._locked_delta(expected_path) as payload:
            entries = payload["additions"]["evaluation_runs"]
            entry_id = _validate_run_id(str(entry.get("id", "")))
            if any(item.get("id") == entry_id for item in entries):
                raise ValueError(f"result delta already contains evaluation id {entry_id!r}")
            record = dict(entry)
            record.setdefault("created_at", _utc_now())
            entries.append(record)
            self._atomic_json_write(self.model_root / "result_delta.json", payload)

    def write_reuse_decision(self, decision_id: str, payload: dict[str, Any]) -> Path:
        directory = self.model_root / "reuse_decisions"
        directory.mkdir(parents=True, exist_ok=True)
        return self.write_json_once(directory / f"{_validate_run_id(decision_id)}.json", payload)

    def write_publication_manifest(self) -> Path:
        with self._locked_file(self.model_root / ".publication_manifest.lock"):
            strategy, delta_path, base = self._refresh_publication_delta()
            derived_reports = refresh_derived_report_views(
                storage=self.storage,
                model_id=self.model_id,
                model_artifact_sha256=self.model_artifact_sha256,
                model_root=self.model_root,
                result_delta_path=delta_path,
            )
            payload = {
                "schema": PUBLICATION_MANIFEST_SCHEMA,
                "status": "pending_human_review",
                "model_id": self.model_id,
                "source_model_root": str(self.model_root.relative_to(self.storage.repo_root)),
                "target_registry_root": str(
                    self.storage.declared_published_results(self.model_id)
                ),
                "result_index": str(self.index_path.relative_to(self.model_root)),
                "result_delta": (
                    str(delta_path.relative_to(self.model_root))
                    if delta_path is not None
                    else None
                ),
                "derived_reports": derived_reports,
                "publication_strategy": strategy,
                "base_published_index": base,
                "created_at": _utc_now(),
                "automatic_publish_allowed": False,
            }
            path = self.model_root / "publication_manifest.json"
            if path.exists():
                self._atomic_json_write(path, payload)
                return path
            return self.write_json_once(path, payload)

    def _refresh_publication_delta(
        self,
    ) -> tuple[str, Path | None, dict[str, str] | None]:
        with self._locked_file(self.model_root / ".result_delta.lock"):
            return self._refresh_publication_delta_unlocked()

    def _refresh_publication_delta_unlocked(
        self,
    ) -> tuple[str, Path | None, dict[str, str] | None]:
        published_root = self.storage.published_results(self.model_id, require_exists=False)
        published_index_path = published_root / "result_index.json"
        if not published_root.exists():
            return "initialize_index", None, None
        if not published_index_path.is_file():
            raise ValueError(
                f"published model result directory has no result_index.json: {published_root}"
            )
        published = validate_result_index(
            published_root,
            expected_model_id=self.model_id,
            require_verified=True,
        )
        if published["model"]["artifact_sha256"] != self.model_artifact_sha256:
            raise ValueError("published result index belongs to a different model artifact")

        local = json.loads(self.index_path.read_text(encoding="utf-8"))
        local_evaluations = list(local["evaluation_runs"])
        local_evaluation_ids = {item["id"] for item in local_evaluations}
        delta_path = self.model_root / "result_delta.json"
        existing_evaluations: list[dict[str, Any]] = []
        delta_created_at = _utc_now()
        if delta_path.is_file():
            existing = json.loads(delta_path.read_text(encoding="utf-8"))
            if existing.get("schema") != RESULT_DELTA_SCHEMA:
                raise ValueError("unsupported local result delta schema")
            if existing.get("status") != "pending_human_review":
                raise ValueError("local result delta must be pending human review")
            if existing.get("automatic_publish_allowed") is not False:
                raise ValueError("local result delta cannot allow automatic publication")
            if existing.get("model") != {
                "id": self.model_id,
                "artifact_sha256": self.model_artifact_sha256,
            }:
                raise ValueError("local result delta belongs to a different model artifact")
            expected_base = {
                "path": str(
                    self.storage.declared_published_results(self.model_id)
                    / "result_index.json"
                ),
                "sha256": file_sha256(published_index_path),
            }
            if existing.get("base_published_index") != expected_base:
                raise ValueError("published result index changed since the delta was created")
            additions = existing.get("additions")
            if not isinstance(additions, dict) or set(additions) != {
                "inference_runs",
                "evaluation_runs",
            }:
                raise ValueError("local result delta additions are invalid")
            delta_created_at = str(existing.get("created_at", ""))
            if not delta_created_at:
                raise ValueError("local result delta created_at is missing")
            existing_evaluations = [
                item
                for item in additions["evaluation_runs"]
                if item.get("id") not in local_evaluation_ids
            ]

        published_inference_ids = {item["id"] for item in published["inference_runs"]}
        local_inference_ids = {item["id"] for item in local["inference_runs"]}
        if published_inference_ids & local_inference_ids:
            raise ValueError("local inference id collides with a published inference id")
        published_evaluation_ids = {item["id"] for item in published["evaluation_runs"]}
        all_new_evaluations = local_evaluations + existing_evaluations
        new_evaluation_ids = [item["id"] for item in all_new_evaluations]
        if len(new_evaluation_ids) != len(set(new_evaluation_ids)):
            raise ValueError("duplicate evaluation id across staged index and result delta")
        if published_evaluation_ids & set(new_evaluation_ids):
            raise ValueError("local evaluation id collides with a published evaluation id")

        base = {
            "path": str(
                self.storage.declared_published_results(self.model_id)
                / "result_index.json"
            ),
            "sha256": file_sha256(published_index_path),
        }
        delta = {
            "schema": RESULT_DELTA_SCHEMA,
            "status": "pending_human_review",
            "model": {
                "id": self.model_id,
                "artifact_sha256": self.model_artifact_sha256,
            },
            "base_published_index": base,
            "additions": {
                "inference_runs": list(local["inference_runs"]),
                "evaluation_runs": all_new_evaluations,
            },
            "created_at": delta_created_at,
            "automatic_publish_allowed": False,
        }
        self._atomic_json_write(delta_path, delta)
        return "merge_delta", delta_path, base

    def _initialize_index(self) -> None:
        if self.index_path.exists():
            current = json.loads(self.index_path.read_text(encoding="utf-8"))
            if current.get("model") != {
                "id": self.model_id,
                "artifact_sha256": self.model_artifact_sha256,
            }:
                raise ValueError(
                    "existing local result index belongs to a different model artifact"
                )
            return
        payload = {
            "schema": RESULT_INDEX_SCHEMA,
            "publication": {"status": "pending_human_review"},
            "model": {
                "id": self.model_id,
                "artifact_sha256": self.model_artifact_sha256,
            },
            "inference_runs": [],
            "evaluation_runs": [],
        }
        self._atomic_json_write(self.index_path, payload)

    def _reserve(self, collection: str, run_id: str) -> Path:
        value = _validate_run_id(run_id)
        target = self.storage.assert_staging_write_path(self.model_root / collection / value)
        target.mkdir(parents=True, exist_ok=False)
        return target

    def _append_entry(self, collection: str, entry: dict[str, Any]) -> None:
        entry_id = _validate_run_id(str(entry.get("id", "")))
        with self._locked_index() as payload:
            entries = payload[collection]
            if any(item.get("id") == entry_id for item in entries):
                raise ValueError(f"{collection} already contains id {entry_id!r}")
            record = dict(entry)
            record.setdefault("created_at", _utc_now())
            entries.append(record)
            self._atomic_json_write(self.index_path, payload)

    @contextmanager
    def _locked_index(self) -> Iterator[dict[str, Any]]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                payload = json.loads(self.index_path.read_text(encoding="utf-8"))
                yield payload
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def _locked_delta(self, published_index_path: Path) -> Iterator[dict[str, Any]]:
        delta_lock = self.model_root / ".result_delta.lock"
        with delta_lock.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                delta_path = self.model_root / "result_delta.json"
                if delta_path.is_file():
                    payload = json.loads(delta_path.read_text(encoding="utf-8"))
                    expected_sha = file_sha256(published_index_path)
                    if payload.get("base_published_index", {}).get("sha256") != expected_sha:
                        raise ValueError(
                            "published result index changed since the delta was created"
                        )
                else:
                    payload = {
                        "schema": RESULT_DELTA_SCHEMA,
                        "status": "pending_human_review",
                        "model": {
                            "id": self.model_id,
                            "artifact_sha256": self.model_artifact_sha256,
                        },
                        "base_published_index": {
                            "path": str(
                                self.storage.declared_published_results(self.model_id)
                                / "result_index.json"
                            ),
                            "sha256": file_sha256(published_index_path),
                        },
                        "additions": {"inference_runs": [], "evaluation_runs": []},
                        "created_at": _utc_now(),
                        "automatic_publish_allowed": False,
                    }
                yield payload
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def _locked_file(self, path: Path) -> Iterator[None]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _atomic_json_write(self, path: Path, payload: dict[str, Any]) -> None:
        target = self.storage.assert_staging_write_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
