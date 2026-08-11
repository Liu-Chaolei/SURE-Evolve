from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import ClassVar
from urllib.parse import quote

from paper_library.application.scanner import SourceDocument
from paper_library.schemas.common import utc_now


class StageName(StrEnum):
    SOURCE = "source"
    PARSE = "parse"
    CANDIDATE_DESIGNS = "candidate_designs"
    EVIDENCE = "evidence"
    CARDS = "cards"
    VALIDATE = "validate"
    PUBLISH = "publish"
    LEXICAL_INDEX = "lexical_index"
    VECTOR_INDEX = "vector_index"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_TERMINAL = "failed_terminal"
    STALE = "stale"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class StageRecord:
    document_id: str
    stage: StageName
    status: StageStatus
    cache_key: str
    input_hash: str
    output_uri: str | None
    output_sha256: str | None
    output_record_count: int | None
    attempt: int
    error_code: str | None
    error_message: str | None


class BuildState:
    _DEPENDENTS: ClassVar[dict[StageName, tuple[StageName, ...]]] = {
        StageName.PARSE: (
            StageName.CANDIDATE_DESIGNS,
            StageName.EVIDENCE,
            StageName.CARDS,
            StageName.VALIDATE,
            StageName.PUBLISH,
            StageName.LEXICAL_INDEX,
            StageName.VECTOR_INDEX,
        ),
        StageName.CANDIDATE_DESIGNS: (
            StageName.EVIDENCE,
            StageName.CARDS,
            StageName.VALIDATE,
            StageName.PUBLISH,
            StageName.LEXICAL_INDEX,
            StageName.VECTOR_INDEX,
        ),
        StageName.EVIDENCE: (
            StageName.CARDS,
            StageName.VALIDATE,
            StageName.PUBLISH,
            StageName.LEXICAL_INDEX,
            StageName.VECTOR_INDEX,
        ),
        StageName.CARDS: (
            StageName.VALIDATE,
            StageName.PUBLISH,
            StageName.LEXICAL_INDEX,
            StageName.VECTOR_INDEX,
        ),
    }

    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def connect_readonly(self) -> Iterator[sqlite3.Connection]:
        uri = f"file:{quote(str(self.path.resolve()), safe='/')}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA query_only = ON")
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    document_id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    pdf_sha256 TEXT NOT NULL,
                    primary_path TEXT NOT NULL,
                    aliases_json TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    present INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS stages (
                    document_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    output_uri TEXT,
                    output_sha256 TEXT,
                    output_record_count INTEGER,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    error_code TEXT,
                    error_message TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (document_id, stage)
                );
                CREATE INDEX IF NOT EXISTS idx_stages_status ON stages(stage, status);
                """
            )
            columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(stages)")
            }
            if "output_sha256" not in columns:
                connection.execute("ALTER TABLE stages ADD COLUMN output_sha256 TEXT")
            if "output_record_count" not in columns:
                connection.execute(
                    "ALTER TABLE stages ADD COLUMN output_record_count INTEGER"
                )

    def sync_sources(self, sources: Sequence[SourceDocument]) -> tuple[int, int]:
        import json

        now = utc_now().isoformat()
        seen: set[str] = set()
        with self.connect() as connection:
            for source in sources:
                document_id = str(source.document_id)
                seen.add(document_id)
                connection.execute(
                    """
                    INSERT INTO sources (
                        document_id, paper_id, pdf_sha256, primary_path, aliases_json,
                        domain, size, mtime_ns, present, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                    ON CONFLICT(document_id) DO UPDATE SET
                        paper_id=excluded.paper_id,
                        primary_path=excluded.primary_path,
                        aliases_json=excluded.aliases_json,
                        domain=excluded.domain,
                        size=excluded.size,
                        mtime_ns=excluded.mtime_ns,
                        present=1,
                        updated_at=excluded.updated_at
                    """,
                    (
                        document_id,
                        str(source.paper_id),
                        str(source.pdf_sha256),
                        str(source.primary_path),
                        json.dumps(source.aliases, ensure_ascii=False),
                        str(source.domain),
                        int(source.size),
                        int(source.mtime_ns),
                        now,
                    ),
                )
            if seen:
                placeholders = ",".join("?" for _ in seen)
                cursor = connection.execute(
                    f"UPDATE sources SET present=0, updated_at=? "
                    f"WHERE document_id NOT IN ({placeholders}) AND present=1",
                    (now, *sorted(seen)),
                )
            else:
                cursor = connection.execute(
                    "UPDATE sources SET present=0, updated_at=? WHERE present=1",
                    (now,),
                )
            missing = cursor.rowcount
        return len(seen), missing

    def present_document_ids(self) -> set[str]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT document_id FROM sources WHERE present=1"
            ).fetchall()
        return {str(row["document_id"]) for row in rows}

    def present_stage_records(
        self,
        stage: StageName,
        *,
        readonly: bool = False,
    ) -> list[StageRecord]:
        manager = self.connect_readonly if readonly else self.connect
        with manager() as connection:
            rows = connection.execute(
                """
                SELECT stages.* FROM stages
                JOIN sources ON sources.document_id = stages.document_id
                WHERE stages.stage=? AND sources.present=1
                ORDER BY stages.document_id
                """,
                (stage.value,),
            ).fetchall()
        return [self._stage_record(row) for row in rows]

    @staticmethod
    def _stage_record(row: sqlite3.Row) -> StageRecord:
        return StageRecord(
            document_id=row["document_id"],
            stage=StageName(row["stage"]),
            status=StageStatus(row["status"]),
            cache_key=row["cache_key"],
            input_hash=row["input_hash"],
            output_uri=row["output_uri"],
            output_sha256=row["output_sha256"],
            output_record_count=row["output_record_count"],
            attempt=row["attempt"],
            error_code=row["error_code"],
            error_message=row["error_message"],
        )

    def get_stage(
        self,
        document_id: str,
        stage: StageName,
        *,
        readonly: bool = False,
    ) -> StageRecord | None:
        manager = self.connect_readonly if readonly else self.connect
        with manager() as connection:
            row = connection.execute(
                "SELECT * FROM stages WHERE document_id=? AND stage=?",
                (document_id, stage.value),
            ).fetchone()
        if row is None:
            return None
        return self._stage_record(row)

    def stage_is_current(self, document_id: str, stage: StageName, cache_key: str) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT status, cache_key FROM stages WHERE document_id=? AND stage=?",
                (document_id, stage.value),
            ).fetchone()
        return bool(
            row
            and row["status"] == StageStatus.SUCCEEDED.value
            and row["cache_key"] == cache_key
        )

    def start_stage(
        self,
        document_id: str,
        stage: StageName,
        cache_key: str,
        input_hash: str,
    ) -> None:
        now = utc_now().isoformat()
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT cache_key FROM stages WHERE document_id=? AND stage=?",
                (document_id, stage.value),
            ).fetchone()
            if existing is not None and existing["cache_key"] != cache_key:
                dependents = self._DEPENDENTS.get(stage, ())
                if dependents:
                    placeholders = ",".join("?" for _ in dependents)
                    connection.execute(
                        f"UPDATE stages SET status=?, updated_at=? "
                        f"WHERE document_id=? AND stage IN ({placeholders})",
                        (
                            StageStatus.STALE.value,
                            now,
                            document_id,
                            *(dependent.value for dependent in dependents),
                        ),
                    )
            connection.execute(
                """
                INSERT INTO stages (
                    document_id, stage, status, cache_key, input_hash, attempt, updated_at
                ) VALUES (?, ?, ?, ?, ?, 1, ?)
                ON CONFLICT(document_id, stage) DO UPDATE SET
                    status=excluded.status,
                    cache_key=excluded.cache_key,
                    input_hash=excluded.input_hash,
                    output_uri=NULL,
                    output_sha256=NULL,
                    output_record_count=NULL,
                    attempt=stages.attempt + 1,
                    error_code=NULL,
                    error_message=NULL,
                    updated_at=excluded.updated_at
                """,
                (
                    document_id,
                    stage.value,
                    StageStatus.RUNNING.value,
                    cache_key,
                    input_hash,
                    now,
                ),
            )

    def finish_stage(
        self,
        document_id: str,
        stage: StageName,
        *,
        output_uri: str,
        output_sha256: str,
        output_record_count: int,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE stages SET status=?, output_uri=?, output_sha256=?,
                    output_record_count=?, updated_at=?
                WHERE document_id=? AND stage=?
                """,
                (
                    StageStatus.SUCCEEDED.value,
                    output_uri,
                    output_sha256,
                    output_record_count,
                    utc_now().isoformat(),
                    document_id,
                    stage.value,
                ),
            )

    def invalidate_stage(self, document_id: str, stage: StageName) -> None:
        now = utc_now().isoformat()
        stages = (stage, *self._DEPENDENTS.get(stage, ()))
        placeholders = ",".join("?" for _ in stages)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE stages SET status=?, updated_at=? "
                f"WHERE document_id=? AND stage IN ({placeholders})",
                (
                    StageStatus.STALE.value,
                    now,
                    document_id,
                    *(item.value for item in stages),
                ),
            )

    def fail_stage(
        self,
        document_id: str,
        stage: StageName,
        *,
        error_code: str,
        error_message: str,
        retryable: bool,
    ) -> None:
        status = (
            StageStatus.FAILED_RETRYABLE if retryable else StageStatus.FAILED_TERMINAL
        )
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE stages SET status=?, error_code=?, error_message=?, updated_at=?
                WHERE document_id=? AND stage=?
                """,
                (
                    status.value,
                    error_code,
                    error_message,
                    utc_now().isoformat(),
                    document_id,
                    stage.value,
                ),
            )

    def counts(self) -> dict[str, int]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT stage, status, COUNT(*) AS count FROM stages GROUP BY stage, status"
            ).fetchall()
        return {f"{row['stage']}:{row['status']}": row["count"] for row in rows}
