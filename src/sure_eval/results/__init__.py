"""Append-only result staging and verified NFS result reuse."""

from .registry import (
    ReuseDecision,
    ReuseRequest,
    ResultRegistry,
    ResultRegistryError,
)
from .schema import (
    RESULT_INDEX_SCHEMA,
    ResultIndexError,
    file_sha256,
    validate_result_index,
)
from .store import AppendOnlyResultStore, RESULT_DELTA_SCHEMA
from .views import DerivedReportError, REPORT_ROW_SCHEMA, refresh_derived_report_views

__all__ = [
    "AppendOnlyResultStore",
    "DerivedReportError",
    "REPORT_ROW_SCHEMA",
    "RESULT_INDEX_SCHEMA",
    "RESULT_DELTA_SCHEMA",
    "ReuseDecision",
    "ReuseRequest",
    "ResultIndexError",
    "ResultRegistry",
    "ResultRegistryError",
    "file_sha256",
    "refresh_derived_report_views",
    "validate_result_index",
]
