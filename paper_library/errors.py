from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    CONFIG_INVALID = "config_invalid"
    SOURCE_INVALID = "source_invalid"
    PARSE_FAILED = "parse_failed"
    EXTRACTION_FAILED = "extraction_failed"
    GROUNDING_FAILED = "grounding_failed"
    VALIDATION_FAILED = "validation_failed"
    STORAGE_FAILED = "storage_failed"
    INDEX_FAILED = "index_failed"
    OPTIONAL_DEPENDENCY_MISSING = "optional_dependency_missing"


class PaperLibraryError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class BuildStageError(PaperLibraryError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code=ErrorCode.EXTRACTION_FAILED)


class ConfigurationError(PaperLibraryError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code=ErrorCode.CONFIG_INVALID)
