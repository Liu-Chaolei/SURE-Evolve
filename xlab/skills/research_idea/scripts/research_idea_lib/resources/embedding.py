"""Production sentence-transformer inference over immutable local snapshots."""

from __future__ import annotations

import importlib
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, Sequence, runtime_checkable

from .manifest import ResourceDescriptor, sha256_file

ADAPTER_ALGORITHM = "sentence-transformers.local_snapshot.v1"
_EXPECTED_OUTCOME_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class ResourceExecutionError(ValueError):
    """Raised when a declared resource cannot be consumed safely."""


@runtime_checkable
class EmbeddingBackend(Protocol):
    dimension: int

    def encode(self, texts: Sequence[str]) -> Any: ...

    def usage(self) -> dict[str, object]: ...


class SentenceTransformerEmbedding:
    """Load an immutable declared snapshot with downloads and remote code disabled."""

    def __init__(
        self,
        descriptor: ResourceDescriptor,
        model_root: Path,
        *,
        expected_model_id: str | None = None,
        model_factory: Any | None = None,
    ) -> None:
        if descriptor.kind != "model":
            raise ResourceExecutionError("sentence-transformer inference requires a model descriptor")
        declared_dimension = int(descriptor.dimension_map.get("embedding") or 0)
        if declared_dimension <= 0:
            raise ResourceExecutionError(
                f"model {descriptor.resource_id!r} must declare a positive embedding dimension"
            )
        if expected_model_id is not None and _declared_model_id(descriptor) != expected_model_id:
            raise ResourceExecutionError(
                f"OutcomeRAG requires immutable model {expected_model_id!r}; "
                f"bundle declares {_declared_model_id(descriptor)!r}"
            )
        root = model_root.resolve()
        if not root.is_dir():
            raise ResourceExecutionError(f"model {descriptor.resource_id!r} snapshot is unreadable")
        _verify_snapshot(descriptor, root)
        factory = model_factory if model_factory is not None else _sentence_transformer_factory()
        try:
            self._model = factory(
                str(root),
                local_files_only=True,
                trust_remote_code=False,
            )
            actual_dimension = int(self._model.get_sentence_embedding_dimension())
        except ResourceExecutionError:
            raise
        except Exception as error:
            raise ResourceExecutionError(
                f"cannot load declared local sentence-transformer {descriptor.resource_id!r}: {error}"
            ) from error
        if actual_dimension != declared_dimension:
            raise ResourceExecutionError(
                f"model {descriptor.resource_id!r} runtime dimension {actual_dimension} "
                f"does not match declared dimension {declared_dimension}"
            )
        self.descriptor = descriptor
        self.model_root = root
        self.dimension = actual_dimension

    @classmethod
    def outcome(
        cls,
        descriptor: ResourceDescriptor,
        model_root: Path,
        *,
        model_factory: Any | None = None,
    ) -> "SentenceTransformerEmbedding":
        return cls(
            descriptor,
            model_root,
            expected_model_id=_EXPECTED_OUTCOME_MODEL,
            model_factory=model_factory,
        )

    def encode(self, texts: Sequence[str]) -> Any:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ResourceExecutionError("sentence-transformer input must contain non-empty strings")
        try:
            vectors = self._model.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            numpy = importlib.import_module("numpy")
            return numpy.ascontiguousarray(vectors, dtype="float32")
        except (ImportError, OSError) as error:
            raise ResourceExecutionError("NumPy runtime is unavailable for embedding inference") from error
        except Exception as error:
            raise ResourceExecutionError(f"sentence-transformer inference failed: {error}") from error

    def usage(self) -> dict[str, object]:
        return {
            "resource_id": self.descriptor.resource_id,
            "descriptor_digest": self.descriptor.digest,
            "declared_model_algorithm": self.descriptor.algorithm_version,
            "declared_model_id": _declared_model_id(self.descriptor),
            "adapter_algorithm": ADAPTER_ALGORITHM,
            "capability_consumed": "sentence_embedding",
            "embedding_dimension": self.dimension,
            "files_consumed": [
                {"logical_path": item.path, "sha256": item.sha256, "size": item.size}
                for item in self.descriptor.files
            ],
            "inference_backend": "sentence_transformers.SentenceTransformer",
            "local_files_only": True,
            "transformer_inference": True,
        }


def _declared_model_id(descriptor: ResourceDescriptor) -> str:
    return descriptor.provenance_map.get("model_id", descriptor.resource_id).strip()


def _sentence_transformer_factory() -> Any:
    try:
        module = importlib.import_module("sentence_transformers")
        return module.SentenceTransformer
    except (ImportError, OSError) as error:
        raise ResourceExecutionError(
            "sentence-transformers runtime is unavailable; install package-pinned ML dependencies"
        ) from error


def _verify_snapshot(descriptor: ResourceDescriptor, root: Path) -> None:
    for declared in descriptor.files:
        path = _contained_file(root, declared.path, descriptor.resource_id)
        try:
            actual_size = path.stat().st_size
            actual_digest = sha256_file(path)
        except OSError as error:
            raise ResourceExecutionError(
                f"model {descriptor.resource_id!r} snapshot file is unreadable: {declared.path!r}"
            ) from error
        if actual_size != declared.size or actual_digest != declared.sha256:
            raise ResourceExecutionError(
                f"model {descriptor.resource_id!r} snapshot file does not match its immutable descriptor: "
                f"{declared.path!r}"
            )


def _contained_file(root: Path, logical_path: str, resource_id: str) -> Path:
    pure = PurePosixPath(logical_path)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ResourceExecutionError(
            f"model {resource_id!r} contains an invalid logical file path: {logical_path!r}"
        )
    path = (root / Path(*pure.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ResourceExecutionError(
            f"model {resource_id!r} logical file escapes its snapshot: {logical_path!r}"
        ) from error
    if not path.is_file():
        raise ResourceExecutionError(
            f"model {resource_id!r} snapshot file is missing: {logical_path!r}"
        )
    return path
