from __future__ import annotations

import hashlib
import json
import multiprocessing
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PACKAGE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from research_idea_lib.resources import (  # noqa: E402
    RESOURCE_MANIFEST_SCHEMA_VERSION,
    FileDigest,
    ModelCache,
    ModelCacheError,
    ModelFile,
    ModelSnapshot,
    ResourceRequirement,
    ResourceValidationError,
    descriptor_digest,
    load_resource_bundle,
    resolve_survey_resources,
    stable_evidence_id,
    stable_paper_id,
)


class ResourceManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="xlab-research-resources-test-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_loads_valid_portable_bundle(self) -> None:
        manifest_path = write_bundle(self.tmp / "bundle")
        requirement = ResourceRequirement(
            resource_id="components:v1",
            schema_versions=("xlab.component_index.v1",),
            algorithm_versions=("faiss-flat-ip.v1",),
            capabilities=("component_similarity",),
            dimensions=(("embedding", 3),),
        )

        bundle = load_resource_bundle(manifest_path, requirements=(requirement,))

        self.assertEqual(bundle.manifest.schema_version, RESOURCE_MANIFEST_SCHEMA_VERSION)
        self.assertEqual(bundle.manifest.survey.parent_artifact_ids, ("sha256:survey-parent",))
        self.assertEqual(bundle.manifest.component_index.dimension_map["embedding"], 3)
        self.assertEqual(bundle.path_for("graph:v1"), manifest_path.parent / "graph")

    def test_rejects_missing_capability(self) -> None:
        manifest_path = write_bundle(self.tmp / "bundle")
        requirement = ResourceRequirement(
            resource_id="components:v1",
            schema_versions=("xlab.component_index.v1",),
            algorithm_versions=("faiss-flat-ip.v1",),
            capabilities=("component_similarity", "filtered_search"),
        )

        with self.assertRaisesRegex(ResourceValidationError, "missing capabilities: filtered_search"):
            load_resource_bundle(manifest_path, requirements=(requirement,))

    def test_rejects_digest_and_dimension_mismatches(self) -> None:
        manifest_path = write_bundle(self.tmp / "bundle")
        component_file = manifest_path.parent / "component-index" / "faiss.index"
        component_file.write_bytes(b"tampered")
        with self.assertRaisesRegex(ResourceValidationError, "digest mismatch"):
            load_resource_bundle(manifest_path)

        manifest_path = write_bundle(self.tmp / "dimension-bundle")
        requirement = ResourceRequirement(
            resource_id="components:v1",
            schema_versions=("xlab.component_index.v1",),
            algorithm_versions=("faiss-flat-ip.v1",),
            dimensions=(("embedding", 4),),
        )
        with self.assertRaisesRegex(ResourceValidationError, "expected 4"):
            load_resource_bundle(manifest_path, requirements=(requirement,))

    def test_rejects_structural_containment_escape(self) -> None:
        manifest_path = write_bundle(self.tmp / "bundle")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["resources"]["graph"]["path"] = "../outside"
        manifest_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

        with self.assertRaisesRegex(ResourceValidationError, "normalized relative POSIX path"):
            load_resource_bundle(manifest_path)

    def test_resolves_survey_linked_bundle_and_immutable_models(self) -> None:
        survey_run = self.tmp / "survey-run"
        artifacts = survey_run / "artifacts"
        artifacts.mkdir(parents=True)
        survey_json = artifacts / "survey.json"
        survey_json.write_text('{"schema_version":"xlab.literature_survey.v1"}', encoding="utf-8")
        survey_id = xlab_file_artifact_id(survey_json)
        manifest_path = write_bundle(survey_run / "resources", survey_parent_id=survey_id, two_models=True)
        survey_manifest = {
            "artifacts": [
                {"type": "literature_survey_json", "path": "artifacts/survey.json"},
                {"type": "research_idea_resource_manifest", "path": "resources/resource-manifest.json"},
            ]
        }
        source_manifest_path = survey_run / "manifest.json"
        source_manifest_path.write_text(json.dumps(survey_manifest), encoding="utf-8")

        resolution = resolve_survey_resources(
            survey_manifest_path=source_manifest_path,
            survey_manifest=survey_manifest,
            survey={},
            survey_json_path=survey_json,
            cwd=survey_run,
            model_cache_root=self.tmp / "cache",
        )

        self.assertTrue(resolution.passed, resolution.blocking_errors)
        self.assertEqual(resolution.manifest_path, manifest_path)
        self.assertEqual(resolution.manifest_logical_path, "resources/resource-manifest.json")
        self.assertEqual(resolution.direct_parent_artifact_ids, (survey_id,))
        self.assertEqual(set(resolution.model_uris), {"outcome_model", "component_model"})
        self.assertTrue(all(uri.startswith("xlab-cache://models/") for uri in resolution.model_uris.values()))
        portable_resources = resolution.portable_resources()
        self.assertEqual(portable_resources["graph"]["files"][0]["logical_path"], "graph.db")
        self.assertEqual(len(portable_resources["graph"]["files"][0]["sha256"]), 64)
        persisted = json.dumps({
            "manifest": resolution.portable_manifest(),
            "resources": resolution.portable_resources(),
        })
        self.assertNotIn(str(self.tmp), persisted)

    def test_blocks_missing_ambiguous_and_unrelated_resource_manifests(self) -> None:
        source_manifest_path = self.tmp / "survey" / "manifest.json"
        source_manifest_path.parent.mkdir(parents=True)
        source_manifest_path.write_text("{}", encoding="utf-8")
        missing = resolve_survey_resources(
            survey_manifest_path=source_manifest_path,
            survey_manifest={},
            survey={},
            survey_json_path=None,
            cwd=self.tmp,
        )
        self.assertIn("No explicit XLab resource manifest", missing.blocking_errors[0])

        ambiguous = resolve_survey_resources(
            survey_manifest_path=source_manifest_path,
            survey_manifest={"resource_manifest": "one.json", "resources": {"manifest": "two.json"}},
            survey={},
            survey_json_path=None,
            cwd=self.tmp,
        )
        self.assertIn("RESOURCE_MANIFEST_DECLARATION_AMBIGUOUS", ambiguous.blocking_errors[0])

        unrelated = resolve_survey_resources(
            survey_manifest_path=source_manifest_path,
            survey_manifest={"resource_manifest": "../neighbor/resource-manifest.json"},
            survey={},
            survey_json_path=None,
            cwd=self.tmp,
        )
        self.assertIn("RESOURCE_MANIFEST_DECLARATION_INVALID", unrelated.blocking_errors[0])

    def test_malformed_resource_manifest_error_uses_public_code_and_logical_path(self) -> None:
        survey_run = self.tmp / "host-path-sentinel" / "survey-run"
        survey_run.mkdir(parents=True)
        manifest_path = survey_run / "manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        malformed = survey_run / "resources" / "resource-manifest.json"
        malformed.parent.mkdir(parents=True)
        malformed.write_text("{not-json", encoding="utf-8")

        resolution = resolve_survey_resources(
            survey_manifest_path=manifest_path,
            survey_manifest={"resource_manifest": "resources/resource-manifest.json"},
            survey={},
            survey_json_path=None,
            cwd=self.tmp,
        )

        serialized = " ".join(resolution.blocking_errors)
        self.assertIn("RESOURCE_MANIFEST_INVALID", serialized)
        self.assertIn("declaration=resources/resource-manifest.json", serialized)
        self.assertNotIn(str(survey_run.resolve()), serialized)

    def test_absolute_resource_declaration_error_omits_host_path(self) -> None:
        survey_run = self.tmp / "survey-run"
        survey_run.mkdir(parents=True)
        manifest_path = survey_run / "manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        unique_host_path = str((self.tmp / "host-path-sentinel" / "resource-manifest.json").resolve())

        resolution = resolve_survey_resources(
            survey_manifest_path=manifest_path,
            survey_manifest={"resource_manifest": unique_host_path},
            survey={},
            survey_json_path=None,
            cwd=self.tmp,
        )

        serialized = " ".join(resolution.blocking_errors)
        self.assertIn("RESOURCE_MANIFEST_DECLARATION_INVALID", serialized)
        self.assertNotIn(unique_host_path, serialized)

    def test_blocks_resource_bundle_with_wrong_survey_lineage(self) -> None:
        survey_run = self.tmp / "survey-run"
        artifacts = survey_run / "artifacts"
        artifacts.mkdir(parents=True)
        survey_json = artifacts / "survey.json"
        survey_json.write_text("{}", encoding="utf-8")
        write_bundle(survey_run / "resources", survey_parent_id="sha256:other", two_models=True)
        survey_manifest = {
            "resource_manifest": "resources/resource-manifest.json",
            "artifacts": [{"type": "literature_survey_json", "path": "artifacts/survey.json"}],
        }
        source_manifest_path = survey_run / "manifest.json"
        source_manifest_path.write_text(json.dumps(survey_manifest), encoding="utf-8")

        resolution = resolve_survey_resources(
            survey_manifest_path=source_manifest_path,
            survey_manifest=survey_manifest,
            survey={},
            survey_json_path=survey_json,
            cwd=survey_run,
            model_cache_root=self.tmp / "cache",
        )

        self.assertFalse(resolution.passed)
        self.assertIn("RESOURCE_MANIFEST_LINEAGE_MISMATCH", resolution.blocking_errors[0])

    def test_stable_evidence_and_paper_ids(self) -> None:
        self.assertEqual(
            stable_paper_id({"doi": "https://doi.org/10.1000/Example"}),
            "doi:10.1000/example",
        )
        first = stable_evidence_id(
            "Claim",
            "  Evidence   improves reliability. ",
            paper_ids=["S2:B", "s2:a"],
        )
        second = stable_evidence_id(
            "claim",
            "evidence improves reliability.",
            paper_ids=["S2:A", "s2:b"],
        )
        self.assertEqual(first, second)


class ModelCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="xlab-model-cache-test-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_populates_atomically_and_reuses_verified_snapshot(self) -> None:
        source = self.tmp / "source"
        model_file = source / "weights.bin"
        model_file.parent.mkdir(parents=True)
        model_file.write_bytes(b"immutable-model-weights")
        snapshot = snapshot_for(model_file)
        cache = ModelCache(self.tmp / "cache")

        populated = cache.resolve(snapshot, source=source)
        reused = cache.resolve(snapshot)

        self.assertFalse(populated.reused)
        self.assertTrue(reused.reused)
        self.assertEqual(populated.path, reused.path)
        self.assertEqual(reused.uri, f"xlab-cache://models/{snapshot.cache_key}")
        self.assertEqual((reused.path / "weights.bin").read_bytes(), b"immutable-model-weights")

        (reused.path / "weights.bin").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ModelCacheError, "size .* expected|digest mismatch"):
            cache.resolve(snapshot)

    def test_rejects_bad_digest_without_populating_cache(self) -> None:
        source = self.tmp / "source"
        model_file = source / "weights.bin"
        model_file.parent.mkdir(parents=True)
        model_file.write_bytes(b"actual")
        snapshot = ModelSnapshot(
            model_id="sentence-transformers/example",
            revision="0123456789abcdef",
            files=(ModelFile(path="weights.bin", sha256="0" * 64, size=len(b"actual")),),
        )
        cache = ModelCache(self.tmp / "cache")

        with self.assertRaisesRegex(ModelCacheError, "digest mismatch"):
            cache.resolve(snapshot, source=source)

        self.assertFalse((self.tmp / "cache" / "models" / snapshot.cache_key).exists())

    def test_concurrent_processes_populate_once(self) -> None:
        source = self.tmp / "source"
        model_file = source / "weights.bin"
        model_file.parent.mkdir(parents=True)
        model_file.write_bytes(b"immutable-model-weights")
        snapshot = snapshot_for(model_file)
        cache_root = self.tmp / "cache"
        context = multiprocessing.get_context("spawn")
        start = context.Event()
        results = context.Queue()
        processes = [
            context.Process(target=resolve_model_cache, args=(cache_root, source, snapshot, start, results))
            for _ in range(4)
        ]

        for process in processes:
            process.start()
        start.set()
        outcomes = [results.get(timeout=15) for _ in processes]
        for process in processes:
            process.join(timeout=15)
            self.assertEqual(process.exitcode, 0)

        self.assertEqual(sum(not reused for reused in outcomes), 1)
        cached = ModelCache(cache_root).resolve(snapshot)
        self.assertEqual((cached.path / "weights.bin").read_bytes(), b"immutable-model-weights")

    def test_stale_lock_file_does_not_block_population(self) -> None:
        source = self.tmp / "source"
        model_file = source / "weights.bin"
        model_file.parent.mkdir(parents=True)
        model_file.write_bytes(b"immutable-model-weights")
        snapshot = snapshot_for(model_file)
        cache_root = self.tmp / "cache"
        lock_path = cache_root / "locks" / f"model-{snapshot.cache_key}.lock"
        lock_path.parent.mkdir(parents=True)
        lock_path.write_text("stale-owner", encoding="utf-8")

        resolved = ModelCache(cache_root, lock_timeout=1).resolve(snapshot, source=source)

        self.assertFalse(resolved.reused)
        self.assertEqual((resolved.path / "weights.bin").read_bytes(), b"immutable-model-weights")


def resolve_model_cache(
    cache_root: Path,
    source: Path,
    snapshot: ModelSnapshot,
    start: Any,
    results: Any,
) -> None:
    start.wait()
    results.put(ModelCache(cache_root).resolve(snapshot, source=source).reused)


def write_bundle(
    root: Path,
    *,
    survey_parent_id: str = "sha256:survey-parent",
    two_models: bool = False,
) -> Path:
    resources = {
        "survey": write_resource(
            root,
            "survey",
            resource_id="survey:v1",
            kind="survey",
            schema_version="xlab.literature_survey.v1",
            algorithm_version="survey-agent.v1",
            files={"survey.json": b'{"topic":"resource contracts"}'},
            capabilities=["citation_traces"],
            parent_artifact_ids=[survey_parent_id],
        ),
        "graph": write_resource(
            root,
            "graph",
            resource_id="graph:v1",
            kind="graph",
            schema_version="xlab.paper_graph.v1",
            algorithm_version="sqlite-paper-graph.v1",
            files={"graph.db": b"sqlite-fixture"},
            capabilities=["paper_neighbors"],
        ),
        "component_index": write_resource(
            root,
            "component-index",
            resource_id="components:v1",
            kind="component_index",
            schema_version="xlab.component_index.v1",
            algorithm_version="faiss-flat-ip.v1",
            files={"faiss.index": b"index", "meta.json": b"[]"},
            capabilities=["component_similarity"],
            dimensions={"embedding": 3},
        ),
        "keynotes": write_resource(
            root,
            "keynotes",
            resource_id="keynotes:v1",
            kind="keynotes",
            schema_version="xlab.paper_keynotes.v1",
            algorithm_version="keynote-cache.v1",
            files={"keynotes.json": b"{}"},
            capabilities=["paper_keynotes"],
        ),
        "models": [
            write_resource(
                root,
                "models/outcome",
                resource_id="model:outcome:v1",
                kind="model",
                schema_version="xlab.model_snapshot.v1",
                algorithm_version="sentence-transformers.v1",
                files={"config.json": b"{}", "weights.bin": b"outcome-weights"},
                capabilities=["sentence_embedding"],
                dimensions={"embedding": 384},
                provenance={
                    "role": "outcome_model",
                    "model_id": "sentence-transformers/all-MiniLM-L6-v2",
                },
            ),
            *(
                [
                    write_resource(
                        root,
                        "models/component",
                        resource_id="model:component-novelty:v1",
                        kind="model",
                        schema_version="xlab.model_snapshot.v1",
                        algorithm_version="sentence-transformers.v1",
                        files={"config.json": b"{}", "weights.bin": b"component-weights"},
                        capabilities=["sentence_embedding"],
                        dimensions={"embedding": 3},
                    )
                ]
                if two_models
                else []
            ),
        ],
    }
    manifest = {
        "schema_version": RESOURCE_MANIFEST_SCHEMA_VERSION,
        "bundle_id": "research-idea-fixture:v1",
        "resources": resources,
    }
    path = root / "resource-manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_resource(
    root: Path,
    relative: str,
    *,
    resource_id: str,
    kind: str,
    schema_version: str,
    algorithm_version: str,
    files: dict[str, bytes],
    capabilities: list[str],
    dimensions: dict[str, int] | None = None,
    parent_artifact_ids: list[str] | None = None,
    provenance: dict[str, str] | None = None,
) -> dict[str, Any]:
    resource_root = root / relative
    file_descriptors: list[FileDigest] = []
    for file_relative, content in files.items():
        path = resource_root / file_relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        file_descriptors.append(
            FileDigest(
                path=file_relative,
                sha256=hashlib.sha256(content).hexdigest(),
                size=len(content),
            )
        )
    return {
        "resource_id": resource_id,
        "kind": kind,
        "schema_version": schema_version,
        "algorithm_version": algorithm_version,
        "path": relative,
        "digest": descriptor_digest(tuple(file_descriptors)),
        "files": [
            {"path": file.path, "sha256": file.sha256, "size": file.size}
            for file in file_descriptors
        ],
        "parent_artifact_ids": parent_artifact_ids or [],
        "capabilities": capabilities,
        "dimensions": dimensions or {},
        "provenance": provenance or {"producer": "test-fixture"},
    }


def xlab_file_artifact_id(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(b"file\0")
    digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()}"


def snapshot_for(path: Path) -> ModelSnapshot:
    content = path.read_bytes()
    return ModelSnapshot(
        model_id="sentence-transformers/example",
        revision="0123456789abcdef",
        files=(
            ModelFile(
                path=path.name,
                sha256=hashlib.sha256(content).hexdigest(),
                size=len(content),
            ),
        ),
        capabilities=("sentence_embedding",),
        dimensions=(("embedding", 3),),
    )


if __name__ == "__main__":
    unittest.main()
