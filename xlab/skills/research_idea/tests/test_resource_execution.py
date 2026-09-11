from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import patch

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PACKAGE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from research_idea_lib.algorithm.component_novelty import (  # noqa: E402
    ComponentQuery,
    ComponentRetrievalRequest,
)
from research_idea_lib.algorithm.runtime_adapters import SurveyRepositoryRetrieval  # noqa: E402
from research_idea_lib.algorithm.workflow import RetrievalRequest  # noqa: E402
from research_idea_lib.resources.component_novelty import build_component_novelty_runtime  # noqa: E402
from research_idea_lib.resources.embedding import SentenceTransformerEmbedding  # noqa: E402
from research_idea_lib.resources.evidence import (  # noqa: E402
    PackageNativeEvidenceResources,
    RetrievalBackends,
)
from research_idea_lib.resources.retrieval import (  # noqa: E402
    CitationRegistry,
    SurveyParagraph,
    outcome_rag,
    parse_survey_markdown,
    search_component_index,
)
from research_idea_lib.resources.manifest import (  # noqa: E402
    FileDigest,
    LoadedResourceBundle,
    ResourceDescriptor,
    ResourceManifest,
    descriptor_digest,
)
from research_idea_lib.resources.resolver import ResourceResolution  # noqa: E402
from research_idea_lib.survey_repository import SurveyArtifactRepository, SurveySource  # noqa: E402


class FakeEmbedding:
    def __init__(self, descriptor: ResourceDescriptor, dimension: int) -> None:
        self.descriptor = descriptor
        self.dimension = dimension
        self.calls: list[tuple[str, ...]] = []

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(tuple(texts))
        vectors = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype="float32")
            vector[0] = 1.0
            if self.dimension > 1 and "unrelated" in text.casefold():
                vector[:] = 0
                vector[1] = 1.0
            vectors.append(vector)
        return np.ascontiguousarray(vectors, dtype="float32")

    def usage(self) -> dict[str, object]:
        return {
            "resource_id": self.descriptor.resource_id,
            "descriptor_digest": self.descriptor.digest,
            "adapter_algorithm": "explicit.fake_embedding.v1",
            "embedding_dimension": self.dimension,
            "transformer_inference": True,
            "local_files_only": True,
        }


class FakeIndex:
    d = 2
    ntotal = 2

    def search(self, vectors: Any, limit: int):
        assert vectors.shape == (1, 2)
        return (
            np.asarray([[0.9, 0.4]], dtype="float32")[:, :limit],
            np.asarray([[1, 0]], dtype="int64")[:, :limit],
        )


class FakeFaiss:
    def __init__(self) -> None:
        self.paths: list[str] = []

    def read_index(self, path: str) -> FakeIndex:
        self.paths.append(path)
        return FakeIndex()


class RecordingRankedIndex:
    d = 2
    ntotal = 5

    def __init__(self) -> None:
        self.limits: list[int] = []

    def search(self, vectors: Any, limit: int):
        assert vectors.shape == (1, 2)
        self.limits.append(limit)
        return (
            np.asarray([[0.99, 0.98, 0.97, 0.70, 0.60]], dtype="float32")[:, :limit],
            np.asarray([[0, 1, 2, 3, 4]], dtype="int64")[:, :limit],
        )


class RecordingRankedFaiss:
    def __init__(self) -> None:
        self.index = RecordingRankedIndex()

    def read_index(self, path: str) -> RecordingRankedIndex:
        return self.index


class PackageNativeResourceExecutionTest(unittest.TestCase):
    def test_executes_model_and_faiss_backends_with_truthful_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository, outcome, component, faiss = build_repository(Path(temporary))
            output = retrieve(repository)

            self.assertEqual(len(outcome.calls), 1)
            self.assertIn("auditable evidence mechanisms", outcome.calls[0][0])
            self.assertEqual(component.calls, [("auditable evidence mechanisms",)])
            self.assertEqual(faiss.paths, [str(repository.source.component_index_dir.resolve() / "faiss.index")])
            receipts = output.metadata["resource_execution"]
            self.assertTrue(receipts["outcome_model"]["transformer_inference"])
            self.assertTrue(receipts["component_model"]["transformer_inference"])
            self.assertTrue(receipts["component_index"]["faiss_file_consumed"]["native_faiss_inference"])
            self.assertEqual(receipts["outcome_model"]["embedding_dimension"], 3)
            self.assertEqual(receipts["component_model"]["embedding_dimension"], 2)
            self.assertNotIn(str(Path(temporary)), json.dumps(output.metadata, sort_keys=True))

    def test_outcomerag_uses_global_adjacent_context_and_resolves_citations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository, _, _, _ = build_repository(Path(temporary))
            output = retrieve(repository)
            survey_evidence = [item for item in output.evidence if item.kind == "survey_markdown_paragraph"]
            self.assertTrue(survey_evidence)
            first = survey_evidence[0]
            self.assertIn("Auditable evidence remains linked [p1].", first.text)
            self.assertIn("A second paragraph in the same section", first.text)
            cross_section = next(
                item
                for item in survey_evidence
                if "A second paragraph in the same section" in item.text
                and "Unrelated material belongs elsewhere" in item.text
            )
            self.assertEqual(first.paper_ids, ("p1",))
            self.assertEqual(cross_section.paper_ids, ("p1", "p2"))
            self.assertEqual(first.provenance["resource"]["adjacent_context_window"], 1)

    def test_outcomerag_parsing_and_embedding_match_upstream_normal_path(self) -> None:
        registry = CitationRegistry(
            references={"p1": {}},
            keynotes={},
            aliases={},
        )
        without_references = parse_survey_markdown(
            "# Retained sufficiently long survey title\n\n## Evidence\n\nA sufficiently long paragraph cites evidence [12] [p1].",
            registry,
        )
        self.assertEqual(
            without_references[0].text,
            "# Retained sufficiently long survey title",
        )

        with_references = parse_survey_markdown(
            "# Removed sufficiently long survey title\n\n# H1 remains sufficiently long ordinary text\n\n## Evidence\n\n"
            "A sufficiently long paragraph cites evidence [12] [p1].\n\n"
            "tiny\n\nReferences:\n[p1] Paper",
            registry,
        )
        self.assertEqual(len(with_references), 2)
        self.assertEqual(
            with_references[0].text,
            "# H1 remains sufficiently long ordinary text",
        )
        self.assertEqual(with_references[1].section_path, ("Evidence",))
        self.assertIn("[12] [p1]", with_references[1].text)

        model_descriptor = ResourceDescriptor(
            resource_id="model:test",
            kind="model",
            schema_version="xlab.model_snapshot.v1",
            algorithm_version="sentence-transformers.v1",
            path="model",
            digest="sha256:test",
            files=(),
            dimensions=(("embedding", 2),),
        )
        model = FakeEmbedding(model_descriptor, 2)
        outcome_rag("query", with_references, model)
        self.assertIn(
            "A sufficiently long paragraph cites evidence  [p1].",
            model.calls[0],
        )

    def test_outcomerag_deduplicates_expanded_context_by_first_200_characters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository, outcome, component, faiss = build_repository(Path(temporary))
            resource = repository.source.resources
            assert resource is not None
            evidence_resource = PackageNativeEvidenceResources(
                bundle=resource.bundle,
                graph_db_path=resource.graph_db_path,
                keynote_cache_path=resource.keynote_cache_path,
                component_index_dir=resource.component_index_dir,
                outcome_model_path=resource.outcome_model_path,
                component_model_path=resource.component_model_path,
                outcome_backend=outcome,
                component_backend=component,
                faiss_backend=faiss,
            )
            shared_prefix = "shared context " + "x" * 210
            paragraphs = (
                SurveyParagraph("p1", "first paragraph text long enough", (), shared_prefix + " one", ("p1",)),
                SurveyParagraph("p2", "second paragraph text long enough", (), shared_prefix + " two", ("p2",)),
                SurveyParagraph(
                    "p3",
                    "third paragraph text long enough",
                    (),
                    "distinct context " + "y" * 210,
                    ("p3",),
                ),
            )
            registry = CitationRegistry(references={"paper": {}}, keynotes={}, aliases={})
            with patch(
                "research_idea_lib.resources.evidence.parse_survey_markdown",
                return_value=paragraphs,
            ):
                selected, ranked = evidence_resource._rank_survey("query", "ignored", registry)

            self.assertIsInstance(selected, tuple)
            self.assertEqual(len(selected), 2)
            self.assertEqual(len(ranked), 2)
            self.assertEqual(
                [hit.paragraph.paragraph_id for hit in selected], ["p1", "p3"]
            )
            self.assertEqual([item["id"] for item in ranked], ["p1", "p3"])
            self.assertNotIn("p2", {paper_id for item in ranked for paper_id in item["paper_ids"]})

    def test_freezes_top_five_outcome_hits_before_expansion_and_final_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository, _, _, _ = build_repository(Path(temporary))
            resource = repository.source.resources
            assert resource is not None
            calls: dict[str, list[str]] = {}

            def graph_neighbors(path, descriptor, survey_items, references):
                calls["graph"] = [str(item["id"]) for item in survey_items]
                return [], {"seed_paper_ids": []}

            def keynote_evidence(root, descriptor, survey_items, graph_items, references):
                calls["keynotes"] = [str(item["id"]) for item in survey_items]
                return [], {"paper_ids_consumed": []}

            with (
                patch(
                    "research_idea_lib.resources.evidence._graph_neighbors",
                    side_effect=graph_neighbors,
                ),
                patch(
                    "research_idea_lib.resources.evidence._keynote_evidence",
                    side_effect=keynote_evidence,
                ),
            ):
                result = repository.retrieve_resource_evidence(
                    "auditable evidence mechanisms", limit=1
                )

            expected = [
                "survey-paragraph-1",
                "survey-paragraph-2",
                "survey-paragraph-3",
            ]
            self.assertEqual(calls, {"graph": expected, "keynotes": expected})
            self.assertEqual(len(result.evidence_items), 1)
            self.assertEqual(
                [hit.paragraph.paragraph_id for hit in result.selected_outcome_hits],
                expected,
            )

    def test_outcome_selection_is_capped_at_five_before_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository, outcome, component, faiss = build_repository(Path(temporary))
            resource = repository.source.resources
            assert resource is not None
            evidence_resource = PackageNativeEvidenceResources(
                bundle=resource.bundle,
                graph_db_path=resource.graph_db_path,
                keynote_cache_path=resource.keynote_cache_path,
                component_index_dir=resource.component_index_dir,
                outcome_model_path=resource.outcome_model_path,
                component_model_path=resource.component_model_path,
                outcome_backend=outcome,
                component_backend=component,
                faiss_backend=faiss,
            )
            paragraphs = tuple(
                SurveyParagraph(
                    paragraph_id=f"paragraph-{index}",
                    text=f"paragraph {index}",
                    section_path=("Survey",),
                    section_context=f"paragraph {index}",
                    paper_ids=("p1",),
                )
                for index in range(7)
            )
            registry = CitationRegistry(
                references={"p1": {}}, keynotes={}, aliases={}
            )

            with patch(
                "research_idea_lib.resources.evidence.parse_survey_markdown",
                return_value=paragraphs,
            ):
                selected, ranked = evidence_resource._rank_survey(
                    "auditable evidence mechanisms", "ignored", registry
                )

            self.assertEqual(len(selected), 5)
            self.assertEqual(len(ranked), 5)

    def test_evidence_and_novelty_share_serialized_faiss_metadata_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository, _, component, faiss = build_repository(Path(temporary))
            metadata_path = repository.source.component_index_dir / "meta.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "meta": {
                            "z-component": {
                                "title": "First vector",
                                "description": "First serialized metadata row.",
                                "paper_ids": ["p1"],
                            },
                            "a-component": {
                                "title": "Second vector",
                                "description": "Second serialized metadata row.",
                                "paper_ids": ["p2"],
                            },
                        }
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            descriptor = existing_descriptor(
                repository.source.resources.bundle.root,
                "components",
                "components:v1",
                "component_index",
                capabilities=("component_similarity",),
                dimensions=(("embedding", 2),),
            )
            old_resolution = repository.source.resources
            manifest = old_resolution.bundle.manifest
            resolution = ResourceResolution(
                bundle=LoadedResourceBundle(
                    old_resolution.bundle.root,
                    ResourceManifest(
                        schema_version=manifest.schema_version,
                        bundle_id=manifest.bundle_id,
                        survey=manifest.survey,
                        graph=manifest.graph,
                        component_index=descriptor,
                        keynotes=manifest.keynotes,
                        models=manifest.models,
                    ),
                ),
                graph_db_path=old_resolution.graph_db_path,
                component_index_dir=old_resolution.component_index_dir,
                outcome_model_path=old_resolution.outcome_model_path,
                component_model_path=old_resolution.component_model_path,
                keynote_cache_path=old_resolution.keynote_cache_path,
                model_uris=old_resolution.model_uris,
                direct_parent_artifact_ids=old_resolution.direct_parent_artifact_ids,
            )
            repository.source.resources = resolution

            evidence_output = retrieve(repository)
            evidence_hits = [
                item
                for item in evidence_output.evidence
                if item.kind == "component_novelty_context"
            ]
            runtime = build_component_novelty_runtime(
                repository.source.resources,
                embedding_backend=component,
                faiss_backend=faiss,
            )
            request = ComponentRetrievalRequest(
                candidate_id="candidate-order",
                candidate_textual_identity="identity-order",
                idea_taste_mode="evidence_first",
                query=ComponentQuery(
                    query_id="query-order",
                    component_name="Order",
                    explanation="Validate positional mapping",
                    query_text="Order: Validate positional mapping",
                ),
                top_k=2,
                embedding_model=runtime.embedding_model,
                component_index=runtime.component_index,
            )
            novelty_output = runtime.retriever.retrieve(request)

            self.assertEqual(
                [item.provenance["resource"]["component_id"] for item in evidence_hits],
                ["a-component", "z-component"],
            )
            self.assertEqual(
                [item.record_id for item in novelty_output.hits],
                ["a-component", "z-component"],
            )

    def test_component_novelty_factory_exposes_native_attested_retriever(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository, _, component, faiss = build_repository(Path(temporary))
            runtime = build_component_novelty_runtime(
                repository.source.resources,
                embedding_backend=component,
                faiss_backend=faiss,
            )
            output = runtime.retriever.retrieve(
                ComponentRetrievalRequest(
                    candidate_id="candidate-1",
                    candidate_textual_identity="identity-1",
                    idea_taste_mode="evidence_first",
                    query=ComponentQuery(
                        query_id="query-1",
                        component_name="Evidence memory",
                        explanation="Retains source identities",
                        query_text="Evidence memory: Retains source identities",
                    ),
                    top_k=2,
                    embedding_model=runtime.embedding_model,
                    component_index=runtime.component_index,
                )
            )
            self.assertTrue(output.native_embedding_inference)
            self.assertTrue(output.native_faiss_search)
            self.assertEqual(output.candidate_id, "candidate-1")
            self.assertEqual(output.candidate_textual_identity, "identity-1")
            self.assertEqual(output.idea_taste_mode, "evidence_first")
            self.assertEqual(output.hits[0].record_id, "component-2")
            self.assertEqual(output.embedding_model, runtime.embedding_model)
            self.assertEqual(output.component_index, runtime.component_index)

    def test_operator_component_retrieval_overretrieves_distinct_core_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository, _, component, _ = build_repository(Path(temporary))
            metadata_path = repository.source.component_index_dir / "meta.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "meta": {
                            f"component-{index}": {
                                "node_id": "node-1" if index < 3 else f"node-{index - 1}",
                                "component": f"Component {index}",
                                "description": f"Description {index}",
                                "domain": "Biology",
                                "paper_ids": [f"paper-{index}"],
                            }
                            for index in range(5)
                        }
                    }
                ),
                encoding="utf-8",
            )
            descriptor = existing_descriptor(
                repository.source.resources.bundle.root,
                "components",
                "components:v1",
                "component_index",
                capabilities=("component_similarity",),
                dimensions=(("embedding", 2),),
            )
            old_resolution = repository.source.resources
            manifest = old_resolution.bundle.manifest
            repository.source.resources = ResourceResolution(
                bundle=LoadedResourceBundle(
                    old_resolution.bundle.root,
                    ResourceManifest(
                        schema_version=manifest.schema_version,
                        bundle_id=manifest.bundle_id,
                        survey=manifest.survey,
                        graph=manifest.graph,
                        component_index=descriptor,
                        keynotes=manifest.keynotes,
                        models=manifest.models,
                    ),
                ),
                graph_db_path=old_resolution.graph_db_path,
                component_index_dir=old_resolution.component_index_dir,
                outcome_model_path=old_resolution.outcome_model_path,
                component_model_path=old_resolution.component_model_path,
                keynote_cache_path=old_resolution.keynote_cache_path,
                model_uris=old_resolution.model_uris,
                direct_parent_artifact_ids=old_resolution.direct_parent_artifact_ids,
            )
            faiss = RecordingRankedFaiss()
            runtime = build_component_novelty_runtime(
                repository.source.resources,
                embedding_backend=component,
                faiss_backend=faiss,
            )

            output = runtime.retriever.retrieve_operator_components(
                "cross-domain control mechanism", limit=2
            )

            self.assertEqual(faiss.index.limits, [5])
            self.assertEqual(
                {hit.core_node_id for hit in output.hits},
                {"node-1", "node-2", "node-3"},
            )
            provenance = json.loads(output.provenance_json)
            self.assertEqual(provenance["raw_limit"], 2)
            self.assertEqual(provenance["internal_search_limit"], 32)

    @unittest.skipUnless(
        os.environ.get("XLAB_RESEARCH_IDEA_REAL_ML_SMOKE") == "1",
        "set XLAB_RESEARCH_IDEA_REAL_ML_SMOKE=1 to run local transformer/FAISS smoke",
    )
    def test_real_local_minilm_and_faiss_smoke(self) -> None:
        model_root = Path(
            os.environ.get(
                "XLAB_RESEARCH_IDEA_MINILM_SNAPSHOT",
                str(
                    Path.home()
                    / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
                    / "snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
                ),
            )
        )
        if not model_root.is_dir():
            self.skipTest("immutable MiniLM snapshot is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local_model_root = root / "minilm-snapshot"
            shutil.copytree(model_root, local_model_root, symlinks=False)
            descriptor = existing_descriptor(
                root,
                local_model_root.name,
                "model:outcome:smoke",
                "model",
                capabilities=("sentence_embedding",),
                dimensions=(("embedding", 384),),
                provenance=(("model_id", "sentence-transformers/all-MiniLM-L6-v2"), ("role", "outcome_model")),
            )
            model = SentenceTransformerEmbedding.outcome(descriptor, local_model_root)
            vectors = model.encode(["auditable scientific agents", "survey evidence"])
            self.assertEqual(vectors.shape, (2, 384))

            import faiss

            index_root = root / "component-index"
            index_root.mkdir()
            index = faiss.IndexFlatIP(384)
            index.add(np.ascontiguousarray(vectors, dtype="float32"))
            faiss.write_index(index, str(index_root / "faiss.index"))
            write_json(
                index_root / "meta.json",
                {"meta": {"component-1": {"title": "Agent"}, "component-2": {"title": "Evidence"}}},
            )
            index_descriptor = existing_descriptor(
                root,
                "component-index",
                "components:smoke",
                "component_index",
                capabilities=("component_similarity",),
                dimensions=(("embedding", 384),),
            )
            results = search_component_index(
                query="auditable scientific agents",
                index_path=index_root / "faiss.index",
                descriptor=index_descriptor,
                records=(("component-1", {"title": "Agent"}), ("component-2", {"title": "Evidence"})),
                model=model,
                limit=2,
            )
            self.assertEqual(len(results), 2)
            self.assertTrue(model.usage()["transformer_inference"])

    def test_declared_resource_mutation_and_missing_bundle_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository, _, _, _ = build_repository(Path(temporary))
            (repository.source.component_index_dir / "meta.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "declared digest"):
                retrieve(repository)

        source = SurveySource(
            survey_path=Path("survey"),
            survey_json_path=None,
            survey_md_path=None,
            citations_path=None,
            report_path=None,
            manifest_path=None,
        )
        repository = SurveyArtifactRepository(source=source, evidence_items=[], references=[])
        with self.assertRaisesRegex(ValueError, "validated executable resource bundle"):
            retrieve(repository)


def retrieve(repository: SurveyArtifactRepository):
    return SurveyRepositoryRetrieval(repository, limit=20).retrieve(
        RetrievalRequest(
            topic="auditable agents",
            query="auditable evidence mechanisms",
            background={},
            source_context={},
        )
    )


def build_repository(
    root: Path,
) -> tuple[SurveyArtifactRepository, FakeEmbedding, FakeEmbedding, FakeFaiss]:
    survey_root = root / "survey"
    survey_root.mkdir()
    survey_markdown = survey_root / "survey.md"
    survey_markdown.write_text(
        "# Survey\n\n## Evidence\n\nAuditable evidence remains linked [p1].\n\n"
        "A second paragraph in the same section adds mechanisms.\n\n"
        "## Other\n\nUnrelated material belongs elsewhere [p2].\n",
        encoding="utf-8",
    )
    citations = {
        "references": [
            {"paper_id": "p1", "title": "Seed Paper"},
            {"paper_id": "p2", "title": "Neighbor Paper"},
        ],
        "traces": [{"claim_id": "c1", "paper_ids": ["p1"]}],
    }

    resource_root = root / "resources"
    graph_root = resource_root / "graph"
    graph_root.mkdir(parents=True)
    graph_path = graph_root / "graph.db"
    with sqlite3.connect(graph_path) as connection:
        connection.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY, paper_id TEXT, paper_title TEXT, summary TEXT)")
        connection.execute("CREATE TABLE edges (source TEXT, target TEXT, relation TEXT, summary TEXT)")
        connection.executemany(
            "INSERT INTO nodes VALUES (?, ?, ?, ?)",
            [
                ("node:p1", "p1", "Seed Paper", "Seed survey evidence."),
                ("node:p2", "p2", "Neighbor Paper", "Neighbor mechanism evidence."),
            ],
        )
        connection.execute(
            "INSERT INTO edges VALUES (?, ?, ?, ?)",
            ("node:p1", "node:p2", "extends", "Graph-linked evidence."),
        )

    keynote_root = resource_root / "keynotes"
    keynote_root.mkdir(parents=True)
    write_json(keynote_root / "keynotes.json", {"p1": "Seed keynote.", "p2": "Neighbor keynote."})

    component_root = resource_root / "components"
    component_root.mkdir(parents=True)
    (component_root / "faiss.index").write_bytes(b"fake-faiss-index")
    write_json(
        component_root / "meta.json",
        {
            "meta": {
                "component-1": {"title": "Auditable memory", "description": "Retain evidence IDs.", "paper_ids": ["p1"]},
                "component-2": {"title": "Graph bridge", "description": "Connect neighbors.", "paper_ids": ["p2"]},
            }
        },
    )

    outcome_root = resource_root / "models" / "outcome"
    component_model_root = resource_root / "models" / "component"
    write_model(outcome_root)
    write_model(component_model_root)
    survey_descriptor = descriptor(resource_root, "survey", "survey:v1", "survey", {"survey.json": b"{}"}, capabilities=("citation_traces",))
    graph_descriptor = existing_descriptor(resource_root, "graph", "graph:v1", "graph", capabilities=("paper_neighbors",))
    component_descriptor = existing_descriptor(
        resource_root, "components", "components:v1", "component_index",
        capabilities=("component_similarity",), dimensions=(("embedding", 2),),
    )
    keynote_descriptor = existing_descriptor(resource_root, "keynotes", "keynotes:v1", "keynotes", capabilities=("paper_keynotes",))
    outcome_descriptor = existing_descriptor(
        resource_root, "models/outcome", "model:outcome:v1", "model",
        capabilities=("sentence_embedding",), dimensions=(("embedding", 3),),
        provenance=(("model_id", "sentence-transformers/all-MiniLM-L6-v2"), ("role", "outcome_model")),
    )
    component_model_descriptor = existing_descriptor(
        resource_root, "models/component", "model:component:v1", "model",
        capabilities=("sentence_embedding",), dimensions=(("embedding", 2),),
        provenance=(("model_id", "package/component-model"), ("role", "component_model")),
    )
    manifest = ResourceManifest(
        schema_version="xlab.research_idea.resources.v1",
        bundle_id="execution-fixture:v1",
        survey=survey_descriptor,
        graph=graph_descriptor,
        component_index=component_descriptor,
        keynotes=keynote_descriptor,
        models=(outcome_descriptor, component_model_descriptor),
    )
    resolution = ResourceResolution(
        bundle=LoadedResourceBundle(resource_root, manifest),
        graph_db_path=graph_path,
        component_index_dir=component_root,
        outcome_model_path=outcome_root,
        component_model_path=component_model_root,
        keynote_cache_path=keynote_root,
        model_uris={"outcome_model": "xlab-cache://models/outcome", "component_model": "xlab-cache://models/component"},
        direct_parent_artifact_ids=("sha256:survey-parent",),
    )
    source = SurveySource(
        survey_path=survey_root,
        survey_json_path=survey_root / "survey.json",
        survey_md_path=survey_markdown,
        citations_path=survey_root / "citations.json",
        report_path=survey_root / "survey_report.json",
        manifest_path=survey_root / "manifest.json",
        survey={"topic": "Auditable agents"},
        citations=citations,
        graph_db_path=graph_path,
        component_index_dir=component_root,
        outcome_model_path=outcome_root,
        component_model_path=component_model_root,
        keynote_cache_path=keynote_root,
        resources=resolution,
    )
    outcome = FakeEmbedding(outcome_descriptor, 3)
    component = FakeEmbedding(component_model_descriptor, 2)
    faiss = FakeFaiss()
    repository = SurveyArtifactRepository(
        source=source,
        evidence_items=[],
        references=list(citations["references"]),
        retrieval_backends=RetrievalBackends(outcome, component, faiss),
    )
    return repository, outcome, component, faiss


def write_model(root: Path) -> None:
    root.mkdir(parents=True)
    write_json(root / "config.json", {"local": True})
    (root / "weights.bin").write_bytes(b"fixture")


def descriptor(
    root: Path,
    relative: str,
    resource_id: str,
    kind: str,
    files: dict[str, bytes],
    **kwargs: Any,
) -> ResourceDescriptor:
    target = root / relative
    target.mkdir(parents=True, exist_ok=True)
    for name, value in files.items():
        (target / name).write_bytes(value)
    return existing_descriptor(root, relative, resource_id, kind, **kwargs)


def existing_descriptor(
    root: Path,
    relative: str,
    resource_id: str,
    kind: str,
    *,
    capabilities: tuple[str, ...] = (),
    dimensions: tuple[tuple[str, int], ...] = (),
    provenance: tuple[tuple[str, str], ...] = (),
) -> ResourceDescriptor:
    target = root / relative
    files = tuple(
        FileDigest(path=path.relative_to(target).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size=path.stat().st_size)
        for path in sorted(target.rglob("*")) if path.is_file()
    )
    algorithms = {"survey": "survey-agent.v1", "graph": "sqlite-paper-graph.v1", "component_index": "faiss-flat-ip.v1", "keynotes": "keynote-cache.v1", "model": "sentence-transformers.v1"}
    schemas = {"survey": "xlab.literature_survey.v1", "graph": "xlab.paper_graph.v1", "component_index": "xlab.component_index.v1", "keynotes": "xlab.paper_keynotes.v1", "model": "xlab.model_snapshot.v1"}
    return ResourceDescriptor(
        resource_id=resource_id, kind=kind, schema_version=schemas[kind], algorithm_version=algorithms[kind],
        path=relative, digest=descriptor_digest(files), files=files, capabilities=capabilities,
        dimensions=dimensions, provenance=provenance,
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
