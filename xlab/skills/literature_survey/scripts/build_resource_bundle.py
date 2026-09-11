#!/usr/bin/env python3
"""The resources phase worker: package completed Survey evidence for native idea retrieval."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from literature_survey_lib.common import atomic_write_json, read_json, utc_now
from literature_survey_lib.manifest import write_final_manifest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research_idea/scripts"))
from research_idea_lib.resources.component_novelty import build_component_novelty_runtime
from research_idea_lib.resources.evidence import PackageNativeEvidenceResources
from research_idea_lib.resources.manifest import FileDigest, descriptor_digest, sha256_file
from research_idea_lib.resources.resolver import resolve_survey_resources
from research_idea_lib.resources.retrieval import CitationRegistry, keynote_text, parse_survey_markdown

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def describe(root, relative, resource_id, kind, schema, algorithm, capability, parents, *, dimensions=None, provenance=None):
    directory = root / relative
    files = tuple(FileDigest(p.relative_to(directory).as_posix(), sha256_file(p), p.stat().st_size)
                  for p in sorted(directory.rglob("*")) if p.is_file())
    if not files:
        raise ValueError(f"Empty resource: {relative}")
    return {"resource_id": resource_id, "kind": kind, "schema_version": schema,
            "algorithm_version": algorithm, "path": relative, "digest": descriptor_digest(files),
            "files": [asdict(f) for f in files], "capabilities": [capability],
            "parent_artifact_ids": parents, "dimensions": dimensions or {}, "provenance": provenance or {}}


def collect_graph_resources(source_graph, destination):
    components = []
    papers = {}
    core_notes = {}
    with sqlite3.connect(f"file:{source_graph}?mode=ro", uri=True) as source, sqlite3.connect(destination) as target:
        source.backup(target)
        for node_id, pid, name, raw in source.execute("SELECT id,paper_id,name,raw_json FROM nodes WHERE node_type='Paper' ORDER BY id"):
            data = json.loads(raw)
            metadata = data.get("metadata") or {}
            papers[pid] = {"title": name, "abstract": metadata.get("abstract") or "", "metadata": metadata}
            target.execute("UPDATE nodes SET paper_title=?,summary=? WHERE id=?", (name, metadata.get("abstract") or "", node_id))
        for node_id, pid, raw in source.execute("SELECT id,paper_id,raw_json FROM nodes WHERE node_type='Core' ORDER BY id"):
            core = json.loads(raw)
            if not pid or core.get("fallback_from_title"):
                continue
            core_notes.setdefault(pid, []).append({k: core[k] for k in ["full_name", "summary", "insight", "quote", "structured_summary", "limitations", "future_work"] if core.get(k)})
            for offset, component in enumerate([core, *(core.get("components") or [])]):
                name = component.get("name") or component.get("full_name")
                summary = component.get("summary") or component.get("description") or ""
                if not name or not summary:
                    continue
                component_id = f"{node_id}:component:{offset}"
                components.append({"id": component_id, "node_id": node_id, "component": name,
                                   "full_name": core.get("full_name") or name, "label": core.get("label") or name,
                                   "summary": summary, "description": summary, "insight": component.get("insight") or core.get("insight") or "",
                                   "paper_ids": [pid], "paper_title": core.get("paper_title") or papers.get(pid, {}).get("title", ""),
                                   "domain": "; ".join(core.get("paper_domain") or []),
                                   "quote": component.get("quote") or ""})
        target.commit()
    if not components:
        raise ValueError("The graph has no evidence-backed components")
    return papers, core_notes, components


def package_keynotes(run, papers, core_notes, citations):
    saved = read_json(run / "state/xcientist/keynotes.json").get("keynotes", {})
    references = {r["paper_id"]: r for r in citations["references"]}
    alias_path = run / "state/xcientist/citation_aliases.json"
    verified = {r["paper_id"]: r for r in read_json(alias_path).get("verified_references", [])} if alias_path.exists() else {}
    result = {}
    levels = {}
    for pid in sorted(set(papers) | set(references)):
        paper = papers.get(pid) or references.get(pid) or {}
        title = paper.get("title") or pid
        if keynote_text(saved.get(pid)):
            content, level = keynote_text(saved[pid]), "extracted_keynotes"
        elif core_notes.get(pid):
            content, level = keynote_text(core_notes[pid]), "graph_extraction"
        elif verified.get(pid, {}).get("verified_reference_text"):
            content, level = verified[pid]["verified_reference_text"], "verified_primary_text"
        elif paper.get("abstract"):
            content, level = paper["abstract"], "abstract"
        else:
            content, level = "Bibliographic metadata only; no extracted scientific findings are available.", "metadata_only"
        content = content.replace("Paper Title: Unknown", f"Paper Title: {title}")
        result[pid] = {"paper_id": pid, "title": title, "evidence_level": level,
                       "keynote": f"Paper Title: {title}\nEvidence level: {level}\n{content}"}
        levels[level] = levels.get(level, 0) + 1
    return {"schema_version": "xlab.paper_keynotes.v1", "keynotes": result}, levels


def build(args):
    run = Path(args.run_dir).resolve()
    cwd = Path(args.cwd).resolve() if args.cwd else run.parents[2]
    original = read_json(run / "manifest.json")
    if original.get("run_id") != args.run_id or original.get("status") != "success":
        raise ValueError("Resources require the selected successful Survey manifest")
    survey = read_json(run / "artifacts/survey.json")
    citations = read_json(run / "artifacts/citations.json")
    markdown = (run / "artifacts/survey.md").read_text()
    source_path = Path(original["inputs"]["graph_path"])
    source_graph = (source_path if source_path.is_absolute() else cwd / source_path) / "artifacts/graph.db"
    snapshot = Path(args.model_snapshot).resolve()
    if not (snapshot / "modules.json").is_file():
        raise ValueError("An explicit local sentence-transformer snapshot is required")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model = SentenceTransformer(str(snapshot), local_files_only=True, trust_remote_code=False, device="cpu")
    dimension = model.get_sentence_embedding_dimension()
    if dimension != 384:
        raise ValueError("OutcomeRAG requires the 384-dimensional all-MiniLM-L6-v2 snapshot")
    root = run / "resources"
    if root.exists():
        raise ValueError("Resource directory already exists; inspect it before rebuilding")
    for directory in ["survey", "graph", "component-index", "keynotes", "models/minilm"]:
        (root / directory).mkdir(parents=True, exist_ok=True)
    for filename in ["survey.json", "survey.md", "citations.json"]:
        shutil.copyfile(run / "artifacts" / filename, root / "survey" / filename)
    for file in sorted(snapshot.rglob("*")):
        if file.is_file():
            target = root / "models/minilm" / file.relative_to(snapshot)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, target)
    print(json.dumps({"run_id": args.run_id, "phase": "graph_and_keynotes"}), flush=True)
    papers, core_notes, components = collect_graph_resources(source_graph, root / "graph/graph.db")
    keynotes, levels = package_keynotes(run, papers, core_notes, citations)
    atomic_write_json(root / "keynotes/keynotes.json", keynotes)
    texts = [f"{r['component']}\n{r['summary']}\n{r['paper_title']}" for r in components]
    print(json.dumps({"run_id": args.run_id, "phase": "embedding_components", "components": len(components)}), flush=True)
    vectors = model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    index = faiss.IndexFlatIP(dimension)
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    faiss.write_index(index, str(root / "component-index/faiss.index"))
    atomic_write_json(root / "component-index/meta.json", {"meta": components})
    parent = "sha256:" + hashlib.sha256(b"file\0" + (run / "artifacts/survey.json").read_bytes()).hexdigest()
    graph_parent = "sha256:" + hashlib.sha256(b"file\0" + source_graph.read_bytes()).hexdigest()
    provenance = {"source_graph_sha256": sha256_file(source_graph), "source_graph_run": source_graph.parent.parent.name}
    resources = {
        "survey": describe(root, "survey", "survey", "survey", "xlab.literature_survey.v1", "survey-agent.v1", "citation_traces", [parent]),
        "graph": describe(root, "graph", "graph", "graph", "xlab.paper_graph.v1", "sqlite-paper-graph.v1", "paper_neighbors", [graph_parent], provenance=provenance),
        "component_index": describe(root, "component-index", "components", "component_index", "xlab.component_index.v1", "faiss-flat-ip.v1", "component_similarity", [graph_parent], dimensions={"embedding": dimension, "vectors": len(components)}, provenance={"model_id": MODEL_ID, "normalization": "l2", **provenance}),
        "keynotes": describe(root, "keynotes", "keynotes", "keynotes", "xlab.paper_keynotes.v1", "keynote-cache.v1", "paper_keynotes", [parent, graph_parent]),
        "models": [describe(root, "models/minilm", role, "model", "xlab.model_snapshot.v1", "sentence-transformers.v1", "sentence_embedding", [], dimensions={"embedding": dimension}, provenance={"model_id": MODEL_ID, "role": role, "source_revision": snapshot.name}) for role in ["outcome_model", "component_model"]],
    }
    atomic_write_json(root / "resource_manifest.json", {"schema_version": "xlab.research_idea.resources.v1", "bundle_id": f"{args.run_id}-resources", "resources": resources})
    candidate = {**original, "resource_manifest": "resources/resource_manifest.json"}
    resolution = resolve_survey_resources(survey_manifest_path=run / "manifest.json", survey_manifest=candidate, survey=survey,
                                          survey_json_path=run / "artifacts/survey.json", cwd=cwd, model_cache_root=run / "state/idea_model_cache")
    if not resolution.passed:
        raise ValueError(resolution.blocking_errors)
    registry = CitationRegistry.from_payloads(citations, keynotes)
    paragraphs = parse_survey_markdown(markdown, registry)
    cited = {pid for p in paragraphs for pid in p.paper_ids}
    if not cited or cited - set(registry.keynotes):
        raise ValueError("Not every cited paper has a declared keynote record")
    runtime = PackageNativeEvidenceResources(bundle=resolution.bundle, graph_db_path=resolution.graph_db_path,
        keynote_cache_path=resolution.keynote_cache_path, component_index_dir=resolution.component_index_dir,
        outcome_model_path=resolution.outcome_model_path, component_model_path=resolution.component_model_path)
    result = runtime.retrieve(survey["topic"], markdown, citations, citations["references"], limit=10)
    novelty = build_component_novelty_runtime(resolution)
    operator_hits = novelty.retriever.retrieve_operator_components(survey["topic"], limit=5)
    if not result.selected_outcome_hits or not operator_hits.hits or not result.usage["keynotes"]["records_consumed"]:
        raise ValueError("Native evidence retrieval did not return usable resources")
    verification = {"passed": True, "generated_at": utc_now(), "run_id": args.run_id,
        "resource_manifest": "resources/resource_manifest.json", "components": len(components), "embedding_dimension": dimension,
        "keynotes": len(keynotes["keynotes"]), "keynote_evidence_levels": levels, "cited_papers": len(cited),
        "survey_paragraphs": len(paragraphs), "operator_hits": len(operator_hits.hits),
        "native_usage": result.usage, "model_uris": resolution.model_uris,
        "source_survey_sha256": sha256_file(run / "artifacts/survey.json")}
    atomic_write_json(root / "verification.json", verification)
    write_final_manifest(run / "manifest.json", candidate)
    print(json.dumps({"run_id": args.run_id, "passed": True, "components": len(components), "keynotes": len(keynotes["keynotes"])}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cwd")
    parser.add_argument("--model-snapshot", required=True)
    build(parser.parse_args())
