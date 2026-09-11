---
name: knowledge-graph
description: Parse PDFs from a successful paper_collect run with MinerU, enrich MinerU structures with PaperGraph Step 1 references, run PaperGraph-style high-recall Step 2 extraction/calibration, and build a PaperGraph-compatible queryable method graph for surveys, ideation, and novelty analysis.
---

# Knowledge graph

Select one successful `paper_collect` run directory and produce MinerU documents,
PaperGraph Step 1 structures, validated Step 2 extractions, `method_graph@2`,
`graph_db@1`, a static graph preview, and `graph_report@2`.

The intended algorithm is PaperGraph-equivalent after an XLab-specific mandatory
MinerU parsing stage:

```text
successful paper_collect run
  -> MinerU PDF parse
  -> PaperGraph Step 1 references and structure
  -> PaperGraph Step 2 high-recall extraction and calibration
  -> PaperGraph Step 3 citation-aware graph merge
  -> PaperGraph-compatible JSONL, SQLite/FTS, visualization, and audit
```

## Environment contract

Before `pre_start`, XLab checks these required values and opens masked input
dialogs for every missing value:

- `KNOWLEDGE_GRAPH_LLM_API_URL`: OpenAI-compatible base URL or full
  `/chat/completions` endpoint.
- `KNOWLEDGE_GRAPH_LLM_API_KEY`: key for that LLM endpoint.
- `KNOWLEDGE_GRAPH_LLM_MODEL`: provider model identifier.

Optional values:

- `HF_TOKEN`: used only by MinerU/Hugging Face when that installation requires
  authenticated model access.
- `SEMANTIC_SCHOLAR_API_KEY` or `S2_API_KEY`: used only by the self-contained
  Step 1 reference resolver for Semantic Scholar Graph API requests.
- `KNOWLEDGE_GRAPH_DISABLE_SEMANTIC_SCHOLAR=1`: disables Semantic Scholar
  enrichment and relies on upstream citation edges plus local References-section
  parsing.
- `KNOWLEDGE_GRAPH_SEMANTIC_SCHOLAR_API_URL`,
  `KNOWLEDGE_GRAPH_SEMANTIC_SCHOLAR_TIMEOUT`, and
  `KNOWLEDGE_GRAPH_SEMANTIC_SCHOLAR_RETRIES`: optional resolver controls.

These settings are intentionally independent of `OPENAI_API_KEY` and the coding
agent's model provider. Never print, log, or persist LLM or Semantic Scholar
keys.

## Rules

- Use only package-local Python code and the pinned MinerU installation.
- Do not import or execute Xcientist, XAgora, XForge, PaperGraph, or another
  skill's scripts at runtime.
- Do not use MCP, `curl`, `wget`, arbitrary shell networking, or ad hoc provider
  scripts.
- The first argument must be a directory from a successful `paper_collect` run.
  Do not accept an arbitrary PDF, Markdown file, or manifest as the workflow
  input.
- The upstream run must provide a schema-version-2 `manifest.json` with
  `skill_name: paper_collect`, `status: success`, and `validation.passed: true`.
- Use only the upstream run's canonical `artifacts/papers.manifest.json`,
  `artifacts/metadata/edges.jsonl`, and `artifacts/pdfs` tree. Downloaded PDFs
  must use safe relative paths and match their upstream `download.sha256`.
- A PDF is parsed only when MinerU produces all three required records:
  Markdown, `*_content_list.json` (or v2), and `*_middle.json`. The middle JSON
  is the MinerU layout/VLM record consumed by the Step 1 adapter.
- Never treat `pdftotext`, an abstract, or metadata as a successful full-paper
  parse.
- Run each paper's LLM passes in order: main ideation extraction, high-recall
  Baseline/Dataset extraction, then calibration and local-reference alignment.
- Calibration may remove false positives and may add missing concrete baselines
  or datasets, but every kept or added semantic entity must have an exact quote
  from MinerU Markdown.
- Do not accept a title-derived fallback Core. Every Core, semantic entity,
  claim, and semantic edge must survive deterministic name and exact-quote
  grounding against MinerU Markdown.
- Keep metadata-only papers as Paper nodes. Do not infer semantic nodes for
  them.
- For an explicitly requested same-model endpoint handoff, the operator-only
  `build_graph.py migrate-endpoint --run-dir <run> --api-url <url> --model <model>`
  command records an immutable migration ledger. Completed records can be
  retained only if their recorded hashes and original producer context still
  validate. Their original endpoint provenance is preserved; unfinished LLM
  passes remain bound to the endpoint that produced them.
- Preserve checksums and per-stage checkpoints. Reuse a checkpoint only when its
  parser/prompt version, model/endpoint, selected text, semantic references,
  candidate hints, and source checksums match.

## Invocation

```text
/xlab build-knowledge-graph "<successful paper_collect run directory>" \
  [--mineru-backend auto|pipeline|vlm-engine|hybrid-engine] \
  [--mineru-method auto|txt|ocr] [--mineru-language en] \
  [--mineru-device auto|cpu|cuda|npu] [--mineru-batch-size N] \
  [--gpu INDEX]... [--mineru-workers N] [--llm-workers N] \
  [--llm-rpm N] [--llm-max-input-chars N] \
  [--llm-disable-thinking] \
  [--min-papers N] [--min-mineru-coverage R] \
  [--min-structure-success R] [--min-extraction-success R]
```

Quote the directory. It may be the prior run root or its `artifacts` directory.
Repeat `--gpu` to explicitly select devices.

The default resource policy probes `nvidia-smi`, respects
`CUDA_VISIBLE_DEVICES`, and selects devices with at least 24 GiB free memory and
at most 20 percent utilization. `auto` uses `vlm-engine` when at least one
eligible GPU exists; otherwise it uses CPU `pipeline` processes. CPU workers are
limited by the CPU affinity allocation and `OMP_NUM_THREADS` (default 16).
GPU parsing
uses at most one MinerU batch process per selected GPU. `--mineru-workers`
limits that count. LLM extraction is separate and defaults to two papers in
parallel; the three passes for one paper remain sequential.

For Qwen-compatible endpoints, `--llm-disable-thinking` sends
`chat_template_kwargs.enable_thinking=false`. The setting participates in LLM
checkpoint validation; omit it for endpoints that do not support that option.

Use `--mineru-device npu --mineru-backend pipeline` inside an Ascend allocation
with `ASCEND_RT_VISIBLE_DEVICES` and the pinned MinerU/torch-npu environment.
NPU workers are limited to allocated visible devices. The packaged NPU launcher
disables online operator compilation in spawned MinerU processes.

Use explicit `--gpu`, `--min-gpu-free-gb`, or `--max-gpu-utilization` only when
the automatically written `artifacts/resource_plan.json` does not reflect the
actual scheduler allocation. Never oversubscribe a busy GPU merely to increase
throughput.

## Procedure

The `pre_start` hook validates the successful upstream run, resolves canonical
`paper_collect` artifacts, checks LLM configuration, probes MinerU/GPU resources,
and writes:

- `artifacts/request.json`
- `artifacts/resource_plan.json`
- `artifacts/paper_documents.json`

`paper_documents.json` records prior-manifest, paper-set, citation-edge, PDF-root,
and per-PDF checksums. Read the resource plan before running long stages.

1. Parse PDFs with MinerU:

```bash
<python> <package_dir>/scripts/build_graph.py mineru \
  --run-id "<run_id>" \
  --run-dir "<run_dir>"
```

Give this command a timeout appropriate for the collection, normally at least
six hours. PDFs are assigned stable hashed names and partitioned into one
worker batches of at most `--mineru-batch-size` PDFs (default 24). Completed
batches update the manifests immediately. Identical PDFs share validated parse
bundles under `.xlab/cache/mineru`; cache keys include source hash, parser and
model configuration, device, method, and language. Each run retains its own
upstream provenance and subsequent extraction context.
Partial outputs are accepted only when the
required Markdown/content-list/middle bundle is complete. Worker logs are under
`artifacts/logs/`.

2. Run the self-contained PaperGraph Step 1 adapter:

```bash
<python> <package_dir>/scripts/build_graph.py structure \
  --run-id "<run_id>" \
  --run-dir "<run_dir>"
```

This converts MinerU text, headings, images, tables, equations, bounding boxes,
page indexes, and middle-layout summaries into per-paper structures under
`artifacts/paper_structures/`. It builds `semantic_references` in PaperGraph
order from upstream `paper_collect` citation edges, optional Semantic Scholar
DOI/ArXiv/paperId/CorpusId/title lookups, then local References-section parsing
as fallback. Step 1 cache validity includes MinerU checksums, upstream
provenance, citation-edge hash, and semantic-reference hash.

3. Run the PaperGraph Step 2 LLM extraction:

```bash
<python> <package_dir>/scripts/build_graph.py extract \
  --run-id "<run_id>" \
  --run-dir "<run_dir>"
```

Give this command at least several hours for a large collection. Provider errors
use bounded exponential retry and `Retry-After`. Invalid JSON, missing Core
contributions, ungrounded Core records, and other deterministic validation
failures trigger a bounded correction request. Validated pass checkpoints are
written atomically under `artifacts/llm/`; final paper records are under
`artifacts/extractions/`. Logs contain request hashes, timing, status, usage,
and errors, but never prompts or keys.

Exhausted service/transport retries pause extraction and leave remaining papers
`pending`, preserving validated passes. Resume the extraction stage after the
service recovers. Pending papers block successful audit even if coverage ratios
would otherwise pass.

Step 2 injects high-recall regex hints for compared methods and datasets. The
calibration pass receives compact local references, local contexts, candidate
hints, current graph data, and Core names; it may delete false positives or add
missing concrete baselines/datasets when they are grounded by exact quotes.

If Markdown exceeds `--llm-max-input-chars`, the extractor prioritizes the
abstract, introduction, method, experiments, results, limitations, conclusion,
and references. Quotes are still checked against the complete Markdown, and
truncation is recorded in `quality.input_truncated`.

4. Build the graph and SQLite database:

```bash
<python> <package_dir>/scripts/build_graph.py build \
  --run-id "<run_id>" \
  --run-dir "<run_dir>"
```

The build uses PaperGraph-style signatures, hard-conflict checks, weighted token
overlap, strong alias matching, and strict registry matching. Baselines first try
to merge into cited Core nodes through `citation_paperId`/`citation_paper_id`;
merge happens only when the Core match is confident, and survey/review target
Cores are skipped. Otherwise Baseline/Dataset nodes are deduplicated through the
registry or created as new semantic nodes.

The Core/Baseline/Dataset semantic subgraph is PaperGraph-compatible. XLab keeps
additional Paper nodes plus `introduces`, `cites`, and `recommended_with` edges
for provenance and workbench navigation. JSONL rows expose flattened
PaperGraph-compatible fields such as `full_name`, `label`, source paper
metadata, aliases, citation metadata, summaries, metrics, insights, quotes, and
`edge_type`.

SQLite is built through a temporary file with foreign keys, integrity checks,
wide PaperGraph-compatible node/edge columns, indexes, and FTS over names,
acronyms, paper titles, summaries, keywords, aliases, citation titles, and TLDR.
A static preview is written to `artifacts/visualizations/overview.html`; it is
not opened automatically.

5. Audit:

```bash
<python> <package_dir>/scripts/build_graph.py audit \
  --run-id "<run_id>" \
  --run-dir "<run_dir>"
```

Read `manifest.json` and `artifacts/graph_report.json`. Finish with the exact
manifest status. Audit exit code 2 means the artifacts are readable but a
configured success gate is incomplete.

Audit blocks on noncanonical upstream provenance, hash drift, missing MinerU
bundles, invalid Step 1/2 checkpoints, ungrounded semantic evidence, invalid
endpoints, JSONL/SQLite count mismatches, missing wide DB columns, `relation` and
`edge_type` mismatch, or failed FTS queries. Low Baseline/Dataset coverage and
ambiguous aliases are warnings.

If interrupted, rerun the same stage. The following command is also
checkpoint-safe, but separate stages are preferred because they expose resource
and provider failures more clearly:

```bash
<python> <package_dir>/scripts/build_graph.py resume \
  --run-id "<run_id>" \
  --run-dir "<run_dir>"
```

## Success gates

Defaults require at least one input paper, MinerU success for 90 percent of
downloaded PDFs, Step 1 success for 98 percent of MinerU parses, Step 2 success
for 90 percent of Step 1 papers, Core coverage of 90 percent, and exact-quote
evidence on 95 percent of semantic edges. Required MinerU bundle checksums,
unique IDs, supported node/relation types, valid endpoints, validated three-pass
LLM records, JSONL/SQLite consistency, SQLite integrity, foreign-key integrity,
wide-column compatibility, `edge_type` consistency, FTS queryability, and a
successful sample query are blocking.

Low Baseline/Dataset coverage and ambiguous aliases are warnings because paper
types legitimately differ. Missing PDFs remain visible as metadata-only papers;
failed MinerU or LLM processing counts against the configured coverage gates.

## Outputs

- `artifacts/resource_plan.json`
- `artifacts/mineru/`
- `artifacts/mineru.manifest.json`
- `artifacts/paper_documents.json`
- `artifacts/paper_structures/`
- `artifacts/paper_structures.manifest.json`
- `artifacts/llm/`
- `artifacts/extractions/`
- `artifacts/extractions.manifest.json`
- `artifacts/nodes.jsonl`
- `artifacts/edges.jsonl`
- `artifacts/entity_aliases.jsonl`
- `artifacts/graph.db`
- `artifacts/visualizations/overview.html`
- `artifacts/method_graph.json`
- `artifacts/graph_report.json`
- `artifacts/failures.jsonl`
- `manifest.json`
