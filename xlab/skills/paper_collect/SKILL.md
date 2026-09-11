---
name: paper-collect
description: Build a broad, graph-ready scholarly paper pool from MCP web-search seed discovery and Semantic Scholar topic, citation, reference, and recommendation expansion, then download all verifiable open-access PDFs. Use before literature surveys, paper knowledge graphs, novelty analysis, and research planning when recall and relation coverage matter more than a short citation list.
---

# Paper collection

Produce `paper_set@3`, `paper_edges@1`, and `collection_report@1`. The collection must be broader than a survey bibliography and suitable for later knowledge-graph construction.

## Rules

- Use only the Python implementation bundled in this skill package.
- Use the configured HTTP MCP web-search endpoint only through the bundled Python provider.
- Do not call web search or Semantic Scholar with `curl`, `wget`, or ad hoc scripts.
- Never import or execute code from Xcientist, XAgora, XForge, or PaperGraph.
- MCP web search is discovery evidence. Semantic Scholar is the canonical metadata and relation source.
- Never invent missing metadata. Keep papers with missing abstracts or PDFs and record missing fields explicitly.
- Do not filter solely by citation count or PDF availability.
- Never echo, log, or persist `SEMANTIC_SCHOLAR_API_KEY`. `WEB_SEARCH_MCP_URL` is non-secret and defaults to `http://127.0.0.1:17890/mcp`.

Read [references/search-strategy.md](references/search-strategy.md) only when explaining or changing the search policy. Read [references/provider-contracts.md](references/provider-contracts.md) only when diagnosing provider response or PDF normalization failures.

## Invocation

```text
/xlab collect-papers "<topic>" [--target-papers N] [--max-papers N] [--facet "<facet>"]... [--download-workers N] [--download-per-host N]
```

The topic must be first and quoted. Repeat `--facet` for at most eight subfields.
Require `1 <= target-papers <= max-papers`. Treat the target as a soft
expansion-stop threshold, not an exact output count or success gate. Defaults
are target 500, safety maximum `max(3 × target, 1500)`, eight global download
workers, and two workers per host.

## Procedure

The `pre_start` hook validates the invocation and creates `artifacts/request.json` and `artifacts/query_plan.json`.

1. Read those two files.
2. Collect provider metadata and graph relations. Give the bash tool at least a
30-minute timeout:

```bash
<python> /absolute/path/to/<package_dir>/scripts/paper_collect.py collect \
  --run-id "<run_id>" \
  --run-dir "<run_dir>"
```

This phase performs MCP web-search seed searches, Semantic Scholar topic/facet searches,
strict title resolution, metadata enrichment, citation/reference/recommendation
expansion, deduplication, and relevance scoring. Every provider response is
checkpointed in `artifacts/logs/provider_results.jsonl`.

If collection is interrupted, rerun it through the collect-only resume alias:

```bash
<python> /absolute/path/to/<package_dir>/scripts/paper_collect.py resume \
  --run-id "<run_id>" \
  --run-dir "<run_dir>"
```

Successful provider calls are skipped. Retryable failures are attempted once per
resume; permanent failures are preserved without being retried indefinitely.

3. Download PDFs as a separate phase. Give the bash tool at least a 60-minute
timeout:

```bash
<python> /absolute/path/to/<package_dir>/scripts/paper_collect.py download \
  --run-id "<run_id>" \
  --run-dir "<run_dir>"
```

If interrupted, rerun the same `download` command. Valid PDFs and `.part` files
are reused. The downloader checkpoints every completed paper, limits both
global and per-host concurrency, and rejects a second mutation phase for the
same run while one is active.

4. Audit separately:

```bash
<python> /absolute/path/to/<package_dir>/scripts/paper_collect.py audit \
  --run-id "<run_id>" \
  --run-dir "<run_dir>"
```

5. Read `manifest.json` and `artifacts/collection_report.json`. Update XLab
state with the reported counters, then finish with the exact manifest status
(`success` or `incomplete`).

Do not use the all-in-one `run` command from the slash workflow. Do not manually
execute query-plan or follow-up entries.

## Success gates

Success requires both provider evidence, unique stable identities, accepted
relevance and metadata ratios, at least one
valid retained edge, the configured graph-connected-paper ratio, terminal PDF
outcomes, valid downloaded PDF files, and an explicit non-failure search stop
reason with no pending follow-ups.

Target shortfall, low abstract coverage, and terminal open-access download
failures remain warnings. Graph connectivity below the configured gate remains
an incomplete outcome.

## Outputs

- `artifacts/papers.manifest.json`
- `artifacts/metadata/papers.jsonl`
- `artifacts/metadata/edges.jsonl`
- `artifacts/metadata/seeds.json`
- `artifacts/pdfs/`
- `artifacts/logs/provider_results.jsonl`
- `artifacts/logs/downloads.jsonl`
- `artifacts/logs/download_results.jsonl`
- `artifacts/followups.json`
- `artifacts/failures.jsonl`
- `artifacts/collection_report.json`
- `manifest.json`

Developer-only offline package check outside the slash-command runtime:

```bash
<python> /absolute/path/to/<package_dir>/scripts/paper_collect.py smoke \
  --run-id "<run_id>" \
  --run-dir "<temporary_run_dir>"
```
