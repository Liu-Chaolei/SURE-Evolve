---
name: paper-search
description: Search Semantic Scholar and arXiv and normalize ranked paper candidates. Use as the evidence-search stage for collection, novelty checking, scholar profiles, and reproduction.
---

# Paper search

Run `scripts/search.mjs` with a query, limit, and output path. Use `--offline` only for tests.

Preserve source IDs, URLs, rank, query, and provider errors. Deduplicate by DOI, arXiv ID, then normalized title. Never represent an index page as a downloaded paper.

Finish only when the `paper_candidates@1` artifact passes the package schema and has no duplicate `dedupe_key`.
