---
name: paper-fetch
description: Fetch full text for normalized paper candidates with checksums and explicit blockers. Use after paper-search and before paper parsing.
---

# Paper fetch

Run `scripts/fetch.mjs --input <paper_candidates.json> --output-dir <dir> --manifest <path>`.

Download only explicit open-access PDF URLs. Limit response size, verify the PDF signature, compute SHA-256, and preserve one status for every candidate. Never silently drop blocked or unavailable papers.

Use `--offline` in tests to create metadata-only records without network access.
