---
name: paper-parse
description: Parse fetched papers, Markdown, text, and MinerU exports into normalized paper documents. Use wherever full text must be shared across graph, survey, novelty, and reproduction skills.
---

# Paper parse

Run `scripts/parse_documents.py --input <fetch_manifest.json> --output-dir <dir> --manifest <path>`.

Prefer existing Markdown or MinerU JSON. For PDF input, use `pdftotext` when available and record a blocker otherwise. Never fabricate missing text. Preserve source metadata and parser provenance.

Finish only when every input paper has a `parsed`, `metadata_only`, or `failed` status and every parsed document has a checksum.
