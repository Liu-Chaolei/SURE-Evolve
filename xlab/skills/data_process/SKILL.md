---
name: data-process
description: Normalize collected CSV, JSON, and JSONL payloads into a provenance-preserving dataset. Use after raw collection and before validation.
---

# Data processing

Define an explicit JSON mapping from output field to input field. Run `scripts/process.py --input <file> --format <csv|json|jsonl> --mapping <mapping> --output <jsonl> --report <report>`.

Do not create train/test splits. Keep `_source` and `_source_row` provenance. Record invalid and duplicate rows.
