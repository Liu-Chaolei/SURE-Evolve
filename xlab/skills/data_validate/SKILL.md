---
name: data-validate
description: Validate a processed JSONL dataset and emit its immutable dataset contract. Use as the final data-workflow gate.
---

# Data validation

Run `scripts/validate.py --input <jsonl> --required-fields <comma-list> --output <dataset.json>`.

Reject malformed JSON, missing fields, duplicate rows, and missing provenance. Record the file checksum and inferred field types. Never mutate the dataset while validating it.
