---
name: data-discover
description: Rank public datasets and benchmarks with access, license, schema, and collection evidence. Use as the first stage of XLab data workflows.
---

# Data discovery

Write a candidate JSON file, then run `scripts/discover.py --input <candidates> --output <source_plan>`. Candidate entries require a stable ID, official URL, role, and collection readiness.

Do not invent sources. Record probe failures and registration blockers. Select a benchmark before training and preserve its metric, split, license, and expected schema.
