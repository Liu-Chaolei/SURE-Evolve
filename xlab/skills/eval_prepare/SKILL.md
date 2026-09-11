---
name: eval-prepare
description: Freeze a model evaluation contract before any benchmark process is started.
---

# Evaluation preparation

Use `scripts/prepare.py` to create `benchmark-plan.json`. Require an existing JSONL dataset whose records contain `id`, `input`, and `expected`. Record its SHA-256 digest, model identity, metric definitions, seed, and optional baseline. Do not silently choose a different dataset or metric after the plan is written.

Example:

```bash
python scripts/prepare.py --model ./model --dataset cases.jsonl --task-type text --output benchmark-plan.json
```
