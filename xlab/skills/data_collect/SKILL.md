---
name: data-collect
description: Download selected official data sources while preserving raw payloads, checksums, and blockers. Use after data discovery.
---

# Data collection

Run `scripts/collect.py --plan <source_plan> --output-dir <raw> --manifest <manifest>`. Bound each download with `--max-bytes`.

Keep one directory per source. Never substitute synthetic data or an HTML index for a quantitative payload. Preserve every skipped source with its blocker.
