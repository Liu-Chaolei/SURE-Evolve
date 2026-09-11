---
name: eval-compare
description: Compare a candidate evaluation with a baseline under declared metric thresholds.
---

# Evaluation comparison

Run `scripts/compare.py` on two reports produced by `eval_run`. The reports must cover the same benchmark plan and case count. Record every metric delta and a threshold-based pass decision; never infer success from a single aggregate string.
