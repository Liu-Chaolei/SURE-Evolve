---
name: eval-run
description: Run a frozen benchmark and preserve case-level model outputs and metric evidence.
---

# Evaluation execution

Use `scripts/evaluate.py` with exactly one inference source:

- `--predictions`: JSONL records containing `id` and `output`.
- `--runner-command`: executable and arguments; each case is sent as one JSON line on stdin and one JSON object with `output` is expected on stdout.

The dataset digest must match the frozen plan. Treat missing predictions, command timeouts, malformed outputs, and partial case coverage as failures.
