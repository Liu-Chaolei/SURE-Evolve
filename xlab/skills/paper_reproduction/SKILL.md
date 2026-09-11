---
name: paper-reproduction
description: Reproduce one declared paper experiment when the paper, official code, data, metrics, and compute budget are explicit.
---

# Paper reproduction

This is a bounded second-phase workflow, not an assertion that arbitrary papers are reproducible. Start only when the user supplies a paper, an official code checkout with an immutable revision, a target experiment, expected metrics, and resource budget.

Use `xlab_stage`:

1. `parse`: extract the target method, data, command, environment, and reported result.
2. `data`: prepare a provenance-preserving dataset.
3. `execute`: run the frozen official-code experiment in an isolated environment.
4. `evaluate`: compare measured and reported metrics under an explicit tolerance.

Run `scripts/report.py` after all stages. Missing official code, unlicensed data, unresolved dependencies, resource exhaustion, or incomparable metrics must produce `incomplete` or `failed`, never success.
