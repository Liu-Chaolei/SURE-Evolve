---
name: novelty-check
description: Produce evidence-linked idea duplication risks and explicit uncertainty, not a binary novelty verdict.
---

# Novelty check

Normalize the idea into question, method, data, experiment, and contribution claims. Run `scripts/check.py` against a paper set for a deterministic candidate ranking, then inspect graph relations and improve the comparison.

Every risk statement and differentiation must cite paper or graph evidence. Fewer than three usable papers, missing claim decomposition, or weak retrieval coverage requires `low` confidence and an `incomplete` finish with a recommendation to run `paper_search` or `paper_collect`. Never describe this output as manuscript peer review or proof of novelty.
