---
name: research-workflow
description: Run the core XLab evidence-to-idea workflow with durable artifact handoff and novelty feedback.
---

# Research workflow

Advance `collect → graph → survey → idea → novelty` using `xlab_stage`. Pass artifact-store references between stages; do not call source scripts behind a dependency package.

If novelty confidence is low, block the novelty stage and resume collection with an expanded query. If duplication risk is high, preserve the report and start a new idea attempt rather than overwriting the prior candidate. Respect each stage's maximum attempts. Finish only when every required stage succeeded and the final novelty report has at least medium evidence confidence.
