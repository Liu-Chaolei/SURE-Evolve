---
name: model-eval
description: Evaluate a model against a frozen dataset and metric contract with reproducible evidence.
---

# Model evaluation workflow

Resolve a concrete model runner and a JSONL benchmark dataset before execution. Run the declared stages through `xlab_stage`:

1. `prepare`: invoke `eval_prepare` and freeze digests, metrics, and seed.
2. `evaluate`: invoke `eval_run`; preserve all case outputs and failures.
3. `compare`: if a baseline report exists, invoke `eval_compare`; otherwise skip this optional stage.

Success requires an `evaluation_report` artifact with nonzero case coverage and no failed cases. Finish with a schema v2 manifest whose artifact path points to the report.
