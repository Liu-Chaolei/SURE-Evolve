# SURE-Master

Run the existing SURE-Master self-evolution loop from XLab. XLab supplies typed research ideas and receives round summaries; SURE remains responsible for candidate execution, metrics, ranking, staged rungs, checkpoints, and holdout decisions.

## Command

```text
/xlab run-sure-master --config <sure-config> --task "<speech task>"
```

The package launcher is intentionally thin. It validates public arguments, starts the authorized SURE process, waits for completion, and emits a bounded JSON result. The SURE process must be configured with an XLab provider endpoint or provider implementation when round-level idea feedback is enabled.

`SURE_EVOLVE_ROOT` must point to the authorized local checkout. It is never written into public artifacts. API keys, VC credentials, absolute source paths, and evaluator changes are not accepted as package arguments.

The package does not submit GPU or VC work itself. Cancellation, timeout, and resume are reported as incomplete so the durable XLab run can be reconciled without silently repeating an unconfirmed experiment.
