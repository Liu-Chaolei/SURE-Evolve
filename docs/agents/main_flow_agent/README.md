# SURE-EVAL Main-Flow Agent

The main-flow agent evaluates a human-published model against versioned
datasets, with exact result reuse and a pinned evaluation engine. Its
authoritative contract is [AGENTS.md](AGENTS.md).

## Input

```yaml
schema: sure.eval.main_flow_input.v2
target:
  model_id: OpenAI__whisper-large-v3
datasets:
  - name: librispeech
    version: v1.0.0
    split: test-clean
inference:
  protocol_id: standard_system  # optional; this is the default
execution:
  mode: auto                    # auto|reuse_result|reevaluate|retest
```

To request the constrained deterministic protocol, set only:

```yaml
inference:
  protocol_id: strict_core
```

Paths are deliberately absent from the input. Models and existing results are
read only from `/hpc_stor03/project/oref/nfs/models` and
`/hpc_stor03/project/oref/nfs/results`. New artifacts are staged only under the
repository's `results/<model_id>/` directory.

## Decisions

| Mode | Exact report | Exact predictions only | No exact predictions |
|---|---|---|---|
| `auto` | reuse report | reevaluate | infer and evaluate |
| `reuse_result` | reuse report | block | block |
| `reevaluate` | reevaluate | reevaluate | block |
| `retest` | infer and evaluate | infer and evaluate | infer and evaluate |

“Exact” includes the model artifact digest, complete versioned dataset set,
protocol resolution digest, and, for report reuse, evaluation engine and
pipeline identities.

## Outputs

All new inference and evaluation runs append below one local model directory:

```text
results/<model_id>/
├── inference_runs/<inference_id>/...
├── evaluation_runs/<evaluation_id>/...
├── result_index.json
├── result_delta.json              # when NFS already contains this model
├── report.jsonl                   # deterministic pipeline rows
├── report_snapshot.md             # human-readable history
└── publication_manifest.json      # pending_human_review
```

Each reevaluation contributes a new immutable evaluation run. After the index
or delta is committed, the two model-level reports are atomically rebuilt from
the effective history. With existing NFS results, that history is the verified
NFS index plus the local delta. These reports are derived views; reuse decisions
continue to read only the verified NFS `result_index.json`.

The agent never publishes automatically. A human verifies and manually merges
or transfers the staged artifacts to the same model directory in NFS.

## Evaluation Engine

`scripts/run_evaluation_bridge.py` executes the exact standalone
`sure-evaluation` checkout pinned by `config/evaluation_engine.lock.yaml`. It
retains plan, route description, environment validation, raw report, pipeline
description, and command trace artifacts. This is the only v2 evaluation
entrypoint.
