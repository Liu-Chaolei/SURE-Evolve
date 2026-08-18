# SURE-EVAL Main-Flow Agent Contract v2

This file is the authoritative main-flow agent instruction. Older v1 examples,
shell templates, and contracts are historical unless this file links them.

## Non-Negotiable Boundaries

```text
published model reads:  /hpc_stor03/project/oref/nfs/models
published result reads: /hpc_stor03/project/oref/nfs/results
model staging writes:   src/sure_eval/models
result staging writes:  results
```

- The two NFS paths are the only published registries. Do not discover models
  or reusable results anywhere else.
- NFS is read-only to both agents. Never create, edit, rename, delete, or copy a
  file there.
- Do not accept model, result, output, registry, or NFS paths from user input.
- All generated artifacts for one model append below `results/<model_id>/`.
- Only a human reviewer may transfer a staged publication into NFS.

The executable policy is `config/storage.yaml` and `sure_eval.storage`.

## Structured Input

Accept only `sure.eval.main_flow_input.v2`:

```yaml
schema: sure.eval.main_flow_input.v2
target:
  model_id: Org__Model
datasets:
  - name: aishell1
    version: v1.0.2
    split: test
inference:
  protocol_id: standard_system
execution:
  mode: auto
```

`inference` and `execution` may be omitted. Defaults are
`standard_system` and `auto`. Allowed protocols are exactly
`standard_system` and `strict_core`. Allowed modes are:

- `auto`: reuse an exact report, otherwise reuse exact predictions, otherwise infer.
- `reuse_result`: require an exact verified report or block.
- `reevaluate`: require exact verified predictions and run evaluation only.
- `retest`: ignore prior runs and perform inference plus evaluation.

Parse with `sure_eval.agent.parse_main_flow_input`. Unknown fields fail closed.

## Protocol Policy

[SYSTEM_CONSTRAINT: MAIN_FLOW_PROTOCOL_SELECTION]

- `standard_system` is the default. It runs the pinned official inference
  entrypoint with its official default parameters and an explicitly empty user
  override map.
- `strict_core` is selected only when the user explicitly requests it.
- `strict_core` must resolve every deterministic control through a concrete
  parameter, a true attestation, or `not_applicable` with a reason. Missing
  controls block execution.
- Never silently translate legacy `strict_one`, custom protocol names, or an
  invalid protocol into a supported value.
- Persist `protocol_id`, protocol version, the complete resolution record, and
  `effective_params_sha256` in predictions and reports.

The executable policy is `config/protocols.yaml` and
`sure_eval.protocols.ProtocolResolver`.

## State Machine

```text
INTAKE
  -> RESOLVE_VERIFIED_MODEL
  -> RESOLVE_DATASET_IDENTITIES
  -> RESOLVE_PROTOCOL
  -> PLAN_PINNED_EVALUATION_PIPELINES
  -> DECIDE_RESULT_REUSE
     -> reuse_result: REPORT
     -> reevaluate: EVALUATE_PUBLISHED_INFERENCE
     -> infer_and_evaluate: INFER -> FINALIZE_INFERENCE -> EVALUATE_STAGED_INFERENCE
     -> blocked: STOP
  -> VALIDATE_STAGED_ARTIFACTS
  -> WRITE_PUBLICATION_REQUEST
  -> REPORT
```

Every transition must be represented by a JSON artifact. Do not infer or
evaluate before the preceding identity gates pass.

## Model Gate

Resolve `target.model_id` only through `ModelRegistry.require_verified()`.
Directory name, `config.yaml`, `publication.json`,
`publication_artifacts.json`, every file checksum, and the aggregate model
digest must agree. A local staged model is not evaluation-ready.

## Dataset Gate

The identity is exactly `<source_name>__<version>@<split>`. A reusable dataset
must match all of: name, version, split, manifest SHA256, and sample count.
Task names and metric names are not part of the dataset ID. Dataset discovery
may inspect the approved `/hpc_stor08` platform, but a versionless or inferred
identity must block reuse.

## Result Reuse Gate

Read reusable artifacts only from
`/hpc_stor03/project/oref/nfs/results/<model_id>/result_index.json` through
`ResultRegistry`. The index must be human-verified and pass all checksum and
cross-reference checks.

Comparison is all-or-nothing across the requested dataset set:

1. model ID and model artifact SHA256
2. every dataset name, version, split, manifest SHA256, and sample count
3. protocol ID, version, and effective parameter SHA256
4. for report reuse, pinned engine commit/tree SHA256 and all pipeline IDs

An invalid published index blocks. It must never silently fall back to a new
test. `reevaluate` reuses predictions only; `retest` always creates new
predictions. New runs use unique immutable IDs but remain in the same model
folder.

## Evaluation Gate

[SYSTEM_CONSTRAINT: PINNED_EVALUATION_ENGINE]

All main-flow evaluation must use `scripts/run_evaluation_bridge.py`. The
bridge verifies `config/evaluation_engine.lock.yaml`, then invokes the pinned
standalone `sure-evaluation` source CLI in three stages:

1. `agent plan`
2. `metric describe`
3. `metric run --validate-env`

Preserve engine verification, command traces, raw `report.json`, full
`pipeline_description.json`, pipeline ID, node trace, and conversion trace.
The local `SUREEvaluator`, ad hoc metrics, and `scripts/evaluate_predictions.py`
are not valid v2 main-flow execution surfaces.

## Write And Publication Gate

- New inference: `results/<model_id>/inference_runs/<inference_id>/`
- New evaluation: `results/<model_id>/evaluation_runs/<evaluation_id>/`
- Initial publication: staged `result_index.json` with strategy `initialize_index`.
- Existing NFS model results: checksum-bound `result_delta.json` with strategy
  `merge_delta`.
- After every successful evaluation registration, atomically rebuild
  model-level `report.jsonl` and `report_snapshot.md` from the effective v2
  state. The effective state is either the staged `result_index.json`, or the
  verified NFS index plus the staged `result_delta.json`.
- `report.jsonl` uses `sure.eval.report_row.v2` and contains one deterministic
  row per evaluation pipeline. It must preserve the exact model artifact,
  dataset name/version/split/manifest/sample count, protocol identity, engine,
  pipeline, metric, lineage, and artifact checksums.
- The JSONL and Markdown files are rebuildable review views. Never use them as
  a result-reuse authority; only the verified NFS `result_index.json` is
  authoritative for reuse.
- Always emit `sure.eval.publication_manifest.v2` with checksums for both
  derived reports and `automatic_publish_allowed: false`.

The agent stops after staging. It must not run a copy, merge, or mutation
against NFS.

## Required Structured Outputs

Store run control artifacts below `results/<model_id>/control/<request_id>/`:

```text
00_main_flow_input.json
01_model_resolution.json
02_dataset_resolution.json
03_protocol_resolution.json
04_evaluation_plan.json
05_result_reuse_decision.json
06_execution_plan.json
07_execution_result.json
08_validation_report.json
09_main_flow_run_report.json
```

Every file requires `schema`, `status`, `created_at`, and the relevant exact
identities. A blocked stage records reason codes and does not materialize later
stage outputs as successful.

## Executable Surfaces

- `scripts/generate_predictions_via_server.py --model-id ... --inference-id ... --protocol-id ...`
- `scripts/finalize_inference_run.py --model-id ... --inference-id ... --dataset ...`
- `scripts/run_evaluation_bridge.py --model-id ... --evaluation-id ... --inference-id ... --inference-source ... --input-manifest ...`

All script paths and run IDs are agent-generated from validated identities.
