# LibriSpeech Main-Flow Full Evaluation Design

## Goal

Run the complete `librispeech_test-clean` company dataset through the SURE-EVAL
main-flow inference and evaluation pipeline, publish standard report artifacts,
and deliver three developer READMEs under `sure_eval_pipnline_refine/`.

## Fixed Inputs

- Audio source root:
  `/hpc_stor08/external_ds/aispeech/g001/store002/ds_pool/librispeech_test-clean`
- Model: `openai__whisper-large-v3-turbo`
- Inference protocol: `strict_core`
- Dataset version: `v1.0.1`
- Canonical dataset id: `librispeech_test-clean__v1.0.1`
- Scope: all 2619 source audio files
- Candidate GPU partitions, in live-capacity order:
  `pdgpu-ezkws`, `pdgpu-3090`, `pdgpu-3090-data`

The current source root contains all 2619 WAV files but does not contain
`sample_files/v1.0.1/sample.jsonl` or `ds.jsonl`. The sibling company dataset
root
`/hpc_stor08/external_ds/aispeech/g001/store002/ds_pool/aispeech_phy_librispeech_test-clean`
contains the v1.0.1 annotation metadata. Its annotation audio basenames match
the selected source root exactly: 2619 of 2619, with no missing or extra files.

## Dataset Preparation

Do not modify either hpc_stor08 dataset. The preparation layer accepts the
selected `librispeech_test-clean` root as the report-facing source root and uses
the sibling v1.0.1 metadata only as a read-only annotation/version source.

The prepared canonical JSONL must:

- contain 2619 rows;
- set every row's `dataset` to `librispeech_test-clean__v1.0.1`;
- remap every audio path to the selected
  `.../ds_pool/librispeech_test-clean/raws/sample/` root;
- record both `source_dataset_root` and `annotation_source_root`;
- record `source_dataset_name=librispeech_test-clean` and
  `version_id=v1.0.1`;
- fail if the annotation and selected audio basename sets differ.

This is an input-materialization compatibility path, not a dataset naming
change. The existing `<source_dataset_name>__<version_id>` rule remains intact.

## Main-Flow Execution

Materialize a fresh isolated run from
`docs/agents/main_flow_agent/templates/run_single_model.sh`. Do not reuse any
prior `eval_runs` artifacts. The run must produce the standard routing and
evidence files:

- `task_classification.json`
- `tool_readiness_routing.json`
- `main_agent_plan.json`
- `dataset_decision.json`
- `script_routing.json`
- `execution_surface.json`
- `execution_readiness_report.json`
- `run_evaluation.sh`
- `assessment_report.json`
- `main_agent_run_report.json`
- `model_eval_manifest.json`

The repository model path is currently unavailable because
`src/sure_eval/models` is a dangling symlink. Do not replace or repair that
user-owned symlink. Use a run-local model descriptor under
`sure_eval_pipnline_refine/runtime/` and discover the actual Python, server, and
checkpoint paths inside the declared model image through a bounded vc probe.
The descriptor must still expose the model-local MCP contract consumed by
`scripts/generate_predictions_via_server.py`.

Use `vc submit` for model inference. Query live capacity immediately before
submission and select the first candidate partition with a free GPU. The
selected partition is pinned in `execution_surface.json`; fallback after
submission is forbidden. Run a bounded vc smoke first, then the complete
2619-sample inference with resume enabled.

## Evaluation Route

Evaluation must run through:

```text
scripts/evaluate_predictions.py
  -> sure_eval.evaluation.scripts.run_task(...)
  -> src/sure_eval/evaluation/tasks/asr/routes.yaml
  -> the selected English WER route and its declared nodes
```

No shell-local WER implementation is allowed. Preserve the actual route ID,
node IDs, conversion steps, runtime versions, `pipeline_description.json`, and
metric `report.json`. The main-agent evaluation README must explain this exact
observed route rather than only describing the intended architecture.

## Standard Reports

The completed run must contain:

- `evaluation_payload.json` with schema `sure.eval.payload.v2`;
- `report.jsonl`, one row per dataset metric;
- `protocol.yaml` with `protocol_id: strict_core`;
- `report_snapshot.md`;
- `metrics/librispeech_test-clean__v1.0.1/wer/report.json`;
- `metrics/librispeech_test-clean__v1.0.1/wer/pipeline_description.json`;
- `sample_reports/librispeech_test-clean__v1.0.1/wer.jsonl`;
- a standard results mirror under
  `results/openai__whisper-large-v3-turbo/strict_core/`.

All report-facing dataset fields and artifact paths must use
`librispeech_test-clean__v1.0.1`. Source provenance fields must remain explicit
and must not be replaced by a legacy alias.

## Developer Documentation

Create exactly these three primary documents under
`sure_eval_pipnline_refine/`:

1. `DATASET_README.md`: accepted source root, read-only metadata overlay,
   validation rules, canonical identity, and preparation command.
2. `MAIN_AGENT_EVALUATION_README.md`: all main-flow units, vc queue/image/job,
   inference command, evaluation router, actual ASR route ID, conversion steps,
   nodes, and resume procedure.
3. `REPORT_README.md`: schemas, field meanings, artifact layout, verification
   commands, and the final measured result.

The READMEs must link to the concrete run artifacts and contain no placeholders.

## Failure Handling

- If no candidate partition has a free GPU, wait and re-query rather than using
  an unapproved queue.
- If the image cannot be pulled or its runtime cannot be discovered, preserve
  the vc job ID and logs and stop before fabricating readiness.
- If smoke inference fails, do not launch the full run.
- If prediction generation succeeds but evaluation fails, reuse predictions
  and run evaluation-only through the repository router.
- If any dataset identity differs across prepare, predictions, validation,
  metrics, payload, or report, the run is incomplete.

## Verification

Automated tests cover source-root resolution, metadata overlay mismatch
rejection, all-2619-row preparation, identity propagation, final report schemas,
and main-flow protocol constraints. Runtime acceptance additionally requires a
Completed vc job, 2619 valid predictions, successful prediction validation,
the route-backed WER artifacts, and the three developer READMEs.
