# SURE Main-Flow Dataset Source Naming Plan

## Goal

Main-flow evaluation should accept only dataset source roots under:

```text
/hpc_stor08/external_ds/aispeech/
```

The canonical dataset output name must be derived from the source dataset
identity:

```text
<source_dataset_name>__<version_id>
```

Example:

```text
/hpc_stor08/external_ds/aispeech/g001/store002/ds_pool/aispeech_phy_aishell-1-test
  + version v1.0.2
  -> aispeech_phy_aishell-1-test__v1.0.2
```

The old projection names such as `aishell1__v1.0.2__asr` should no longer be the
reported dataset identity.

## Current Lifecycle

Current data flow:

```text
allowed_datasets / user input
  -> dataset_decision.selected_datasets
  -> execution surface DATASETS / DATASET
  -> run_single_model.sh DATASET_ARRAY
  -> scripts/prepare_sure_dataset.py
  -> prepare_summary.prepared[].dataset
  -> shell overwrites DATASET_ARRAY
  -> predictions/<dataset>.txt
  -> validation_payload.results[].dataset
  -> metrics/<dataset>/<metric_slug>/
  -> report.jsonl dataset.name
  -> evaluation_payload.results[].dataset
  -> report_snapshot.md
```

The real identity pivot today is `prepare_summary.prepared[].dataset`, which is
currently `jsonl_path.stem`.

## Current Problem

The current code mixes multiple identity sources:

- OREF fallback projection names, for example `aishell1__v1.0.2__asr`
- legacy config names, for example `librispeech_other`
- local JSONL stems, for example `seedtts_test_eval_en`
- source roots under `/hpc_stor08/...`, but only for two ASR datasets

Only these current datasets clearly preserve `/hpc_stor08` source and version:

- `aishell1__v1.0.2__asr`
- `librispeech_clean__v1.0.1__asr`

The source roots are recorded in:

```text
config/oref_datasets.yaml
data/datasets/sure_benchmark/aishell1/oref/source.json
data/datasets/sure_benchmark/librispeech_clean/oref/source.json
data/datasets/sure_benchmark/*/projections/asr_transcription_v1/conversion_report.json
```

## Target Contract

### Accepted Dataset Input

Only accept source roots like:

```text
/hpc_stor08/external_ds/aispeech/g001/store002/ds_pool/<source_dataset_name>
```

Optionally support an explicit version selector later, but the source root must
remain the identity base.

### Rejected Dataset Input

Reject or require migration for:

```text
aishell1
librispeech_clean
seedtts_test_eval_en
data/datasets/sure_benchmark/jsonl/*.jsonl
/hpc_stor03/...
/mnt/gpfs_dataset_hdd/...
```

### Canonical Dataset ID

Use:

```text
dataset_id = source_dataset_name + "__" + version_id
```

Do not append task suffix.

## Implementation Plan

### 1. Add Source Resolver

Add a resolver in `src/sure_eval/datasets/dataset_manager.py`, for example:

```python
resolve_aispeech_source_entry(entry: str) -> DatasetSourceRef
```

It should:

- require the path to be under `/hpc_stor08/external_ds/aispeech/`
- require a `ds_pool/<source_dataset_name>` component
- resolve `sample_files/<version_id>/sample.jsonl`
- resolve `sample_files/<version_id>/ds.jsonl`
- resolve `raws/sample`
- select the version only if exactly one version exists
- fail if multiple versions exist and no explicit version is provided
- produce `dataset_id = <source_dataset_name>__<version_id>`

### 2. Convert Source Root To SURE JSONL

Replace or generalize `_convert_oref_platform_to_jsonl(...)` with a source-root
conversion path.

Output:

```text
data/datasets/sure_benchmark/jsonl/<dataset_id>.jsonl
```

Each JSONL row should include:

```json
{
  "dataset": "<source_dataset_name>__<version_id>",
  "metadata": {
    "source": "aispeech_ds_pool",
    "source_dataset_root": "/hpc_stor08/...",
    "source_dataset_name": "<source_dataset_name>",
    "version_id": "v1.0.2"
  }
}
```

### 3. Update Prepare Script

Modify `scripts/prepare_sure_dataset.py` so `--dataset` means source root, not
legacy dataset name.

`prepare_summary.json` should emit:

```json
{
  "dataset": "<source_dataset_name>__<version_id>",
  "requested_name": "/hpc_stor08/...",
  "source_dataset_root": "/hpc_stor08/...",
  "source_dataset_name": "<source_dataset_name>",
  "version_id": "v1.0.2",
  "jsonl_path": "data/datasets/sure_benchmark/jsonl/<dataset_id>.jsonl"
}
```

The shell can keep its existing lifecycle: after prepare, overwrite
`DATASET_ARRAY` with `prepared[].dataset`.

### 4. Tighten Dataset Normalization

Modify `DatasetManager.normalize_dataset_name(...)`:

- source-root input returns the resolved `dataset_id`
- existing `<dataset_id>.jsonl` returns `<dataset_id>`
- legacy aliases should not silently become reported output identities
- non-`/hpc_stor08/external_ds/aispeech` roots should fail with a clear message

### 5. Update Main-Flow Contracts And Templates

Update:

```text
docs/agents/main_flow_agent/contracts/main_agent_dataset_unit.md
docs/agents/main_flow_agent/templates/main_agent_dataset_decision.json
docs/agents/main_flow_agent/templates/main_agent_execution_surface.json
docs/agents/main_flow_agent/examples/input_*.md
```

The contract should state:

- `selected_datasets[]` stores source roots
- `resolved_datasets[]` stores `dataset_id`, source root, source dataset name,
  version, task, language, and JSONL path
- report-facing identity is always `dataset_id`

### 6. Extend Report Fields

Keep `dataset.name` as the canonical report identity:

```json
"dataset": {
  "name": "<source_dataset_name>__<version_id>",
  "source_dataset_name": "<source_dataset_name>",
  "version_id": "v1.0.2",
  "source_root": "/hpc_stor08/...",
  "task": "ASR",
  "language": "zh"
}
```

Also add the same source object to `evaluation_payload.results[]` so downstream
consumers do not need to inspect local conversion reports.

### 7. Keep Artifact Layout Stable

After prepare, all later artifacts should use the same `dataset_id`:

```text
predictions/<dataset_id>.txt
predictions/<dataset_id>.jsonl
metrics/<dataset_id>/<metric_slug>/
sample_reports/<dataset_id>/<metric_slug>.jsonl
report.jsonl dataset.name = <dataset_id>
evaluation_payload.results[].dataset = <dataset_id>
```

### 8. Migrate Existing Results Deliberately

Do not rewrite historical `results/` in place by default.

Recommended migration:

- mark old runs as legacy naming
- rerun or republish using source-root inputs
- optionally write a one-time migration tool that rewrites report rows and paths
  only when source metadata is complete

### 9. Tests

Add focused tests:

- source root with one version resolves to `<source_dataset_name>__<version_id>`
- multiple versions without explicit version fails
- non-`/hpc_stor08/external_ds/aispeech` input fails
- `prepare_summary.prepared[].dataset` equals dataset_id
- prediction files use dataset_id
- validation payload uses dataset_id
- evaluation payload uses dataset_id
- report.jsonl uses dataset_id
- old `__task` suffix format no longer appears for new runs

## Acceptance Criteria

A new main-flow run is valid only if:

- every selected dataset input is an `/hpc_stor08/external_ds/aispeech/...`
  source root
- every output dataset name is `<source_dataset_name>__<version_id>`
- `report.jsonl`, `evaluation_payload.json`, prediction files, metric artifacts,
  and sample reports all use the same dataset_id
- source root and version are visible in machine-readable reports
- legacy alias names do not appear as new report identities
