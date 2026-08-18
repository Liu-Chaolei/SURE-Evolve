# Dataset Read Path

This validation reads company data from hpc_08. It does not copy audio into
`sure_eval_pipnline_refine` and does not maintain a second persistent dataset
tree there.

## Selected Source

- Audio source: `/hpc_stor08/external_ds/aispeech/g001/store002/ds_pool/librispeech_test-clean`
- Audio layout: `raws/sample/*.wav`
- Source audio count: 2619
- Annotation overlay: `/hpc_stor08/external_ds/aispeech/g001/store002/ds_pool/aispeech_phy_librispeech_test-clean`
- Annotation version: `sample_files/v1.0.1`
- Access mode: read-only for both hpc_08 roots

The selected audio root contains the 2619 WAV files but no `sample_files`
metadata. The annotation overlay supplies the matching `sample.jsonl` and
`ds.jsonl`. Before evaluation, the reader verified that the annotation and
audio basename sets match exactly.

## Bounded Selection

The run prepared the canonical source once, retained only the first three rows
in its run-local input index, and removed the full generated projection copy.
The three selected audio paths still point directly to `hpc_stor08`.

The audit file is:

`/hpc_stor03/project/oref/nfs/models/openai__whisper-large-v3-turbo/eval_runs/main_agent_openai_whisper_librispeech_test-clean_3sample_20260808_001/input_subset_manifest.json`

## Dataset Identity

The existing naming rule was not changed:

`source_dataset_name__version_id`

For this source it resolves to `librispeech_test-clean__v1.0.1`. The same value
appears in `prepare_summary.json`, prediction filenames,
`evaluation_payload.json`, `protocol.yaml`, and the formal `report.jsonl`.
