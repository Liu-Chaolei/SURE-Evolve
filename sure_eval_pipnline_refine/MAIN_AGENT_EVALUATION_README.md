# Main-Agent Evaluation Route

## Actual Execution Surface

- VC job: `job-178618239700545279127-junhao-du`
- Queue: `pdgpu-ezkws`
- Compute node: `d6-hpc-gpu-032`
- GPU allocation: 1
- Image: `docker.v2.aispeech.com/sjtu/sjtu_yukai-dujunhao-sure_openai__whisper-large-v3-turbo:v1.0`
- Python: `/opt/conda/bin/python` (CPython 3.11.9)
- Torch: `2.4.1+cu121`
- Checkpoint: `/hpc_stor03/project/oref/nfs/models/openai__whisper-large-v3-turbo/checkpoints/large-v3-turbo.pt`
- Checkpoint SHA-256: `aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a`
- Protocol: `strict_core`

`d6-hpc-gpu-033` was excluded after a mount probe showed that its
`/hpc_stor03` was local XFS and did not expose the sibling `nfs` directory.
The alternate node 032 exposed `/hpc_stor03` through the expected NFS mount and
could read both the repository and checkpoint.

## Inference Chain

```text
hpc_08 audio root + hpc_08 annotation overlay
  -> scripts/prepare_sure_dataset.py
  -> run-local three-row canonical input index
  -> scripts/materialize_predictions_template.py
  -> scripts/generate_predictions_via_server.py
  -> MCP stdio tool: asr_transcribe
  -> model server.py
  -> OpenAI Whisper large-v3-turbo local checkpoint on CUDA
  -> prediction TXT + structured JSONL
  -> scripts/validate_prediction_files.py
```

Validation reported 3 expected, 3 provided, no missing/extra/duplicate/empty
rows, and `is_valid=true`.

## Evaluation Chain And Nodes

```text
scripts/evaluate_predictions.py
  -> sure_eval.evaluation.scripts.run_task(...)
  -> src/sure_eval/evaluation/tasks/asr/routes.yaml
  -> pipeline asr.en.wer.whisper_norm_english_v1.wenet_wer_v1
  -> normalization/whisper_norm v1
  -> scoring/wenet_wer v1
  -> metric report + sample report + protocol.yaml + report.jsonl
```

`normalization/whisper_norm` used the vendored Whisper
`EnglishTextNormalizer`. `scoring/wenet_wer` used corpus edit distance over 53
reference words and recorded 0 substitutions, 0 insertions, 0 deletions, and no
missing or extra utterance keys.

## Why The Earlier TTS Run Could Emit Another Protocol

The earlier main-flow shell treated `strict_core` as a default rather than a
hard policy:

```bash
PROTOCOL_ID="${PROTOCOL_ID:-strict_core}"
```

Therefore an upstream environment value could pass through to the result path
and `evaluate_predictions.py`. TTS made this easier to expose because inference
and audio evaluation were separate surfaces and evaluation-only segments could
inherit or re-supply orchestration variables. The TTS model did not choose the
protocol; the orchestration layer propagated it.

The current main-flow contract separates inference protocol from task, metric,
pipeline, and TTS evaluation segment. It requires `strict_core` at every
boundary, and the canonical shell rejects `standard_system` or custom protocol
values instead of silently accepting them.
