# SURE Master

`sure_master` is a multi-task speech self-evolution playground. It treats
SURE as a read-only metric backend: candidate scripts generate standard
artifacts in the experiment workspace, then EvoMaster calls SURE to score them.

For a complete setup and operation guide, see [USAGE.md](USAGE.md).
For the internal architecture, self-evolution loop, runtime lifecycle, and data
flow, see [TECHNICAL_MANUAL.md](TECHNICAL_MANUAL.md).

The SURE source tree under `/hpc_stor03/sjtu_home/chaolei.liu/sure` must not be
modified by candidates or by the playground. Candidate model code, checkpoints,
configs, predictions, audio, and intermediate files belong under the current
experiment workspace.

## Workflow

```text
draft -> execute run_sure.py -> SURE metric -> debug if needed
research -> improve ideas -> execute + SURE metric -> select best
knowledge_promotion -> next round
```

## Workspace Contract

Each experiment uses:

```text
run_sure.py
models/
artifacts/
metric/
working/
```

`run_sure.py` may change model structure and save independent model artifacts
under `models/`, but it should only generate the SURE-required outputs under
`artifacts/`. SURE scoring is called by `SureMetricRunner`, not by the generated
script.

## Task Cards

`task_cards/sure_tasks.yaml` maps SURE task IDs to canonical task names,
primary metrics, score direction, required roles, and default artifact paths.
Select a task through:

```yaml
sure:
  task_id: "asr_en_wer"
```

Override input or artifact paths with:

```yaml
sure:
  inputs:
    ref: "/path/to/ref.txt"
    hyp: "artifacts/hyp.txt"
  execution_env:
    SURE_MAX_TRAIN_EPOCHS: "1"
    SURE_MAX_DURATION: "600"
```

When `split_workspace_for_exp` is enabled, relative paths are resolved inside
each experiment workspace. Use absolute paths for shared read-only inputs.
Required configured inputs are copied to their task-card default `input/` paths
inside each experiment workspace before `run_sure.py` executes, so legacy
candidate scripts can still read paths such as `input/ref.txt`. Use
`session.local.symlinks` only when a large shared input directory should appear
inside every workspace.

## Base Model Profiles

Tasks may declare a required `base_model` profile. The resolved profile is
injected into prompts and converted into workspace symlinks such as
`base_model/recipe`, `base_model/data`, and `base_model/root`. Candidate scripts
must optimize from those workspace-relative paths and must not write to the
external source directories.

`asr_en_wer` is configured to start from the icefall LibriSpeech Zipformer
recipe. Set `sure.base_models.asr_en_wer.source_paths` in the run config to the
actual recipe, data, and icefall root paths available on the machine.
If no Zipformer checkpoint is available, keep `SURE_MAX_TRAIN_EPOCHS` positive
so candidates train a workspace-local checkpoint before decoding.
ASR self-evolution configs should use a dev-tier reference such as
`playground/sure_master/data/asr_librispeech_regular_ref.txt` together with
`SURE_ASR_EVAL_SPLITS=dev-clean,dev-other`. Keep
`playground/sure_master/data/asr_en_wer_ref.txt` and
`SURE_ASR_EVAL_SPLITS=test-clean,test-other` for final holdout benchmark runs.
By default `sure.require_base_model` is true, so other tasks must define their
own `sure.base_models.<task_id>` profile before they can run.

## Production Remote Resources

Production ASR/TTS profiles keep the coordinator GPU-free with
`sure.coordinator.local_gpu_policy: disabled` and submit every candidate to VC.
Inference uses 1 GPU / 8 CPU / 32G; draft training, training, and architecture
candidates use 8 GPU / 64 CPU / 256G. Each resource profile uses `num_task: 1`.
Do not apply a global remote GPU override, because it would replace these
candidate-specific profiles. Remote Icefall uses `/opt/conda/envs/icefall/bin/python`.
