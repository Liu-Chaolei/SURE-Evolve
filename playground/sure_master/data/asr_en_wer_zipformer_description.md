# SURE English ASR WER Optimization with Zipformer

## Task and Metric

This task optimizes the SURE `asr_en_wer` task.

- **Canonical task**: ASR
- **Language**: English
- **Primary metric**: WER
- **Optimization direction**: lower is better
- **SURE route**: `asr.en.wer.whisper_norm.wenet_wer`

SURE is used only as the read-only evaluator. Candidate scripts must generate
ASR hypotheses; EvoMaster calls SURE after artifact generation to compute WER.

## Base Model and Recipe Location

The task must start from the configured icefall LibriSpeech Zipformer base
model/recipe. In every SURE Master experiment workspace, the base model is
available through workspace-relative paths:

- **Recipe directory**: `base_model/recipe`
- **Data directory**: `base_model/data`
- **Icefall root**: `base_model/root`

The current default source paths in the SURE Master config are:

- **Icefall repository**: `/hpc_stor03/sjtu_home/chaolei.liu/ASR/icefall`
- **Recipe source**: `/hpc_stor03/sjtu_home/chaolei.liu/ASR/icefall/egs/librispeech/ASR/zipformer`
- **Data source**: `/hpc_stor03/sjtu_home/chaolei.liu/ASR/icefall/egs/librispeech/ASR/data`

Use the workspace paths in generated code, not the external source paths. The
recipe contains:

- `train.py` - Training entry point
- `decode.py` - Decoding entry point
- `model.py` - Transducer model definition
- `zipformer.py` - Zipformer encoder architecture
- `asr_datamodule.py` - Lhotse data loading and batching
- `optim.py`, `scaling.py`, `subsampling.py` - Supporting modules

## Model Architecture

- **Name**: Zipformer
- **Framework**: icefall
- **Type**: Encoder-transducer ASR model
- **Typical recipe scale**: normal Zipformer, around 65M parameters
- **Key components**:
  - Zipformer encoder
  - Transducer decoder
  - Joiner
  - Lhotse-based data module
  - k2/icefall training and decoding utilities

The agent may optimize model structure only through workspace-local code,
wrappers, subclasses, monkey patches, copied configs, or checkpoints saved under
the current experiment workspace. It must not edit the external icefall source
tree.

## Dataset and Inputs

The Zipformer recipe expects prepared LibriSpeech-style icefall data:

- Lhotse cuts/manifests
- Precomputed fbank features
- BPE language resources such as `lang_bpe_500`

SURE scoring for `asr_en_wer` requires a reference transcript:

```text
input/ref.txt
```

SURE Master also passes the configured absolute role path from `sure.inputs.ref`.
For compatibility, the framework copies required configured input files into
their task-card default `input/` paths inside each experiment workspace before
running `run_sure.py`.

The reference file should use key-tab-text lines:

```text
utt_id<TAB>reference text
```

For self-evolution search, the default configured reference is a tiered
LibriSpeech dev subset such as:

```text
playground/sure_master/data/asr_librispeech_regular_ref.txt
```

The full `test-clean/test-other` reference remains available as the final
holdout/official benchmark:

```text
playground/sure_master/data/asr_en_wer_ref.txt
```

Do not use the holdout/test reference for routine search or candidate
selection.

The candidate script must generate:

```text
artifacts/hyp.txt
```

with matching utterance keys:

```text
utt_id<TAB>recognized text
```

The configured reference path is read-only input. Candidate scripts may read it
directly or read the workspace-local `input/ref.txt` copy, but must not write
back to the configured reference file.

## Runtime Environment

The GPT/Kimi SURE Master configs inject these environment variables when
executing `run_sure.py`:

```bash
SURE_MAX_TRAIN_EPOCHS=1
SURE_MAX_DURATION=1000
SURE_USE_FP16=1
SURE_DECODE_ONLY=0
SURE_ENABLE_MUSAN=0
SURE_ICEFALL_PYTHON=/opt/conda/envs/icefall/bin/python
SURE_ASR_EVAL_SPLITS=dev-clean,dev-other
SURE_ASR_ZIPFORMER_WRAPPER=/hpc_stor03/sjtu_home/chaolei.liu/Agent/SURE-Evolve/playground/sure_master/tools/run_icefall_zipformer_candidate.py
```

Use these values to decide whether to train, how large the batches should be,
whether FP16 is allowed, whether MUSAN augmentation is enabled, and which Python
executable should run the icefall recipe. `SURE_MAX_DURATION` may be configured
as a fixed integer or as `auto`; when it is `auto`, resolve it inside
`run_sure.py` after constructing the final `train.py` architecture/loss
arguments. Run `SURE_RUNTIME_ENV_HELPER resolve-max-duration` with
`SURE_DURATION_TRAIN_ARGS_JSON` set to those exact final extra arguments, write
the helper output to a workspace log file, and parse the final integer line as
the resolved bound. This ensures compact and large Zipformer candidates probe
their own real structure instead of a shared baseline structure.
Only accept that integer when the helper exits successfully. Never recover a
duration from failed helper logs, tracebacks, command lines, argparse usage text,
or diagnostics; if the helper fails, exit non-zero. Do not train with a
`--max-duration` lower than `SURE_DURATION_AUTOTUNE_MIN` or
`SURE_TRAIN_DURATION_MIN`.
The Zipformer recipe declares `--max-duration` as an integer option: normalize
values with `int(float(value))` and pass an integer string such as `1000`, never
`1000.0`.

For generated training/decode candidates, call `SURE_ASR_ZIPFORMER_WRAPPER`
instead of spawning `base_model/recipe/train.py` or `decode.py` directly. The
wrapper keeps logging, duration probing, retry behavior, checkpoint placement,
decode split handling, and `artifacts/hyp.txt` formatting consistent. Candidate
freedom is still expressed through JSON CLI args:

```python
cmd = [
    os.environ["SURE_ASR_ZIPFORMER_WRAPPER"],
    "--candidate-type", "arch",
    "--action", "train_decode",
    "--idea-text", "reduce middle encoder dimensions",
    "--train-args-json", json.dumps(["--encoder-dim", "192,256,448,640,448,256"]),
    "--decode-args-json", json.dumps(["--decoding-method", "modified_beam_search"]),
]
```

SURE Master may also inject `CUDA_VISIBLE_DEVICES` and `ASR_WORLD_SIZE` from the
local GPU scheduler. For multi-GPU training, set:

```python
world_size = int(os.environ.get("ASR_WORLD_SIZE", "0") or "0")
if world_size <= 0:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    world_size = len([x for x in visible.split(",") if x.strip()]) or 1
```

Pass `--world-size str(world_size)` to `train.py` and use a unique
`--master-port`. Do not hard-code `CUDA_VISIBLE_DEVICES=0` or `--world-size 1`
when multiple visible GPUs are available. Keep `decode.py` single-process.

The prepared Zipformer data currently does not include `data/fbank/musan_feats`.
When `SURE_ENABLE_MUSAN=0`, pass `--enable-musan 0` to `train.py`. Do not
recover from missing decode/training outputs by copying the reference transcript
or by extracting supervision text from Lhotse manifests into `artifacts/hyp.txt`.
`artifacts/hyp.txt` must contain model-decoded ASR hypotheses.
Do not fill `artifacts/hyp.txt` with a constant token, fallback token,
BPE-only prior, or other low-diversity placeholder when real model decoding is
unavailable; exit non-zero instead.

This Zipformer recipe's `train.py` accepts `--bpe-model` and `--manifest-dir`.
It does **not** accept `--lang-dir`. Use `--lang-dir` only with `decode.py`
when the selected decoding method needs the language directory.

Some LibriSpeech Zipformer `decode.py` versions hardcode
`test-clean/test-other`, while the configured local recipe may already honor
`SURE_ASR_EVAL_SPLITS`. Candidate scripts must decode the splits named by
`SURE_ASR_EVAL_SPLITS`, usually `dev-clean,dev-other` during search and
`test-clean,test-other` only for final benchmark runs. First inspect
`base_model/recipe/decode.py`: if it already supports `SURE_ASR_EVAL_SPLITS`,
call it directly through the workspace-relative path. Only create a
workspace-local wrapper for older recipe versions that still hardcode
`test-clean/test-other`. Do not edit the external icefall source tree.

Icefall `recogs-*.txt` keys may be Lhotse cut ids rather than canonical
LibriSpeech supervision ids. Before writing `artifacts/hyp.txt`, normalize keys
to the `speaker-chapter-utterance` id used by `input/ref.txt` by removing any
trailing numeric cut/segment suffix regardless of suffix length. For example,
`1089-134686-0000-0` and `2086-149214-0000-1708` must both map to
`1089-134686-0000` and `2086-149214-0000` respectively. Do not use a suffix
length guard such as `len(parts[-1]) <= 3`; write hypotheses in `input/ref.txt`
order.

## Evaluation Metric

SURE computes English ASR WER by comparing:

- reference: `input/ref.txt` or the configured `sure.inputs.ref`
- hypothesis: `artifacts/hyp.txt`

The metric route applies SURE's English normalization and WeNet WER scoring.
The final score is the SURE-reported WER, and lower is better.

Unlike `asr_master`, the candidate script should not print or spoof a boxed WER
score. It should only produce the required artifacts. SURE Master runs the
metric after the script exits successfully.

## Expected Candidate Script Behavior

Generated `run_sure.py` should:

1. Run from the current experiment workspace.
2. Use `base_model/recipe`, `base_model/data`, and `base_model/root` as the
   only base-model paths.
3. Train, fine-tune, decode, or run inference with the Zipformer recipe.
4. Save all new model code, configs, checkpoints, logs, and intermediate files
   under `models/` or `working/`.
5. Generate `artifacts/hyp.txt` in SURE ASR format.
6. Exit non-zero if real hypotheses cannot be produced.

It must not:

- import or call `sure_eval`
- write under `/hpc_stor03/sjtu_home/chaolei.liu/sure`
- write to the external icefall recipe/data/root source directories
- switch to an unrelated ASR model family such as Whisper or Wav2Vec2
- generate placeholder transcripts or placeholder scores
- fill hypotheses with one constant/fallback token when real decoding is unavailable
- copy the reference transcript or Lhotse supervision transcripts as hypotheses

## Example Workspace-Oriented Training Command

The candidate script may launch training through subprocesses from the
experiment workspace. A typical command shape is:

```bash
cd base_model/recipe
python train.py \
  --world-size "$ASR_WORLD_SIZE" \
  --num-epochs 5 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir ../../models/zipformer_exp \
  --causal 0 \
  --full-libri 1 \
  --max-duration "$SURE_MAX_DURATION" \
  --enable-musan 0 \
  --bpe-model data/lang_bpe_500/bpe.model \
  --manifest-dir data/fbank
```

Adjust `python` to the configured icefall Python executable if needed.

If no existing `epoch-*.pt` checkpoint is available under the workspace-local
experiment directory, do not fail decode-only by default. Train the configured
Zipformer recipe for the allowed number of epochs, save checkpoints under
`models/zipformer_exp`, and decode from that checkpoint. If `SURE_DECODE_ONLY=1`,
fail explicitly when no checkpoint is available.

## Example Workspace-Oriented Decoding Command

```bash
cd base_model/recipe
python decode.py \
  --epoch 5 \
  --avg 1 \
  --use-averaged-model 0 \
  --exp-dir ../../models/zipformer_exp \
  --max-duration "$SURE_MAX_DURATION" \
  --decoding-method modified_beam_search \
  --bpe-model data/lang_bpe_500/bpe.model \
  --lang-dir data/lang_bpe_500 \
  --manifest-dir data/fbank
```

The candidate script must convert decode output into SURE's required
`artifacts/hyp.txt` key-tab-text format.

Lhotse decode outputs may use cut ids with a final segment suffix, for example
`1089-134686-0000-0`, while SURE refs use the supervision id
`1089-134686-0000`. Before writing `artifacts/hyp.txt`, normalize LibriSpeech
cut ids to the reference key format by stripping the final numeric segment
suffix.

## Optimization Ideas

Useful directions include:

1. **Model structure**
   - Encoder layer counts
   - Encoder dimensions
   - Feedforward dimensions
   - Causal vs non-causal mode
   - Dropout or module-dropout changes

2. **Training**
   - Learning rate and schedule
   - Epoch count for fast iteration vs final runs
   - Batch duration and memory-aware settings
   - FP16 usage
   - Checkpoint averaging strategy

3. **Data and regularization**
   - SpecAugment parameters
   - Speed perturbation if available
   - Filtering or batching choices compatible with the prepared manifests

4. **Decoding**
   - Greedy search vs modified beam search vs fast beam search
   - Beam size and decoding batch duration
   - Checkpoint averaging and epoch selection
   - Text normalization/post-processing before writing `hyp.txt`

5. **Artifact generation**
   - Preserve exact utterance keys from the reference set.
   - Ensure one hypothesis line per required utterance.
   - Avoid extra score text inside `artifacts/hyp.txt`.

## Baseline Reference

The upstream icefall Zipformer LibriSpeech recipe reports WERs around:

| decoding method | test-clean | test-other |
| --- | ---: | ---: |
| greedy_search | 2.22 | 4.87 |
| modified_beam_search | 2.21 | 4.79 |
| fast_beam_search | 2.21 | 4.82 |

These numbers are reference points for Zipformer recipe quality. SURE Master's
actual optimization target is the configured SURE `asr_en_wer` dataset and its
SURE-computed WER.

During search, compare candidates on the configured dev tier. Reserve the full
`test-clean/test-other` reference for the final reported benchmark only.

## Success Criteria

A successful candidate:

- Uses the configured Zipformer base model through `base_model/*`.
- Produces `artifacts/hyp.txt`.
- Passes SURE boundary checks.
- Runs to completion in the experiment workspace.
- Receives a valid SURE WER score.
- Improves WER relative to previous SURE Master candidates.
