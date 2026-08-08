# Zipformer ASR Model Optimization on LibriSpeech

## Codebase and Recipe Location

- **Icefall repository**: `/home/liuchaolei/ASR/icefall`
- **Recipe directory**: `egs/librispeech/ASR/zipformer`
- **Full recipe path**: `/home/liuchaolei/ASR/icefall/egs/librispeech/ASR/zipformer`

The recipe contains:
- `train.py` — Training entry point with `get_parser()` and `run()` functions
- `decode.py` — Decoding entry point that computes WER on test sets
- `model.py` — Model definition (encoder, decoder, joiner)
- `zipformer.py` — Zipformer encoder architecture
- `asr_datamodule.py` — Lhotse data loading and batching
- `optim.py`, `scaling.py`, `subsampling.py` — Supporting modules

## Model Architecture

- **Name**: Zipformer (pruned transducer stateless)
- **Type**: Streaming-capable encoder-transducer ASR model
- **Default configuration (normal-scaled)**:
  - Model parameters: ~65.55 M
  - Encoder layers: configurable (default 2,2,4,5,4,2 per stack)
  - Encoder dimensions: configurable (default 192,256,512,768,512,256)
  - Uses downsampling, reduced attention, and scalable architectures

Reference: https://github.com/k2-fsa/icefall/pull/1058

## Dataset

- **LibriSpeech**:
  - `test-clean` — Clean test set, ~5.7 hours
  - `test-other` — Noisy/other test set, ~5.1 hours
- **Training data**: `train-clean-100`, `train-clean-360`, `train-other-500` (combined via `--full-libri 1`)
- **Data format**: Lhotse cuts manifests (precomputed fbank features)
- **BPE model**: 500 tokens (`lang_bpe_500`)

## Evaluation Metric

**WER (Word Error Rate)** computed on both test sets:
- `test-clean WER` — Lower is better
- `test-other WER` — Lower is better

The decode.py script outputs WER in formats like:
```
test-clean WER 2.25
test-other WER 5.06
```

**Optimization goal**: Reduce WER on both test-clean and test-other. A common practice is to report the **average WER** or focus on improving the harder set (test-other).

## Baseline Results (from RESULTS.md)

Normal-scaled model (65.55 M parameters), trained for 50 epochs:

| decoding method      | test-clean | test-other | epoch/avg |
|----------------------|------------|------------|-----------|
| greedy_search        | 2.22       | 4.87       | 50/25     |
| modified_beam_search | 2.21       | 4.79       | 50/25     |
| fast_beam_search     | 2.21       | 4.82       | 50/25     |

With LM rescoring:
| modified_beam_search_rescore | 2.04 | 4.39 | 40/16 |

**State-of-the-art reference** (large model + more epochs):
- test-clean: 2.00
- test-other: 4.38

Your goal is to approach or beat these baselines through hyperparameter/architecture improvements.

## Example Training Command

```bash
cd /home/liuchaolei/ASR/icefall/egs/librispeech/ASR

export CUDA_VISIBLE_DEVICES="0,1,2,3"
./zipformer/train.py \
  --world-size 4 \
  --num-epochs 5 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir ./zipformer/exp_opt \
  --causal 0 \
  --full-libri 1 \
  --max-duration 1000
```

For single-GPU smoke tests, set `CUDA_VISIBLE_DEVICES="0"` and `--world-size 1`.

Key hyperparameters you can modify:
- `--num-epochs`: Training epochs (use reduced number like 5 for fast iteration)
- `--base-lr`: Base learning rate (default varies by model size)
- `--num-encoder-layers`: Encoder stack layers (e.g., `2,2,2,2,2,2` for small)
- `--encoder-dim`: Encoder dimensions per stack (e.g., `192,256,256,256,256,256`)
- `--feedforward-dim`: Feedforward dimensions
- `--max-duration`: Max batch duration (affects batch size)
- `--causal`: Causal vs non-causal model (0 = non-streaming, 1 = streaming)

## Example Decoding Command

```bash
cd /home/liuchaolei/ASR/icefall/egs/librispeech/ASR

export CUDA_VISIBLE_DEVICES="0"
./zipformer/decode.py \
  --epoch 5 \
  --avg 1 \
  --use-averaged-model 0 \
  --exp-dir ./zipformer/exp_opt \
  --max-duration 600 \
  --decoding-method modified_beam_search
```

Decoding outputs WER for both test-clean and test-other. Your wrapper script should parse this output and print the final WER as `\boxed{WER_value}` (use average WER or test-other WER).

## What You Should Optimize

You can explore improvements in:

1. **Model Architecture**
   - `--num-encoder-layers`: Number of encoder layers per stack (e.g., `2,2,3,4,3,2`)
   - `--encoder-dim`: Encoder hidden dimensions (e.g., `256,384,512,640,512,384`)
   - `--feedforward-dim`: Feedforward layer dimensions
   - `--encoder-unmasked-dim`: Dimensions for unmasked (non-attention) layers
   - `--causal`: Streaming vs non-streaming mode

2. **Training Schedule**
   - `--base-lr`: Learning rate (e.g., 0.04, 0.08, 0.1)
   - `--num-epochs`: Total epochs (5 for fast iteration, 30-50 for final)
   - `--start-epoch`: Starting epoch (usually 1)
   - LR scheduler parameters (inside train.py, can be patched)

3. **Data Augmentation** (SpecAugment params in train.py)
   - Time mask factor, frequency mask factor
   - Speed perturbation (if enabled)
   - `--max-duration`: Batch duration limit

4. **Regularization**
   - Dropout rates (inside model.py)
   - Module dropout
   - Weight decay (optimizer params)

5. **Decoding Strategy**
   - `--decoding-method`: `greedy_search`, `modified_beam_search`, `fast_beam_search`
   - `--beam-size`: Beam width for beam search methods
   - LM rescoring parameters (if external LM available)

## Constraints

- Use the **existing icefall recipe** (`train.py` + `decode.py`) — do NOT rewrite the entire training pipeline
- Use a **reduced number of epochs** (e.g., 5) per iteration for faster experimentation
- Single-node GPU training. Use `--world-size 1` for one GPU or `--world-size N` with `CUDA_VISIBLE_DEVICES` listing N GPUs for DDP.
- The data is **already prepared** (lhotse cuts + BPE lang) — do NOT run `prepare.sh`
- Your wrapper script must:
  1. Set `sys.path` and `os.chdir()` to the icefall recipe directory
  2. Call `train.py` via subprocess with your hyperparameter overrides
  3. Call `decode.py` via subprocess on the trained checkpoint
  4. Parse decode output for WER (test-clean and test-other)
  5. Print the final WER as `\boxed{WER_value}` (average or test-other)

## Success Metric

**WER (Word Error Rate)** — lower is better.

Target: Beat the baseline (test-clean ~2.2, test-other ~4.8) or approach SOTA (test-clean 2.0, test-other 4.38).

Report the **test-other WER** or **average WER** as the optimization metric.
