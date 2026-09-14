# TEDLIUM3 Zipformer evolution

Optimize the native Icefall TEDLIUM3 Zipformer RNN-T using the complete supplied
training split and the final epoch budget in execution_contract. Every search
candidate must finish that entire training budget before its WER can be compared.
Do not use a small training subset, shortened screening training, or later finalist
retraining. Keep the shared 500-token BPE, FP32, seed 42 and allocated world size.
SURE evaluates English WER on the regular dev tier. Never read selection/test
transcripts or optimize against their scores.

XLab proposes exactly three distinct evidence-backed architecture-only hypotheses
per round, with explicit ablations. Keep the optimizer, loss, augmentation, data
sampling and inference/decoding settings fixed. Training each changed architecture
is required execution, not a training-strategy optimization. All candidates train
from scratch using the same full data and final epoch budget as the baseline.

Use SURE_TASK_WRAPPER with --action arch and --parameters-json. For ASR, only
structural train_args_json overrides are allowed. Architecture source edits occur
in the materialized workspace recipe before invoking the wrapper. Generate
artifacts/hyp.txt through real decoding of the final checkpoint.

Train the full-budget baseline once and reuse it throughout the search rounds.
After search, selection compares the retained final checkpoints of the baseline
and strongest candidates using inference only. Holdout runs after the winner is
frozen. Neither phase retrains any model.
