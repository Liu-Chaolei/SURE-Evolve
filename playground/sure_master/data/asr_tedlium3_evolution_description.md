# TEDLIUM3 Zipformer evolution

Optimize the native Icefall TEDLIUM3 Zipformer RNN-T. Use the supplied fixed
100-hour train subset, shared 500-token BPE, FP32, seed 42, eight NPU ranks,
and exactly ten training epochs. SURE evaluates English WER on the regular
dev tier. Never read selection/test transcripts or optimize against their scores.

XLab proposes four distinct evidence-backed hypotheses per round, freely mixing
architecture, training and supported inference changes, with explicit ablations.
All candidates that update weights train from scratch with the fixed budget;
inherit implementation improvements, never extra training epochs. Inference uses
the controller's frozen round-parent artifact. Keep RNN-T; CUDA-only beam kernels
are unavailable on NPU. Do not change world size, seed, training data, tokenizer,
common max-duration, training budget, evaluator, or accelerator environment.

Use SURE_TASK_WRAPPER with --action arch|fine_tune|infer and --parameters-json.
For ASR, parameters accept train_args_json, decode_args_json, decode_method, decode_avg and use_averaged_model.
Architecture source edits must occur in the materialized workspace recipe before
invoking the wrapper. Always generate artifacts/hyp.txt through real decoding.

The search baseline is trained once, then reused across six to ten rounds. The
controller retrains the unchanged baseline and at most two selected complete
recipes from scratch for thirty epochs on full train. Selection chooses the final
model; holdout scoring is performed only after the choice is frozen.
