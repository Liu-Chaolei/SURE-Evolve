# TEDLIUM-3 Zipformer model evolution

Use the configured Icefall TEDLIUM Zipformer RNN-T recipe. The original audio
and STM transcripts are read-only. Prepared manifests, BPE and references are
provided through `base_model/data`; recipe code through `base_model/recipe`.

STM intervals define the utterances. Preserve their IDs exactly: do not apply
LibriSpeech ID suffix stripping. Train only on the train manifest. Search uses
the supplied dev reference tier. Selection uses independent dev recordings;
test is a final held-out evaluation, never a source of training or search data.

Generate `artifacts/hyp.txt` with one `ID<TAB>recognized text` row for every
reference ID. Obtain text from real audio decoding. Do not copy supervision
text, spoof scores, import the evaluator, or modify the external source trees.

Call `SURE_ASR_ZIPFORMER_WRAPPER` to execute a candidate. Training and architecture
candidates use `--action train_decode` and produce an independent checkpoint.
Inference candidates use `--action decode_only` and reuse the controller's
checkpoint. Declare the correct `--candidate-type`, and provide explicit
`--train-args-json`, `--decode-args-json`, and `--changed-fields-json` changes.
Set epochs with the wrapper's `--train-epochs` option. Output directories,
world size, epoch selection and decode method are controlled wrapper/config
options; do not override them through the extra-argument JSON arrays.

For the initial compatibility run, keep the RNN-T objective, FP32, one device,
one epoch, and the fixed training subset and seed. On Ascend, the wrapper adapts
the network to NPU and computes k2 losses on CPU with autograd preserved.
Use greedy decoding; unsupported CUDA/k2 beam-search kernels are not NPU
alternatives. Honor the execution contract attached to the XLab idea request.

The controller independently computes English WER (a fraction; lower is better).
A valid experiment may fail to improve WER. Report actual behavior and let the
controller decide whether to adopt the change.
