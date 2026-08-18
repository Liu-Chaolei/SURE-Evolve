# SA-ASR Fixture Index

SA-ASR fixtures validate speaker-attributed ASR outputs. Metric inputs must be
MeetEval-loadable annotation files, not ordinary ASR `key<TAB>text` files.

Common STM-style smoke line:

```text
rec1 1 spk1 0.00 3.00 hello world
```

Metric route:

```text
src/sure_eval/evaluation/tasks/sa_asr/
src/sure_eval/evaluation/conversion/sa_asr__cpwer/
src/sure_eval/evaluation/nodes/normalization/gstar_norm/
src/sure_eval/evaluation/nodes/scoring/meeteval/
```

Default scoring converts STM transcript fields to key-text, applies
`normalization/gstar_norm`, converts the normalized text back to STM, then uses
MeetEval cpWER as the main metric and reports DER as a companion metric with
`collar=0.5`. Keep smoke-test segments comfortably longer than the collar on
both boundaries; one-second single-segment DER checks can fail in
`md-eval-22.pl` because the scored region becomes empty.
