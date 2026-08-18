# SD Fixture Index

SD fixtures validate speaker diarization model outputs. Model input fixtures may
be ordinary audio files, but metric inputs must be MeetEval-loadable annotation
files.

Recommended DER smoke format:

```text
SPEAKER rec1 1 0.00 1.00 <NA> <NA> spk1 <NA> <NA>
```

Metric route:

```text
src/sure_eval/evaluation/tasks/sd/
src/sure_eval/evaluation/nodes/scoring/meeteval/
```

Default scoring uses MeetEval DER with `collar=0.25`. The report records the
selected collar and loader `meeteval.io.load`.
