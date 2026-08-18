# Result Reuse Contract v2

Reuse candidates exist only in the verified NFS `result_index.json` for the
requested model. Local staging, archived reports, database rows, report
snapshots, and model-local `eval_runs` are never reusable evidence.

The requested dataset collection and a candidate inference collection must be
equal as sets. Partial overlap does not permit partial reuse.

Required comparisons:

1. model ID and artifact SHA256
2. dataset ID (`name__version`), split, manifest SHA256, and sample count
3. protocol ID, version, and effective parameter SHA256
4. engine commit, engine tree SHA256, and exact pipeline set for report reuse
5. checksums of every referenced prediction and evaluation artifact

Malformed or tampered published data returns `blocked`, not “no result.”
`retest` bypasses lookup intentionally. `reevaluate` requires a matching
published inference and never runs model inference.
