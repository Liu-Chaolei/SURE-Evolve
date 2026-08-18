# Model Storage And Publication v2

The only published model registry is
`/hpc_stor03/project/oref/nfs/models`. Agent access is read-only. All adapter
generation occurs under `src/sure_eval/models/<model_id>`.

`scripts/finalize_model_adapter.py` creates a content-addressed file manifest
and pending publication request. Runtime caches, prior eval runs, Python cache,
and publication records are excluded. Symlinks are rejected.

The published `publication.json` must contain a human verifier and timestamp,
reference `publication_artifacts.json`, and repeat its checksums. The main-flow
registry recomputes every file checksum and aggregate digest before accepting
the model.
