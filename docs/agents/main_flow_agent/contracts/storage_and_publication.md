# Storage And Publication Contract v2

## Capabilities

| Capability | Fixed root | Agent access |
|---|---|---|
| published model registry | `/hpc_stor03/project/oref/nfs/models` | read only |
| published result registry | `/hpc_stor03/project/oref/nfs/results` | read only |
| model staging | `src/sure_eval/models` | read/write |
| result staging | `results` | read/write |

No environment variable, user input, compatibility alias, or discovered path
may replace these roots. `config/storage.yaml` is the single declaration.

## Publication

Agents stage a request with `automatic_publish_allowed: false` and stop.

- A new model uses `publication_request.json` plus
  `publication_artifacts.json`.
- A first result publication uses the staged `result_index.json`.
- Additional results use `result_delta.json`, bound to the current published
  index by path and SHA256.
- `report.jsonl` and `report_snapshot.md` are deterministic derived views over
  the staged index, or over the verified published index plus the staged delta.
  They are refreshed only under `results/<model_id>/`; NFS remains read-only.
- `publication_manifest.json` uses `sure.eval.publication_manifest.v2` and
  binds both report views by path and SHA256, including the exact index/delta
  source hashes used to build them.

The report JSONL is logically append-only because immutable evaluation entries
are appended to the canonical state. Its physical file is rebuilt under a lock
and atomically replaced, so a partial line is never exposed. The snapshot is a
human-readable projection of the same rows. Neither file is an independent
source of truth.

The reviewer copies immutable run directories first, verifies all checksums,
then creates or merges the index and adds human identity/timestamp fields.
Automatic NFS mutation is a contract violation.
