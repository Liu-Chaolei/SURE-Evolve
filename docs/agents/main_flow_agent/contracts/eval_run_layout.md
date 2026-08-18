# Result Layout v2

The v1 model-local `eval_runs` layout is retired. Use the append-only
`results/<model_id>/` layout in [storage_and_publication.md](storage_and_publication.md).

```text
results/<model_id>/
├── inference_runs/<inference_id>/...
├── evaluation_runs/<evaluation_id>/...
├── result_index.json
├── result_delta.json              # only when a verified NFS base exists
├── report.jsonl                   # sure.eval.report_row.v2, one row/pipeline
├── report_snapshot.md             # derived review summary
└── publication_manifest.json      # checksums every publication input/view
```

Run directories are immutable. The index and delta receive new entries. The
two report files are deterministic projections rebuilt atomically after a
successful evaluation entry is registered.
