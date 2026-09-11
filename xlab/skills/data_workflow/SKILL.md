---
name: data-workflow
description: Run a resumable scientific data workflow from source discovery through validated dataset publication. Use for dataset acquisition, curation, and benchmark preparation.
---

# Data workflow

Run the declared stages with `xlab_stage`. Reuse a valid upstream artifact instead of repeating its stage.

Keep raw payloads immutable and source-separated. Process through explicit mappings. Finish only after `data_validate` emits a valid `dataset@1` artifact with checksum and provenance.
