# Dataset Resolution Unit v2

Resolve each requested `{name, version, split}` against the approved
`/hpc_stor08` dataset platform. Do not select a latest version implicitly.

The output records source path, annotation path when applicable, exact
`name__version` ID, split, manifest SHA256, sample count, task, language, and
resolution evidence. The set must equal the user's requested set. Missing,
ambiguous, versionless, task-suffixed, or duplicate identities block.

Dataset paths are resolution evidence, not user-selectable output roots.
