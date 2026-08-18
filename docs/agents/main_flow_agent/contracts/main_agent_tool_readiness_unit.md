# Verified Model Resolution Unit v2

Resolve the model only with `ModelRegistry.require_verified(model_id)` from the
fixed NFS models registry. Local adapters, alternative shared roots, and
unverified NFS directories are not ready.

The output records model ID, model artifact SHA256, publication manifest
SHA256, task capability, protocol capabilities, and either `verified` or a
blocking reason. Integration repair is handed to the model tool agent.
