---
name: science-gateway
description: Provide durable forum threads and direct messages through a local stdio MCP service and control CLI.
---

# Science gateway

This package owns both the persistent service and its control plane:

- `scripts/server.py --database <path>` starts the MCP `2025-11-25` stdio server.
- `scripts/gateway.py` performs the same create/post/reply/direct/list/export operations for lifecycle checks and recovery.
- SQLite is the source of truth; every message records sender, channel, task id, run id, artifact references, and timestamp.

Use `gateway.py export` to create the final `communication_manifest`. A post is successful only when the stored message can be read back. Keep this service local unless authentication and transport security are added by a later deployment layer.
