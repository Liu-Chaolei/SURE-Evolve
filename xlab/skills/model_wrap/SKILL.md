---
name: model-wrap
description: Generate a standalone stdio MCP tool server for a local model runner.
---

# Model wrapper generation

Require an explicit model command, tool name, input JSON Schema, and an example input. Run `scripts/generate.py`. It generates a Python-standard-library server that implements MCP `2025-11-25` initialization and `tools/list` / `tools/call`, uses newline-delimited stdio JSON-RPC, and sends logs only to stderr.

The generated package owns its server, schema, configuration, and dependency lock. It must not import this skill package or any source repository.
