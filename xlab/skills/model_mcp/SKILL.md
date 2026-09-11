---
name: model-mcp
description: Wrap a local model runner as a self-contained MCP tool and verify the complete protocol lifecycle.
---

# Model MCP workflow

Use `xlab_stage` for each boundary:

1. `wrap`: invoke `model_wrap` with an explicit command, input schema, and example input.
2. `validate`: invoke `mcp_validate`; success requires initialization, discovery, and a real tool call.
3. `register`: skip unless the user explicitly requested registration; registration requires its own `--approve`.

An untested package is incomplete. An unvalidated package must not be registered. The final manifest must include both `mcp_package` and `mcp_validation` artifacts.
