---
name: mcp-validate
description: Verify MCP initialization, tool discovery, and a real example tool call over stdio.
---

# MCP validation

Run `scripts/validate.py` against `mcp-package.json`. The validator launches the generated server, negotiates protocol `2025-11-25`, sends `notifications/initialized`, lists tools, invokes the declared tool with the saved example, and records stderr plus every JSON-RPC response. A generated file without a successful call is not valid.
