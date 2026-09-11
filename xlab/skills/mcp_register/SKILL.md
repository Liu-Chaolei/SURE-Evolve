---
name: mcp-register
description: Register a validated MCP package through an explicit, atomic, reversible config update.
---

# MCP registration

Registration changes external configuration and must be explicitly approved with `--approve`. Run `scripts/register.py` only after `mcp_validate` reports `passed: true`. The script preserves an exact backup, writes through a same-directory temporary file, and emits rollback metadata. Re-running an identical registration is idempotent.
