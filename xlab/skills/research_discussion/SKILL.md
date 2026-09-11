---
name: research-discussion
description: Run a bounded evidence-linked scientific review among explicit roles and produce decisions, not chat logs.
---

# Research discussion

Define the goal, allowed roles, maximum rounds, and source artifacts before the first turn. Scholar profiles may inform role perspectives but never grant additional permissions. Use the `science_gateway` MCP interface when durable forum or direct messaging is requested.

Record turns as JSONL with `round`, `role`, `statement`, `stance`, `citations`, and `action`. Run `scripts/finalize.py` to validate role membership, round bounds, citation presence, and actionable conclusions. The final artifact must separate key points, consensus, disputes, actions, and open questions. Budget exhaustion without actions is incomplete.
