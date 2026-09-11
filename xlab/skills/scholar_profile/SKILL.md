---
name: scholar-profile
description: Build an evidence-grounded scholar research profile and persona prompt from public academic sources. Use for PI personas, research discussions, and scholar analysis.
---

# /xlab build-scholar-profile

Run `scripts/run_pipeline.py` for one scholar. Read configuration only from environment variables; never read or write `.env`.

Stages:

1. Resolve the scholar and DBLP identity.
2. Discover and verify public sources.
3. Build the research mainline graph.
4. Score representative papers.
5. Generate the evidence-grounded persona prompt.

Write JSON and Markdown as authoritative outputs. DOCX is optional. Record unavailable sources and low-confidence conclusions instead of inventing evidence.

Before finishing, create `scholar_profile.json`, `system_prompt.md`, and the XLab manifest v2. The profile must contain the scholar name, source list, mainline graph, representative papers, evidence warnings, and prompt path.
