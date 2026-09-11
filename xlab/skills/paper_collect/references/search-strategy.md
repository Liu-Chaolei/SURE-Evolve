# Search strategy

## Objective

Build a high-recall paper pool for graph construction. The later survey selects a smaller core from this collection.

## Seed coverage

Create three global MCP web-search lanes:

- surveys and reviews, whose references cover the field backbone
- seminal and foundational papers
- recent work from roughly the last three years

Add one MCP web-search and one Semantic Scholar query per user-relevant facet. Prefer three to eight facets. Avoid many near-duplicate keyword variants.

## Semantic Scholar expansion

Canonicalize MCP web-search candidates by DOI, arXiv ID, Semantic Scholar paper ID, or exact title. Then combine:

1. topic and facet relevance searches
2. references from survey and seminal seeds
3. citations from seminal and recent seeds
4. joint recommendations from the strongest seeds

Use one relation hop first. Promote only `core` or strongly supported `related` papers to seeds for another round.

For title-only canonicalization, query Semantic Scholar with at most three
results and retain only a strict normalized-title match. A provider rank alone
is not relevance evidence for title resolution.

## Relevance classes

- `core`: direct high-ranked topic result or strong title/abstract match
- `related`: adequate relevance score, support from at least two seeds, or one graph edge plus textual support
- `boundary`: weak or unresolved evidence

Keep boundary candidates in `raw_candidates.jsonl`, but do not place them in the selected paper set or expand them.

## Stop policy

Stop when any condition holds:

- soft target paper count reached
- two consecutive rounds add less than the configured novelty ratio
- maximum candidate count reached
- Semantic Scholar call budget exhausted
- no unseen enrichment or graph-expansion actions remain

The audit requires an explicit stop reason and no pending actions.
Target shortfall is a warning rather than a failed collection.
Provider-expansion failure, exhausted graph expansion with no edges, and a
graph-connected-paper ratio below the configured threshold are incomplete
outcomes rather than successful convergence.

## Survey coverage

Make the survey consume `paper_set@3`. If a later survey discovers an additional DOI, arXiv ID, or Semantic Scholar ID, feed it back as an exact enrichment candidate instead of maintaining an independent collection.
