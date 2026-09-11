# Provider and download contracts

## HTTP MCP web search

Treat web-search results as discovery evidence only. Read `title`, `url`, `content` or `snippet`, and provider score. Extract DOI and arXiv identifiers conservatively. Never treat snippets as canonical abstracts.

Use academic-domain filtering for seed discovery. The bundled client connects to `WEB_SEARCH_MCP_URL` (default `http://127.0.0.1:17890/mcp`) using the stateless MCP JSON-RPC lifecycle and calls the `search` tool. It normalizes structured or JSON text results into `{"results": [{"title", "url", "content", "score"}]}`. No web-search credential is stored by XLab.

## Semantic Scholar

Accept structured paper records from search, exact paper lookup, citations, references, and recommendations. Normalize:

- missing arrays to empty arrays
- missing scalar fields to null
- author objects to `{author_id, name}`
- external ID key variants to stable keys
- citation wrappers through `citingPaper`
- reference wrappers through `citedPaper`

The bundled client accepts JSON responses from search, exact paper lookup, citations, references, and recommendations. Preserve unrecognized responses as explicit provider failures rather than inventing records.

Retry incomplete search records once through exact paper lookup. Keep records with a stable ID and title even when the abstract, authors, year, venue, or PDF is missing.

The bundled client sends `SEMANTIC_SCHOLAR_API_KEY` only in the case-sensitive
`x-api-key` header. It uses Academic Graph paper search/detail/citation/reference
endpoints and the Recommendations API. Search is capped at 100 papers per call,
relations at 1000, and recommendations at 500.

Use endpoint-specific field sets. Search and exact paper detail may request
`tldr`; citation, reference, and recommendation endpoints must omit it. The
client filters caller-supplied fields against the relevant allowlist before
sending a request.

Title-based enrichment is resolution, not topic search. Accept a returned paper
only when its normalized title is exact or passes the configured high-similarity
check. Do not retain the other top search results.

## Request failures

- never store request headers or API keys
- serialize Semantic Scholar calls through one rate limiter shared by Graph and
  Recommendations endpoints
- retry network timeouts, 408, 425, 429, and 5xx responses with bounded backoff
- honor `Retry-After` up to 60 seconds
- fail immediately on 401 or 403 so XLab can repair credentials
- checkpoint both successful and failed logical calls as JSONL
- skip identical successful calls on resume; retry recorded failures once per resume

## PDF candidates

Try candidates in this order:

1. `openAccessPdf.url`
2. arXiv PDF derived from `externalIds.ArXiv`
3. Web-search direct PDF URL tied to the same title or stable identifier

Do not use a Semantic Scholar landing-page URL as a PDF. Do not bypass paywalls.

## Download validation

- allow only public HTTP or HTTPS hosts
- block credentials, loopback, private, link-local, multicast, reserved, and unresolved addresses
- revalidate redirect destinations
- honor size, timeout, retry, and per-host concurrency limits
- interleave hosts and bound queued work so one dominant host cannot monopolize
  the worker pool
- checkpoint each completed paper from one writer thread
- reject concurrent mutation phases for the same run
- write to `.part`, flush, fsync, validate, and atomically rename
- require a PDF header, EOF marker, minimum size, and SHA-256
- classify HTML, JSON, login, and error responses as invalid content

Keep `failed` and `unavailable` as terminal metadata states. Never delete the paper record because its PDF cannot be downloaded.
