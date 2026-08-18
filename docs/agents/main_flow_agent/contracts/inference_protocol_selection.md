# Inference Protocol Selection v2

## Enumeration And Default

The only protocol IDs are `standard_system` and `strict_core`.
`standard_system` is the default. `strict_core` is valid only after an explicit
user selection in `MAIN_FLOW_INPUT.inference.protocol_id`.

Legacy `strict_one`, model names, tasks, metrics, pipeline IDs, and run IDs are
invalid protocol identifiers. Invalid input stops before inference.

## `standard_system`

This protocol means “run the official pinned inference flow unchanged.” The
model adapter must record:

- official repository URL and commit
- weights revision
- official inference entrypoint
- `invocation_overrides: {}`
- effective official parameters for audit

The main-flow layer injects no decoding parameters. Any user override, even one
that happens to equal an official value, violates this protocol.

## `strict_core`

This protocol constrains the semantic controls in `config/protocols.yaml`,
including greedy/sampling behavior, temperature, one beam, one hypothesis,
batch size one, and disabled external LM, hotwords, retrieval, and multi-pass
selection.

Every control requires one of:

- `parameter`: concrete model parameter and enforced value
- `attestation`: named evidence set to `true`
- `not_applicable`: non-empty model-specific reason

An incomplete map blocks. Temperature may be `not_applicable` only when the API
does not accept temperature while sampling is disabled; the reason must say so.

## Comparable Identity

Protocol equality requires all three fields:

```json
{
  "id": "standard_system",
  "version": "2.0",
  "effective_params_sha256": "<64 lowercase hex>"
}
```

The selected ID alone is insufficient for result reuse.
