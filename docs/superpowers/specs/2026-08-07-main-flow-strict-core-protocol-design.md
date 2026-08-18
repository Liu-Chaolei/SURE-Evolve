# Main-Flow Strict-Core Protocol Design

## Problem

SURE-EVAL defines exactly two inference protocols in `config/protocols.yaml`:
`strict_core` and `standard_system`. The main-flow agent documentation does not
currently define an authoritative protocol enumeration or selection algorithm.
Its shell templates default to `strict_core`, but accept any externally supplied
`PROTOCOL_ID` and propagate it into result paths, `protocol.yaml`, and report
artifacts. Most input examples omit protocol selection entirely. These gaps let
an agent invent task-, metric-, or run-specific protocol identifiers, especially
when materializing segmented TTS evaluation surfaces.

## Required Invariants

1. The only globally defined protocol identifiers are `strict_core` and
   `standard_system`.
2. Every inference run orchestrated by `docs/agents/main_flow_agent` uses
   `strict_core`. The agent does not choose `standard_system`, even though it
   remains a valid global protocol for workflows outside this main-flow policy.
3. The main-flow agent never derives a protocol identifier from a model, task,
   dataset, language, metric list, pipeline, segment, run ID, or output path.
4. TTS/VC evaluation-only and segmented evaluation inherit `strict_core` from
   the inference run. They do not create or rename protocols.
5. `protocol_id`, `PROTOCOL_ID`, the results-directory suffix, `protocol.yaml`,
   `report.jsonl`, and report snapshots must all agree on `strict_core`.
6. Supplying any other value, including `standard_system`, to a main-flow shell
   is a contract error and stops before preparation, inference, or evaluation.

## Contract Structure

Add one focused protocol-selection contract under
`docs/agents/main_flow_agent/contracts/`. It is the main-flow source of truth
for the global enumeration, the main-flow selection rule, propagation, failure
behavior, and audit evidence.

Reference that contract from:

- the non-bypassable system constraints in `AGENTS.md`;
- the architecture, execution-surface, shell, TTS/VC, and run-layout contracts;
- `README.md` and the example input documentation.

The main-flow input shape explicitly carries `harness.protocol_id: strict_core`
as a fixed field. It is documentation and audit evidence, not a user-extensible
namespace.

## Defense In Depth

### Agent Decision Layer

The system constraint declares the two global IDs and pins main-flow inference
to `strict_core`. Planning, script-routing, execution-surface, assessment, and
run-report artifacts must use that exact value. An absent value resolves to
`strict_core`; a different value is rejected rather than normalized or silently
rewritten.

### Materialization Layer

`main_agent_execution_surface.json` records:

```json
{
  "protocol": {
    "allowed_protocol_ids": ["strict_core", "standard_system"],
    "selected_protocol_id": "strict_core",
    "selection_policy": "main_flow_strict_minimum_inference",
    "allow_override": false
  }
}
```

Script-routing inputs also use `protocol_id: strict_core`. Result publication
paths are always `<output_root>/<model_name>/strict_core/` for main-flow runs.

### Runtime Layer

All three executable templates validate `PROTOCOL_ID` before doing work:

- unset or empty becomes `strict_core`;
- `strict_core` continues;
- every other value exits with a clear error explaining that main-flow inference
  is pinned to `strict_core` and that custom protocol IDs are forbidden.

The check is duplicated intentionally because materialized execution surfaces
must remain self-contained and may not source another template.

### TTS/VC Segmentation

Audio evaluation handoffs, metric-family segments, retries, payload merging,
and results mirrors preserve the original `strict_core`. Segment names and
metric slugs belong in segment directories and metric artifact paths, never in
`protocol_id`.

## Regression Coverage

Add a repository test that audits the main-flow documentation surface:

- the system constraint and protocol-selection contract state the two global
  IDs and the main-flow `strict_core` invariant;
- all executable templates reject non-`strict_core` values before work begins;
- execution-surface and script-routing templates materialize `strict_core`;
- every `input_*.md` example explicitly declares
  `harness.protocol_id: strict_core`;
- no example or template declares a third literal protocol identifier.

Run shell syntax checks for all executable templates and the focused protocol
policy test. Finally, search the complete `docs/agents/main_flow_agent` tree to
confirm every protocol-selection statement is consistent with this contract.

## Scope

This change updates main-flow documentation, contracts, templates, and examples,
plus focused regression coverage. It does not add or remove global protocols,
change `config/protocols.yaml`, reinterpret existing historical run artifacts,
or alter metric/pipeline identifiers.
