---
name: run-experiment
description: Execute a canonical Research Idea with the native durable XLab experiment runtime.
---

# Native Experiment workflow

`/xlab run-experiment` executes an existing successful Research Idea through the native XLab/Pi control plane. The command never invents or repairs an Idea. `--idea` must resolve exactly to an existing structured Idea by file path, artifact ID, producer run ID, workspace `idea-vN` handle, current workspace Idea, or case-sensitive indexed title. Missing, ambiguous, malformed, incomplete, corrupted, or legacy inputs fail before a run is created.

The resolved semantic input is frozen as `inputs/idea.json` with its binding metadata. The standard profile uses one macro workflow in this exact order:

1. `prepare`: resolve and validate repositories, dataset, model, environment, and synthesis inputs.
2. `code`: implement the canonical components and finish with one real terminal `final_integration_smoke`.
3. `science`: execute one all-components full run and exactly one full component-disabled run for every canonical Idea component.
4. `finalize`: perform a blocking final review, materialize ablation and audit artifacts, update symbolic memory with compare-and-swap, and publish the final manifest.

`workflow.json` is the only macro-stage state machine. The native experiment journal and current projection own the stage-internal execution graph. Do not call generic `xlab_stage` or `xlab_finish` tools for this skill.

## Direct formal ASR profile

`/xlab run-experiment --profile <asr-profile.json> --idea <reference>` selects the native ASR execution profile.
`/xlab run-experiment --baseline-profile <asr-profile.json>` creates an explicit original-model baseline instead; it is mutually exclusive with `--idea` and does not fabricate a Research Idea.
The baseline has zero scientific components, one all-components reference condition, and an explicit `ablation_status: not_applicable` result.

The ASR profile preserves prepare/code/science/finalize and all durable child/reviewer/lineage contracts, with these explicit policy changes:

- Code ends in `final_static_integration`, and its matrix uses `static_integration` instead of `integration_smoke`.
- No separate model smoke, pilot, benchmark or API probe is scheduled. Runtime integration evidence is deferred to the real full science conditions; an omitted smoke is never reported as passed.
- Workers execute actual bound work with `experiment_execute`; the native executor submits exclusive four-NPU Slurm jobs, captures training and SURE scoring evidence, and reattaches to recorded jobs after interruption.
- Science work units with satisfied dependencies execute concurrently without an application-level count cap. Slurm resources and policy determine actual concurrency.
- Every canonical component gets a full disabled condition, with no fixed component-count or ablation-count limit.
- The original-model baseline and every trained scientific condition use the same frozen data, seed, epoch and per-device duration budgets.

The default smoke requirements below apply to the standard profile. Direct formal success instead requires the accepted full science cohort and its runtime evidence.

## Durable child execution

The parent runtime creates isolated durable planner, worker, reviewer, and final-reviewer sessions. Children submit structured candidates only; they cannot modify the parent workflow, protocol, reviewer aggregate, artifact index, final manifest, or symbolic memory.

Each logical work unit keeps one stable worker session across retries and review repairs. A blocking review failure reopens that exact session and sends the structured issues directly to it. Repair increments the assignment attempt, generation, and review round, then reruns the complete reviewer matrix from its first role.

Every code work unit is reviewed serially in this order:

`idea_alignment`, `implementation_correctness`, `scientific_invariants`, `protocol_semantics`, `integration_smoke`, `reproducibility`, `code_cleanliness`.

Every science condition is reviewed serially in this order:

`protocol_compliance`, `protocol_semantics`, `condition_toggle`, `evidence_plausibility`, `statistical_interpretation`, `idea_alignment`.

Each reviewer role and round uses a fresh read-only child session. The parent publishes each accepted report as byte-identical `attempts/<NNN>.json` and `latest.json`, then creates the ordered aggregate with report paths and SHA-256 lineage.

For component-disabled science conditions, only `statistical_interpretation.structured_findings.component_results` is authoritative for component conclusions. Finalization reloads and validates the accepted matrix; neither another reviewer nor the final reviewer may override those results.

## Execution contracts

Prepare work units are fixed and ordered as `repos`, `dataset`, `model`, `env`, and `synthesis`. Evidence must bind real local resources and provenance. Missing required evidence produces a truthful incomplete result; it is never replaced with a mock, quick, pilot, or dry-run shortcut.

The code plan contains exactly one terminal `final_integration_smoke`. Its evidence must execute the integrated path and a component-disabled path using concrete prepared inputs, bounded commands, real outputs, and finite metrics. Import-only checks, synthetic/random-only inputs, mocks, dry runs, and timeout-as-success are invalid.

The science plan contains exactly one all-components reference condition followed by one disabled full-run condition for every canonical component, with no extra formal conditions. Each condition records commands, resource bindings, raw outputs, logs, finite metrics, and lineage under the managed run scope.

## Recovery and cancellation

Planner and worker session mappings are durable. Restart and resume reopen the recorded session file and deterministically continue the unique next action without repeating accepted work. Identical result replay is idempotent; conflicting, superseded, stale, late, or post-cancellation submissions fail closed.

Cancellation first persists the cancellation intent and terminal generation, then aborts live children. A user cancellation is `cancelled`; a terminal runtime error is `failed`; missing evidence, lineage, finalization, or symbolic-memory writeback is `incomplete`.

## Parent-owned finalization

The parent deterministically publishes plans, worker results, reviewer matrices, ablation results, materialization report, final audit, experiment report, symbolic-memory receipt, and final manifest from accepted native submissions. The symbolic-memory receipt is emitted only after a real compare-and-swap write and digest verification; conflicts cannot be reported as success.

Do not invoke a Python experiment agent, command hook scheduler, OpenHarness, an external Xcientist checkout, or an obsolete experiment workspace reader. The native protocol and XLab/Pi child sessions are the only supported runtime.
