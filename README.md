# XLab

XLab is an experimental, durable research-workflow layer built into the [Pi coding agent](https://github.com/badlogic/pi-mono) source tree. Pi remains available as a general coding agent; XLab adds structured research commands, resumable runs, validation gates, artifact lineage, and workspace management through one public command form:

```text
/xlab <verb-noun>
```

The runtime owns orchestration, persistence, tool and artifact gates, recovery, and indexing. Skill packages own task-specific prompts, scripts, schemas, metrics, and validators.

## Run from source

XLab currently targets source-based development and requires Node.js 22.19 or newer.

```bash
npm install --ignore-scripts
./pi-test.sh
```

Then check and configure the local runtime from the TUI:

```text
/xlab check-xlab-setup
/xlab configure-xlab runtime
/xlab show-help
```

Some skills require additional providers, credentials, executables, or scholarly services. `/xlab check-xlab-setup` reports missing requirements without exposing secret values.

## Commands

Research workflows include:

```text
/xlab collect-papers
/xlab build-knowledge-graph
/xlab generate-research-ideas
/xlab check-idea-novelty
/xlab run-research-workflow
/xlab build-research-dataset
/xlab run-experiment
/xlab evaluate-model
/xlab reproduce-paper
```

Run `/xlab show-help` for the installed catalog or `/xlab show-help <task>` for one command's contract. XLab also provides controls for durable state:

```text
/xlab list-workspaces
/xlab show-workspace <slug>
/xlab select-workspace <slug>
/xlab list-runs
/xlab show-run <run-id>
/xlab resume-run <run-id>
/xlab cancel-run <run-id>
/xlab show-artifact <artifact-id>
```

Repository-managed packages live under `xlab/skills/`. A project can provide local packages under `.xlab/skills/`; a local package with the same internal name overrides the repository package. See [XLab skill packages](xlab/skills/README.md) for the package contract and authoring guide.

## Native experiment flow

`/xlab run-experiment` executes an existing successful, structured Research Idea. It does not invent or silently repair an Idea.

```text
/xlab generate-research-ideas <research question>
/xlab run-experiment --idea <idea-reference>
```

The native TypeScript XLab/Pi runtime freezes the selected Idea and advances one durable macro workflow:

```text
prepare → code → science → finalize
```

Planner, worker, reviewer, and final-reviewer sessions submit structured evidence to the parent runtime. The parent alone advances workflow state, accepts results, publishes reports, updates symbolic memory, and indexes final artifacts. Failed reviews reopen the responsible durable worker session; resume does not intentionally repeat accepted work.

The full execution and evidence contract is documented in [`xlab/skills/run_experiment/SKILL.md`](xlab/skills/run_experiment/SKILL.md).

## Runs, workspaces, and artifacts

Local XLab state is stored below `.xlab/` and is ignored by Git. A run has a stable directory such as:

```text
.xlab/runs/<run-id>/
```

Run records, workflow state, events, checkpoints, and manifests are persisted so supported workflows can be inspected, cancelled, or resumed. Successful artifacts are content addressed, indexed with producer occurrences and parent lineage, and exposed through `/xlab show-artifact`.

Skill packages must write typed schema-versioned manifests and satisfy their declared artifact gates. A run may finish as `incomplete` or `failed` when required evidence, validation, publication, or recovery conditions are not met.

## Current limitations

- XLab is source-first and under active development; it is not yet a separately packaged end-user application.
- Availability depends on the installed skill packages and their declared external services, models, data, and credentials.
- Durable orchestration and validation improve traceability, but they do not guarantee that an experiment succeeds or that a scientific conclusion is correct.
- Generated ideas, literature claims, metrics, and final reports still require human scientific review.
- Local `.xlab/` state can contain research inputs and outputs; protect it according to the sensitivity of the project.

## Development

Start with [AGENTS.md](AGENTS.md) and [CONTRIBUTING.md](CONTRIBUTING.md). Extension APIs are documented in [`packages/coding-agent/docs/extensions.md`](packages/coding-agent/docs/extensions.md).

Useful checks include:

```bash
npm run check
npm test
npm run build
```

The root `npm run check` applies Biome formatting before running repository checks, so review its resulting diff. During focused development, run the relevant package test or Vitest file first.

Keep domain-specific behavior in skill packages. The shared runtime should enforce only common lifecycle, isolation, state, workflow, and artifact contracts.

## Pi foundation and license

This repository builds on Pi and retains the original packages under `packages/`. XLab-specific runtime code is under `packages/coding-agent/src/core/xlab/`, with repository skills under `xlab/skills/`.

The repository is distributed under the terms in [LICENSE](LICENSE). Individual integrations or bundled third-party components may have separate licenses and notices; review those terms before redistribution.
