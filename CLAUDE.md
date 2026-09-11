# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Before editing

Read `AGENTS.md` and `CONTRIBUTING.md`. `AGENTS.md` is the authoritative source for code style, test safety, dependency handling, shared-worktree Git rules, and release procedures.

This is an npm-workspace ESM monorepo requiring Node.js `>=22.19.0`. A local Conda environment named `xlab` provides CI-aligned Node 22, Python 3.12, and native utilities:

```bash
conda activate xlab
```

Run repository commands from the root. Direct dependencies are exactly pinned; install without lifecycle scripts:

```bash
npm install --ignore-scripts       # hydrate/update the existing tree
npm ci --ignore-scripts            # clean, lockfile-reproducible install
./pi-test.sh                        # run the TypeScript source CLI
./pi-test.sh --help                 # startup smoke test without a model call
```

## Checks, tests, and builds

```bash
npm run check                       # format/lint, repository validations, typecheck, browser smoke
./test.sh                           # credential-isolated workspace tests
npm run build                       # build packages in dependency order
npm test                            # all workspace tests, including enabled e2e tests
```

`npm run check` starts with `biome check --write`, so it can modify files; inspect its diff. It does not run tests. Follow `AGENTS.md`: after non-documentation code changes run `npm run check`; do not run `npm run build`, `npm test`, or a full Vitest suite unless the user requests it. Use `./test.sh` for the complete non-e2e TypeScript test run; it temporarily hides Pi auth and clears provider credentials.

Run a focused Vitest file from the owning package root:

```bash
cd packages/coding-agent
node ../../node_modules/vitest/dist/cli.js --run test/suite/xlab-experiment-protocol.test.ts
```

The same pattern applies in `packages/ai` and `packages/agent`. TUI tests use Node's runner:

```bash
cd packages/tui
node --test test/<name>.test.ts
```

Some XLab skills also have standalone Python `unittest` suites not included by root `npm test`:

```bash
python xlab/skills/paper_collect/tests/test_paper_collect.py
python xlab/skills/knowledge_graph/tests/test_pipeline.py
```

Coding-agent tests under `packages/coding-agent/test/suite/` must use `test/suite/harness.ts` and faux providers; never use real credentials, network APIs, or paid tokens. Put issue regressions in `test/suite/regressions/`.

CI performs `npm ci --ignore-scripts`, `npm run build`, `npm run check`, and `npm test`. Before a PR, the documented local gate is `npm run check && ./test.sh`.

## Monorepo architecture

The package dependency flow is:

```text
pi-ai ──→ pi-agent-core ──→ pi-coding-agent ──→ pi-orchestrator
pi-tui ───────────────────→ pi-coding-agent
```

- `packages/tui`: terminal abstraction, differential renderer, components, input, overlays, and configurable keybindings.
- `packages/ai`: provider/model catalog, authentication/OAuth, streaming implementations, and unified LLM APIs.
- `packages/agent`: stateful `Agent`, message types, the provider-facing agent loop, tool execution, and steering/follow-up queues.
- `packages/coding-agent`: the `pi` CLI and application layer: sessions, settings/trust, tools, extensions, skills, packages, resources, TUI/print/JSON/RPC modes, and XLab.
- `packages/orchestrator`: experimental orchestration on top of coding-agent; its API is unstable.

The root build order is `tui → ai → agent → coding-agent → orchestrator`. TypeScript path aliases resolve workspace imports directly to source during development.

The CLI enters through `packages/coding-agent/src/cli.ts` and `main.ts`. Startup resolves the mode, cwd/session, project trust, settings, packages/resources, model registry, and `AgentSession`, then dispatches interactive, text/JSON, or RPC mode. `AgentSessionRuntime` owns cwd-bound services; session replacement or cwd changes rebuild settings, packages, extensions, models, and session services, so previously captured extension contexts are stale.

At the lower layer, the agent loop converts internal messages to provider messages only at the LLM boundary, streams lifecycle events, executes tool calls, applies steering before the next assistant turn, and processes follow-ups after the agent would otherwise stop. Sibling tool calls may execute concurrently.

## Settings, trust, extensions, and skills

`SettingsManager` merges global settings with project `.pi/settings.json`; project settings and executable project resources load only after trust. `DefaultResourceLoader` coordinates settings, package discovery, extensions/providers, Agent Skills, prompt templates, themes, and context files. Preserve the bootstrap trust pass—do not load project-local executable resources before trust is resolved.

Pi packages may declare `extensions`, `skills`, `prompts`, and `themes`. Extensions are TypeScript/JavaScript factories loaded with the user's process permissions and can register tools, commands, flags, shortcuts, providers, renderers, and lifecycle handlers. Agent Skills are Markdown instruction resources (normally `SKILL.md`); they are separate from XLab skill packages.

Custom mutating tools must use coding-agent's file-mutation coordination because sibling tools can run in parallel. Keybindings must remain configurable through `DEFAULT_EDITOR_KEYBINDINGS` or `DEFAULT_APP_KEYBINDINGS`, not hardcoded key checks.

## XLab architecture

XLab is the built-in durable research layer inside coding-agent, not a separate application. The shared TypeScript control plane is under `packages/coding-agent/src/core/xlab/`; repository-owned domain packages are under `xlab/skills/`, with project overrides under `.xlab/skills/`.

The boundary is intentional:

- Shared TypeScript runtime: command dispatch, trust and environment preparation, runs/workflows, child sessions and assignments, persistence/recovery/cancellation, validation, artifacts, lineage, and publication.
- Skill packages: domain prompts, schemas, scripts, metrics, validators, permissions, hooks, workflow DAGs, and artifact declarations.

Do not add a second Python or external experiment control plane. An XLab package is independently reviewable and described by schema-v2 `xlab.skill.json`; exact-version dependencies and package-contained paths enforce reproducibility. Python packages with substantive lockfiles receive content-keyed virtual environments under `.xlab/environments/<package>/<digest>`; the activated Conda Python is the bootstrap interpreter.

`/xlab <verb-noun>` discovers packages and creates durable state under `.xlab/runs/<run-id>/`. Workflows use validated `xlab_stage` transitions and explicit `xlab_finish`; reaching `agent_end` alone does not finish a run. Successful artifacts are content-addressed under `.xlab/artifacts/objects/sha256` and indexed with producer occurrences and parent lineage.

`/xlab run-experiment` is a native runtime with fixed macro stages `prepare → code → science → finalize`. It owns isolated planner/worker/reviewer sessions, durable protocol state, fencing, recovery, repair, cancellation, and parent-owned finalization; do not drive it with generic `xlab_stage`/`xlab_finish`. Its authoritative scientific/evidence contract is `xlab/skills/run_experiment/SKILL.md`.

Useful source-start commands are:

```text
/xlab check-xlab-setup
/xlab configure-xlab runtime
/xlab show-help
```

`.xlab/` is ignored durable state and may contain sensitive research inputs, credentials-related metadata, outputs, journals, and artifacts. Never commit it.

## Repository-specific implementation constraints

- TypeScript is strict ESM with Node16 resolution, explicit relative `.ts` imports, and `erasableSyntaxOnly`; avoid enums, namespaces, parameter properties, `import =`, and other syntax requiring TypeScript emit.
- Use top-level imports only; no dynamic or inline type imports. Avoid `any`.
- Never edit `packages/ai/src/models.generated.ts` directly; change its generator and regenerate.
- Keep Pi core minimal. Features that do not belong in core should normally be extensions.
- Normal contributor changes do not edit package changelogs.
- Multiple sessions may share the working tree. Follow the explicit staging and non-destructive Git rules in `AGENTS.md`, and commit only when requested.
