# XLab skill packages

Every directory under `xlab/skills` is an independently reviewable XLab package. Runtime discovery also checks project-local `.xlab/skills`; a project package overrides a repository package with the same internal package name.

## Package contract

```text
xlab/skills/<package>/
  xlab.skill.json
  SKILL.md
  hooks/
  scripts/
  schemas/
  requirements.lock | package-lock.json
  examples/
```

`xlab.skill.json` uses schema version 2 and declares:

- an exact package version and `public`, `internal`, or `service-control` visibility;
- one runtime kind, entrypoint, lockfile, executable requirements, and secret names;
- tool, network, subprocess, and external-mutation permissions;
- exact-version dependencies and an optional acyclic stage DAG;
- input/output schemas, typed artifact requirements, hooks, and UI hints;
- optional `ui.argumentHint` and declarative `ui.arguments` entries for slash-command usage and parameter completion.

`ui.arguments` describes the user-facing slash contract, not package-internal phase, audit, output, or run-state CLI options. Long flags use names such as `--depth`; assignment-style packages may use names such as `language`, which completes as `language=`. Free-form workflow commands should declare a concise `argumentHint` without inventing stage flags. Every XLab skill command also receives the harness-level `--workspace <slug>` option automatically.

Public and service-control packages participate in the curated `/xlab <verb-noun>` product catalog. Manifests retain only stable internal package identities; internal packages are reusable workflow stages and do not expose commands.

Use `/xlab check-xlab-setup` before scientific workflows, then `/xlab configure-xlab <runtime|knowledge-graph|scholarly-services|all>` to initialize missing values with masked prompts. Runtime model configuration is persisted in Pi config; integration environment values remain process-local and should be persisted through a shell or secrets manager when needed. Active XLab runs show a compact Pi Agent TUI dashboard with phase, workflow, artifacts, blockers, and resume hints.

## Self-contained rule

A package may consume another package only through a declared exact-version dependency and its artifact or service contract. It must not:

- import scripts from another source checkout;
- contain host-specific absolute paths;
- use symlinks that escape its directory;
- refer to an undeclared local editable dependency;
- require the original source repositories at runtime.

Code adapted from an existing project must be copied and reduced into the owning package. Keep package-specific logic, schemas, repair behavior, and smoke fixtures beside that code.

## Runtime and artifacts

Python packages with a lockfile receive a content-keyed virtual environment under `.xlab/environments/<package>/<digest>`. Workflow stage startup returns the interpreter path for the resolved dependency package.

All final manifests use schema version 2 and contain:

- `run_id`, `skill_name`, `skill_version`, status, timestamps, inputs, outputs, and validation;
- typed artifact entries with `type`, `schema_version`, `path`, optional parents, and metadata.

Successful artifacts are committed to `.xlab/artifacts/objects/sha256` and indexed in SQLite. The payload is immutable and content addressed; producer occurrences and parent lineage are retained separately. Failed runs may finish with partial or no artifacts, while successful runs must satisfy all required artifact gates.

## Workflow lifecycle

Composite packages use `xlab_stage` at stage boundaries. The runtime enforces prerequisites, maximum attempts, terminal transitions, durable stage state, and successful completion of every non-optional stage. `/xlab resume-run <run-id>` restores the locked environment and checkpoint without repeating successful stages.

Hooks may be TypeScript modules or package-local commands. Command hooks receive one JSON context object on stdin and return one JSON result on stdout. Use:

- `pre_start` for input and environment gates;
- `pre_stage` / `post_stage` for stage contracts;
- `pre_finish` for successful or incomplete artifact validation;
- `post_finish`, `on_error`, `on_resume`, and `on_cancel` for terminal state and recovery.

## Current capability split

- Evidence: `paper_search`, `paper_fetch`, `paper_parse`, `paper_collect`, `knowledge_graph`.
- Ideation: `research_idea`, `novelty_check`, `research_workflow`.
- Data: `data_discover`, `data_collect`, `data_process`, `data_validate`, `data_workflow`.
- Experiments: `run_experiment` uses the native durable XLab/Pi runtime for canonical Research Idea execution, review, recovery, and finalization.
- Evaluation: `eval_prepare`, `eval_run`, `eval_compare`, `model_eval`.
- Tooling: `model_wrap`, `mcp_validate`, `mcp_register`, `model_mcp`.
- Collaboration: `scholar_profile`, `science_gateway`, `research_discussion`.
- Phase two: `paper_reproduction`, gated to papers with explicit official code, data, target metrics, and compute budget.
