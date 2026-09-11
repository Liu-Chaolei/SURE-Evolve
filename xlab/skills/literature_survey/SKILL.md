---
name: literature_survey
description: Generate a readable, structured, and citation-traceable literature survey from a topic and knowledge graph.
---

# Literature survey

Use this skill to turn a research topic plus a `knowledge_graph` run into durable survey artifacts for downstream research planning. The canonical command is `/xlab write-literature-survey`. The runtime is self-contained in this package under `scripts/`; do not call an external Xcientist checkout or write outside the project/run workspace.

Product path: `/xlab collect-papers` gathers papers, `/xlab build-knowledge-graph` builds the current graph, `/xlab write-literature-survey` produces the survey and citation trace, `/xlab generate-research-ideas` consumes the current survey, and `/xlab check-idea-novelty` checks proposed ideas. Use `/xlab run-research-workflow` for the composed end-to-end path.

## API configuration

SurveyAgent uses the same PyAgent-style LLM API contract as the surrounding XLab Python agents:

- `OPENAI_API_KEY`: canonical required LLM API key, entered through XLab's masked startup dialog when neither it nor `LLM_API_KEY` is set;
- `LLM_API_KEY`: accepted alias for the canonical LLM API key;
- `LLM_BASE_URL`: LLM API base URL, default `https://api.minimaxi.com/v1`;
- `LLM_MODEL`: LLM model, default `MiniMax-M3`;
- `LLM_CONTEXT_WINDOW`: LLM context-window hint, default `512000` tokens;
- `SEMANTIC_SCHOLAR_API_KEY`: canonical required scholarly metadata key, entered through the same masked startup dialog when neither it nor `S2_API_KEY` is set;
- `S2_API_KEY`: accepted alias for the canonical Semantic Scholar key.

Never echo, log, or persist API key values. `state/runtime_config.json` records only whether keys were present, plus non-secret model/base-url defaults needed for resume.

## Normal input

For normal runs, provide the topic and use the current graph handle from the active XLab workspace:

```bash
/xlab write-literature-survey "Retrieval augmented generation for scientific agents"
```

To make the dependency explicit, pass a readable workspace slug and graph handle:

```bash
/xlab write-literature-survey "Retrieval augmented generation for scientific agents" \
  --workspace scientific-discovery-agents \
  --graph graph-v1
```

Run `/xlab list-workspaces` to list workspace slugs and `/xlab show-workspace <slug>` to inspect available `graph-vN`, `survey-vN`, and `idea-vN` handles. The preferred product input is the active workspace graph or an explicit `--graph graph-vN` handle. The XLab harness resolves `graph-vN` to the concrete `knowledge_graph` manifest before this Python runtime starts. Absolute paths remain supported for compatibility: `--graph` may point to a `knowledge_graph` run directory, its `manifest.json`, a direct `artifacts/method_graph.json`, or a direct `artifacts/graph.db`. The survey runtime extracts graph paper records as references, uses bounded Core/edge context for orientation, then runs the integrated Xcientist SurveyAgent outline, paper-assignment, drafting, review, refinement, citation-normalization, and evaluation flow.

## Compatibility input

If a knowledge graph has not been built yet, a paper manifest remains supported as a source compatibility path:

```bash
/xlab write-literature-survey "Retrieval augmented generation for scientific agents" \
  --input .xlab/runs/<paperCollectRun>/artifacts/papers.manifest.json
```

`--input`, `--papers`, and `--paper-set` accept a JSON paper list, a `paper_collect` manifest, or a final manifest that references a `paper_set` artifact. This path is kept for compatibility; ordinary Shanghai Cloud usage should prefer `--graph`.

Topic-only runs do not produce successful real-paper surveys. They can initialize state and emit an incomplete manifest with repair diagnostics, but success requires real papers from `--graph` or `--input`.

## Hidden defaults and advanced flags

Most Survey Agent knobs are internal runtime defaults, snapshotted into `state/runtime_config.json` during `init` so resume uses the same configuration. Real synthesis uses the integrated full-original `deep_survey_fast` SurveyAgent path, including local graph expansion, graph keynotes, relation graph/table context, review/revise loops, refinement, and judge/evaluation defaults. Normal users should not tune these knobs.

The CLI still accepts legacy/advanced flags for compatibility:

- `--max-papers` / `--min-papers`: bounded by internal defaults and limits;
- `--depth`, `--language`, `--facet`: recorded as metadata and surfaced as compatibility warnings when they do not change the upstream evidence set;
- `--full-text`: warns unless the runtime explicitly enables full-text synthesis;
- `--resume`: legacy request metadata; use the `resume` subcommand for real checkpoint recovery.

## Required workflow

Hooks normally run `init` for you. Only the packaged CLI phases below may create or mutate survey artifacts; do not use ad hoc writes, direct provider calls, or external checkout scripts.

For manual execution:

```bash
python xlab/skills/literature_survey/scripts/run_survey_phase.py init \
  --run-id <runId> \
  --run-dir .xlab/runs/<runId> \
  --arguments '"<topic>" --graph .xlab/runs/<knowledgeGraphRun>'
```

Generate or resume the disk-backed pipeline:

```bash
python xlab/skills/literature_survey/scripts/run_survey_phase.py synthesize \
  --run-id <runId> \
  --run-dir .xlab/runs/<runId>
```

If interrupted, rerun the staged resume path. Completed state files with matching request/runtime signatures are reused:

```bash
python xlab/skills/literature_survey/scripts/run_survey_phase.py resume \
  --run-id <runId> \
  --run-dir .xlab/runs/<runId>
```

Audit before finishing:

```bash
python xlab/skills/literature_survey/scripts/run_survey_phase.py audit \
  --run-id <runId> \
  --run-dir .xlab/runs/<runId>
```

Audit exits with `0` for success, `2` for a completed audit whose final manifest is `incomplete`, and `1` for runtime errors that prevented report/manifest generation. Use `xlab_finish` with the status written to `.xlab/runs/<runId>/manifest.json`.

## Resources for native idea generation

Before handing a completed survey to `research_idea`, package its explicit resources through the native phase CLI:

```bash
python xlab/skills/literature_survey/scripts/run_survey_phase.py resources \
  --run-id <runId> --run-dir .xlab/runs/<runId> \
  --model-snapshot <local-all-MiniLM-L6-v2-snapshot>
```

This phase uses the selected graph and saved keynotes, constructs a normalized 384-dimensional FAISS component index, copies the explicit local model snapshot, and declares separate OutcomeRAG/component model identities. It verifies content digests, Survey lineage, native embedding and FAISS inference, graph access, citation parsing, and keynote retrieval before linking `resources/resource_manifest.json` from the Survey manifest. It does not call an LLM or generate ideas. `resources/verification.json` records actual resource consumption and distinguishes extracted keynotes from abstract-only or metadata-only evidence.

Survey completion alone does not imply native idea resource readiness. Keep the bundled Survey files synchronized with the published Survey; changed Survey content requires rebuilding and verifying its resource bundle.

## Run-local state

The pipeline writes intermediate state instead of keeping the whole survey in memory:

- `state/request.json` and `state/runtime_config.json`;
- `state/checkpoint.json` with completed stages, request/runtime signatures, warnings, outputs, and counts;
- `state/papers.jsonl`, `state/references.jsonl`, `state/paper_index.json`;
- `state/graph_context.json`, `state/clusters.json`, `state/chronology.json`;
- `state/survey_agent_outline.json`, `state/survey_agent_assignment.json`, `state/survey_agent_draft.md`, and `state/survey_agent_result.json` when the LLM Survey Agent path runs;
- `state/xcientist/graph.db`, `state/xcientist/config.snapshot.yaml`, `state/xcientist/progress.json`, `state/xcientist/seed_papers.json`, `state/xcientist/expanded_papers.json`, `state/xcientist/collected_papers.json`, `state/xcientist/keynotes.json`, `state/xcientist/clustering_result.json`, `state/xcientist/analysis_context.json`, `state/xcientist/outline.raw.json`, `state/xcientist/draft.raw.json`, `state/xcientist/draft.raw.md`, `state/xcientist/refined.raw.md`, `state/xcientist/references.raw.json`, `state/xcientist/evaluation.json`, and `state/xcientist/engine_result.json` for mid-SurveyAgent resume and inspection;
- `state/sections.jsonl`, `state/key_claims.jsonl`, `state/research_gaps.jsonl`, `state/traces.jsonl`;
- `logs/pipeline.jsonl` and `logs/diagnostics.jsonl`.

## Artifacts

The run must produce:

- `artifacts/survey.md` (`literature_survey_document`, schema version `1`): readable Markdown with `[paper:<id>]` citations;
- `artifacts/survey.json` (`literature_survey_json`, schema version `1`): structured sections, clusters, claims, gaps, references, source metadata, and optional graph context;
- `artifacts/citations.json` (`citation_trace`, schema version `1`): claim-to-paper traceability;
- `artifacts/survey_report.json` (`survey_report`, schema version `1`): audit checks, counts, warnings, blockers, source metadata, checkpoint metadata, and file paths;
- `manifest.json`: XLab final manifest with `schema_version: "2"`, `skill_version`, and typed artifact entries.

## Success gates

A successful survey requires:

- real papers from `--graph` or `--input`, default minimum 3;
- nonempty clusters, sections, key claims, research gaps, references, and traces;
- every cited paper id in sections, claims, gaps, and traces resolves to the reference list;
- readable Markdown with `[paper:<id>]` citations;
- no runtime dependency on external research-agent checkouts or host-specific absolute paths.

If graph enrichment, optional full-text parsing, or Survey Agent provider access is unavailable, preserve completed state, write diagnostics into `survey_report.json`, and finish as `incomplete` when success gates fail. Do not generate or present a secondary summary branch as the intended Survey Agent output.
