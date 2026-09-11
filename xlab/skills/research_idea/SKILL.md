---
name: research-idea
description: Generate structured, evidence-linked research ideas and experiment-plan fields from a literature survey.
---

# Research idea

Use `/xlab generate-research-ideas` after `/xlab write-literature-survey` has produced a validated Survey artifact and its explicit resource manifest. This skill executes a package-native XLab research-idea workflow identified as `xlab.research_idea.algorithm.v2`, independent of any external Xcientist checkout: Survey-grounded retrieval, structured analysis, MCTS candidate search, multi-mode fusion, optional experiment-feedback replanning, and final idea materialization. It still requires the declared `literature_survey` artifact/resource contract, installed Python dependencies, local model/index snapshots, and configured OpenAI-compatible provider access.

The implementation uses Xcientist-2 as attributed source material without claiming source or runtime parity. Algorithm v2 preserves the audited revision's successful normal-path scientific decisions while retaining XLab-owned execution, reliability, provenance, and fail-closed policies. Execution is entirely package-native and neither invokes nor depends on an external Xcientist checkout. See `PROVENANCE.md`.

## User interface

Provide a Survey handle or path plus optional textual constraints:

```text
/xlab generate-research-ideas --workspace scientific-discovery-agents \
  --survey survey-v1 \
  [--topic "..."] \
  [--mature-idea "..."] \
  [--refinement-scope "..."] \
  [--discussion "..."] \
  [--experiment-feedback "..."]
```

If a workspace is active, users may omit `--workspace`; if the workspace has a current Survey, users may also omit `--survey`:

```text
/xlab generate-research-ideas --topic "Retrieval augmented generation for scientific agents"
```

- `--workspace` is optional and selects a readable XLab workspace slug. Use `/xlab list-workspaces`, `/xlab show-workspace <slug>`, and `/xlab select-workspace <slug>` to inspect or set it.
- `--survey` accepts a workspace handle such as `survey-v1`, a `/xlab write-literature-survey` run directory, or its `manifest.json`. The XLab harness resolves handles before the Python runtime starts.
- A successful run requires the selected Survey manifest to declare exactly one Survey-linked XLab resource manifest. That content-addressed bundle must declare the Survey, paper graph, component index, keynotes, and immutable model snapshots with compatible schemas, algorithms, capabilities, dimensions, digests, and Survey artifact lineage.
- Model snapshots are copied only from that verified bundle into a run-local, content-addressed XLab model cache and represented in portable artifacts by `xlab-cache://models/<cache-key>` URIs. OutcomeRAG loads the declared local `sentence-transformers/all-MiniLM-L6-v2` snapshot with remote downloads and remote code disabled, embeds Survey Markdown at paragraph level, and returns context bounded to each paragraph's Markdown section with citations resolved through `citations.json`. Component retrieval loads the separately declared component embedding model and executes native FAISS search against the declared `faiss.index`, validating runtime, model, index, and metadata dimensions/counts before returning results. The OutcomeRAG dimension is independent of the component index dimension. Missing ML dependencies, malformed runtime resources, undeclared model identities, or incompatible dimensions fail closed. The runtime does not search neighboring directories or download an undeclared model.
- Keynote grounding is citation-driven: only papers cited by selected OutcomeRAG hits are eligible. The package resolves their validated keynote records, scores each eligible keynote once, orders them deterministically, compresses the top five individually, and rolls lower-ranked keynotes into one cited capsule. Provider failures or malformed keynote outputs fail closed; the resource layer itself performs no hidden provider calls.
- `--topic` is optional; the runtime infers it from the Survey when omitted.
- `--mature-idea` anchors contract-mode refinement. It can be used without `--refinement-scope`.
- `--refinement-scope` supplies optional free-text guidance for analysis, search, fusion, and materialization and is most useful with `--mature-idea`. It is never parsed as an exact component-name allowlist. Enforceable field, component, and edit-kind limits use the separate internal typed `RefinementBoundary`, which is not exposed by this public option.
- `--discussion` adds scientific discussion context. It does not select the experiment-feedback replanning path.
- `--experiment-feedback` is the only public feedback input and requires `--mature-idea`. It supplies experiment findings and selects the re-analysis/replanning path; an optional refinement scope further constrains refinement. Ordinary prose remains replanning context only. If the string contains a supported JSON object with `records`, `components`, or nested `ablation_results`, recognized component-removal findings also become bounded symbolic MCTS hints (positive removal result means removal helped; negative means removal hurt). Vector memory is disabled in the XLab research-idea workflow memory profile.

Do not ask users for MCTS, fusion, model-routing, memory, retrieval-threshold, or retry settings. Those are internal runtime defaults and are snapshotted into run-local state. Fusion is evaluated under `xlab.research_idea.fusion-referee.v2` using the audited default Pro path's active `moonshot_inventor` taste weights over ten validated metrics; the provider's aggregate score is diagnostic only, and risk/complexity are inverted during deterministic scalarization. Semantic fusion drafting allows up to five validated attempts. Local fusion repair is restricted to remove/replace/rewire operations, at most ten steps with patience five, and accepts only scores strictly greater than the current best plus 0.02. Never place API keys or other secrets in command arguments, Survey metadata, or artifacts; configure provider credentials through the runtime environment.

## Runtime contract

Run the packaged phase CLI only:

```text
python scripts/run_idea_phase.py init --arguments "$XLAB_ARGS" --run-dir "$RUN_DIR" --run-id "$RUN_ID"
python scripts/run_idea_phase.py synthesize --run-dir "$RUN_DIR" --run-id "$RUN_ID"
python scripts/run_idea_phase.py resume --run-dir "$RUN_DIR" --run-id "$RUN_ID"
python scripts/run_idea_phase.py audit --run-dir "$RUN_DIR" --run-id "$RUN_ID"
```

These are package-internal phase commands. Users resume an incomplete XLab run with `/xlab resume-run <run-id>`; the harness restores the recorded run, and the package resume phase reuses valid checkpoints. Do not pass `--resume` to `/xlab generate-research-ideas`: the parser retains it only as unused compatibility metadata, and it does not initiate resume.

Do not call an external Xcientist checkout, search for undeclared resources, invoke direct MCP search tools, or issue provider HTTP commands outside the package runtime. The package must produce artifacts, checkpoints, diagnostics, and the final manifest through `scripts/run_idea_phase.py` alone.

### Fail-closed completion

`success` means the final structural audit found the required artifact fields and non-placeholder generated content, resolved evidence/resource identities, expected workflow and provider traces, recorded MCTS/fusion structure, portable paths, and no detected secret leakage. It verifies those recorded contracts and trace shapes; it does not prove scientific quality or identical stochastic output. A requested `success` status cannot override a failed audit.

Once a package phase has started, Survey validation, provider readiness, resource validation, generation, or any final audit check failure makes the run exactly `incomplete`. For `/xlab generate-research-ideas`, the harness first resolves required runtime prerequisites and may stop before package artifacts exist if provider-secret setup is cancelled or unavailable:

- preserve completed checkpoints and diagnostics for resume;
- write `status: "incomplete"`, `blockers`, and `incomplete_reason` into the public artifacts and final report;
- leave required generated idea fields empty rather than fabricating provider output;
- write `manifest.json` with final status `incomplete`; and
- return exit code 2 from `synthesize`, `resume`, or `audit`.

Never substitute a lightweight, single-mode, raw-candidate, or otherwise unaudited idea and present it as a successful result. An explicit `incomplete` finalization request may downgrade an otherwise passing run, but no request may upgrade a failing run.

## Outputs

The run writes:

- `artifacts/idea_result.json` — public XLab research-idea workflow-shaped idea result with algorithm provenance.
- `artifacts/research_idea.json` — XLab downstream artifact for `/xlab check-idea-novelty` and experiment planning.
- `artifacts/idea_trace.json` — workflow, operation, MCTS, source, resource, and fusion provenance.
- `artifacts/idea_report.json` — final audit checks and success/incomplete blockers.
- `manifest.json` — XLab manifest v2 with final status and artifact paths.

A successful `research_idea.json` includes a research question, hypothesis, method, expected contribution, experiment plan, data requirements, baselines, metrics, risks, source evidence tied to the Survey, and provenance identifying the package-native XLab research-idea workflow as `xlab.research_idea.algorithm.v2`. That identifier records the selected internal scientific contract; it does not assert source/runtime identity or identical stochastic output.
