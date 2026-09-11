# research_idea examples

The package-native XLab research-idea workflow is identified as `xlab.research_idea.algorithm.v2` and consumes a completed `/xlab write-literature-survey` run or its `manifest.json`. The selected Survey manifest must declare exactly one explicit, Survey-linked XLab resource manifest containing content-verified Survey, graph, component-index, keynote, and immutable model resources. Verified model snapshots are stored in a run-local, content-addressed XLab model cache and published through portable `xlab-cache://models/<cache-key>` identities; the runtime neither searches for an external checkout nor downloads undeclared resources. Production retrieval performs local-only sentence-transformer inference over Survey passages and native FAISS search over the declared component index, with independent model dimensions and strict model/index identity checks. Algorithm v2 preserves successful normal-path scientific decisions from the audited Xcientist-2 revision while retaining XLab's package-native runtime and stricter completion policy.

Initialize the run before synthesis:

```bash
python scripts/run_idea_phase.py init \
  --run-dir .xlab/runs/research-idea-example \
  --run-id research-idea-example \
  --arguments '--survey .xlab/runs/literature-survey-run --topic "Survey-grounded scientific agents"'

python scripts/run_idea_phase.py synthesize \
  --run-dir .xlab/runs/research-idea-example \
  --run-id research-idea-example
```

For refinement after an experiment, use the only public feedback argument, `--experiment-feedback`, together with the required mature idea:

```bash
python scripts/run_idea_phase.py init \
  --run-dir .xlab/runs/research-idea-refinement \
  --run-id research-idea-refinement \
  --arguments '--survey .xlab/runs/literature-survey-run --mature-idea "Existing idea text" --experiment-feedback "Observed experiment findings"'
```

An optional `--refinement-scope "..."` provides free-text guidance to analysis, search, fusion, and materialization. It is not interpreted as an exact component-name allowlist. Enforceable field, component, and edit-kind limits use the separate internal typed `RefinementBoundary`, which this public option does not expose.

Feedback drives re-analysis and replanning. Ordinary prose remains replanning context only. If the string is a supported JSON object with `records`, `components`, or nested `ablation_results`, recognized component-removal findings also become bounded symbolic MCTS hints. A positive removal result means removal helped; a negative result means removal hurt. Vector memory remains disabled in the XLab research-idea workflow profile.

Fusion uses the package-owned `xlab.research_idea.fusion-referee.v2` ten-metric profile. The provider's aggregate score is retained only as diagnostic metadata; validated metrics determine the authoritative score, with risk and complexity inverted.

Set provider credentials in the runtime environment, never in `--arguments` or another command-line option.

Completion is fail-closed. `success` is emitted only when the final structural audit finds the required non-placeholder artifact content, evidence/resource linkage, expected workflow/provider traces, recorded MCTS/fusion structure, portable paths, and no detected secret leakage. It does not prove scientific quality or identical stochastic output. Once a package phase starts, missing provider access, an invalid or absent resource manifest, incompatible resources, generation failure, or any audit blocker produces `status: "incomplete"`, records blockers and an incomplete reason, preserves checkpointed state, and returns exit code 2; no lightweight or unaudited idea is promoted as success. The `/xlab` harness may instead stop before package artifacts exist if required provider-secret setup is cancelled or unavailable.

Resume an incomplete harness-managed run with `/xlab resume-run <run-id>`. Do not add `--resume` to `/xlab generate-research-ideas`; that parsed field is unused compatibility metadata and does not initiate resume.

The skill writes `artifacts/idea_result.json`, `artifacts/research_idea.json`, `artifacts/idea_trace.json`, `artifacts/idea_report.json`, and `manifest.json`.
