# Provenance and notices

## Xcientist-2 source material

This package-native `research_idea` implementation was derived in part from Xcientist-2 Idea Agent and memory source material, including workflow structure, prompts, MCTS concepts, fusion behavior, retrieval adapters, and memory interfaces.

The audited source is the Xcientist mirror at the following authoritative mirror URL and immutable revision:

- Repository: <https://github.com/KotoHanon/Xcientist-mirror>
- Audited revision: [`a06fec2c3d6361cdcee4850b7447348c75adb47d`](https://github.com/KotoHanon/Xcientist-mirror/tree/a06fec2c3d6361cdcee4850b7447348c75adb47d)
- Upstream source scope used for this derivation: every file recursively under `src/agents/idea_agent/` and `src/memory/` at that revision; no material from `src/harness/` is identified as derived content in this package.
- Content history context: those two source directories are unchanged between the audited revision and their sole introducing/content commit, [`10ac7358adbc69b6b6e79a004a4a19c99c720796`](https://github.com/KotoHanon/Xcientist-mirror/commit/10ac7358adbc69b6b6e79a004a4a19c99c720796). This history observation identifies content context; it does not supply licensing terms.

The corresponding derived XLab implementation scope is `scripts/research_idea_lib/algorithm/`, `scripts/research_idea_lib/resources/`, `scripts/research_idea_lib/research_idea_artifacts.py`, `scripts/research_idea_lib/research_idea_spec.py`, `scripts/research_idea_lib/pipeline.py`, and `scripts/research_idea_lib/survey_repository.py`. These paths are a package-native reconstruction informed by the scoped upstream material; this notice does not claim that every line in them is upstream-derived or that any upstream file was copied verbatim.

Xcientist-2 is source material and attribution, not a source-checkout execution dependency. The package executes its own package-native XLab research-idea workflow identified as `xlab.research_idea.algorithm.v2`; it does not claim source or runtime parity, invoke an Xcientist checkout, or require one to be present. Package-owned does not mean offline or standalone: the declared Survey/resource contract, installed Python environment, local immutable resources, and configured provider remain runtime dependencies. Against audited revision `a06fec2c3d6361cdcee4850b7447348c75adb47d`, algorithm v2 preserves successful normal-path scientific decisions for retrieval scope, prompt selection, adaptive MCTS operator planning, operator grounding, graph/evaluation reuse, component novelty, and fusion scoring. This claim excludes source organization, concurrency, persistence, transport retries, stochastic output identity, and exceptional failure policy.

## Intentional XLab corrections

The package preserves the research workflow while intentionally adding or tightening robustness properties for XLab operation:

- an explicit, Survey-linked resource-manifest contract instead of discovery through neighboring or machine-specific paths;
- content digests, containment checks, schema/algorithm/capability validation, and direct Survey artifact lineage for every required resource;
- immutable, content-verified model snapshots in a run-local, content-addressed XLab model cache with portable logical URIs instead of implicit model lookup or download;
- explicit local-only ML execution: declared immutable sentence-transformer snapshots perform real normalized embedding inference and declared FAISS indexes perform native search, with independent OutcomeRAG/component dimensions and strict model, index, digest, dimension, and vector-count checks; deterministic lightweight backends exist only as explicit test injection;
- package-owned configuration, checkpoints, diagnostics, portable logical resource identifiers, and atomic JSON artifact writes;
- fail-closed completion: any Survey, provider, resource, generation, workflow, provenance, MCTS/fusion, portability, secret-safety, or final-audit blocker yields `incomplete`, never a substitute successful idea;
- provider credentials read only from the runtime environment, with checks against persistence or disclosure in artifacts; and
- bounded and validated provider responses, structural generated-content checks, explicit algorithm provenance, and checkpointed incomplete state resumable through `/xlab resume-run <run-id>`; and
- versioned fusion scoring under `xlab.research_idea.fusion-referee.v2`: the audited default Pro call path's active `moonshot_inventor` taste vector scalarizes ten finite provider metrics (alignment 0.14, complexity 0.06, novelty 0.27, surprise 0.22, impact 0.18, feasibility 0.05, clarity 0.03, conciseness 0.02, risk 0.02, protocol 0.01), with risk and complexity treated as penalties; the provider aggregate remains diagnostic, semantic drafts receive up to five validated attempts, and repairs allow only remove/replace/rewire edits with a maximum of ten steps, patience five, and strict improvement greater than epsilon 0.02.

The production workflow can construct bounded symbolic component-removal records when the `--experiment-feedback` string contains a supported JSON object with `records`, `components`, or nested `ablation_results`, and inject their hints into MCTS. Positive removal results mean removal helped; negative results mean removal hurt. Free-form feedback still reaches re-analysis/replanning without being misrepresented as structured ablation evidence. Vector memory remains disabled under the XLab research-idea workflow profile.

These differences mean this package should not be described as a full copy or a source-equivalent runtime. It is a package-native reconstruction with authentic local transformer and FAISS execution. It also intentionally requires all five canonical taste-mode winners, strict public materialization, and fail-closed finalization instead of successful-subset, raw-candidate, or permissive novelty/fusion fallbacks. The default automated suite remains hermetic: end-to-end tests inject deterministic provider, embedding, and FAISS backends to verify contracts and fail-closed behavior. A separate opt-in smoke (`XLAB_RESEARCH_IDEA_REAL_ML_SMOKE=1`) exercises a locally available immutable MiniLM snapshot and native FAISS, but is not a full live-provider conformance test. The final audit checks artifact completeness, trace/resource linkage, required workflow shapes, portability, and secret safety; it is structural validation, not proof of scientific quality or identical stochastic output.

## License handling

The XLab repository containing this package has a repository-level MIT license, copyright © 2025 Mario Zechner. Its copyright and permission notice must be retained with copies or substantial portions covered by that license; see the repository root `LICENSE`.

No authoritative upstream repository license was located in the audited Xcientist mirror revision. Specifically, revision `a06fec2c3d6361cdcee4850b7447348c75adb47d` has no repository-level `LICENSE`, `NOTICE`, or `COPYING` file and no such file in `src/agents/idea_agent/` or `src/memory/`. It does contain `src/harness/LICENSE`, an MIT license copyright © 2025 OpenHarness Contributors, but that license belongs to the separate OpenHarness subproject and must not be assumed to license the Xcientist-2 Idea Agent or memory materials. No OpenHarness material is identified in the derived XLab scope above.

This provenance file does not grant rights to upstream materials, and the absence of an upstream license must not be interpreted as permission to use, copy, modify, or redistribute them. Redistribution of this package or its derived Xcientist-informed material requires legal review and confirmation of sufficient rights. Retain the XLab MIT license and every license or notice that is determined to apply; if authoritative licensing terms for the Xcientist-2 Idea Agent or memory materials become available, include and retain them rather than inferring permission from the separate OpenHarness license.
