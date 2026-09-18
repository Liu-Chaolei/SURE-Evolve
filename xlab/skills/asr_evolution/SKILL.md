---
name: asr-evolution
description: Run six durable ASR research rounds with native experiments, full component ablations, and SURE scoring.
---

# Native ASR evolution

Use `/xlab run-evolution --config <profile.json> [--workspace <slug>]`.
The parent owns a fresh baseline, six sequential research rounds, and final selection/test evaluation.
Each round invokes the packaged research_idea phases and native run_experiment lifecycle.
Every canonical component has one full disabled condition; component and Slurm concurrency counts are uncapped.
Slurm decides actual simultaneous resource allocation. Each condition exclusively requests four NPUs.

This profile is direct_formal: no separate smoke, pilot, benchmark or API probe is executed.
Static integration checks precede full science; runtime evidence must come from the real full conditions.
All model roles use XI gpt-6-astra with credentials resolved privately from the profile env file.
SURE scoring uses frozen reference files and the declared English WER pipeline.

Use `/xlab resume-run <run-id>` after interruption and `/xlab cancel-run <run-id>` to cancel the parent and its recorded jobs.
Never launch SURE_master, synthesize experimental results, or substitute partial-epoch runs for full evidence.
Do not use generic xlab_stage or xlab_finish for this native workflow.
