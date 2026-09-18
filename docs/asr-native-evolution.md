# Native TEDLIUM3 evolution

The `asr_evolution` control plane calls package-native `research_idea` and the native `run_experiment` lifecycle. It does not launch or import SURE_master. The independent `sure-eval` package supplies frozen English WER scoring.

```text
/xlab run-evolution --config xlab/configs/asr_tedlium3_evolution.json
/xlab show-run <run-id>
/xlab resume-run <run-id>
/xlab cancel-run <run-id>
```

The controller is detached from the interactive session. Closing the UI does not cancel multi-hour training. Explicit cancellation records intent, cancels only jobs carrying this run's ownership tag, and rejects subsequent submissions. Logs are in the run's `logs/controller.log`; durable parent state is `evolution.json`. Experiment child sessions, assignments and reviews remain in the native experiment journal.

The checked-in local profile starts a new full TEDLIUM3 Zipformer baseline, then six rounds with one idea each. The baseline explicitly disables MUSAN because the additional noise corpus is outside the TEDLIUM-only data contract; its frozen `baseline_configuration` also declares greedy decoding. Its zero-component code phase verifies the exact prepared source and configuration. Each idea is evaluated with one all-components full run and one full disabled run for every canonical component. Counts are uncapped; all independent science conditions are submitted to Slurm. Each job requests four exclusive Ascend NPUs, 80 CPUs, 500G memory and 1T temporary storage, with a 72-hour limit. Every condition trains from scratch for 30 epochs at duration 900, FP32, seed 42. A resumed job restores only a complete checkpoint from the identical source and training binding.

Model architecture, training methods and decoding may change within the frozen data and compute budget. `project/method.py` returns train/decode arguments and the decoding method. `component_enabled(name)` exposes the immutable disabled-component list to recipe code. Reviews must verify a meaningful disabled path for every component. Frozen execution code owns data splits, checkpoint completion and evaluation; workers cannot edit parent-owned results or SURE scoring code.

The direct-formal profile replaces the standard terminal smoke with static integration checks. Actual runtime validity is established by the full science runs, not by a synthetic check or a fabricated smoke receipt. Source code, the Python experiment runtime and the scoring implementation are snapshotted when the evolution is created. XI credentials remain outside snapshots; all model roles use gpt-6-astra.

Regular dev references (200 utterances) provide round feedback and promotion. Selection references (196 separate utterances) are used only after all six rounds to compare successful variants. Only the baseline and the selected final model are evaluated on test (1155 utterances). All hypothesis IDs must exactly match the frozen reference IDs. Promotion includes actually evaluated disabled variants; equal WER preserves the incumbent. Disabled parent components remain disabled in descendants unless the new canonical Idea explicitly reintroduces them. No final retraining is added.

Ascend decoding runs through `asr_runtime.npu_decode`. In the validated torch_npu 2.10 environment, native `pack_padded_sequence` retained padded rows and returned incorrect data for unequal lengths, shifting frames during batched decoding. The decoding subprocess packs valid frames on CPU and returns the packed data and sorting indices to NPU; model inference remains on NPU. The wrapper also covers previously prepared frozen recipes without editing their scientific source. Runtime repairs to an existing deployment must update `source_manifest.json` and append the changed hashes and validation evidence to `deployment_repairs.jsonl`.

Native statistical reviewers report removal benefit as full-model WER minus disabled-model WER. The executor validates its value and sign against SURE reports before acceptance. Positive means removal helped. One-seed observations are not claims of statistical significance. The accepted reports feed native symbolic memory and the next round's feedback replanning.

Offline regressions cover unbounded component inventories, full ablation coverage, concurrent work units, failed cohorts, Slurm acknowledgement recovery, idempotent replay and cancellation ownership. They do not submit Slurm jobs or call provider APIs.
