# LibriSpeech Main-Flow Full Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read all 2619 files from the approved hpc_stor08 LibriSpeech root, run `openai__whisper-large-v3-turbo` through the strict main-flow and SURE-EVAL ASR router, publish standard reports, and write three developer READMEs.

**Architecture:** Add an explicit read-only annotation overlay to the AiSpeech source resolver because the approved audio root has no `sample_files` metadata. Materialize an isolated model descriptor and main-flow run under `sure_eval_pipnline_refine/`, use a live-capacity-selected approved vc partition for model inference, then validate route-backed WER artifacts and document the observed pipeline.

**Tech Stack:** Python 3.11, pytest, Bash, SURE-EVAL dataset/evaluation APIs, JSONL/YAML, Volcano `vc`, OpenAI Whisper MCP server.

---

### Task 1: Explicit AiSpeech Annotation Overlay

**Files:**
- Modify: `src/sure_eval/datasets/dataset_manager.py`
- Modify: `scripts/prepare_sure_dataset.py`
- Modify: `tests/test_aispeech_source_dataset_identity.py`

- [ ] **Step 1: Write failing resolver and conversion tests**

Add tests that create an audio-only source root and a separate annotation root.
Assert that resolution fails without an explicit overlay, succeeds with it,
returns `dataset_id == "librispeech_test-clean__v1.0.1"`, and rejects any
basename mismatch. Assert all converted `path` values use the audio-only source
root and metadata records both roots.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH=src .venv.hostbak/bin/pytest -q tests/test_aispeech_source_dataset_identity.py
```

Expected: the new overlay tests fail because the resolver only reads
`<source_root>/sample_files`.

- [ ] **Step 3: Extend the source reference and resolver**

Add `annotation_source_root` to `DatasetSourceRef`. Extend
`resolve_aispeech_source_entry(...)` with an explicit optional annotation root.
Both roots must remain under the configured AiSpeech roots. Dataset identity
must continue to use the selected audio source name and annotation version:

```python
dataset_id = f"{source_dataset_name}__{selected_version}"
```

When the roots differ, compare the complete annotation audio basename set with
the complete selected-root WAV basename set and raise on any difference.

- [ ] **Step 4: Remap paths and preserve provenance**

During conversion, resolve each annotation path to
`source_ref.raw_dir / Path(raw_path).name`. Add
`annotation_source_root` to row metadata, source metadata,
`conversion_report.json`, and `dataset_manifest.json`.

- [ ] **Step 5: Expose the overlay in the deterministic prepare CLI**

Add:

```text
--annotation-source-root /hpc_stor08/.../ds_pool/aispeech_phy_librispeech_test-clean
```

The option is valid only with exactly one `--dataset` source root. Include the
annotation root in `prepare_summary.json`.

- [ ] **Step 6: Run focused tests and commit**

Run the focused test command from Step 2. Expected: all tests pass.

Commit only the three Task 1 files:

```bash
git add src/sure_eval/datasets/dataset_manager.py scripts/prepare_sure_dataset.py tests/test_aispeech_source_dataset_identity.py
git commit -m "feat: support explicit aispeech annotation overlays"
```

### Task 2: Main-Flow Overlay Propagation

**Files:**
- Modify: `docs/agents/main_flow_agent/templates/run_single_model.sh`
- Modify: `docs/agents/main_flow_agent/templates/main_agent_execution_surface.json`
- Modify: `docs/agents/main_flow_agent/contracts/main_agent_dataset_unit.md`
- Modify: `docs/agents/main_flow_agent/contracts/main_agent_execution_surface_unit.md`
- Modify: `tests/test_evaluation_scripts_contracts.py`

- [ ] **Step 1: Add failing shell-contract tests**

Assert the shell accepts `ANNOTATION_SOURCE_ROOT`, rejects it for multiple
datasets, and forwards it to `prepare_sure_dataset.py` as
`--annotation-source-root`. Assert the execution-surface template records the
annotation root separately from the audio source root.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=src .venv.hostbak/bin/pytest -q tests/test_evaluation_scripts_contracts.py
```

Expected: new assertions fail because the template does not propagate an
annotation overlay.

- [ ] **Step 3: Implement shell and contract propagation**

Add `ANNOTATION_SOURCE_ROOT="${ANNOTATION_SOURCE_ROOT:-}"`, build the prepare
argument array deterministically, and record the field under
`resolved_inputs.dataset_sources[]`. State that the annotation root is
read-only provenance and never determines the canonical dataset name.

- [ ] **Step 4: Verify and commit**

Run Task 2 tests plus `bash -n` on the shell. Commit only Task 2 files.

### Task 3: Isolated Whisper MCP Runtime Descriptor

**Files:**
- Create: `sure_eval_pipnline_refine/runtime/openai__whisper-large-v3-turbo/config.yaml`
- Create: `sure_eval_pipnline_refine/runtime/openai__whisper-large-v3-turbo/model.spec.yaml`
- Create: `sure_eval_pipnline_refine/runtime/openai__whisper-large-v3-turbo/model.py`
- Create: `sure_eval_pipnline_refine/runtime/openai__whisper-large-v3-turbo/server.py`
- Create: `sure_eval_pipnline_refine/runtime/openai__whisper-large-v3-turbo/probe_runtime.sh`
- Create: `tests/test_pipeline_refine_whisper_runtime.py`

- [ ] **Step 1: Write MCP contract tests with a fake model**

Test `initialize`, `tools/list`, both ASR tool aliases, lazy model loading,
healthcheck, error response, and newline-delimited JSON-RPC serving.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=src .venv.hostbak/bin/pytest -q tests/test_pipeline_refine_whisper_runtime.py
```

Expected: fail because the isolated runtime files do not exist.

- [ ] **Step 3: Implement the minimal descriptor**

The wrapper loads `whisper.load_model("turbo", download_root=...)`, selects CUDA
when visible, calls `transcribe(..., language="en", fp16=True)`, and returns a
JSON-serializable `text` result. The server implements only the MCP methods
needed by `generate_predictions_via_server.py`.

- [ ] **Step 4: Add a read-only vc probe script**

The script records `python`, installed Whisper/Torch versions, CUDA visibility,
`ffmpeg`, and candidate checkpoint paths without downloading weights or writing
outside the run output directory.

- [ ] **Step 5: Verify and commit**

Run the Task 3 test and `bash -n probe_runtime.sh`. Commit only Task 3 files.

### Task 4: Materialize Main-Flow Run

**Files:**
- Create under: `sure_eval_pipnline_refine/runtime/openai__whisper-large-v3-turbo/eval_runs/main_agent_openai_whisper_librispeech_full_20260808_001/`

- [ ] **Step 1: Materialize all unit artifacts from templates**

Create the classification, readiness routing, plan, dataset decision, script
routing, execution surface, readiness report, assessment, run report, and
manifest JSON files. Use `strict_core`, the approved source and annotation
roots, and `librispeech_test-clean__v1.0.1` throughout.

- [ ] **Step 2: Select the live partition**

Run `vc info` and select the first partition with a free GPU from:

```text
pdgpu-ezkws
pdgpu-3090
pdgpu-3090-data
```

Record the observation time, allocations, and pinned selection in the surface.

- [ ] **Step 3: Materialize the shell**

Copy only `docs/agents/main_flow_agent/templates/run_single_model.sh`, record
its SHA-256, and set the approved model, roots, full scope (`MAX_SAMPLES=0`),
English WER, resume, run directory, results directory, and annotation overlay.

- [ ] **Step 4: Run static readiness gates**

Run `bash -n`, JSON parsing, protocol rejection, dataset source rejection, and
`scripts/check_execution_surface_compliance.py`. Readiness remains false until
the vc smoke succeeds.

### Task 5: VC Runtime Probe and Bounded Smoke

**Files:**
- Update run artifacts from Task 4 with job IDs, logs, and readiness evidence.

- [ ] **Step 1: Submit the runtime probe to the pinned partition**

Use the declared image:

```text
docker.v2.aispeech.com/sjtu/sjtu_yukai-dujunhao-sure_openai__whisper-large-v3-turbo:v1.0
```

Record the exact `vc submit` command, job ID, task ID, and persistent log path.

- [ ] **Step 2: Inspect the probe result**

Confirm the image interpreter, Whisper import, CUDA, ffmpeg, and a local
`large-v3-turbo.pt`/`turbo.pt` checkpoint. Update the runtime descriptor with
observed paths. Do not permit network model fallback.

- [ ] **Step 3: Submit a 3-sample vc smoke through the main-flow shell**

Use a separate smoke run ID with `MAX_SAMPLES=3`. Require three non-empty
predictions, valid prediction payload, route-backed WER output, and
`strict_core` reports.

- [ ] **Step 4: Promote readiness**

Only after smoke success, set `execution_ready=true`, record smoke evidence,
and preserve the smoke job ID and logs.

### Task 6: Complete 2619-Sample Inference and Evaluation

**Files:**
- Update the full run directory and standard results mirror.

- [ ] **Step 1: Re-query capacity and submit the full run**

Keep the approved partition order, pin the selected queue, use one GPU, and
enable resume. Record the full job ID immediately.

- [ ] **Step 2: Monitor to terminal state**

Use `vc list -j`, `vc logs -t`, and persistent run logs. If the job terminates,
use `vc describe -j` and classify the failure before retrying. Never fall back
to local model inference.

- [ ] **Step 3: Validate predictions and evaluation**

Require exactly 2619 prediction keys, zero missing/extra/duplicate/empty rows,
`sure.eval.payload.v2`, a numeric WER, and canonical route artifacts produced
through `sure_eval.evaluation.scripts.run_task(...)`.

- [ ] **Step 4: Finalize assessment, run report, manifest, and snapshot**

Record actual execution path, queue, image, job ID, dataset provenance,
protocol, route, metric result, and all artifact paths. Persist reports because
the user explicitly requested final report generation.

### Task 7: Three Developer READMEs

**Files:**
- Create: `sure_eval_pipnline_refine/DATASET_README.md`
- Create: `sure_eval_pipnline_refine/MAIN_AGENT_EVALUATION_README.md`
- Create: `sure_eval_pipnline_refine/REPORT_README.md`

- [ ] **Step 1: Write dataset documentation from actual artifacts**

Document the selected audio root, annotation overlay, exact set validation,
version, 2619-row preparation, canonical ID, and reproducible prepare command.

- [ ] **Step 2: Write main-agent and evaluation-route documentation**

Document every main-flow unit, selected queue, image and job IDs, MCP inference,
the actual ASR route ID, ordered node IDs, conversion steps, node inputs and
outputs, pipeline descriptions, and resume procedure.

- [ ] **Step 3: Write report documentation**

Document `evaluation_payload.json`, `report.jsonl`, `protocol.yaml`, metric and
sample reports, field invariants, measured WER, and verification commands.

### Task 8: Completion Audit

**Files:**
- Create: `sure_eval_pipnline_refine/completion_audit.json`

- [ ] **Step 1: Run relevant automated tests**

Run:

```bash
PYTHONPATH=src .venv.hostbak/bin/pytest -q \
  tests/test_aispeech_source_dataset_identity.py \
  tests/test_evaluation_scripts_contracts.py \
  tests/test_main_flow_protocol_policy.py \
  tests/test_prediction_scripts.py \
  tests/test_pipeline_refine_whisper_runtime.py
```

- [ ] **Step 2: Run the artifact auditor**

Verify all required files exist, all JSON/YAML parses, dataset identity is
identical at every boundary, protocol is always `strict_core`, prediction and
sample counts equal 2619, the route nodes match the pipeline description, and
all README links resolve.

- [ ] **Step 3: Write the completion audit and inspect diffs**

Record each objective requirement with its authoritative artifact and pass/fail
result. Run `git diff --check` on touched source, test, contract, and README
files. Do not mark the goal complete if any audit item is false.
