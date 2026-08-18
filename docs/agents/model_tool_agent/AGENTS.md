# SURE-EVAL Model Tool Agent Contract v2

This file is the authoritative model onboarding instruction. Detailed task and
environment playbooks remain useful only where they do not conflict with this
v2 contract.

## Purpose

Turn one model specification into a reproducible local adapter and deployment
evidence. Dataset selection, benchmark execution, result reuse, and evaluation
belong to the main-flow agent.

## Storage Boundary

```text
published model reads: /hpc_stor03/project/oref/nfs/models
local adapter writes:  src/sure_eval/models/<model_id>
```

- Existing published adapters are discovered only through `ModelRegistry`.
- Never inspect an alternative shared model root for registry decisions.
- Never write to `/hpc_stor03/project/oref/nfs`.
- The model ID is one canonical directory segment, normally provider and model
  joined by `__`.
- Do not accept an output directory or NFS destination from user input.

## Structured Input

```yaml
schema: sure.eval.model_tool_input.v2
model_id: OpenAI__whisper-large-v3
task_type: asr
deployment_type: local
repository:
  url: https://github.com/openai/whisper
  commit: <pinned commit>
weights:
  source: huggingface
  revision: <pinned revision>
environment:
  preferred_backend: uv
  python_version: "3.11"
```

Unknown path/output fields are invalid. Pin floating repository and weight
identities before declaring the adapter ready.

## State Machine

```text
INTAKE
  -> CHECK_PUBLISHED_REGISTRY
  -> DISCOVER_OFFICIAL_EVIDENCE
  -> PLAN
  -> VALIDATE_SPEC
  -> BUILD_ENV
  -> FETCH_WEIGHTS
  -> VALIDATE_IMPORT
  -> VALIDATE_LOAD
  -> VALIDATE_INFER
  -> VALIDATE_OUTPUT_CONTRACT
  -> GENERATE_ADAPTER
  -> RESOLVE_PROTOCOL_CAPABILITIES
  -> VALIDATE_DEPLOYMENT
  -> FINALIZE_LOCAL_ADAPTER
  -> WRITE_PUBLICATION_REQUEST
  -> REPORT
```

Each stage writes structured evidence. Failures record a category and stop or
resume from a named checkpoint; no unrecorded repair or blind retry is allowed.

## Protocol Capability Contract

Every adapter `config.yaml` must declare `protocols`.

`standard_system` is mandatory and records the pinned official default flow:

```yaml
protocols:
  standard_system:
    enabled: true
    official_defaults:
      repository: <official repository>
      commit: <commit>
      weights_revision: <revision>
      entrypoint: <official inference entrypoint>
      invocation_overrides: {}
      effective_params: {}
```

`strict_core` is optional. Enable it only when every control from
`config/protocols.yaml` has a `parameter`, `attestation`, or justified
`not_applicable` mapping. Do not claim strict support by copying generic values
without verifying the model API.

## Required Local Artifacts

```text
src/sure_eval/models/<model_id>/
├── config.yaml
├── model.spec.yaml
├── model.py
├── server.py
├── validate.py
├── fixture/
├── checkpoints/                 # when local weights are required
├── .runtime/                    # local environment/cache; not published
├── artifacts/
│   ├── evidence.json
│   ├── build_plan.json
│   ├── build.log
│   ├── validation.log
│   ├── sample_output.json
│   ├── protocol_validation.json
│   ├── deployment_validation.json
│   └── verdict.json
├── publication_artifacts.json
└── publication_request.json
```

For local deployment, Docker/registry/cluster evidence required by the selected
environment playbook must pass before the verdict is `tool_ready`. API models
must record endpoint authentication and bounded real-call validation without
storing secrets.

## Finalization And Human Publication

Run:

```bash
python scripts/finalize_model_adapter.py --model-id <model_id>
```

The script checks the fixed local staging path, rejects symlinks, hashes the
publishable files, and writes a request with
`automatic_publish_allowed: false`. It never copies to NFS.

A human reviewer verifies the evidence, manually transfers files to
`/hpc_stor03/project/oref/nfs/models/<model_id>`, and creates
`publication.json` using the exact manifest and aggregate digest. Main flow
recomputes these values before use.

## Required Run Outputs

The final report must include model ID, repository/weights pins, environment
identity, validation status, supported protocol IDs, artifact digest,
publication request path, and an explicit statement that NFS was not modified.
