# SURE-EVAL Model Tool Agent

The model tool agent creates and validates one local model adapter. See the
authoritative [AGENTS.md](AGENTS.md).

## Input

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

The agent reads published adapters only from
`/hpc_stor03/project/oref/nfs/models`. It writes a new or repaired adapter only
to `src/sure_eval/models/<model_id>`.

## Protocol Evidence

Every adapter must support `standard_system`, the official pinned entrypoint
with official defaults and no user overrides. `strict_core` is an optional,
explicit capability whose deterministic controls must all be mapped and
verified.

## Publication

After validation:

```bash
python scripts/finalize_model_adapter.py --model-id OpenAI__whisper-large-v3
```

This creates `publication_artifacts.json` and
`publication_request.json` locally. It does not write to NFS. A human reviewer
must transfer the adapter and create the verified `publication.json` record.
