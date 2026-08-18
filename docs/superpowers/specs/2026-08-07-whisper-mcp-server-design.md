# Whisper MCP Server Design

## Context

The `openai__whisper-large-v3-turbo` model entrypoint currently loads the model,
prints one health payload, and exits. The SURE prediction runner communicates
with model servers over line-delimited MCP/JSON-RPC, so the process exits before
it can answer `initialize`, `tools/list`, or `tools/call`.

## Design

Replace the one-shot entrypoint with a small stdio MCP server consistent with
the existing ASR model wrappers:

- implement `initialize`, `notifications/initialized`, `tools/list`, and
  `tools/call`;
- advertise `asr_transcribe`, its configured backward-compatible alias
  `transcribe_audio`, and `healthcheck`;
- lazily construct and load `ModelWrapper` only for transcription calls;
- pass `audio_path` and optional Whisper language codes through unchanged;
- return successful tool payloads in both MCP text content and `raw` form;
- keep all diagnostics out of stdout so the JSON-RPC stream remains valid.

The server accepts an injectable model factory solely to enable protocol tests
without importing model runtimes or loading weights. Dataset preparation,
dataset identity derivation, report naming, and scoring behavior are unchanged.

## Verification

A standalone `unittest` module checks protocol initialization, tool discovery,
both transcription aliases, lazy loading, language forwarding, health status,
and unknown-tool errors. After the protocol test passes in the declared model
image, rerun the existing three-sample main-flow evaluation on `pdgpu-ezkws` and
inspect its canonical evaluation artifacts and dataset identifier.
