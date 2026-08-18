# Whisper MCP Server Implementation Plan

1. Add a model-runtime-independent protocol regression test for the configured
   Whisper tools and MCP request lifecycle.
2. Run the test in the declared model container and confirm it fails against
   the current one-shot server.
3. Implement a line-delimited MCP/JSON-RPC server in the model entrypoint with
   lazy model loading and both configured transcription tool names.
4. Rerun the protocol test in the same container and confirm it passes.
5. Resume the run-local three-sample evaluation on `pdgpu-ezkws`.
6. Validate prediction, scoring, report artifacts, and the dataset identity
   emitted by the existing dataset loader without changing its naming logic.
