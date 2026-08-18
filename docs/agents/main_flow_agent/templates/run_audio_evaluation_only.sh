#!/usr/bin/env bash
set -euo pipefail

: "${REPO_ROOT:?required}"
: "${MODEL_ID:?required}"
: "${INFERENCE_ID:?required}"
: "${EVALUATION_ID:?required}"
: "${INFERENCE_SOURCE:?published|staged required}"
: "${EVALUATION_INPUT_MANIFEST:?required}"

cd "$REPO_ROOT"
python scripts/run_evaluation_bridge.py \
  --model-id "$MODEL_ID" \
  --evaluation-id "$EVALUATION_ID" \
  --inference-id "$INFERENCE_ID" \
  --inference-source "$INFERENCE_SOURCE" \
  --input-manifest "$EVALUATION_INPUT_MANIFEST"
