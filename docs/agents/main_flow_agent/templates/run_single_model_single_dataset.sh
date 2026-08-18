#!/usr/bin/env bash
set -euo pipefail

: "${REPO_ROOT:?required}"
: "${MODEL_ID:?required}"
: "${ACTION:?reuse_result|reevaluate|infer_and_evaluate required}"

PROTOCOL_ID="${PROTOCOL_ID:-standard_system}"
case "$PROTOCOL_ID" in
  standard_system|strict_core) ;;
  *) echo "invalid protocol_id: $PROTOCOL_ID" >&2; exit 2 ;;
esac

cd "$REPO_ROOT"
case "$ACTION" in
  reuse_result)
    exit 0
    ;;
  reevaluate)
    : "${INFERENCE_ID:?required}"
    : "${EVALUATION_ID:?required}"
    : "${EVALUATION_INPUT_MANIFEST:?required}"
    python scripts/run_evaluation_bridge.py \
      --model-id "$MODEL_ID" \
      --evaluation-id "$EVALUATION_ID" \
      --inference-id "$INFERENCE_ID" \
      --inference-source published \
      --input-manifest "$EVALUATION_INPUT_MANIFEST"
    ;;
  infer_and_evaluate)
    : "${INFERENCE_ID:?required}"
    : "${EVALUATION_ID:?required}"
    : "${EVALUATION_INPUT_MANIFEST:?required}"
    : "${DATASET:?prepared dataset id required}"
    : "${DATASET_IDENTITY:?name__version@split required}"
    python scripts/generate_predictions_via_server.py \
      --model-id "$MODEL_ID" \
      --dataset "$DATASET" \
      --inference-id "$INFERENCE_ID" \
      --protocol-id "$PROTOCOL_ID"
    python scripts/finalize_inference_run.py \
      --model-id "$MODEL_ID" \
      --inference-id "$INFERENCE_ID" \
      --dataset "$DATASET_IDENTITY"
    python scripts/run_evaluation_bridge.py \
      --model-id "$MODEL_ID" \
      --evaluation-id "$EVALUATION_ID" \
      --inference-id "$INFERENCE_ID" \
      --inference-source staged \
      --input-manifest "$EVALUATION_INPUT_MANIFEST"
    ;;
  *) echo "invalid action: $ACTION" >&2; exit 2 ;;
esac
