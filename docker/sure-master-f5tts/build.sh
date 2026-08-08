#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BASE_IMAGE="${BASE_IMAGE:-docker.v2.aispeech.com/sjtu/sjtu_yukai-chaolei-suremaster_icefall:v1.0}"
PYTORCH_WHEEL_BASE="${PYTORCH_WHEEL_BASE:-https://mirrors.aliyun.com/pytorch-wheels/cu121}"
IMAGE="docker.v2.aispeech.com/sjtu/sjtu_yukai-chaolei-suremaster_f5tts"
TAG="${1:-v1.0}"

rm -rf build
mkdir -p build/res

cp -rf source/* build/res/
cp -f Dockerfile build/

docker build \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg PYTORCH_WHEEL_BASE="${PYTORCH_WHEEL_BASE}" \
  -t "${IMAGE}:${TAG}" \
  build

docker push "${IMAGE}:${TAG}"
