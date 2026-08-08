#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CUDA_VERSION="11.6.2"
CUDNN_VERSION="8"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime-ubuntu20.04}"
MINICONDA_URL="${MINICONDA_URL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh}"
PYTORCH_WHEEL_BASE="${PYTORCH_WHEEL_BASE:-https://mirrors.aliyun.com/pytorch-wheels/cu116}"
NUM2WORDS_URL="${NUM2WORDS_URL:-https://files.pythonhosted.org/packages/f6/58/ad645bd38b4b648eb2fc2ba1b909398e54eb0cbb6a7dbd2b4953e38c9621/num2words-0.5.14.tar.gz}"
IMAGE="docker.v2.aispeech.com/sjtu/sjtu_yukai-chaolei-suremaster_icefall"
TAG="${1:-v1.0}"

rm -rf build
mkdir -p build/res

cp -rf source/* build/res/
cp -f Dockerfile build/

docker build \
  --build-arg CUDA_VERSION="${CUDA_VERSION}" \
  --build-arg CUDNN_VERSION="${CUDNN_VERSION}" \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg MINICONDA_URL="${MINICONDA_URL}" \
  --build-arg PYTORCH_WHEEL_BASE="${PYTORCH_WHEEL_BASE}" \
  --build-arg NUM2WORDS_URL="${NUM2WORDS_URL}" \
  -t "${IMAGE}:${TAG}" \
  build

docker push "${IMAGE}:${TAG}"
