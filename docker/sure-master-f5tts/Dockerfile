ARG BASE_IMAGE="docker.v2.aispeech.com/sjtu/sjtu_yukai-chaolei-suremaster_icefall:v1.0"
FROM ${BASE_IMAGE}

USER root

ARG DEBIAN_FRONTEND=noninteractive
ARG CONDA_DIR="/opt/conda"
ARG PYTORCH_WHEEL_BASE="https://mirrors.aliyun.com/pytorch-wheels/cu121"

ENV PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple"
ENV PIP_TRUSTED_HOST="mirrors.aliyun.com"
ENV PIP_NO_CACHE_DIR=0
ENV PYTHONUNBUFFERED=1
ENV PYTHONNOUSERSITE=1
ENV PATH="${CONDA_DIR}/envs/evomaster/bin:${CONDA_DIR}/bin:${PATH}"

COPY res /tmp/res

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      ffmpeg \
      git \
      libgl1 \
      libglib2.0-0 \
      libsndfile1 \
      libsox-fmt-all \
      less \
      sox \
      sudo \
      vim \
      wget; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    "${CONDA_DIR}/bin/conda" config --system --set auto_update_conda false; \
    if ! "${CONDA_DIR}/bin/conda" env list | awk '{print $1}' | grep -qx evomaster; then \
      "${CONDA_DIR}/bin/conda" create -y -n evomaster python=3.10; \
    fi; \
    if ! "${CONDA_DIR}/bin/conda" env list | awk '{print $1}' | grep -qx f5tts; then \
      "${CONDA_DIR}/bin/conda" create -y -n f5tts python=3.10; \
    fi; \
    "${CONDA_DIR}/bin/conda" clean -afy

RUN --mount=type=cache,target=/root/.cache/pip set -eux; \
    "${CONDA_DIR}/envs/evomaster/bin/python" -m pip install --upgrade pip setuptools wheel; \
    "${CONDA_DIR}/envs/evomaster/bin/python" -m pip install -r /tmp/res/evomaster-requirements.txt; \
    "${CONDA_DIR}/envs/evomaster/bin/python" -m pip install -r /tmp/res/sure-requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip set -eux; \
    "${CONDA_DIR}/envs/f5tts/bin/python" -m pip install --upgrade pip setuptools wheel; \
    "${CONDA_DIR}/envs/f5tts/bin/python" -m pip install \
      "${PYTORCH_WHEEL_BASE}/torch-2.4.0%2Bcu121-cp310-cp310-linux_x86_64.whl" \
      "${PYTORCH_WHEEL_BASE}/torchaudio-2.4.0%2Bcu121-cp310-cp310-linux_x86_64.whl"

RUN --mount=type=cache,target=/root/.cache/pip set -eux; \
    "${CONDA_DIR}/envs/f5tts/bin/python" -m pip install f5-tts==1.1.20 --no-deps; \
    "${CONDA_DIR}/envs/f5tts/bin/python" -m pip install \
      --constraint /tmp/res/f5tts-constraints.txt \
      -r /tmp/res/f5tts-requirements.txt

RUN set -eux; \
    "${CONDA_DIR}/envs/evomaster/bin/python" -c "import openai, yaml; print('evomaster deps ok')"; \
    "${CONDA_DIR}/envs/f5tts/bin/python" -c "import torch, torchaudio, soundfile, f5_tts; print('f5tts deps ok', torch.__version__, torch.version.cuda)"; \
    rm -rf /tmp/res

RUN set -eux; \
    if ! "${CONDA_DIR}/bin/conda" env list | awk '{print $1}' | grep -qx nodepy311; then \
      "${CONDA_DIR}/bin/conda" create -y -n nodepy311 python=3.11; \
    fi; \
    ln -sf "${CONDA_DIR}/envs/nodepy311/bin/python3.11" /usr/bin/python3.11; \
    /usr/bin/python3.11 -V; \
    "${CONDA_DIR}/bin/conda" clean -afy

ENV HF_HUB_DISABLE_XET=1
ENV HF_HUB_OFFLINE=1

WORKDIR /workspace
