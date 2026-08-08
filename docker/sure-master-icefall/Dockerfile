# Docker Build Arguments
ARG CUDA_VERSION="11.6.2"
ARG CUDNN_VERSION="8"
ARG BASE_IMAGE="nvcr.io/nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04"
FROM ${BASE_IMAGE}

ARG CONDA_DIR="/opt/conda"
ARG MINICONDA_URL="https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh"
ARG PYTORCH_WHEEL_BASE="https://mirrors.aliyun.com/pytorch-wheels/cu116"
ARG K2_WHEEL_URL="https://hf-mirror.com/csukuangfj/k2/resolve/main/ubuntu-cuda/k2-1.24.3.dev20230725%2Bcuda11.6.torch1.13.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="${CONDA_DIR}/envs/evomaster/bin:${CONDA_DIR}/bin:${PATH}"
ENV PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple"
ENV PIP_TRUSTED_HOST="mirrors.aliyun.com"
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONNOUSERSITE=1

COPY res /tmp/res

RUN sed -i \
      -e 's@http://archive.ubuntu.com/ubuntu/@http://mirrors.aliyun.com/ubuntu/@g' \
      -e 's@http://security.ubuntu.com/ubuntu/@http://mirrors.aliyun.com/ubuntu/@g' \
      /etc/apt/sources.list \
    && apt-get clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      ffmpeg \
      git \
      gawk \
      libgl1 \
      libglib2.0-0 \
      libsndfile1 \
      net-tools \
      sox \
      sudo \
      vim \
      wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q --timeout=60 --tries=3 "${MINICONDA_URL}" -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm -f /tmp/miniconda.sh \
    && "${CONDA_DIR}/bin/conda" config --system --set auto_update_conda false \
    && ( "${CONDA_DIR}/bin/conda" config --system --remove-key channels || true ) \
    && "${CONDA_DIR}/bin/conda" config --system --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
    && "${CONDA_DIR}/bin/conda" config --system --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r \
    && "${CONDA_DIR}/bin/conda" config --system --set show_channel_urls true \
    && "${CONDA_DIR}/bin/conda" create -y -n evomaster python=3.10 \
    && "${CONDA_DIR}/bin/conda" create -y -n icefall python=3.8 \
    && "${CONDA_DIR}/bin/conda" clean -afy

RUN "${CONDA_DIR}/envs/evomaster/bin/python" -m pip install --upgrade pip setuptools wheel \
    && "${CONDA_DIR}/envs/evomaster/bin/python" -m pip install -r /tmp/res/evomaster-requirements.txt \
    && "${CONDA_DIR}/envs/evomaster/bin/python" -m pip install -r /tmp/res/sure-requirements.txt

ARG NUM2WORDS_URL="https://files.pythonhosted.org/packages/f6/58/ad645bd38b4b648eb2fc2ba1b909398e54eb0cbb6a7dbd2b4953e38c9621/num2words-0.5.14.tar.gz"

RUN "${CONDA_DIR}/envs/icefall/bin/python" -m pip install --upgrade pip setuptools wheel \
    && "${CONDA_DIR}/envs/icefall/bin/python" -m pip install \
      "${PYTORCH_WHEEL_BASE}/torch-1.13.0%2Bcu116-cp38-cp38-linux_x86_64.whl" \
      "${PYTORCH_WHEEL_BASE}/torchaudio-0.13.0%2Bcu116-cp38-cp38-linux_x86_64.whl" \
    && "${CONDA_DIR}/envs/icefall/bin/python" -m pip install "${K2_WHEEL_URL}" \
    && "${CONDA_DIR}/envs/icefall/bin/python" -m pip install \
      numpy==1.24.4 \
      lhotse==1.23.0 \
    && "${CONDA_DIR}/envs/icefall/bin/python" -m pip install "${NUM2WORDS_URL}"

RUN "${CONDA_DIR}/envs/icefall/bin/python" -m pip install -r /tmp/res/icefall-requirements.txt

RUN "${CONDA_DIR}/envs/evomaster/bin/python" -c "import pydantic, openai, yaml; print('evomaster deps ok')" \
    && "${CONDA_DIR}/envs/icefall/bin/python" -c "import importlib.metadata as m; import torch, torchaudio, lhotse; print('icefall deps ok', torch.__version__, torch.version.cuda, lhotse.__version__, m.version('k2'))" \
    && rm -rf /tmp/res

WORKDIR /workspace
