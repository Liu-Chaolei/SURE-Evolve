#!/bin/bash
set -eo pipefail

REPO_DIR="${SURE_MASTER_REPO_DIR:-/hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster}"
LOG_DIR="${SURE_MASTER_LOG_DIR:-/hpc_stor03/sjtu_home/chaolei.liu/log}"
PYTHON_BIN="${SURE_MASTER_PYTHON:-/opt/conda/envs/evomaster/bin/python}"
F5TTS_PYTHON="${SURE_TTS_PYTHON:-/opt/conda/envs/f5tts/bin/python}"
CONFIG_PATH="${SURE_MASTER_CONFIG:-configs/sure_master/gpt-5-f5tts-docker.yaml}"
TASK_PATH="${SURE_MASTER_TASK:-playground/sure_master/data/tts_en_wer_f5tts_description.md}"
F5TTS_ROOT="${SURE_TTS_SOURCE_ROOT:-/hpc_stor03/sjtu_home/chaolei.liu/TTS/F5-TTS}"

cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}"

LOG_FILE="${SURE_MASTER_LOG_FILE:-${LOG_DIR}/sure_master_f5tts_${VC_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "repo_dir=${REPO_DIR}"
echo "log_file=${LOG_FILE}"
echo "python=${PYTHON_BIN}"
echo "f5tts_python=${F5TTS_PYTHON}"
echo "config=${CONFIG_PATH}"
echo "task=${TASK_PATH}"
echo "f5tts_root=${F5TTS_ROOT}"

if [[ ! -f .env ]]; then
  echo "Missing .env in ${REPO_DIR}; API settings are required at runtime."
  exit 2
fi

if [[ ! -d "${F5TTS_ROOT}" ]]; then
  echo "Missing F5-TTS source directory: ${F5TTS_ROOT}"
  exit 2
fi

set -a
source .env
set +a

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

HF_CACHE_DIR="${HUGGINGFACE_HUB_CACHE:-${HF_HOME:-/hpc_stor03/sjtu_home/chaolei.liu/.cache/huggingface}/hub}"
VOCOS_CACHE_DIR="${HF_CACHE_DIR}/models--charactr--vocos-mel-24khz"
if [[ "${HF_HUB_OFFLINE}" == "1" && ! -d "${VOCOS_CACHE_DIR}/snapshots" ]]; then
  echo "Missing cached Vocos model while HF_HUB_OFFLINE=1: ${VOCOS_CACHE_DIR}"
  echo "Pre-cache charactr/vocos-mel-24khz or set HF_HUB_OFFLINE=0 for a networked run."
  exit 2
fi

for required_var in OPENAI_API_KEY GPT_BASE_URL GPT_CHAT_MODEL; do
  if [[ -z "${!required_var:-}" ]]; then
    echo "Missing ${required_var} after sourcing .env."
    exit 2
  fi
done

"${PYTHON_BIN}" - <<'PY'
import openai
import yaml
print("evomaster runtime ok")
PY

"${F5TTS_PYTHON}" - <<'PY'
import torch
import torchaudio
import f5_tts
print("f5tts runtime ok", torch.__version__, torch.version.cuda)
PY

KEEPALIVE_PID=""
if [[ "${SURE_GPU_KEEPALIVE:-1}" == "1" ]]; then
  "${F5TTS_PYTHON}" - <<'PY' &
import os
import signal
import sys
import time

import torch

running = True


def _stop(signum, frame):
    global running
    running = False


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)

if not torch.cuda.is_available():
    sys.exit(0)

device = torch.device("cuda:0")
size = int(float(os.environ.get("SURE_GPU_KEEPALIVE_SIZE", "4096")))
interval = float(os.environ.get("SURE_GPU_KEEPALIVE_INTERVAL", "0.05"))
size = max(256, min(size, 8192))
interval = max(0.05, interval)
dtype = torch.float16
x = torch.randn((size, size), device=device, dtype=dtype)
y = torch.randn((size, size), device=device, dtype=dtype)
print(f"gpu_keepalive_active size={size} interval={interval}", flush=True)
while running:
    x = torch.mm(x, y)
    x.mul_(0.0001)
    torch.cuda.synchronize(device)
    time.sleep(interval)
PY
  KEEPALIVE_PID=$!
  echo "gpu_keepalive_pid=${KEEPALIVE_PID}"
fi

cleanup() {
  if [[ -n "${KEEPALIVE_PID}" ]]; then
    kill "${KEEPALIVE_PID}" >/dev/null 2>&1 || true
    wait "${KEEPALIVE_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

set +e
"${PYTHON_BIN}" -u run.py \
  --agent sure_master \
  --config "${CONFIG_PATH}" \
  --task "${TASK_PATH}"
run_rc=$?
set -e
cleanup
exit "${run_rc}"
