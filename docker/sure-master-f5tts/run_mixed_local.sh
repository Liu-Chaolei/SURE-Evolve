#!/bin/bash
set -eo pipefail

REPO_DIR="${SURE_MASTER_REPO_DIR:-/hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster}"
LOCAL_ENV_PYTHON="${SURE_LOCAL_PYTHON:-/hpc_stor03/sjtu_home/chaolei.liu/anaconda3/envs/suremaster-f5tts-local/bin/python}"
CONFIG_PATH="${SURE_MASTER_CONFIG:-configs/sure_master/gpt-5-f5tts-regular-search-mixed.yaml}"
TASK_PATH="${SURE_MASTER_TASK:-playground/sure_master/data/tts_en_wer_f5tts_description.md}"
RUN_DIR="${SURE_MASTER_RUN_DIR:-}"
LOG_DIR="${SURE_MASTER_LOG_DIR:-/hpc_stor03/sjtu_home/chaolei.liu/log}"
MIN_LOCAL_GPUS="${SURE_LOCAL_MIN_GPUS:-4}"
SKIP_LOCAL_F5TTS_CHECK="${SURE_SKIP_LOCAL_F5TTS_CHECK:-auto}"

PRESET_MASTER_PYTHON="${SURE_MASTER_PYTHON:-}"
PRESET_TTS_PYTHON="${SURE_TTS_PYTHON:-}"
if [[ -n "${SURE_LOCAL_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${SURE_LOCAL_CUDA_VISIBLE_DEVICES}"
fi

cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}"

LOG_FILE="${SURE_MASTER_LOG_FILE:-${LOG_DIR}/sure_master_f5tts_mixed_local_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "repo_dir=${REPO_DIR}"
echo "log_file=${LOG_FILE}"
echo "config=${CONFIG_PATH}"
echo "task=${TASK_PATH}"
echo "run_dir=${RUN_DIR:-[default]}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-[unset; config may auto-select]}"

if [[ ! -f .env ]]; then
  echo "Missing .env in ${REPO_DIR}; API settings are required at runtime."
  exit 2
fi

set -a
source .env
set +a

PYTHON_BIN="${PRESET_MASTER_PYTHON:-${SURE_MASTER_PYTHON:-${LOCAL_ENV_PYTHON}}}"
F5TTS_PYTHON="${PRESET_TTS_PYTHON:-${SURE_LOCAL_F5TTS_PYTHON:-${LOCAL_ENV_PYTHON}}}"
export SURE_TTS_PYTHON="${F5TTS_PYTHON}"

SURE_ROOT="${SURE_ROOT:-/hpc_stor03/sjtu_home/chaolei.liu/sure}"
SURE_PYTHONPATH="${SURE_PYTHONPATH:-${SURE_ROOT}/src}"
export PYTHONPATH="base_model/root/src:${SURE_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "python=${PYTHON_BIN}"
echo "f5tts_python=${SURE_TTS_PYTHON}"
echo "pythonpath=${PYTHONPATH}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python is not executable: ${PYTHON_BIN}"
  exit 2
fi
if [[ ! -x "${SURE_TTS_PYTHON}" ]]; then
  echo "F5-TTS Python is not executable: ${SURE_TTS_PYTHON}"
  exit 2
fi
if ! command -v vc >/dev/null 2>&1; then
  echo "vc command is required because training-level candidates run as VC child jobs."
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
import sure_eval
print("evomaster runtime ok")
PY

REMOTE_CANDIDATE_TYPES="$("${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as f:
    payload = yaml.safe_load(f) or {}

remote = ((payload.get("sure") or {}).get("remote_training") or {})
raw_types = remote.get("candidate_types")
if raw_types is None:
    raw_types = (payload.get("sure") or {}).get("remote_candidate_types")
if raw_types is None:
    values = ["training"]
elif isinstance(raw_types, (list, tuple)):
    values = [str(value).strip().lower() for value in raw_types]
else:
    values = [part.strip().lower() for part in str(raw_types).split(",")]
if "all" in values or "*" in values:
    values = ["inference", "training"]
print(",".join(value for value in values if value))
PY
)"

if [[ "${SKIP_LOCAL_F5TTS_CHECK}" == "auto" ]]; then
  if [[ ",${REMOTE_CANDIDATE_TYPES}," == *",inference,"* && ",${REMOTE_CANDIDATE_TYPES}," == *",training,"* ]]; then
    SKIP_LOCAL_F5TTS_CHECK=1
  else
    SKIP_LOCAL_F5TTS_CHECK=0
  fi
fi
echo "remote_candidate_types=${REMOTE_CANDIDATE_TYPES:-[default:training]}"
echo "skip_local_f5tts_check=${SKIP_LOCAL_F5TTS_CHECK}"

if [[ "${SKIP_LOCAL_F5TTS_CHECK}" == "1" ]]; then
  echo "Skipping local F5-TTS/GPU check because all F5-TTS candidate execution is remote."
else
  "${SURE_TTS_PYTHON}" - <<PY
import sys
import torch
import torchaudio
import torchcodec
import f5_tts

min_gpus = int("${MIN_LOCAL_GPUS}")
available = torch.cuda.is_available()
count = torch.cuda.device_count()
print("f5tts runtime ok", torch.__version__, torch.version.cuda)
print("cuda_available", available)
print("cuda_device_count", count)
if not available or count < min_gpus:
    print(f"Need at least {min_gpus} local CUDA devices for mixed regular_search.", file=sys.stderr)
    sys.exit(3)
for index in range(count):
    props = torch.cuda.get_device_properties(index)
    print(f"gpu[{index}]={props.name} memory_total_mib={props.total_memory // 1024 // 1024}")
PY
fi

RUN_ARGS=(
  -u run.py
  --agent sure_master
  --config "${CONFIG_PATH}"
  --task "${TASK_PATH}"
)
if [[ -n "${RUN_DIR}" ]]; then
  RUN_ARGS+=(--run-dir "${RUN_DIR}")
fi

"${PYTHON_BIN}" "${RUN_ARGS[@]}"
