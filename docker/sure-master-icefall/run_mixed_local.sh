#!/bin/bash
set -eo pipefail

REPO_DIR="${SURE_MASTER_REPO_DIR:-/hpc_stor03/sjtu_home/chaolei.liu/SURE-Evolve}"
LOCAL_ENV_PYTHON="${SURE_LOCAL_PYTHON:-/hpc_stor03/sjtu_home/chaolei.liu/anaconda3/envs/suremaster-f5tts-local/bin/python}"
CONFIG_PATH="${SURE_MASTER_CONFIG:-configs/sure_master/zai-tedlium3-cuda-vc-evolution.yaml}"
TASK_PATH="${SURE_MASTER_TASK:-playground/sure_master/data/asr_tedlium3_evolution_description.md}"
RUN_DIR="${SURE_MASTER_RUN_DIR:-}"
RUN_NAME="${SURE_MASTER_RUN_NAME:-}"
LOG_DIR="${SURE_MASTER_LOG_DIR:-/hpc_stor03/sjtu_home/chaolei.liu/log}"

PRESET_MASTER_PYTHON="${SURE_MASTER_PYTHON:-}"

cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}"

if [[ -n "${SURE_MASTER_LOG_FILE:-}" ]]; then
  LOG_FILE="${SURE_MASTER_LOG_FILE}"
elif [[ -n "${RUN_NAME}" ]]; then
  LOG_FILE="${LOG_DIR}/sure_master_icefall_${RUN_NAME}.log"
else
  LOG_FILE="${LOG_DIR}/sure_master_icefall_mixed_local_$(date +%Y%m%d_%H%M%S).log"
fi
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "repo_dir=${REPO_DIR}"
echo "log_file=${LOG_FILE}"
echo "config=${CONFIG_PATH}"
echo "task=${TASK_PATH}"

RUN_ARGS=()
if [[ -n "${RUN_DIR}" && -n "${RUN_NAME}" ]]; then
  echo "Set only one of SURE_MASTER_RUN_DIR or SURE_MASTER_RUN_NAME."
  exit 2
elif [[ -n "${RUN_DIR}" ]]; then
  RUN_ARGS+=(--run-dir "${RUN_DIR}")
  echo "run_dir=${RUN_DIR}"
elif [[ -n "${RUN_NAME}" ]]; then
  RUN_ARGS+=(--run-dir "${RUN_NAME}")
  echo "run_name=${RUN_NAME} (resolved under EvoMaster's default run root)"
fi

if [[ ! -f .env ]]; then
  echo "Missing .env in ${REPO_DIR}; API settings are required at runtime."
  exit 2
fi

set -a
source .env
set +a

PYTHON_BIN="${PRESET_MASTER_PYTHON:-${SURE_LOCAL_PYTHON:-${LOCAL_ENV_PYTHON}}}"
export XLAB_PYTHON="${XLAB_PYTHON:-${PYTHON_BIN}}"
SURE_ROOT="${SURE_ROOT:-/hpc_stor03/sjtu_home/chaolei.liu/sure}"
SURE_PYTHONPATH="${SURE_PYTHONPATH:-${SURE_ROOT}/src}"

export PYTHONPATH="${SURE_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "python=${PYTHON_BIN}"
echo "pythonpath=${PYTHONPATH}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python is not executable: ${PYTHON_BIN}"
  exit 2
fi
if ! command -v vc >/dev/null 2>&1; then
  echo "vc command is required because training-level candidates run as VC child jobs."
  exit 2
fi

for required_var in ZAI_API_KEY ZAI_BASE_URL; do
  if [[ -z "${!required_var:-}" ]]; then
    echo "Missing ${required_var} after sourcing .env."
    exit 2
  fi
done

# SURE and XLab speak OpenAI-compatible protocols; keep the source credential
# names ZAI_* while exposing only the coordinator-side aliases they require.
export OPENAI_API_KEY="${ZAI_API_KEY}"
export OPENAI_BASE_URL="${ZAI_BASE_URL}"
export LLM_BASE_URL="${ZAI_BASE_URL}"
export LLM_MODEL="glm-5.3-flash"

"${PYTHON_BIN}" - <<'PY'
import evomaster
import openai
import sure_eval
import yaml

print("evomaster runtime ok")
PY

"${XLAB_PYTHON}" - <<'PY'
import faiss
import sentence_transformers

print("xlab research runtime ok")
PY

if [[ "${SURE_SKIP_ZAI_PREFLIGHT:-0}" != "1" ]]; then
  "${PYTHON_BIN}" - <<'PY'
import json
import os
import sys

from playground.sure_master.tools.zai_preflight import check_zai_api

result = check_zai_api({
    "ZAI_API_KEY": os.environ["ZAI_API_KEY"],
    "ZAI_BASE_URL": os.environ["ZAI_BASE_URL"],
})
print(json.dumps(result, ensure_ascii=True))
if result["status"] != "passed":
    sys.exit(2)
PY
fi

exec "${PYTHON_BIN}" -u run.py \
  --agent sure_master \
  --config "${CONFIG_PATH}" \
  --task "${TASK_PATH}" \
  "${RUN_ARGS[@]}"
