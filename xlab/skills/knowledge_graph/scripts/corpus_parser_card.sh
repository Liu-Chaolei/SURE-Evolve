#!/usr/bin/env bash
set -euo pipefail
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
task_root=${1:?pipeline root required}
task_card=${2:?card index required}
task_runtime=/workspace/XLab/.xlab/benchmarks/mineru-ascend-311-20260908/mineru344
task_scripts=/shared/chaolei.liu/XLab/xlab/skills/knowledge_graph/scripts
export MINERU_TOOLS_CONFIG_JSON="$task_runtime/home/mineru.json"
export PYTHONPATH="$task_runtime/venv-pypi-3.4.4/lib/python3.11/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$task_runtime/runtime/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1 MINERU_MODEL_SOURCE=local MINERU_DEVICE_MODE=npu
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MINERU_INTRA_OP_NUM_THREADS=4 MINERU_INTER_OP_NUM_THREADS=1 MINERU_API_MAX_CONCURRENT_REQUESTS=1
export MINERU_VIRTUAL_VRAM_SIZE=12 MINERU_PROCESSING_WINDOW_SIZE=16 MINERU_PDF_RENDER_THREADS=4
export HCCL_OP_EXPANSION_MODE=AIV MINERU_ENABLE_PIPELINE_INFERENCE_LOCKS=true
task_pids=()
for task_lane in 0 1 2 3; do
  task_cache=/local/job/card-$task_card-lane-$task_lane
  mkdir -p "$task_cache"
  ASCEND_CACHE_PATH="$task_cache/ascend-cache" ASCEND_PROCESS_LOG_PATH="$task_cache/ascend-log" \
    PYTHONPYCACHEPREFIX="$task_cache/pycache" XDG_CACHE_HOME="$task_cache/cache" \
    python3 "$task_scripts/corpus_parse_worker.py" --run-dir "$task_root" --worker "$task_card-$task_lane" \
    > "$task_root/parser-$task_card-$task_lane.log" 2>&1 &
  task_pids+=("$!")
done
task_failed=0
for task_pid in "${task_pids[@]}"; do
  wait "$task_pid" || task_failed=1
done
exit "$task_failed"
