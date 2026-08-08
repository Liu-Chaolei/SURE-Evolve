# SURE Master F5-TTS 启动手册

这份手册只保留当前 F5-TTS 自进化会用到的内容：
官方 F5-TTS v1 Base draft、`staged_axes` 三轴自进化、mixed-local
coordinator、远端 VC 候选执行、日志和结果路径。

## 当前入口

推荐使用的 staged 配置：

```text
configs/sure_master/gpt-5-f5tts-staged-axes-mixed.yaml
```

普通 mixed search 仍可使用：

```text
configs/sure_master/gpt-5-f5tts-regular-search-mixed.yaml
```

使用脚本：

```text
docker/sure-master-f5tts/run_mixed_local.sh
```

运行方式：

```text
本地 coordinator：负责 SURE Master / LLM 编排和提交 VC 子任务
本地 inference 候选：使用 coordinator 可见 GPU 执行 batch inference
远端 VC 子任务：负责 fine-tune、arch 训练型候选，每个候选 8 GPU
```

## 前置条件

确认这些路径存在：

```bash
test -f /hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster/.env
test -d /hpc_stor03/sjtu_home/chaolei.liu/TTS/F5-TTS
test -d /hpc_stor03/sjtu_home/chaolei.liu/sure
test -d /hpc_stor03/public/shared/data
test -f /hpc_stor03/sjtu_home/chaolei.liu/models/official_drafts/SWivid-F5-TTS/F5TTS_v1_Base/model_1250000.safetensors
test -f /hpc_stor03/sjtu_home/chaolei.liu/models/official_drafts/SWivid-F5-TTS/F5TTS_v1_Base/vocab.txt
command -v vc
```

`.env` 至少需要：

```bash
OPENAI_API_KEY=...
GPT_BASE_URL=...
GPT_CHAT_MODEL=...
```

F5-TTS runtime 默认离线使用 Hugging Face cache：

```bash
HF_HUB_OFFLINE=1
```

如果缺少 Vocos cache，需要先缓存 `charactr/vocos-mel-24khz`，或者临时设置：

```bash
HF_HUB_OFFLINE=0
```

## Official Draft

F5-TTS draft 固定为官方 `SWivid/F5-TTS` 的 `F5TTS_v1_Base`：

```text
playground/sure_master/baselines/f5tts_v1_base_official_baseline.py
```

checkpoint：

```text
/hpc_stor03/sjtu_home/chaolei.liu/models/official_drafts/SWivid-F5-TTS/F5TTS_v1_Base/model_1250000.safetensors
/hpc_stor03/sjtu_home/chaolei.liu/models/official_drafts/SWivid-F5-TTS/F5TTS_v1_Base/vocab.txt
```

baseline 脚本只调用 `SURE_TTS_BATCH_INFER_WRAPPER`，不训练、不改结构。

已校验过的文件：

```text
model_1250000.safetensors sha256=670900fd14e6c458b95da6e9ed317cdb20dbaf7a1c02ac06a05475a9d32b6a38
vocab.txt                  sha256=2a05f992e00af9b0bd3800a8d23e78d520dbd705284ed2eedb5f4bd29398fa3c
```

## 当前数据

staged 配置使用：

```text
playground/sure_master/data/f5tts_staged/smoke
playground/sure_master/data/f5tts_staged/regular_search
playground/sure_master/data/f5tts_staged/selection
playground/sure_master/data/f5tts_staged/holdout
playground/sure_master/data/f5tts_train_manifests
```

## Staged Axes

当前 F5-TTS staged 配置已经写好，不需要手动复制 YAML：

```text
Stage0 official draft/baseline: smoke 20 rows
Stage1 arch scratch screening: 4 rounds x 4 ideas, 16 -> 8 at 1000 steps, then 8 -> 2 at 2000 steps
Stage2 train strategy: 4 rounds x 4 ideas, 16 -> 8 -> 2
Stage3 inference/decoding: 4 rounds x 4 ideas, 16 -> 8 -> 3
Stage4 limited combination: top-2 arch x top-2 train x top-3 inference = 12, retrained with partial_load at 5000 steps
Stage5 selection rerank: search best + top-5 runner-up + baseline/draft
Stage6 holdout final check: final selected candidate + baseline/draft
```

三条轴的边界：

```text
arch:       只通过 SURE_TTS_ARCH_WRAPPER 改白名单结构字段
fine_tune: 只通过 SURE_TTS_FINETUNE_WRAPPER 改训练策略，不改结构
inference: 只通过 SURE_TTS_BATCH_INFER_WRAPPER 改推理/解码/文本处理
```

arch 初筛阶段会设置 `SURE_TTS_ARCH_INIT_MODE=scratch`，目的是让不同结构在相同
随机初始化条件下比较；这些 scratch checkpoint 只用于筛选结构。组合、selection
和 holdout 会设置 `SURE_TTS_ARCH_INIT_MODE=partial_load`，从官方 F5-TTS checkpoint
加载形状兼容的参数后再训练，最终只比较这类重新训练出来的 checkpoint。staged
配置还会同步设置 `SURE_TTS_ARCH_FORCE_INIT_MODE`，防止候选脚本用 CLI 参数覆盖
当前阶段的初始化策略。

远端调度只覆盖训练型候选：

```yaml
remote_training:
  candidate_types:
    - "training"
```

因此 `fine_tune` 和 `arch` 会提交 VC 子任务，`inference` 留在本地 coordinator
的 `CUDA_VISIBLE_DEVICES` 上执行。

## 启动

后台启动当前 F5-TTS staged 搜索：

```bash
cd /hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster

CONFIG=configs/sure_master/gpt-5-f5tts-staged-axes-mixed.yaml
RUN_NAME=f5tts_staged
RUN_DIR="runs/${RUN_NAME}"
RUN_ID="${RUN_NAME}"
LOG_DIR=/hpc_stor03/sjtu_home/chaolei.liu/log
MASTER_LOG="${LOG_DIR}/sure_master_${RUN_ID}.log"
NOHUP_LOG="${LOG_DIR}/sure_master_${RUN_ID}.nohup.log"

REMOTE_PARTITION=""
REMOTE_PARTITIONS=pdgpu-3090,pdgpu-4090,pdgpu-a10
REMOTE_PARTITION_POLICY=most_free_gpu
REMOTE_PARTITION_FALLBACK=queue_first
REMOTE_GPU_PER_TASK=8
REMOTE_CPU_PER_TASK=64
REMOTE_MEM_PER_TASK=256G
REMOTE_MAX_PARALLEL=4

mkdir -p "${LOG_DIR}"

nohup env \
  SURE_MASTER_CONFIG="${CONFIG}" \
  SURE_MASTER_RUN_DIR="${RUN_DIR}" \
  SURE_MASTER_LOG_FILE="${MASTER_LOG}" \
  SURE_REMOTE_PARTITION="${REMOTE_PARTITION}" \
  SURE_REMOTE_PARTITIONS="${REMOTE_PARTITIONS}" \
  SURE_REMOTE_PARTITION_POLICY="${REMOTE_PARTITION_POLICY}" \
  SURE_REMOTE_PARTITION_FALLBACK="${REMOTE_PARTITION_FALLBACK}" \
  SURE_REMOTE_GPU_PER_TASK="${REMOTE_GPU_PER_TASK}" \
  SURE_REMOTE_CPU_PER_TASK="${REMOTE_CPU_PER_TASK}" \
  SURE_REMOTE_MEM_PER_TASK="${REMOTE_MEM_PER_TASK}" \
  SURE_REMOTE_MAX_PARALLEL="${REMOTE_MAX_PARALLEL}" \
  bash docker/sure-master-f5tts/run_mixed_local.sh \
  > "${NOHUP_LOG}" 2>&1 &

echo "pid=$!"
echo "master_log=${MASTER_LOG}"
echo "nohup_log=${NOHUP_LOG}"
```

本地 inference 现在默认由配置自动选择空闲卡：

```yaml
gpu_devices: idle
idle_gpu_min_free_mib: 9000
idle_gpu_max_utilization: 20
```

含义是通过 `nvidia-smi` 只选择空闲显存不少于 9GB 且 GPU 利用率不高于 20% 的
卡。本地并发数会按实际筛出的卡数自动收缩，避免把 F5-TTS 推理派到已被其他
进程占满的 GPU 上。若想限制候选卡范围，启动前设置：

```bash
SURE_LOCAL_CUDA_VISIBLE_DEVICES=1,2,3
```

若没有满足阈值的本地卡，任务会在 setup 阶段直接报错，而不是退回忙卡继续
OOM。阈值可通过 `SURE_IDLE_GPU_MIN_FREE_MIB` 和
`SURE_IDLE_GPU_MAX_UTILIZATION` 覆盖。

SURE metric 评分也有独立 GPU 调度，不复用候选生成时的卡选择：

```yaml
sure:
  metric_gpu:
    enabled: true
    devices: "idle"
    gpus_per_metric: 1
    min_free_mib: 9500
    max_utilization: 20
    wait_timeout_sec: 900
    poll_interval_sec: 10
    oom_retry: true
    max_retries: 4
    cleanup_before_score: true
```

含义是每次调用 SURE/Whisper 评分前重新用 `nvidia-smi` 找空闲卡，并通过
`/tmp/sure_master_metric_gpu_locks` 做文件锁，避免多个评分任务同时抢同一张卡。
如果评分阶段遇到 CUDA OOM，会释放当前卡、换下一张可用卡重试。每次尝试会写到
`metric/metric_gpu_attempts.json`；失败详情在 `metric/metric_error.json`。

这里故意设置：

```bash
REMOTE_PARTITION=
```

这样框架会在 `SURE_REMOTE_PARTITIONS` 中按 `most_free_gpu` 自动选队列。若写成
`REMOTE_PARTITION=pdgpu-3090`，就会固定使用 `pdgpu-3090`。

## 查看状态

主日志：

```bash
tail -f /hpc_stor03/sjtu_home/chaolei.liu/log/sure_master_f5tts_staged_axes_mixed.log
```

后台包装日志：

```bash
tail -f /hpc_stor03/sjtu_home/chaolei.liu/log/sure_master_f5tts_staged_axes_mixed.nohup.log
```

查 coordinator 进程：

```bash
ps -ef | rg 'sure_master|run_mixed_local|f5tts_staged_axes_mixed'
```

查 VC 子任务：

```bash
vc info
```

## 结果位置

主输出目录：

```text
runs/f5tts_staged_axes_mixed/
```

staged 汇总：

```text
runs/f5tts_staged_axes_mixed/staged_axes/baseline_draft.json
runs/f5tts_staged_axes_mixed/staged_axes/ideas_<axis>.json
runs/f5tts_staged_axes_mixed/staged_axes/leaderboard_<axis>_<rung>.json
runs/f5tts_staged_axes_mixed/staged_axes/top_<axis>.json
runs/f5tts_staged_axes_mixed/staged_axes/leaderboard_combination_search.json
runs/f5tts_staged_axes_mixed/staged_axes/leaderboard_selection.json
runs/f5tts_staged_axes_mixed/staged_axes/summary.json
```

每个候选 workspace 内常用文件：

```text
run_sure.py
artifacts/samples.jsonl
artifacts/wavs/*.wav
artifacts/candidate_changes.json
metric/score_summary.json
metric/remote_training_result.json
```

## 常用覆盖

换输出目录：

```bash
SURE_MASTER_RUN_DIR=runs/f5tts_staged_axes_mixed_v2
```

换成普通 mixed search：

```bash
SURE_MASTER_CONFIG=configs/sure_master/gpt-5-f5tts-regular-search-mixed.yaml
SURE_MASTER_RUN_DIR=runs/f5tts_regular_search_mixed_v2
```

限制远端并发：

```bash
SURE_REMOTE_MAX_PARALLEL=2
```

固定队列：

```bash
SURE_REMOTE_PARTITION=pdgpu-4090
```

临时允许联网下载 cache：

```bash
HF_HUB_OFFLINE=0
```

## 快速排错

`Missing .env`：确认仓库根目录存在 `.env`。

`Missing OPENAI_API_KEY/GPT_BASE_URL/GPT_CHAT_MODEL`：检查 `.env` 变量名。

`vc command is required`：需要在能调用 `vc submit` 的机器上启动 coordinator。

`Missing F5-TTS source directory`：检查：

```text
/hpc_stor03/sjtu_home/chaolei.liu/TTS/F5-TTS
```

`Missing cached Vocos model while HF_HUB_OFFLINE=1`：先缓存
`charactr/vocos-mel-24khz`，或在启动命令里加 `HF_HUB_OFFLINE=0`。

`model_1250000.safetensors` 不存在：检查官方 draft checkpoint 路径，或重新下载
`SWivid/F5-TTS` 的 `F5TTS_v1_Base` 文件。
