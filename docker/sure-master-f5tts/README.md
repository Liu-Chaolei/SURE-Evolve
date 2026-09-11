# SURE Master F5-TTS 启动手册

这份手册只保留当前 F5-TTS 自进化会用到的内容：
官方 F5-TTS v1 Base draft、`staged_axes` 三轴自进化、CPU-only coordinator、
全远端 VC 候选执行、日志和结果路径。脚本名沿用 `run_mixed_local.sh`，但当前
production 配置默认不是 mixed-local GPU 执行。

## 当前入口

中文 Premium 正式结构搜索使用：

```text
configs/sure_master/tts-zh-zai-vc.yaml
```

该配置使用 `tts_zh_cer`、Seed-TTS zh holdout、ZAI `glm-5.3-flash` 和 8-GPU
F5-TTS DDP 完整训练。先按 extracted-only 模式准备数据：

```bash
python playground/sure_master/tools/prepare_task_data.py tts \\
  --root /hpc_stor03/sjtu_home/chaolei.liu/data/datasets/data/tts/WenetSpeech4TTS/Premium \\
  --seed-root /hpc_stor03/sjtu_home/chaolei.liu/data/datasets/seed-tts-eval \\
  --output /hpc_stor03/sjtu_home/chaolei.liu/data/sure_premium_f5tts \\
  --source-mode extracted_only
```

正式运行时设置 `SURE_MASTER_CONFIG` 指向该配置；不会执行 smoke 阶段。

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
本地 coordinator：CPU-only，负责 SURE Master / LLM 编排和提交 VC 子任务
远端 inference：每个 VC child job 使用 1 GPU / 8 CPU / 32G
远端 fine_tune、arch 及 train-from-scratch draft：每个 VC child job 使用 8 GPU / 64 CPU / 256G
SURE metric：在对应 VC child job 内运行
```

staged 和 regular-search production YAML 均设置
`coordinator.local_gpu_policy: disabled`、`remote_training.draft_enabled: true`，并完整
覆盖 `inference`、`fine_tune`、`arch`。`remote_training.max_parallel` 表示并行 VC
job 数，不是 GPU 数。

## 前置条件

确认 coordinator 所需路径存在：

```bash
REPO_DIR=/hpc_stor03/sjtu_home/chaolei.liu/Agent/SURE-Evolve

test -f "${REPO_DIR}/.env"
test -d /hpc_stor03/sjtu_home/chaolei.liu/sure
test -x /hpc_stor03/sjtu_home/chaolei.liu/anaconda3/envs/suremaster-f5tts-local/bin/python
command -v vc
```

全远端模式下 coordinator 不要求本地 GPU、`nvidia-smi`、F5-TTS source/runtime、
F5-TTS Python 或本地 Vocos cache。F5-TTS、CUDA、数据和以下 checkpoint 必须在 VC
child image 或挂载路径内可读：

```bash
test -d /hpc_stor03/public/shared/data
test -f /hpc_stor03/sjtu_home/chaolei.liu/models/official_drafts/SWivid-F5-TTS/F5TTS_v1_Base/model_1250000.safetensors
test -f /hpc_stor03/sjtu_home/chaolei.liu/models/official_drafts/SWivid-F5-TTS/F5TTS_v1_Base/vocab.txt
```

当前仓库和 launcher/config 默认路径均为 `SURE-Evolve`；通常无需覆盖
`SURE_MASTER_REPO_DIR`。迁移仓库时，YAML 中的数据、wrapper、VC workdir 和 `.env`
路径必须同步调整。启动前可扫描：

```bash
rg -n '/Agent/(EvoMaster|SURE-Evolve-old)' \
  configs/sure_master/gpt-5-f5tts-staged-axes-mixed.yaml \
  configs/sure_master/gpt-5-f5tts-regular-search-mixed.yaml
```

所选 launcher/YAML 不应包含旧仓库路径。

`.env` 至少需要 OpenAI-compatible 配置之一：

```bash
OPENAI_API_KEY=...
GPT_BASE_URL=...
GPT_CHAT_MODEL=...
```

也可以只配置 ZAI 兼容端点；launcher 会将其映射到 coordinator 使用的
`OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `SURE_AGENT_MODEL`：

```bash
ZAI_API_KEY=...
ZAI_BASE_URL=...
SURE_AGENT_MODEL=glm-5.3-flash
```

F5-TTS child runtime 默认离线使用 Hugging Face cache：

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

远端调度覆盖所有候选类型：

```yaml
coordinator:
  local_gpu_policy: disabled
remote_training:
  draft_enabled: true
  candidate_types:
    - inference
    - fine_tune
    - arch
```

因此 draft、`inference`、`fine_tune` 和 `arch` 均提交 VC child；profile 按类型提供
1-GPU inference 或 8-GPU training 资源。

## 启动

后台启动当前 F5-TTS staged 搜索：

```bash
cd /hpc_stor03/sjtu_home/chaolei.liu/Agent/SURE-Evolve

REPO_DIR=/hpc_stor03/sjtu_home/chaolei.liu/Agent/SURE-Evolve
CONFIG=configs/sure_master/gpt-5-f5tts-staged-axes-mixed.yaml
RUN_NAME=f5tts_staged
RUN_DIR="runs/${RUN_NAME}"
LOG_DIR=/hpc_stor03/sjtu_home/chaolei.liu/log
MASTER_LOG="${LOG_DIR}/sure_master_${RUN_NAME}.log"
NOHUP_LOG="${LOG_DIR}/sure_master_${RUN_NAME}.nohup.log"

REMOTE_PARTITION=""
REMOTE_PARTITIONS=pdgpu-3090,pdgpu-4090,pdgpu-a10
REMOTE_PARTITION_POLICY=most_free_gpu
REMOTE_PARTITION_FALLBACK=queue_first
REMOTE_MAX_PARALLEL=4

mkdir -p "${LOG_DIR}"

nohup env \
  SURE_MASTER_REPO_DIR="${REPO_DIR}" \
  SURE_MASTER_CONFIG="${CONFIG}" \
  SURE_MASTER_RUN_DIR="${RUN_DIR}" \
  SURE_MASTER_LOG_FILE="${MASTER_LOG}" \
  SURE_REMOTE_PARTITION="${REMOTE_PARTITION}" \
  SURE_REMOTE_PARTITIONS="${REMOTE_PARTITIONS}" \
  SURE_REMOTE_PARTITION_POLICY="${REMOTE_PARTITION_POLICY}" \
  SURE_REMOTE_PARTITION_FALLBACK="${REMOTE_PARTITION_FALLBACK}" \
  SURE_REMOTE_MAX_PARALLEL="${REMOTE_MAX_PARALLEL}" \
  bash docker/sure-master-f5tts/run_mixed_local.sh \
  > "${NOHUP_LOG}" 2>&1 &

PID=$!
echo "pid=${PID}"
echo "run_dir=${RUN_DIR}"
echo "master_log=${MASTER_LOG}"
echo "nohup_log=${NOHUP_LOG}"
```

这里使用 `SURE_MASTER_RUN_DIR=runs/f5tts_staged`，所以它作为显式 path-like 值
按 launcher/CLI 语义直接使用，不追加时间戳（从仓库根目录启动时即
`<REPO_DIR>/runs/f5tts_staged`）。若改用 `SURE_MASTER_RUN_NAME=f5tts_staged`，launcher
会把裸名称传给 `--run-dir`，由 CLI 解析到默认 run root 并追加时间戳。两者不能同时
设置；若都不设置，则使用 CLI 自动生成的默认目录。

`SURE_MASTER_CONFIG` 可省略，因为 launcher 当前默认即为
`gpt-5-f5tts-staged-axes-mixed.yaml`；切换 regular search 时再显式覆盖。

不要在推荐启动命令中设置 `SURE_REMOTE_GPU_PER_TASK`、
`SURE_REMOTE_CPU_PER_TASK` 或 `SURE_REMOTE_MEM_PER_TASK`：全局覆盖会压过 YAML 的
inference/training profiles。`SURE_REMOTE_MAX_PARALLEL=4` 表示最多并行 4 个 VC
child job。

如果 coordinator Python 不在默认环境中，可在 `nohup env` 后增加：

```bash
SURE_MASTER_PYTHON=/path/to/coordinator/python \
```

coordinator Python 只需承载编排依赖；全远端模式不使用本地 F5-TTS Python。VC
scheduler 注入的 `CUDA_VISIBLE_DEVICES` 是 child 内 GPU 可见性的权威来源，不要从
coordinator 传入或在 wrapper 中覆盖。

如确需 mixed-local，这是显式 opt-in：修改 `coordinator.local_gpu_policy`，并从
`remote_training.candidate_types`（以及按需的 draft routing）移除要留在本地的类型，
再配置本地 runtime、session 和 GPU 选择。仅设置 `SURE_LOCAL_CUDA_VISIBLE_DEVICES`
不会改变当前全远端路由。

这里故意设置：

```bash
REMOTE_PARTITION=
```

这样框架会在 `SURE_REMOTE_PARTITIONS` 中按 `most_free_gpu` 自动选队列。若写成
`REMOTE_PARTITION=pdgpu-3090`，就会固定使用 `pdgpu-3090`。

## 查看状态

以上面的 `RUN_NAME=f5tts_staged` 为例。

主日志：

```bash
tail -f /hpc_stor03/sjtu_home/chaolei.liu/log/sure_master_f5tts_staged.log
```

后台包装日志：

```bash
tail -f /hpc_stor03/sjtu_home/chaolei.liu/log/sure_master_f5tts_staged.nohup.log
```

查 coordinator 进程：

```bash
ps -ef | rg 'sure_master|run_mixed_local|f5tts_staged'
```

查 VC 子任务：

```bash
vc info
```

## 结果位置

主输出目录（相对于仓库根目录）：

```text
runs/f5tts_staged/
```

staged 汇总：

```text
runs/f5tts_staged/staged_axes/baseline_draft.json
runs/f5tts_staged/staged_axes/ideas_<axis>.json
runs/f5tts_staged/staged_axes/leaderboard_<axis>_<rung>.json
runs/f5tts_staged/staged_axes/top_<axis>.json
runs/f5tts_staged/staged_axes/leaderboard_combination_search.json
runs/f5tts_staged/staged_axes/leaderboard_selection.json
runs/f5tts_staged/staged_axes/summary.json
```

每个候选 workspace 内常用文件：

```text
run_sure.py
artifacts/samples.jsonl
artifacts/wavs/*.wav
artifacts/candidate_changes.json
working/batch_infer/diagnostics.json
metric/score_summary.json
metric/remote_training_result.json
metric/metric_gpu_attempts.json
metric/metric_error.json
```

`artifacts/candidate_changes.json` 记录 F5 adaptation 的实际结构、训练或 inference
改动；`working/batch_infer/diagnostics.json` 记录 shard/worker 推理诊断。metric 尝试和
错误文件由 child 内评分生成；启用 workspace cleanup 时，`working/` 等大产物可能在
候选结束后被清理，应结合保留的 metric/remote result 与日志尾部排查。

## 常用覆盖

换输出目录：

```bash
SURE_MASTER_RUN_DIR=runs/f5tts_staged_v2
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

`Missing .env`：确认 `SURE_MASTER_REPO_DIR` 指向当前仓库根目录，且该目录中
存在 `.env`。

若 launcher 无法进入仓库：确认 `SURE_MASTER_REPO_DIR` 指向：

```bash
SURE_MASTER_REPO_DIR=/hpc_stor03/sjtu_home/chaolei.liu/Agent/SURE-Evolve
```

VC 子任务找不到数据、wrapper、workdir 或 `.env`：检查所选 YAML 中是否仍有旧路径：

```bash
rg -n '/Agent/(EvoMaster|SURE-Evolve-old)' "${CONFIG}"
```

`Missing OPENAI_API_KEY/OPENAI_BASE_URL/SURE_AGENT_MODEL`：检查 `.env`；如果使用 ZAI，确认 `ZAI_API_KEY` 和 `ZAI_BASE_URL` 已配置。

`Python is not executable` 或运行时依赖导入失败：使用
`SURE_MASTER_PYTHON` 指定 coordinator Python。

child 内 CUDA 卡号异常：不要从 coordinator 传入或在候选中重写
`CUDA_VISIBLE_DEVICES`；以 VC scheduler 注入的值为准。

`vc command is required`：需要在能调用 `vc submit` 的机器上启动 coordinator。

远端 child 报 F5-TTS source/runtime 或 Vocos cache 缺失：检查 production image、
挂载路径和 child 的 `HF_HUB_OFFLINE` 设置；全远端模式不要求 coordinator 本地安装
F5-TTS。

`model_1250000.safetensors` 不存在：检查官方 draft checkpoint 路径，或重新下载
`SWivid/F5-TTS` 的 `F5TTS_v1_Base` 文件。
