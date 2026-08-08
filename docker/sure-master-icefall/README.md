# SURE Master Icefall ASR 启动手册

这份手册只保留当前要用的 ASR staged self-evolution 启动流程：
官方 Zipformer draft + `staged_axes` 三轴自进化 + mixed-local coordinator。

## 当前入口

使用配置：

```text
configs/sure_master/gpt-5-icefall-staged-axes-mixed.yaml
```

使用脚本：

```text
docker/sure-master-icefall/run_mixed_local.sh
```

运行方式：

```text
本地 coordinator：负责 SURE Master / LLM 编排和提交 VC 子任务
远端 VC 子任务：负责 arch/train 等训练型候选，每个候选 8 GPU
inference 候选：按配置和候选类型由框架调度
```

## 前置条件

确认这些路径存在：

```bash
test -f /hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster/.env
test -d /hpc_stor03/sjtu_home/chaolei.liu/ASR/icefall
test -d /hpc_stor03/sjtu_home/chaolei.liu/sure
test -f /hpc_stor03/sjtu_home/chaolei.liu/models/official_drafts/icefall-asr-librispeech-zipformer-large-transducer-with-CR-CTC-20241019/exp/pretrained.pt
test -f /hpc_stor03/sjtu_home/chaolei.liu/models/official_drafts/icefall-asr-librispeech-zipformer-large-transducer-with-CR-CTC-20241019/exp/epoch-50.pt
command -v vc
```

`.env` 至少需要：

```bash
OPENAI_API_KEY=...
GPT_BASE_URL=...
GPT_CHAT_MODEL=...
```

## Official Draft

ASR draft 固定为 icefall 官方 Zipformer large CR-CTC-RNNT baseline：

```text
playground/sure_master/baselines/zipformer_large_cr_ctc_rnnt_baseline.py
```

checkpoint 目录：

```text
/hpc_stor03/sjtu_home/chaolei.liu/models/official_drafts/icefall-asr-librispeech-zipformer-large-transducer-with-CR-CTC-20241019/exp
```

当前配置使用：

```yaml
SURE_BASELINE_USE_PRETRAINED: "1"
```

也就是 Stage0 baseline decode 使用官方包里的 `pretrained.pt`。`epoch-50.pt`
保留用于复现和后续训练。

已校验过的 checkpoint：

```text
pretrained.pt sha256=2ca2bf48b5ae52de95402749df9756da28b50f85d75933e361f135ceb4290026
epoch-50.pt   sha256=11a0a03fc768125266e50bb55a9bae766f83e532daa93700ffee78a6db4ffd95
```

## Staged Axes

当前 ASR staged 配置已经写好，不需要手动复制 YAML：

```text
Stage0 official draft/baseline
Stage1 arch: 4 rounds x 4 ideas, 16 -> 8 -> 2
Stage2 train strategy: 4 rounds x 4 ideas, 16 -> 8 -> 2
Stage3 inference/decoding: 4 rounds x 4 ideas, 16 -> 8 -> 3
Stage4 limited combination: top-2 arch x top-2 train x top-3 inference = 12
Stage5 selection rerank: search best + top-5 runner-up + baseline/draft
```

核心配置在：

```text
configs/sure_master/gpt-5-icefall-staged-axes-mixed.yaml
```

Stage0 official pretrained draft 会固定使用 decode duration，并关闭训练
duration autotune：

```yaml
staged_axes:
  draft:
    execution_env:
      SURE_MAX_DURATION: "300"
      SURE_DURATION_AUTOTUNE: "0"
```

## 启动

后台启动：

```bash
cd /hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster

CONFIG=configs/sure_master/gpt-5-icefall-staged-axes-mixed.yaml
RUN_NAME=zipformer_staged
LOG_DIR=/hpc_stor03/sjtu_home/chaolei.liu/log
MASTER_LOG="${LOG_DIR}/sure_master_${RUN_NAME}.log"
NOHUP_LOG="${LOG_DIR}/sure_master_${RUN_NAME}.nohup.log"

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
  SURE_MASTER_RUN_NAME="${RUN_NAME}" \
  SURE_MASTER_LOG_FILE="${MASTER_LOG}" \
  SURE_REMOTE_PARTITION="${REMOTE_PARTITION}" \
  SURE_REMOTE_PARTITIONS="${REMOTE_PARTITIONS}" \
  SURE_REMOTE_PARTITION_POLICY="${REMOTE_PARTITION_POLICY}" \
  SURE_REMOTE_PARTITION_FALLBACK="${REMOTE_PARTITION_FALLBACK}" \
  SURE_REMOTE_GPU_PER_TASK="${REMOTE_GPU_PER_TASK}" \
  SURE_REMOTE_CPU_PER_TASK="${REMOTE_CPU_PER_TASK}" \
  SURE_REMOTE_MEM_PER_TASK="${REMOTE_MEM_PER_TASK}" \
  SURE_REMOTE_MAX_PARALLEL="${REMOTE_MAX_PARALLEL}" \
  bash docker/sure-master-icefall/run_mixed_local.sh \
  > "${NOHUP_LOG}" 2>&1 &

echo "pid=$!"
echo "master_log=${MASTER_LOG}"
echo "nohup_log=${NOHUP_LOG}"
```

如果要切换到 TEDLIUM3 / 新数据集 ASR 自进化，启动方式不变，只替换配置和 run 名称：

```bash
cd /hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster

CONFIG=configs/sure_master/gpt-5-icefall-tedlium3-staged-axes-mixed.yaml
RUN_NAME=zipformer_tedlium3_staged
LOG_DIR=/hpc_stor03/sjtu_home/chaolei.liu/log
MASTER_LOG="${LOG_DIR}/sure_master_${RUN_NAME}.log"
NOHUP_LOG="${LOG_DIR}/sure_master_${RUN_NAME}.nohup.log"

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
  SURE_MASTER_RUN_NAME="${RUN_NAME}" \
  SURE_MASTER_LOG_FILE="${MASTER_LOG}" \
  SURE_REMOTE_PARTITION="${REMOTE_PARTITION}" \
  SURE_REMOTE_PARTITIONS="${REMOTE_PARTITIONS}" \
  SURE_REMOTE_PARTITION_POLICY="${REMOTE_PARTITION_POLICY}" \
  SURE_REMOTE_PARTITION_FALLBACK="${REMOTE_PARTITION_FALLBACK}" \
  SURE_REMOTE_GPU_PER_TASK="${REMOTE_GPU_PER_TASK}" \
  SURE_REMOTE_CPU_PER_TASK="${REMOTE_CPU_PER_TASK}" \
  SURE_REMOTE_MEM_PER_TASK="${REMOTE_MEM_PER_TASK}" \
  SURE_REMOTE_MAX_PARALLEL="${REMOTE_MAX_PARALLEL}" \
  bash docker/sure-master-icefall/run_mixed_local.sh \
  > "${NOHUP_LOG}" 2>&1 &

echo "pid=$!"
echo "master_log=${MASTER_LOG}"
echo "nohup_log=${NOHUP_LOG}"
```

## 查看状态

主日志：

```bash
tail -f /hpc_stor03/sjtu_home/chaolei.liu/log/sure_master_zipformer_staged_axes.log
```

后台包装日志：

```bash
tail -f /hpc_stor03/sjtu_home/chaolei.liu/log/sure_master_zipformer_staged_axes.nohup.log
```

查 coordinator 进程：

```bash
ps -ef | rg 'sure_master|run_mixed_local|zipformer_staged_axes'
```

查 VC 子任务：

```bash
vc info
```

## 结果位置

主输出目录：

```text
runs/zipformer_staged_axes/
```

staged 汇总：

```text
runs/zipformer_staged_axes/staged_axes/baseline_draft.json
runs/zipformer_staged_axes/staged_axes/ideas_<axis>.json
runs/zipformer_staged_axes/staged_axes/leaderboard_<axis>_<rung>.json
runs/zipformer_staged_axes/staged_axes/top_<axis>.json
runs/zipformer_staged_axes/staged_axes/leaderboard_combination_search.json
runs/zipformer_staged_axes/staged_axes/leaderboard_selection.json
runs/zipformer_staged_axes/staged_axes/summary.json
```

每个候选 workspace 内常用文件：

```text
run_sure.py
artifacts/candidate_status.json
artifacts/hyp.txt
metric/score_summary.json
metric/remote_training_result.json
metric/cleanup_manifest.json
metric/log_tails/*.tail.log
```

`artifacts/candidate_status.json` 会记录 `reason_code`、stage/phase/rung、
candidate type、metric 是否被接受、缺失 artifact 和 artifact guard 错误。分析
失败候选时优先看这个文件，再看 `metric/log_tails/*.tail.log` 和
`metric/remote_training_result.json`。

默认配置开启 `sure.workspace_cleanup`，候选结束后会删除当前候选 workspace 内的
`models/`、`working/`、`.sure_runtime/` 和完整 `metric/*.log`，只保留评分、
状态、hyp、远端结果 JSON 和日志尾部，避免每轮 Zipformer 自进化堆积大量
checkpoint。如果需要保留完整候选产物调试，可以临时关闭：

```bash
SURE_WORKSPACE_CLEANUP_ENABLED=0 bash docker/sure-master-icefall/run_task.sh
```

清理已有 run 目录时先 dry-run：

```bash
python playground/sure_master/tools/cleanup_sure_workspaces.py --root runs/zipformer_staged_axes
```

确认列表后执行：

```bash
python playground/sure_master/tools/cleanup_sure_workspaces.py --root runs/zipformer_staged_axes --apply
```

## 本地 GPU 调度

mixed-local 启动时，draft baseline decode 和 inference 类候选会在本地 GPU 上跑；
训练类候选才会提交 VC。ASR mixed 配置默认使用 `gpu_devices: idle`：

```yaml
session:
  local:
    gpu_devices: idle
    idle_gpu_min_free_mib: 10000
    idle_gpu_max_utilization: 20
    idle_gpu_allow_busy_fallback: false
    gpu_lock_enabled: true
```

启动时会筛掉忙 GPU；每次本地候选真正执行前还会再次刷新 `nvidia-smi`，
如果原 GPU 变忙就重新选择空闲 GPU。GPU 文件锁位于
`/tmp/sure_master_gpu_locks`，用于避免多个 sure_master 进程抢同一张本地卡。

常用覆盖：

```bash
SURE_IDLE_GPU_MIN_FREE_MIB=10000
SURE_IDLE_GPU_MAX_UTILIZATION=20
SURE_IDLE_GPU_ALLOW_BUSY_FALLBACK=0
SURE_GPU_LOCK_WAIT_SECONDS=30
```

## 常用覆盖

换 run 名称：

```bash
SURE_MASTER_RUN_NAME=zipformer_staged_axes_v2
```

限制远端并发：

```bash
SURE_REMOTE_MAX_PARALLEL=2
```

固定队列：

```bash
SURE_REMOTE_PARTITION=pdgpu-3090
```

增加候选队列：

```bash
SURE_REMOTE_PARTITIONS=pdgpu-3090,pdgpu-4090,pdgpu-a10
```

## 快速排错

`Missing .env`：确认仓库根目录存在 `.env`，且不是只存在于登录 shell。

`Missing OPENAI_API_KEY/GPT_BASE_URL/GPT_CHAT_MODEL`：检查 `.env` 变量名。

`vc command is required`：需要在能调用 `vc submit` 的机器上启动 coordinator。

`local icefall Python is not executable`：检查：

```text
/hpc_stor03/sjtu_home/chaolei.liu/anaconda3/envs/icefall/bin/python
```

`pretrained.pt` 不存在：重新下载官方 draft checkpoint，或检查
`SURE_BASELINE_CHECKPOINT_DIR` 是否指向 `exp/` 目录。
