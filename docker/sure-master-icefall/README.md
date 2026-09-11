# SURE Master Icefall ASR 启动手册

当前 TEDLIUM3 CUDA + VC 入口为
`configs/sure_master/zai-tedlium3-cuda-vc-evolution.yaml`。训练候选统一申请 8
张 GPU；提交前从 `vc info` 的配置 partition 列表中选择空闲 GPU 数不少于 8 的
partition，具体 GPU 由 VC scheduler 分配。coordinator 只需要本地 CPU 和 ZAI API
访问，模型训练、解码及 SURE 评分均在 VC child 中完成。

这份手册同时保留旧的 staged 配置说明；新的 TEDLIUM3 CUDA 入口使用 ordinary + XLab
architecture-only search。脚本名沿用 `run_mixed_local.sh`，但 coordinator 默认不占用本地 GPU。

## 当前入口

旧 staged 配置（二选一）：

```text
# LibriSpeech 官方预训练 draft
configs/sure_master/gpt-5-icefall-staged-axes-mixed.yaml

# TEDLIUM3 从训练开始的 draft
configs/sure_master/gpt-5-icefall-tedlium3-staged-axes-mixed.yaml
```

使用脚本：

```text
docker/sure-master-icefall/run_mixed_local.sh
```

当前 TEDLIUM3 CUDA 运行直接使用：

```bash
cd /hpc_stor03/sjtu_home/chaolei.liu/SURE-Evolve
SURE_MASTER_CONFIG=configs/sure_master/zai-tedlium3-cuda-vc-evolution.yaml \
SURE_MASTER_RUN_NAME=tedlium3_cuda_vc \
bash docker/sure-master-icefall/run_mixed_local.sh
```

首次运行前在 host 上创建完整数据视图（不会修改或复制原始特征）：

```bash
PYTHONPATH=/hpc_stor03/sjtu_home/chaolei.liu/SURE-Evolve \
/hpc_stor03/sjtu_home/chaolei.liu/anaconda3/envs/icefall/bin/python \
-m playground.sure_master.tools.prepare_tedlium \
--corpus /hpc_stor03/sjtu_home/chaolei.liu/data/datasets/data/asr/rawdata/TEDLIUM_release-3 \
--output /hpc_stor03/sjtu_home/chaolei.liu/data/sure_tedlium3_full \
--train-hours 0 --jobs 1 \
--reuse-feature-data /hpc_stor03/sjtu_home/chaolei.liu/data/datasets/data/asr/am/icefall/tedlium3 \
--reuse-bpe-dir /hpc_stor03/sjtu_home/chaolei.liu/ASR/icefall/egs/tedlium3/ASR/data/lang_bpe_500
```

该步骤复用 central lowercase train/dev/test 特征和 recipe 的 500-token BPE，仅生成
SURE refs、软链接及带数据谱系的 `preparation.json`。不指定两个 `--reuse-*` 参数时才会
从原始 SPH 重新提取约 50G 特征。

训练候选默认申请 8 GPU；`remote_training.partitions` 中的候选队列由每次
VC 提交前的 `vc info` 空闲 GPU 数动态排序。需要限制队列时可设置
`SURE_REMOTE_PARTITIONS`，但不要设置 `SURE_REMOTE_GPU_PER_TASK`，以免覆盖
8-GPU training profile。

运行方式：

```text
本地 coordinator：CPU-only，负责 SURE Master / LLM 编排和提交 VC 子任务
远端 inference：每个 VC child job 使用 1 GPU / 8 CPU / 12G
远端 fine_tune、arch 及需从头训练的 draft：每个 VC child job 使用 8 GPU / 64 CPU / 96G
SURE metric：在对应 VC child job 内运行
```

当前两份 production YAML 均设置 `coordinator.local_gpu_policy: disabled`，并以
`remote_training.draft_enabled: true` 和完整的 `inference`、`fine_tune`、`arch`
远端覆盖运行；`remote_training.max_parallel` 表示同时运行的 VC job 数，不是 GPU 数。

## 前置条件

确认 coordinator 所需路径存在：

```bash
REPO_DIR=/hpc_stor03/sjtu_home/chaolei.liu/SURE-Evolve

test -f "${REPO_DIR}/.env"
test -d /hpc_stor03/sjtu_home/chaolei.liu/sure
test -x /hpc_stor03/sjtu_home/chaolei.liu/anaconda3/envs/suremaster-f5tts-local/bin/python
command -v vc
```

XLab idea generation 需要单独的 Python 环境并安装
`playground/sure_master/requirements-xlab.txt`（至少可导入
`sentence_transformers` 和 `faiss`）。默认使用 `SURE_MASTER_PYTHON`；也可通过
`XLAB_PYTHON=/path/to/xlab-python` 指定该环境。

全远端模式下 coordinator 不要求本地 GPU、`nvidia-smi`、Icefall runtime 或
Icefall Python；Icefall、CUDA、数据和 checkpoint 的可用性由 VC child image 及其
挂载路径保证。LibriSpeech 官方 draft checkpoint 仍须可由远端 child 读取：

```bash
test -f /hpc_stor03/sjtu_home/chaolei.liu/models/official_drafts/icefall-asr-librispeech-zipformer-large-transducer-with-CR-CTC-20241019/exp/pretrained.pt
test -f /hpc_stor03/sjtu_home/chaolei.liu/models/official_drafts/icefall-asr-librispeech-zipformer-large-transducer-with-CR-CTC-20241019/exp/epoch-50.pt
```

当前仓库和 launcher/config 默认路径均为 `SURE-Evolve`；通常无需覆盖
`SURE_MASTER_REPO_DIR`，但迁移仓库时应让它与所选 YAML 中的
`remote_training.workdir`、`sure.inputs.ref` 和 wrapper 路径一致。可以这样扫描旧路径：

```bash
rg -n '/Agent/(EvoMaster|SURE-Evolve-old)' \
  docker/sure-master-icefall/run_mixed_local.sh \
  configs/sure_master/gpt-5-icefall-staged-axes-mixed.yaml \
  configs/sure_master/gpt-5-icefall-tedlium3-staged-axes-mixed.yaml
```

所选 launcher/YAML 不应包含旧仓库路径。

当前 launcher 的 `.env` 只需要：

```bash
ZAI_API_KEY=...
ZAI_BASE_URL=...
```

launcher 会在 coordinator 进程内将它们映射为 OpenAI-compatible 环境变量；不会
把 key 注入 VC training child。

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

## Historical Staged Reference

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

旧的 staged 配置和 smoke 流程仅作为历史参考保留，不适用于当前 TEDLIUM3 CUDA
8-GPU ordinary search。当前运行命令见本文开头的 `zai-tedlium3-cuda-vc-evolution.yaml`
示例；它会直接提交完整 baseline 和候选训练。

`SURE_REMOTE_PARTITION=""` 表示不固定单一队列。空值会被忽略，框架会从
`SURE_REMOTE_PARTITIONS` 中按照 `most_free_gpu` 选择；查询失败时按照
`queue_first` 回退到列表中的首个队列。

不要在推荐启动命令中设置 `SURE_REMOTE_GPU_PER_TASK`、
`SURE_REMOTE_CPU_PER_TASK` 或 `SURE_REMOTE_MEM_PER_TASK`：这些全局覆盖会压过 YAML
中的 inference/training resource profiles，使 1-GPU inference 也被错误申请为训练规格。
`SURE_REMOTE_MAX_PARALLEL=4` 表示最多并行 4 个 VC child job。

如果 coordinator Python 不在脚本默认环境中，可在 `nohup env` 后增加：

```bash
SURE_MASTER_PYTHON=/path/to/coordinator/python \
```

该 Python 环境需要能够导入 `evomaster`、`openai`、`sure_eval` 和 `yaml`。

如果需要固定、不带时间戳的结果目录，应使用 `SURE_MASTER_RUN_DIR` 代替
`SURE_MASTER_RUN_NAME`，两者不能同时设置。

## 查看状态

假设启动时使用：

```bash
RUN_NAME=zipformer_tedlium3_staged
```

主日志：

```bash
tail -f /hpc_stor03/sjtu_home/chaolei.liu/log/sure_master_zipformer_tedlium3_staged.log
```

后台包装日志：

```bash
tail -f /hpc_stor03/sjtu_home/chaolei.liu/log/sure_master_zipformer_tedlium3_staged.nohup.log
```

查 coordinator 进程：

```bash
ps -ef | rg 'sure_master|run_mixed_local|zipformer_tedlium3_staged'
```

查 VC 子任务：

```bash
vc info
```

## 结果位置

使用 `SURE_MASTER_RUN_NAME=zipformer_tedlium3_staged` 时，裸 run 名称会被解析到
默认实验根目录并自动追加启动时间：

```text
/hpc_stor03/sjtu_home/chaolei.liu/data/experiments/SURE-Evolve/runs/zipformer_tedlium3_staged_<YYYYMMDD_HHMMSS>/
```

以下用 `<RUN_DIR>` 表示该次实际运行目录。启动后可定位最新目录：

```bash
RUN_ROOT=/hpc_stor03/sjtu_home/chaolei.liu/data/experiments/SURE-Evolve/runs
ls -dt "${RUN_ROOT}"/zipformer_tedlium3_staged_* | head -n 1
```

staged 汇总：

```text
<RUN_DIR>/staged_axes/baseline_draft.json
<RUN_DIR>/staged_axes/ideas_<axis>.json
<RUN_DIR>/staged_axes/leaderboard_<axis>_<rung>.json
<RUN_DIR>/staged_axes/top_<axis>.json
<RUN_DIR>/staged_axes/leaderboard_combination_search.json
<RUN_DIR>/staged_axes/leaderboard_selection.json
<RUN_DIR>/staged_axes/summary.json
```

每个候选 workspace 内常用文件：

```text
run_sure.py
artifacts/candidate_status.json
artifacts/candidate_changes.json
artifacts/hyp.txt
metric/score_summary.json
metric/remote_training_result.json
metric/cleanup_manifest.json
metric/log_tails/*.tail.log
```

`artifacts/candidate_changes.json` 是 ASR adaptation/diagnostic 记录，包含实际结构、训练、
decode 参数、默认值差异、训练/解码尝试和产物路径；排查“候选是否真正应用改动”时应与
`candidate_status.json` 一起查看。`artifacts/candidate_status.json` 会记录 `reason_code`、stage/phase/rung、
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
python playground/sure_master/tools/cleanup_sure_workspaces.py --root "<RUN_DIR>"
```

确认列表后执行：

```bash
python playground/sure_master/tools/cleanup_sure_workspaces.py --root "<RUN_DIR>" --apply
```

## 本地与远程 GPU 调度

当前 production 配置是全远端模式：`coordinator.local_gpu_policy: disabled`，draft
以及 `inference`、`fine_tune`、`arch` 都提交 VC。coordinator 只使用 CPU，不探测、
选择或锁定本地 GPU；SURE metric 也在每个 child 内完成。VC scheduler 分配的
`CUDA_VISIBLE_DEVICES` 是 child 内 GPU 可见性的权威来源，候选或 wrapper 不应以登录
节点卡号覆盖它。

如确需 mixed-local，这是显式 opt-in：把 `coordinator.local_gpu_policy` 改为所需
本地策略，并从 `remote_training.candidate_types`（以及按需的 draft routing）移除要
留在本地的类型，同时配置本地 session/GPU 选择。仅设置
`SURE_LOCAL_CUDA_VISIBLE_DEVICES` 不会改变全远端路由。

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

`Missing .env`：确认 `SURE_MASTER_REPO_DIR` 指向当前仓库根目录，且该目录中
存在 `.env`；脚本不会只依赖登录 shell 中已有的变量。

若 launcher 无法进入仓库：确认 `SURE_MASTER_REPO_DIR` 指向：

```bash
SURE_MASTER_REPO_DIR=/hpc_stor03/sjtu_home/chaolei.liu/SURE-Evolve
```

VC 子任务找不到 runner、ref 或 wrapper：检查所选 YAML 中是否仍有旧路径：

```bash
rg -n '/Agent/(EvoMaster|SURE-Evolve-old)' "${CONFIG}"
```

`Missing ZAI_API_KEY/ZAI_BASE_URL`：检查 `.env` 变量名。

`Python is not executable` 或运行时依赖导入失败：通过 `SURE_MASTER_PYTHON`
指定 coordinator Python，并确认该环境能导入 `evomaster`、`openai`、
`sure_eval` 和 `yaml`。

`vc command is required`：需要在能调用 `vc submit` 的机器上启动 coordinator。

child 内 CUDA 卡号异常：不要从 coordinator 传入或在候选中重写
`CUDA_VISIBLE_DEVICES`；以 VC scheduler 注入的值为准。

远端 child 报 Icefall runtime 或 checkpoint 缺失：检查 production image 和挂载路径；
全远端模式不要求 coordinator 本地安装 Icefall。

`pretrained.pt` 不存在：重新下载官方 draft checkpoint，或检查
`SURE_BASELINE_CHECKPOINT_DIR` 是否指向 `exp/` 目录。
