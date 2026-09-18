# ASR-Native-MCTS-v2 异机交接文档

本文用于把 **ASR-Native-MCTS-v2** 迁移到另一台昇腾机器后继续运行。

当前部署目录：

```text
/shared/chaolei.liu/SURE-Evolve/runs/asr_native_mcts_v2_20260917
```

当前状态：

- 研究状态为 `blocked`，原因是之前的 API 请求失败后耗尽了该操作的恢复预算。
- 旧的失败研究树和 API 回执已归档在 `recovery_history/`，不能直接当作新策略的断点。
- `TRANSITION_STOP` 存在，交接器已停止；它防止自动占用训练资源。
- 基线已导入，`baseline_import.json` 中 `training_performed` 应为 `false`。
- 没有候选训练结果需要迁移，训练恢复前必须由用户明确提供目标机器的资源。

## 1. 两台机器的文件是否需要复制

如果目标机器和当前机器使用同一个 `/shared` 文件系统，不需要复制
`/shared/chaolei.liu/SURE-Evolve` 或 `/shared/chaolei.liu/XLab`。直接在目标机器通过
Slurm 提交即可，路径仍然有效。

如果目标机器使用独立文件系统，仅复制两个顶层代码目录不够。至少需要迁移下面的内容，
并保持目录结构和文件哈希：

```text
/shared/chaolei.liu/SURE-Evolve/
  runs/asr_native_mcts_v2_20260917/       # 整个目录，包含 source、external、缓存、回执、归档
  runs/asr_formal_6x4_npu4_v2_20260913_151000/  # baseline 和 icefall_source
  playground/
  evomaster/

/shared/chaolei.liu/XLab/                 # 仅用于重新构建或核对 XLab；运行时优先使用新部署内置副本

/shared/chaolei.liu/data/sure_tedlium3_unigram500_musan_v2/
/shared/chaolei.liu/data/TEDLIUM_release-3/       # 目标机没有原始数据时才需要

/shared/chaolei.liu/data/sure_asr_controller/
/shared/chaolei.liu/data/sure_xlab_runtime312/
```

训练镜像也必须能在目标节点拉取：

```text
registry.cluster.local:5000/users/chaolei-liu/sure-zipformer@sha256:d557a61e7beb3b675917946061e63790997fc9f672563fe7882f3336b1a8fa64
```

目标节点必须具有匹配的昇腾驱动、CANN、Docker/Slurm 权限和
`gpu:ascend910b3` 资源。容器镜像不会随着代码目录自动迁移。

## 2. 必须保留的运行内容

不要只复制 `deployment.yaml`。以下内容共同构成可恢复身份：

- `source/<digest>/` 冻结的 SURE 源码；
- `external/XLab/` 和其中的 `literature_bundle/`；
- `external/XLab/.xlab/runs/asr_native/`；
- `parent_provider_cache/`、路由冷却和 API 回执；
- `search/workspace/metric/`；
- `api_stage_revision_20260917_xcode/` 及其 `policy.json`、`recovery.json`；
- `dependencies.json`、`deployment.json`、`external_manifest.json`；
- `recovery_history/`，用于审计旧失败请求，不用于伪造新结果。

`xlab_operations` 是指向
`external/XLab/.xlab/runs/asr_native` 的符号链接。独立文件系统迁移时必须保留目标目录，
或者在迁移后重新创建等价链接：

```bash
ln -s external/XLab/.xlab/runs/asr_native \
  runs/asr_native_mcts_v2_20260917/xlab_operations
```

不要复制运行中的 PID、锁文件语义或旧节点的 Slurm allocation 身份。

## 3. API 配置

在目标机器创建本地 `.env`，不要把密钥提交到 Git 或放入运行目录的公开配置。
当前分环节策略为：

| 环节 | 路由 | 模型 |
|---|---|---|
| 检索、摘要、证据整理、MCTS 评估/诊断/新颖性、候选代码调试 | ZAI → ZHOU → OPENAI → XCODE | GLM 环节首选 `glm-5.3-flash`，最后 `gpt-6-astra` |
| 初始分析、MCTS 候选生成、融合、融合评审/修复、知识更新/重规划 | OPENAI → XCODE | `gpt-6-astra` |
| 训练、解码、SURE WER | 不调用 API | - |

`.env` 至少需要这些键：

```text
ZAI_API_KEY=...
ZAI_BASE_URL=...
ZHOU_API_KEY=...
ZHOU_API_BASE_URL=...
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
XCODE_API_KEY=...
XCODE_API_BASE_URL=...
```

启动前检查键是否存在即可，不要打印值：

```bash
python -P -c 'from dotenv import dotenv_values; d=dotenv_values(".env", interpolate=False); print({k: bool(d.get(k)) for k in ("ZAI_API_KEY", "ZHOU_API_KEY", "OPENAI_API_KEY", "XCODE_API_KEY")})'
```

## 4. 路径重映射

最稳妥的方式是在目标机器保持相同绝对路径：

```text
/shared/chaolei.liu/SURE-Evolve
/shared/chaolei.liu/data/...
/shared/chaolei.liu/data/sure_asr_controller
/shared/chaolei.liu/data/sure_xlab_runtime312
```

如果必须使用不同根目录，不能直接运行现有脚本。需要对
`deployment.yaml`、`execution.yaml`、`input_config.yaml`、`prepare_ideas.sh`、
`launch.sh`、`api_stage_revision_20260917_xcode/` 中的路径统一重写，至少包括：

- `task_cards_path`、`task_description_path`；
- `initial_baseline_run`、`baseline_run`、`icefall_source`；
- `SURE_ASR_PREPARATION`、`SURE_ASR_ZIPFORMER_WRAPPER`；
- `XLAB_ROOT`、`XLAB_SURE_SURVEY_PATH`、`XLAB_SURE_RUN_ROOT`；
- `XLAB_SURE_PROVIDER_CACHE_ROOT`、`XLAB_API_ROUTING_STATE_DIR`；
- `api_profile.env_file`、`stage_policy`、`stage_controller_entry`；
- `PYTHONPATH`、监控路径和活动指针。

重写后必须重新检查所有路径存在，并重新计算冻结清单。不要只改启动脚本，否则运行时会
继续引用旧机器的绝对路径。

## 5. 目标机器环境检查

在目标节点、且已获得 Slurm allocation 后执行：

```bash
hostname
python --version
which sudo
which docker
srun --jobid="$SLURM_JOB_ID" --overlap --exact -c1 \
  --gres=gpu:ascend910b3:1 --ntasks=1 npu-smi info
```

确认容器镜像可拉取。正式候选训练使用每候选 4 NPU、32 CPU、128G；同一节点可按资源
情况运行两个候选。不要用 CPU 机器替代昇腾训练节点。

## 6. 恢复顺序

在确认目标机器文件、数据、镜像和 `.env` 都就绪后：

1. 先运行只读身份校验，确认 `source_manifest.json`、`external_manifest.json`、基线
   manifest 和数据 `preparation.json` 未改变。
2. 保留现有 `recovery_history/`；不要删除 `generation.json`、operation receipt 或
   provider cache。
3. 由于本部署包含 `TRANSITION_STOP` 且当前研究曾因 API 策略变更失败，先由维护者确认
   是否要从新的 API 策略重新开始首轮研究。不能把旧策略的 MCTS 模式直接拼接到新策略。
4. 若要从新策略重新开始，应在新目录或新的 recovery revision 中归档旧
   `search/` 与 `external/XLab/.xlab/runs/asr_native`，再调用 `prepare_ideas.sh`。
5. ideas-only 阶段发布四个完整候选后，才可移除资源阻断并提交训练。训练前必须确认
   `TRAINING_ENABLED`，并由用户明确授权目标节点和 NPU 数量。
6. 训练启动后检查每个候选首个 epoch/batch 的有限 loss，再写入
   `training_verification.json`。

恢复入口必须使用部署目录中的脚本：

```bash
cd /shared/chaolei.liu/SURE-Evolve
bash runs/asr_native_mcts_v2_20260917/prepare_ideas.sh
# 仅在候选评审完成、资源授权且 TRANSITION_STOP 已按维护决定处理后：
bash runs/asr_native_mcts_v2_20260917/launch.sh
```

不要直接运行冻结 source 中的旧模块入口，也不要复用旧 PID。

## 7. 验收清单

交接完成前应记录：

- 目标节点 hostname、Slurm job ID、NPU 型号和可见设备；
- SURE/XLab Python 环境版本；
- 容器镜像 digest；
- 数据 preparation fingerprint；
- baseline manifest digest；
- 新部署 source/external manifest 校验结果；
- API 策略 digest、四个 API 键是否配置（只记布尔值）；
- `workflow_state.json`、`transition_state.json` 和 `TRAINING_ENABLED` 状态；
- 候选数量、训练并行度、首个有限 loss；
- 不得把 test 集结果反馈给搜索或重规划。

## 8. 资源和安全边界

Slurm 中使用 `--gres=gpu:ascend910b3:<数量>`，并遵循
`/shared/chaolei.liu/rules.md`、`http://127.0.0.1:18088/` 手册和
`/shared/cluster/SLURM-USAGE.md`。不取消别人的作业，不唤醒被暂停的服务 holder，
不在无 allocation 的节点上查询 NPU，不把 API key、token、checkpoint 或完整 `/shared`
目录放入容器镜像。

本文件描述的是迁移和恢复边界，不代表已自动启动训练。当前部署仍需在目标机器上完成
路径、依赖、API 和资源验收后，才能继续研究。
