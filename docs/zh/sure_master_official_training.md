# F5-TTS 与 DiariZen：官方预算的完整训练

四个 `ordinary-{tts,sd}-{cuda,npu}.yaml` 配置现在都使用完整训练入口。
基线也训练原结构；每轮四个候选只改变结构。候选达到结束条件后才进入 SURE 评分，
selection/holdout 只恢复模型推理，不重新训练。旧的 `max_steps: 1000` 配置会被拒绝。

## 预算依据

| 项目 | F5-TTS | DiariZen |
|---|---|---|
| 官方依据 | `src/f5_tts/train/finetune_cli.py` | `recipes/diar_ssl/conf/wavlm_updated_conformer.toml`、`run_stage.sh` |
| 数据 | WenetSpeech4TTS-Premium 的完整 train 划分 | AMI 的完整 train 划分 |
| 初始化 | 固定 F5TTS_v1_Base 官方权重；结构改动部分加载，匹配率至少 70% | WavLM-Base+ SSL 权重，其他网络随机初始化；不读成品 DiariZen v2 |
| 终止 | 100 epoch，无短训截断或早停 | 最多 100 epoch，完整训练验证损失连续 10 次不改善可早停 |
| 优化器 | 官方 AdamW，LR `1e-5` | 双 AdamW：WavLM `2e-5`，其他参数 `1e-3` |
| Batch | 每卡 3200 mel frames，最多 64 条/batch，累积 1 | 每卡 16 个片段、4 进程，全局 64，累积 1 |
| 数据处理 | 24 kHz，固定官方词表，shuffle seed 666 | 16 kHz，8 秒片段，train shift 6 秒、validation shift 8 秒 |
| 其他 | warmup 20000 updates，按实际 100 epoch 计算衰减 | seed 3407，官方梯度裁剪，验证 batch 8/进程，无额外 LR decay |
| 推理权重 | 完成后 checkpoint 的 EMA | 按完整 train_validation Loss 选最优 5 个 checkpoint 平均 |

F5 默认一个训练进程；SD 默认四个。CUDA/NPU 采用相同预算和 FP32。
`max_samples: 64` 是每个动态 batch 的上限，不是只训练 64 条数据。
F5 使用微调 CLI 的 100 epoch，不使用预训练 YAML 的 11 epoch 或 GUI 自动推算预算。

SD 的新基线是官方 WavLM-updated 路线，不再使用当前下载的 pruned large v2 成品配置。
固定推理采用 8 秒分段、AHC threshold 0.70、min_cluster_size 30，以及 Loss/best/5 平均规则。
SURE 的 DER 仍使用之前选定的 collar=0.25 秒、会话平均和 UEM 裁剪；不是论文 collar=0 的复现分数。

## 数据和 SSL 权重准备

Premium 数据必须全部下载并解压。准备工具依据 `Premium_md5check.txt` 校验每个压缩包，
记录完整成员清单，并检查每个音频/文本文件已经解压。首次验证前不要删除压缩包。
校验记录保存在输出目录的 `source_integrity.json`，之后可复用已验证的压缩包记录。
只下载或只解压一部分会失败，不能把这部分当作完整训练集。

```bash
python playground/sure_master/tools/prepare_task_data.py tts \
  --root /shared/chaolei.liu/data/WenetSpeech4TTS-Premium \
  --output /shared/chaolei.liu/data/sure_premium
```

标准 `..._S001-S002` 样本 ID 自动映射到原始录音分组；特殊命名必须提供 `--groups` JSON。
沿用既定 train/train_validation/search/selection 划分，完整 Seed-zh 留作 holdout。

```bash
python playground/sure_master/tools/prepare_task_data.py sd \
  --root /shared/chaolei.liu/data/ami-corpus \
  --recipe-data /shared/chaolei.liu/SD/DiariZen/recipes/diar_ssl/data/AMI_AliMeeting_AISHELL4 \
  --output /shared/chaolei.liu/data/sure_ami
```

两个准备工具都会生成 `preparation.json`，记录各划分摘要及来源完整性。
配置中的 `data_preparation` 必须指向它；旧的清单需重新验证生成准备报告。

在 SD 模型环境中准备官方 SSL 主干；`--download` 可选，已有本地 HF 快照时省略：

```bash
python playground/sure_master/tools/prepare_wavlm_initialization.py \
  --diarizen-root /shared/chaolei.liu/SD/DiariZen \
  --source /path/to/wavlm-base-plus \
  --output /path/to/wavlm-init --download
```

工具复用 DiariZen 的转换器，严格验证转换后权重加载，并生成
`wavlm-base-plus-converted.bin.provenance.json`。将 `SURE_WAVLM_INIT` 指向对应 `.bin`。
不允许把训练好的 DiariZen 权重标作 SSL 初始化。模型环境见
`playground/sure_master/runtime/environments/README.md`；该转换需安装 Transformers。

## 完成证明与恢复

每个候选使用 `models/f5tts_training/` 或 `models/diarizen_training/`：

- `job.json`：冻结的 recipe、结构、数据摘要、初始化、后端和进程数。
- `state/contract.json`：训练身份；变化后拒绝恢复。
- `state/checkpoint-*/`：模型、优化器、调度器、EMA（F5）、各 rank RNG、梯度历史和 epoch/batch 游标。
- `evidence/training_completion.json`：实际 epoch、更新数、停止原因与 checkpoint 摘要。
- SD 额外记录完整训练验证历史及平均使用的五个 checkpoint。

快照由所有 rank 共同保存，原子提交，保留最近两份；最新快照损坏时回退上一份。
只在优化器更新边界保存，默认 F5 每 5000 updates、SD 每 500 updates，另在 epoch 结束保存。
F5 续训恢复原生 optimizer/scheduler/EMA；SD 恢复双 optimizer、梯度裁剪历史和早停状态。
新候选从固定初始化来源开始，不续用上一候选的优化器状态。

只有报告与当前训练契约一致、结束条件成立、所有 checkpoint 校验通过才可评分。
超时、OOM 和部分写入不是完成。训练已完成而评分失败时，重试会直接复用训练结果。
推理模型与完成证据一起进入运行级不可变模型包，清理工作区后仍可验证和复评。

Slurm 调度按任务分配设备：ASR 训练保持 8，TTS 训练 1，SD 训练 4，冻结推理 1。
不允许为了适应资源不足而静默缩 batch、改进程数或减训练量。
修改预算/数据/初始化/后端后使用新的运行目录；旧短训 checkpoint 不能升级为完整训练结果。

## 轻量检查

本次实现不运行 100 epoch 验收，也不提交真实多轮 XLab 实验。

```bash
python -m unittest playground.sure_master.core.utils.test_official_training
# 在有 Torch 的环境中：小型 CPU 模型与 Gloo，不加载 F5/DiariZen 大模型。
python -m unittest playground.sure_master.core.utils.test_training_state
```

`probe_task.py --component train|arch` 仍是一次小型梯度检查，不会调用完整训练循环。
`--component inference` 对 TTS 只检查原始模型单样本合成；SD 则需要已完成训练的模型包。
`--component replay --model-artifact ...` 只做冻结推理。组件结果不能用作训练完成证明。
真实 CUDA/NPU 模型与算子兼容性需要在分配到的模型环境中单独验证。
