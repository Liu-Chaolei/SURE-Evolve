> 当前部署（2026-09-13 更新）：50 epochs、每卡 25600 frames、BF16 autocast、fused AdamW、expandable_segments。主权重、优化器状态及推理保持 FP32。以下历史性能测试中的 100-epoch 时间仅为当时的规模换算。

# F5-TTS：Premium 完整微调与三方案自由进化

正式配置为 `configs/sure_master/xlab-f5tts-premium-npu-evolution.yaml`。
以本地 F5TTS_v1_Base 官方权重为起点，在 Premium 完整 train 划分上微调。
基线与每个更新权重的方案均完成 100 epoch，使用最终 EMA；这是固定配方完成标准，
不是验证集收敛承诺。搜索期间就完成微调，结束后不再重训。

XLab 每轮提出三个完整方案，没有结构/训练/推理类型配额，允许一个方案同时修改
模型、loss、优化器、训练增强与推理方法。至少六轮；之后连续三轮无改善停止，
最多十轮。纯推理方案继承轮初冻结的最佳模型；训练方案从同一官方权重重新初始化。

每个训练方案单节点八张 Ascend NPU、160 CPU、1000G 内存、1T 临时盘。
最多三个候选并行，训练峰值 24 张 NPU；独立推理使用一张卡。
遵循 `/shared/chaolei.liu/rules.md` 和 <http://127.0.0.1:18088/>：计算节点必须由
Slurm 分配，并在真实 step 内执行 `sudo -n slurm-docker-run`。不手工覆盖 Ascend
可见卡变量。镜像使用内部 Registry digest；缓存进入 `/local/job`，持久结果写 `/shared`。

## 数据与资源

`prepare_task_data.py tts` 校验十份压缩包及解压清单，从片段 ID 推导原始录音分组。
按种子 42 划分 train/train_validation/search/selection，分组比例为 90/2/4/4。
原始录音分组不等于说话人身份。保留 1–30 秒训练片段，3–10 秒参考音频。
正式 search 固定 400 条、selection 固定 800 条；Seed 中文标准集 2020 条仅用于 holdout。
`preparation.json` 保存全量来源与 manifest 摘要，不接受未经说明的部分数据。

工作环境从 `docker/sure-master-ascend/Dockerfile.f5tts` 构建，保持 Torch、torch_npu、
torchaudio 版本配对。构建上下文只需要 Dockerfile 和
`playground/sure_master/runtime/environments/f5tts.txt`，不要把数据或凭据放进镜像。
`prepare_tts_resources.py --output /shared/chaolei.liu/data/sure_tts_resources` 下载固定
Vocos 24 kHz 和 Paraformer 中文评分资源，保存文件摘要。需要联网的下载阶段在 n01
执行；正式训练使用本地资源。配置引用已通过 XLab 检查的 `20260910-survey-tts-v2`。
评分采用 `metric_runtime.tts_runtime: worker`：在独立评分进程里调用 SURE 的公开 TTS API，
注入同一 Paraformer 实现，保持 pipeline ID、归一化与 CER 计算不变；不依赖 SURE 源码目录
内的 `.venv`，也不修改其源码。推理与评分的零超时表示由 Slurm 作业时限管理。

## 冻结并启动

使用控制器 Python，传入构建发布后得到的真实 digest：

```bash
/shared/chaolei.liu/data/sure_asr_controller/bin/python \
  -m playground.sure_master.tools.prepare_tts_deployment \
  --image registry.cluster.local:5000/users/chaolei-liu/sure-f5tts@sha256:IMAGE_DIGEST \
  --output /shared/chaolei.liu/SURE-Evolve/runs/f5tts_premium_evolution

bash /shared/chaolei.liu/SURE-Evolve/runs/f5tts_premium_evolution/launch.sh
```

入口先准备完整数据，再提交八卡组件验收，之后执行正式基线和搜索。
组件验收使用真实 F5 的少量更新、重新启动后的 checkpoint 恢复、音频生成重放和 CPU CER；
不会生成可以进入正式排名的训练完成证明，也不会启动“两轮验证搜索”。
可在冻结源码目录分别调用 `tts_workflow --stage prepare|probe|search|all`，
并传入相同的 `--config deployment.yaml --output RUN_DIR`。

控制器使用现有私有 `.env` 的 ZAI 配置，XLab 使用 XI 配置。部署保留凭据占位符，
通过 `with_api_profile` 注入；模型 worker 请求不包含 LLM 密钥。

## 候选扩展接口

生成的 `run_sure.py` 使用 `SURE_WORKER_PYTHON` 和 `SURE_TASK_WRAPPER`：

```python
subprocess.run([python, wrapper, "--action", "prepare_source",
                "--candidate-source", "working/f5_candidate"], check=True)
# 在 working/f5_candidate/src/f5_tts 下实现方案；不要改共享源码。
subprocess.run([python, wrapper, "--action", "candidate",
                "--candidate-source", "working/f5_candidate",
                "--parameters-json", json.dumps({
                    "requires_training": True,
                    "training": {"learning_rate": 2e-5},
                    "architecture": {"depth": 20},
                    "inference": {"nfe_step": 24},
                })], check=True)
```

模型/loss 与推理源码可修改。框架保护训练循环、数据遍历与 CSV 准备逻辑。
需要改变优化器、调度或增强时，在 `src/f5_tts/sure_candidate.py` 定义可选函数：

- `build_optimizer(parameters, training)`：返回 Torch 优化器。
- `build_scheduler(optimizer, total_updates, training)`：返回可保存恢复的 scheduler。
- `transform_batch(batch)`：返回保持字段和批大小的 batch。

固定项包括完整训练数据、100 epoch、初始化种子、八个训练进程、FP32、最终 EMA、
词表、24 kHz 梅尔接口、vocoder 和评分文本。实际源码及参数进入训练身份和模型包；
纯推理不能通过修改模型/训练源码绕过训练。旧的仅结构配置和固定官方配方仍可使用。

## 结果、恢复与验收

search 按固定 SURE 中文 CER 比较。结束后 selection 比较完整微调基线与 search 前两名；
Seed 只评估 selection 赢家和基线，测试结果不反馈给 XLab。
保存逐样本生成音频、评分报告、源代码、模型与 XLab 成功/失败历史。

`workflow_state.json` 记录阶段，`search/workspace/metric/controller_state.json` 记录轮次，
候选的 `metric/slurm/*/job.json` 记录 Slurm 作业。只有 100 个完整 epoch、合法更新次数、
最终 EMA 和一致摘要通过检查，才发布 `training_completion.json` 并参与排名。
节点故障或 TIMEOUT 从同一方案持久 checkpoint 恢复；取消不自动重提，提交状态不明确
时先核对已有作业，避免重复运行。部署身份变化必须使用新运行目录。

验证命令：

```bash
python -m unittest playground.sure_master.core.utils.test_f5_evolution \
  playground.sure_master.core.utils.test_tts_formal_search \
  playground.sure_master.core.utils.test_official_training \
  playground.sure_master.core.utils.test_full_search_training \
  playground.sure_master.core.utils.test_free_idea_generation
# F5 镜像内另外运行：
SURE_F5_TEST_SOURCE=/path/to/F5-TTS python -m unittest \
  playground.sure_master.core.utils.test_f5_runtime
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m unittest \
  playground.sure_master.core.utils.test_training_state
```

两组 CPU 测试使用独立进程，避免 Accelerate 导入的 torch_npu 全局补丁影响纯 Torch 测试。
CPU/模拟测试通过不等于真实 NPU 验收通过；流程完成和 CER 改善分别报告。


### 与官方微调默认值的对齐

基线以 F5-TTS `finetune_cli.py` 的默认值为参照：100 epochs、学习率 1e-5、
每卡 3200 frames、最大 64 样本、梯度累积 1、裁剪 1.0、warmup 配置 20000、
每 50000 updates 保存定期检查点，每 5000 updates 保存最近检查点。
正式基线直接载入官方模型及 EMA 权重，同时恢复 EMA 的 `step` 和 `initted`；
优化器仍从新状态开始。续训时再完整恢复本次微调的模型、EMA、优化器、scheduler 和 RNG。
每训练进程默认使用官方的 16 个 DataLoader workers；复用 256 GB 服务分配时，
设置 `SURE_F5_DATALOADER_WORKERS=4`。16 workers 在这种配额下已于跨 epoch 时触发
Slurm OOM，因此该资源适配必须保留。实际 worker 数写入 `initialization.json`。
所有定期检查点保留；另有两个原子提交的训练状态用于可靠断点恢复。

NPU/CUDA 默认使用官方 `AdamW(fused=True)`，CPU 保留非融合实现。
每步将 loss 与梯度范数一起检查，确认有限后才更新参数；loss 日志记录首步和每 50 步，
避免重复的 host 同步。检查点源码身份包含这些运行时实现。

保留的适配差异：NPU 上固定 FP32；DataLoader 使用
`spawn`、180 秒超时，避免初始化计算库后 fork 导致重采样死锁；
分布式尾批次补齐及超预算单样本保留，以满足完整训练数据遍历。
验收训练使用基线优化器，不再注入非默认 weight_decay hook。


### 性能验证与 batch 试验

2026-09-13 在独立 8 卡分配上，以 16384 条真实训练样本进行了组件测试。
非融合、fused、fused 加合并同步的同条件耗时分别约为 0.357、0.302、0.290 秒/update。
每卡 12800 frames 时约 0.680 秒/update，峰值框架预留显存 48.1 GiB，40 次更新无 OOM。
这些短测仅验证运行性能，不替代完整微调或候选 CER 评测。

Fused 与非融合的模型、EMA、优化器比较通过 FP32 容差。checkpoint 刚加载时，
1819 个张量及 CPU/NPU RNG 逐元素一致。恢复后继续训练的动量最大差约 1.76e-6，
严格的 1e-6 动量比较有两个元素超限；模型最大差约 3.73e-8，EMA 约 2.91e-11。
性能试验按动量 1e-5 的工程界限继续，并保留原始严格比较的失败记录。
详细数据在 `runs/f5tts_acceleration_validation_20260913/summary.json`。

12800 frames 改变有效 batch、总 updates 以及 warmup 覆盖的数据量，应作为独立训练方案；
100 epochs 和最终 EMA 评测要求不变。正在运行的冻结 v8 基线仍使用其原配置。


### 当前模型保留规则

按照用户选择保留官方格式的全部定期模型：`keep_last_n_checkpoints=-1`。
每 50000 updates 保存 `model_<update>.pt` 并全部保留；每 5000 updates
覆盖更新 `model_last.pt`，训练结束另保留最终 EMA 评测文件。
每个 epoch 结束和定期保存点还会提交完整续训状态（模型、EMA、优化器、scheduler、各 rank RNG），
原子状态保留最新两代用于校验与回退。实验成功后不清理 `models/` 和 `working/`，
确保定期模型及其源码可用于后续测试。

50 epochs 是当前 BF16 部署的完整预算，所有更新权重的候选都必须完整完成；
旧 FP32 配置和旧训练结果保持原有协议，不作为新基线继续训练。
