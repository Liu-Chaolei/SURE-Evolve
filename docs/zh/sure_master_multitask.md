# SureMaster：ordinary + XLab 多任务进化

新运行只支持 `ordinary`（`origin` 为别名），且必须启用 XLab。staged_axes 配置
归档到 `configs/sure_master/archive/staged_axes/`，历史协议与报告仍可读取；不再执行旧搜索引擎。

## 同一条流程

固定基线 → 每轮冻结最佳代码和模型 → XLab 生成 4 个 idea → 执行与评分 →
提交最佳代码/模型 → XLab 总结 → 下一轮 → selection → 冻结方案 → holdout。

当前 `sure.search_scope: architecture_only`：每轮四个候选都必须是结构改动，
按固定初始化来源和预算重新实验，不在上一轮的 checkpoint 上累计训练。
训练方法、训练预算和推理设置固定；每个候选在搜索阶段直接完成配置中的最终训练预算，再进行评分。
不再采用短训筛选后对少数方案重训的流程。失败也进入 XLab 反馈；
XLab 失败不切回内置 research。selection/holdout 由可信 wrapper 恢复模型推理，
不重训、不 debug 改代码，holdout 不参与后续选优。

## 配置与资源

复制 `configs/sure_master/multitask.env.example` 到私有位置并设置真实资源、解释器及
任务对应的 XLab literature survey。新配置为 `ordinary-{asr,tts,sd}-{cuda,npu}.yaml`。
模型解释器、协调器和 CPU SURE 评分解释器分开。运行环境说明在
`playground/sure_master/runtime/environments/README.md`。

TTS 中文训练数据使用 WenetSpeech4TTS-Premium。数据未下载解压完成时，工具会明确报错。
标准样本 ID 自动按原始录音分组；特殊命名提供可信的 utterance→speaker/source-recording
JSON 映射（`--groups`）。首次准备需保留完整压缩包以完成校验：

```bash
python playground/sure_master/tools/prepare_task_data.py tts \
  --root /shared/chaolei.liu/data/WenetSpeech4TTS-Premium \
  --groups /private/premium_groups.json \
  --output /shared/chaolei.liu/data/sure_premium
```

按组、种子 42 划分 train/train_validation/search/selection 为 90/2/4/4%；
参考音频和目标来自不同片段。完整 Seed-zh 是 holdout；英文 Seed/WER 接口保留。
转换保留基础模型词表。训练预算由运行配置控制，候选只能改允许的参数。

SD 使用官方 WavLM-updated 训练 recipe、AMI Array1-01 和模型配套会话清单：

```bash
python playground/sure_master/tools/prepare_task_data.py sd \
  --root /shared/chaolei.liu/data/ami-corpus \
  --recipe-data /shared/chaolei.liu/SD/DiariZen/recipes/diar_ssl/data/AMI_AliMeeting_AISHELL4 \
  --output /shared/chaolei.liu/data/sure_ami
```

保留 train/dev/test 划分；train 按会议组留出 5% 验证，dev 的会议组平分 search/selection。
因此会话数不一定各占一半。DER 使用固定 SURE 口径：collar=0.25 秒、会话平均，
ref/hyp 同时裁剪到 UEM。记录 SURE pipeline/report，不直接与论文的 collar=0 分数比较。

## 候选与模型接口

任务由 `TaskAdapter` 提供 preflight、环境/划分映射、产物校验、模型收集和冻结推理。
用 `register_adapter` 增加任务适配器；公共搜索循环不需要增加任务分支。

候选调用 `SURE_TASK_WRAPPER`，例如：

```python
subprocess.run([
    os.environ['SURE_WORKER_PYTHON'], os.environ['SURE_TASK_WRAPPER'],
    '--action', 'arch',
    '--parameters-json', json.dumps({'architecture': {'depth': 20}}),
], check=True)
```

F5/SD 参数分为 `inference`、`training`、`architecture` 三个对象。
ASR wrapper 兼容 `train_args_json`、`decode_args_json`、`decode_method`。
底层执行接口仍有 `infer`、`fine_tune`、`arch`；当前搜索仅允许 `arch`，不允许用
`training` 或 `inference` 参数覆盖夹带其他优化。固定基线使用 `baseline`。
冻结复评只接受 `infer --model-artifact ...`，禁止传入候选参数覆盖。

`sure.model_artifact.v2` 保存权重、模型/推理配置、词表、vocoder 或 embedding/PLDA、
源码快照及摘要。文件先复制到运行级 `model_artifacts/` 再清理工作区。
旧 ASR v1 清单可读取。相同权重但不同推理参数是不同候选产物。

## 轻量验证

本次不要求真实多轮 XLab 或全量 selection/holdout。先运行不访问 LLM 的契约测试：

```bash
python -m unittest playground.sure_master.core.utils.test_multitask
```

组件检查不会启动搜索。默认子进程超时为 300 秒，模型依赖或硬件缺失记为 `unverified`：

```bash
python playground/sure_master/tools/probe_task.py \
  --config configs/sure_master/ordinary-tts-npu.yaml \
  --component imports --workspace /tmp/sure-tts-npu-imports
python playground/sure_master/tools/probe_task.py \
  --config configs/sure_master/ordinary-sd-cuda.yaml \
  --component train --workspace /tmp/sure-sd-cuda-grad
```

`train`/`arch` 是极小合成 batch 的前后向、一次参数更新及保存恢复，不是数据集训练验收。
`inference` 只取评测 manifest 的一条，`replay --model-artifact ...` 检查冻结产物。
ASR 的模型梯度检查复用 `probe_zipformer.py`，真实音频解码使用现有 `smoke_baseline.py`。
真实硬件缺失时不能把模拟测试报告当作 CUDA/NPU 验证通过。

后续需要正式运行时，显式指定运行目录：

```bash
python run.py --agent sure_master \
  --config configs/sure_master/ordinary-tts-npu.yaml \
  --run-dir /shared/chaolei.liu/SURE-Evolve/runs/tts_npu \
  --task '在固定预算下改进 F5-TTS 中文 CER；由 XLab 产生方案。'
```

## 当前搜索范围

`search_strategy` 继续是 `ordinary`，`axis` 留空；结构限制由 `search_scope` 单独控制，
不会恢复 staged_axes。SureMaster 将结构范围、固定训练/推理设置随 execution_contract
送给 XLab，审核后的四个候选还必须通过本地领域校验，才能进入实现阶段。
旧批次会按当前范围重新校验，不会把以前的混合候选当成结构候选执行。
改动范围后应使用新的运行目录；已有断点的执行契约不一致时会明确拒绝恢复。

## 搜索即完整训练

正式 ASR 配置使用 `training_mode: full_during_search`，数据是完整 TEDLIUM train，
`SURE_MAX_TRAIN_EPOCHS` 和基线 epoch 都为 30。`ordinary-asr-*` 的数据路径由
`SURE_FULL_DATA_DIR` 指定，不能指向 1h 或 `search_100h` 准备结果。
TTS 使用官方微调 CLI 的 100 epoch；SD 使用 WavLM-updated 的最多 100 epoch 和
验证损失 patience=10。两者都训练原结构基线，不再使用 1000 步上限。
完整预算、资源准备和续训约束见 [官方训练指南](sure_master_official_training.md)。

旧配置中的 `full_training.enabled=true` 会在模型和工作区初始化之前，把最终数据和 epoch
迁移到搜索预算，避免仅删除后置阶段却仍然短训。旧运行与新预算的契约不同，必须新建运行目录。
轻量组件测试和性能校准保留；它们不参与候选选优，不属于小规模搜索训练。
