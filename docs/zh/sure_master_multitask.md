# SureMaster：ordinary + XLab 多任务进化

新运行只支持 `ordinary`（`origin` 为别名），且必须启用 XLab。staged_axes 配置
归档到 `configs/sure_master/archive/staged_axes/`，历史协议与报告仍可读取；不再执行旧搜索引擎。

## 同一条流程

固定基线 → 每轮冻结最佳代码和模型 → XLab 生成 4 个 idea → 执行与评分 →
提交最佳代码/模型 → XLab 总结 → 下一轮 → selection → 冻结方案 → holdout。

同轮所有推理候选继承同一个轮初最佳模型。训练候选按固定初始化来源和预算重新实验，
不在上一轮的 checkpoint 上累计训练。候选类型没有配额。失败也进入 XLab 反馈；
XLab 失败不切回内置 research。selection/holdout 由可信 wrapper 恢复模型推理，
不重训、不 debug 改代码，holdout 不参与后续选优。

## 配置与资源

复制 `configs/sure_master/multitask.env.example` 到私有位置并设置真实资源、解释器及
任务对应的 XLab literature survey。新配置为 `ordinary-{asr,tts,sd}-{cuda,npu}.yaml`。
模型解释器、协调器和 CPU SURE 评分解释器分开。运行环境说明在
`playground/sure_master/runtime/environments/README.md`。

TTS 中文训练数据使用 WenetSpeech4TTS-Premium。数据未下载解压完成时，工具会明确报错。
必须提供可信的 utterance→speaker/source-recording JSON 映射（`--groups`），避免按句随机切分：

```bash
python playground/sure_master/tools/prepare_task_data.py tts \
  --root /shared/chaolei.liu/data/WenetSpeech4TTS-Premium \
  --groups /private/premium_groups.json \
  --output /shared/chaolei.liu/data/sure_premium
```

按组、种子 42 划分 train/train_validation/search/selection 为 90/2/4/4%；
参考音频和目标来自不同片段。完整 Seed-zh 是 holdout；英文 Seed/WER 接口保留。
转换保留基础模型词表。训练预算由运行配置控制，候选只能改允许的参数。

SD 使用本地 DiariZen v2、AMI Array1-01 和模型配套会话清单：

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
    '--action', 'infer',
    '--parameters-json', json.dumps({'inference': {'speed': 0.95}}),
], check=True)
```

F5/SD 参数分为 `inference`、`training`、`architecture` 三个对象。
ASR wrapper 兼容 `train_args_json`、`decode_args_json`、`decode_method`。
动作是 `infer`、`fine_tune`、`arch`；固定基线使用 `baseline`。
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
