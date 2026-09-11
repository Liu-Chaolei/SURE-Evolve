# TEDLIUM-3 / Zipformer：CUDA 与昇腾运行说明

## 数据与评分

TEDLIUM 自带的 STM 是标注来源，不需要额外参考转写。准备工具读取 STM
起止时间，排除 `ignore_time_segment_in_scoring`，为语音片段和参考文本生成
同一个 ID。支持标准目录和当前 `TEDLIUM_release3/legacy/{train,dev,test}`
下音频、STM 平铺的目录。原始语料和 Icefall 源目录保持只读。

在仓库根目录、模型 Python 环境中运行：

```bash
python -m playground.sure_master.tools.prepare_tedlium \
  --corpus /shared/chaolei.liu/data/TEDLIUM_release-3 \
  --output /shared/chaolei.liu/data/sure_tedlium3_1h \
  --train-hours 1 --jobs 4
```

`--refs-only` 只生成参考文本和准备记录，不提取特征；`--train-hours 0` 准备完整
train，必须使用新的输出目录。工具以数据内容摘要检查缓存兼容性。

默认 dev 按演讲隔离：smoke 10、early 50、regular 200，selection 使用剩余
演讲。当前语料得到 selection 196、test 1155 条；train 子集 611 条、约 1 小时。
部分演讲为保持分组隔离只抽取所需数量，其未选片段不进入其他 tier。
test 不参与搜索。Fbank 为 80 维，BPE 为仅用 train 文本训练的 500 词元模型。

## 环境与配置

控制器使用 Python 3.10+、根目录 requirements.txt 和
`playground/sure_master/requirements.txt`。模型环境需要匹配的 Torch/k2、Lhotse、
torchaudio、SentencePiece、sph2pipe。SURE 英文 WER 在 CPU 上计算，返回比例，
例如 `0.1667` 表示约 16.67%。

将 `configs/sure_master/tedlium3.env.example` 复制为私有配置，填写服务地址、密钥、
模型名、XLab survey 路径，以及本机模型 Python 路径。不要提交私有配置。
XLab survey 必须来自真实的 literature_survey 流程，并包含 research_idea 所需
的证据资源；仅将一个空 JSON 或说明文档命名为 survey.json 不满足输入契约。

已有 XLab API-key 登录配置时，可用 `with_xlab_environment` 启动命令，直接读取
指定 agent 目录的配置和认证，不复制密钥到仓库。它将同一服务注入 SURE/XLab，
并为 SurveyAgent 启用流式响应，减少长响应经过代理时连接中断的风险：

```bash
python -m playground.sure_master.tools.with_xlab_environment \
  --agent-dir /shared/chaolei.liu/.pi/agent -- \
  python -m playground.sure_master.tools.preflight \
  --config configs/sure_master/xlab-tedlium3-npu-smoke.yaml
```

XLab 原生 research_idea 与 literature_survey 的 Python 依赖单独安装，使用
`XLAB_PYTHON` 指定该解释器；不能把仅安装控制器依赖的 Python 当作完整 XLab
运行环境。语料、survey 和模型 Python 等非秘密路径仍需配置。
当前基于已有知识图谱的运行路径可在独立环境安装
`playground/sure_master/requirements-xlab.txt`；需要现场解析 PDF 时再安装
XLab 原生 literature_survey 的额外解析依赖。

本次提供两个配置：

- `configs/sure_master/xlab-tedlium3-npu-smoke.yaml`
- `configs/sure_master/xlab-tedlium3-cuda-smoke.yaml`

两者均启用启动预检：检查数据、BPE、参考文本、真实 WER 小样例、XLab 配置和
模型设备。外部路径和模型凭据通过环境注入，不依赖旧 `/hpc_stor03` 路径。

```bash
python -m playground.sure_master.tools.preflight \
  --config configs/sure_master/xlab-tedlium3-npu-smoke.yaml
python run.py --agent sure_master \
  --config configs/sure_master/xlab-tedlium3-npu-smoke.yaml \
  --task playground/sure_master/data/asr_tedlium3_zipformer_description.md
```

CUDA 使用匹配的原生 Torch/k2 环境，将配置换成 cuda 文件。CUDA 多卡旧配置
保留原有行为。当前机器没有 NVIDIA GPU，CUDA 实机结果需要在目标机器验证。

## 昇腾单卡兼容环境

昇腾首版保留 Zipformer RNN-T：网络计算在 NPU，k2 loss 在 CPU，并通过
Tensor.to 的 autograd 回传梯度。默认 FP32、单卡、greedy 解码。没有将 RNN-T
替换为 CTC，也没有对全局 torch.cuda 做别名替换。

```bash
docker build --network=host -t sure-master-ascend:tedlium3 \
  -f docker/sure-master-ascend/Dockerfile docker/sure-master-ascend
```

镜像固定 Ascend 基础镜像摘要和 k2 源码提交，CPU 构建阶段禁用 NPU 自动加载。
工作区副本适配设备、标签和激活函数；外部 Icefall 源码不修改。

在 Slurm 单卡分配内使用 `docker/sure-master-ascend/worker.sh`。它将实际分配的
物理卡映射到容器，并在取消时停止容器。若目标节点尚未加载镜像，先使用
`docker save` 导出到共享目录，设置 `SURE_NPU_IMAGE_ARCHIVE`。私有环境文件通过
`SURE_PRIVATE_ENV_FILE` 传入 Docker；该文件应使用绝对路径、普通 `KEY=value`
赋值，路径中不使用尚未展开的变量。

```bash
srun --partition=compute --gres=gpu:ascend910b3:1 \
  --cpus-per-task=8 --mem=32G --time=00:30:00 \
  bash docker/sure-master-ascend/worker.sh \
  python -m playground.sure_master.tools.probe_zipformer \
  --icefall /shared/chaolei.liu/ASR/icefall --backend npu \
  --output /shared/chaolei.liu/data/sure_zipformer_probe_npu
```

探测执行真实模型前向、loss、梯度和一次权重更新，并比较 NPU 桥接结果与 CPU
参考。只有探测通过才继续长训练。训练仍受 NPU 算子覆盖和 CPU loss 性能限制。

## 模型产物与 XLab 反馈

每个成功 ASR 实验在清理前生成 `retained_model/manifest.json`，保留模型权重、
recipe、候选代码、BPE 和候选修改记录。`models/`、`working/` 的清理不删除这份
保留产物。后续阶段按产物引用加载，不从空目录隐式重新训练。

普通 XLab 搜索默认两轮、每轮四个 idea。反馈包含逐候选 WER、成功状态、是否
改善、失败类别和 checkpoint。成功但未改善的实验不标为运行失败。
生成采用逐个自由探索：不规定结构、训练、推理的数量比例，后续调用会看到
本轮已接纳候选和拒绝原因。生成后由 XLab 的 evaluation model 检查证据、
执行预算、机制重复和实现忠实度；仅换措辞或同一参数的数值不算新机制。
`xlab.idea_generation.max_attempts` 默认为 8，表示原生生成流程的最多调用次数，
不是底层 LLM 请求总数。凑齐四个即发布；不足四个则保持未完成，不启动该批次训练。

候选可以围绕一个假设组合多类改动，并必须给出消融计划。`IdeaSpec.change_domains`
记录 `arch/train/inference` 改动领域，`requires_training` 记录训练需求。
结构改动归为 `arch`，无结构改动但更新权重归为 `fine_tune`，完全不训练归为
`inference`；实际执行仍需通过 wrapper 校验。普通 XLab 批次不受旧的每类数量限制。
明确启用的 `staged_axes` 仍遵守各阶段范围。smoke 的 `require_training_candidate`
是整次运行的验收条件，不会强迫生成训练候选。

历史保存原生 idea、实现代码摘要、评估范围、训练预算和实际结果。当前最佳模型
有对应原生 idea 时，以它为 `mature_idea`，通过 `experiment_feedback` 触发 XLab
原生重规划。最佳模型仍是初始基线时，通过 discussion 提供历史，不伪造父 idea。
公共产物中的 risks 列表会无损转成原生搜索所需文本，原始产物仍保留。
总结中的 `observations` 区分 `improved/no_improvement/execution_failed/inconclusive`；
这些是给定预算下的观察，不代表机制被普遍证实或证伪。

XLab 运行目录保存 `generation.json`、`attempt-N/sure_request.json`、
`attempt-N/review.json`、`attempt-N/accepted.json` 和成功时的 `batch.json`。
SURE 同时保存 `artifacts/xlab_batches/`。已发布请求可重放；未完成请求必须先
核对产物，不能自动重复原生生成。审查记录包含原因及 provider 返回的用量。

仅有通过验证的 survey 文本还不够：survey 必须声明原生 research_idea 所需的
resource manifest，包含有 lineage 的图谱、组件索引、keynotes 和模型快照。
两份 smoke 配置通过 `idea_provider.preflight_command` 在训练前检查此条件：

```bash
"$XLAB_PYTHON" "$XLAB_ROOT/xlab/skills/sure_master/scripts/xlab_idea_client.py" \
  --check-survey "$XLAB_SURE_SURVEY_PATH"
```

此检查不生成 idea、不训练模型。缺少声明或资源校验失败时明确报告 blocked。

selection 对基线及入选模型做冻结权重复评，test 对最终选择和基线做最终评估。
报告为 `metric/final_evaluation.json`；最佳模型引用为
`best_solution/model_artifact.json`。

每次回传前将请求保存到 `artifacts/xlab_rounds/*.request.json`。仅重试回传：

```bash
python -m playground.sure_master.tools.retry_xlab_summary \
  --config configs/sure_master/xlab-tedlium3-npu-smoke.yaml \
  --workspace /path/to/original/workspace \
  --request /path/to/original/workspace/artifacts/xlab_rounds/ROUND.request.json
```

该命令不生成新候选、不重新训练。失败的 generate 操作仍须核对原操作，不能
当作 summary 重试。

## 验证边界

CPU 模型探测和真实文本评分可以验证模型代码与评测链，模拟控制器测试用于
验证调度、回传和排名。它们均不替代真实 NPU/CUDA 训练验收。运行记录应分别
记录 CPU、NPU、CUDA 和真实 XLab 两轮闭环的结果；没有资源或凭据时保持待验证。
