# SURE Master 技术手册

本文面向需要维护、扩展或审查 `sure_master` 的工程人员。`USAGE.md` 侧重
如何配置和运行，本手册侧重系统内部如何工作、如何实现自进化、启动一个
自进化进程后每一步会做什么，以及每类数据在流程中的定位。

## 1. 系统定位

`sure_master` 是 EvoMaster 中面向 SURE 语音评测体系的自进化
playground。它的核心思想是：

```text
LLM 生成候选 run_sure.py
-> 在独立实验工作区执行候选脚本
-> 候选脚本生成标准 artifact
-> EvoMaster 调用 SURE 只读评分
-> 根据评分继续 debug、research、improve 和知识沉淀
```

这里的“自进化”不是训练 LLM 本身，而是围绕候选程序和实验策略形成闭环。
系统不断改进的是候选脚本中的模型调用方式、训练或解码参数、后处理逻辑、
artifact 生成方式，以及在允许范围内保存到当前实验工作区的模型代码、
checkpoint 和配置。

核心边界：

- 候选脚本负责生成 artifact，不负责评分。
- SURE 是只读 metric backend，由框架统一调用。
- 候选脚本不能 import SURE、调用 SURE、修改 SURE 源码目录。
- required base model 必须通过当前实验工作区的 `base_model/*` 路径使用。
- 候选脚本只能写当前 experiment workspace，例如 `models/`、`working/`、
  `artifacts/`。

## 2. 关键模块

主要模块关系如下：

```text
run.py
  -> SureMasterPlayground
       -> SureRunExp
            -> generated run_sure.py
            -> artifact guards
            -> SureMetricRunner
       -> ResearchExp / KnowledgePromotionExp / WisdomPromotionExp
       -> SureMasterLocalSession / SureMasterLocalEnv

task_cards/sure_tasks.yaml
  -> SureTaskCard
  -> BaseModelProfile
```

关键文件：

| 文件 | 作用 |
| --- | --- |
| `core/playground.py` | 注册 `sure_master`，编排完整自进化流程 |
| `core/exp/run_exp.py` | 生成、执行、debug 候选 `run_sure.py`，并触发评分 |
| `core/utils/metric.py` | SURE 只读评分适配器 |
| `core/utils/task_cards.py` | 任务卡和基础模型 profile 的解析与校验 |
| `core/utils/code.py` | 候选代码提取和越界静态检查 |
| `core/utils/vc_remote.py` | F5-TTS mixed local/VC 远端训练调度 |
| `env/local.py` | SURE Master local workspace 和 symlink 准备 |
| `task_cards/sure_tasks.yaml` | 内置任务、metric、required roles 和 artifact contract |

`SureMasterPlayground` 是总控。它加载配置、解析任务卡、准备 session、控制
research 轮数、维护 best solution，并决定一个候选结果是否优于当前最佳。

`SureRunExp` 是单个候选实验的执行单元。它负责让 agent 产出
`run_sure.py`，保存代码，做静态边界检查，执行候选脚本，检查 required
artifact，调用 `SureMetricRunner`，并把 terminal output 和 metric feedback
交给 debug agent。

`SureMetricRunner` 是 SURE 的适配层。它会构建 SURE pipeline spec，把 task
card 中的 role path 转成 SURE CLI adapter 需要的参数，然后调用 SURE
pipeline，输出 `metric/pipeline_spec.json` 和 `metric/score_summary.json`。

## 3. 自进化如何实现

### 3.1 闭环

`sure_master` 的自进化闭环由五类动作组成：

```text
生成 -> 执行 -> 评分 -> 选择 -> 反思
```

更具体地说：

1. `draft_agent` 生成初始候选脚本。
2. `SureRunExp` 执行候选脚本，候选脚本生成 artifact。
3. `SureMetricRunner` 调用 SURE 得到真实 metric score。
4. 如果失败，`debug_agent` 根据代码、terminal output 和 metric feedback
   修复候选脚本。
5. 如果成功，系统把新分数和当前 `best_score` 比较。
6. `reseach_agent` 基于当前 best、历史方案和历史结果提出下一批改进 idea。
7. `improve_agent` 针对每个 idea 生成新候选脚本。
8. 每个 improve 候选独立执行、评分、比较，优者成为新的 best。
9. `knowledge_promotion_agent` 总结本轮经验，影响下一轮 research。

分数比较由任务卡决定。对于 WER、CER、DER、tts_wer 等指标，越低越好；
对于 accuracy、BLEU 等指标，越高越好。实现上使用 `task_card.metric_direction`
和 `compare_score()` 统一处理。

### 3.2 自进化状态

一次 run 中，系统维护以下核心状态：

| 状态 | 含义 |
| --- | --- |
| `initial_code` | draft 阶段得到的初始候选代码 |
| `best_solution` | 当前正式最佳候选代码 |
| `best_score` | 当前最佳 SURE 分数 |
| `real_time_best_solution` | 实时最佳代码，超时时用于 wisdom promotion |
| `research_plan_and_result` | 历轮 research plan 和 knowledge summary |
| `prefetch_descriptor` | prefetch 阶段生成的任务理解摘要 |

这些状态构成“记忆”。下一轮 research 不是从零开始，而是看到此前的计划、
尝试结果、成功和失败经验，再提出更有针对性的 idea。

### 3.3 多 agent 分工

| Agent | 作用 | 主要输入 | 主要输出 |
| --- | --- | --- | --- |
| `prefetch_agent` | 形成任务和数据理解 | task description、task card、base model profile | descriptor、data/model knowledge |
| `draft_agent` | 生成初始 `run_sure.py` | task、data preview、role paths、execution env | 初始候选代码 |
| `debug_agent` | 修复失败候选 | buggy code、terminal output、metric feedback | 修复后的候选代码 |
| `reseach_agent` | 提出改进方向 | initial code、best code、历史结果 | JSON research plan |
| `improve_agent` | 实现某个 idea | previous solution、idea、任务上下文 | 改进候选代码 |
| `knowledge_promotion_agent` | 总结本轮经验 | research plan、各 idea 分数、best code | 文本总结 |
| `wisdom_promotion_agent` | 超时或结束时沉淀经验 | best solution、task card | 可复用 JSON wisdom |

注意：代码中 agent key 保持为 `reseach_agent` 和 `prompts/reseach_*.txt`，
这是当前实现约定，不应在未做兼容迁移时随意改名。

### 3.4 `reseach_agent` 的实现方式

`reseach_agent` 不是执行候选代码的 agent，也不直接生成新的 `run_sure.py`。
它的职责是生成“下一轮该尝试什么”的结构化计划。具体实现由
`core/exp/research_exp.py` 中的 `ResearchExp` 包装。

每一轮 research 开始时，`SureMasterPlayground` 会创建一个 `ResearchExp`，
并把当前上下文传给它：

```text
task_description
data_preview
task_card
base_model_profile
initial_code
best_code
research_plan_and_result_text
candidate_type_guidance
```

这些字段会被写入 `reseach_agent` 的 prompt format kwargs，然后通过
`research_agent.run(TaskInstance(...))` 触发一次 LLM 调用。对应 user prompt
是 `prompts/reseach_user_prompt.txt`。

该 prompt 明确要求 agent 输出 JSON，而不是代码或 shell 命令：

```json
{
  "major direction 1": {
    "1": "specific idea",
    "2": "specific idea"
  },
  "major direction 2": {
    "1": "specific idea"
  },
  "major direction 3": {
    "1": "specific idea"
  }
}
```

`ResearchExp` 会从 LLM trajectory 中抽取最后响应，并用
`_parse_json_from_response()` 解析 JSON。这个解析器会优先找 key 以
`major direction` 开头的对象，也能容忍模型把 JSON 包在 markdown code
block 中。解析成功后返回的 `research_plan` 会交回
`SureMasterPlayground`。

随后 playground 会遍历 `research_plan.items()`：

- 外层 key 是 major direction。
- 内层 key/value 是某个方向下的具体 idea。
- 每个 idea 会创建一个独立的 improve experiment。
- `improve_agent` 才负责把这个 idea 落成新的 `run_sure.py`。
- `SureRunExp` 再负责执行、artifact 检查和 SURE 评分。

因此 research 阶段的本质是“规划器”：它把历史 best、失败经验和任务约束
转化成下一轮可执行的搜索空间。真正的代码生成和评分发生在 improve 阶段。

### 3.5 候选类型和混合执行

普通任务中，idea 都按本地候选执行。mixed local/VC 模式下，research idea
会被拆成三个明确类别：

- `[inference]`：只改推理参数、文本处理、采样、后处理，使用本地 GPU。
- `[fine_tune]`：训练或微调权重，但不改变模型结构，提交到 VC 子任务。
- `[arch]`：训练结构变体，必须改变模型结构或参数量，提交到 VC 子任务。

当前 regular mixed 配置要求每轮 research 输出 4 个 `[inference]`、2 个
`[fine_tune]` 和 2 个 `[arch]` idea。旧标签 `[training]` 仍可兼容解析，
但在新流程中会被归一化为 `[fine_tune]`；如果旧 `[training]` idea 明确包含
结构参数变更，系统会提升为 `[arch]`。

`candidate_type_from_idea()` 和 `candidate_type_from_code()` 会根据 idea 前缀
或代码 marker 判断候选类型。训练级候选通过
`VcRemoteTrainingExecutor` 提交远端 VC job，子任务执行同一套候选执行、
artifact guard 和 SURE metric 流程，并把结果 JSON 写回共享 workspace。

### 3.6 分阶段三轴搜索

当配置 `sure.search_strategy: staged_axes` 或 `sure.staged_axes.enabled: true`
时，`SureMasterPlayground` 不走默认 mixed research/improve loop，而是执行
固定的三轴协议：

```text
Stage0 draft/baseline
Stage1 arch successive halving: 16 -> 8 -> 2
Stage2 train strategy successive halving: 16 -> 8 -> 2
Stage3 inference successive halving: 16 -> 8 -> 3
Stage4 limited combination: top-2 arch × top-2 train × top-3 inference
Stage5 selection rerank: search best + top-5 runner-up + baseline/draft
Stage6 optional holdout: final candidate + baseline/draft
```

Stage1-3 用 axis-specific research request 生成单一类型 idea，再用
`SureRunExp` 评测。successive halving 的后续 rung 通过
`run_existing_code()` 复跑同一份候选代码，并注入更高预算的 execution env；
不会调用 debug agent 修改代码。Stage4 用 improve agent 生成组合候选，但
组合 prompt 明确要求重新训练 `arch × train` checkpoint 后再跑 inference。

selection 和 holdout 可按 phase 覆盖 role paths、execution env 和 base model
source symlink。远端 VC 子任务通过 experiment workspace 下的
`metric/remote_candidate_context.json` 读取这些覆盖，保证 local/remote 运行
使用同一套 staged context。所有阶段排行榜和 summary 写入
`<workspace>/staged_axes/`。

## 4. 启动后的完整流程

典型启动命令：

```bash
python run.py \
  --agent sure_master \
  --config configs/sure_master/kimi-2.6.yaml \
  --task playground/sure_master/data/asr_en_wer_zipformer_description.md
```

启动后，系统会经历以下阶段。

### 4.1 入口和配置加载

`run.py` 自动导入 playground，找到注册名 `sure_master` 对应的
`SureMasterPlayground`。随后加载：

- `--config` 指定的 YAML。
- `--task` 指定的任务描述文件。
- LLM 配置、agent prompt、session 配置。
- `sure` 配置块。

关键配置字段：

| 字段 | 作用 |
| --- | --- |
| `sure.root` | SURE 仓库根目录 |
| `sure.pythonpath` | SURE Python 源码路径 |
| `sure.task_id` | 当前优化任务 |
| `sure.task_cards_path` | 任务卡文件 |
| `sure.inputs` | 输入或 artifact role path 覆盖 |
| `sure.execution_env` | 执行 `run_sure.py` 时注入的环境变量 |
| `sure.base_models` | base model source paths 和 profile override |
| `session.local` | workspace、GPU、并行和 symlink 配置 |

实际可用配置位于 `configs/sure_master/`，例如 `kimi-2.6.yaml`、
`gpt-5-example.yaml`、`gpt-5-docker.yaml`、`gpt-5-f5tts-*.yaml`。
部分配置名包含 `docker`，但 SURE Master v1 的 session 类型仍是 local；
代码中 Docker session 会直接报不支持。

### 4.2 任务卡和 base model 解析

系统读取 `task_cards/sure_tasks.yaml`，根据 `sure.task_id` 解析
`SureTaskCard`。

任务卡提供：

- canonical task，例如 `asr`、`tts`、`classification`。
- primary metric，例如 `WER`、`tts_wer`、`accuracy`。
- metric direction，例如 `lower` 或 `higher`。
- required roles 和 optional roles。
- artifact contract。
- 可选 base model profile。

如果 `sure.require_base_model` 为 true，当前任务必须解析出 base model
profile。对于 `asr_en_wer`，任务卡内置了 Zipformer profile，但真实路径必须
由配置里的 `sure.base_models.asr_en_wer.source_paths` 提供。

base model 解析完成后，系统会把外部 source path 注入 session symlinks，
使每个实验工作区都能看到类似：

```text
base_model/recipe
base_model/data
base_model/root
```

候选代码必须使用这些 workspace-relative 路径，不能硬编码外部 source path。

### 4.3 Metric runner 和 session 准备

`SureMetricRunner` 根据 `sure.root`、`sure.pythonpath`、`sure.device`、
`sure.cache_dir` 创建。它只在候选脚本结束后由框架调用。

local session 准备内容：

- 创建主 workspace。
- 创建 `best_solution/`、`artifacts/`、`models/`、`metric/`、`working/`。
- 创建 base model symlink。
- 对 icefall 兼容额外创建 `base_model/recipe/data -> ../data` 和
  `workspace/data -> base_model/data`。
- 自动解析 GPU 配置。

GPU 配置规则：

- `gpu_devices: null`、`auto` 或 `all` 会尝试发现所有可见 GPU。
- 并行开启时，按 `max_parallel` 和 `gpus_per_exp` 给 experiment 分配资源。
- ASR 可设置 `set_asr_world_size`，使候选训练脚本看到正确的
  `ASR_WORLD_SIZE`。

### 4.4 构建数据预览

进入候选生成前，系统会构建 `data_preview`。它包含：

- task card JSON。
- base model profile JSON。
- 配置后的 role paths。
- 可读取输入文件的前若干内容。

例如 ASR 任务会预览 `ref` 文件；TTS 任务可能预览 `prompts.jsonl`。这些
内容进入 agent prompt，帮助 LLM 生成符合数据格式的候选脚本。

### 4.5 Prefetch 阶段

`PrefetchExp` 使用 `prefetch_agent` 对 task description、task card 和 base
model profile 做轻量理解，输出 descriptor。当前 v1 不依赖外部 wisdom
database，但保留了以后做 RAG 或经验检索的接口。

Prefetch 失败不会使整个 run 失败；系统会记录 warning 并继续 draft。

### 4.6 Draft 阶段

系统创建一个 `SureRunExp(stage="draft")`。如果配置中有
`sure.initial_solution_path`，draft 直接读取该文件作为初始候选。例如
`gpt-5-docker.yaml` 可使用 Zipformer baseline 作为初稿。

ASR/F5-TTS 的正式实验应把 draft 固定为官方强 baseline：

- ASR 使用 `zipformer_large_cr_ctc_rnnt_baseline.py`，并通过
  `SURE_BASELINE_CHECKPOINT_DIR` 指向 icefall 官方 checkpoint 的 `exp/`
  目录。官方 HF 包提供 `pretrained.pt` 和 `epoch-50.pt`；配置
  `SURE_BASELINE_USE_PRETRAINED=1` 时 baseline 会用 `pretrained.pt`
  作为官方已导出的 averaged checkpoint。若关闭 pretrained 模式，baseline
  会 fail fast 校验 icefall averaging 所需的完整 epoch checkpoint 序列。
- F5-TTS 使用 `f5tts_v1_base_official_baseline.py`。该脚本只调用
  `SURE_TTS_BATCH_INFER_WRAPPER`，不触发训练或结构修改，并写
  `artifacts/official_baseline.json`。

这样 Stage0 的分数就是 official draft/baseline 分数，Stage5 selection 和
Stage6 holdout 中的 baseline 对照也来自同一份初始候选。

否则 `draft_agent` 会根据以下输入生成候选 `run_sure.py`：

- task description。
- data preview。
- task card。
- base model profile。
- role paths。
- execution env。
- prefetch 得到的 data/model knowledge。

输出必须是一个 Python 代码块。`SureRunExp` 从模型响应中提取代码，并保存到
当前 experiment workspace 下的 `run_sure.py`。

### 4.7 候选代码边界检查

执行前，`validate_sure_candidate_boundary()` 会静态检查候选代码。常见拒绝
原因包括：

- import 或引用 `sure_eval`。
- 硬编码 SURE root。
- required base model 存在，但代码没有引用 `base_model/*`。
- 直接引用 base model 外部 source path。
- ASR subprocess 使用 `capture_output=True` 或 `stdout=subprocess.PIPE`。
- Zipformer `train.py` 错误传入 `--lang-dir`。

边界检查失败时，候选不会执行，SURE metric 也不会运行。

### 4.8 候选脚本执行

候选脚本从当前 experiment workspace 执行：

```text
<workspace>/exp_X_*/run_sure.py
```

执行命令由 `_execution_command()` 构建。它会：

- 注入 `sure.execution_env`。
- 支持环境变量覆盖运行时 `SURE_*`、`ASR_*`、`HF_*` 等值。
- 处理 `SURE_MAX_DURATION=auto`，通过 `runtime_env.py` 解析安全 batch
  duration。
- 使用 `SURE_RUN_TIMEOUT` 控制候选脚本执行超时；显式设置为 `0` 表示候选进程无执行期限。

对于通过 VC 执行的训练，需要区分三层期限：

- `SURE_BASELINE_TRAIN_TIMEOUT` 限制 Zipformer 训练子进程；显式设置为 `0` 表示不限制训练时长。
- `SURE_RUN_TIMEOUT` 限制 VC child 内完整 `run_sure.py` 候选进程；显式设置为 `0` 表示不限制候选时长。
- `sure.remote_training.submit_timeout` 限制本地 coordinator 同步等待 `vc submit` 的时长；显式设置为 `0` 表示 coordinator 也无限等待。

长时间 draft 训练可以将这三个值都设为 `0`，使训练子进程、VC child
和本地 coordinator 都不受时间限制。此时 coordinator 会一直停留在当前远程
候选上，直到 `vc submit --sync` 返回；只有在配置正数 submit timeout 时，
超时后的 `remote_training.result_recovery` 和 staged-axes
`late_result_recovery` 才用于消费 VC child 较晚写入的结果。普通搜索候选应
继续使用正数期限，避免失控任务无限运行。

候选脚本的职责：

- 读取只读输入和 base model。
- 运行训练、推理、解码或后处理。
- 生成任务卡声明的 required artifact。
- 出错时非零退出。

候选脚本不应打印或伪造 score，也不应调用 SURE。

### 4.9 Required artifact 和 artifact guard

候选脚本正常退出后，`SureRunExp` 先检查 required roles 对应文件是否存在。
例如：

- ASR `asr_en_wer` 必须有 `artifacts/hyp.txt`。
- TTS `tts_en_wer` 必须有 `artifacts/samples.jsonl`。

随后执行任务相关 artifact guard：

ASR guard：

- 如果 `hyp.txt` 大规模复制 `ref.txt`，拒绝。
- 如果覆盖率很高但 transcript 多样性极低，拒绝。
- 目的是防止用 reference、supervision manifest 或常量 token 伪造 ASR 输出。

TTS guard：

- `samples.jsonl` 必须是有效 JSONL。
- `prediction_audio` 文件不能过小。
- `prediction_audio` 不能指向或复制 `reference_audio`。
- 多样本任务不能全部复用同一个预测 wav。

artifact guard 只判断明显不可信或格式错误的产物；真正 metric 仍由 SURE 计算。

### 4.10 SURE 评分

artifact 检查通过后，`SureMetricRunner.run()` 会：

1. 把 SURE `pythonpath` 放入 `sys.path` 和 `PYTHONPATH`。
2. 调用 SURE CLI adapter 构建 pipeline spec。
3. 把 task card role paths 转为 SURE 运行参数，例如 `ref -> ref_file`、
   `hyp -> hyp_file`、`samples_jsonl -> samples_jsonl`。
4. 调用 SURE `run_pipeline_spec()`。
5. 写出 metric 文件。

成功时输出：

```text
metric/pipeline_spec.json
metric/score_summary.json
```

失败时输出：

```text
metric/metric_error.json
```

SURE import 期间会临时设置 `sys.dont_write_bytecode = True`，避免向只读 SURE
源码树写 `.pyc`。

### 4.11 Debug 阶段

如果候选失败，`SureRunExp` 最多调用 `debug_agent` 三次。debug prompt 会拿到：

- task description。
- task card。
- base model profile。
- role paths。
- execution env。
- buggy code。
- terminal output。
- metric feedback。

debug 产生的新代码会再次经过边界检查、执行、artifact guard 和 SURE metric。

### 4.12 Research 和 Improve 阶段

draft 成功后，系统进入最多 `max_research_rounds` 轮 research/improve。

每轮 research：

- `reseach_agent` 看到 initial code、current best code、data preview 和历史
  research 结果。
- 输出 JSON 结构的 major directions 和具体 idea。
- mixed execution 模式下，idea 应按 4/2/2 输出 `[inference]`、`[fine_tune]`
  和 `[arch]` 前缀。

research 输出不会直接执行。系统会先把 JSON plan 解析成：

```text
direction -> idea_id -> idea_text
```

然后按配置限制裁剪：

- `max_improve_directions_per_round` 控制每轮最多尝试多少个方向。
- `max_ideas_per_direction` 控制每个方向最多尝试多少个 idea。
- mixed execution 模式下，`candidate_type_from_idea()` 会识别
  `[inference]`、`[fine_tune]` 和 `[arch]`，并应用每轮候选数量限制。

每个 idea 会创建一个 improve experiment：

- 复制 `improve_agent` 和 `debug_agent`，避免并行状态互相污染。
- 使用当前方向的 best solution 作为 previous solution。
- 生成新的候选 `run_sure.py`。
- 独立执行和评分。

如果 `split_workspace_for_exp` 开启，每个 improve experiment 使用独立目录：

```text
<main_workspace>/exp_N_improve/
```

所有 idea 完成后，系统按 task metric direction 选择该方向中最优的候选，并
更新全局 best。

### 4.13 Knowledge promotion 和结束

每轮 improve 完成后，`knowledge_promotion_agent` 会总结：

- 哪些 idea 改善了 metric。
- 哪些失败或应避免。
- 下一轮应尝试什么。
- artifact contract 中发现了哪些坑。
- base model 相关约束或可复用改法。

总结文本追加到 `research_plan_and_result`，下一轮 research 会看到这些经验。

结束条件：

- 达到 `max_research_rounds`。
- 全局 watchdog 超时。
- fatal exception。

正常结束返回：

```text
status
task_id
metric
best_score
is_lower_better
```

当前最佳代码保存到：

```text
<main_workspace>/best_solution/best_solution.py
```

超时时，系统会尝试运行 `WisdomPromotionExp`，从 `real_time_best_solution` 中
提炼可复用经验。

## 5. 流程中的数据定位

一次自进化 run 中会同时出现多类数据。理解每类数据的定位，是避免数据泄漏
和错误扩展的关键。

| 数据类型 | 来源 | 使用位置 | 定位 |
| --- | --- | --- | --- |
| 任务描述 | `--task` 指向的 Markdown | 所有 agent prompt | 目标说明和任务背景 |
| 任务卡 | `task_cards/sure_tasks.yaml` | playground、agent prompt、metric runner | 系统契约 |
| 运行配置 | `configs/sure_master/*.yaml` | 初始化、session、候选执行 | 运行参数和资源策略 |
| 输入数据 | `sure.inputs` 或 task card 默认路径 | data preview、candidate、SURE metric | 只读评测输入 |
| Base model | `sure.base_models.*.source_paths` | candidate | 只读优化起点 |
| Candidate 代码 | LLM 输出或 initial solution | `run_sure.py` | 可变候选方案 |
| Candidate 中间文件 | experiment workspace | candidate | 可写临时和模型产物 |
| Artifact | `artifacts/` | artifact guard、SURE metric | 候选提交给 SURE 的标准产物 |
| Metric 报告 | `metric/` | best 选择、debug feedback | 唯一可信评分来源 |
| Evolution memory | playground 内存状态 | research、knowledge promotion | 驱动下一轮自进化 |
| Best solution | `best_solution/best_solution.py` | 最终输出、超时总结 | 当前最优候选代码 |

### 5.1 输入数据

输入数据是只读的。对于 ASR，典型输入是 `ref.txt`；对于 TTS/VC，典型输入是
`prompts.jsonl` 或 samples 描述。

如果 `sure.inputs` 配置了绝对路径，框架会在每个 experiment 开始前把 required
input 复制到任务卡默认 `input/` 路径。例如：

```text
/abs/path/to/ref.txt
-> <exp_workspace>/input/ref.txt
```

这样 legacy candidate 可以读 `input/ref.txt`，同时原始输入不会被候选脚本
写回。

### 5.2 Base model 数据

Base model 数据通过 symlink 暴露给当前 workspace：

```text
base_model/recipe
base_model/data
base_model/root
```

它的定位是“只读起点”。候选可以基于它训练 workspace-local checkpoint，
也可以复制配置或写 wrapper，但不能直接修改 symlink 背后的外部源目录。

### 5.3 Candidate 产物

候选脚本可写：

```text
models/
working/
artifacts/
```

其中：

- `models/` 存放 checkpoint、模型代码、局部配置。
- `working/` 存放日志、临时文件、patch 后的本地副本。
- `artifacts/` 存放 SURE required artifact。

只有 `artifacts/` 中符合任务卡 contract 的文件会进入 SURE metric。

### 5.4 Metric 数据

`metric/` 是框架写入的评分区域，不应由候选脚本伪造。主要文件：

```text
metric/pipeline_spec.json
metric/score_summary.json
metric/metric_error.json
```

系统选择 best 只看 SURE 返回的 score，而不是候选脚本打印的内容。

### 5.5 数据分层定位

自进化系统会反复看数据、改方案、比较候选。因此数据分层必须明确：

| 阶段 | 定位 |
| --- | --- |
| smoke | 链路检查，不用于效果结论 |
| early search | 快速排错和粗筛方向 |
| regular search | 正式搜索和日常比较 |
| selection | 对少数候选做稳定性确认 |
| holdout/test | 最终评测，不进入搜索和调参 |
| official benchmark | 对齐公开 benchmark，不参与日常自进化 |

基本原则：

- 被 agent 反复看到或用于选择方案的数据，不是最终测试集。
- 被模型训练用到的数据，不是评测集。
- holdout/test 不进入 prompt、不参与 debug、不参与 improve 选择。

## 6. 典型任务流程

### 6.1 ASR Zipformer `asr_en_wer`

`asr_en_wer` 的目标是生成 `artifacts/hyp.txt`，由 SURE 计算英文 WER。

关键数据：

| 数据 | 位置 | 定位 |
| --- | --- | --- |
| ref | `input/ref.txt` 或 `sure.inputs.ref` | 只读参考文本 |
| recipe | `base_model/recipe` | icefall Zipformer recipe |
| data | `base_model/data` | LibriSpeech prepared data |
| root | `base_model/root` | icefall root |
| hyp | `artifacts/hyp.txt` | 候选 ASR 输出 |

候选脚本通常会：

1. 检查 `base_model/recipe`、`base_model/data`、`base_model/root`。
2. 根据 `SURE_MAX_TRAIN_EPOCHS` 和 `SURE_DECODE_ONLY` 决定是否训练。
3. 用 workspace-local `models/` 保存 checkpoint。
4. 根据 `SURE_ASR_EVAL_SPLITS` decode dev 或 test split。
5. 从 icefall `recogs-*.txt` 收集 hypothesis。
6. 归一化 LibriSpeech cut id，例如去掉 `1089-134686-0000-0` 的尾部 segment。
7. 按 ref key 顺序写 `artifacts/hyp.txt`。

搜索阶段应使用 dev 分层 ref，例如：

```text
playground/sure_master/data/asr_librispeech_regular_ref.txt
SURE_ASR_EVAL_SPLITS=dev-clean,dev-other
```

最终 benchmark 才切换：

```text
playground/sure_master/data/asr_en_wer_ref.txt
SURE_ASR_EVAL_SPLITS=test-clean,test-other
```

### 6.2 F5-TTS `tts_en_wer`

`tts_en_wer` 的目标是生成音频和 `artifacts/samples.jsonl`，由 SURE 使用 ASR
语义转写后计算 TTS WER。

关键数据：

| 数据 | 位置 | 定位 |
| --- | --- | --- |
| eval prompts | `base_model/eval_data/prompts.jsonl` | 只读评测 prompts |
| F5-TTS root | `base_model/root` | 只读模型源码 |
| train manifests | `base_model/train_data` | 受控 fine-tune 训练数据 |
| wavs | `artifacts/wavs/*.wav` | 候选生成音频 |
| samples | `artifacts/samples.jsonl` | SURE TTS 输入 |
| candidate changes | `artifacts/candidate_changes.json` | 候选改动记录 |

候选脚本通常会：

1. 读取 `base_model/eval_data/prompts.jsonl`。
2. 调用 `SURE_TTS_BATCH_INFER_WRAPPER` 做批量推理，不要对每条 prompt
   单独启动 `infer_cli.py`。
3. 由 wrapper 把音频写到 `artifacts/wavs/`。
4. 由 wrapper 写 `artifacts/samples.jsonl`，其中 `prediction_audio` 相对
   `samples.jsonl` 解析。
5. 由 wrapper 写 `artifacts/candidate_changes.json`，记录候选类型、改动字段、
   默认值、diff 和产物路径。
6. 如果使用普通微调，只能调用 `SURE_TTS_FINETUNE_WRAPPER`。
7. 如果使用结构搜索，只能调用 `SURE_TTS_ARCH_WRAPPER`。
8. fine-tune 成功后用 `models/f5tts_finetune/final_checkpoint.pt` 做 inference；
   结构训练成功后用 `models/f5tts_arch/final_checkpoint.pt` 和
   `models/f5tts_arch/model_cfg.yaml` 调用 batch inference wrapper。

`SURE_TTS_BATCH_INFER_WRAPPER` 会在每个 worker 内只加载一次 F5-TTS 模型和
vocoder，并按当前候选可见的 `CUDA_VISIBLE_DEVICES` 切分 prompts。这样避免
`N` 条样本启动 `N` 次 Python/模型加载。

F5-TTS 结构搜索第一版是受控白名单，不允许任意改源码。允许搜索：

```text
depth: 18, 20, 22, 24
ff_mult: 2, 3, 4
conv_layers: 2, 4, 6
qk_norm: none, rms_norm
attn_mask_enabled: 0, 1
checkpoint_activations: 0, 1
```

结构 wrapper 会从 base checkpoint 部分加载 shape 匹配的权重，匹配率低于
`SURE_TTS_ARCH_MATCH_THRESHOLD` 时失败。训练有 `max_steps` 硬上限，并支持
基于 smoothed training loss 的 early stop。selection 固定候选复评时会复制
regular search 候选已有的 `models/f5tts_finetune`、`models/f5tts_arch` 和
`candidate_changes.json`，结构 wrapper 默认复用匹配的已有 checkpoint。

F5-TTS 数据已分层：

```text
smoke/          20 条
early_search/   200 条
regular_search/ 800 条
selection/      500 条
holdout/        1000 条
```

训练数据来自：

```text
f5tts_train_manifests/libritts_train_clean_100_1h
f5tts_train_manifests/libritts_train_clean_100_5h
f5tts_train_manifests/libritts_train_clean_100_10h
```

候选脚本不能把 eval/search/selection/holdout 数据用于训练。

## 7. 任务卡和 Artifact 契约

任务卡是 `sure_master` 和 SURE 之间的结构化契约。一个任务卡至少要说明：

- `canonical_task`
- `task_alias`
- `primary_metric`
- `metric_direction`
- `required_roles`
- `artifact_contract`

以 ASR 为例：

```yaml
required_roles: [ref, hyp]
artifact_contract:
  ref: input/ref.txt
  hyp: artifacts/hyp.txt
```

含义是：候选脚本必须生成 `artifacts/hyp.txt`，SURE 会用 `input/ref.txt`
或配置覆盖后的 ref 和 hyp 计算 metric。

新增任务时，应同时确认：

1. SURE 是否支持该 task、language、metric route。
2. required roles 是否和 SURE adapter 一致。
3. artifact_contract 是否能被候选脚本稳定生成。
4. 是否需要 required base model profile。
5. `sure.inputs` 是否提供所有只读输入。
6. 是否需要新增 artifact guard。
7. 是否有独立的 search/selection/holdout 数据划分。

## 8. 安全边界和可信评测

`sure_master` 的可信性依赖三层约束。

第一层是 prompt 约束。draft、debug、improve prompt 都明确要求候选脚本：

- 不调用 SURE。
- 不写 SURE。
- 不写 base model 外部源目录。
- 不伪造 score。
- 不复制 reference 作为预测。

第二层是静态检查。`validate_sure_candidate_boundary()` 会在执行前拒绝明显
越界代码。

第三层是 artifact guard。它不替代 SURE metric，但会拒绝明显作弊或无效产物。

这些约束的目标不是限制候选优化能力，而是保证“分数来自同一套 SURE
pipeline”，并保证实验结果可复现、可比较。

## 9. 运维和排错

### 9.1 验证命令

运行单测：

```bash
python -m unittest playground.sure_master.core.utils.test_sure_master
```

语法检查：

```bash
python -m py_compile \
  playground/sure_master/core/playground.py \
  playground/sure_master/core/exp/*.py \
  playground/sure_master/core/utils/*.py \
  playground/sure_master/env/local.py \
  playground/sure_master/agent/session/local.py
```

### 9.2 按阶段定位失败

| 阶段 | 常见问题 | 排查方向 |
| --- | --- | --- |
| 初始化 | unknown task、base model profile 缺失 | 检查 `sure.task_id` 和 `sure.base_models` |
| session | symlink 不存在、GPU 分配异常 | 检查 source paths、`gpu_devices`、并行配置 |
| draft | 没有生成 Python 代码 | 检查 LLM 输出和 prompt |
| 边界检查 | candidate 被拒绝 | 查看 boundary errors |
| 执行 | 非零退出或 timeout | 查看 terminal output、`working/*.log` |
| artifact | required artifact missing | 检查任务卡 contract 和候选输出路径 |
| guard | copy ref、空音频、复用 wav | 检查 artifact 内容 |
| SURE metric | 模型加载或环境失败 | 检查 `metric/metric_error.json`、cache、device |
| research | idea JSON 解析失败 | 检查 research agent 输出是否为纯 JSON |
| remote training | VC 提交失败或未写结果 | 查看 `metric/remote_training_*` 文件 |

### 9.3 常见配置问题

- `sure.require_base_model: true` 时，非内置 base model 的任务必须配置
  `sure.base_models.<task_id>`。
- required profile 中每个 `required_paths` 都必须有对应 `source_paths`。
- 并行时应开启 `split_workspace_for_exp`，避免多个候选写同一目录。
- ASR 如果没有 checkpoint 且 `SURE_DECODE_ONLY=1`，候选应明确失败。
- ASR 如果配置了 `SURE_BASELINE_CHECKPOINT_DIR`，官方 baseline 必须找到
  averaged-model decode 所需的完整 checkpoint；缺文件不会退回短训。
- F5-TTS 在离线环境中需要提前准备 Hugging Face 或 ModelScope cache。
- Zipformer ASR 的 `SURE_MAX_DURATION=auto` 由
  `SURE_ASR_ZIPFORMER_WRAPPER`/baseline 在构造最终 `train.py` 结构/loss 参数后
  通过 `SURE_RUNTIME_ENV_HELPER resolve-max-duration` 解析。生成候选不要直接调用
  helper；候选只把最终 train/decode extra args 传给 wrapper，由 wrapper 设置
  `SURE_DURATION_TRAIN_ARGS_JSON`。duration autotune 是 fail-closed：所有 probe
  失败时不会返回未验证的最小值；非 OOM 的启动/导入/CLI/数据错误不会继续降
  `max-duration` 伪装成显存不足。probe 目录按候选/cache key 隔离，并且默认看到
  少量正常 training batch log 即可成功，不要求跑完整 epoch。ASR Zipformer
  baseline 在正式训练阶段负责 OOM 降档重试，并输出
  `artifacts/resource_profile.json`。

## 10. 扩展 Checklist

新增任务或新模型接入时，建议按以下顺序：

1. 在 SURE 中确认 task、metric、required roles 和 pipeline route。
2. 在 `task_cards/sure_tasks.yaml` 添加或更新任务卡。
3. 配置 base model profile，包括 workspace-relative `required_paths`。
4. 在运行配置中提供真实 `source_paths`。
5. 准备 search、selection、holdout 数据，并明确哪些能进入 prompt。
6. 写任务描述 Markdown，说明候选脚本应生成的 artifact。
7. 如有必要，新增数据构建工具或 baseline。
8. 如存在明显作弊风险，新增 artifact guard。
9. 运行 `test_sure_master`。
10. 使用 smoke 数据先跑通完整链路，再扩大到 regular search。

## 11. 阅读路径建议

第一次接手 `sure_master` 时，推荐按以下顺序阅读：

1. 本技术手册，理解系统机制。
2. `USAGE.md`，理解运行配置和常见命令。
3. `task_cards/sure_tasks.yaml`，理解任务契约。
4. `core/playground.py`，理解总控流程。
5. `core/exp/run_exp.py`，理解候选执行和评分。
6. 当前任务描述文件，例如 `data/asr_en_wer_zipformer_description.md` 或
   `data/tts_en_wer_f5tts_description.md`。

这样可以先建立“自进化闭环”和“数据定位”的整体模型，再进入具体任务实现。
