# SURE Master 使用说明

本文说明 `sure_master` 的用途、配置方式、运行命令、输入/输出约定、数据集分层、任务卡格式、安全边界和常见问题。

## 1. 功能定位

`sure_master` 是 EvoMaster 中面向 SURE 语音评测体系的自进化 playground。它会让大模型生成候选 Python 程序，在独立实验工作区中运行候选程序，再调用 SURE 计算指标，并根据指标结果继续 debug、research、improve 和知识沉淀。

核心边界：

- 候选代码可以修改模型结构、训练逻辑、推理逻辑、解码策略、后处理和产物生成方式。
- 候选代码可以在当前实验工作区保存模型代码、配置、checkpoint、音频、预测结果和中间文件。
- SURE 只作为只读评测工具使用。候选代码不能 import SURE、不能调用 SURE、不能写入 SURE 源码目录。

默认 SURE 路径：

```text
/hpc_stor03/sjtu_home/chaolei.liu/sure
```

## 2. 相关文件

主要代码和配置：

```text
playground/sure_master/
  README.md
  USAGE.md
  core/playground.py
  core/exp/run_exp.py
  core/utils/metric.py
  core/utils/task_cards.py
  core/utils/code.py
  task_cards/sure_tasks.yaml
  prompts/

configs/sure_master/
  kimi-2.6.yaml
  gpt-5-example.yaml
```

关键文件说明：

| 文件 | 作用 |
| --- | --- |
| `core/playground.py` | 注册并编排 `sure_master` 自进化流程 |
| `core/exp/run_exp.py` | 生成、执行 `run_sure.py`，然后调用 SURE 评测 |
| `core/utils/metric.py` | SURE 只读评测适配器，负责构建 pipeline spec 并运行 SURE |
| `core/utils/task_cards.py` | 加载和解析 SURE 任务卡 |
| `core/utils/code.py` | 提取候选代码并检查 SURE 只读边界 |
| `task_cards/sure_tasks.yaml` | 所有内置语音任务、指标和输入/输出契约 |
| `configs/sure_master/*.yaml` | 运行配置 |
| `prompts/*.txt` | 各阶段 agent 的提示词 |

## 3. 环境准备

进入 EvoMaster 仓库根目录：

```bash
cd /mnt/cloudstorfs/sjtu_home/chaolei.liu/Agent/EvoMaster
```

安装依赖：

```bash
pip install -r requirements.txt
```

`run.py` 的入口会加载 MCP 工具框架，因此当前 Python 环境需要安装 `mcp`。如果报错 `ModuleNotFoundError: No module named 'mcp'`，执行：

```bash
pip install mcp
```

检查 SURE 源码是否存在：

```bash
test -d /hpc_stor03/sjtu_home/chaolei.liu/sure/src && echo "SURE source OK"
```

配置模型 API 环境变量。

Kimi 配置需要：

```bash
export KIMI_API_KEY="..."
export KIMI_BASE_URL="..."
```

GPT/OpenAI 兼容配置需要：

```bash
export OPENAI_API_KEY="..."
export GPT_BASE_URL="..."
export GPT_CHAT_MODEL="..."
```

## 4. 快速运行

使用 Kimi 配置：

```bash
python run.py \
  --agent sure_master \
  --config configs/sure_master/kimi-2.6.yaml \
  --task playground/sure_master/data/asr_en_wer_zipformer_description.md
```

使用 GPT/OpenAI 兼容配置：

```bash
python run.py \
  --agent sure_master \
  --config configs/sure_master/gpt-5-example.yaml \
  --task playground/sure_master/data/asr_en_wer_zipformer_description.md
```

默认任务是：

```yaml
sure:
  task_id: "asr_en_wer"
```

## 5. 选择任务

修改配置文件中的 `sure.task_id`。

示例：

```yaml
sure:
  task_id: "tts_en_wer"
```

当前内置任务：

| task_id | SURE 任务 | 指标 | 优化方向 |
| --- | --- | --- | --- |
| `asr_zh_cer` | 中文 ASR | `CER` | 越低越好 |
| `asr_en_wer` | 英文 ASR | `WER` | 越低越好 |
| `asr_cs_mer` | 中英混合 ASR | `MER` | 越低越好 |
| `s2tt_bleu` | 语音翻译 | `BLEU` | 越高越好 |
| `kws_accuracy` | 关键词检测 | `accuracy` | 越高越好 |
| `slu_accuracy` | 口语理解 | `accuracy` | 越高越好 |
| `classification_accuracy` | 通用分类 | `accuracy` | 越高越好 |
| `ser_accuracy` | 语音情感识别 | `accuracy` | 越高越好 |
| `gr_accuracy` | 性别识别 | `accuracy` | 越高越好 |
| `sd_der` | 说话人分离 | `DER` | 越低越好 |
| `sa_asr_cpwer` | 带说话人归属的 ASR | `cpWER` | 越低越好 |
| `tts_zh_cer` | 中文 TTS | `tts_cer` | 越低越好 |
| `tts_en_wer` | 英文 TTS | `tts_wer` | 越低越好 |
| `vc_zh_cer` | 中文声音转换 | `vc_cer` | 越低越好 |
| `vc_en_wer` | 英文声音转换 | `vc_wer` | 越低越好 |

任务卡定义文件：

```text
playground/sure_master/task_cards/sure_tasks.yaml
```

## 6. 配置字段说明

典型配置：

```yaml
sure:
  root: "/hpc_stor03/sjtu_home/chaolei.liu/sure"
  pythonpath: "/hpc_stor03/sjtu_home/chaolei.liu/sure/src"
  task_cards_path: "./playground/sure_master/task_cards/sure_tasks.yaml"
  task_id: "asr_en_wer"
  device: "cuda"
  cache_dir: null
  validate_env: false
  require_base_model: true
  inputs:
    ref: "/hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster/playground/sure_master/data/asr_librispeech_regular_ref.txt"
  execution_env:
    SURE_ASR_EVAL_SPLITS: "dev-clean,dev-other"
    SURE_MAX_TRAIN_EPOCHS: "1"
    SURE_MAX_DURATION: "600"
    SURE_USE_FP16: "1"
    SURE_DECODE_ONLY: "0"
    SURE_ICEFALL_PYTHON: "/hpc_stor03/sjtu_home/chaolei.liu/anaconda3/envs/icefall/bin/python"

max_research_rounds: 3
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `sure.root` | SURE 仓库根目录 |
| `sure.pythonpath` | SURE Python 源码路径，通常是 `<sure.root>/src` |
| `sure.task_cards_path` | sure_master 的任务卡文件 |
| `sure.task_id` | 当前要优化的任务 ID |
| `sure.device` | SURE 评测使用的设备，例如 `cuda` 或 `cpu` |
| `sure.cache_dir` | SURE 加载模型时可使用的缓存目录 |
| `sure.validate_env` | 是否在评测前调用 SURE 的环境检查 |
| `sure.require_base_model` | 是否要求当前任务必须解析出基础模型 profile，默认 `true` |
| `sure.inputs` | 覆盖任务卡中的输入/产物路径 |
| `sure.execution_env` | 执行候选 `run_sure.py` 时注入的环境变量，例如训练 epoch、batch duration、icefall Python |
| `sure.base_models` | 覆盖任务卡中的基础模型源路径或入口配置 |
| `max_research_rounds` | research/improve 轮数 |

## 7. 输入和产物契约

每个任务卡包含：

- `required_roles`：SURE 必需的输入或产物角色。
- `optional_roles`：SURE 可选输入角色。
- `artifact_contract`：每个角色的默认路径。

例如 `asr_en_wer`：

```yaml
required_roles: [ref, hyp]
artifact_contract:
  ref: input/ref.txt
  hyp: artifacts/hyp.txt
```

含义：

- `ref` 默认读当前实验工作区下的 `input/ref.txt`，除非在 `sure.inputs` 中覆盖。
- 候选脚本必须生成 `artifacts/hyp.txt`。

### 路径解析规则

相对路径会按当前实验工作区解析。

例如：

```yaml
sure:
  inputs:
    ref: "input/ref.txt"
```

在开启独立实验工作区时，实际路径类似：

```text
<run_dir>/workspace/exp_X_*/input/ref.txt
```

共享只读数据推荐写绝对路径：

```yaml
sure:
  inputs:
    ref: "/abs/path/to/ref.txt"
```

如果某个 required input 在 `sure.inputs` 中配置为绝对路径，sure_master 会在每个实验开始前把它复制到任务卡默认的 `input/` 路径。例如 `asr_en_wer` 会得到：

```text
<exp_workspace>/input/ref.txt
```

这样候选脚本既可以读取配置中的绝对路径，也可以读取兼容路径 `input/ref.txt`。复制发生在实验工作区内，不会把候选脚本的写入透传回原始输入文件。

也可以额外配置 symlink，让每个实验工作区都有同一个大输入目录：

```yaml
session:
  local:
    symlinks:
      "/abs/path/to/input": "input"
```

symlink 的格式是：

```text
源路径: 工作区内目标路径
```

### 常见 role

| role | 含义 |
| --- | --- |
| `ref` | 参考文本、RTTM、STM 或标签文件 |
| `hyp` | 候选脚本生成的预测文本、RTTM、STM 或标签文件 |
| `src` | S2TT 可选源文本 |
| `prompt_jsonl` | SLU prompt-choice 元信息 |
| `label_spec` | 分类任务可选标签定义 |
| `reference_jsonl` | KWS 参考 JSONL |
| `sample_output` | KWS 预测输出 |
| `keyword` | KWS 可选关键词 |
| `samples_jsonl` | TTS/VC 的样本列表文件 |

字面量可以用 `literal:` 前缀传入：

```yaml
sure:
  inputs:
    keyword: "literal:hello"
```

## 8. 数据集分层和使用边界

`sure_master` 会根据评测反馈反复生成、执行和选择候选方案，因此数据集必须按用途分层。能被 agent 反复查看或用于选择方案的数据，都不应再作为最终测试结论；能被模型参数训练用到的数据，也不应再作为评测集。

推荐把数据分成以下几类：

| 类别 | 定位 | 作用 | 推荐数据集或来源 |
| --- | --- | --- | --- |
| Smoke set | 链路检查集 | 验证 Docker、VC、基础模型路径、候选脚本、SURE metric 和 artifact contract 能跑通 | 当前最小 demo；少量 LJSpeech、LibriTTS-R、LibriSpeech 或任务自带样例 |
| Fine-tune train set | 训练集 | 仅当候选方案会更新模型参数时使用，用于训练或微调模型 | 优先使用自有授权数据；TTS 可用 LibriTTS-R train、VCTK train speakers、Emilia 子集或全量；ASR 可用对应 recipe 的 train split |
| Early search set | 早期搜索集 | 快速排错和筛方向，判断方案能否运行、是否明显有希望 | TTS 可用 LibriTTS-R dev_clean/dev_other 100-300 条；ASR 可用 dev-clean/dev-other 的小采样 |
| Regular search set | 正式搜索集 | agent 反复比较候选方案，寻找稳定有效的改法 | TTS 可用 LibriTTS-R dev_clean/dev_other 500-1000 条；ASR 可用更大的 dev split 或业务 dev set |
| Selection set | 候选确认集 | 对 regular search 选出的少数方案做二次验证，防止只适配 search set | TTS 可用 VCTK held-out speakers + 另一批 LibriTTS-R；ASR 可用不同 speaker/domain 的 held-out dev |
| Holdout/test set | 最终隐藏测试集 | 不给 agent 看，不参与训练或方案选择，只用于最终效果结论 | LibriTTS-R test、VCTK held-out speakers、业务隐藏集；ASR 可用标准 test split 或业务隐藏集 |
| Official benchmark | 官方可比 benchmark | 和论文、官方脚本或社区结果对齐 | F5-TTS 可用 Seed-TTS test-en/zh、LibriSpeech-PC test-clean；icefall/LibriSpeech 可用 test-clean/test-other |

### 阶段含义

`smoke` 只检查链路是否跑通，不用于判断模型效果。当前 F5-TTS demo prompt 或 1-20 条样本都属于这个阶段。

`early search` 是接入初期和方向粗筛阶段，重点是排除会崩溃、OOM、产物格式错误、空音频、明显退化的方案。它的结果可信度有限。

`regular search` 是正式搜索阶段，链路已经稳定后，用更大的数据规模比较候选方案。这里的数据会被 agent 反复使用，因此不是最终测试集。

`selection` 是候选确认阶段，只对 regular search 中少数较好的方案做复测，用来判断改动是否泛化到另一批 speaker、文本长度或领域。selection set 可以少量重复使用，但不应每一步都拿来调参。

`holdout/test` 是最终评测阶段。它不进入 prompt、不参与 debug、不参与 improve 方案选择。最终报告中的核心效果应来自这部分数据。

`official benchmark` 也属于最终评测，但它回答的是“结果是否能和官方或社区设置对齐”。如果 official benchmark 被用于日常搜索，它就失去 benchmark 意义。

### F5-TTS 推荐划分

对于 `tts_en_wer` / F5-TTS，自进化初期可以按以下方式落地：

```text
smoke:
  10-20 条
  当前 F5 demo + LibriTTS-R dev_clean 少量样本

fine-tune train:
  前期推理级优化可不使用
  需要微调时先用 LibriTTS-R train_clean_100 或自有授权数据
  最终训练可扩到 LibriTTS-R train 全量、VCTK train speakers 或 Emilia 子集/全量

early search:
  100-300 条
  LibriTTS-R dev_clean/dev_other

regular search:
  500-1000 条，或至少几万词
  LibriTTS-R dev_clean/dev_other，和 early search 不完全重叠

selection:
  500-1000 条
  VCTK held-out speakers + 另一批 LibriTTS-R

holdout/test:
  1000+ 条
  LibriTTS-R test_clean/test_other + 自有业务隐藏集

official benchmark:
  Seed-TTS test-en/zh
  LibriSpeech-PC test-clean
```

如果 F5-TTS 任务只调推理参数、文本规范化、长文本切分或后处理，`fine-tune train set` 可以为空。如果候选方案开始训练 checkpoint，训练数据必须只来自 train split，不能混入 selection、holdout 或 official benchmark。

当前仓库已为 F5-TTS 的现有 SURE Master 流程落地了分层 eval prompt：

```bash
python playground/sure_master/tools/build_f5tts_eval_data.py
python playground/sure_master/tools/build_f5tts_train_manifests.py
```

生成结果位于 `playground/sure_master/data/f5tts_staged/`：

```text
smoke/          20 条，LibriTTS dev-clean
early_search/   200 条，LibriTTS dev-clean/dev-other
regular_search/ 800 条，LibriTTS dev-clean/dev-other
selection/      500 条，VCTK held-out speakers + LibriTTS dev
holdout/        1000 条，LibriTTS test-clean/test-other
```

日常自进化用 `configs/sure_master/gpt-5-f5tts-smoke.yaml`、
`configs/sure_master/gpt-5-f5tts-early-search.yaml`、
`configs/sure_master/gpt-5-f5tts-regular-search.yaml` 或
`configs/sure_master/gpt-5-f5tts-selection.yaml`。`holdout/` 不提供常规运行
config，避免被 agent 在搜索或调参时使用。

训练级动作已按白名单微调方式开放，训练数据位于
`playground/sure_master/data/f5tts_train_manifests/`：

```text
libritts_train_clean_100_1h/   LibriTTS train-clean-100，约 1 小时
libritts_train_clean_100_5h/   LibriTTS train-clean-100，约 5 小时
libritts_train_clean_100_10h/  LibriTTS train-clean-100，约 10 小时
```

候选脚本不能直接调用 F5-TTS 的训练入口，只能调用受控 wrapper：

```bash
$SURE_TTS_PYTHON $SURE_TTS_FINETUNE_WRAPPER \
  --action finetune_short \
  --train-manifest libritts_train_clean_100_1h \
  --max-steps 1000 \
  --learning-rate 3e-6 \
  --effective-batch-size 8 \
  --freeze-policy none
```

当前白名单动作空间固定模型结构和 base checkpoint，只允许 `no_train` 或
`finetune_short`；`max_steps` 限定为 `500/1000/2000/5000`，learning rate 限定为
`1e-6/3e-6/5e-6/1e-5`，effective batch size 限定为 `8/16/32`，freeze policy
当前只允许 `none`。wrapper 只写当前 experiment 的 `working/` 和 `models/`，
输出 checkpoint 为 `models/f5tts_finetune/final_checkpoint.pt`。

### Zipformer ASR 推荐划分

对于 `asr_en_wer` / icefall LibriSpeech Zipformer，自进化默认使用 `dev-clean/dev-other` 分层子集，保留 `test-clean/test-other` 只做最终 benchmark。

已提供的 ref 文件：

```text
smoke:
  playground/sure_master/data/asr_librispeech_smoke_ref.txt
  20 条，dev-clean/dev-other 各 10 条

early search:
  playground/sure_master/data/asr_librispeech_early_ref.txt
  300 条，dev-clean/dev-other 各 150 条

regular search:
  playground/sure_master/data/asr_librispeech_regular_ref.txt
  1000 条，dev-clean/dev-other 各 500 条

selection:
  playground/sure_master/data/asr_librispeech_selection_ref.txt
  4077 条，和前三层 speaker-disjoint

holdout / official benchmark:
  playground/sure_master/data/asr_en_wer_ref.txt
  5559 条，test-clean/test-other 全量
```

生成或重建这些 dev ref：

```bash
python playground/sure_master/tools/build_asr_librispeech_refs.py
```

自优化配置应同时切换 ref 和 decode split：

```yaml
sure:
  inputs:
    ref: "/hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster/playground/sure_master/data/asr_librispeech_regular_ref.txt"
  execution_env:
    SURE_ASR_EVAL_SPLITS: "dev-clean,dev-other"
    SURE_MAX_DURATION: "auto"
    SURE_DURATION_AUTOTUNE: "1"
    SURE_DURATION_AUTOTUNE_MIN: "100"
    SURE_DURATION_AUTOTUNE_STEP: "100"
    SURE_DURATION_AUTOTUNE_MAX: "1400"
    SURE_DURATION_PROBE_SUCCESS_BATCHES: "2"
    SURE_DURATION_PROBE_FULL_LIBRI: "0"
    SURE_DURATION_PRINT_DIAGNOSTICS: "0"
    SURE_TRAIN_DURATION_RETRY: "1"
    SURE_TRAIN_DURATION_MIN: "100"
    SURE_TRAIN_DURATION_RETRY_STEP: "100"
    SURE_ASR_ZIPFORMER_WRAPPER: "/hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster/playground/sure_master/tools/run_icefall_zipformer_candidate.py"
    SURE_ASR_REQUIRE_ZIPFORMER_WRAPPER: "1"
```

`SURE_MAX_DURATION=auto` 由 Zipformer wrapper/baseline 在构造好最终
`train.py` 结构/loss 参数后解析。生成的 Zipformer 候选不要直接调用
`SURE_RUNTIME_ENV_HELPER resolve-max-duration`，而是把候选控制的 train/decode
参数传给 `SURE_ASR_ZIPFORMER_WRAPPER`；wrapper 会通过
`SURE_DURATION_TRAIN_ARGS_JSON` 把最终 extra args 交给 helper。这样 compact
和 large Zipformer 会分别探测自己的真实显存需求，且 icefall `PYTHONPATH`、
checkpoint、decode split、hyp 格式都由统一协议管理。

探测目录按候选/cache key 隔离，默认看到
`SURE_DURATION_PROBE_SUCCESS_BATCHES=2` 个正常 training batch log 即认为该
duration 可训练，不要求跑完整 epoch。探测仍然是 fail-closed：如果降到
`SURE_DURATION_AUTOTUNE_MIN` 仍然没有成功，不会再返回一个失败过的 duration。
非 OOM 的启动/导入/CLI/数据错误不会继续降 `max-duration` 伪装成显存不足，而会
直接作为候选或环境错误失败。正式训练阶段如果仍遇到 CUDA OOM，Zipformer baseline 会按
`SURE_TRAIN_DURATION_RETRY_STEP` 降低 `max-duration` 重试，并写出
`artifacts/resource_profile.json`。

生成的 ASR Zipformer 训练/结构候选应调用 `SURE_ASR_ZIPFORMER_WRAPPER`，不要在候选
脚本里直接 `subprocess` 调 `base_model/recipe/train.py` / `decode.py`。wrapper 只固定
执行边界，不固定搜索空间；候选仍可通过 JSON 参数改变结构、loss、训练和 decode：

```python
cmd = [
    os.environ["SURE_ASR_ZIPFORMER_WRAPPER"],
    "--candidate-type", "arch",
    "--action", "train_decode",
    "--idea-text", "reduce middle Zipformer widths",
    "--train-args-json", json.dumps(["--encoder-dim", "192,256,448,640,448,256"]),
    "--decode-args-json", json.dumps(["--decoding-method", "modified_beam_search"]),
]
```

如果配置中保留 `SURE_ASR_REQUIRE_ZIPFORMER_WRAPPER: "1"`，非官方 baseline 的候选
直接调用 `train.py`/`decode.py` 会在执行前被拒绝。

最终评测再切回官方 test：

```yaml
sure:
  inputs:
    ref: "/hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster/playground/sure_master/data/asr_en_wer_ref.txt"
  execution_env:
    SURE_ASR_EVAL_SPLITS: "test-clean,test-other"
```

### 切换到 TEDLIUM3 跑 ASR 自进化

ASR 现在通过 dataset profile 和 recipe profile 区分“数据集”和“icefall recipe/model”。LibriSpeech 默认配置仍然可用；如果要在 TEDLIUM_release-3 / TEDLIUM3 上跑 Zipformer 自优化，不要只改 ref 路径，还要同时切换数据集 profile、recipe profile、recipe/data 源路径、decode split、BPE/ref 产物和训练参数。

当前 TEDLIUM3 Zipformer 配置使用：

```yaml
sure:
  task_id: "asr_en_wer"
  initial_solution_path: "playground/sure_master/baselines/zipformer_large_cr_ctc_rnnt_baseline.py"
  inputs:
    ref: "/hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster/playground/sure_master/data/asr_tedlium3_regular_ref.txt"
  execution_env:
    SURE_ASR_DATASET: "tedlium3"
    SURE_ASR_RECIPE_PROFILE: "tedlium3_zipformer"
    SURE_ASR_EVAL_SPLITS: "dev"
    SURE_DURATION_PROBE_FULL_LIBRI: "0"
    SURE_ENABLE_MUSAN: "0"
    SURE_ASR_ZIPFORMER_WRAPPER: "/hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster/playground/sure_master/tools/run_icefall_zipformer_candidate.py"
    SURE_ASR_REQUIRE_ZIPFORMER_WRAPPER: "1"
    SURE_ICEFALL_PYTHON: "/hpc_stor03/sjtu_home/chaolei.liu/anaconda3/envs/icefall/bin/python"
    PYTHONPATH: "/hpc_stor03/sjtu_home/chaolei.liu/ASR/icefall"
  base_models:
    asr_en_wer:
      source_paths:
        recipe: "/hpc_stor03/sjtu_home/chaolei.liu/ASR/icefall/egs/tedlium3/ASR/zipformer"
        data: "/hpc_stor03/sjtu_home/chaolei.liu/ASR/icefall/egs/tedlium3/ASR/data"
        root: "/hpc_stor03/sjtu_home/chaolei.liu/ASR/icefall"
```

TEDLIUM3 的 `SURE_DURATION_TRAIN_ARGS` 只能包含 TEDLIUM3 recipe 支持的结构/训练参数，例如：

```yaml
SURE_DURATION_TRAIN_ARGS: >-
  --num-encoder-layers 2,2,4,5,4,2
  --feedforward-dim 512,768,1536,2048,1536,768
  --encoder-dim 192,256,512,768,512,256
  --encoder-unmasked-dim 192,192,256,320,256,192
  --enable-spec-aug 0
```

不要把 LibriSpeech large CR-CTC/RNNT 专用参数带到 TEDLIUM3，例如 `--full-libri`、`--use-cr-ctc`、`--ctc-loss-scale`、`--cr-loss-scale`、`--time-mask-ratio` 等。

#### TEDLIUM3 预处理

从 icefall TEDLIUM3 ASR recipe 根目录运行预处理：

```bash
cd /hpc_stor03/sjtu_home/chaolei.liu/ASR/icefall/egs/tedlium3/ASR
```

当前 SURE 配置不启用 MUSAN，也不使用 HLG/LM rescoring，因此最小必需阶段是 TEDLIUM3 manifests、fbank cuts 和 BPE：

```bash
bash prepare.sh --stage 0 --stop-stage 1
bash prepare.sh --stage 3 --stop-stage 3
bash prepare.sh --stage 5 --stop-stage 6
```

如果 TEDLIUM3 已经下载在其它目录，则用已有下载根目录替换 stage 0：

```bash
bash prepare.sh --dl-dir /path/to/download_root --stage 1 --stop-stage 1
bash prepare.sh --dl-dir /path/to/download_root --stage 3 --stop-stage 3
bash prepare.sh --dl-dir /path/to/download_root --stage 5 --stop-stage 6
```

其中 `/path/to/download_root/tedlium3` 必须存在。预处理完成后至少确认：

```bash
test -f /hpc_stor03/sjtu_home/chaolei.liu/ASR/icefall/egs/tedlium3/ASR/data/fbank/tedlium_cuts_train.jsonl.gz
test -f /hpc_stor03/sjtu_home/chaolei.liu/ASR/icefall/egs/tedlium3/ASR/data/fbank/tedlium_cuts_dev.jsonl.gz
test -f /hpc_stor03/sjtu_home/chaolei.liu/ASR/icefall/egs/tedlium3/ASR/data/fbank/tedlium_cuts_test.jsonl.gz
test -f /hpc_stor03/sjtu_home/chaolei.liu/ASR/icefall/egs/tedlium3/ASR/data/lang_bpe_500/bpe.model
```

然后在 EvoMaster 根目录生成 SURE ref：

```bash
cd /hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster

/opt/conda/envs/evomaster/bin/python -m playground.sure_master.tools.build_asr_refs \
  --dataset tedlium3 \
  --manifest-dir /hpc_stor03/sjtu_home/chaolei.liu/ASR/icefall/egs/tedlium3/ASR/data/fbank \
  --output-dir /hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster/playground/sure_master/data
```

预期生成：

```text
playground/sure_master/data/asr_tedlium3_smoke_ref.txt
playground/sure_master/data/asr_tedlium3_early_ref.txt
playground/sure_master/data/asr_tedlium3_regular_ref.txt
playground/sure_master/data/asr_tedlium3_selection_ref.txt
```

日常 staged search 使用 `asr_tedlium3_regular_ref.txt`，默认应为 500 行：

```bash
wc -l /hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster/playground/sure_master/data/asr_tedlium3_regular_ref.txt
```

#### 先 smoke，再 full staged search

不要在预处理后直接启动完整 TEDLIUM3 staged search。先跑小规模 smoke，验证训练、解码、hyp/ref key 匹配和 SURE WER scoring：

```bash
CONFIG=configs/sure_master/gpt-5-icefall-tedlium3-smoke-staged-axes-mixed.yaml
RUN_NAME=zipformer_tedlium3_smoke
LOG_FILE="/hpc_stor03/sjtu_home/chaolei.liu/log/${RUN_NAME}.log"

nohup python run.py \
  --agent sure_master \
  --config "$CONFIG" \
  --run-dir "$RUN_NAME" \
  --task playground/sure_master/data/asr_en_wer_zipformer_description.md \
  > "$LOG_FILE" 2>&1 &
```

裸 `--run-dir` 名称会解析为
`/hpc_stor03/sjtu_home/chaolei.liu/data/experiments/SURE-Evolve/runs/<name>_<timestamp>/`。
如果传入 `./name`、`runs/name` 或绝对路径，则该路径会被原样使用。

smoke 通过的标准是当前实验 workspace 中：

```text
artifacts/hyp.txt
metric/report.json
metric/score_summary.json
candidate_status.json
```

并且 `candidate_status.json` 中 `success=true`、`metric_accepted=true`、`score_valid=true`。ASR 真实解码可能出现空 hypothesis；这代表 WER 的全删除预测，可以进入 SURE WER 计算，不应手工替换成 placeholder。

smoke 通过后，再启动完整 TEDLIUM3 staged self-evolution：

```bash
CONFIG=configs/sure_master/gpt-5-icefall-tedlium3-staged-axes-mixed.yaml
RUN_NAME=zipformer_tedlium3_staged
LOG_FILE="/hpc_stor03/sjtu_home/chaolei.liu/log/${RUN_NAME}.log"

nohup python run.py \
  --agent sure_master \
  --config "$CONFIG" \
  --run-dir "$RUN_NAME" \
  --task playground/sure_master/data/asr_en_wer_zipformer_description.md \
  > "$LOG_FILE" 2>&1 &
```

之前停止的旧任务仍保留在 `runs/zipformer_tedlium3_staged/`，不会自动迁移。需要明确复用旧路径时，应传入精确路径 `--run-dir runs/zipformer_tedlium3_staged`，而不是裸运行名。

如果使用 VC 容器启动，命令形式和 LibriSpeech staged run 相同，只需要把 `SURE_MASTER_CONFIG` 和 `SURE_MASTER_RUN_NAME` 改成 TEDLIUM3 对应值：

```bash
SURE_MASTER_RUN_NAME=zipformer_tedlium3_staged
SURE_MASTER_CONFIG=configs/sure_master/gpt-5-icefall-tedlium3-staged-axes-mixed.yaml
```

#### TEDLIUM3 checkpoint 策略

当前 TEDLIUM3 配置默认按从头训练 smoke / search：

```yaml
SURE_BASELINE_USE_PRETRAINED: "0"
SURE_BASELINE_CHECKPOINT_DIR: ""
SURE_ASR_REQUIRE_LOCAL_CHECKPOINT: "1"
```

如果后续要做 decode-only baseline 或 warm start，必须使用 TEDLIUM3 recipe 兼容的 checkpoint 和 TEDLIUM3 BPE，例如 TEDLIUM3 Zipformer pretrained package；不要沿用 LibriSpeech checkpoint 或 LibriSpeech BPE。

### 用 VC 直接启动 ASR 自进化

推荐显式指定 `SURE_MASTER_RUN_NAME`。启动脚本会把它作为裸运行名交给 EvoMaster，输出到
`/hpc_stor03/sjtu_home/chaolei.liu/data/experiments/SURE-Evolve/runs/<name>_<timestamp>/`。
如需不加时间戳的固定路径或兼容旧任务，请改用 `SURE_MASTER_RUN_DIR`；该值会作为精确路径使用。二者不能同时设置：

```bash
vc submit \
  --image docker.v2.aispeech.com/sjtu/sjtu_yukai-chaolei-suremaster_icefall:v1.0 \
  --partition pdgpu-a10 \
  --nopassenv \
  --num-task 1 \
  --gpu-per-task 8 \
  --cpu-per-task 64 \
  --mem-per-task 256G \
  --volume /hpc_stor03/sjtu_home/chaolei.liu:/hpc_stor03/sjtu_home/chaolei.liu \
  --dir /hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster \
  --job sure-master-icefall-zipformer-staged \
  --cmd "SURE_MASTER_RUN_NAME=zipformer_staged SURE_MASTER_CONFIG=configs/sure_master/gpt-5-docker.yaml bash docker/sure-master-icefall/run_task.sh" \
  --sync \
  --debug
```

如果要用 staged axes 配置，把 `SURE_MASTER_CONFIG` 改成：

```bash
configs/sure_master/gpt-5-icefall-staged-axes-mixed.yaml
```

mixed-local 启动时不要用本地 shell 的 `SURE_ICEFALL_PYTHON` 覆盖远端容器。约定是：

```bash
SURE_LOCAL_ICEFALL_PYTHON=/hpc_stor03/sjtu_home/chaolei.liu/anaconda3/envs/icefall/bin/python
```

远端 VC child 默认使用配置中的容器路径：

```text
/opt/conda/envs/icefall/bin/python
```

需要改远端路径时使用 `SURE_REMOTE_ICEFALL_PYTHON`，不要复用本地路径。

输出目录：

```text
runs/zipformer_staged/
```

候选脚本的 decode 输出 key 可能来自 Lhotse cut id，例如 `1089-134686-0000-0` 或 `2086-149214-0000-1708`；SURE ref 使用 supervision id，例如 `1089-134686-0000`。候选脚本写 `artifacts/hyp.txt` 前必须保持和 ref 一致，像 baseline 一样去掉最后的数字 segment 后缀，不要用 `len(parts[-1]) <= 3` 这类长度限制。

### 全量训练和短训练

稳定验证时可以使用全量 train split 只跑 1 个 epoch。它适合验证训练脚本、数据加载、显存、IO、checkpoint 和 loss 是否正常，但不适合作为每个候选方案的内循环筛选方式。

推荐顺序：

```text
候选搜索:
  小/中训练子集，短训练

稳定验证:
  全量 train split，1 epoch

最终训练:
  全量 train split，按固定 epoch 或固定 steps 训练到预算
```

全量训练只能使用 train split 的全量；不要把 dev、selection、holdout 或 official benchmark 混入训练。

### 基本规则

- 被 agent 反复用来改方案的数据，不是最终测试集。
- 被模型参数训练用到的数据，不是评测集。
- Search/dev 分数只能用于选择方向，不能作为最终结论。
- Selection 分数用于确认候选稳定性，但仍不是最终结论。
- Holdout/test 和 official benchmark 不参与日常自进化搜索。
- 数据划分应尽量按 speaker、文本和 domain 隔离，避免同一 speaker-target pair 同时出现在 train/search/selection/test 中。

## 9. 基础模型自优化

`sure_master` 支持为每个任务绑定一个基础模型 profile。这样 agent 不是自由选择模型，而是必须从任务指定的模型或 recipe 开始自优化。

任务卡中的 `base_model` 描述工作区内的相对契约：

```yaml
base_model:
  model_id: icefall_librispeech_zipformer
  model_type: asr_recipe
  framework: icefall
  required_paths:
    recipe: base_model/recipe
    data: base_model/data
    root: base_model/root
  entrypoints:
    train: base_model/recipe/train.py
    infer: base_model/recipe/decode.py
  usage_policy: required
```

配置文件中的 `sure.base_models.<task_id>.source_paths` 提供真实外部路径：

```yaml
sure:
  base_models:
    asr_en_wer:
      python: "/path/to/icefall/bin/python"
      source_paths:
        recipe: "/hpc_stor03/sjtu_home/chaolei.liu/ASR/icefall/egs/librispeech/ASR/zipformer"
        data: "/path/to/librispeech/ASR/data"
        root: "/hpc_stor03/sjtu_home/chaolei.liu/ASR/icefall"
```

启动后，每个实验工作区都会得到：

```text
base_model/recipe
base_model/data
base_model/root
```

候选 `run_sure.py` 必须使用这些工作区相对路径。它可以在当前实验工作区里写 wrapper、monkey patch、subclass、复制配置、训练/解码参数、checkpoint、`models/` 和 `artifacts/`，但不能写 SURE 源码目录，也不能直接写基础模型外部源目录。

`asr_en_wer` 默认从 icefall Zipformer recipe 开始自优化。具体使用 LibriSpeech 还是 TEDLIUM3 等数据集，由 ASR dataset profile、recipe profile 和 `sure.base_models.asr_en_wer.source_paths` 共同决定；它不是自由切换到 Whisper、Wav2Vec2 或其它无关模型。

默认 `sure.require_base_model: true`，所以其它任务如果还没有在任务卡中内置 profile，就必须在配置文件里补 `sure.base_models.<task_id>`；缺少 profile 或 required profile 缺少 `source_paths` 都会 fail fast。

## 10. 官方 draft/baseline

为了证明自进化是从强基线出发，ASR 和 F5-TTS 配置都应通过
`sure.initial_solution_path` 使用官方 baseline 脚本，而不是让 LLM 生成弱
draft。Stage0 会先执行该脚本并用 SURE 打分，结果写入
`<workspace>/staged_axes/baseline_draft.json`；baseline 脚本还会写
`artifacts/official_baseline.json` 记录模型来源、checkpoint 和推理参数。

ASR 使用 icefall 官方 Zipformer large CR-CTC-RNNT：

```bash
huggingface-cli download \
  Zengwei/icefall-asr-librispeech-zipformer-large-transducer-with-CR-CTC-20241019 \
  --local-dir /hpc_stor03/sjtu_home/chaolei.liu/models/official_drafts/icefall-asr-librispeech-zipformer-large-transducer-with-CR-CTC-20241019
```

配置中的 `SURE_BASELINE_CHECKPOINT_DIR` 指向下载目录下的 `exp/`。官方 HF
包提供 `pretrained.pt` 和 `epoch-50.pt`；配置
`SURE_BASELINE_USE_PRETRAINED=1` 时，Zipformer baseline 会用
`pretrained.pt` 作为官方已导出的 averaged checkpoint 进行 decode。若关闭
`SURE_BASELINE_USE_PRETRAINED`，则必须提供 icefall averaging 所需的完整 epoch
checkpoint 序列，否则直接失败，不会退回短训。

F5-TTS 使用 `SWivid/F5-TTS` 的 `F5TTS_v1_Base`：

```bash
huggingface-cli download \
  SWivid/F5-TTS \
  --include "F5TTS_v1_Base/model_1250000.safetensors" \
  --include "F5TTS_v1_Base/vocab.txt" \
  --local-dir /hpc_stor03/sjtu_home/chaolei.liu/models/official_drafts/SWivid-F5-TTS
```

F5 official baseline 只调用 `SURE_TTS_BATCH_INFER_WRAPPER`，不训练、不改结构，
并生成合法的 `artifacts/samples.jsonl`、wav 和 `candidate_changes.json`。

## 11. 分阶段三轴自进化

默认 `sure_master` 仍使用原来的 research/improve loop。如果要启用三轴分阶段
搜索，配置：

```yaml
sure:
  search_strategy: "staged_axes"
  staged_axes:
    enabled: true
    rounds_per_axis: 4
    ideas_per_round: 4
    top_arch: 2
    top_train: 2
    top_inference: 3
    runner_up_count: 5
```

启用后流程变为：

```text
Stage0 draft/baseline
Stage1 arch: 16 -> 8 -> 2
Stage2 train strategy: 16 -> 8 -> 2
Stage3 inference: 16 -> 8 -> 3
Stage4 top-2 arch × top-2 train × top-3 inference = 12 个组合
Stage5 selection: search best + top-5 runner-up + baseline/draft 重排
Stage6 holdout/test: 可选，只评最终候选和 baseline/draft
```

Stage2/Stage3 不做额外复验；结构、训练策略和推理方式的交互统一在 Stage4
有限组合中验证。Stage4 的 `arch × train` 组合必须重新训练 checkpoint，不能把
arch-only checkpoint 和 train-only checkpoint 直接拼接。

successive halving 的预算通过每个 axis 的 rung 配置注入候选运行环境：

```yaml
sure:
  staged_axes:
    axes:
      arch:
        rungs:
          - name: short
            keep: 8
            execution_env:
              SURE_TTS_TRAIN_MAX_STEPS: "500"
          - name: medium
            keep: 2
            execution_env:
              SURE_TTS_TRAIN_MAX_STEPS: "1000"
          - name: final
            keep: 2
            execution_env:
              SURE_TTS_TRAIN_MAX_STEPS: "2000"
```

`train` 和 `inference` 也使用同样的 `rungs` 格式。ASR 可用 epoch、steps、
duration 或数据子集相关环境变量控制预算；F5-TTS 可用 wrapper 白名单中的
`SURE_TTS_TRAIN_MAX_STEPS`、`SURE_TTS_MAX_SAMPLES` 等控制预算。

selection 和 holdout 可以覆盖输入 role、运行环境和 base model symlink。F5-TTS
常用 `base_model/eval_data` 切换 prompt set：

```yaml
sure:
  staged_axes:
    selection:
      execution_env:
        SURE_TTS_MAX_SAMPLES: "500"
      base_model_source_paths:
        eval_data: "/abs/path/to/f5tts_staged/selection"
    holdout:
      enabled: true
      execution_env:
        SURE_TTS_MAX_SAMPLES: "1000"
      base_model_source_paths:
        eval_data: "/abs/path/to/f5tts_staged/holdout"
```

ASR selection/holdout 通常覆盖 `inputs.ref` 和 `SURE_ASR_EVAL_SPLITS`：

```yaml
sure:
  staged_axes:
    selection:
      inputs:
        ref: "/abs/path/to/asr_librispeech_selection_ref.txt"
      execution_env:
        SURE_ASR_EVAL_SPLITS: "dev-clean,dev-other"
```

分阶段运行的结果写入：

```text
<workspace>/staged_axes/
  baseline_draft.json
  ideas_<axis>.json
  leaderboard_<axis>_<rung>.json
  top_<axis>.json
  leaderboard_combination_search.json
  leaderboard_selection.json
  leaderboard_holdout.json
  summary.json
```

最终候选由 selection set 重排决定；如果配置了 holdout，holdout 只用于最终
报告，不参与搜索和候选选择。

## 12. 任务配置示例

### 英文 ASR WER

```yaml
sure:
  task_id: "asr_en_wer"
  inputs:
    ref: "/hpc_stor03/sjtu_home/chaolei.liu/Agent/EvoMaster/playground/sure_master/data/asr_librispeech_regular_ref.txt"
  execution_env:
    SURE_ASR_EVAL_SPLITS: "dev-clean,dev-other"
  base_models:
    asr_en_wer:
      python: "/path/to/icefall/bin/python"
      source_paths:
        recipe: "/hpc_stor03/sjtu_home/chaolei.liu/ASR/icefall/egs/librispeech/ASR/zipformer"
        data: "/path/to/librispeech/ASR/data"
        root: "/hpc_stor03/sjtu_home/chaolei.liu/ASR/icefall"
```

候选脚本需要生成：

```text
artifacts/hyp.txt
```

### SLU Accuracy

`sure.require_base_model` 默认是 `true`，因此切到 `slu_accuracy` 时也必须指定一个 SLU 基础模型。下面是结构示例，路径需要按实际模型替换：

```yaml
sure:
  task_id: "slu_accuracy"
  inputs:
    ref: "/data/slu/ref.txt"
    prompt_jsonl: "/data/slu/prompt.jsonl"
  base_models:
    slu_accuracy:
      model_id: "your_slu_model"
      model_type: "slu_model"
      framework: "custom"
      usage_policy: "required"
      required_paths:
        model: "base_model/model"
      source_paths:
        model: "/path/to/your_slu_model"
      entrypoints:
        infer: "base_model/model/infer.py"
```

候选脚本需要生成：

```text
artifacts/hyp.txt
```

该任务的 SURE 必需角色是：

```text
ref, hyp, prompt_jsonl
```

### KWS Accuracy

KWS 任务同样需要绑定一个 KWS 基础模型：

```yaml
sure:
  task_id: "kws_accuracy"
  inputs:
    reference_jsonl: "/data/kws/reference.jsonl"
    keyword: "literal:hey_sure"
  base_models:
    kws_accuracy:
      model_id: "your_kws_model"
      model_type: "kws_model"
      framework: "custom"
      usage_policy: "required"
      required_paths:
        model: "base_model/model"
      source_paths:
        model: "/path/to/your_kws_model"
      entrypoints:
        infer: "base_model/model/infer.py"
```

候选脚本需要生成：

```text
artifacts/sample_output.jsonl
```

### 英文 TTS WER

TTS 任务需要绑定一个 TTS 基础模型，候选代码只能基于它生成或改进音频：

```yaml
sure:
  task_id: "tts_en_wer"
  device: "cuda"
  cache_dir: "/path/to/sure_cache"
  base_models:
    tts_en_wer:
      model_id: "your_tts_model"
      model_type: "tts_model"
      framework: "custom"
      usage_policy: "required"
      required_paths:
        model: "base_model/model"
      source_paths:
        model: "/path/to/your_tts_model"
      entrypoints:
        infer: "base_model/model/infer.py"
```

候选脚本需要生成：

```text
artifacts/samples.jsonl
```

`samples.jsonl` 中应引用当前实验工作区内生成的音频文件，通常放在 `artifacts/` 或 `working/` 下。SURE 会读取这些样本并执行对应的转写和评分 pipeline。

### 英文 VC WER

VC 任务需要绑定一个 VC 基础模型：

```yaml
sure:
  task_id: "vc_en_wer"
  device: "cuda"
  cache_dir: "/path/to/sure_cache"
  base_models:
    vc_en_wer:
      model_id: "your_vc_model"
      model_type: "vc_model"
      framework: "custom"
      usage_policy: "required"
      required_paths:
        model: "base_model/model"
      source_paths:
        model: "/path/to/your_vc_model"
      entrypoints:
        infer: "base_model/model/infer.py"
```

候选脚本需要生成：

```text
artifacts/samples.jsonl
```

## 13. 工作区结构

每个实验会创建：

```text
run_sure.py
base_model/
models/
artifacts/
metric/
working/
```

推荐用途：

| 目录 | 用途 |
| --- | --- |
| `base_model/` | 当前任务的基础模型或 recipe symlink 入口 |
| `models/` | 模型代码、模型结构、配置、checkpoint、tokenizer 等 |
| `artifacts/` | SURE 必需产物，例如 `hyp.txt`、`sample_output.jsonl`、`samples.jsonl` |
| `metric/` | EvoMaster 调用 SURE 后写入的评测结果 |
| `working/` | 临时文件、预处理结果、日志 |

候选脚本只能写当前实验工作区，不应写仓库其它目录、SURE 目录或基础模型外部源目录。

### 工作区自动清理

ASR/Zipformer 这类训练候选会在每个 workspace 下生成 checkpoint、decode 中间文件和长日志。推荐开启候选级清理：

```yaml
sure:
  workspace_cleanup:
    enabled: true
    on_success: true
    on_failure: true
    remove_models: true
    remove_working: true
    remove_runtime_cache: true
    compact_metric_logs: true
    keep_log_tails: true
    log_tail_bytes: 65536
```

清理发生在候选写完 `artifacts/candidate_status.json` 和 metric 结果之后。默认保留：

```text
run_sure.py
artifacts/candidate_status.json
artifacts/candidate_changes.json
artifacts/hyp.txt
metric/score_summary.json
metric/metric_error.json
metric/remote_training_result.json
metric/remote_training_vc_command.json
metric/cleanup_manifest.json
metric/log_tails/*.tail.log
```

会删除当前候选 workspace 内的：

```text
models/
working/
.sure_runtime/
metric/*.log
```

`metric/*.log` 和 `working/**/*.log` 删除前会把尾部保存到 `metric/log_tails/`。清理不会删除外部的官方模型、LibriSpeech 数据、SURE 源码或 `base_model/` 软链接。

调试某个候选时可以临时关闭：

```bash
SURE_WORKSPACE_CLEANUP_ENABLED=0 python run.py --agent sure_master --config configs/sure_master/gpt-5-icefall-staged-axes-mixed.yaml
```

也可以只保留失败候选的完整产物：

```yaml
sure:
  workspace_cleanup:
    enabled: true
    on_success: true
    on_failure: false
```

清理已有 run 目录时先 dry-run：

```bash
python playground/sure_master/tools/cleanup_sure_workspaces.py --root runs/zipformer_staged_axes
```

确认候选 workspace 列表正确后执行：

```bash
python playground/sure_master/tools/cleanup_sure_workspaces.py --root runs/zipformer_staged_axes --apply
```

### 本地 GPU 调度

ASR mixed-local 模式中，coordinator、draft baseline decode 和 inference 类候选会在本地 GPU 上运行；训练类候选再提交到 VC。为避免本地 draft decode 抢到忙 GPU，ASR 配置推荐使用：

```yaml
session:
  local:
    gpu_devices: idle
    idle_gpu_min_free_mib: 10000
    idle_gpu_max_utilization: 20
    idle_gpu_min_count: 1
    idle_gpu_allow_busy_fallback: false
    gpu_lock_enabled: true
    gpu_lock_dir: "/tmp/sure_master_gpu_locks"
    gpu_lock_wait_seconds: 30
    gpu_lock_poll_seconds: 2
```

`idle` 会在 session 初始化时通过 `nvidia-smi` 选择空闲 GPU。执行每个本地候选前，ResourceAllocator 还会再次刷新 GPU 状态；如果原分配 GPU 变忙，会重新选择仍满足阈值的 GPU。`gpu_lock_enabled` 会为每张本地 GPU 创建文件锁，避免多个 sure_master 进程同时抢同一张 GPU。

运行时可用环境变量覆盖阈值：

```bash
SURE_IDLE_GPU_MIN_FREE_MIB=10000
SURE_IDLE_GPU_MAX_UTILIZATION=20
SURE_IDLE_GPU_ALLOW_BUSY_FALLBACK=0
SURE_GPU_LOCK_ENABLED=1
SURE_GPU_LOCK_WAIT_SECONDS=30
```

当基础模型包含 `base_model/recipe` 和 `base_model/data` 时，环境会额外尝试创建：

```text
base_model/recipe/data -> ../data
```

这是为了兼容 icefall/Zipformer 这类 recipe 在脚本内部按 `data/...` 访问 manifests 或 fbank 的习惯。

## 14. 输出文件

候选脚本运行成功并通过 SURE 评分后，实验目录中会出现：

```text
metric/pipeline_spec.json
metric/score_summary.json
```

如果 SURE 评分失败：

```text
metric/metric_error.json
```

当前最佳代码会保存到主工作区：

```text
best_solution/best_solution.py
```

`sure_master` 的返回结果包含：

```text
status
task_id
metric
best_score
is_lower_better
```

## 15. 自进化流程

整体流程：

```text
prefetch
draft -> execute run_sure.py -> SURE metric -> debug if needed
research -> improve idea(s) -> execute + SURE metric -> select best
knowledge_promotion
repeat until max_research_rounds or timeout
```

阶段说明：

| 阶段 | 作用 |
| --- | --- |
| `prefetch` | 生成任务和数据理解摘要 |
| `draft` | 生成初始 `run_sure.py` |
| `debug` | 当代码运行失败或产物缺失时修复候选代码 |
| `research` | 根据当前最佳结果提出改进方向 |
| `improve` | 针对某个 idea 生成新候选方案 |
| `knowledge_promotion` | 总结一轮实验经验 |
| `wisdom_promotion` | 超时或结束时沉淀可复用知识 |

research 轮数：

```yaml
max_research_rounds: 3
```

并行实验配置：

```yaml
session:
  local:
    parallel:
      enabled: true
      max_parallel: 1
      split_workspace_for_exp: true
```

如果把 `max_parallel` 调大，需要确保：

- GPU/CPU 资源足够。
- 输入数据使用绝对路径，或 symlink 能正确创建到每个实验工作区。
- 各实验不会写同一个模型输出文件。

## 16. SURE 只读边界

生成的 `run_sure.py` 不允许：

- `import sure_eval`
- 调用 SURE 评分
- 引用 `/hpc_stor03/sjtu_home/chaolei.liu/sure`
- 写 SURE 源码目录

边界由两层保证：

1. prompt 明确要求候选代码不要修改或调用 SURE。
2. `validate_sure_candidate_boundary()` 会在执行候选代码前做静态检查；如果发现直接 import SURE、引用 SURE 根目录、绕过 required 基础模型，或直接引用基础模型外部源目录，会拒绝执行。

框架侧调用 SURE 时，`SureMetricRunner` 会临时设置：

```python
sys.dont_write_bytecode = True
```

这样可以避免 Python import SURE 时向 SURE 源码树写入新的 `.pyc` 文件。

## 17. 验证命令

语法检查：

```bash
python -m py_compile \
  playground/sure_master/core/playground.py \
  playground/sure_master/core/exp/*.py \
  playground/sure_master/core/utils/*.py \
  playground/sure_master/env/local.py \
  playground/sure_master/agent/session/local.py
```

运行单测：

```bash
python -m unittest playground.sure_master.core.utils.test_sure_master
```

用真实 SURE adapter 校验所有任务卡：

```bash
python - <<'PY'
from pathlib import Path
from playground.sure_master.core.utils.metric import SureMetricRunner
from playground.sure_master.core.utils.task_cards import load_task_cards

sure_root = Path("/hpc_stor03/sjtu_home/chaolei.liu/sure")
runner = SureMetricRunner(sure_root, pythonpath=sure_root / "src", device="cpu")
runner._prepare_import_path()

from sure_eval.evaluation.cli_adapters import build_pipeline_spec

cards = load_task_cards("playground/sure_master/task_cards/sure_tasks.yaml")
for task_id, card in cards.items():
    spec = runner._build_pipeline_spec(build_pipeline_spec, card)
    assert set(card.required_roles) == set(spec["required_roles"]), task_id
    print(task_id, spec["pipeline_id"], spec["metric"], spec["required_roles"])
print("TOTAL", len(cards))
PY
```

验证 `sure_master` 是否能被入口注册：

```bash
python - <<'PY'
import logging
import run
from evomaster.core import list_registered_playgrounds, get_playground_class

logging.getLogger().setLevel(logging.ERROR)
run.auto_import_playgrounds()
print("sure_master_registered", "sure_master" in list_registered_playgrounds())
pg = get_playground_class(
    "sure_master",
    config_path=run.project_root / "configs" / "sure_master" / "kimi-2.6.yaml",
)
print(type(pg).__name__, pg.task_card.task_id)
PY
```

## 18. 常见问题

### 入口报 `ModuleNotFoundError: No module named 'mcp'`

当前 Python 环境缺少项目入口依赖。执行：

```bash
pip install -r requirements.txt
```

或：

```bash
pip install mcp
```

### 报 required artifact missing

候选脚本没有生成任务卡要求的产物。检查：

```text
playground/sure_master/task_cards/sure_tasks.yaml
```

例如 `asr_en_wer` 必须生成：

```text
artifacts/hyp.txt
```

### 报 required input missing

输入文件找不到。优先用绝对路径配置 `sure.inputs`：

```yaml
sure:
  inputs:
    ref: "/abs/path/to/ref.txt"
```

配置为绝对路径后，sure_master 会自动复制一份到每个实验工作区的任务卡默认输入位置，例如 `input/ref.txt`。如果仍然报 missing，通常说明源文件不存在或路径写错。

如果使用相对路径，并且开启了：

```yaml
split_workspace_for_exp: true
```

需要确认每个实验工作区下都有相应输入文件，或者配置了 `session.local.symlinks`。

### asr_en_wer 没有 checkpoint，decode-only 失败

当前 Zipformer 源目录如果没有 `epoch-*.pt` checkpoint，候选脚本必须先训练出 workspace-local checkpoint，再 decode 生成 `artifacts/hyp.txt`。推荐在配置中保留：

```yaml
sure:
  execution_env:
    SURE_MAX_TRAIN_EPOCHS: "1"
    SURE_MAX_DURATION: "600"
    SURE_USE_FP16: "1"
    SURE_DECODE_ONLY: "0"
    SURE_ICEFALL_PYTHON: "/hpc_stor03/sjtu_home/chaolei.liu/anaconda3/envs/icefall/bin/python"
```

如果你后来补齐了现成 Zipformer checkpoint，可以把 checkpoint 目录作为额外 base model source path 挂进 workspace，并把 `SURE_DECODE_ONLY` 改成 `"1"`。

### 启动时报 `requires a base_model profile`

默认 `sure.require_base_model: true`，当前任务必须有基础模型 profile。处理方式二选一：

1. 在 `playground/sure_master/task_cards/sure_tasks.yaml` 给该任务增加 `base_model`。
2. 在运行配置中增加 `sure.base_models.<task_id>`。

临时调试时可以关闭：

```yaml
sure:
  require_base_model: false
```

但这会允许 agent 自由选择模型，不符合“每个任务都从某个基础模型开始自优化”的目标。

### 启动时报 `requires base_model source_paths`

任务卡或配置里声明了 required 基础模型，但没有给每个 `required_paths` 提供真实外部源目录。

例如任务卡有：

```yaml
required_paths:
  recipe: base_model/recipe
  data: base_model/data
```

配置里就必须补：

```yaml
sure:
  base_models:
    asr_en_wer:
      source_paths:
        recipe: "/abs/path/to/recipe"
        data: "/abs/path/to/data"
```

`source_paths` 必须指向目录。路径不存在时，环境创建 symlink 会 warning 并跳过，对应候选脚本通常会运行失败。

### 候选代码被执行前拒绝

说明候选代码违反了 SURE 只读边界。常见原因：

- 代码里出现 `import sure_eval`
- 代码里硬编码了 `/hpc_stor03/sjtu_home/chaolei.liu/sure`
- 代码尝试自己调用 SURE 评分
- required 基础模型存在，但候选代码没有引用 `base_model/...`
- 候选代码硬编码了基础模型外部源目录，例如 `/abs/path/to/recipe`

候选脚本只应通过工作区相对路径使用基础模型并生成模型和产物，评分由 `SureMetricRunner` 统一完成。

### SURE 环境检查失败

默认配置是：

```yaml
validate_env: false
```

如果你改成 `true`，SURE 会先检查所选 pipeline 的环境。失败通常说明对应 SURE node 的本地环境、依赖或模型缓存尚未准备好。

### TTS/VC 很慢或模型加载失败

TTS/VC 的语义指标会通过 SURE 加载转写模型，例如 Whisper 或 Paraformer。建议配置：

```yaml
sure:
  device: "cuda"
  cache_dir: "/path/to/sure_cache"
```

## 19. 新增任务卡

在下面文件添加新任务：

```text
playground/sure_master/task_cards/sure_tasks.yaml
```

最小示例：

```yaml
new_task_id:
  canonical_task: asr
  task_alias: asr
  language: en
  primary_metric: WER
  metric_direction: lower
  route: asr.en.wer.whisper_norm.wenet_wer
  required_roles: [ref, hyp]
  artifact_contract:
    ref: input/ref.txt
    hyp: artifacts/hyp.txt
  base_model:
    model_id: example_asr_model
    model_type: asr_model
    framework: custom
    usage_policy: required
    required_paths:
      model: base_model/model
    entrypoints:
      infer: base_model/model/infer.py
  prompt_guidance: Generate hypotheses in the expected format.
  description: Short task description.
```

如果不想把真实路径写进任务卡，可以只在任务卡中写 `required_paths` 和 `entrypoints`，在运行配置里补：

```yaml
sure:
  base_models:
    new_task_id:
      source_paths:
        model: "/abs/path/to/example_asr_model"
```

添加后至少运行：

```bash
python -m unittest playground.sure_master.core.utils.test_sure_master
```

如果新任务对应真实 SURE route，还应运行第 14 节的 SURE adapter 校验命令。
