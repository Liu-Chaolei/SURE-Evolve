# ASR Master

**ASR Master** 是基于 EvoMaster 的 `ml_master_2` 改造而来的**自动语音识别（ASR）科研 agent**，用于自动优化 ASR 模型（如 icefall Zipformer）以降低 WER（字错误率）。

它复用了 ml_master_2 的多 agent 编排架构（prefetch → draft → research → improve → knowledge/wisdom promotion），但把所有面向「表格 Kaggle 竞赛」的假设替换成了面向「深度学习 ASR 训练」的逻辑。

本文档说明 **asr_master 与 ml_master_2 的区别**。

---

## 一句话总结

| | ml_master_2 | asr_master |
|---|---|---|
| 面向场景 | MLE-Bench 表格 Kaggle 竞赛 | ASR 模型优化（icefall Zipformer / LibriSpeech） |
| 评测指标 | 竞赛 metric（accuracy / AUC 等，方向可配） | WER（字错误率，**固定越低越好**） |
| agent 产出的代码 | 单个 `run.py`，产出 `submission/submission.csv` | 单个 `run_asr.py`，**wrapper 脚本**调用 train.py + decode.py |
| 评分来源 | grading server 校验提交格式 + metric agent 从终端读分 | **只靠 metric agent 从终端读 `\boxed{WER}`**（无 grading server） |
| 执行方式 | 多 idea 并行（workspace 隔离） | **顺序执行**（GPU 训练不并行） |
| 迭代轮数 | 固定 20 轮 | **可配置**（默认 3 轮，适配长训练） |
| 数据预览 | CSV/JSON 列结构（`data_preview.py`） | lhotse cuts / lang_bpe / RESULTS.md（`asr_data_preview.py`） |

---

## 详细对比

### 1. 注册与类名

| | ml_master_2 | asr_master |
|---|---|---|
| 注册名 | `@register_playground("ml_master_2")` | `@register_playground("asr_master")` |
| 类名 | `MLMaster2Playground` | `ASRMasterPlayground` |

### 2. 评测指标方向

ml_master_2 通过配置 `is_lower_better` 决定优化方向（不同竞赛 metric 方向不同）。

asr_master **硬编码 `is_lower_better = True`**，因为 WER 永远是越低越好。`compare_score()` 简化为：

```python
# asr_master: 两者都有效时，直接比大小
return new_score < old_score
```

### 3. agent 产出的代码形式

**ml_master_2**：agent 写一个自包含的 `run.py`，训练模型并产出 `submission/submission.csv`。成功判据是 `exit_code == 0 且 submission_{uid}.csv 存在`，且经过 grading server 校验格式。

**asr_master**：agent 写一个 **wrapper 脚本 `run_asr.py`**，内部通过 subprocess 调用 icefall recipe 的 `train.py` + `decode.py`，解析 decode 输出得到 WER，最后 `print(f"\boxed{WER}")`。成功判据是 `exit_code == 0`（脚本跑完），再由 metric agent 从终端输出提取 WER。

典型 wrapper 结构（见 `prompts/draft_user_prompt.txt`）：

```python
# train
subprocess.run([sys.executable, "train.py", "--world-size", "1",
                "--num-epochs", "5", "--exp-dir", exp_dir, ...], check=True)
# decode
result = subprocess.run([sys.executable, "decode.py",
                         "--method", "modified_beam_search", "--exp-dir", exp_dir, ...],
                        capture_output=True, text=True)
# parse WER + print boxed
wer = re.findall(r'(?:WER|wer)[\s:=]+(\d+\.?\d*)', result.stdout + result.stderr)[-1]
print(f"\\boxed{{{wer}}}")
```

### 4. 评分机制（关键差异）

**ml_master_2**：`DraftExp` / `ImproveExp` 里有完整的 grading server 调用链：
- `_check_grading_valid()` → `validate_submission()` → 请求 `http://127.0.0.1:6666/validate` 校验提交 CSV 格式
- metric agent 从终端读 `\boxed{score}`

**asr_master**：**完全移除了 grading server**：
- 删掉了 `_check_grading_valid()` 方法
- 不再检查 `submission_{uid}.csv` 是否存在
- 不再 import `grading.py`
- 成功 = `exit_code == 0` + metric agent 成功提取 `\boxed{WER}`
- 新增 `_extract_wer_from_metric()` 方法统一解析 WER

### 5. 执行并行度

**ml_master_2**：research 阶段产出的多个 idea **并行执行**（`max_parallel` 个 worker，每个独立 workspace）。表格任务训练快，并行划算。

**asr_master**：改为**顺序执行**（`max_workers=1`）。ASR 训练是 GPU 密集 + 耗时长，并行会抢显存、不划算。`execute_parallel_tasks()` 默认 `max_workers=1`。

### 6. 迭代轮数

**ml_master_2**：`for reseach_round in range(20)` 固定 20 轮。

**asr_master**：`for reseach_round in range(self.max_research_rounds)`，默认 3 轮，可在配置里改：

```yaml
# configs/asr_master/kimi-2.6.yaml
max_research_rounds: 3   # 每轮 = 一次完整 train + decode，可能数小时
```

### 7. 数据预览

**ml_master_2**：`utils/data_preview.py` 的 `generate()` 扫描 workspace，预览 `.csv` 列结构、`.json` schema、目录树。面向表格数据。

**asr_master**：新增 `utils/asr_data_preview.py` 的 `generate_asr_preview()`，读取 ASR 专属资源：
- **recipe 文件**：train.py / decode.py / model.py / zipformer.py 等大小
- **lhotse cuts 清单**：`fbank/*.cuts.jsonl.gz` 的 cut 数、时长、说话人数
- **语言资源**：`lang_bpe_*` 的 token 数、词表大小
- **基线结果**：`RESULTS.md` 里的 WER

### 8. Prompt 内容

ml_master_2 的 prompt 大量出现 `submission.csv`、`input/description.md`、`test.csv`、Kaggle、A100、12h 等字眼。

asr_master 的 16 个 prompt **全部重写**，0 处 Kaggle 残留，改为：
- `draft_user_prompt.txt`：wrapper 脚本模式 + 示例代码 + `\boxed{WER}` 要求
- `improve_user_prompt.txt`：基于 best_solution 改进 + ASR 改进方向（架构/增强/训练/解码）
- `reseach_user_prompt.txt`：ASR 领域研究方向（编码器维度、SpecAugment、LR schedule、beam width 等）
- `debug_user_prompt.txt`：修 wrapper 脚本 bug，不产出 CSV
- `metric_user_prompt.txt`：从 icefall decode 输出提取 WER（兼容多种 WER 格式）

### 9. 配置差异

| 配置项 | ml_master_2 | asr_master |
|---|---|---|
| 任务 ID | `competition_id` | `asr_task_id` |
| 指标方向 | `is_lower_better`（可配） | 固定 True（移除该字段） |
| 迭代轮数 | 无（硬编码 20） | `max_research_rounds` |
| 框架路径 | 无 | `icefall_root` / `recipe_dir` / `data_dir` |
| grading server | `grading_servers: [...]` | 移除 |
| 默认 LLM | `local_sglang`（DeepSeek） | `kimi`（Kimi-k2.6） |
| symlinks | 指向竞赛数据目录 | 指向 icefall recipe + data + root |
| workspace 子目录 | `best_submission/best_solution/submission/working` | `best_solution/working`（移除 submission/best_submission） |

### 10. 返回结果

**ml_master_2**：`{"status": "completed", "steps": 0}`

**asr_master**：增加 `best_wer` 字段，`{"status": "completed", "steps": 0, "best_wer": 3.52}`

---

## 架构复用 vs 改造对照

| 模块 | 是否复用 | 说明 |
|---|---|---|
| 三层架构（Playground/Exp/Agent） | ✅ 复用 | EvoMaster 框架不变 |
| 多 agent 编排流程 | ✅ 复用 | prefetch→draft→(research→improve)*→promotion |
| `knowledge_promotion` / `wisdom_promotion` | ✅ 复用 | 仅 prompt 文案 ASR 化 |
| `research_exp.py` / `prefetch_exp.py` | ✅ 复用 | 逻辑不变，prompt 驱动 ASR 方向 |
| `watch_dog.py`（超时看门狗） | ✅ 复用 | 不变 |
| `agent/session/local.py`（symlink session） | ✅ 复用 | 不变 |
| `draft_exp.py` / `improve_exp.py` | ⚠️ **改造** | 移除 submission/grading，改成功判据 |
| `playground.py` | ⚠️ **改造** | sequential、可配轮数、移除 grading |
| `data_preview.py` | ❌ **替换** | → `asr_data_preview.py` |
| `grading.py` | ❌ **弃用** | ASR 不需要 |
| 全部 prompts | ❌ **重写** | Kaggle → ASR |

---

## 运行方式

```bash
# 确保 .env 里有 Kimi 变量
# KIMI_API_KEY=...
# KIMI_BASE_URL=https://api.moonshot.cn/v1

# 确保 icefall 数据已 prepare（fbank 目录有 .cuts.jsonl.gz）

python run.py --agent asr_master \
  --config configs/asr_master/gpt-5-example.yaml \
  --task playground/asr_master/data/librispeech_zipformer_description.md
```

---

## 改造时的核心设计决策

1. **为什么用 wrapper 脚本而不是直接改 icefall 源码？**
   icefall 是多文件大代码库（train.py 5 万行、zipformer.py 9 万行），agent 每轮重写整库不现实。wrapper 脚本只覆盖超参/局部组件，便于迭代和回滚。

2. **为什么顺序执行而不是并行？**
   ASR 训练占满 GPU 显存，并行会 OOM 或互相拖慢。表格任务训练秒级，并行划算；ASR 训练小时级，顺序更稳。

3. **为什么固定 WER 越低越好？**
   WER 是 ASR 通用指标，方向唯一，没必要做成可配项，简化了 `compare_score` 逻辑。

4. **为什么去掉 grading server？**
   grading server 是 MLE-Bench 校验提交 CSV 格式的工具，ASR 没有 CSV 提交，WER 直接从 decode 终端输出解析，grading server 无用武之地。
