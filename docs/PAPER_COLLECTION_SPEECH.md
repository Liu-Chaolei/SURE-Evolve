# 语音领域论文收集参考

本文说明如何使用 XLab 的 `paper_collect` 流程收集语音领域论文，包括运行前配置、输入参数、搜索范围和输出位置。

## 1. 流程概览

Paper 收集的目标是建立一个适合后续知识图谱构建、Survey 筛选和新颖性分析的**高召回论文池**，而不是只返回一小份引用列表。

基本流程如下：

```text
主题与子领域配置
  → Tavily 发现搜索种子
  → Semantic Scholar 补全元数据和论文关系
  → 去重与相关性评分
  → 下载可验证的开放获取 PDF
  → 审计并发布论文集合
```

当前实现对应的 XLab 包是 `paper_collect@2.2.0`。

## 2. 运行前配置

建议先检查 XLab 环境：

```text
/xlab check-xlab-setup
```

然后配置学术服务：

```text
/xlab configure-xlab scholarly-services
```

Paper 收集需要以下两个密钥：

| 密钥 | 用途 |
| --- | --- |
| `TAVILY_API_KEY` | 发现 Survey、综述、经典论文和近期论文 |
| `SEMANTIC_SCHOLAR_API_KEY` | 获取规范化元数据、引用关系、参考文献和相关推荐 |

缺失的密钥会在运行开始前通过掩码输入提示配置。不要把密钥写进命令、主题、Markdown 文件或日志中。

## 3. 最小运行示例

先用较小规模验证搜索方向：

```text
/xlab collect-papers "speech processing and spoken language technologies" \
  --target-papers 100 \
  --max-papers 300 \
  --facet "automatic speech recognition" \
  --facet "text-to-speech synthesis" \
  --facet "speech representation learning"
```

其中：

- 主主题必须是第一个参数，并使用引号包裹。
- `--target-papers` 是希望达到的论文数量。
- `--max-papers` 是本次收集的安全上限。
- `--facet` 用于描述具体子领域，可以重复使用。
- `target-papers` 和 `max-papers` 必须满足 `1 <= target-papers <= max-papers`。

目标数量是软停止阈值，不保证最终结果恰好等于该数量。数量不足通常会作为警告记录，是否成功还取决于元数据、关系和下载审计结果。

## 4. 语音领域的推荐配置

### 4.1 覆盖较完整的语音技术领域

```text
/xlab collect-papers "speech processing and spoken language technologies" \
  --target-papers 500 \
  --max-papers 1500 \
  --facet "automatic speech recognition" \
  --facet "text-to-speech synthesis" \
  --facet "speaker recognition and verification" \
  --facet "speech enhancement and separation" \
  --facet "spoken language understanding" \
  --facet "speech representation learning" \
  --facet "audio language models" \
  --facet "speech datasets and evaluation"
```

这适合希望建立语音领域基础语料库，并计划继续构建知识图谱的情况。

### 4.2 语音大模型方向

```text
/xlab collect-papers "large language models for speech understanding and generation" \
  --target-papers 300 \
  --max-papers 900 \
  --facet "speech language models" \
  --facet "audio tokenization and neural codecs" \
  --facet "speech instruction tuning" \
  --facet "spoken dialogue models" \
  --facet "multimodal audio language models" \
  --facet "speech-to-speech generation"
```

### 4.3 自动语音识别方向

```text
/xlab collect-papers "automatic speech recognition" \
  --target-papers 300 \
  --max-papers 900 \
  --facet "end-to-end speech recognition" \
  --facet "self-supervised speech representation learning" \
  --facet "speech recognition for low-resource languages" \
  --facet "streaming and real-time speech recognition" \
  --facet "robust and noisy speech recognition" \
  --facet "ASR datasets and benchmarks"
```

### 4.4 指定工作区

如果已经建立了工作区，可以将 run 关联到该工作区：

```text
/xlab collect-papers "speech foundation models" \
  --workspace speech-research \
  --target-papers 300 \
  --max-papers 900
```

`--workspace` 是 XLab harness 自动提供的工作区参数，不属于 Python 收集脚本自身的内部参数。

## 5. 输入参数

用户通常只需要提供主题、子领域和规模参数。完整参数如下：

| 参数 | 必填 | 说明 | 约束或默认值 |
| --- | --- | --- | --- |
| `"<topic>"` | 是 | 总体研究主题 | 第一个参数；1–500 个字符 |
| `--facet "<facet>"` | 否 | 一个子领域或检索角度 | 最多 8 个；建议使用 3–8 个 |
| `--target-papers N` | 否 | 软目标论文数 | 默认 500 |
| `--max-papers N` | 否 | 硬上限 | 默认 `max(3 × target, 1500)` |
| `--download-workers N` | 否 | PDF 全局并发下载数 | 默认 8；范围 1–16 |
| `--download-per-host N` | 否 | 每个网站的下载并发数 | 默认 2；范围 1–16 |
| `--workspace <slug>` | 否 | 关联到指定 XLab 工作区 | 工作区必须存在 |

例如，下面的命令含义是：以语音表示学习为总主题，覆盖五个子方向，目标收集 200 篇，最多处理 600 篇候选论文：

```text
/xlab collect-papers "speech representation learning" \
  --target-papers 200 \
  --max-papers 600 \
  --facet "self-supervised learning for speech" \
  --facet "wav2vec and HuBERT" \
  --facet "speech foundation models" \
  --facet "multilingual speech representation" \
  --facet "robust speech representation"
```

## 6. 搜索范围和扩展方式

### 6.1 Tavily 搜索

系统会建立三类全局搜索通道：

1. Survey 和 Review：用于覆盖领域骨架和参考文献网络。
2. Seminal 和 Foundational papers：用于找到经典、奠基性工作。
3. Recent work：重点发现大约最近三年的研究。

每个 `facet` 还会增加一个 Tavily 搜索和一个 Semantic Scholar 搜索。建议使用有明确研究含义的 3–8 个 facet，避免只改变词序的大量重复关键词。

### 6.2 Semantic Scholar 扩展

Semantic Scholar 负责规范化论文信息和关系扩展，主要包括：

- 主题和 facet 相关性搜索。
- Survey 和经典论文的参考文献。
- 经典论文和近期论文的引用论文。
- 高质量种子的相关推荐论文。
- DOI、arXiv ID、Semantic Scholar ID 或严格标题匹配去重。

系统会优先进行一跳关系扩展。只有被判定为 `core` 或有充分证据的 `related` 论文，才可能成为下一轮扩展种子。

### 6.3 相关性分类

论文候选大致分为三类：

- `core`：直接高度相关的主题结果，或标题和摘要匹配度很高。
- `related`：相关性足够，或者得到多个种子支持，或者同时有图关系和文本证据。
- `boundary`：证据较弱或无法充分确认相关性。

`boundary` 候选会保留在原始候选记录中，但不会进入最终论文集合，也不会继续扩展。

### 6.4 停止条件

收集会在以下情况之一发生时停止：

- 达到软目标论文数。
- 连续两轮新增论文低于配置的创新比例。
- 达到候选数量上限。
- Semantic Scholar 调用预算耗尽。
- 没有新的元数据补全或图关系扩展动作。

因此，搜索结果不是简单的“关键词命中前 N 篇”，而是结合主题搜索、引用图和推荐关系形成的论文池。

## 7. 年份、会议和期刊范围

当前命令没有提供严格的 `--start-year`、`--end-year`、会议或期刊过滤参数。可以把时间和范围写进主题或 facet，例如：

```text
/xlab collect-papers "speech foundation models since 2022" \
  --target-papers 200 \
  --max-papers 600 \
  --facet "recent self-supervised speech representation learning since 2022" \
  --facet "recent audio language models since 2022"
```

这种方式属于查询语义约束，不是严格的结构化年份过滤。正式研究使用时，应根据输出的 `papers.jsonl` 元数据进行二次筛选和人工核验。

## 8. 输出位置

每次执行都会创建一个持久化 run：

```text
.xlab/runs/<run-id>/
```

如果从仓库根目录 `/shared/chaolei.liu/XLab` 运行，实际位置通常是：

```text
/shared/chaolei.liu/XLab/.xlab/runs/<run-id>/
```

主要输出结构如下：

```text
.xlab/runs/<run-id>/
├── manifest.json
└── artifacts/
    ├── request.json
    ├── query_plan.json
    ├── papers.manifest.json
    ├── collection_report.json
    ├── followups.json
    ├── failures.jsonl
    ├── metadata/
    │   ├── papers.jsonl
    │   ├── edges.jsonl
    │   └── seeds.json
    ├── pdfs/
    └── logs/
        ├── provider_results.jsonl
        ├── downloads.jsonl
        └── download_results.jsonl
```

最重要的文件和目录：

| 路径 | 内容 |
| --- | --- |
| `artifacts/papers.manifest.json` | 论文集合及集合级信息 |
| `artifacts/metadata/papers.jsonl` | 每篇论文一行的规范化元数据 |
| `artifacts/metadata/edges.jsonl` | 引用、参考文献和推荐关系 |
| `artifacts/metadata/seeds.json` | 初始搜索种子 |
| `artifacts/pdfs/` | 成功获取的开放获取 PDF |
| `artifacts/collection_report.json` | 数量、相关性、元数据、图关系和下载审计 |
| `artifacts/failures.jsonl` | 失败项和缺失信息 |
| `artifacts/logs/provider_results.jsonl` | Tavily 和 Semantic Scholar 调用记录 |
| `manifest.json` | run 状态、输入、输出和校验结果 |

成功产物还会被内容寻址地提交到：

```text
.xlab/artifacts/objects/sha256/
```

这些内容寻址对象用于后续知识图谱等流程，不能当作普通临时文件随意修改。

## 9. 中断、恢复和查看

查看所有 run：

```text
/xlab list-runs
```

查看指定 run：

```text
/xlab show-run <run-id>
```

恢复被中断的 run：

```text
/xlab resume-run <run-id>
```

Paper 收集内部会将收集、PDF 下载和审计作为独立阶段。下载阶段中断时，已经验证有效的 PDF 和临时文件会尽量复用，不需要从头下载。

## 10. 运行后的检查

运行结束后，建议重点检查：

1. `manifest.json` 的最终状态是 `success` 还是 `incomplete`。
2. `collection_report.json` 中的实际论文数量和目标数量。
3. `papers.jsonl` 的摘要、年份、作者、DOI 或 arXiv ID 覆盖情况。
4. `edges.jsonl` 是否有足够的引用或推荐关系。
5. `pdfs/` 中的有效 PDF 数量及下载失败原因。
6. `failures.jsonl` 和报告中的 warnings。
7. 语音主题是否过宽，是否需要重新按子方向分别收集。

缺少摘要或 PDF 的论文不会被自动静默删除，缺失字段会被明确记录。PDF 下载失败也不一定意味着论文从集合中移除。

## 11. 后续构建知识图谱

Paper 收集成功后，可以把该 run 作为知识图谱的输入：

```text
/xlab build-knowledge-graph <successful-paper-collect-run>
```

知识图谱要求输入一个成功的 `paper_collect` run，而不是任意手工创建的 PDF 目录。建议保留原始 run、manifest 和 artifact lineage，以便后续追溯论文来源。

## 12. 推荐执行顺序

第一次使用时，可以按以下顺序执行：

```text
/xlab check-xlab-setup
/xlab configure-xlab scholarly-services
```

然后用小规模配置测试：

```text
/xlab collect-papers "speech processing" \
  --target-papers 100 \
  --max-papers 300 \
  --facet "automatic speech recognition" \
  --facet "text-to-speech synthesis" \
  --facet "speech representation learning"
```

确认结果质量后，再使用正式规模：

```text
/xlab collect-papers "speech processing and spoken language technologies" \
  --target-papers 500 \
  --max-papers 1500 \
  --facet "automatic speech recognition" \
  --facet "text-to-speech synthesis" \
  --facet "speaker recognition and verification" \
  --facet "speech enhancement and separation" \
  --facet "spoken language understanding" \
  --facet "speech representation learning" \
  --facet "audio language models" \
  --facet "speech datasets and evaluation"
```

最后，再将成功的论文收集 run 交给 `build-knowledge-graph`。
