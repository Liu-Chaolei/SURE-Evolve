# ASR/TTS Paper 资料库：建设动机与实现方案

## 1. 建设动机

### 1.1 为什么需要 Paper 资料库

ASR/TTS 自进化 Agent 的核心能力之一，是根据已有实验结果提出下一轮研究方案。现有的结果驱动探索能够告诉 Agent“当前实验发生了什么”，但仅依赖单次任务内的轨迹，仍可能遇到以下问题：

- 不清楚研究者通常如何把现象转化为可证伪假设；
- 容易直接尝试方法，而没有设计足够的 baseline、control 和 ablation；
- 面对 WER、自然度、说话人相似度等指标的异常或 trade-off 时，缺少系统化诊断参考；
- 难以复用论文中已经验证过的实验组织方式、适用条件和失败经验；
- 普通 PDF RAG 往往只返回相似文本，无法保证结论能追溯到原文、表格数值和实验条件。

因此，需要建设一个面向 ASR/TTS 研究设计的资料库。它不是简单的“PDF 向量库”，而是将论文转化为可核验、可检索、可增量维护的研究证据和设计知识。

### 1.2 资料库希望提供什么

资料库主要提供三类辅助能力：

1. **研究背景参考**：了解常见问题、研究方向、变量和适用条件；
2. **实验设计参考**：学习如何提出假设、设置对照、设计消融并判断观测信号；
3. **失败诊断参考**：根据已有实验现象，查找可能机制、负结果、限制和诊断实验。

资料库的价值不只在于记录“论文用了什么方法”，更在于保存研究者如何完成以下推理链：

```text
研究问题
  → 假设或 tested relationship
  → base method 与 intervention delta
  → baseline / control / ablation
  → 训练与评测条件
  → 实验结果
  → 适用条件、限制、负结果与 trade-off
```

### 1.3 Paper 资料不是决策主路径

资料库必须遵循以下使用边界：

- Task card、SURE 约束和 artifact contract 是不可突破的硬边界；
- 当前任务真实运行得到的实验结果是下一轮决策的主要证据；
- 任务开始时尚无本地实验，论文信息只能形成弱先验，不能过早锁定方法；
- 每轮实验后的方案探索仍以现有结果驱动逻辑为主，Paper 检索只负责补充显式信息缺口；
- “论文中有效”不等于“当前任务中有效”，迁移到当前任务后仍需本地实验验证；
- 论文证据与本地实验经验必须分层保存，不能互相改写。

因此，本阶段只实现独立的资料库构建和查询能力，暂不修改 `sure_master` 的自进化流程。

## 2. 设计目标

资料库实现遵循以下目标：

### 2.1 可追溯

任何结构化结论都必须能够返回原始 PDF，而不是只保留模型摘要：

```text
ResearchDesignCard 字段
  → EvidenceRecord
  → ParsedDocument 中的 block / table cell / figure / equation
  → PDF page + bbox + verbatim text
  → 原始 PDF
```

### 2.2 结构化

资料库使用三个相互独立的严格 Pydantic 顶层模型：

1. **`ParsedDocument`**：保存 PDF 的布局和原始内容，不产生科研结论；
2. **`EvidenceRecord`**：保存可独立核验的原子 claim 及其精确来源；
3. **`ResearchDesignCard`**：保存论文中的 intervention、ablation、diagnostic、negative result、trade-off 或 system 设计。

### 2.3 可增量构建

新增、修改、移动或删除 PDF 时，不应重新处理整个语料库。系统依据内容 hash 和阶段 cache key，只重做受影响的文档及其下游阶段。

### 2.4 可验证和可复现

所有正式发布的记录必须通过 schema、引用、grounding 和语料一致性检查。发布快照、manifest 和索引都与内容 hash、配置和版本绑定。

### 2.5 代码与数据分离

- 可复用代码：`paper_library/`
- 默认配置：`configs/paper_library/default.yaml`
- 只读 PDF 输入：`data/papers/`
- 本地构建产物：`data/paper_library/`

源 PDF 不被修改，生成物也不与代码混放。

## 3. 总体实现方案

### 3.1 分层架构

```text
PDF corpus
   │
   ▼
ParsedDocument
页面、文本块、span、bbox、表格、图、公式和解析告警
   │
   ▼
Candidate Research Designs
识别一篇论文中的多个 intervention / ablation / diagnostic 单元
   │
   ▼
EvidenceRecord
原子 claim、实验条件、数值事实和精确 source refs
   │
   ▼
ResearchDesignCard
问题、假设、base+delta、controls、results、limitations、relations
   │
   ├──► 版本化 corpus snapshot
   └──► Evidence / Design 双索引
```

将抽取拆成多层，而不是让模型一次性总结整篇论文，可以减少无来源结论，并允许每一层独立验证、缓存和人工复核。

### 3.2 构建阶段

完整阶段 DAG 为：

```text
source
  → parse
  → candidate_designs
  → evidence
  → cards
  → validate
  → publish
  → lexical_index
```

预留了 `vector_index` 阶段，但首版只实现 SQLite FTS5/BM25 词法索引。

## 4. 各阶段如何实现

### 4.1 Source 扫描

扫描器确定性遍历 `data/papers/**/*.pdf`，计算 SHA-256 并同步 SQLite 状态：

- 相同内容的不同路径只解析一次，同时保留 source aliases；
- 文件重命名但内容不变时，不重新解析；
- PDF 内容变化时，只失效该文档及其下游阶段；
- 文件消失时标记为 missing，不自动删除历史快照；
- source root 始终作为只读目录使用。

### 4.2 布局感知 PDF 解析

首版解析器基于 PyMuPDF，不把 PDF 简化成一个平坦字符串。`ParsedDocument` 保存：

- PDF 内容 hash 和文档元数据；
- page、block、line/span 和 reading order；
- 原始文本、规范化文本和位置映射；
- page/bbox/offset；
- 基础 table、cell、figure、caption 和 equation 对象；
- parser 名称、版本和解析 warning。

对扫描件、低文本量或结构恢复不可靠的页面，系统记录 warning 或进入复核路径，而不是静默生成看似完整的结果。

### 4.3 候选研究设计发现

一篇论文可能包含主方法、多个子模块、消融、诊断和负结果，因此不能固定生成“一篇论文一张摘要卡”。候选发现阶段识别多个 design units，并保存：

- candidate 类型；
- 相关 section、block、table、figure 或 equation ID；
- 可能的 parent/child 关系；
- 候选 intervention 或 tested relationship。

默认不配置外部模型时，该阶段可使用本地结构规则运行。

### 4.4 Evidence 抽取

Evidence 层将论文内容拆成尽量单一、可独立核验的 claim，例如：

- research problem / gap；
- author hypothesis / proposed mechanism；
- method 或 implementation detail；
- training setup / evaluation protocol；
- baseline / control definition；
- reported result / ablation / negative result；
- trade-off / limitation / future work。

每条 Evidence 保存原文、条件、数值事实、抽取来源和质量信息。表格数值应关联到具体 cell，并记录 metric、direction、dataset/split、candidate/baseline value 等条件。

### 4.5 Research Design Card 构建

Design Card 只消费已通过 grounding 的 Evidence。每张详细卡对应一个主要 intervention 或 tested relationship，重点记录：

- 研究问题与 gap；
- 显式作者假设、重建假设或 tested relationship；
- base method 和修改增量；
- baseline、显式 control 和假定 control；
- 训练目标与最终评测指标；
- 结构化结果；
- applicability、limitations、negative results 和 trade-offs；
- parent/child 以及 `depends_on`、`part_of`、`extends`、`evaluated_with` 关系。

卡片中的事实字段通过 `field_evidence` 映射到 Evidence ID，避免只提供卡片级笼统引用。

### 4.6 验证与人工复核

验证分为多个层次：

1. **Schema 验证**：类型、枚举、ID、页码、bbox、offset 和字段联动规则；
2. **引用完整性**：Evidence 引用的文档和 source object 必须存在；Card 引用的 Evidence 必须存在；
3. **Grounding**：verbatim text 必须能在指定 block、cell 或 equation 中定位；
4. **字段覆盖率**：reported 数值、metric、dataset、split、backend 等事实必须具有字段级证据；
5. **科学一致性**：检查 metric direction、训练目标与最终评测、base+delta、control、数值差异和关系一致性；
6. **语料一致性**：检查 ID 唯一性、JSONL 数量和 hash、manifest 以及索引版本。

外部模型输出始终被视为不可信候选数据。未通过检查或需要人工判断的记录写入不可变 review queue，不直接进入正式快照。

### 4.7 增量状态与错误恢复

`data/paper_library/state/build.sqlite3` 记录每篇文档每个阶段的：

- status；
- cache key 和 input hash；
- attempt；
- output URI、SHA-256 和 record count；
- error code、message 和 retryable 分类。

Cache key 纳入相关上游内容、schema、实现版本、配置、prompt 和 provider identity。变化只向下游传播，例如 Card 抽取配置变化不应重新解析 PDF。

缓存命中时还会检查物理产物的 hash 和记录数，防止损坏的缓存继续被下游消费。

### 4.8 原子发布

只有通过必要验证的对象才能整理为正式语料快照：

```text
data/paper_library/records/<corpus-version>/
├── parsed_documents.jsonl
├── evidence_records.jsonl
└── research_design_cards.jsonl
```

每个 corpus manifest 绑定：

- 输入文档及内容 hash；
- schema、parser、配置、prompt 和 provider 信息；
- 文件路径、record count 和 SHA-256；
- 构建与验证统计。

完成后通过原子替换更新 `manifests/current.json`。如果新版本发布失败，消费者仍可读取上一个完整版本。

### 4.9 双索引与查询

首版在同一个版本化 SQLite 文件中维护两套独立 FTS5 表：

- **Evidence Index**：用于核查原文、数值、实验条件和限制；
- **Research Design Index**：用于检索相似问题、干预、消融和诊断设计。

推荐查询流程：

```text
检索 Research Design Card
  → 获得候选实验设计或诊断模式
  → 根据 evidence IDs 查询 Evidence
  → 核查原文、条件和具体数值
```

两类 BM25 分数只在各自索引内有意义，不跨类型比较。

## 5. 主要产物

默认构建目录如下：

```text
data/paper_library/
├── state/build.sqlite3
├── objects/
│   ├── parsed/
│   ├── candidates/
│   ├── evidence/
│   └── cards/
├── records/<corpus-version>/
├── indexes/<corpus-version>/lexical.sqlite3
├── manifests/
├── reviews/
├── failures/
├── reports/
└── staging/
```

其中：

- `objects/` 是逐文档、内容寻址的内部缓存；
- `records/` 是经过验证的稳定语料快照；
- `indexes/` 是可以根据快照重建的派生产物；
- `manifests/current.json` 是消费者读取当前完整版本的入口；
- `reviews/` 保存待复核候选和复核决定；
- `state/`、`failures/` 和 `reports/` 用于增量调度与质量审计。

## 6. 配置与运行方式

默认配置位于：

```text
configs/paper_library/default.yaml
```

安全的本地流程不会调用外部模型：

```bash
CONFIG=configs/paper_library/default.yaml

python -m paper_library validate-config --config "$CONFIG"
python -m paper_library scan --config "$CONFIG"
python -m paper_library parse --config "$CONFIG"
python -m paper_library extract --config "$CONFIG" candidates
python -m paper_library status --config "$CONFIG"
python -m paper_library validate --config "$CONFIG"
```

完整 Evidence 和 Design Card 抽取需要显式配置 OpenAI 或 Anthropic provider、model 和 API key：

```bash
python -m paper_library build --config configs/paper_library/local.yaml
python -m paper_library index --config configs/paper_library/local.yaml
```

启用 provider 会将相关论文内容发送到对应外部服务，因此不能在未明确配置和授权的情况下自动启用。

具体命令和配置示例见[构建器使用说明](./paper_library_builder.md)。

## 7. 当前实现范围与后续工作

首版已完成：

- 独立 package 和 CLI；
- 三层严格 schema；
- PDF 扫描与布局解析；
- 增量状态、缓存和失效传播；
- Candidate、Evidence 和 Design Card 流程；
- grounding、验证和 review queue；
- 版本化原子发布；
- Evidence/Design FTS5 索引和只读查询；
- 自动测试、真实 PDF 解析和文档。

当前没有包含：

- Paper 选择和下载；
- OCR、GROBID 或视觉解析后端；
- FAISS/向量检索的正式实现；
- review decision 自动应用到发布；
- `retry` 命令的完整实现；
- `sure_master` 的资料库消费接口；
- 将 Paper 检索直接变成强约束或自动实验决策。

后续接入自进化时，应继续坚持“本地实验结果为主、Paper Evidence 为辅”的原则：任务开始阶段只生成弱 Research Brief；实验后仅围绕明确的信息缺口检索资料库；任何迁移建议都必须转化为当前任务内可证伪、可执行、可评分的实验假设。
